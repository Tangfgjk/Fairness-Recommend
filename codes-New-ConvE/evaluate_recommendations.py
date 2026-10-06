import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from experiment_utils import update_timing, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ER recommendation scores with ACC, NOV and model-estimated Ep.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-ks", default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--nov-alpha", type=float, default=1.0)
    parser.add_argument("--ep-thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--ep-device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--ep-calibration-root", type=Path, default=Path(__file__).resolve().parents[1] / "Data_Fin_New" / "_ep_readouts")
    parser.add_argument("--ep-data-root", type=Path, default=None,
                        help="Base Data_Fin_New root when evaluating a graph in a nested variant directory.")
    parser.add_argument("--timing-file", type=Path, default=None)
    return parser.parse_args()


def load_q_matrix(path):
    matrix = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                matrix.append([int(value) for value in line.split(",")])
    return matrix


def load_json_matrix(path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_uid_ex_scores(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as fp:
            return pickle.load(fp)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_uid_ex_scores_json(uid_ex_scores, path):
    rows = [{"uid": uid, "scores": scores} for uid, scores in uid_ex_scores]
    write_json(rows, path)


def load_test_cognitive_relations(test_triples_path):
    uid_mlkc = {}
    with test_triples_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            head, relation, tail = line.strip().split("\t")
            if relation.startswith("mlkc"):
                uid_mlkc.setdefault(tail, {})[head] = float(relation[4:])
    return uid_mlkc


def load_uid_kc_response(data_dir, dataset_name):
    candidates = [
        data_dir / f"{dataset_name}_uid_kc_response.txt",
        data_dir / f"{dataset_name.replace('_prepared', '')}_uid_kc_response.txt",
    ]
    matches = list(data_dir.glob("*_uid_kc_response.txt"))
    candidates.extend(matches)
    for path in candidates:
        if not path.exists():
            continue
        rows = {}
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                values = [int(value) for value in parts[1].split(",") if value != ""]
                rows[parts[0]] = values
        return rows, path
    raise FileNotFoundError(f"No *_uid_kc_response.txt found in {data_dir}")


def top_exercises(scores, top_k):
    return [idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]]


def exercise_concepts(q_matrix, exercise_idx):
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def calculate_acc(uid_mlkc, uid_ex_scores, q_matrix, target_mastery, top_k):
    values = []
    for uid, scores in uid_ex_scores:
        if uid not in uid_mlkc:
            continue
        total = 0.0
        for exercise_idx in top_exercises(scores, top_k):
            ex_mastery = 1.0
            for kc_idx in exercise_concepts(q_matrix, exercise_idx):
                ex_mastery *= float(uid_mlkc[uid].get(f"kc{kc_idx}", 0.0))
            total += 1.0 - abs(float(target_mastery) - ex_mastery)
        values.append(total / max(1, top_k))
    return float(np.mean(values)) if values else 0.0, float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def calculate_time_weighted_nov(uid_kc_response, uid_ex_scores, q_matrix, top_k, alpha):
    values = []
    for uid, scores in uid_ex_scores:
        if uid not in uid_kc_response:
            continue
        history = uid_kc_response[uid]
        t_plus_1 = len(history)
        kc_last_time = {kc: idx for idx, kc in enumerate(history, 1)}
        total = 0.0
        for exercise_idx in top_exercises(scores, top_k):
            rec_kcs = set(exercise_concepts(q_matrix, exercise_idx))
            history_set = set(history)
            union = history_set.union(rec_kcs)
            if not union:
                total += 0.0
                continue
            intersection = history_set.intersection(rec_kcs)
            weighted_intersection = sum(
                np.exp(-alpha * (t_plus_1 - kc_last_time.get(kc, t_plus_1 + 1))) for kc in intersection
            )
            weighted_union = sum(
                np.exp(-alpha * (t_plus_1 - kc_last_time.get(kc, t_plus_1 + 1))) for kc in union
            )
            total += 1.0 - weighted_intersection / weighted_union
        values.append(total / max(1, top_k))
    return float(np.mean(values)) if values else 0.0, float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_csv(metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "top_k", "mean", "std"])
        for metric_name, by_k in metrics.items():
            if not isinstance(by_k, dict):
                continue
            for top_k, payload in by_k.items():
                writer.writerow([metric_name, top_k, payload.get("mean"), payload.get("std")])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = [int(value.strip()) for value in args.top_ks.split(",") if value.strip()]

    metric_start = time.perf_counter()
    q_matrix = load_q_matrix(args.data_dir / "Q.txt")
    uid_ex_scores = load_uid_ex_scores(args.scores_file)
    uid_mlkc = load_test_cognitive_relations(args.data_dir / "test_triples.txt")
    uid_kc_response, response_file = load_uid_kc_response(args.data_dir, args.dataset_name)

    acc = {}
    nov = {}
    for top_k in top_ks:
        acc_mean, acc_std = calculate_acc(uid_mlkc, uid_ex_scores, q_matrix, args.target_mastery, top_k)
        nov_mean, nov_std = calculate_time_weighted_nov(uid_kc_response, uid_ex_scores, q_matrix, top_k, args.nov_alpha)
        acc[str(top_k)] = {"mean": round(acc_mean, 6), "std": round(acc_std, 6)}
        nov[str(top_k)] = {"mean": round(nov_mean, 6), "std": round(nov_std, 6)}

    import sys
    import torch
    ep_module_dir = Path(__file__).resolve().parents[1] / "mastery_ep_v2"
    sys.path.insert(0, str(ep_module_dir))
    from evaluate import evaluate_entry, load_context, parse_float_list
    data_root = args.ep_data_root or args.data_dir.parent.parent
    context = load_context(args.dataset_name, data_root, args.ep_calibration_root, torch.device(args.ep_device))
    thresholds = parse_float_list(args.ep_thresholds)
    if thresholds != [0.4, 0.5, 0.6, 0.7]:
        raise ValueError("Comparison tables require the four fixed Ep thresholds: 0.4,0.5,0.6,0.7")
    ep_result = evaluate_entry(context, args.scores_file, args.output_dir / "ep",
                               top_ks, thresholds)
    ep = {f"Ep_t{threshold:.1f}": {} for threshold in thresholds}
    for item in ep_result["results"]:
        ep[f"Ep_t{item['threshold']:.1f}"][str(item["top_n"])] = {
            "mean": round(item["mean"], 6),
            "std": round(item["std_across_users"], 6) if item["std_across_users"] is not None else None,
        }
    metrics = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "seed": args.seed,
        "scores_file": str(args.scores_file),
        "uid_kc_response_file": str(response_file),
        "ACC": acc,
        "NOV": nov,
        **ep,
        "ep_thresholds": thresholds,
        "ep_calibration_root": str(args.ep_calibration_root),
    }
    write_json(metrics, args.output_dir / "metrics.json")
    write_csv({"ACC": acc, "NOV": nov, **ep}, args.output_dir / "metrics.csv")
    write_uid_ex_scores_json(uid_ex_scores, args.output_dir / "uid_ex_scores.json")
    if args.timing_file:
        update_timing(args.timing_file, "evaluation_metric", time.perf_counter() - metric_start)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
