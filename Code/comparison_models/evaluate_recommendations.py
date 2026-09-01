import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ep_sim import calculate_ep_sim
from experiment_utils import update_timing, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ER recommendation scores with Ada, NOV and Ep_sim.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-ks", default="10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--nov-alpha", type=float, default=1.0)
    parser.add_argument("--ep-top-k", type=int, default=10)
    parser.add_argument("--learning-gain", type=float, default=0.1)
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


def calculate_ada(uid_mlkc, uid_ex_scores, q_matrix, target_mastery, top_k):
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
    mastery = load_json_matrix(args.data_dir / "stu2know_mastery.json")
    uid_ex_scores = load_uid_ex_scores(args.scores_file)
    uid_mlkc = load_test_cognitive_relations(args.data_dir / "test_triples.txt")
    uid_kc_response, response_file = load_uid_kc_response(args.data_dir, args.dataset_name)

    ada = {}
    nov = {}
    for top_k in top_ks:
        ada_mean, ada_std = calculate_ada(uid_mlkc, uid_ex_scores, q_matrix, args.target_mastery, top_k)
        nov_mean, nov_std = calculate_time_weighted_nov(uid_kc_response, uid_ex_scores, q_matrix, top_k, args.nov_alpha)
        ada[str(top_k)] = {"mean": round(ada_mean, 6), "std": round(ada_std, 6)}
        nov[str(top_k)] = {"mean": round(nov_mean, 6), "std": round(nov_std, 6)}

    ep = calculate_ep_sim(
        uid_ex_scores=uid_ex_scores,
        q_matrix=q_matrix,
        mastery=mastery,
        top_k=args.ep_top_k,
        learning_gain=args.learning_gain,
    )
    metrics = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "seed": args.seed,
        "scores_file": str(args.scores_file),
        "uid_kc_response_file": str(response_file),
        "Ada": ada,
        "NOV": nov,
        "Ep_sim": {"top_k": args.ep_top_k, "mean": ep["mean"], "std": ep["std"]},
    }
    write_json(metrics, args.output_dir / "metrics.json")
    write_csv({"Ada": ada, "NOV": nov}, args.output_dir / "metrics.csv")
    write_json(ep, args.output_dir / "ep_sim.json")
    write_uid_ex_scores_json(uid_ex_scores, args.output_dir / "uid_ex_scores.json")
    if args.timing_file:
        update_timing(args.timing_file, "evaluation_metric", time.perf_counter() - metric_start)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
