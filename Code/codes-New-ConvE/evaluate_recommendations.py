import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from experiment_utils import update_timing, write_json
from fairness_metrics import calculate_fairness_metrics
from semantic_experiment_utils import DEFAULT_TOP_KS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ER recommendation scores with Ada, NOV and fairness metrics.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--nov-alpha", type=float, default=1.0)
    parser.add_argument("--fairness-head-ratio", type=float, default=0.2)
    parser.add_argument("--fairness-long-tail-ratio", type=float, default=0.8)
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


def load_uid_ex_scores(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as fp:
            rows = pickle.load(fp)
    else:
        with path.open("r", encoding="utf-8") as fp:
            rows = json.load(fp)

    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append((row["uid"], row["scores"]))
        else:
            normalized.append((row[0], row[1]))
    return normalized


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


def write_fairness_csv(fairness_metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_k = fairness_metrics.get("top_k", {})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "top_k", "value"])
        for top_k, payload in by_k.items():
            if not isinstance(payload, dict):
                continue
            for metric_name, value in payload.items():
                if metric_name == "recommendations":
                    continue
                writer.writerow([metric_name, top_k, value])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = [int(value.strip()) for value in args.top_ks.split(",") if value.strip()]

    metric_start = time.perf_counter()
    q_matrix = load_q_matrix(args.data_dir / "Q.txt")
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

    fairness = calculate_fairness_metrics(
        uid_ex_scores=uid_ex_scores,
        q_matrix=q_matrix,
        train_triples_path=args.data_dir / "triples.txt",
        top_ks=top_ks,
        head_ratio=args.fairness_head_ratio,
        long_tail_ratio=args.fairness_long_tail_ratio,
    )
    metrics = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "seed": args.seed,
        "scores_file": str(args.scores_file),
        "uid_kc_response_file": str(response_file),
        "Ada": ada,
        "NOV": nov,
        "Fairness": fairness["top_k"],
        "FairnessDefinition": fairness["definition"],
    }
    write_json(metrics, args.output_dir / "metrics.json")
    write_csv({"Ada": ada, "NOV": nov}, args.output_dir / "metrics.csv")
    write_json(fairness, args.output_dir / "fairness_metrics.json")
    write_fairness_csv(fairness, args.output_dir / "fairness_metrics.csv")
    write_uid_ex_scores_json(uid_ex_scores, args.output_dir / "uid_ex_scores.json")
    if args.timing_file:
        update_timing(args.timing_file, "evaluation_metric", time.perf_counter() - metric_start)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
