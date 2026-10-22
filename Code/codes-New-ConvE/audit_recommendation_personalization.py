"""Audit saved rankings, score/popularity association and reference overlap.

Accepts both score and logit files. Correlations are descriptive, not causal.
Undefined correlations from constant vectors are reported as null, never zero.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from fairness_context import load_item_popularity, read_q_matrix
from fairness_rerank import load_uid_ex_scores
from semantic_experiment_utils import write_json


def checked_rows(path: Path, exercise_count: int) -> dict[str, np.ndarray]:
    result = {}
    for uid, values in load_uid_ex_scores(path):
        array = np.asarray(values, dtype=np.float64)
        if uid in result or array.shape != (exercise_count,) or not np.isfinite(array).all():
            raise ValueError(f"Duplicate student, mismatched candidates or nonfinite scores: {path}/{uid}")
        result[uid] = array
    if not result:
        raise ValueError(f"No students: {path}")
    return result


def pearson(x, y) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(), y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else None


def summarize(rows: dict[str, np.ndarray], popularity, top_ks, reference=None) -> dict:
    if reference is not None and (rows.keys() != reference.keys()
                                  or any(rows[u].shape != reference[u].shape for u in rows)):
        raise ValueError("Reference must contain exactly the same students and candidates")
    ranks = {u: np.lexsort((np.arange(len(a)), -a)).tolist() for u, a in rows.items()}
    corrs = [pearson(a, np.log1p(popularity)) for a in rows.values()]
    defined = [c for c in corrs if c is not None]
    result = {
        "students": len(rows), "exercise_count": len(popularity),
        "mean_within_student_score_logpop_pearson": float(np.mean(defined)) if defined else None,
        "undefined_correlation_students": len(corrs) - len(defined),
        "mean_distinct_score_values": float(np.mean([len(np.unique(a)) for a in rows.values()])),
        "top_n": {},
    }
    for n in top_ks:
        k = min(n, len(popularity))
        if k <= 0 or n <= 0:
            raise ValueError("Top-N and candidate count must be positive")
        lists = [tuple(a[:k]) for a in ranks.values()]
        counts = Counter(lists)
        row = {"effective_n": k, "unique_ordered_lists": len(counts),
               "unique_items": len({i for a in lists for i in a}),
               "largest_identical_list_fraction": max(counts.values()) / len(rows)}
        if reference is not None:
            overlaps, exact = [], []
            for u, a in ranks.items():
                b = np.lexsort((np.arange(len(reference[u])), -reference[u])).tolist()[:k]
                overlaps.append(len(set(a[:k]) & set(b)) / k)
                exact.append(a[:k] == b)
            row.update(mean_reference_overlap=float(np.mean(overlaps)),
                       exact_reference_list_fraction=float(np.mean(exact)))
        result["top_n"][str(n)] = row
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--reference-scores-file", type=Path)
    parser.add_argument("--core-file", type=Path)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--top-ks", default="5,10,15,20,25,30,35,40,45,50")
    args = parser.parse_args(argv)
    count = len(read_q_matrix(args.data_dir / "Q.txt"))
    popularity, metadata = load_item_popularity(args.data_dir, count, source="train_interactions", aggregation="unique_users")
    rows = checked_rows(args.scores_file, count)
    reference = checked_rows(args.reference_scores_file, count) if args.reference_scores_file else None
    top_ks = [int(v) for v in args.top_ks.split(",")]
    result = summarize(rows, popularity, top_ks, reference)
    if args.core_file:
        core = checked_rows(args.core_file, count)
        result["core"] = summarize(core, popularity, top_ks, rows)
    paths = [args.scores_file, args.reference_scores_file, args.core_file]
    result["inputs"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p}
    result["popularity_source"] = metadata
    result["interpretation"] = "Descriptive diagnostics; coverage and correlation alone do not establish debiasing or educational quality."
    write_json(args.output_file, result)
    print(f"Personalization audit -> {args.output_file}")


if __name__ == "__main__":
    main()
