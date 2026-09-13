"""E0 popularity-bias chain diagnostics for recommendation exposure."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from fairness_context import load_item_popularity, read_q_matrix
from fairness_metrics import coverage, exposure_counts, gini, kc_popularity_from_items, load_item_popularity_from_triples
from semantic_experiment_utils import DEFAULT_TOP_KS, write_json


def parse_top_ks(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def load_uid_ex_scores(path: Path) -> List[Tuple[str, List[float]]]:
    if path.suffix == ".pkl":
        with path.open("rb") as fp:
            rows = pickle.load(fp)
    else:
        with path.open("r", encoding="utf-8") as fp:
            rows = json.load(fp)

    normalized: List[Tuple[str, List[float]]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append((str(row["uid"]), [float(value) for value in row["scores"]]))
        else:
            normalized.append((str(row[0]), [float(value) for value in row[1]]))
    return normalized


def normalize_distribution(values: Sequence[float]) -> np.ndarray:
    array = np.asarray([max(0.0, float(value)) for value in values], dtype=np.float64)
    total = float(array.sum())
    if array.size == 0:
        return array
    if total <= 0.0:
        return np.full(array.shape, 1.0 / array.size, dtype=np.float64)
    return array / total


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = normalize_distribution(left)
    q = normalize_distribution(right)
    if p.size != q.size:
        raise ValueError("JS divergence requires distributions with the same support size")
    if p.size == 0:
        return 0.0
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def head_mid_tail_groups(popularity: Sequence[float], head_ratio: float, tail_ratio: float) -> Tuple[set[int], set[int], set[int]]:
    if not 0 <= head_ratio <= 1:
        raise ValueError("head_ratio must be between 0 and 1")
    if not 0 <= tail_ratio <= 1:
        raise ValueError("tail_ratio must be between 0 and 1")
    if head_ratio + tail_ratio > 1:
        raise ValueError("head_ratio + tail_ratio must be at most 1")
    count = len(popularity)
    ranked = sorted(range(count), key=lambda idx: (-float(popularity[idx]), idx))
    head_count = min(count, int(np.ceil(count * head_ratio)))
    tail_count = min(count - head_count, int(np.ceil(count * tail_ratio)))
    head = set(ranked[:head_count])
    tail = set(ranked[count - tail_count :]) if tail_count > 0 else set()
    mid = set(ranked[head_count : count - tail_count])
    return head, mid, tail


def share(values: Sequence[float], selected: set[int]) -> float:
    total = float(sum(values))
    if total <= 0.0:
        return 0.0
    return float(sum(float(values[idx]) for idx in selected if 0 <= idx < len(values)) / total)


def average_recommendation_popularity(values: Sequence[float], reference_popularity: Sequence[float]) -> float:
    distribution = normalize_distribution(values)
    reference = np.asarray([float(value) for value in reference_popularity], dtype=np.float64)
    if distribution.size == 0:
        return 0.0
    return float(np.sum(distribution * reference))


def metrics_for_distribution(
    *,
    stage: str,
    target: str,
    values: Sequence[float],
    reference_data: Sequence[float],
    reference_label: Sequence[float],
    head: set[int],
    mid: set[int],
    tail: set[int],
    top_k: int | None = None,
) -> Dict[str, Any]:
    row = {
        "target": target,
        "stage": stage,
        "top_k": "" if top_k is None else int(top_k),
        "total_exposure": round(float(sum(values)), 6),
        "Gini": round(gini(values), 6),
        "DeltaGini_vs_p_data": round(gini(values) - gini(reference_data), 6),
        "JS_vs_p_data": round(js_divergence(values, reference_data), 6),
        "JS_vs_p_label": round(js_divergence(values, reference_label), 6),
        "ARP": round(average_recommendation_popularity(values, reference_data), 6),
        "Coverage": round(coverage(values), 6),
        "HeadExposureShare": round(share(values, head), 6),
        "MidExposureShare": round(share(values, mid), 6),
        "TailExposureShare": round(share(values, tail), 6),
    }
    return row


def score_exposure_rows(
    *,
    stage: str,
    scores_file: Path,
    q_matrix: Sequence[Sequence[int]],
    top_ks: Sequence[int],
) -> Dict[int, Tuple[List[float], List[float]]]:
    scores = load_uid_ex_scores(scores_file)
    by_k: Dict[int, Tuple[List[float], List[float]]] = {}
    for top_k in top_ks:
        item_exposure, kc_exposure, _recommendation_count = exposure_counts(scores, q_matrix, int(top_k))
        by_k[int(top_k)] = (item_exposure, kc_exposure)
    return by_k


def write_csv_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "stage",
        "top_k",
        "total_exposure",
        "Gini",
        "DeltaGini_vs_p_data",
        "JS_vs_p_data",
        "JS_vs_p_label",
        "ARP",
        "Coverage",
        "HeadExposureShare",
        "MidExposureShare",
        "TailExposureShare",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_diagnostics(args: argparse.Namespace) -> Dict[str, Any]:
    q_matrix = read_q_matrix(args.data_dir / "Q.txt")
    p_data_item, p_data_metadata = load_item_popularity(
        args.data_dir,
        len(q_matrix),
        source="train_interactions",
        aggregation=args.popularity_aggregation,
    )
    p_label_item = load_item_popularity_from_triples(args.data_dir / "triples.txt", len(q_matrix))
    p_data_kc = kc_popularity_from_items(p_data_item, q_matrix)
    p_label_kc = kc_popularity_from_items(p_label_item, q_matrix)

    item_head, item_mid, item_tail = head_mid_tail_groups(p_data_item, args.head_ratio, args.tail_ratio)
    kc_head, kc_mid, kc_tail = head_mid_tail_groups(p_data_kc, args.head_ratio, args.tail_ratio)

    rows: List[Dict[str, Any]] = [
        metrics_for_distribution(
            stage="p_data",
            target="item",
            values=p_data_item,
            reference_data=p_data_item,
            reference_label=p_label_item,
            head=item_head,
            mid=item_mid,
            tail=item_tail,
        ),
        metrics_for_distribution(
            stage="p_label",
            target="item",
            values=p_label_item,
            reference_data=p_data_item,
            reference_label=p_label_item,
            head=item_head,
            mid=item_mid,
            tail=item_tail,
        ),
        metrics_for_distribution(
            stage="p_data",
            target="kc",
            values=p_data_kc,
            reference_data=p_data_kc,
            reference_label=p_label_kc,
            head=kc_head,
            mid=kc_mid,
            tail=kc_tail,
        ),
        metrics_for_distribution(
            stage="p_label",
            target="kc",
            values=p_label_kc,
            reference_data=p_data_kc,
            reference_label=p_label_kc,
            head=kc_head,
            mid=kc_mid,
            tail=kc_tail,
        ),
    ]

    score_specs = [("p_base", args.base_scores_file), ("p_debias", args.debiased_scores_file)]
    for stage, scores_file in score_specs:
        if scores_file is None:
            continue
        for top_k, (item_exposure, kc_exposure) in score_exposure_rows(stage=stage, scores_file=scores_file, q_matrix=q_matrix, top_ks=args.top_ks).items():
            rows.append(
                metrics_for_distribution(
                    stage=stage,
                    target="item",
                    values=item_exposure,
                    reference_data=p_data_item,
                    reference_label=p_label_item,
                    head=item_head,
                    mid=item_mid,
                    tail=item_tail,
                    top_k=top_k,
                )
            )
            rows.append(
                metrics_for_distribution(
                    stage=stage,
                    target="kc",
                    values=kc_exposure,
                    reference_data=p_data_kc,
                    reference_label=p_label_kc,
                    head=kc_head,
                    mid=kc_mid,
                    tail=kc_tail,
                    top_k=top_k,
                )
            )

    return {
        "metadata": {
            "diagnostic": "E0_bias_chain",
            "data_dir": str(args.data_dir),
            "base_scores_file": str(args.base_scores_file) if args.base_scores_file else None,
            "debiased_scores_file": str(args.debiased_scores_file) if args.debiased_scores_file else None,
            "top_ks": args.top_ks,
            "popularity_aggregation": args.popularity_aggregation,
            "head_ratio": args.head_ratio,
            "tail_ratio": args.tail_ratio,
            "bucket_definition": "Head/Mid/Tail buckets are ranked by p_data popularity.",
            "p_data_source": p_data_metadata,
            "p_label_source": {"source": "rec_triples", "path": str(args.data_dir / "triples.txt"), "relation": "rec"},
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare p_data, p_label, p_base and p_debias popularity-bias chains.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-scores-file", type=Path, default=None)
    parser.add_argument("--debiased-scores-file", type=Path, default=None)
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--tail-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.top_ks = parse_top_ks(args.top_ks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = build_diagnostics(args)
    write_json(args.output_dir / "bias_chain_metrics.json", diagnostics)
    write_csv_rows(diagnostics["rows"], args.output_dir / "bias_chain_metrics.csv")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
