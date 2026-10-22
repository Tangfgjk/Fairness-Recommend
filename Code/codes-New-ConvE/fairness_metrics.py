"""Resource-side fairness metrics for 2CKG4ER recommendations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_exercise_id(value: Any) -> int | None:
    text = str(value)
    if text.startswith("ex"):
        text = text[2:]
    try:
        return int(text)
    except ValueError:
        return None


def top_exercises(scores: Sequence[float], top_k: int, exercise_count: int | None = None) -> List[int]:
    limit = len(scores) if exercise_count is None else min(len(scores), int(exercise_count))
    ranked = sorted(range(limit), key=lambda idx: (float(scores[idx]), -idx), reverse=True)
    return ranked[: max(0, int(top_k))]


def exercise_concepts(q_matrix: Sequence[Sequence[int]], exercise_idx: int) -> List[int]:
    if exercise_idx < 0 or exercise_idx >= len(q_matrix):
        return []
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def gini(values: Sequence[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0:
        return 0.0
    if np.any(array < 0):
        raise ValueError("Gini is undefined for negative values")
    total = float(array.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(array)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values)) / (n * total) - (n + 1.0) / n)


def load_item_popularity_from_triples(path: Path, exercise_count: int) -> List[float]:
    popularity = [0.0 for _ in range(exercise_count)]
    if not path.exists():
        return popularity

    fallback_counts = [0.0 for _ in range(exercise_count)]
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            head, relation, tail = parts
            tail_idx = parse_exercise_id(tail)
            if tail_idx is not None and 0 <= tail_idx < exercise_count:
                fallback_counts[tail_idx] += 1.0
                if relation == "rec":
                    popularity[tail_idx] += 1.0
            head_idx = parse_exercise_id(head)
            if head_idx is not None and 0 <= head_idx < exercise_count:
                fallback_counts[head_idx] += 1.0

    if sum(popularity) <= 0:
        return fallback_counts
    return popularity


def kc_popularity_from_items(item_popularity: Sequence[float], q_matrix: Sequence[Sequence[int]]) -> List[float]:
    concept_count = max((len(row) for row in q_matrix), default=0)
    popularity = [0.0 for _ in range(concept_count)]
    for exercise_idx, item_count in enumerate(item_popularity):
        concepts = exercise_concepts(q_matrix, exercise_idx)
        if not concepts:
            continue
        share = float(item_count) / len(concepts)
        for concept_idx in concepts:
            popularity[concept_idx] += share
    return popularity


def head_and_long_tail(popularity: Sequence[float], head_ratio: float, long_tail_ratio: float) -> Tuple[set[int], set[int]]:
    count = len(popularity)
    if count == 0:
        return set(), set()
    if not 0 <= head_ratio <= 1:
        raise ValueError("head_ratio must be between 0 and 1")
    if not 0 <= long_tail_ratio <= 1:
        raise ValueError("long_tail_ratio must be between 0 and 1")

    ranked = sorted(range(count), key=lambda idx: (-float(popularity[idx]), idx))
    head_count = min(count, max(0, math.ceil(count * head_ratio)))
    long_tail_count = min(count, max(0, math.ceil(count * long_tail_ratio)))
    head = set(ranked[:head_count])
    long_tail_start = max(head_count, count - long_tail_count)
    long_tail = set(ranked[long_tail_start:])
    return head, long_tail


def exposure_counts(
    uid_ex_scores: Iterable[Tuple[str, Sequence[float]]],
    q_matrix: Sequence[Sequence[int]],
    top_k: int,
) -> Tuple[List[float], List[float], int]:
    exercise_count = len(q_matrix)
    concept_count = max((len(row) for row in q_matrix), default=0)
    item_exposure = [0.0 for _ in range(exercise_count)]
    kc_exposure = [0.0 for _ in range(concept_count)]
    recommendation_count = 0

    for _, scores in uid_ex_scores:
        for exercise_idx in top_exercises(scores, top_k, exercise_count=exercise_count):
            item_exposure[exercise_idx] += 1.0
            recommendation_count += 1
            concepts = exercise_concepts(q_matrix, exercise_idx)
            if not concepts:
                continue
            share = 1.0 / len(concepts)
            for concept_idx in concepts:
                kc_exposure[concept_idx] += share

    return item_exposure, kc_exposure, recommendation_count


def coverage(exposure: Sequence[float]) -> float:
    if not exposure:
        return 0.0
    return float(sum(1 for value in exposure if float(value) > 0.0) / len(exposure))


def exposure_share(exposure: Sequence[float], selected: set[int]) -> float:
    total = float(sum(exposure))
    if total <= 0:
        return 0.0
    return float(sum(float(exposure[idx]) for idx in selected if 0 <= idx < len(exposure)) / total)


def calculate_fairness_metrics(
    uid_ex_scores: Iterable[Tuple[str, Sequence[float]]],
    q_matrix: Sequence[Sequence[int]],
    top_ks: Sequence[int],
    train_triples_path: Path | None = None,
    item_popularity: Sequence[float] | None = None,
    popularity_metadata: Dict[str, Any] | None = None,
    head_ratio: float = 0.2,
    long_tail_ratio: float = 0.8,
) -> Dict[str, Any]:
    scores = list(uid_ex_scores)
    exercise_count = len(q_matrix)
    if item_popularity is None:
        if train_triples_path is None:
            raise ValueError("Either item_popularity or train_triples_path must be provided")
        item_popularity_values = load_item_popularity_from_triples(train_triples_path, exercise_count)
        popularity_metadata = popularity_metadata or {
            "source": "rec_triples",
            "path": str(train_triples_path),
            "relation": "rec",
            "aggregation": "edge_count",
        }
    else:
        item_popularity_values = [float(value) for value in item_popularity[:exercise_count]]
        if len(item_popularity_values) < exercise_count:
            item_popularity_values.extend([0.0] * (exercise_count - len(item_popularity_values)))
        popularity_metadata = popularity_metadata or {"source": "provided_item_popularity"}
    kc_popularity = kc_popularity_from_items(item_popularity_values, q_matrix)
    head_items, long_tail_items = head_and_long_tail(item_popularity_values, head_ratio, long_tail_ratio)
    head_kcs, long_tail_kcs = head_and_long_tail(kc_popularity, head_ratio, long_tail_ratio)

    by_k: Dict[str, Any] = {}
    for top_k in top_ks:
        item_exposure, kc_exposure, recommendation_count = exposure_counts(scores, q_matrix, int(top_k))
        by_k[str(top_k)] = {
            "ItemExposureGini": round(gini(item_exposure), 6),
            "KCExposureGini": round(gini(kc_exposure), 6),
            "ItemCoverage": round(coverage(item_exposure), 6),
            "KCCoverage": round(coverage(kc_exposure), 6),
            "LongTailItemExposureShare": round(exposure_share(item_exposure, long_tail_items), 6),
            "LongTailKCExposureShare": round(exposure_share(kc_exposure, long_tail_kcs), 6),
            "HeadItemExposureShare": round(exposure_share(item_exposure, head_items), 6),
            "HeadKCExposureShare": round(exposure_share(kc_exposure, head_kcs), 6),
            "recommendations": recommendation_count,
        }

    return {
        "top_k": by_k,
        "definition": {
            "fairness_subject": "exercise_and_knowledge_concept_exposure",
            "item_popularity_source": popularity_metadata,
            "head_ratio": head_ratio,
            "long_tail_ratio": long_tail_ratio,
            "kc_exposure": "fractional_credit_by_q_matrix",
            "kc_popularity": "fractional_item_popularity_by_q_matrix",
        },
    }
