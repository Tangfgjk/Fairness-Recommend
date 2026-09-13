"""Post-process 2CKG4ER recommendation scores with resource fairness reranking."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from experiment_utils import write_json
from fairness_context import load_item_popularity
from fairness_metrics import (
    exercise_concepts,
    head_and_long_tail,
    kc_popularity_from_items,
    top_exercises,
)
from semantic_experiment_utils import DEFAULT_TOP_KS


VALID_METHODS = {"baseline", "item", "kc", "item_kc", "quota_strict", "quota_ratio", "quota_ratio_hybrid"}
QUOTA_METHODS = {"quota_strict", "quota_ratio", "quota_ratio_hybrid"}
RATIO_QUOTA_METHODS = {"quota_ratio", "quota_ratio_hybrid"}


@dataclass(frozen=True)
class RerankConfig:
    method: str = "item_kc"
    top_k: int = max(DEFAULT_TOP_KS)
    candidate_multiplier: float = 3.0
    min_candidates: int = max(DEFAULT_TOP_KS)
    lambda_item: float = 0.3
    lambda_kc: float = 0.3
    beta_item: float = 0.1
    beta_kc: float = 0.1
    quota_top_k: int = 10
    min_long_tail_items: int = 1
    min_kc_coverage: int = 10
    quota_prefixes: Tuple[int, ...] = field(default_factory=lambda: (10, 20, 50, 100))
    long_tail_item_ratio: float = 0.1
    kc_coverage_ratio: float = 0.2
    head_ratio: float = 0.2
    long_tail_ratio: float = 0.8

    def validate(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(f"Unknown rerank method: {self.method}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be at least 1.0")
        if self.min_candidates <= 0:
            raise ValueError("min_candidates must be positive")
        if self.quota_top_k <= 0:
            raise ValueError("quota_top_k must be positive")
        if self.min_long_tail_items < 0:
            raise ValueError("min_long_tail_items must be non-negative")
        if self.min_kc_coverage < 0:
            raise ValueError("min_kc_coverage must be non-negative")
        if not self.quota_prefixes:
            raise ValueError("quota_prefixes must not be empty")
        if any(prefix <= 0 for prefix in self.quota_prefixes):
            raise ValueError("quota_prefixes must contain positive integers")
        if not 0 <= self.long_tail_item_ratio <= 1:
            raise ValueError("long_tail_item_ratio must be between 0 and 1")
        if not 0 <= self.kc_coverage_ratio <= 1:
            raise ValueError("kc_coverage_ratio must be between 0 and 1")
        if not 0 <= self.head_ratio <= 1:
            raise ValueError("head_ratio must be between 0 and 1")
        if not 0 <= self.long_tail_ratio <= 1:
            raise ValueError("long_tail_ratio must be between 0 and 1")
        for name in ("lambda_item", "lambda_kc", "beta_item", "beta_kc"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def load_q_matrix(path: Path) -> List[List[int]]:
    matrix: List[List[int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                matrix.append([int(value) for value in line.split(",")])
    return matrix


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


def save_uid_ex_scores(uid_ex_scores: Sequence[Tuple[str, Sequence[float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        with path.open("wb") as fp:
            pickle.dump([(uid, list(scores)) for uid, scores in uid_ex_scores], fp)
        return
    rows = [{"uid": uid, "scores": list(scores)} for uid, scores in uid_ex_scores]
    write_json(rows, path)


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_value = min(float(value) for value in values)
    max_value = max(float(value) for value in values)
    if math.isclose(max_value, min_value):
        return [0.0 for _ in values]
    return [(float(value) - min_value) / (max_value - min_value) for value in values]


def candidate_count_for(scores: Sequence[float], exercise_count: int, config: RerankConfig) -> int:
    requested = max(
        config.top_k,
        config.quota_top_k,
        config.min_candidates,
        math.ceil(config.top_k * config.candidate_multiplier),
    )
    return min(len(scores), exercise_count, requested)


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def bounded_penalty(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(value / max_value)


def score_candidate(
    exercise_idx: int,
    relevance: float,
    concepts: Sequence[int],
    selected_concepts: set[int],
    item_tail_bonus: Sequence[float],
    kc_tail_bonus: Sequence[float],
    item_exposure: Sequence[float],
    kc_exposure: Sequence[float],
    config: RerankConfig,
) -> float:
    score = float(relevance)
    if config.method in {"item", "item_kc", "quota_ratio_hybrid"}:
        item_max_exposure = max(item_exposure) if item_exposure else 0.0
        score += config.lambda_item * float(item_tail_bonus[exercise_idx])
        score -= config.beta_item * bounded_penalty(float(item_exposure[exercise_idx]), float(item_max_exposure))

    if config.method in {"kc", "item_kc", "quota_ratio_hybrid"}:
        if concepts:
            new_concept_share = sum(1 for kc_idx in concepts if kc_idx not in selected_concepts) / len(concepts)
            tail_bonus = average([kc_tail_bonus[kc_idx] for kc_idx in concepts if 0 <= kc_idx < len(kc_tail_bonus)])
            kc_max_exposure = max(kc_exposure) if kc_exposure else 0.0
            exposure_penalty = average(
                [bounded_penalty(float(kc_exposure[kc_idx]), float(kc_max_exposure)) for kc_idx in concepts if 0 <= kc_idx < len(kc_exposure)]
            )
        else:
            new_concept_share = 0.0
            tail_bonus = 0.0
            exposure_penalty = 0.0
        score += config.lambda_kc * (0.5 * tail_bonus + 0.5 * new_concept_share)
        score -= config.beta_kc * exposure_penalty
    return score


def rewrite_scores_by_order(original_scores: Sequence[float], ordered_exercises: Sequence[int]) -> List[float]:
    new_scores = [-1.0 for _ in original_scores]
    base = len(ordered_exercises)
    for rank, exercise_idx in enumerate(ordered_exercises):
        if 0 <= exercise_idx < len(new_scores):
            new_scores[exercise_idx] = float(base - rank)
    return new_scores


def concept_set(q_matrix: Sequence[Sequence[int]], exercise_order: Iterable[int]) -> set[int]:
    concepts: set[int] = set()
    for exercise_idx in exercise_order:
        concepts.update(exercise_concepts(q_matrix, exercise_idx))
    return concepts


def quota_prefixes_for(config: RerankConfig, target_count: int) -> List[int]:
    prefixes = sorted({min(int(prefix), target_count) for prefix in config.quota_prefixes if int(prefix) > 0})
    return [prefix for prefix in prefixes if prefix > 0]


def max_concept_coverage_with_limit(
    original_order: Sequence[int],
    q_matrix: Sequence[Sequence[int]],
    limit: int,
) -> int:
    selected_concepts: set[int] = set()
    selected_set: set[int] = set()
    for _ in range(max(0, min(limit, len(original_order)))):
        best_idx = None
        best_new_count = 0
        for exercise_idx in original_order:
            if exercise_idx in selected_set:
                continue
            new_count = sum(1 for kc_idx in exercise_concepts(q_matrix, exercise_idx) if kc_idx not in selected_concepts)
            if best_idx is None or (new_count, -exercise_idx) > (best_new_count, -best_idx):
                best_idx = exercise_idx
                best_new_count = new_count
        if best_idx is None:
            break
        selected_set.add(best_idx)
        selected_concepts.update(exercise_concepts(q_matrix, best_idx))
        if best_new_count == 0:
            break
    return len(selected_concepts)


def quota_ratio_targets(
    prefix: int,
    original_order: Sequence[int],
    q_matrix: Sequence[Sequence[int]],
    long_tail_items: set[int],
    concept_count: int,
    config: RerankConfig,
) -> Tuple[int, int]:
    long_tail_target = min(
        prefix,
        sum(1 for idx in original_order if idx in long_tail_items),
        int(math.ceil(prefix * config.long_tail_item_ratio)),
    )
    raw_kc_target = int(math.ceil(concept_count * config.kc_coverage_ratio))
    kc_target = min(raw_kc_target, max_concept_coverage_with_limit(original_order, q_matrix, prefix))
    return long_tail_target, kc_target


def quota_fixed_targets(
    prefix: int,
    original_order: Sequence[int],
    q_matrix: Sequence[Sequence[int]],
    long_tail_items: set[int],
    config: RerankConfig,
) -> Tuple[int, int]:
    long_tail_target = min(
        config.min_long_tail_items,
        prefix,
        sum(1 for idx in original_order if idx in long_tail_items),
    )
    kc_target = min(config.min_kc_coverage, max_concept_coverage_with_limit(original_order, q_matrix, prefix))
    return long_tail_target, kc_target


def choose_best_quota_candidate(
    original_order: Sequence[int],
    selected_set: set[int],
    relevance_by_exercise: Dict[int, float],
    q_matrix: Sequence[Sequence[int]],
    selected_concepts: set[int],
    required_items: set[int] | None = None,
    prefer_hybrid_score: bool = False,
    prefer_new_concepts: bool = True,
    item_tail_bonus: Sequence[float] | None = None,
    kc_tail_bonus: Sequence[float] | None = None,
    item_exposure: Sequence[float] | None = None,
    kc_exposure: Sequence[float] | None = None,
    config: RerankConfig | None = None,
) -> int | None:
    best_idx = None
    best_score = None
    for exercise_idx in original_order:
        if exercise_idx in selected_set:
            continue
        if required_items is not None and exercise_idx not in required_items:
            continue
        concepts = exercise_concepts(q_matrix, exercise_idx)
        new_concepts = sum(1 for kc_idx in concepts if kc_idx not in selected_concepts)
        if prefer_hybrid_score and config is not None:
            hybrid_score = score_candidate(
                exercise_idx=exercise_idx,
                relevance=relevance_by_exercise[exercise_idx],
                concepts=concepts,
                selected_concepts=selected_concepts,
                item_tail_bonus=item_tail_bonus or [],
                kc_tail_bonus=kc_tail_bonus or [],
                item_exposure=item_exposure or [],
                kc_exposure=kc_exposure or [],
                config=config,
            )
        else:
            hybrid_score = relevance_by_exercise[exercise_idx]
        if prefer_new_concepts:
            tie_breaker = (new_concepts, hybrid_score, relevance_by_exercise[exercise_idx], -exercise_idx)
        else:
            tie_breaker = (hybrid_score, new_concepts, relevance_by_exercise[exercise_idx], -exercise_idx)
        if best_idx is None or tie_breaker > best_score:
            best_idx = exercise_idx
            best_score = tie_breaker
    return best_idx


def add_selected_exercise(
    exercise_idx: int,
    selected: List[int],
    selected_set: set[int],
    selected_concepts: set[int],
    q_matrix: Sequence[Sequence[int]],
    item_exposure: List[float],
    kc_exposure: List[float],
) -> None:
    selected.append(exercise_idx)
    selected_set.add(exercise_idx)
    if 0 <= exercise_idx < len(item_exposure):
        item_exposure[exercise_idx] += 1.0
    concepts = exercise_concepts(q_matrix, exercise_idx)
    if concepts:
        share = 1.0 / len(concepts)
        for kc_idx in concepts:
            if 0 <= kc_idx < len(kc_exposure):
                kc_exposure[kc_idx] += share
        selected_concepts.update(concepts)


def rerank_one_user_quota_strict(
    scores: Sequence[float],
    q_matrix: Sequence[Sequence[int]],
    long_tail_items: set[int],
    item_exposure: List[float],
    kc_exposure: List[float],
    config: RerankConfig,
) -> List[int]:
    exercise_count = len(q_matrix)
    candidate_count = candidate_count_for(scores, exercise_count, config)
    original_order = top_exercises(scores, candidate_count, exercise_count=exercise_count)
    if not original_order:
        return original_order

    normalized_relevance_values = min_max_normalize([float(scores[idx]) for idx in original_order])
    relevance_by_exercise = dict(zip(original_order, normalized_relevance_values))
    target_count = min(config.top_k, len(original_order))
    quota_prefix_k = min(config.quota_top_k, target_count)
    available_long_tail = {idx for idx in original_order if idx in long_tail_items}
    target_long_tail = min(config.min_long_tail_items, quota_prefix_k, len(available_long_tail))
    available_concepts = concept_set(q_matrix, original_order)
    target_kc_coverage = min(config.min_kc_coverage, len(available_concepts))

    selected: List[int] = []
    selected_set: set[int] = set()
    selected_concepts: set[int] = set()

    while len(selected) < quota_prefix_k and len(selected_set.intersection(long_tail_items)) < target_long_tail:
        exercise_idx = choose_best_quota_candidate(
            original_order=original_order,
            selected_set=selected_set,
            relevance_by_exercise=relevance_by_exercise,
            q_matrix=q_matrix,
            selected_concepts=selected_concepts,
            required_items=available_long_tail,
        )
        if exercise_idx is None:
            break
        add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

    while len(selected) < quota_prefix_k and len(selected_concepts) < target_kc_coverage:
        exercise_idx = choose_best_quota_candidate(
            original_order=original_order,
            selected_set=selected_set,
            relevance_by_exercise=relevance_by_exercise,
            q_matrix=q_matrix,
            selected_concepts=selected_concepts,
        )
        if exercise_idx is None:
            break
        before = len(selected_concepts)
        add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)
        if len(selected_concepts) == before and len(selected_concepts) >= target_kc_coverage:
            break

    for exercise_idx in original_order:
        if len(selected) >= target_count:
            break
        if exercise_idx in selected_set:
            continue
        add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

    selected.extend(exercise_idx for exercise_idx in original_order if exercise_idx not in selected_set)
    return selected


def rerank_one_user_quota_ratio(
    scores: Sequence[float],
    q_matrix: Sequence[Sequence[int]],
    item_popularity_norm: Sequence[float],
    kc_popularity_norm: Sequence[float],
    long_tail_items: set[int],
    item_exposure: List[float],
    kc_exposure: List[float],
    config: RerankConfig,
) -> List[int]:
    exercise_count = len(q_matrix)
    concept_count = max((len(row) for row in q_matrix), default=0)
    candidate_count = candidate_count_for(scores, exercise_count, config)
    original_order = top_exercises(scores, candidate_count, exercise_count=exercise_count)
    if not original_order:
        return original_order

    normalized_relevance_values = min_max_normalize([float(scores[idx]) for idx in original_order])
    relevance_by_exercise = dict(zip(original_order, normalized_relevance_values))
    use_hybrid_score = config.method == "quota_ratio_hybrid"
    item_tail_bonus = [1.0 - float(value) for value in item_popularity_norm]
    kc_tail_bonus = [1.0 - float(value) for value in kc_popularity_norm]
    target_count = min(config.top_k, len(original_order))
    prefixes = quota_prefixes_for(config, target_count)
    if not prefixes:
        prefixes = [target_count]

    selected: List[int] = []
    selected_set: set[int] = set()
    selected_concepts: set[int] = set()

    for prefix in prefixes:
        long_tail_target, kc_target = quota_ratio_targets(
            prefix=prefix,
            original_order=original_order,
            q_matrix=q_matrix,
            long_tail_items=long_tail_items,
            concept_count=concept_count,
            config=config,
        )
        while len(selected) < prefix and len(selected_set.intersection(long_tail_items)) < long_tail_target:
            exercise_idx = choose_best_quota_candidate(
                original_order=original_order,
                selected_set=selected_set,
                relevance_by_exercise=relevance_by_exercise,
                q_matrix=q_matrix,
                selected_concepts=selected_concepts,
                required_items=long_tail_items,
                prefer_hybrid_score=use_hybrid_score,
                prefer_new_concepts=True,
                item_tail_bonus=item_tail_bonus,
                kc_tail_bonus=kc_tail_bonus,
                item_exposure=item_exposure,
                kc_exposure=kc_exposure,
                config=config,
            )
            if exercise_idx is None:
                break
            add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

        while len(selected) < prefix and len(selected_concepts) < kc_target:
            exercise_idx = choose_best_quota_candidate(
                original_order=original_order,
                selected_set=selected_set,
                relevance_by_exercise=relevance_by_exercise,
                q_matrix=q_matrix,
                selected_concepts=selected_concepts,
                prefer_hybrid_score=use_hybrid_score,
                prefer_new_concepts=True,
                item_tail_bonus=item_tail_bonus,
                kc_tail_bonus=kc_tail_bonus,
                item_exposure=item_exposure,
                kc_exposure=kc_exposure,
                config=config,
            )
            if exercise_idx is None:
                break
            add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

        while len(selected) < prefix:
            if use_hybrid_score:
                exercise_idx = choose_best_quota_candidate(
                    original_order=original_order,
                    selected_set=selected_set,
                    relevance_by_exercise=relevance_by_exercise,
                    q_matrix=q_matrix,
                    selected_concepts=selected_concepts,
                    prefer_hybrid_score=True,
                    prefer_new_concepts=False,
                    item_tail_bonus=item_tail_bonus,
                    kc_tail_bonus=kc_tail_bonus,
                    item_exposure=item_exposure,
                    kc_exposure=kc_exposure,
                    config=config,
                )
            else:
                exercise_idx = next((idx for idx in original_order if idx not in selected_set), None)
            if exercise_idx is None:
                break
            add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

    while len(selected) < target_count:
        if use_hybrid_score:
            exercise_idx = choose_best_quota_candidate(
                original_order=original_order,
                selected_set=selected_set,
                relevance_by_exercise=relevance_by_exercise,
                q_matrix=q_matrix,
                selected_concepts=selected_concepts,
                prefer_hybrid_score=True,
                prefer_new_concepts=False,
                item_tail_bonus=item_tail_bonus,
                kc_tail_bonus=kc_tail_bonus,
                item_exposure=item_exposure,
                kc_exposure=kc_exposure,
                config=config,
            )
        else:
            exercise_idx = next((idx for idx in original_order if idx not in selected_set), None)
        if exercise_idx is None:
            break
        add_selected_exercise(exercise_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

    selected.extend(exercise_idx for exercise_idx in original_order if exercise_idx not in selected_set)
    return selected


def rerank_one_user(
    scores: Sequence[float],
    q_matrix: Sequence[Sequence[int]],
    item_popularity_norm: Sequence[float],
    kc_popularity_norm: Sequence[float],
    long_tail_items: set[int],
    item_exposure: List[float],
    kc_exposure: List[float],
    config: RerankConfig,
) -> List[int]:
    exercise_count = len(q_matrix)
    candidate_count = candidate_count_for(scores, exercise_count, config)
    original_order = top_exercises(scores, candidate_count, exercise_count=exercise_count)
    if config.method == "baseline" or not original_order:
        return original_order
    if config.method == "quota_strict":
        return rerank_one_user_quota_strict(scores, q_matrix, long_tail_items, item_exposure, kc_exposure, config)
    if config.method in RATIO_QUOTA_METHODS:
        return rerank_one_user_quota_ratio(
            scores,
            q_matrix,
            item_popularity_norm,
            kc_popularity_norm,
            long_tail_items,
            item_exposure,
            kc_exposure,
            config,
        )

    candidate_scores = {idx: float(scores[idx]) for idx in original_order}
    normalized_relevance_values = min_max_normalize([candidate_scores[idx] for idx in original_order])
    relevance_by_exercise = dict(zip(original_order, normalized_relevance_values))
    item_tail_bonus = [1.0 - float(value) for value in item_popularity_norm]
    kc_tail_bonus = [1.0 - float(value) for value in kc_popularity_norm]

    selected: List[int] = []
    selected_set: set[int] = set()
    selected_concepts: set[int] = set()
    target_count = min(config.top_k, len(original_order))

    while len(selected) < target_count:
        best_idx = None
        best_score = None
        for exercise_idx in original_order:
            if exercise_idx in selected_set:
                continue
            concepts = exercise_concepts(q_matrix, exercise_idx)
            candidate_score = score_candidate(
                exercise_idx=exercise_idx,
                relevance=relevance_by_exercise[exercise_idx],
                concepts=concepts,
                selected_concepts=selected_concepts,
                item_tail_bonus=item_tail_bonus,
                kc_tail_bonus=kc_tail_bonus,
                item_exposure=item_exposure,
                kc_exposure=kc_exposure,
                config=config,
            )
            tie_breaker = (candidate_score, relevance_by_exercise[exercise_idx], -exercise_idx)
            if best_idx is None or tie_breaker > best_score:
                best_idx = exercise_idx
                best_score = tie_breaker

        if best_idx is None:
            break
        add_selected_exercise(best_idx, selected, selected_set, selected_concepts, q_matrix, item_exposure, kc_exposure)

    selected.extend(exercise_idx for exercise_idx in original_order if exercise_idx not in selected_set)
    return selected


def rerank_uid_ex_scores(
    uid_ex_scores: Iterable[Tuple[str, Sequence[float]]],
    q_matrix: Sequence[Sequence[int]],
    item_popularity: Sequence[float],
    config: RerankConfig,
) -> Tuple[List[Tuple[str, List[float]]], Dict[str, Any]]:
    config.validate()
    rows = list(uid_ex_scores)
    exercise_count = len(q_matrix)
    concept_count = max((len(row) for row in q_matrix), default=0)
    item_popularity_values = list(item_popularity[:exercise_count])
    if len(item_popularity_values) < exercise_count:
        item_popularity_values.extend([0.0] * (exercise_count - len(item_popularity_values)))
    item_popularity_norm = min_max_normalize(item_popularity_values)
    kc_popularity = kc_popularity_from_items(item_popularity, q_matrix)
    kc_popularity_norm = min_max_normalize(kc_popularity)
    _, long_tail_items = head_and_long_tail(item_popularity_values, config.head_ratio, config.long_tail_ratio)
    item_exposure = [0.0 for _ in range(exercise_count)]
    kc_exposure = [0.0 for _ in range(concept_count)]
    reranked_rows: List[Tuple[str, List[float]]] = []
    changed_users = 0
    quota_satisfied_users = 0
    quota_long_tail_counts: List[int] = []
    quota_kc_counts: List[int] = []
    effective_quota_long_tail_targets: List[int] = []
    effective_quota_kc_targets: List[int] = []
    quota_prefix_records: Dict[str, Dict[str, List[float]]] = {}

    for uid, scores in rows:
        valid_original_order = top_exercises(scores, min(len(scores), exercise_count), exercise_count=exercise_count)
        reranked_order = rerank_one_user(
            scores=scores,
            q_matrix=q_matrix,
            item_popularity_norm=item_popularity_norm,
            kc_popularity_norm=kc_popularity_norm,
            long_tail_items=long_tail_items,
            item_exposure=item_exposure,
            kc_exposure=kc_exposure,
            config=config,
        )
        full_order = list(reranked_order)
        selected = set(full_order)
        full_order.extend(exercise_idx for exercise_idx in valid_original_order if exercise_idx not in selected)
        full_order.extend(idx for idx in range(len(scores)) if idx not in selected and idx >= exercise_count)
        if full_order[: config.top_k] != valid_original_order[: config.top_k]:
            changed_users += 1
        if config.method in QUOTA_METHODS:
            target_count = min(config.top_k, len(full_order))
            prefixes = [min(config.quota_top_k, target_count)] if config.method == "quota_strict" else quota_prefixes_for(config, target_count)
            candidate_count = candidate_count_for(scores, exercise_count, config)
            original_candidates = top_exercises(scores, candidate_count, exercise_count=exercise_count)
            user_satisfied = True
            for prefix in prefixes:
                quota_prefix = full_order[: min(prefix, len(full_order))]
                if config.method == "quota_strict":
                    effective_long_tail_target, effective_kc_target = quota_fixed_targets(
                        prefix=len(quota_prefix),
                        original_order=original_candidates,
                        q_matrix=q_matrix,
                        long_tail_items=long_tail_items,
                        config=config,
                    )
                else:
                    effective_long_tail_target, effective_kc_target = quota_ratio_targets(
                        prefix=len(quota_prefix),
                        original_order=original_candidates,
                        q_matrix=q_matrix,
                        long_tail_items=long_tail_items,
                        concept_count=concept_count,
                        config=config,
                    )
                long_tail_count = sum(1 for idx in quota_prefix if idx in long_tail_items)
                kc_count = len(concept_set(q_matrix, quota_prefix))
                satisfied = long_tail_count >= effective_long_tail_target and kc_count >= effective_kc_target
                user_satisfied = user_satisfied and satisfied
                bucket = quota_prefix_records.setdefault(
                    str(prefix),
                    {
                        "long_tail_items": [],
                        "kc_coverage": [],
                        "effective_long_tail_target": [],
                        "effective_kc_target": [],
                        "satisfied": [],
                    },
                )
                bucket["long_tail_items"].append(float(long_tail_count))
                bucket["kc_coverage"].append(float(kc_count))
                bucket["effective_long_tail_target"].append(float(effective_long_tail_target))
                bucket["effective_kc_target"].append(float(effective_kc_target))
                bucket["satisfied"].append(1.0 if satisfied else 0.0)
                if prefix == prefixes[0]:
                    quota_long_tail_counts.append(long_tail_count)
                    quota_kc_counts.append(kc_count)
                    effective_quota_long_tail_targets.append(effective_long_tail_target)
                    effective_quota_kc_targets.append(effective_kc_target)
            if user_satisfied:
                quota_satisfied_users += 1
        reranked_rows.append((uid, rewrite_scores_by_order(scores, full_order)))

    quota_prefix_summary = {
        prefix: {
            "average_long_tail_items": round(average(values["long_tail_items"]), 6),
            "average_kc_coverage": round(average(values["kc_coverage"]), 6),
            "average_effective_long_tail_target": round(average(values["effective_long_tail_target"]), 6),
            "average_effective_kc_target": round(average(values["effective_kc_target"]), 6),
            "satisfied_user_ratio": round(average(values["satisfied"]), 6),
        }
        for prefix, values in sorted(quota_prefix_records.items(), key=lambda item: int(item[0]))
    }
    metadata = {
        "method": config.method,
        "config": asdict(config),
        "users": len(rows),
        "exercise_count": exercise_count,
        "concept_count": concept_count,
        "changed_users": changed_users,
        "changed_user_ratio": round(changed_users / len(rows), 6) if rows else 0.0,
        "quota_satisfied_users": quota_satisfied_users if config.method in QUOTA_METHODS else None,
        "quota_satisfied_user_ratio": round(quota_satisfied_users / len(rows), 6) if config.method in QUOTA_METHODS and rows else None,
        "average_quota_long_tail_items": round(average(quota_long_tail_counts), 6) if quota_long_tail_counts else None,
        "average_quota_kc_coverage": round(average(quota_kc_counts), 6) if quota_kc_counts else None,
        "average_effective_quota_long_tail_target": round(average(effective_quota_long_tail_targets), 6)
        if effective_quota_long_tail_targets
        else None,
        "average_effective_quota_kc_target": round(average(effective_quota_kc_targets), 6) if effective_quota_kc_targets else None,
        "quota_prefix_summary": quota_prefix_summary or None,
        "long_tail_item_count": len(long_tail_items),
        "score_semantics": "rank_only_scores; evaluate_recommendations uses these scores for ordering",
    }
    return reranked_rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank 2CKG4ER scores with item/KC exposure fairness.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--method", choices=sorted(VALID_METHODS), default="item_kc")
    parser.add_argument("--top-k", type=int, default=max(DEFAULT_TOP_KS))
    parser.add_argument("--candidate-multiplier", type=float, default=3.0)
    parser.add_argument("--min-candidates", type=int, default=max(DEFAULT_TOP_KS))
    parser.add_argument("--lambda-item", type=float, default=0.3)
    parser.add_argument("--lambda-kc", type=float, default=0.3)
    parser.add_argument("--beta-item", type=float, default=0.1)
    parser.add_argument("--beta-kc", type=float, default=0.1)
    parser.add_argument("--quota-top-k", type=int, default=10)
    parser.add_argument("--min-long-tail-items", type=int, default=1)
    parser.add_argument("--min-kc-coverage", type=int, default=10)
    parser.add_argument("--quota-prefixes", default="10,20,50,100")
    parser.add_argument("--long-tail-item-ratio", type=float, default=0.1)
    parser.add_argument("--kc-coverage-ratio", type=float, default=0.2)
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--long-tail-ratio", type=float, default=0.8)
    parser.add_argument("--popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    return parser.parse_args()


def parse_int_list(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def main() -> None:
    args = parse_args()
    config = RerankConfig(
        method=args.method,
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        min_candidates=args.min_candidates,
        lambda_item=args.lambda_item,
        lambda_kc=args.lambda_kc,
        beta_item=args.beta_item,
        beta_kc=args.beta_kc,
        quota_top_k=args.quota_top_k,
        min_long_tail_items=args.min_long_tail_items,
        min_kc_coverage=args.min_kc_coverage,
        quota_prefixes=parse_int_list(args.quota_prefixes),
        long_tail_item_ratio=args.long_tail_item_ratio,
        kc_coverage_ratio=args.kc_coverage_ratio,
        head_ratio=args.head_ratio,
        long_tail_ratio=args.long_tail_ratio,
    )
    config.validate()
    q_matrix = load_q_matrix(args.data_dir / "Q.txt")
    uid_ex_scores = load_uid_ex_scores(args.scores_file)
    item_popularity, popularity_metadata = load_item_popularity(
        args.data_dir,
        len(q_matrix),
        source=args.popularity_source,
        aggregation=args.popularity_aggregation,
    )
    reranked_rows, metadata = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)
    save_uid_ex_scores(reranked_rows, args.output_file)
    metadata.update(
        {
            "data_dir": str(args.data_dir),
            "scores_file": str(args.scores_file),
            "output_file": str(args.output_file),
            "item_popularity_source": popularity_metadata,
        }
    )
    metadata_file = args.metadata_file or args.output_file.with_suffix(args.output_file.suffix + ".metadata.json")
    write_json(metadata, metadata_file)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
