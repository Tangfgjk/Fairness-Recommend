"""Shared fairness context for 2CKG4ER item/KC fairness work."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from fairness_metrics import (
    exercise_concepts,
    head_and_long_tail,
    kc_popularity_from_items,
    load_item_popularity_from_triples,
)


def read_q_matrix(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append([int(float(value)) for value in line.replace("\t", ",").split(",") if value != ""])
    return rows


def normalize_distribution(values: Sequence[float], epsilon: float = 1e-8) -> List[float]:
    adjusted = [max(0.0, float(value)) + float(epsilon) for value in values]
    total = float(sum(adjusted))
    if total <= 0:
        return [1.0 / len(adjusted) for _ in adjusted] if adjusted else []
    return [value / total for value in adjusted]


def smoothed_popularity_target(popularity: Sequence[float], gamma: float, epsilon: float = 1e-6) -> List[float]:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    values = [(max(0.0, float(value)) + float(epsilon)) ** float(gamma) for value in popularity]
    return normalize_distribution(values, epsilon=0.0)


def q_matrix_to_fractional_array(q_matrix: Sequence[Sequence[int]]) -> np.ndarray:
    concept_count = max((len(row) for row in q_matrix), default=0)
    matrix = np.zeros((len(q_matrix), concept_count), dtype=np.float32)
    for exercise_idx, row in enumerate(q_matrix):
        concepts = [idx for idx, value in enumerate(row) if int(value) == 1]
        if not concepts:
            continue
        share = 1.0 / len(concepts)
        for concept_idx in concepts:
            matrix[exercise_idx, concept_idx] = share
    return matrix


@dataclass(frozen=True)
class FairnessContext:
    q_matrix: List[List[int]]
    exercise_to_kc: Dict[int, List[int]]
    item_popularity: List[float]
    kc_popularity: List[float]
    item_long_tail_flag: List[int]
    kc_long_tail_flag: List[int]
    item_target_distribution: List[float]
    kc_target_distribution: List[float]
    ex_kc_matrix: np.ndarray
    metadata: Dict[str, Any]

    @property
    def exercise_count(self) -> int:
        return len(self.q_matrix)

    @property
    def kc_count(self) -> int:
        return max((len(row) for row in self.q_matrix), default=0)

    def to_tensors(self, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        return {
            "item_popularity": torch.tensor(self.item_popularity, dtype=torch.float32, device=device),
            "kc_popularity": torch.tensor(self.kc_popularity, dtype=torch.float32, device=device),
            "item_long_tail_flag": torch.tensor(self.item_long_tail_flag, dtype=torch.float32, device=device),
            "kc_long_tail_flag": torch.tensor(self.kc_long_tail_flag, dtype=torch.float32, device=device),
            "item_target_distribution": torch.tensor(self.item_target_distribution, dtype=torch.float32, device=device),
            "kc_target_distribution": torch.tensor(self.kc_target_distribution, dtype=torch.float32, device=device),
            "ex_kc_matrix": torch.tensor(self.ex_kc_matrix, dtype=torch.float32, device=device),
        }

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "exercise_count": self.exercise_count,
            "kc_count": self.kc_count,
            "item_popularity": self.item_popularity,
            "kc_popularity": self.kc_popularity,
            "item_long_tail_flag": self.item_long_tail_flag,
            "kc_long_tail_flag": self.kc_long_tail_flag,
            "item_target_distribution": self.item_target_distribution,
            "kc_target_distribution": self.kc_target_distribution,
            "metadata": self.metadata,
        }


def build_fairness_context(
    data_dir: str | Path,
    head_ratio: float = 0.2,
    long_tail_ratio: float = 0.8,
    target_gamma: float = 0.5,
) -> FairnessContext:
    data_dir = Path(data_dir)
    q_matrix = read_q_matrix(data_dir / "Q.txt")
    item_popularity = load_item_popularity_from_triples(data_dir / "triples.txt", len(q_matrix))
    kc_popularity = kc_popularity_from_items(item_popularity, q_matrix)
    _head_items, long_tail_items = head_and_long_tail(item_popularity, head_ratio, long_tail_ratio)
    _head_kcs, long_tail_kcs = head_and_long_tail(kc_popularity, head_ratio, long_tail_ratio)
    exercise_to_kc = {idx: exercise_concepts(q_matrix, idx) for idx in range(len(q_matrix))}
    item_target = smoothed_popularity_target(item_popularity, target_gamma)
    kc_target = smoothed_popularity_target(kc_popularity, target_gamma)
    ex_kc_matrix = q_matrix_to_fractional_array(q_matrix)
    return FairnessContext(
        q_matrix=q_matrix,
        exercise_to_kc=exercise_to_kc,
        item_popularity=[float(value) for value in item_popularity],
        kc_popularity=[float(value) for value in kc_popularity],
        item_long_tail_flag=[1 if idx in long_tail_items else 0 for idx in range(len(q_matrix))],
        kc_long_tail_flag=[1 if idx in long_tail_kcs else 0 for idx in range(max((len(row) for row in q_matrix), default=0))],
        item_target_distribution=item_target,
        kc_target_distribution=kc_target,
        ex_kc_matrix=ex_kc_matrix,
        metadata={
            "data_dir": str(data_dir),
            "head_ratio": float(head_ratio),
            "long_tail_ratio": float(long_tail_ratio),
            "target_gamma": float(target_gamma),
            "q_matrix_rows": len(q_matrix),
            "q_matrix_cols": max((len(row) for row in q_matrix), default=0),
            "item_popularity_source": str(data_dir / "triples.txt"),
            "item_popularity_relation": "rec",
            "kc_popularity": "fractional_item_popularity_by_q_matrix",
        },
    )


def write_fairness_context_summary(context: FairnessContext, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")


def long_tail_count(flags: Sequence[int]) -> int:
    return int(sum(1 for value in flags if int(value) == 1))


def min_nonzero(values: Sequence[float]) -> float:
    nonzero = [float(value) for value in values if float(value) > 0.0]
    return min(nonzero) if nonzero else 0.0


def context_stats(context: FairnessContext) -> Dict[str, Any]:
    return {
        "exercise_count": context.exercise_count,
        "kc_count": context.kc_count,
        "item_long_tail_count": long_tail_count(context.item_long_tail_flag),
        "kc_long_tail_count": long_tail_count(context.kc_long_tail_flag),
        "item_popularity_min_nonzero": min_nonzero(context.item_popularity),
        "item_popularity_max": max(context.item_popularity) if context.item_popularity else 0.0,
        "kc_popularity_min_nonzero": min_nonzero(context.kc_popularity),
        "kc_popularity_max": max(context.kc_popularity) if context.kc_popularity else 0.0,
        "q_nonzero_edges": int(sum(len(values) for values in context.exercise_to_kc.values())),
        "target_gamma": context.metadata.get("target_gamma"),
    }

