"""Popularity debiasing context for KPPD-2CKG4ER."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from fairness_context import load_item_popularity, read_q_matrix, resolve_raw_dir
from fairness_metrics import parse_exercise_id


def parse_uid(value: str) -> int | None:
    value = str(value or "").strip()
    if value.startswith("uid") and value[3:].isdigit():
        return int(value[3:])
    return None


def standardize(values: Sequence[float]) -> tuple[List[float], Dict[str, float]]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    if std <= 1e-8:
        return [0.0 for _ in values], {"mean": mean, "std": std}
    return [float((v - mean) / std) for v in arr], {"mean": mean, "std": std}


def head_mid_tail_sets(popularity: Sequence[float], head_ratio: float = 0.2, tail_ratio: float = 0.2) -> tuple[set[int], set[int], set[int]]:
    if head_ratio < 0 or tail_ratio < 0 or head_ratio + tail_ratio > 1:
        raise ValueError("head_ratio and tail_ratio must be non-negative and sum to at most 1")
    count = len(popularity)
    ranked = sorted(range(count), key=lambda idx: (-float(popularity[idx]), idx))
    head_count = int(math.ceil(count * head_ratio))
    tail_count = int(math.ceil(count * tail_ratio))
    head = set(ranked[:head_count])
    tail = set(ranked[count - tail_count :]) if tail_count else set()
    mid = set(ranked[head_count : count - tail_count])
    return head, mid, tail


def normalize_distance_rows(distances: np.ndarray) -> np.ndarray:
    if distances.ndim != 2:
        raise ValueError("distance matrix must be two-dimensional")
    result = np.zeros_like(distances, dtype=np.float32)
    for row_idx in range(distances.shape[0]):
        row = distances[row_idx].astype(np.float64)
        finite = np.isfinite(row)
        if not finite.any():
            continue
        row_min = float(row[finite].min())
        row_max = float(row[finite].max())
        spread = row_max - row_min
        if spread <= 1e-12:
            result[row_idx, finite] = 1.0
        else:
            result[row_idx, finite] = 1.0 - ((row[finite] - row_min) / spread)
    return result


@dataclass(frozen=True)
class PopularityDebiasContext:
    exercise_count: int
    item_popularity: List[float]
    item_popularity_z: List[float]
    user_profiles: np.ndarray
    pedagogical_relevance: np.ndarray
    metadata: Dict[str, Any]

    def to_tensors(
        self,
        *,
        entity2id: Dict[str, int],
        exercise_entity_ids: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        nentity = max(entity2id.values(), default=-1) + 1
        ex_entity_to_index = torch.full((nentity,), -1, dtype=torch.long, device=device)
        for ex_idx, entity_id in enumerate(exercise_entity_ids.detach().cpu().tolist()):
            if 0 <= int(entity_id) < nentity:
                ex_entity_to_index[int(entity_id)] = int(ex_idx)

        uid_entity_to_index = torch.full((nentity,), -1, dtype=torch.long, device=device)
        for name, entity_id in entity2id.items():
            uid_idx = parse_uid(name)
            if uid_idx is not None and 0 <= uid_idx < self.user_profiles.shape[0]:
                uid_entity_to_index[int(entity_id)] = int(uid_idx)

        return {
            "item_popularity_z": torch.tensor(self.item_popularity_z, dtype=torch.float32, device=device),
            "user_profiles": torch.tensor(self.user_profiles, dtype=torch.float32, device=device),
            "pedagogical_relevance": torch.tensor(self.pedagogical_relevance, dtype=torch.float32, device=device),
            "exercise_entity_to_index": ex_entity_to_index,
            "uid_entity_to_index": uid_entity_to_index,
        }


def build_user_popularity_profiles(
    data_dir: Path,
    item_popularity_z: Sequence[float],
    *,
    head: set[int],
    mid: set[int],
    tail: set[int],
) -> tuple[np.ndarray, Dict[str, Any]]:
    raw_dir = resolve_raw_dir(data_dir)
    if raw_dir is None:
        return np.zeros((0, 7), dtype=np.float32), {"source": "missing_raw_interactions"}
    interactions_path = raw_dir / "interactions_all.csv"
    by_uid: Dict[int, List[int]] = {}
    rows_seen = 0
    train_rows = 0
    with interactions_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows_seen += 1
            split = str(row.get("split", "train")).strip().lower()
            if split and split != "train":
                continue
            train_rows += 1
            uid_idx = parse_uid(row.get("entity_id") or row.get("uid") or "")
            ex_idx = parse_exercise_id(row.get("entity_exercise_id", ""))
            if ex_idx is None:
                ex_idx = parse_exercise_id(row.get("question", ""))
            if uid_idx is None or ex_idx is None or not 0 <= ex_idx < len(item_popularity_z):
                continue
            by_uid.setdefault(uid_idx, []).append(ex_idx)

    max_uid = max(by_uid.keys(), default=-1)
    profiles = np.zeros((max_uid + 1, 7), dtype=np.float32)
    observed_profiles: List[np.ndarray] = []
    for uid_idx, exercises in by_uid.items():
        values = np.asarray([float(item_popularity_z[ex]) for ex in exercises], dtype=np.float32)
        count = max(1, len(exercises))
        profile = np.asarray(
            [
                float(values.mean()) if values.size else 0.0,
                float(values.std()) if values.size else 0.0,
                sum(1 for ex in exercises if ex in head) / count,
                sum(1 for ex in exercises if ex in mid) / count,
                sum(1 for ex in exercises if ex in tail) / count,
                math.log1p(len(exercises)),
                1.0,
            ],
            dtype=np.float32,
        )
        profiles[uid_idx] = profile
        observed_profiles.append(profile)

    if observed_profiles:
        neutral = np.stack(observed_profiles, axis=0).mean(axis=0)
        neutral[5] = 0.0
        neutral[6] = 0.0
        for uid_idx in range(profiles.shape[0]):
            if profiles[uid_idx, 6] <= 0.0:
                profiles[uid_idx] = neutral

    return profiles, {
        "source": "train_interactions",
        "path": str(interactions_path),
        "rows_seen": rows_seen,
        "train_rows": train_rows,
        "profile_dim": 7,
        "profile_fields": [
            "mean_pop_z",
            "std_pop_z",
            "head20_share",
            "mid60_share",
            "tail20_share",
            "log_history_length",
            "history_available",
        ],
    }


def load_pedagogical_relevance(data_dir: Path, exercise_count: int) -> tuple[np.ndarray, Dict[str, Any]]:
    path = data_dir / "stu2ex_recommend_full_precision.json"
    if not path.exists():
        return np.zeros((0, exercise_count), dtype=np.float32), {"source": "missing_distance_matrix"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    distances = np.asarray(payload, dtype=np.float32)
    if distances.ndim != 2:
        raise ValueError(f"distance matrix must be two-dimensional: {path}")
    if distances.shape[1] < exercise_count:
        padded = np.zeros((distances.shape[0], exercise_count), dtype=np.float32)
        padded[:, : distances.shape[1]] = distances
        distances = padded
    elif distances.shape[1] > exercise_count:
        distances = distances[:, :exercise_count]
    relevance = normalize_distance_rows(distances)
    return relevance, {
        "source": "stu2ex_recommend_full_precision",
        "path": str(path),
        "normalization": "row_minmax_relevance_1_minus_scaled_distance",
        "shape": list(relevance.shape),
    }


def build_popularity_debias_context(
    data_dir: str | Path,
    *,
    popularity_source: str = "train_interactions",
    popularity_aggregation: str = "unique_users",
    head_ratio: float = 0.2,
    tail_ratio: float = 0.2,
) -> PopularityDebiasContext:
    data_dir = Path(data_dir)
    q_matrix = read_q_matrix(data_dir / "Q.txt")
    item_popularity, popularity_metadata = load_item_popularity(
        data_dir,
        len(q_matrix),
        source=popularity_source,
        aggregation=popularity_aggregation,
    )
    item_log = [math.log1p(max(0.0, float(value))) for value in item_popularity]
    item_popularity_z, standardization = standardize(item_log)
    head, mid, tail = head_mid_tail_sets(item_popularity, head_ratio=head_ratio, tail_ratio=tail_ratio)
    profiles, profile_metadata = build_user_popularity_profiles(
        data_dir,
        item_popularity_z,
        head=head,
        mid=mid,
        tail=tail,
    )
    pedagogical_relevance, relevance_metadata = load_pedagogical_relevance(data_dir, len(q_matrix))
    return PopularityDebiasContext(
        exercise_count=len(q_matrix),
        item_popularity=[float(value) for value in item_popularity],
        item_popularity_z=item_popularity_z,
        user_profiles=profiles,
        pedagogical_relevance=pedagogical_relevance,
        metadata={
            "data_dir": str(data_dir),
            "popularity_source": popularity_metadata,
            "log_pop_standardization": standardization,
            "bucket_definition": {
                "head": f"top {head_ratio:.2f} by p_data popularity",
                "mid": f"middle {1.0 - head_ratio - tail_ratio:.2f} by p_data popularity",
                "tail": f"bottom {tail_ratio:.2f} by p_data popularity",
                "note": "Head20/Mid60/Tail20 are distinct from LongTail80 visibility metrics.",
            },
            "user_profile": profile_metadata,
            "pedagogical_relevance": relevance_metadata,
        },
    )

