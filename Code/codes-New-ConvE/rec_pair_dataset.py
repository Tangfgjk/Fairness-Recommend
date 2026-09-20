"""Pairwise recommendation datasets and negative sampling for KPPD."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_loader import relation_kind


Triple = Tuple[int, int, int]


@dataclass(frozen=True)
class RecPair:
    uid_entity_id: int
    pos_ex_entity_id: int
    neg_ex_entity_id: int
    pair_weight: float


class RecPairDataset(Dataset):
    """Build deterministic BPR pairs for the rec relation."""

    def __init__(
        self,
        triples: Sequence[Triple],
        *,
        id2relation: Dict[int, str],
        id2entity: Dict[int, str],
        exercise_entity_ids: Sequence[int],
        rec_relation_id: int,
        distances: np.ndarray | None,
        item_popularity: Sequence[float],
        margin: float = 0.0,
        negative_sampling_mode: str = "random",
        pair_weighting: str = "none",
        seed: int = 2024,
    ) -> None:
        if negative_sampling_mode not in {"random", "popular", "mixed"}:
            raise ValueError("--negative-sampling-mode for RecPairDataset must be one of: random, popular, mixed")
        if pair_weighting not in {"none", "distance_gap"}:
            raise ValueError("--bpr-pair-weighting must be one of: none, distance_gap")
        self.id2entity = id2entity
        self.exercise_entity_ids = list(exercise_entity_ids)
        self.rec_relation_id = int(rec_relation_id)
        self.distances = distances
        self.margin = float(margin)
        self.negative_sampling_mode = negative_sampling_mode
        self.pair_weighting = pair_weighting
        self.seed = int(seed)
        self.ex_entity_to_index = {
            int(entity_id): int(str(id2entity[int(entity_id)])[2:])
            for entity_id in self.exercise_entity_ids
            if str(id2entity[int(entity_id)]).startswith("ex") and str(id2entity[int(entity_id)])[2:].isdigit()
        }
        self.positive_by_user: Dict[int, set[int]] = {}
        self.rec_triples: List[Triple] = []
        for h, r, t in triples:
            if int(r) == self.rec_relation_id or relation_kind(id2relation.get(int(r), "")) == "rec":
                self.rec_triples.append((int(h), int(r), int(t)))
                self.positive_by_user.setdefault(int(h), set()).add(int(t))
        ranked_by_pop = sorted(
            self.exercise_entity_ids,
            key=lambda entity_id: (-float(item_popularity[self.ex_entity_to_index.get(int(entity_id), 0)]), int(entity_id)),
        )
        self.popular_entities = ranked_by_pop or list(self.exercise_entity_ids)

    def __len__(self) -> int:
        return len(self.rec_triples)

    def uid_index(self, entity_id: int) -> int | None:
        name = self.id2entity.get(int(entity_id), "")
        if name.startswith("uid") and name[3:].isdigit():
            return int(name[3:])
        return None

    def distance(self, uid_entity_id: int, ex_entity_id: int) -> float | None:
        if self.distances is None:
            return None
        uid_idx = self.uid_index(uid_entity_id)
        ex_idx = self.ex_entity_to_index.get(int(ex_entity_id))
        if uid_idx is None or ex_idx is None:
            return None
        if not (0 <= uid_idx < self.distances.shape[0] and 0 <= ex_idx < self.distances.shape[1]):
            return None
        return float(self.distances[uid_idx, ex_idx])

    def eligible_negative(self, uid_entity_id: int, pos_ex_entity_id: int, candidate: int) -> bool:
        if candidate in self.positive_by_user.get(int(uid_entity_id), set()):
            return False
        pos_distance = self.distance(uid_entity_id, pos_ex_entity_id)
        neg_distance = self.distance(uid_entity_id, candidate)
        if pos_distance is None or neg_distance is None:
            return True
        return neg_distance > pos_distance + self.margin

    def choose_negative(self, uid_entity_id: int, pos_ex_entity_id: int, idx: int) -> int:
        rng = random.Random(self.seed + idx * 1000003)
        mode = self.negative_sampling_mode
        if mode == "mixed":
            mode = ["random", "popular", "near_boundary"][idx % 3]
        if mode == "popular":
            pool = self.popular_entities
        elif mode == "near_boundary" and self.distances is not None:
            uid_idx = self.uid_index(uid_entity_id)
            if uid_idx is not None and 0 <= uid_idx < self.distances.shape[0]:
                ranked = sorted(
                    self.exercise_entity_ids,
                    key=lambda entity_id: (
                        self.distance(uid_entity_id, int(entity_id)) if self.distance(uid_entity_id, int(entity_id)) is not None else float("inf"),
                        int(entity_id),
                    ),
                )
                pool = ranked
            else:
                pool = self.exercise_entity_ids
        else:
            pool = self.exercise_entity_ids
        for _ in range(200):
            candidate = pool[rng.randrange(len(pool))] if mode != "near_boundary" else pool[_ % len(pool)]
            candidate = int(candidate)
            if self.eligible_negative(uid_entity_id, pos_ex_entity_id, candidate):
                return candidate
        fallback = [int(candidate) for candidate in self.exercise_entity_ids if int(candidate) not in self.positive_by_user.get(int(uid_entity_id), set())]
        if not fallback:
            raise ValueError("No eligible negative exercise for BPR pair")
        return fallback[rng.randrange(len(fallback))]

    def pair_weight(self, uid_entity_id: int, pos_ex_entity_id: int, neg_ex_entity_id: int) -> float:
        if self.pair_weighting != "distance_gap":
            return 1.0
        pos_distance = self.distance(uid_entity_id, pos_ex_entity_id)
        neg_distance = self.distance(uid_entity_id, neg_ex_entity_id)
        if pos_distance is None or neg_distance is None:
            return 1.0
        return max(0.0, float(neg_distance - pos_distance))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _r, pos_t = self.rec_triples[idx]
        neg_t = self.choose_negative(h, pos_t, idx)
        weight = self.pair_weight(h, pos_t, neg_t)
        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(pos_t, dtype=torch.long),
            torch.tensor(neg_t, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )

