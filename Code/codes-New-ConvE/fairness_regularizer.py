"""Differentiable expected-exposure fairness regularizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

VALID_CANDIDATE_MODES = {"random", "popular", "top_score", "mixed"}


@dataclass(frozen=True)
class FairnessRegularizerConfig:
    loss_type: str = "none"
    alpha_item: float = 0.0
    alpha_kc: float = 0.0
    loss_scale: float = 1.0
    temperature: float = 1.0
    candidate_size: int = 150
    candidate_mode: str = "random"
    target_gamma: float = 0.5

    @property
    def enabled(self) -> bool:
        return (
            self.loss_type == "expected_exposure"
            and (float(self.alpha_item) > 0.0 or float(self.alpha_kc) > 0.0)
        )


class FairnessRegularizer:
    """Compute item/KC expected exposure losses over sampled exercise candidates."""

    def __init__(
        self,
        config: FairnessRegularizerConfig,
        rec_relation_id: int,
        exercise_entity_ids: torch.Tensor,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        if config.loss_type not in {"none", "expected_exposure"}:
            raise ValueError("--fairness-loss must be one of: none, expected_exposure")
        if config.loss_scale <= 0:
            raise ValueError("--fairness-loss-scale must be positive")
        if config.temperature <= 0:
            raise ValueError("--fairness-temperature must be positive")
        if config.candidate_size <= 0:
            raise ValueError("--fairness-candidate-size must be positive")
        if config.candidate_mode not in VALID_CANDIDATE_MODES:
            raise ValueError(f"--fairness-candidate-mode must be one of: {sorted(VALID_CANDIDATE_MODES)}")
        self.config = config
        self.rec_relation_id = int(rec_relation_id)
        self.exercise_entity_ids = exercise_entity_ids
        self.item_popularity = tensors.get("item_popularity", tensors["item_target_distribution"])
        self.item_target = tensors["item_target_distribution"]
        self.kc_target = tensors["kc_target_distribution"]
        self.ex_kc_matrix = tensors["ex_kc_matrix"]

    def sample_candidate_indices(self) -> torch.Tensor:
        exercise_count = int(self.exercise_entity_ids.numel())
        sample_count = min(exercise_count, int(self.config.candidate_size))
        return torch.randperm(exercise_count, device=self.exercise_entity_ids.device)[:sample_count]

    def popular_candidate_indices(self) -> torch.Tensor:
        exercise_count = int(self.exercise_entity_ids.numel())
        sample_count = min(exercise_count, int(self.config.candidate_size))
        popularity = self.item_popularity[:exercise_count].to(self.exercise_entity_ids.device)
        return torch.argsort(popularity, descending=True)[:sample_count]

    def top_score_candidate_indices(self, model, unique_users: torch.Tensor, sample_count: int | None = None) -> torch.Tensor:
        exercise_count = int(self.exercise_entity_ids.numel())
        sample_count = min(exercise_count, int(sample_count or self.config.candidate_size))
        relation_ids = torch.full(
            (unique_users.numel(),),
            self.rec_relation_id,
            dtype=torch.long,
            device=unique_users.device,
        )
        with torch.no_grad():
            scores = model.score_tails(unique_users, relation_ids, self.exercise_entity_ids)
            mean_scores = scores.mean(dim=0)
            return torch.topk(mean_scores, k=sample_count, largest=True).indices

    def ordered_unique(self, indices: torch.Tensor) -> torch.Tensor:
        seen: set[int] = set()
        values: list[int] = []
        for value in indices.detach().cpu().tolist():
            idx = int(value)
            if idx not in seen:
                seen.add(idx)
                values.append(idx)
        return torch.tensor(values, dtype=torch.long, device=self.exercise_entity_ids.device)

    def fill_candidates(self, indices: torch.Tensor, sample_count: int) -> torch.Tensor:
        unique = self.ordered_unique(indices)
        if unique.numel() >= sample_count:
            return unique[:sample_count]
        random_order = torch.randperm(int(self.exercise_entity_ids.numel()), device=self.exercise_entity_ids.device)
        filled = self.ordered_unique(torch.cat([unique, random_order]))
        return filled[:sample_count]

    def select_candidate_indices(self, model, unique_users: torch.Tensor) -> torch.Tensor:
        mode = self.config.candidate_mode
        exercise_count = int(self.exercise_entity_ids.numel())
        sample_count = min(exercise_count, int(self.config.candidate_size))
        if mode == "random":
            return self.sample_candidate_indices()
        if mode == "popular":
            return self.popular_candidate_indices()
        if mode == "top_score":
            return self.top_score_candidate_indices(model, unique_users, sample_count=sample_count)

        part = max(1, sample_count // 3)
        top_score_indices = self.top_score_candidate_indices(model, unique_users, sample_count=part)
        popular_indices = self.popular_candidate_indices()[:part]
        random_indices = torch.randperm(exercise_count, device=self.exercise_entity_ids.device)[: max(1, sample_count - 2 * part)]
        return self.fill_candidates(torch.cat([top_score_indices, popular_indices, random_indices]), sample_count)

    @staticmethod
    def normalize(values: torch.Tensor) -> torch.Tensor:
        total = values.sum()
        if bool((total <= 0).detach().cpu().item()):
            return torch.full_like(values, 1.0 / max(1, values.numel()))
        return values / torch.clamp(total, min=1e-8)

    def __call__(self, model, user_entity_ids: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self.config.enabled or user_entity_ids.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=self.exercise_entity_ids.device)
            return zero, {"item_fair_loss": 0.0, "kc_fair_loss": 0.0, "fair_loss": 0.0}

        unique_users = torch.unique(user_entity_ids)
        if unique_users.numel() < 2:
            zero = torch.zeros((), dtype=torch.float32, device=self.exercise_entity_ids.device)
            return zero, {"item_fair_loss": 0.0, "kc_fair_loss": 0.0, "fair_loss": 0.0}
        candidate_indices = self.select_candidate_indices(model, unique_users)
        candidate_entity_ids = self.exercise_entity_ids[candidate_indices]
        relation_ids = torch.full(
            (unique_users.numel(),),
            self.rec_relation_id,
            dtype=torch.long,
            device=unique_users.device,
        )
        scores = model.score_tails(unique_users, relation_ids, candidate_entity_ids)
        probabilities = torch.softmax(scores / float(self.config.temperature), dim=1)

        expected_item = probabilities.mean(dim=0)
        item_target = self.normalize(self.item_target[candidate_indices].to(expected_item.device))
        item_fair_loss = F.mse_loss(expected_item, item_target)

        ex_kc = self.ex_kc_matrix[candidate_indices].to(probabilities.device)
        expected_kc = probabilities.matmul(ex_kc).mean(dim=0)
        expected_kc = self.normalize(expected_kc)
        kc_target = self.normalize(self.kc_target.to(expected_kc.device))
        kc_fair_loss = F.mse_loss(expected_kc, kc_target) if expected_kc.numel() else torch.zeros_like(item_fair_loss)

        fair_loss = (
            float(self.config.alpha_item) * item_fair_loss
            + float(self.config.alpha_kc) * kc_fair_loss
        ) * float(self.config.loss_scale)
        details = {
            "item_fair_loss": float(item_fair_loss.detach().cpu().item()),
            "kc_fair_loss": float(kc_fair_loss.detach().cpu().item()),
            "fair_loss": float(fair_loss.detach().cpu().item()),
            "fairness_loss_scale": float(self.config.loss_scale),
            "candidate_mode": self.config.candidate_mode,
        }
        return fair_loss, details

    def metadata(self) -> Dict[str, object]:
        return {
            "loss_type": self.config.loss_type,
            "enabled": self.config.enabled,
            "alpha_item": float(self.config.alpha_item),
            "alpha_kc": float(self.config.alpha_kc),
            "loss_scale": float(self.config.loss_scale),
            "temperature": float(self.config.temperature),
            "candidate_size": int(self.config.candidate_size),
            "candidate_mode": self.config.candidate_mode,
            "target_gamma": float(self.config.target_gamma),
            "rec_relation_id": self.rec_relation_id,
            "exercise_count": int(self.exercise_entity_ids.numel()),
            "kc_count": int(self.kc_target.numel()),
        }
