"""Differentiable expected-exposure fairness regularizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

VALID_CANDIDATE_MODES = {"random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"}
VALID_EXPOSURE_PROXIES = {"softmax", "sigmoid_topk"}
VALID_FAIRNESS_DISTANCES = {"mse", "l1", "kl_target_model", "js"}


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
    exposure_proxy: str = "softmax"
    surrogate_top_k: int = 10
    distance: str = "mse"
    top_score_ratio: float = 0.5
    popular_ratio: float = 0.25

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
        if config.exposure_proxy not in VALID_EXPOSURE_PROXIES:
            raise ValueError(f"--fairness-exposure-proxy must be one of: {sorted(VALID_EXPOSURE_PROXIES)}")
        if config.surrogate_top_k <= 0:
            raise ValueError("--fairness-surrogate-k must be positive")
        if config.distance not in VALID_FAIRNESS_DISTANCES:
            raise ValueError(f"--fairness-distance must be one of: {sorted(VALID_FAIRNESS_DISTANCES)}")
        if not 0 <= config.top_score_ratio <= 1:
            raise ValueError("--fairness-top-score-ratio must be between 0 and 1")
        if not 0 <= config.popular_ratio <= 1:
            raise ValueError("--fairness-popular-ratio must be between 0 and 1")
        if config.top_score_ratio + config.popular_ratio > 1:
            raise ValueError("--fairness-top-score-ratio + --fairness-popular-ratio must be <= 1")
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

    def score_tails_logits(self, model, users: torch.Tensor, relation_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "score_tails_logits"):
            return model.score_tails_logits(users, relation_ids, tail_ids)
        probabilities = model.score_tails(users, relation_ids, tail_ids)
        return torch.logit(probabilities.clamp(min=1e-6, max=1 - 1e-6))

    def score_tail_matrix_logits(
        self,
        model,
        users: torch.Tensor,
        relation_ids: torch.Tensor,
        candidate_entity_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(model, "score_tail_matrix_logits"):
            return model.score_tail_matrix_logits(users, relation_ids, candidate_entity_ids)
        rows = [
            self.score_tails_logits(
                model,
                users[row : row + 1],
                relation_ids[row : row + 1],
                candidate_entity_ids[row],
            )[0]
            for row in range(users.numel())
        ]
        return torch.stack(rows, dim=0)

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
            scores = self.score_tails_logits(model, unique_users, relation_ids, self.exercise_entity_ids)
            mean_scores = scores.mean(dim=0)
            return torch.topk(mean_scores, k=sample_count, largest=True).indices

    def top_score_user_candidate_indices(self, model, unique_users: torch.Tensor, sample_count: int | None = None) -> torch.Tensor:
        exercise_count = int(self.exercise_entity_ids.numel())
        sample_count = min(exercise_count, int(sample_count or self.config.candidate_size))
        relation_ids = torch.full(
            (unique_users.numel(),),
            self.rec_relation_id,
            dtype=torch.long,
            device=unique_users.device,
        )
        with torch.no_grad():
            scores = self.score_tails_logits(model, unique_users, relation_ids, self.exercise_entity_ids)
            return torch.topk(scores, k=sample_count, dim=1, largest=True).indices

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

    def fill_candidate_row(self, row_indices: torch.Tensor, sample_count: int) -> torch.Tensor:
        unique = self.ordered_unique(row_indices)
        if unique.numel() >= sample_count:
            return unique[:sample_count]
        random_order = torch.randperm(int(self.exercise_entity_ids.numel()), device=self.exercise_entity_ids.device)
        return self.fill_candidates(torch.cat([unique, random_order]), sample_count)

    def mixed_user_candidate_indices(self, model, unique_users: torch.Tensor, sample_count: int) -> torch.Tensor:
        top_count = int(round(sample_count * float(self.config.top_score_ratio)))
        popular_count = int(round(sample_count * float(self.config.popular_ratio)))
        top_count = min(sample_count, max(1, top_count))
        popular_count = min(sample_count - top_count, max(0, popular_count))
        random_count = max(0, sample_count - top_count - popular_count)

        top_indices = self.top_score_user_candidate_indices(model, unique_users, sample_count=top_count)
        popular_indices = self.popular_candidate_indices()[:popular_count]
        rows = []
        exercise_count = int(self.exercise_entity_ids.numel())
        for row in range(unique_users.numel()):
            parts = [top_indices[row]]
            if popular_count > 0:
                parts.append(popular_indices)
            if random_count > 0:
                parts.append(torch.randperm(exercise_count, device=self.exercise_entity_ids.device)[:random_count])
            rows.append(self.fill_candidate_row(torch.cat(parts), sample_count))
        return torch.stack(rows, dim=0)

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
        if mode == "top_score_user":
            return self.top_score_user_candidate_indices(model, unique_users, sample_count=sample_count)
        if mode == "mixed_user":
            return self.mixed_user_candidate_indices(model, unique_users, sample_count)

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

    def exposure_weights(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.exposure_proxy == "softmax":
            return torch.softmax(logits / float(self.config.temperature), dim=1)
        k = min(int(self.config.surrogate_top_k), int(logits.shape[1]))
        kth_scores = torch.topk(logits, k=k, dim=1, largest=True).values[:, -1].detach()
        gates = torch.sigmoid((logits - kth_scores.unsqueeze(1)) / float(self.config.temperature))
        row_totals = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return gates * (float(k) / row_totals)

    def distribution_loss(self, exposure: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        exposure = self.normalize(exposure).clamp(min=1e-8)
        target = self.normalize(target).clamp(min=1e-8)
        if self.config.distance == "mse":
            return F.mse_loss(exposure, target)
        if self.config.distance == "l1":
            return F.l1_loss(exposure, target)
        if self.config.distance == "kl_target_model":
            return torch.sum(target * (torch.log(target) - torch.log(exposure)))
        midpoint = self.normalize(0.5 * (exposure + target)).clamp(min=1e-8)
        kl_exposure = torch.sum(exposure * (torch.log(exposure) - torch.log(midpoint)))
        kl_target = torch.sum(target * (torch.log(target) - torch.log(midpoint)))
        return 0.5 * (kl_exposure + kl_target)

    def candidate_logits(
        self,
        model,
        unique_users: torch.Tensor,
        relation_ids: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_indices.dim() == 1:
            candidate_entity_ids = self.exercise_entity_ids[candidate_indices]
            logits = self.score_tails_logits(model, unique_users, relation_ids, candidate_entity_ids)
            expanded_indices = candidate_indices.unsqueeze(0).expand(unique_users.numel(), -1)
            return logits, expanded_indices
        candidate_entity_ids = self.exercise_entity_ids[candidate_indices]
        logits = self.score_tail_matrix_logits(model, unique_users, relation_ids, candidate_entity_ids)
        return logits, candidate_indices

    def item_exposure_distribution(
        self,
        candidate_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exercise_count = int(self.exercise_entity_ids.numel())
        exposure = torch.zeros(exercise_count, dtype=weights.dtype, device=weights.device)
        exposure.scatter_add_(0, candidate_indices.reshape(-1), weights.reshape(-1))
        exposure = exposure / max(1, weights.shape[0])
        support = exposure > 0
        if not bool(support.any().detach().cpu().item()):
            support = torch.ones_like(exposure, dtype=torch.bool)
        item_distribution = self.normalize(exposure[support])
        item_target = self.normalize(self.item_target[:exercise_count].to(weights.device)[support])
        return exposure, item_distribution, item_target

    def __call__(self, model, user_entity_ids: torch.Tensor) -> tuple[torch.Tensor, Dict[str, object]]:
        if not self.config.enabled or user_entity_ids.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=self.exercise_entity_ids.device)
            return zero, {"item_fair_loss": 0.0, "kc_fair_loss": 0.0, "fair_loss": 0.0}

        unique_users = torch.unique(user_entity_ids)
        if unique_users.numel() < 2:
            zero = torch.zeros((), dtype=torch.float32, device=self.exercise_entity_ids.device)
            return zero, {"item_fair_loss": 0.0, "kc_fair_loss": 0.0, "fair_loss": 0.0}
        candidate_indices = self.select_candidate_indices(model, unique_users)
        relation_ids = torch.full(
            (unique_users.numel(),),
            self.rec_relation_id,
            dtype=torch.long,
            device=unique_users.device,
        )
        logits, expanded_candidate_indices = self.candidate_logits(model, unique_users, relation_ids, candidate_indices)
        weights = self.exposure_weights(logits)

        item_exposure, expected_item, item_target = self.item_exposure_distribution(expanded_candidate_indices, weights)
        item_fair_loss = self.distribution_loss(expected_item, item_target)

        ex_kc = self.ex_kc_matrix[: int(self.exercise_entity_ids.numel())].to(weights.device)
        expected_kc = item_exposure.matmul(ex_kc)
        expected_kc = self.normalize(expected_kc)
        kc_target = self.normalize(self.kc_target.to(expected_kc.device))
        kc_fair_loss = self.distribution_loss(expected_kc, kc_target) if expected_kc.numel() else torch.zeros_like(item_fair_loss)

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
            "candidate_granularity": "per_user" if candidate_indices.dim() == 2 else "shared",
            "candidate_support_size": int(torch.unique(expanded_candidate_indices.detach()).numel()),
            "exposure_proxy": self.config.exposure_proxy,
            "surrogate_top_k": int(self.config.surrogate_top_k),
            "fairness_distance": self.config.distance,
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
            "exposure_proxy": self.config.exposure_proxy,
            "surrogate_top_k": int(self.config.surrogate_top_k),
            "fairness_distance": self.config.distance,
            "top_score_ratio": float(self.config.top_score_ratio),
            "popular_ratio": float(self.config.popular_ratio),
            "rec_relation_id": self.rec_relation_id,
            "exercise_count": int(self.exercise_entity_ids.numel()),
            "kc_count": int(self.kc_target.numel()),
        }
