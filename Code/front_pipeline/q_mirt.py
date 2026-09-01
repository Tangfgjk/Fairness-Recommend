from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class QConstrainedMIRT(nn.Module):
    """MIRT with a fixed Q-mask that anchors latent dimensions to knowledge concepts."""

    def __init__(self, train_user_count: int, q_matrix: torch.Tensor, q_normalization: str = "mean") -> None:
        super().__init__()
        if q_matrix.ndim != 2:
            raise ValueError("q_matrix must have shape [exercise_count, concept_count]")
        q = (q_matrix > 0).float()
        if torch.any(q.sum(dim=1) <= 0):
            raise ValueError("Every exercise must be linked to at least one concept in Q")
        if q_normalization not in {"none", "mean"}:
            raise ValueError("q_normalization must be 'none' or 'mean'")

        exercise_count, concept_count = q.shape
        self.exercise_count = int(exercise_count)
        self.concept_count = int(concept_count)
        self.q_normalization = q_normalization
        self.theta = nn.Embedding(int(train_user_count), self.concept_count)
        self.raw_a = nn.Embedding(self.exercise_count, self.concept_count)
        self.b = nn.Embedding(self.exercise_count, 1)
        self.register_buffer("q_matrix", q)
        self.register_buffer("q_counts", q.sum(dim=1, keepdim=True).clamp_min(1.0))
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.raw_a.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.b.weight)

    def effective_a(self) -> torch.Tensor:
        values = self.q_matrix * functional.softplus(self.raw_a.weight)
        if self.q_normalization == "mean":
            values = values / self.q_counts
        return values

    def logits(self, local_user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        theta = self.theta(local_user_ids)
        a = self.effective_a()[item_ids]
        b = self.b(item_ids).squeeze(-1)
        return torch.sum(theta * a, dim=-1) - b

    def forward(self, local_user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(local_user_ids, item_ids))

    def logits_with_theta(self, theta: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        a = self.effective_a()[item_ids]
        b = self.b(item_ids).squeeze(-1)
        return torch.sum(theta * a, dim=-1) - b

    @torch.no_grad()
    def set_unseen_item_fallbacks(self, observed_item_ids: torch.Tensor) -> dict[str, int]:
        observed = torch.zeros(self.exercise_count, dtype=torch.bool, device=self.q_matrix.device)
        observed[torch.unique(observed_item_ids)] = True
        unseen = ~observed
        if not unseen.any():
            return {"observed_items": int(observed.sum().item()), "fallback_items": 0}

        raw_a = functional.softplus(self.raw_a.weight.detach())
        active_mask = self.q_matrix > 0
        fallback = torch.zeros_like(raw_a)
        global_values = raw_a[observed][active_mask[observed]]
        global_median = torch.median(global_values) if global_values.numel() else torch.tensor(1.0, device=raw_a.device)
        for concept in range(self.concept_count):
            candidates = raw_a[observed, concept][active_mask[observed, concept]]
            value = torch.median(candidates) if candidates.numel() else global_median
            fallback[:, concept] = value
        target_raw = torch.where(active_mask, fallback, torch.zeros_like(fallback))
        inverse_softplus = torch.where(
            target_raw > 1e-8,
            torch.log(torch.expm1(target_raw).clamp_min(1e-8)),
            torch.full_like(target_raw, -20.0),
        )
        self.raw_a.weight[unseen] = inverse_softplus[unseen]
        b_median = torch.median(self.b.weight.detach()[observed]) if observed.any() else torch.tensor(0.0, device=raw_a.device)
        self.b.weight[unseen] = b_median
        return {"observed_items": int(observed.sum().item()), "fallback_items": int(unseen.sum().item())}
