"""KPPD popularity nuisance and pedagogical guardrail head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_POPULARITY_DEBIAS_MODES = {"none", "global", "global_personal", "global_personal_need"}


def repair_legacy_context_state(model: nn.Module, state: dict, context_version: int = 1) -> tuple[dict, bool]:
    """Repair v1 identity/fallback buffers without changing learned parameters.

    Train-profile values and existing positive identity mappings must agree with
    the supplied graph. A changed dataset cannot be disguised as this migration.
    """
    head = model.popularity_debias_head
    if head is None:
        return state, False
    if context_version not in {1, 2}:
        raise ValueError(f"Unsupported KPPD context version: {context_version}")
    if context_version == 2:
        return state, False
    prefix = "popularity_debias_head."
    saved_profiles = state[prefix + "user_profiles"]
    saved_map = state[prefix + "uid_entity_to_index"]
    if (saved_profiles.ndim != 2 or saved_profiles.shape[1] != head.user_profiles.shape[1]
            or saved_profiles.shape[0] > head.user_profiles.shape[0]
            or not torch.allclose(saved_profiles.to(head.user_profiles.device), head.user_profiles[:saved_profiles.shape[0]])
            or saved_map.shape != head.uid_entity_to_index.shape):
        raise ValueError("Legacy KPPD training profiles or entity cohort differ from the current dataset")
    saved_map = saved_map.to(head.uid_entity_to_index.device)
    if torch.any((saved_map >= 0) & (saved_map != head.uid_entity_to_index)):
        raise ValueError("Legacy KPPD learner identity mapping differs from the current dataset")
    changed = saved_profiles.shape != head.user_profiles.shape or not torch.equal(saved_map, head.uid_entity_to_index)
    repaired = dict(state)
    repaired[prefix + "user_profiles"] = head.user_profiles.detach().clone()
    repaired[prefix + "uid_entity_to_index"] = head.uid_entity_to_index.detach().clone()
    return repaired, changed


@dataclass(frozen=True)
class PopularityDebiasConfig:
    mode: str = "none"
    beta_global: float = 1.0
    beta_personal: float = 1.0
    beta_need: float = 1.0
    lambda_global: float = 1.0
    lambda_personal: float = 1.0
    personal_hidden_dim: int = 16

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def use_global(self) -> bool:
        return self.mode in {"global", "global_personal", "global_personal_need"}

    @property
    def use_personal(self) -> bool:
        return self.mode in {"global_personal", "global_personal_need"}

    @property
    def use_need(self) -> bool:
        return self.mode == "global_personal_need"


class PopularityDebiasHead(nn.Module):
    """Produce global, personal, and pedagogical logit components."""

    def __init__(self, config: PopularityDebiasConfig, tensors: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        if config.mode not in VALID_POPULARITY_DEBIAS_MODES:
            raise ValueError(f"unknown popularity debias mode: {config.mode}")
        self.config = config
        self.register_buffer("item_popularity_z", tensors["item_popularity_z"].detach().clone().float())
        self.register_buffer("user_profiles", tensors["user_profiles"].detach().clone().float())
        self.register_buffer("pedagogical_relevance", tensors["pedagogical_relevance"].detach().clone().float())
        self.register_buffer("exercise_entity_to_index", tensors["exercise_entity_to_index"].detach().clone().long())
        self.register_buffer("uid_entity_to_index", tensors["uid_entity_to_index"].detach().clone().long())
        self.global_scale_unconstrained = nn.Parameter(torch.zeros(()))
        self.global_intercept = nn.Parameter(torch.zeros(()))
        hidden = max(4, int(config.personal_hidden_dim))
        self.personal_mlp = nn.Sequential(
            nn.Linear(8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _indices(self, h: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex_idx = self.exercise_entity_to_index[t].clamp(min=-1)
        uid_idx = self.uid_entity_to_index[h].clamp(min=-1)
        valid = (ex_idx >= 0) & (uid_idx >= 0)
        return uid_idx, ex_idx, valid

    def components(self, h: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        uid_idx, ex_idx, valid = self._indices(h, t)
        zeros = torch.zeros_like(h, dtype=torch.float32)
        safe_ex = ex_idx.clamp(min=0)
        safe_uid = uid_idx.clamp(min=0)
        item_pop = self.item_popularity_z[safe_ex].to(zeros.device)

        global_pop = F.softplus(self.global_scale_unconstrained) * item_pop + self.global_intercept
        global_pop = torch.where(ex_idx >= 0, global_pop, zeros)

        if self.user_profiles.numel() and self.config.use_personal:
            safe_uid_profile = safe_uid.clamp(max=max(0, self.user_profiles.shape[0] - 1))
            uid_valid = (uid_idx >= 0) & (uid_idx < self.user_profiles.shape[0])
            profiles = self.user_profiles[safe_uid_profile].to(zeros.device)
            personal_input = torch.cat([profiles, item_pop.unsqueeze(1)], dim=1)
            personal_pop = self.personal_mlp(personal_input).squeeze(1)
            personal_pop = torch.where(valid & uid_valid, personal_pop, zeros)
        else:
            personal_pop = zeros

        if self.pedagogical_relevance.numel() and self.config.use_need:
            uid_ok = (uid_idx >= 0) & (uid_idx < self.pedagogical_relevance.shape[0])
            ex_ok = (ex_idx >= 0) & (ex_idx < self.pedagogical_relevance.shape[1])
            relevance_valid = uid_ok & ex_ok
            safe_uid_rel = uid_idx.clamp(min=0, max=max(0, self.pedagogical_relevance.shape[0] - 1))
            safe_ex_rel = ex_idx.clamp(min=0, max=max(0, self.pedagogical_relevance.shape[1] - 1))
            need = self.pedagogical_relevance[safe_uid_rel, safe_ex_rel].to(zeros.device)
            need = torch.where(relevance_valid, need, zeros)
        else:
            need = zeros

        if not self.config.use_global:
            global_pop = zeros
        if not self.config.use_personal:
            personal_pop = zeros
        if not self.config.use_need:
            need = zeros

        return {
            "global_pop": global_pop,
            "personal_pop": personal_pop,
            "need": need,
        }

    def combine(self, core_logits: torch.Tensor, h: torch.Tensor, t: torch.Tensor, score_mode: str = "raw") -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if score_mode not in {"core", "raw", "debiased", "global_pop", "personal_pop", "need"}:
            raise ValueError("score_mode must be one of: core, raw, debiased, global_pop, personal_pop, need")
        components = self.components(h, t)
        global_term = float(self.config.beta_global) * components["global_pop"]
        personal_term = float(self.config.beta_personal) * components["personal_pop"]
        need_term = float(self.config.beta_need) * components["need"]
        raw = core_logits + global_term + personal_term + need_term
        debiased = raw - float(self.config.lambda_global) * global_term - float(self.config.lambda_personal) * personal_term
        if score_mode == "core":
            logits = core_logits
        elif score_mode == "debiased":
            logits = debiased
        elif score_mode == "global_pop":
            logits = global_term
        elif score_mode == "personal_pop":
            logits = personal_term
        elif score_mode == "need":
            logits = need_term
        else:
            logits = raw
        details = {
            **components,
            "global_term": global_term,
            "personal_term": personal_term,
            "need_term": need_term,
            "raw": raw,
            "debiased": debiased,
            "core": core_logits,
        }
        return logits, details

    def metadata(self) -> Dict[str, object]:
        return {
            "mode": self.config.mode,
            "beta_global": float(self.config.beta_global),
            "beta_personal": float(self.config.beta_personal),
            "beta_need": float(self.config.beta_need),
            "lambda_global": float(self.config.lambda_global),
            "lambda_personal": float(self.config.lambda_personal),
            "global_popularity": "softplus(theta_g) * standardized_log1p_train_popularity + c_g",
            "personal_profile_dim": int(self.user_profiles.shape[1]) if self.user_profiles.dim() == 2 else 0,
            "pedagogical_guardrail": "row-minmax relevance from stu2ex_recommend_full_precision",
            "need_is_trainable": False,
            "personal_popularity_interpretation": "descriptive historical popularity profile; not an identified conformity effect",
        }
