"""Training objective helpers for fair 2CKG4ER variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from feature_loader import relation_kind


Triple = Tuple[int, int, int]


@dataclass(frozen=True)
class RelationWeightConfig:
    mode: str = "none"
    rec: float = 1.0
    mlkc: float = 1.0
    exfr: float = 1.0
    exkc: float = 1.0
    other: float = 1.0


def canonical_relation_group(relation_name: str) -> str:
    kind = relation_kind(relation_name)
    if kind in {"ex_has_kc", "kc_has_ex"}:
        return "exkc"
    if kind in {"rec", "mlkc", "exfr"}:
        return kind
    return "other"


def relation_group_counts(triples: Sequence[Triple], id2relation: Dict[int, str]) -> Dict[str, int]:
    counts = {"rec": 0, "mlkc": 0, "exfr": 0, "exkc": 0, "other": 0}
    for _h, relation_id, _t in triples:
        group = canonical_relation_group(id2relation.get(int(relation_id), ""))
        counts[group] = counts.get(group, 0) + 1
    return counts


def balanced_group_weights(counts: Dict[str, int]) -> Dict[str, float]:
    active = {group: count for group, count in counts.items() if count > 0}
    if not active:
        return {group: 1.0 for group in counts}
    total = float(sum(active.values()))
    group_count = float(len(active))
    weights = {group: 1.0 for group in counts}
    for group, count in active.items():
        weights[group] = total / (group_count * float(count))
    return weights


def custom_group_weights(config: RelationWeightConfig) -> Dict[str, float]:
    return {
        "rec": float(config.rec),
        "mlkc": float(config.mlkc),
        "exfr": float(config.exfr),
        "exkc": float(config.exkc),
        "other": float(config.other),
    }


def build_relation_weight_tensor(
    id2relation: Dict[int, str],
    triples: Sequence[Triple],
    config: RelationWeightConfig,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, Dict[str, object]]:
    mode = str(config.mode).lower()
    if mode not in {"none", "balanced", "custom"}:
        raise ValueError("--relation-loss-weights must be one of: none, balanced, custom")

    counts = relation_group_counts(triples, id2relation)
    if mode == "balanced":
        group_weights = balanced_group_weights(counts)
    elif mode == "custom":
        group_weights = custom_group_weights(config)
    else:
        group_weights = {group: 1.0 for group in counts}

    max_relation_id = max(id2relation.keys(), default=-1)
    values = []
    for relation_id in range(max_relation_id + 1):
        group = canonical_relation_group(id2relation.get(relation_id, ""))
        values.append(float(group_weights.get(group, 1.0)))

    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    metadata: Dict[str, object] = {
        "mode": mode,
        "relation_group_counts": counts,
        "relation_group_weights": group_weights,
        "relation_weights": {id2relation[idx]: float(tensor[idx].detach().cpu().item()) for idx in id2relation},
    }
    return tensor, metadata


def weighted_bce_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    relation_ids: torch.Tensor,
    relation_weight_tensor: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy(predictions.view(-1), labels.view(-1), reduction="none")
    weights = relation_weight_tensor[relation_ids.view(-1)].to(device=losses.device, dtype=losses.dtype)
    denominator = torch.clamp(weights.sum(), min=1e-8)
    return torch.sum(losses * weights) / denominator


def weighted_bce_with_logits_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    relation_ids: torch.Tensor,
    relation_weight_tensor: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), reduction="none")
    weights = relation_weight_tensor[relation_ids.view(-1)].to(device=losses.device, dtype=losses.dtype)
    denominator = torch.clamp(weights.sum(), min=1e-8)
    return torch.sum(losses * weights) / denominator


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, pair_weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = -F.logsigmoid(pos_logits.view(-1) - neg_logits.view(-1))
    if pair_weights is None:
        return losses.mean()
    weights = pair_weights.view(-1).to(device=losses.device, dtype=losses.dtype).clamp(min=0.0)
    if bool((weights.sum() <= 0).detach().cpu().item()):
        return losses.mean()
    return torch.sum(losses * weights) / torch.clamp(weights.sum(), min=1e-8)


def correlation_penalty(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = values.view(-1).float()
    reference = reference.view(-1).float().to(device=values.device)
    if values.numel() < 2 or reference.numel() < 2:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    values = values - values.mean()
    reference = reference - reference.mean()
    denom = torch.sqrt(torch.sum(values * values) * torch.sum(reference * reference)).clamp(min=1e-8)
    corr = torch.sum(values * reference) / denom
    return corr * corr
