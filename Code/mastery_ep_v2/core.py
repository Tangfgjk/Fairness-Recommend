"""Full-history state extraction and a response-supervised concept readout.

The frozen EKTM trunk is the paper checkpoint. The new readout predicts a
response *before* that response is applied to the GRU; it is therefore trained
without the old post-answer label leakage. No raw or paper graph file is edited.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "ep_kt_evaluation"))
from evaluate_ep_kt import exercise_representations, load_frozen_kt  # noqa: E402

sys.path.insert(0, str(PROJECT / "front_pipeline"))
from common import load_raw_dataset  # noqa: E402


DATASETS = ("Eedi", "algebra2005", "XES3G5M-sub-small")


def raw_name_for(dataset: str) -> str:
    return "raw_compact" if dataset == "XES3G5M-sub-small" else "raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(dataset: str, data_root: Path, device: torch.device):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    raw = load_raw_dataset(dataset, data_root, raw_name_for(dataset))
    model, config, checkpoint_path = load_frozen_kt(dataset, data_root, raw, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    representations = exercise_representations(model, raw.exercise_count, device)
    return raw, model, config, checkpoint_path, representations


def gru_from_cell(cell: nn.GRUCell, device: torch.device) -> nn.GRU:
    """Use the equivalent cuDNN GRU for efficient 200-step chunks."""
    gru = nn.GRU(cell.input_size, cell.hidden_size, batch_first=True).to(device)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(cell.weight_ih)
        gru.weight_hh_l0.copy_(cell.weight_hh)
        gru.bias_ih_l0.copy_(cell.bias_ih)
        gru.bias_hh_l0.copy_(cell.bias_hh)
    gru.eval()
    return gru


@torch.inference_mode()
def verify_gru_equivalence(cell: nn.GRUCell, gru: nn.GRU, device: torch.device) -> float:
    generator = torch.Generator(device=device).manual_seed(731)
    inputs = torch.randn(1, 13, cell.input_size, generator=generator, device=device)
    state = torch.zeros(1, cell.hidden_size, device=device)
    step_outputs = []
    for step in range(inputs.shape[1]):
        state = cell(inputs[:, step, :], state)
        step_outputs.append(state)
    batched, _ = gru(inputs)
    difference = float((batched - torch.stack(step_outputs, dim=1)).abs().max().item())
    if difference > 2e-5:
        raise AssertionError(f"GRU/GRUCell mismatch: {difference}")
    return difference


@dataclass
class HistoryFeatures:
    uids: np.ndarray
    items: np.ndarray
    labels: np.ndarray
    before: np.ndarray
    tags: np.ndarray
    tag_mask: np.ndarray
    final_hidden: np.ndarray
    processed_attempts: int
    source_attempts: int
    gru_max_difference: float


def concept_tags(frame: pd.DataFrame, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use per-interaction tags where present; Eedi falls back to Q."""
    tags: list[np.ndarray] = []
    if frame["concepts"].isna().all():
        for item in frame["question"].to_numpy(dtype=np.int64):
            tags.append(np.flatnonzero(q[item]).astype(np.int64))
    else:
        if frame["concepts"].isna().any():
            raise ValueError("Partially missing interaction concept tags")
        for item, value in zip(frame["question"], frame["concepts"]):
            parsed = np.unique(np.asarray([int(part) for part in str(value).split("_")], dtype=np.int64))
            if len(parsed) == 0 or parsed.min() < 0 or parsed.max() >= q.shape[1]:
                raise ValueError(f"Invalid interaction concept tags for exercise {item}: {value}")
            if not np.all(q[int(item), parsed] > 0):
                raise ValueError(f"Interaction concept tags outside Q for exercise {item}: {value}")
            tags.append(parsed)
    if any(len(tag) == 0 for tag in tags):
        raise ValueError("At least one interaction maps to no knowledge concept")
    width = max(map(len, tags))
    indices = np.zeros((len(tags), width), dtype=np.int64)
    mask = np.zeros((len(tags), width), dtype=np.float32)
    for index, tag in enumerate(tags):
        indices[index, : len(tag)] = tag
        mask[index, : len(tag)] = 1.0
    return indices, mask


@torch.inference_mode()
def extract_history(raw, model, representations: torch.Tensor, device: torch.device,
                    chunk_len: int = 200, selected_uids: set[int] | None = None) -> HistoryFeatures:
    if chunk_len < 1:
        raise ValueError("chunk_len must be positive")
    frame = raw.interactions
    if selected_uids is not None:
        frame = frame[frame["uid"].isin(selected_uids)].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No interactions selected")
    gru = gru_from_cell(model.gru_cell, device)
    # On RTX 5060, cuDNN GRU used a different accumulation path than GRUCell
    # (observed max difference 5.5e-4). Disable cuDNN only for this replay so
    # the exported state matches the exact GRUCell used by Ep simulation.
    with torch.backends.cudnn.flags(enabled=False):
        max_diff = verify_gru_equivalence(model.gru_cell, gru, device)
    before = np.empty((len(frame), model.hidden_size), dtype=np.float32)
    final_hidden = np.full((raw.student_count, model.hidden_size), np.nan, dtype=np.float32)
    for uid, group in frame.groupby("uid", sort=True):
        row_ids = group.index.to_numpy(dtype=np.int64)
        items = torch.as_tensor(group["question"].to_numpy(dtype=np.int64), device=device)
        scores = torch.as_tensor(group["response"].to_numpy(dtype=np.float32), device=device)
        hidden = torch.zeros(1, 1, model.hidden_size, device=device)
        with torch.backends.cudnn.flags(enabled=False):
            for start in range(0, len(group), chunk_len):
                stop = min(start + chunk_len, len(group))
                rep = representations[items[start:stop]].unsqueeze(0)
                block = torch.cat((rep, scores[start:stop].view(1, -1, 1)), dim=-1)
                outputs, next_hidden = gru(block, hidden)
                prior = torch.cat((hidden.transpose(0, 1), outputs[:, :-1, :]), dim=1)
                before[row_ids[start:stop]] = prior.squeeze(0).cpu().numpy()
                hidden = next_hidden
        final_hidden[int(uid)] = hidden.squeeze(0).squeeze(0).cpu().numpy()
    tags, mask = concept_tags(frame, raw.q_matrix)
    if not np.isfinite(before).all():
        raise ValueError("Non-finite hidden state in extracted history")
    return HistoryFeatures(
        uids=frame["uid"].to_numpy(dtype=np.int64),
        items=frame["question"].to_numpy(dtype=np.int64),
        labels=frame["response"].to_numpy(dtype=np.float32),
        before=before, tags=tags, tag_mask=mask, final_hidden=final_hidden,
        processed_attempts=len(frame), source_attempts=len(raw.interactions),
        gru_max_difference=max_diff,
    )


class ConceptReadout(nn.Module):
    """Mastery logit per concept plus a regularized exercise difficulty offset."""

    def __init__(self, hidden_size: int, concept_count: int, initial_item_bias: np.ndarray):
        super().__init__()
        bias = torch.as_tensor(initial_item_bias.reshape(-1), dtype=torch.float32)
        self.mastery_head = nn.Linear(hidden_size, concept_count)
        self.item_bias = nn.Embedding(len(bias), 1)
        nn.init.zeros_(self.mastery_head.weight)
        nn.init.zeros_(self.mastery_head.bias)
        with torch.no_grad():
            self.item_bias.weight[:, 0].copy_(bias)
        self.register_buffer("initial_item_bias", bias)

    def forward(self, states: torch.Tensor, items: torch.Tensor,
                tags: torch.Tensor, tag_mask: torch.Tensor) -> torch.Tensor:
        mastery_logits = self.mastery_head(states)
        selected = mastery_logits.gather(1, tags)
        concept_logit = (selected * tag_mask).sum(1) / tag_mask.sum(1).clamp_min(1)
        return concept_logit + self.item_bias(items).squeeze(1)

    def mastery(self, states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mastery_head(states))

    def bias_penalty(self) -> torch.Tensor:
        return (self.item_bias.weight.squeeze(1) - self.initial_item_bias).square().mean()


def initial_bias_from_model(base_model) -> np.ndarray:
    """Use the frozen checkpoint's MIRT buffer, not mutable disk-side exports."""
    return -base_model.b_param.detach().cpu().numpy().astype(np.float32).reshape(-1)
