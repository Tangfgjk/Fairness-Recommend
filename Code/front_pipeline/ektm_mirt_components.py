from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]", str(text).lower())


def build_vocab_and_tokens(texts: list[str], min_count: int, max_len: int) -> tuple[dict[str, int], np.ndarray]:
    counts: dict[str, int] = {}
    tokenized = []
    for text in texts:
        tokens = tokenize_text(text)
        tokenized.append(tokens)
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_count:
            vocab[token] = len(vocab)
    matrix = np.zeros((len(texts), max_len), dtype=np.int64)
    for row_idx, tokens in enumerate(tokenized):
        ids = [vocab.get(token, 1) for token in tokens[:max_len]]
        matrix[row_idx, : len(ids)] = ids
    return vocab, matrix


class LearnerSequenceDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, q_matrix: np.ndarray, learner_count: int, max_seq_len: int) -> None:
        self.samples: list[dict[str, np.ndarray | int]] = []
        concept_count = int(q_matrix.shape[1])
        for uid in range(learner_count):
            group = interactions[interactions["user_id"] == uid]
            if group.empty:
                continue
            if max_seq_len > 0 and len(group) > max_seq_len:
                group = group.tail(max_seq_len)
            items = group["item_id"].to_numpy(dtype=np.int64)
            scores = group["score"].to_numpy(dtype=np.float32)
            concepts = np.zeros((len(items), concept_count), dtype=np.float32)
            for row_idx, item_id in enumerate(items):
                concepts[row_idx] = q_matrix[int(item_id)].astype(np.float32)
            self.samples.append({"user_id": uid, "items": items, "scores": scores, "concepts": concepts})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        return self.samples[index]


def collate_sequences(batch: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    max_len = max(len(item["items"]) for item in batch)  # type: ignore[arg-type]
    concept_count = batch[0]["concepts"].shape[1]  # type: ignore[index,union-attr]
    users = torch.zeros(len(batch), dtype=torch.long)
    items = torch.zeros((len(batch), max_len), dtype=torch.long)
    scores = torch.zeros((len(batch), max_len), dtype=torch.float32)
    concepts = torch.zeros((len(batch), max_len, concept_count), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for row_idx, sample in enumerate(batch):
        seq_items = torch.as_tensor(sample["items"], dtype=torch.long)
        seq_scores = torch.as_tensor(sample["scores"], dtype=torch.float32)
        seq_concepts = torch.as_tensor(sample["concepts"], dtype=torch.float32)
        length = len(seq_items)
        users[row_idx] = int(sample["user_id"])
        items[row_idx, :length] = seq_items
        scores[row_idx, :length] = seq_scores
        concepts[row_idx, :length] = seq_concepts
        mask[row_idx, :length] = True
    return {"users": users, "items": items, "scores": scores, "concepts": concepts, "mask": mask}


class EKTMmirt(nn.Module):
    def __init__(
        self,
        token_matrix: np.ndarray,
        concept_token_matrix: np.ndarray,
        q_matrix: np.ndarray,
        a_param: np.ndarray,
        b_param: np.ndarray,
        vocab_size: int,
        text_emb_size: int,
        text_hidden_size: int,
        knowledge_emb_size: int,
        hidden_size: int,
        mirt_proj_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.exercise_count, self.text_len = token_matrix.shape
        self.concept_text_count, self.concept_text_len = concept_token_matrix.shape
        self.concept_count = int(q_matrix.shape[1])
        if self.concept_text_count != self.concept_count:
            raise ValueError(
                f"concept_token_matrix rows={self.concept_text_count} must match concept_count={self.concept_count}"
            )
        self.text_hidden_size = int(text_hidden_size)
        self.knowledge_emb_size = int(knowledge_emb_size)
        self.hidden_size = int(hidden_size)
        self.register_buffer("token_matrix", torch.as_tensor(token_matrix, dtype=torch.long))
        self.register_buffer("concept_token_matrix", torch.as_tensor(concept_token_matrix, dtype=torch.long))
        q = torch.as_tensor(q_matrix, dtype=torch.float32)
        q_sum = q.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("q_matrix", q)
        self.register_buffer("q_norm", q / q_sum)
        self.register_buffer("a_param", torch.as_tensor(a_param, dtype=torch.float32))
        self.register_buffer("b_param", torch.as_tensor(b_param, dtype=torch.float32))

        self.word_embedding = nn.Embedding(vocab_size, text_emb_size, padding_idx=0)
        self.text_encoder = nn.GRU(text_emb_size, text_hidden_size // 2, batch_first=True, bidirectional=True)
        self.knowledge_embedding = nn.Embedding(self.concept_count, knowledge_emb_size)
        self.mirt_projector = nn.Sequential(
            nn.Linear(self.concept_count + 1, mirt_proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        step_dim = text_hidden_size + knowledge_emb_size + mirt_proj_size
        self.step_norm = nn.LayerNorm(step_dim)
        self.gru_cell = nn.GRUCell(step_dim + 1, hidden_size)
        self.response_head = nn.Sequential(
            nn.Linear(hidden_size + step_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.mastery_head = nn.Linear(hidden_size, self.concept_count)

    def encode_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = token_ids.shape[:-1]
        flat = token_ids.reshape(-1, token_ids.shape[-1])
        emb = self.word_embedding(flat)
        _, hidden = self.text_encoder(emb)
        text = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return text.reshape(*shape, -1)

    def topic_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_text_tokens(self.token_matrix[item_ids])

    def exercise_representation(self, item_ids: torch.Tensor) -> torch.Tensor:
        text = self.topic_embeddings(item_ids)
        q = self.q_norm[item_ids]
        knowledge = q @ self.knowledge_embedding.weight
        mirt = self.mirt_projector(torch.cat([self.a_param[item_ids], self.b_param[item_ids]], dim=-1))
        return self.step_norm(torch.cat([text, knowledge, mirt], dim=-1))

    def forward(self, items: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = items.shape
        reps = self.exercise_representation(items)
        hidden = torch.zeros(batch_size, self.hidden_size, device=items.device)
        logits = []
        mastery = []
        for step in range(seq_len):
            rep = reps[:, step, :]
            logits.append(self.response_head(torch.cat([hidden, rep], dim=-1)).squeeze(-1))
            gru_input = torch.cat([rep, scores[:, step : step + 1]], dim=-1)
            next_hidden = self.gru_cell(gru_input, hidden)
            active = mask[:, step].float().unsqueeze(-1)
            hidden = next_hidden * active + hidden * (1.0 - active)
            mastery.append(torch.sigmoid(self.mastery_head(hidden)))
        return torch.stack(logits, dim=1), torch.stack(mastery, dim=1)

    @torch.no_grad()
    def export_exercise_topics(self, batch_size: int = 512) -> np.ndarray:
        device = next(self.parameters()).device
        outputs = []
        for start in range(0, self.exercise_count, batch_size):
            ids = torch.arange(start, min(self.exercise_count, start + batch_size), device=device)
            outputs.append(self.topic_embeddings(ids).detach().cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def batch_loss(model: EKTMmirt, batch: dict[str, torch.Tensor], device: torch.device, mastery_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    items = batch["items"].to(device)
    scores = batch["scores"].to(device)
    concepts = batch["concepts"].to(device)
    mask = batch["mask"].to(device)
    logits, mastery = model(items, scores, mask)
    response_loss = F.binary_cross_entropy_with_logits(logits[mask], scores[mask])
    concept_mask = (concepts > 0) & mask.unsqueeze(-1)
    if concept_mask.any():
        target = scores.unsqueeze(-1).expand_as(mastery)
        mastery_loss = F.binary_cross_entropy(mastery[concept_mask], target[concept_mask])
    else:
        mastery_loss = torch.zeros((), device=device)
    loss = response_loss + float(mastery_weight) * mastery_loss
    return loss, {"response_loss": float(response_loss.detach().cpu()), "mastery_loss": float(mastery_loss.detach().cpu())}


def run_epoch(
    model: EKTMmirt,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    mastery_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "response_loss": 0.0, "mastery_loss": 0.0}
    batches = 0
    for batch in tqdm(loader, disable=not training):
        if training:
            optimizer.zero_grad()
        loss, pieces = batch_loss(model, batch, device, mastery_weight)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["response_loss"] += pieces["response_loss"]
        totals["mastery_loss"] += pieces["mastery_loss"]
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def export_mastery(model: EKTMmirt, dataset: LearnerSequenceDataset, learner_count: int, concept_count: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_sequences)
    mastery = np.full((learner_count, concept_count), 0.5, dtype=np.float32)
    model.eval()
    for batch in loader:
        users = batch["users"].numpy()
        items = batch["items"].to(device)
        scores = batch["scores"].to(device)
        mask = batch["mask"].to(device)
        _, mastery_seq = model(items, scores, mask)
        lengths = mask.sum(dim=1).clamp_min(1) - 1
        for row_idx, user_id in enumerate(users):
            mastery[int(user_id)] = mastery_seq[row_idx, lengths[row_idx]].detach().cpu().numpy().astype(np.float32)
    return np.clip(mastery, 0.0, 1.0)

