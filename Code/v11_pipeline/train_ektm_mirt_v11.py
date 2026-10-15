from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    copy_file,
    ensure_large_csv_field_limit,
    entity_count,
    locate_sequence_interactions,
    locate_source_feature_dir,
    locate_source_graph_dir,
    locate_test_sequences,
    output_data_root,
    read_entity_dict,
    read_json,
    read_q_matrix,
    source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
    write_matrix_json,
)
from prepare_mirt_inputs_v11 import eedi_valid_question_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a V11 EKTM_mirt-style model on ER-aligned learners and export "
            "stu2know_mastery.json plus EKTM topic text embeddings."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-emb-size", type=int, default=128)
    parser.add_argument("--text-hidden-size", type=int, default=100)
    parser.add_argument("--knowledge-emb-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--mirt-proj-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--mastery-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Export from existing best.pt without retraining.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_v11_core_files(source_graph_dir: Path, v11_dir: Path) -> None:
    for name in ["entities.dict", "Q.txt", "relations.dict"]:
        src = source_graph_dir / name
        dst = v11_dir / name
        if src.exists() and not dst.exists():
            copy_file(src, dst)


def find_single(pattern: str, directory: Path, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Cannot locate {label}: {directory / pattern}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {label} files found under {directory}: {[str(x) for x in matches]}")
    return matches[0]


def load_mirt_params(v11_dir: Path, exercise_count: int, concept_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    output_dir = v11_dir / "mirt" / "outputs"
    manifest = read_json(output_dir / "mirt_export_manifest.json")
    latent_dim = int(manifest["latent_dim"])
    a_path = output_dir / f"a_param_{latent_dim}.csv"
    b_path = output_dir / f"b_param_{latent_dim}.csv"
    theta_path = output_dir / f"theta_param_{latent_dim}.csv"
    if not a_path.exists():
        a_path = find_single("a_param_*.csv", output_dir, "MIRT a parameter")
    if not b_path.exists():
        b_path = find_single("b_param_*.csv", output_dir, "MIRT b parameter")
    if not theta_path.exists():
        theta_path = find_single("theta_param_*.csv", output_dir, "MIRT theta parameter")

    a = np.loadtxt(a_path, delimiter=",", dtype=np.float32)
    b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
    theta = np.loadtxt(theta_path, delimiter=",", dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
    if a.shape[0] != exercise_count or b.shape[0] != exercise_count:
        raise ValueError(
            f"MIRT item parameter rows must match exercises={exercise_count}; "
            f"a_shape={a.shape}, b_shape={b.shape}"
        )
    if a.shape[1] != concept_count:
        raise ValueError(
            f"V11 no-Q MIRT latent_dim should equal concept_count for EKTM-style training; "
            f"a_dim={a.shape[1]}, concept_count={concept_count}"
        )
    return a, b[:, :1], theta, {
        "a_file": a_path,
        "b_file": b_path,
        "theta_file": theta_path,
        "a_shape": list(a.shape),
        "b_shape": list(b[:, :1].shape),
        "theta_shape": list(theta.shape),
        "latent_dim": latent_dim,
    }


def tokenize_text(text: str) -> list[str]:
    # Keep English/math tokens and Chinese characters. This avoids external
    # tokenizers while still letting the Bi-GRU learn dataset-specific text use.
    return re.findall(r"[A-Za-z]+|[0-9]+|[\u4e00-\u9fff]|[^\s]", str(text).lower())


def load_exercise_texts(dataset: str, source_dir: Path, source_graph_dir: Path, exercise_count: int) -> list[str]:
    feature_dir = locate_source_feature_dir(dataset, source_dir, source_graph_dir)
    if feature_dir is None:
        raise FileNotFoundError(f"Cannot locate semantic feature directory for {dataset}")
    path = feature_dir / "entity_features" / "exercise_semantics.json"
    payload = read_json(path)
    exercises = payload.get("exercises", {})
    texts: list[str] = []
    for idx in range(exercise_count):
        item = exercises.get(f"ex{idx}", {})
        text = item.get("text_for_embedding") or item.get("question_text") or item.get("problem_name") or f"exercise {idx}"
        texts.append(str(text))
    return texts


def load_concept_texts(dataset: str, source_dir: Path, source_graph_dir: Path, concept_count: int) -> tuple[list[str], dict[str, Any]]:
    feature_dir = locate_source_feature_dir(dataset, source_dir, source_graph_dir)
    if feature_dir is None:
        raise FileNotFoundError(f"Cannot locate semantic feature directory for {dataset}")
    path = feature_dir / "entity_features" / "concept_semantics.json"
    payload = read_json(path)
    concepts = payload.get("concepts", {})
    texts: list[str] = []
    source_summary: dict[str, int] = {}
    for idx in range(concept_count):
        item = concepts.get(f"kc{idx}", {})
        text = (
            item.get("text_for_embedding")
            or " ".join(str(x) for x in [item.get("name", ""), item.get("definition", "")] if str(x).strip())
            or f"knowledge concept {idx}"
        )
        texts.append(str(text))
        source = str(item.get("definition_source", "missing"))
        source_summary[source] = source_summary.get(source, 0) + 1
    return texts, {
        "source_file": path,
        "definition_source_summary": source_summary,
        "concept_count": len(texts),
    }


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


def graph_uid_map(dataset: str, v11_dir: Path, learner_count: int) -> tuple[dict[str, int], dict[str, Any]]:
    manifest_path = v11_dir / "mirt" / "inputs" / "mirt_input_manifest.json"
    manifest = read_json(manifest_path)
    uid_info = manifest.get("uid_alignment", {})
    mapping: dict[str, int] = {}
    if "sequence_uid_by_entity" in uid_info:
        seq_map = uid_info["sequence_uid_by_entity"]
        mapping = {str(seq_map[f"uid{idx}"]): idx for idx in range(learner_count)}
        key_source = "sequence_uid_by_entity"
    elif "raw_uid_by_entity" in uid_info:
        raw_map = uid_info["raw_uid_by_entity"]
        mapping = {str(raw_map[f"uid{idx}"]): idx for idx in range(learner_count)}
        key_source = "raw_uid_by_entity"
    else:
        mapping = {str(idx): idx for idx in range(learner_count)}
        key_source = "identity_uid"
    return mapping, {"manifest": manifest_path, "key_source": key_source, "mapped_users": len(mapping)}


def parse_sequence_field(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [token.strip() for token in str(value).split(",")]


def eedi_interactions_from_test_sequences(
    source_dir: Path,
    source_graph_dir: Path,
    learner_count: int,
    exercise_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_sequences = locate_test_sequences("Eedi", source_dir, source_graph_dir)
    ensure_large_csv_field_limit()
    selected: list[dict[str, str]] = []
    with test_sequences.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            valid_len = eedi_valid_question_length(row)
            if 10 <= valid_len <= 198:
                selected.append(dict(row, _valid_question_length=str(valid_len)))
    selected = selected[:learner_count]
    if len(selected) != learner_count:
        raise ValueError(f"Eedi selected test rows={len(selected)} does not match learner_count={learner_count}")

    records: list[dict[str, Any]] = []
    for user_id, row in enumerate(selected):
        questions = parse_sequence_field(row.get("questions", ""))
        responses = parse_sequence_field(row.get("responses", ""))
        masks = parse_sequence_field(row.get("selectmasks", ""))
        usable = min(len(questions), len(responses))
        for order_idx in range(usable):
            question = questions[order_idx]
            response = responses[order_idx]
            mask = masks[order_idx] if order_idx < len(masks) else "1"
            if mask != "1" or question in {"", "-1"} or response in {"", "-1"}:
                continue
            try:
                item_id = int(float(question))
                score = 1.0 if int(float(response)) > 0 else 0.0
            except ValueError:
                continue
            if 0 <= item_id < exercise_count:
                records.append({"user_id": user_id, "item_id": item_id, "score": score, "_order": order_idx})
    df = pd.DataFrame.from_records(records, columns=["user_id", "item_id", "score", "_order"])
    return df, {
        "source": test_sequences,
        "raw_rows": len(selected),
        "aligned_rows": int(len(df)),
        "aligned_users": int(df["user_id"].nunique()) if len(df) else 0,
        "uid_alignment": "Eedi pyKT test rows with 10 <= valid question length <= 198; row order maps to uid0..uidN",
        "order_policy": "preserve question order in pyKT test_sequences.csv",
    }


def load_ordered_interactions(dataset: str, source_dir: Path, v11_dir: Path, learner_count: int, exercise_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_graph_dir = locate_source_graph_dir(source_dir)
    if dataset == "Eedi":
        return eedi_interactions_from_test_sequences(source_dir, source_graph_dir, learner_count, exercise_count)

    seq_path = locate_sequence_interactions(source_dir)
    ensure_large_csv_field_limit()
    raw = pd.read_csv(seq_path, low_memory=False)
    required = {"uid", "question", "response"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{seq_path} missing columns: {sorted(missing)}")
    uid_map, uid_report = graph_uid_map(dataset, v11_dir, learner_count)
    work = raw.copy()
    work["_order"] = np.arange(len(work))
    work["uid_key"] = work["uid"].astype(str)
    work = work[work["uid_key"].isin(uid_map)].copy()
    work["user_id"] = work["uid_key"].map(uid_map).astype(int)
    work["item_id"] = pd.to_numeric(work["question"], errors="coerce")
    work["score"] = pd.to_numeric(work["response"], errors="coerce")
    work = work.dropna(subset=["item_id", "score"])
    work["item_id"] = work["item_id"].astype(int)
    work["score"] = (work["score"] > 0).astype(float)
    work = work[(work["item_id"] >= 0) & (work["item_id"] < exercise_count)]
    work = work[(work["user_id"] >= 0) & (work["user_id"] < learner_count)]
    df = work[["user_id", "item_id", "score", "_order"]].sort_values(["user_id", "_order"]).reset_index(drop=True)
    return df, {
        "source": seq_path,
        "raw_rows": int(len(raw)),
        "aligned_rows": int(len(df)),
        "aligned_users": int(df["user_id"].nunique()) if len(df) else 0,
        "uid_alignment": uid_report,
        "order_policy": "preserve source sequence row order within each aligned graph learner",
    }


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


def split_dataset(dataset: LearnerSequenceDataset, valid_ratio: float, seed: int) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    valid_size = max(1, int(len(indices) * valid_ratio)) if len(indices) > 1 else 0
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:] or indices
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, valid_idx or train_idx[:1])


class V11EKTMmirt(nn.Module):
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
        self.dropout = nn.Dropout(dropout)

    def encode_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = token_ids.shape[:-1]
        flat = token_ids.reshape(-1, token_ids.shape[-1])
        emb = self.word_embedding(flat)
        _, hidden = self.text_encoder(emb)
        # hidden: [2, N, H/2], concatenate final states from both directions.
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

    @torch.no_grad()
    def export_concept_text_embeddings(self, batch_size: int = 512) -> np.ndarray:
        device = next(self.parameters()).device
        outputs = []
        for start in range(0, self.concept_count, batch_size):
            token_ids = self.concept_token_matrix[start : min(self.concept_count, start + batch_size)].to(device)
            outputs.append(self.encode_text_tokens(token_ids).detach().cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def batch_loss(model: V11EKTMmirt, batch: dict[str, torch.Tensor], device: torch.device, mastery_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
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
    model: V11EKTMmirt,
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
def export_mastery(model: V11EKTMmirt, dataset: LearnerSequenceDataset, learner_count: int, concept_count: int, device: torch.device) -> np.ndarray:
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


def save_exports(
    args: argparse.Namespace,
    v11_dir: Path,
    model: V11EKTMmirt,
    dataset: LearnerSequenceDataset,
    learner_count: int,
    concept_count: int,
    device: torch.device,
    report: dict[str, Any],
) -> None:
    export_dir = v11_dir / "ektm_mirt" / "exports"
    text_dir = v11_dir / "semantic_kg_features" / "text_embeddings"
    export_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    topics = model.export_exercise_topics()
    concepts = model.export_concept_text_embeddings()
    mastery = export_mastery(model, dataset, learner_count, concept_count, device)

    np.save(export_dir / "exercise_topic_v.npy", topics)
    np.save(export_dir / "know_output.npy", mastery)
    np.save(text_dir / "exercise_text_embeddings.npy", topics)
    np.save(text_dir / "concept_text_embeddings.npy", concepts)
    write_matrix_json(export_dir / "stu2know_mastery.json", mastery, decimals=6)
    write_matrix_json(v11_dir / "ektm_mirt" / "stu2know_mastery.json", mastery, decimals=6)
    write_matrix_json(v11_dir / "stu2know_mastery.json", mastery, decimals=6)
    write_json(
        text_dir / "text_embedding_manifest.json",
        {
            "dataset": args.dataset,
            "model": "V11_EKTM_mirt_style_BiGRU_topic_encoder",
            "source": "topic_v_from_train_ektm_mirt_v11",
            "embedding_dim": int(topics.shape[1]),
            "concept_embedding_source": "concept_name_definition_encoded_by_the_same_trainable_BiGRU_text_encoder",
            "exercise_entity_ids": [f"ex{idx}" for idx in range(topics.shape[0])],
            "concept_entity_ids": [f"kc{idx}" for idx in range(concepts.shape[0])],
            "files": {
                "exercise_text_embeddings": "exercise_text_embeddings.npy",
                "concept_text_embeddings": "concept_text_embeddings.npy",
            },
            "notes": (
                "No BGE/SentenceTransformer is used. Exercise text is encoded by the trainable Bi-GRU topic encoder. "
                "Concept name and definition are encoded by the same Bi-GRU after training; concept text does not enter "
                "the EKTM_mirt forward pass for mastery estimation."
            ),
        },
    )
    write_json(
        export_dir / "ektm_train_export_manifest.json",
        {
            "dataset": args.dataset,
            "model": "V11_EKTM_mirt_style",
            "checkpoint": v11_dir / "ektm_mirt" / "best.pt",
            "mastery_json": v11_dir / "stu2know_mastery.json",
            "topic_output_file": export_dir / "exercise_topic_v.npy",
            "know_output_file": export_dir / "know_output.npy",
            "text_embedding_dir": text_dir,
            "topic_shape": list(topics.shape),
            "concept_embedding_shape": list(concepts.shape),
            "mastery_shape": list(mastery.shape),
            "mastery_range": {
                "min": float(np.min(mastery)),
                "max": float(np.max(mastery)),
                "mean": float(np.mean(mastery)),
                "std": float(np.std(mastery)),
            },
            "training_report": report,
        },
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    source_graph_dir = locate_source_graph_dir(source_dir)
    v11_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    v11_dir.mkdir(parents=True, exist_ok=True)
    ensure_v11_core_files(source_graph_dir, v11_dir)

    entity2id = read_entity_dict(v11_dir / "entities.dict")
    learner_count = entity_count(entity2id, "uid")
    exercise_count = entity_count(entity2id, "ex")
    concept_count = entity_count(entity2id, "kc")
    q_matrix = read_q_matrix(v11_dir / "Q.txt").astype(np.float32)
    if q_matrix.shape != (exercise_count, concept_count):
        raise ValueError(f"Q.txt shape={q_matrix.shape} does not match exercises={exercise_count}, concepts={concept_count}")
    a_param, b_param, _theta, mirt_report = load_mirt_params(v11_dir, exercise_count, concept_count)
    texts = load_exercise_texts(args.dataset, source_dir, source_graph_dir, exercise_count)
    concept_texts, concept_text_report = load_concept_texts(args.dataset, source_dir, source_graph_dir, concept_count)
    vocab, combined_token_matrix = build_vocab_and_tokens(texts + concept_texts, args.min_token_count, args.max_text_len)
    token_matrix = combined_token_matrix[:exercise_count]
    concept_token_matrix = combined_token_matrix[exercise_count:]
    interactions, interaction_report = load_ordered_interactions(args.dataset, source_dir, v11_dir, learner_count, exercise_count)
    seq_dataset = LearnerSequenceDataset(interactions, q_matrix, learner_count, args.max_seq_len)
    if len(seq_dataset) == 0:
        raise ValueError(f"No aligned learner sequences found for {args.dataset}")

    model = V11EKTMmirt(
        token_matrix=token_matrix,
        concept_token_matrix=concept_token_matrix,
        q_matrix=q_matrix,
        a_param=a_param,
        b_param=b_param,
        vocab_size=len(vocab),
        text_emb_size=args.text_emb_size,
        text_hidden_size=args.text_hidden_size,
        knowledge_emb_size=args.knowledge_emb_size,
        hidden_size=args.hidden_size,
        mirt_proj_size=args.mirt_proj_size,
        dropout=args.dropout,
    ).to(device)

    output_dir = v11_dir / "ektm_mirt"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if best_path.exists() and not args.force and not args.export_only:
        raise FileExistsError(f"{best_path} exists. Use --force to retrain or --export-only to export.")

    history: list[dict[str, Any]] = []
    best_valid = float("inf")
    best_epoch = 0
    if args.export_only:
        if not best_path.exists():
            raise FileNotFoundError(f"{best_path} does not exist. Train before --export-only.")
        payload = load_checkpoint(best_path, device)
        model.load_state_dict(payload["model_state_dict"])
        history = payload.get("history", [])
        best_valid = float(payload.get("best_valid_loss", float("inf")))
        best_epoch = int(payload.get("best_epoch", payload.get("epoch", 0)))
    else:
        train_set, valid_set = split_dataset(seq_dataset, args.valid_ratio, args.seed)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sequences)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            train_stats = run_epoch(model, train_loader, device, optimizer, args.mastery_loss_weight)
            with torch.no_grad():
                valid_stats = run_epoch(model, valid_loader, device, None, args.mastery_loss_weight)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_response_loss": train_stats["response_loss"],
                "train_mastery_loss": train_stats["mastery_loss"],
                "valid_loss": valid_stats["loss"],
                "valid_response_loss": valid_stats["response_loss"],
                "valid_mastery_loss": valid_stats["mastery_loss"],
            }
            history.append(row)
            payload = {
                "model": "V11_EKTM_mirt_style",
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
                "vocab": vocab,
                "token_matrix_shape": list(token_matrix.shape),
                "concept_token_matrix_shape": list(concept_token_matrix.shape),
                "mirt_params": mirt_report,
                "concept_text": concept_text_report,
                "interaction_report": interaction_report,
                "history": history,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid,
            }
            torch.save(payload, last_path)
            if valid_stats["loss"] < best_valid:
                best_valid = valid_stats["loss"]
                best_epoch = epoch + 1
                payload["best_epoch"] = best_epoch
                payload["best_valid_loss"] = best_valid
                torch.save(payload, best_path)
            print(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_stats['loss']:.6f} valid_loss={valid_stats['loss']:.6f} best_epoch={best_epoch}"
            )
        payload = load_checkpoint(best_path, device)
        model.load_state_dict(payload["model_state_dict"])

    report = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid,
        "history": history,
        "mirt_params": mirt_report,
        "concept_text": concept_text_report,
        "interaction_source": interaction_report,
        "vocab_size": len(vocab),
        "token_matrix_shape": list(token_matrix.shape),
        "concept_token_matrix_shape": list(concept_token_matrix.shape),
        "exercise_count": exercise_count,
        "learner_count": learner_count,
        "concept_count": concept_count,
        "export_only": bool(args.export_only),
    }
    save_exports(args, v11_dir, model, seq_dataset, learner_count, concept_count, device, report)
    write_json(output_dir / "ektm_train_manifest.json", report)
    print(f"saved V11 EKTM_mirt-style checkpoint and exports: {output_dir}")


if __name__ == "__main__":
    main()
