from __future__ import annotations

"""Train a text- and MIRT-enhanced multi-label next-concept GRU.

This is an experimental alternative to ``train_multilabel_seq.py``.  It keeps
the same target and outer student-split protocol, while using the frozen
exercise ``topic_v`` exported by ``train_ektm_mirt.py`` and the normalized
Q-constrained MIRT item features.  The default output is intentionally kept
separate from ``stu2know_seq.json`` so existing ER graphs are not overwritten.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import data_fin_root, front_dir, load_raw_dataset, read_json, write_json, write_matrix_json


@dataclass
class SequenceSample:
    uid: int
    inputs: np.ndarray
    targets: np.ndarray


def interaction_features(
    question_ids: np.ndarray,
    responses: np.ndarray,
    q_matrix: np.ndarray,
    topic_vectors: np.ndarray,
    difficulty: np.ndarray,
    discrimination: np.ndarray,
) -> np.ndarray:
    """Construct one event vector per answered exercise.

    The two Q channels preserve the previous sequence model's response-aware
    representation.  ``topic_vectors``, difficulty, and discrimination are
    frozen front features rather than trainable copies in this GRU.
    """
    q = q_matrix[question_ids]
    incorrect = q * (1.0 - responses[:, None])
    correct = q * responses[:, None]
    text = topic_vectors[question_ids]
    item_features = np.concatenate(
        [difficulty[question_ids, None], discrimination[question_ids, None]], axis=1
    )
    return np.concatenate([incorrect, correct, text, item_features], axis=1).astype(np.float32)


class TextMIRTNextConceptDataset(Dataset):
    def __init__(
        self,
        interactions,
        q_matrix: np.ndarray,
        topic_vectors: np.ndarray,
        difficulty: np.ndarray,
        discrimination: np.ndarray,
        uids: list[int],
    ) -> None:
        self.samples: list[SequenceSample] = []
        for uid in uids:
            frame = interactions[interactions["uid"] == uid]
            if len(frame) < 2:
                continue
            question_ids = frame["question"].to_numpy(dtype=np.int64)
            responses = frame["response"].to_numpy(dtype=np.float32)
            features = interaction_features(
                question_ids,
                responses,
                q_matrix,
                topic_vectors,
                difficulty,
                discrimination,
            )
            self.samples.append(
                SequenceSample(
                    uid=int(uid),
                    inputs=features[:-1],
                    targets=q_matrix[question_ids[1:]].astype(np.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SequenceSample:
        return self.samples[index]


def collate(samples: list[SequenceSample]) -> dict[str, torch.Tensor]:
    max_len = max(sample.inputs.shape[0] for sample in samples)
    input_dim = samples[0].inputs.shape[1]
    concept_count = samples[0].targets.shape[1]
    inputs = torch.zeros((len(samples), max_len, input_dim), dtype=torch.float32)
    targets = torch.zeros((len(samples), max_len, concept_count), dtype=torch.float32)
    mask = torch.zeros((len(samples), max_len), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = sample.inputs.shape[0]
        inputs[row, :length] = torch.from_numpy(sample.inputs)
        targets[row, :length] = torch.from_numpy(sample.targets)
        mask[row, :length] = True
    return {"inputs": inputs, "targets": targets, "mask": mask}


class TextMIRTMultiLabelNextConceptGRU(nn.Module):
    """Project concatenated event features, then predict the next KC labels."""

    def __init__(self, input_dim: int, concept_count: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, concept_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.input_projection(self.input_norm(inputs)))
        values, _ = self.gru(values)
        return self.output(self.dropout(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-EKTM-topic + Q-MIRT multi-label next-concept GRU."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--replace-stu2know-seq",
        action="store_true",
        help="Also overwrite front_features/stu2know_seq.json after a successful export.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_weights(dataset: TextMIRTNextConceptDataset, concept_count: int, cap: float) -> np.ndarray:
    positives = np.zeros(concept_count, dtype=np.float64)
    total = 0
    for sample in dataset.samples:
        positives += sample.targets.sum(axis=0)
        total += sample.targets.shape[0]
    negatives = total - positives
    weights = np.ones(concept_count, dtype=np.float32)
    valid = positives > 0
    weights[valid] = np.clip(negatives[valid] / positives[valid], 1.0, cap)
    return weights


def loss_for_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    losses = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    return losses[mask].mean()


def run_epoch(model, loader, optimizer, pos_weight, device) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        if training:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_for_batch(logits, targets, mask, pos_weight)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def export_predictions(
    model,
    raw,
    topic_vectors: np.ndarray,
    difficulty: np.ndarray,
    discrimination: np.ndarray,
    fallback: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    output = np.tile(fallback.astype(np.float32), (raw.student_count, 1))
    model.eval()
    for uid in range(raw.student_count):
        frame = raw.interactions[raw.interactions["uid"] == uid]
        if frame.empty:
            continue
        question_ids = frame["question"].to_numpy(dtype=np.int64)
        responses = frame["response"].to_numpy(dtype=np.float32)
        features = interaction_features(
            question_ids,
            responses,
            raw.q_matrix,
            topic_vectors,
            difficulty,
            discrimination,
        )
        logits = model(torch.from_numpy(features).unsqueeze(0).to(device))
        output[uid] = torch.sigmoid(logits[0, -1]).cpu().numpy()
    return np.clip(output, 0.0, 1.0)


def load_frozen_features(raw, front: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topic_path = front / "ektm_mirt" / "exports" / "exercise_topic_v.npy"
    mirt_dir = front / "mirt"
    difficulty_path = mirt_dir / "exercise_difficulty_norm.npy"
    discrimination_path = mirt_dir / "exercise_discrimination_norm.npy"
    missing = [
        str(path)
        for path in [topic_path, difficulty_path, discrimination_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing frozen EKTM/MIRT inputs. Run train_q_mirt.py and train_ektm_mirt.py first: "
            f"{missing}"
        )
    topics = np.load(topic_path).astype(np.float32)
    difficulty = np.load(difficulty_path).astype(np.float32).reshape(-1)
    discrimination = np.load(discrimination_path).astype(np.float32).reshape(-1)
    if topics.ndim != 2 or topics.shape[0] != raw.exercise_count:
        raise ValueError(f"Unexpected exercise_topic_v shape: {topics.shape}")
    if difficulty.shape != (raw.exercise_count,) or discrimination.shape != (raw.exercise_count,):
        raise ValueError(
            "MIRT item feature shape mismatch: "
            f"difficulty={difficulty.shape}, discrimination={discrimination.shape}"
        )
    if not (np.isfinite(topics).all() and np.isfinite(difficulty).all() and np.isfinite(discrimination).all()):
        raise ValueError("Frozen EKTM/MIRT features contain non-finite values")
    return topics, difficulty, discrimination


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root)
    front = front_dir(args.dataset, root)
    output = front / "sequence_text_mirt_gru"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "text_mirt_multilabel_gru.pt"
    if checkpoint_path.exists() and not args.force:
        raise FileExistsError(f"{checkpoint_path} exists. Use --force to retrain.")
    protocol_path = front / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("Run prepare_front_protocol.py before sequence training")
    protocol = read_json(protocol_path)
    topics, difficulty, discrimination = load_frozen_features(raw, front)
    inner_train = [int(value) for value in protocol["inner_train_uids"]]
    inner_valid = [int(value) for value in protocol["inner_valid_uids"]]
    outer_train = [int(value) for value in protocol["outer_train_uids"]]
    train_set = TextMIRTNextConceptDataset(
        raw.interactions,
        raw.q_matrix,
        topics,
        difficulty,
        discrimination,
        inner_train,
    )
    valid_set = TextMIRTNextConceptDataset(
        raw.interactions,
        raw.q_matrix,
        topics,
        difficulty,
        discrimination,
        inner_valid,
    )
    if len(train_set) == 0 or len(valid_set) == 0:
        raise ValueError("Both inner train and validation cohorts need a sequence of length >= 2")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_dim = int(train_set.samples[0].inputs.shape[1])
    weights = positive_weights(train_set, raw.concept_count, args.max_pos_weight)
    pos_weight = torch.from_numpy(weights).to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = TextMIRTMultiLabelNextConceptGRU(input_dim, raw.concept_count, args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_epoch, best_valid = 1, float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, pos_weight, device)
        valid_loss = run_epoch(model, valid_loader, None, pos_weight, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid:
            best_epoch, best_valid = epoch, valid_loss
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} best_epoch={best_epoch}")

    # Select the epoch on inner validation, then retrain on every outer-train learner.
    set_seed(args.seed)
    final_set = TextMIRTNextConceptDataset(
        raw.interactions,
        raw.q_matrix,
        topics,
        difficulty,
        discrimination,
        outer_train,
    )
    final_weights = torch.from_numpy(
        positive_weights(final_set, raw.concept_count, args.max_pos_weight)
    ).to(device)
    final_loader = DataLoader(final_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    final_model = TextMIRTMultiLabelNextConceptGRU(
        input_dim,
        raw.concept_count,
        args.hidden_size,
        args.dropout,
    ).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate)
    for _ in range(best_epoch):
        run_epoch(final_model, final_loader, final_optimizer, final_weights, device)

    target_prior = np.zeros(raw.concept_count, dtype=np.float64)
    total_targets = 0
    for sample in final_set.samples:
        target_prior += sample.targets.sum(axis=0)
        total_targets += sample.targets.shape[0]
    target_prior = target_prior / max(1, total_targets)
    sequence = export_predictions(
        final_model,
        raw,
        topics,
        difficulty,
        discrimination,
        target_prior,
        device,
    )
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "config": vars(args),
            "input_dim": input_dim,
            "topic_dim": int(topics.shape[1]),
            "selected_epoch": best_epoch,
            "selected_valid_loss": best_valid,
            "positive_weights": weights,
            "target_prior": target_prior,
        },
        checkpoint_path,
    )
    output_json = output / "stu2know_seq_text_mirt_gru.json"
    write_matrix_json(output_json, sequence)
    if args.replace_stu2know_seq:
        write_matrix_json(front / "stu2know_seq.json", sequence)
    write_json(
        output / "sequence_text_mirt_gru_manifest.json",
        {
            "dataset": raw.name,
            "model": "TextMIRTMultiLabelNextConceptGRU",
            "input": {
                "response_aware_q": "incorrect-Q and correct-Q channels",
                "text": "frozen EKTM_mirt exercise_topic_v",
                "mirt": "outer-train-fitted difficulty_norm and discrimination_norm",
            },
            "target": "full K-dimensional Q vector of the next exercise; all linked concepts are label 1",
            "loss": "weighted BCEWithLogits over all K concepts",
            "selected_epoch": best_epoch,
            "selected_valid_loss": best_valid,
            "final_training_users": len(outer_train),
            "test_users_frozen_inference_only": len(raw.test_uids),
            "output": str(output_json),
            "replaces_default_stu2know_seq": bool(args.replace_stu2know_seq),
            "history": history,
        },
    )
    print(f"saved text-MIRT multi-label GRU sequence outputs: {output}")


if __name__ == "__main__":
    main()
