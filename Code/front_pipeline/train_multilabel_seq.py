from __future__ import annotations

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


class NextConceptDataset(Dataset):
    def __init__(self, interactions, q_matrix: np.ndarray, uids: list[int]) -> None:
        self.samples: list[SequenceSample] = []
        for uid in uids:
            frame = interactions[interactions["uid"] == uid]
            if len(frame) < 2:
                continue
            question_ids = frame["question"].to_numpy(dtype=np.int64)
            responses = frame["response"].to_numpy(dtype=np.float32)
            q = q_matrix[question_ids]
            incorrect = q[:-1] * (1.0 - responses[:-1, None])
            correct = q[:-1] * responses[:-1, None]
            self.samples.append(
                SequenceSample(
                    uid=int(uid),
                    inputs=np.concatenate([incorrect, correct], axis=1).astype(np.float32),
                    targets=q[1:].astype(np.float32),
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


class MultiLabelNextConceptLSTM(nn.Module):
    def __init__(self, concept_count: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.concept_count = int(concept_count)
        self.lstm = nn.LSTM(2 * concept_count, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, concept_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, _ = self.lstm(inputs)
        return self.output(self.dropout(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train raw-data multi-label next-concept LSTM and export stu2know_seq.json.")
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_weights(dataset: NextConceptDataset, concept_count: int, cap: float) -> np.ndarray:
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


def loss_for_batch(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    losses = functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
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
def export_predictions(model, raw, fallback: np.ndarray, device: torch.device) -> np.ndarray:
    output = np.tile(fallback.astype(np.float32), (raw.student_count, 1))
    model.eval()
    for uid in range(raw.student_count):
        frame = raw.interactions[raw.interactions["uid"] == uid]
        if frame.empty:
            continue
        item_ids = frame["question"].to_numpy(dtype=np.int64)
        responses = frame["response"].to_numpy(dtype=np.float32)
        q = raw.q_matrix[item_ids]
        step_input = np.concatenate([q * (1.0 - responses[:, None]), q * responses[:, None]], axis=1).astype(np.float32)
        logits = model(torch.from_numpy(step_input).unsqueeze(0).to(device))
        output[uid] = torch.sigmoid(logits[0, -1]).cpu().numpy()
    return np.clip(output, 0.0, 1.0)


def build_model(raw, args) -> MultiLabelNextConceptLSTM:
    return MultiLabelNextConceptLSTM(raw.concept_count, args.hidden_size, args.dropout)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root)
    output = front_dir(args.dataset, root) / "sequence"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "multilabel_next_concept_lstm.pt"
    if checkpoint_path.exists() and not args.force:
        raise FileExistsError(f"{checkpoint_path} exists. Use --force to retrain.")
    protocol = read_json(front_dir(args.dataset, root) / "protocol.json")
    inner_train = [int(value) for value in protocol["inner_train_uids"]]
    inner_valid = [int(value) for value in protocol["inner_valid_uids"]]
    outer_train = [int(value) for value in protocol["outer_train_uids"]]
    train_set = NextConceptDataset(raw.interactions, raw.q_matrix, inner_train)
    valid_set = NextConceptDataset(raw.interactions, raw.q_matrix, inner_valid)
    if len(train_set) == 0 or len(valid_set) == 0:
        raise ValueError("Both inner train and inner validation cohorts need at least one sequence of length >= 2")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = positive_weights(train_set, raw.concept_count, args.max_pos_weight)
    pos_weight = torch.from_numpy(weights).to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model(raw, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_epoch, best_valid, best_state = 1, float("inf"), None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, pos_weight, device)
        valid_loss = run_epoch(model, valid_loader, None, pos_weight, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid:
            best_epoch, best_valid = epoch, valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} best_epoch={best_epoch}")

    # Final training uses every outer-train learner for the selected number of epochs.
    set_seed(args.seed)
    final_set = NextConceptDataset(raw.interactions, raw.q_matrix, outer_train)
    final_weights = torch.from_numpy(positive_weights(final_set, raw.concept_count, args.max_pos_weight)).to(device)
    final_loader = DataLoader(final_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    final_model = build_model(raw, args).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate)
    for _ in range(best_epoch):
        run_epoch(final_model, final_loader, final_optimizer, final_weights, device)
    target_prior = np.zeros(raw.concept_count, dtype=np.float64)
    total_targets = 0
    for sample in final_set.samples:
        target_prior += sample.targets.sum(axis=0)
        total_targets += sample.targets.shape[0]
    target_prior = target_prior / max(1, total_targets)
    sequence = export_predictions(final_model, raw, target_prior, device)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid,
            "positive_weights": weights,
            "target_prior": target_prior,
        },
        checkpoint_path,
    )
    write_matrix_json(output / "stu2know_seq.json", sequence)
    write_matrix_json(front_dir(args.dataset, root) / "stu2know_seq.json", sequence)
    write_json(
        output / "sequence_manifest.json",
        {
            "dataset": raw.name,
            "model": "MultiLabelNextConceptLSTM",
            "input": "two channels: incorrect-Q and correct-Q for each observed exercise",
            "target": "full K-dimensional Q vector of the next exercise; all linked concepts are label 1",
            "loss": "weighted BCEWithLogits over all K concepts",
            "positive_weight": weights,
            "selected_epoch": best_epoch,
            "selected_valid_loss": best_valid,
            "final_training_users": len(outer_train),
            "test_users_frozen_inference_only": len(raw.test_uids),
            "output": "stu2know_seq.json",
            "history": history,
        },
    )
    print(f"saved raw multi-label sequence outputs: {output}")


if __name__ == "__main__":
    main()
