"""Train a response-supervised mastery readout on full EKTM histories."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import (DATASETS, PROJECT, ConceptReadout, extract_history,
                  initial_bias_from_model, load_source, sha256_file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(indices: np.ndarray, size: int, *, shuffle: bool, seed: int):
    order = np.array(indices, copy=True)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield order[start:start + size]


def make_readout(raw, base_model, device: torch.device) -> ConceptReadout:
    readout = ConceptReadout(base_model.hidden_size, raw.concept_count,
                             initial_bias_from_model(base_model))
    return readout.to(device)


def batch_tensors(features, rows: np.ndarray, device: torch.device):
    return (
        torch.as_tensor(features.before[rows], device=device),
        torch.as_tensor(features.items[rows], device=device),
        torch.as_tensor(features.tags[rows], device=device),
        torch.as_tensor(features.tag_mask[rows], device=device),
        torch.as_tensor(features.labels[rows], device=device),
    )


@torch.no_grad()
def predict_readout(readout: ConceptReadout, features, indices: np.ndarray,
                    device: torch.device, batch_size: int) -> np.ndarray:
    readout.eval()
    values = []
    for rows in batches(indices, batch_size, shuffle=False, seed=0):
        states, items, tags, mask, _ = batch_tensors(features, rows, device)
        values.append(torch.sigmoid(readout(states, items, tags, mask)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def metrics(prob: np.ndarray, labels: np.ndarray) -> dict:
    if len(prob) == 0:
        return {"attempts": 0}
    p = np.clip(prob.astype(np.float64), 1e-7, 1 - 1e-7)
    y = labels.astype(np.float64)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float(np.mean(bins == index)) *
              abs(float(np.mean(p[bins == index]) - np.mean(y[bins == index])))
              for index in range(10) if np.any(bins == index))
    return {
        "attempts": len(p),
        "observed_correct_rate": float(y.mean()),
        "mean_predicted_correct": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "accuracy_at_0_5": float(np.mean((p >= .5) == y)),
        "ece_10_bins": ece,
    }


def train_epochs(readout: ConceptReadout, features, train_idx: np.ndarray,
                 valid_idx: np.ndarray | None, device: torch.device,
                 epochs: int, batch_size: int, learning_rate: float,
                 bias_penalty: float, seed: int) -> tuple[int, list[dict]]:
    optimizer = torch.optim.AdamW(readout.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    best_epoch = 1
    best_brier = float("inf")
    for epoch in range(1, epochs + 1):
        readout.train()
        total_loss = 0.0
        for rows in batches(train_idx, batch_size, shuffle=True, seed=seed + epoch):
            states, items, tags, mask, labels = batch_tensors(features, rows, device)
            logits = readout(states, items, tags, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + bias_penalty * readout.bias_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_idx)}
        if valid_idx is not None:
            val_prob = predict_readout(readout, features, valid_idx, device, batch_size)
            row["validation"] = metrics(val_prob, features.labels[valid_idx])
            if row["validation"]["brier"] < best_brier:
                best_brier = row["validation"]["brier"]
                best_epoch = epoch
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if valid_idx is not None and epoch >= best_epoch + 5:
            break
    return best_epoch if valid_idx is not None else epochs, history


@torch.no_grad()
def predict_base_response(base_model, representations, features, indices: np.ndarray,
                          device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    for rows in batches(indices, batch_size, shuffle=False, seed=0):
        states = torch.as_tensor(features.before[rows], device=device)
        items = torch.as_tensor(features.items[rows], device=device)
        logits = base_model.response_head(torch.cat((states, representations[items]), dim=1))
        values.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def trailing_indices(features, test_uids: set[int]) -> np.ndarray:
    selected = []
    for uid in sorted(test_uids):
        rows = np.flatnonzero(features.uids == uid)
        if len(rows) < 5:
            continue
        selected.extend(rows[max(1, int(len(rows) * .8)):])
    return np.asarray(selected, dtype=np.int64)


@torch.no_grad()
def counterfactual_direction(base, readout, representations, features,
                             indices: np.ndarray, device: torch.device,
                             batch_size: int) -> dict:
    """Compare post-answer mastery after y=1 versus y=0 at the same state/item."""
    related = []
    for rows in batches(indices, batch_size, shuffle=False, seed=0):
        states, items, tags, mask, _ = batch_tensors(features, rows, device)
        reps = representations[items]
        zeros = torch.zeros(len(rows), 1, device=device)
        ones = torch.ones(len(rows), 1, device=device)
        wrong = base.gru_cell(torch.cat((reps, zeros), dim=1), states)
        right = base.gru_cell(torch.cat((reps, ones), dim=1), states)
        delta = readout.mastery(right) - readout.mastery(wrong)
        relevant = (delta.gather(1, tags) * mask).sum(1) / mask.sum(1).clamp_min(1)
        related.append(relevant.cpu().numpy())
    values = np.concatenate(related) if related else np.empty(0)
    return {"attempts": len(values),
            "mean_related_concept_delta_right_minus_wrong": float(values.mean()) if len(values) else None,
            "positive_fraction": float(np.mean(values > 0)) if len(values) else None,
            "negative_fraction": float(np.mean(values < 0)) if len(values) else None}


def train_one(dataset: str, args) -> Path:
    output = args.output_root / dataset
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Output already exists: {output}; select another root or use --force")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    data_root = args.data_fin_root
    raw, base, config, checkpoint_path, representations = load_source(dataset, data_root, device)
    protocol = json.loads((data_root / dataset / "front_features" / "protocol.json").read_text(encoding="utf-8"))
    inner_train = set(map(int, protocol["inner_train_uids"]))
    inner_valid = set(map(int, protocol["inner_valid_uids"]))
    outer_train = set(map(int, protocol["outer_train_uids"]))
    test = set(map(int, raw.test_uids))
    if inner_train & inner_valid or inner_train | inner_valid != outer_train or outer_train & test:
        raise ValueError("Student protocol has overlapping or incomplete partitions")
    selected_uids = None
    if args.smoke_users_per_group:
        count = args.smoke_users_per_group
        selected_uids = set(sorted(inner_train)[:count] + sorted(inner_valid)[:count] + sorted(test)[:count])
        inner_train &= selected_uids
        inner_valid &= selected_uids
        outer_train &= selected_uids
        test &= selected_uids
    print(f"[{dataset}] extracting full histories for {len(selected_uids) if selected_uids else raw.student_count} students", flush=True)
    features = extract_history(raw, base, representations, device, args.chunk_len, selected_uids)
    train_idx = np.flatnonzero(np.isin(features.uids, list(inner_train)))
    valid_idx = np.flatnonzero(np.isin(features.uids, list(inner_valid)))
    final_idx = np.flatnonzero(np.isin(features.uids, list(outer_train)))
    test_idx = np.flatnonzero(np.isin(features.uids, list(test)))
    if not all(len(x) for x in [train_idx, valid_idx, final_idx, test_idx]):
        raise ValueError("Empty training, validation, final training or test split")
    candidate = make_readout(raw, base, device)
    best_epoch, selection_history = train_epochs(
        candidate, features, train_idx, valid_idx, device, args.epochs,
        args.batch_size, args.learning_rate, args.bias_penalty, args.seed)
    set_seed(args.seed)
    readout = make_readout(raw, base, device)
    _, final_history = train_epochs(
        readout, features, final_idx, None, device, best_epoch,
        args.batch_size, args.learning_rate, args.bias_penalty, args.seed)
    tail_idx = trailing_indices(features, test)
    with torch.no_grad():
        mastery = np.full((raw.student_count, raw.concept_count), np.nan, dtype=np.float32)
        for start in range(0, raw.student_count, args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, raw.student_count))
            available = rows[np.isfinite(features.final_hidden[rows]).all(axis=1)]
            if len(available):
                states = torch.as_tensor(features.final_hidden[available], device=device)
                mastery[available] = readout.mastery(states).cpu().numpy()
    test_prob = predict_readout(readout, features, tail_idx, device, args.batch_size)
    base_prob = predict_base_response(base, representations, features, tail_idx,
                                      device, args.batch_size)
    with torch.no_grad():
        item_only = torch.sigmoid(readout.item_bias(
            torch.as_tensor(features.items[tail_idx], device=device)).squeeze(1)).cpu().numpy()
    direction = counterfactual_direction(base, readout, representations, features,
                                         tail_idx, device, args.batch_size)
    report = {
        "dataset": dataset, "protocol": "frozen EKTM trunk; pre-answer concept response supervision; full history with carried GRU state",
        "smoke_only": bool(args.smoke_users_per_group),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_interactions_sha256": sha256_file(raw.root / "interactions_all.csv"),
        "source_q_sha256": sha256_file(raw.root / "Q.txt"),
        "original_max_seq_len": int(config.max_seq_len),
        "chunk_len": args.chunk_len, "source_attempts": features.source_attempts,
        "processed_attempts": features.processed_attempts,
        "gru_max_difference": features.gru_max_difference,
        "student_counts": {"inner_train": len(inner_train), "inner_valid": len(inner_valid),
                           "outer_train": len(outer_train), "test": len(test)},
        "selected_epoch": best_epoch,
        "selection_history": selection_history,
        "final_training_history": final_history,
        "test_trailing_20pct": {
            "mastery_readout": metrics(test_prob, features.labels[tail_idx]),
            "item_difficulty_only": metrics(item_only, features.labels[tail_idx]),
            "frozen_response_head": metrics(base_prob, features.labels[tail_idx]),
            "counterfactual_answer_direction": direction,
            "protocol": "For each unseen test student, predict the last 20% sequentially from all preceding actual answers; score before each update",
        },
        "exported_mastery": {"mean": float(np.nanmean(mastery)),
                             "std": float(np.nanstd(mastery)),
                             "min": float(np.nanmin(mastery)),
                             "max": float(np.nanmax(mastery))},
        "interpretation": "Estimated concept propensity conditional on an average-difficulty item, not observed ground-truth mastery",
    }
    torch.save({"model_state_dict": readout.cpu().state_dict(),
                "hidden_size": base.hidden_size, "concept_count": raw.concept_count,
                "exercise_count": raw.exercise_count,
                "smoke_only": bool(args.smoke_users_per_group),
                "source_checkpoint_sha256": report["source_checkpoint_sha256"]},
               output / "readout.pt")
    np.save(output / "full_history_hidden.npy", features.final_hidden)
    np.save(output / "stu2know_mastery.npy", mastery)
    if not args.smoke_users_per_group:
        (output / "stu2know_mastery.json").write_text(
            json.dumps(np.round(mastery, 6).tolist(), ensure_ascii=False), encoding="utf-8")
    else:
        stale_full_export = output / "stu2know_mastery.json"
        if stale_full_export.exists():
            stale_full_export.unlink()
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": dataset, "output": str(output),
                      "selected_epoch": best_epoch,
                      "test_mastery_brier": report["test_trailing_20pct"]["mastery_readout"]["brier"],
                      "test_response_brier": report["test_trailing_20pct"]["frozen_response_head"]["brier"]},
                     ensure_ascii=False), flush=True)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    result.add_argument("--data-fin-root", type=Path, default=PROJECT / "Data_Fin_New")
    result.add_argument("--output-root", type=Path, default=PROJECT / "mastery_ep_v2" / "results")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=8192)
    result.add_argument("--chunk-len", type=int, default=200)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--bias-penalty", type=float, default=0.01)
    result.add_argument("--seed", type=int, default=2024)
    result.add_argument("--smoke-users-per-group", type=int, default=0,
                        help="Only for code checks. Artifacts are rejected by the full evaluator.")
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.chunk_len < 1 or args.smoke_users_per_group < 0:
        raise ValueError("epochs, batch-size, chunk-len must be positive; smoke count must be nonnegative")
    if args.learning_rate <= 0 or args.bias_penalty < 0:
        raise ValueError("learning-rate must be positive and bias-penalty nonnegative")
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        train_one(dataset, args)


if __name__ == "__main__":
    main()
