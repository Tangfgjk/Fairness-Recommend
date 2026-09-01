from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from common import data_fin_root, front_dir, load_raw_dataset, minmax_from_train, read_json, write_json
from q_mirt import QConstrainedMIRT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-constrained MIRT from Data_Fin/raw and adapt only test-student theta.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument("--epoch", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--q-normalization", choices=["none", "mean"], default="mean")
    parser.add_argument("--test-theta-steps", type=int, default=50)
    parser.add_argument("--test-theta-lr", type=float, default=0.005)
    parser.add_argument(
        "--test-theta-prior-weight",
        type=float,
        default=0.01,
        help="MAP prior weight for frozen-item test-learner theta estimation.",
    )
    parser.add_argument(
        "--test-theta-bound-std",
        type=float,
        default=3.0,
        help="Clamp adapted theta componentwise to train_mean +/- this many train standard deviations.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def loader(frame: pd.DataFrame, uid_to_local: dict[int, int], batch_size: int, shuffle: bool) -> DataLoader:
    local = frame["uid"].map(uid_to_local)
    if local.isna().any():
        raise ValueError("MIRT training frame contains a user outside the outer-train cohort")
    dataset = TensorDataset(
        torch.as_tensor(local.to_numpy(dtype=np.int64), dtype=torch.long),
        torch.as_tensor(frame["question"].to_numpy(dtype=np.int64), dtype=torch.long),
        torch.as_tensor(frame["response"].to_numpy(dtype=np.float32), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def adapt_theta(
    model: QConstrainedMIRT,
    interactions: pd.DataFrame,
    theta_mean: torch.Tensor,
    theta_std: torch.Tensor,
    steps: int,
    lr: float,
    prior_weight: float,
    bound_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Estimate one held-out learner's theta while keeping all item parameters frozen.

    This is MAP person scoring, not an additional fit of the MIRT item model.  The
    outer-train theta distribution supplies both the initialization and a
    dimension-wise Gaussian prior, which prevents sparse/all-correct histories
    from producing implausibly large ability vectors.
    """
    if interactions.empty:
        return theta_mean.detach().cpu()
    items = torch.as_tensor(interactions["question"].to_numpy(dtype=np.int64), device=device)
    labels = torch.as_tensor(interactions["response"].to_numpy(dtype=np.float32), device=device)
    prior_mean = theta_mean.detach().to(device=device, dtype=torch.float32)
    prior_std = theta_std.detach().to(device=device, dtype=torch.float32).clamp_min(1e-3)
    lower = prior_mean - float(bound_std) * prior_std
    upper = prior_mean + float(bound_std) * prior_std
    theta = prior_mean.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    model.eval()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model.logits_with_theta(theta.unsqueeze(0).expand(len(items), -1), items)
        response_loss = functional.binary_cross_entropy_with_logits(logits, labels)
        prior_loss = ((theta - prior_mean) / prior_std).square().mean()
        loss = response_loss + float(prior_weight) * prior_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.copy_(torch.maximum(torch.minimum(theta, upper), lower))
    return theta.detach().cpu()


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    output = front_dir(args.dataset, root) / "mirt"
    protocol = read_json(front_dir(args.dataset, root) / "protocol.json")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "mirt_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} exists. Use --force to retrain.")

    train_uids = [int(uid) for uid in protocol["outer_train_uids"]]
    test_uids = [int(uid) for uid in protocol["outer_test_uids"]]
    uid_to_local = {uid: index for index, uid in enumerate(train_uids)}
    train_frame = raw.interactions[raw.interactions["uid"].isin(train_uids)].copy()
    train_loader = loader(train_frame, uid_to_local, args.batch_size, shuffle=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = QConstrainedMIRT(len(train_uids), torch.as_tensor(raw.q_matrix), args.q_normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epoch + 1):
        model.train()
        losses: list[float] = []
        for local_ids, items, labels in train_loader:
            local_ids, items, labels = local_ids.to(device), items.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.logits(local_ids, items)
            loss = functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        history.append(row)
        print(f"[Epoch {epoch}] q_mirt_train_loss={row['train_loss']:.6f}")

    observed_items = torch.as_tensor(sorted(train_frame["question"].unique().tolist()), dtype=torch.long, device=device)
    fallback_report = model.set_unseen_item_fallbacks(observed_items)
    with torch.no_grad():
        effective_a = model.effective_a().detach().cpu().numpy().astype(np.float32)
        b_values = model.b.weight.detach().cpu().numpy().astype(np.float32)
        train_theta = model.theta.weight.detach().cpu().numpy().astype(np.float32)

    train_theta_mean = train_theta.mean(axis=0).astype(np.float32)
    train_theta_std = train_theta.std(axis=0).astype(np.float32)
    # A zero variance dimension should behave as a fixed prior dimension rather
    # than creating an undefined standardized MAP penalty.
    train_theta_std = np.maximum(train_theta_std, 1e-3)

    theta_all = np.zeros((raw.student_count, raw.concept_count), dtype=np.float32)
    theta_all[np.asarray(train_uids, dtype=np.int64)] = train_theta
    test_frame = raw.interactions[raw.interactions["uid"].isin(test_uids)]
    for position, uid in enumerate(test_uids, start=1):
        theta_all[uid] = adapt_theta(
            model,
            test_frame[test_frame["uid"] == uid],
            torch.as_tensor(train_theta_mean),
            torch.as_tensor(train_theta_std),
            args.test_theta_steps,
            args.test_theta_lr,
            args.test_theta_prior_weight,
            args.test_theta_bound_std,
            device,
        ).numpy()
        if position % 50 == 0 or position == len(test_uids):
            print(f"adapted frozen-item theta for {position}/{len(test_uids)} outer-test learners")

    item_train_mask = np.zeros(raw.exercise_count, dtype=bool)
    item_train_mask[np.unique(train_frame["question"].to_numpy(dtype=np.int64))] = True
    a_norm, a_norm_stats = minmax_from_train(np.linalg.norm(effective_a, axis=1), item_train_mask, default=0.0)
    b_norm, b_norm_stats = minmax_from_train(b_values[:, 0], item_train_mask, default=0.5)
    theta_norm, theta_norm_stats = minmax_from_train(np.linalg.norm(theta_all, axis=1), np.isin(np.arange(raw.student_count), train_uids), default=0.5)
    np.savetxt(output / "a_param_q_constrained.csv", effective_a, delimiter=",")
    np.savetxt(output / "b_param.csv", b_values, delimiter=",")
    np.savetxt(output / "theta_param.csv", theta_all, delimiter=",")
    np.save(output / "a_param_q_constrained.npy", effective_a)
    np.save(output / "b_param.npy", b_values)
    np.save(output / "theta_param.npy", theta_all)
    np.save(output / "exercise_discrimination_norm.npy", a_norm.astype(np.float32))
    np.save(output / "exercise_difficulty_norm.npy", b_norm.astype(np.float32))
    np.save(output / "learner_theta_norm.npy", theta_norm.astype(np.float32))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_uids": train_uids,
            "q_normalization": args.q_normalization,
            "concept_count": raw.concept_count,
        },
        output / "q_mirt.pt",
    )
    write_json(
        manifest_path,
        {
            "dataset": raw.name,
            "model": "QConstrainedMIRT",
            "formula": "sigmoid(theta_u^T ((Q_e * softplus(a_e)) / q_count_e) - b_e)",
            "q_normalization": args.q_normalization,
            "q_used_in_forward": True,
            "train_users": len(train_uids),
            "test_users_theta_adapted_with_frozen_item_parameters": len(test_uids),
            "test_theta_estimation": {
                "method": "MAP_person_scoring_with_frozen_item_parameters",
                "initialization": "outer_train_theta_mean",
                "prior": "dimensionwise_gaussian_from_outer_train_theta",
                "item_parameters_updated_for_test_users": False,
                "empty_history_policy": "outer_train_theta_mean",
            },
            "item_count": raw.exercise_count,
            "concept_count": raw.concept_count,
            "train_interactions": int(len(train_frame)),
            "test_adaptation_interactions": int(len(test_frame)),
            "fallback": fallback_report,
            "normalization": {
                "difficulty": b_norm_stats,
                "discrimination": a_norm_stats,
                "theta": theta_norm_stats,
                "fitted_on": "outer_train_only",
            },
            "hyperparameters": {
                "epoch": args.epoch,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "test_theta_steps": args.test_theta_steps,
                "test_theta_lr": args.test_theta_lr,
                "test_theta_prior_weight": args.test_theta_prior_weight,
                "test_theta_bound_std": args.test_theta_bound_std,
            },
            "history": history,
            "files": {
                "a": "a_param_q_constrained.csv",
                "b": "b_param.csv",
                "theta": "theta_param.csv",
                "checkpoint": "q_mirt.pt",
            },
        },
    )
    print(f"saved Q-constrained MIRT outputs: {output}")


if __name__ == "__main__":
    main()
