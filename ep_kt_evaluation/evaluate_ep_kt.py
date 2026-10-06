"""Post-hoc Ep evaluation using the frozen EKTM response model.

This does not alter recommendation checkpoints or legacy Ep_sim files.  The
state is a per-concept average of predicted correctness over fixed Q-linked
anchor exercises, because the saved EKTM mastery heads had zero loss weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "front_pipeline"))
from common import load_raw_dataset  # noqa: E402
from train_ektm_mirt import legacy_components, make_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simulated KT mastery gain after top-N recommendations.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scores-file", type=Path, required=True, help="A run's eval/uid_ex_scores.json or uid_ex_scores.pkl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-fin-root", type=Path, default=PROJECT / "Data_Fin")
    parser.add_argument("--raw-name", default=None)
    parser.add_argument("--top-n", default="10", help="Comma-separated recommendation lengths, e.g. 5,10,20")
    parser.add_argument("--draws", type=int, default=30, help="Independent Bernoulli draws per student")
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=None, help="Smoke-test limit; omit for final evaluation")
    parser.add_argument("--save-traces", action="store_true", help="Also save every simulated response; may be large")
    return parser.parse_args()


def load_scores(path: Path) -> list[tuple[str, list[float]]]:
    if path.suffix.lower() == ".pkl":
        import pickle

        with path.open("rb") as fp:
            data = pickle.load(fp)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    rows = [(str(row["uid"]), row["scores"]) if isinstance(row, dict) else (str(row[0]), row[1]) for row in data]
    if len({uid for uid, _ in rows}) != len(rows):
        raise ValueError("Recommendation scores contain duplicate student IDs")
    return rows


def choose_response(prob: float, mode: str, rng: np.random.Generator | None) -> int:
    if mode == "threshold":
        return int(prob >= 0.5)
    if mode == "bernoulli" and rng is not None:
        return int(rng.random() < prob)
    raise ValueError(f"Unsupported response mode: {mode}")


def normalized_gain(before: np.ndarray, after: np.ndarray) -> float:
    denominator = float(np.sum(1.0 - before))
    # Match Eval_Global_Gain_Mastery in the supplied reference script.
    if denominator <= 1e-5:
        return 0.0
    return float(np.sum(after - before) / denominator)


def load_frozen_kt(dataset: str, root: Path, raw, device: torch.device):
    front = root / dataset / "front_features"
    checkpoint_path = front / "ektm_mirt" / "best.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported KT checkpoint: {checkpoint_path}")
    config = SimpleNamespace(**checkpoint["config"])
    tokens = np.load(front / "semantic_kg_features" / "text_embeddings" / "exercise_text_tokens.npy")
    concept_tokens = np.zeros((raw.concept_count, int(config.max_text_len)), dtype=np.int64)
    mirt = front / "mirt"
    a_param = np.load(mirt / "a_param_q_constrained.npy").astype(np.float32)
    b_param = np.load(mirt / "b_param.npy").astype(np.float32)
    model = make_model(
        legacy_components(), tokens, concept_tokens, raw, a_param, b_param,
        len(checkpoint["vocab"]), config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    # The checkpoint restores its own Q, token, and MIRT buffers.  The external
    # Q used to aggregate concept states must still name the same concepts.
    if not np.array_equal(model.q_matrix.cpu().numpy(), raw.q_matrix):
        raise ValueError(f"Current Q matrix differs from frozen KT checkpoint: {dataset}")
    return model, config, checkpoint_path


@torch.inference_mode()
def exercise_representations(model, exercise_count: int, device: torch.device) -> torch.Tensor:
    blocks = []
    for start in range(0, exercise_count, 256):
        ids = torch.arange(start, min(start + 256, exercise_count), device=device)
        blocks.append(model.exercise_representation(ids))
    return torch.cat(blocks, dim=0)


@torch.inference_mode()
def replay_history(model, representations: torch.Tensor, history, device: torch.device) -> torch.Tensor:
    hidden = torch.zeros(1, model.hidden_size, device=device)
    for row in history.itertuples(index=False):
        exercise = int(row.question)
        answer = float(row.response)
        step = torch.cat((representations[exercise : exercise + 1], torch.tensor([[answer]], device=device)), dim=1)
        hidden = model.gru_cell(step, hidden)
    return hidden


@torch.inference_mode()
def answer_probability(model, hidden: torch.Tensor, representation: torch.Tensor) -> float:
    value = model.response_head(torch.cat((hidden, representation.unsqueeze(0)), dim=1))
    return float(torch.sigmoid(value).item())


@torch.inference_mode()
def concept_state(model, hidden: torch.Tensor, representations: torch.Tensor, q: np.ndarray) -> np.ndarray:
    """Probe the same fixed exercise bank before and after recommendations."""
    predicted = []
    for block in representations.split(512, dim=0):
        repeated_hidden = hidden.expand(block.size(0), -1)
        logits = model.response_head(torch.cat((repeated_hidden, block), dim=1)).squeeze(1)
        predicted.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(predicted)
    coverage = q.sum(axis=0)
    if np.any(coverage == 0):
        raise ValueError("At least one knowledge concept has no Q-linked anchor exercise")
    return (q.T @ probabilities / coverage).astype(np.float64)


@torch.inference_mode()
def simulate_one(model, initial_hidden: torch.Tensor, representations: torch.Tensor, q: np.ndarray,
                 ranked_exercises: list[int], top_values: list[int], before: np.ndarray,
                 mode: str, rng: np.random.Generator | None) -> tuple[dict[int, float], list[dict]]:
    hidden = initial_hidden.clone()
    gains = {}
    steps = []
    snapshots = set(top_values)
    for step_no, exercise in enumerate(ranked_exercises[: max(top_values)], start=1):
        rep = representations[exercise]
        prob = answer_probability(model, hidden, rep)
        answer = choose_response(prob, mode, rng)
        hidden = model.gru_cell(torch.cat((rep.unsqueeze(0), torch.tensor([[float(answer)]], device=rep.device)), dim=1), hidden)
        steps.append({"position": step_no, "exercise": exercise, "prob_correct": prob, "simulated_response": answer})
        if step_no in snapshots:
            gains[step_no] = normalized_gain(before, concept_state(model, hidden, representations, q))
    return gains, steps


@torch.inference_mode()
def validate_on_history(model, representations: torch.Tensor, history, device: torch.device) -> list[tuple[float, int]]:
    """Predict held-out trailing historical answers using only preceding answers."""
    if len(history) < 5:
        return []
    cutoff = max(1, int(len(history) * 0.8))
    hidden = replay_history(model, representations, history.iloc[:cutoff], device)
    outcomes = []
    for row in history.iloc[cutoff:].itertuples(index=False):
        exercise = int(row.question)
        observed = int(row.response)
        rep = representations[exercise]
        probability = answer_probability(model, hidden, rep)
        outcomes.append((probability, observed))
        hidden = model.gru_cell(torch.cat((rep.unsqueeze(0), torch.tensor([[float(observed)]], device=device)), dim=1), hidden)
    return outcomes


def main() -> None:
    args = parse_args()
    if args.draws < 1:
        raise ValueError("--draws must be positive")
    top_values = sorted({int(item) for item in args.top_n.split(",") if item.strip()})
    if not top_values or min(top_values) < 1:
        raise ValueError("--top-n must contain positive integers")
    raw_name = args.raw_name or ("raw_compact" if args.dataset == "XES3G5M-sub-small" else "raw")
    raw = load_raw_dataset(args.dataset, args.data_fin_root, raw_name)
    if max(top_values) > raw.exercise_count:
        raise ValueError("Recommendation length exceeds exercise count")
    device = torch.device(args.device)
    model, config, checkpoint_path = load_frozen_kt(args.dataset, args.data_fin_root, raw, device)
    representations = exercise_representations(model, raw.exercise_count, device)
    q = raw.q_matrix.astype(np.float64)
    scores = load_scores(args.scores_file)
    test_uids = set(raw.test_uids)
    score_uids = {int(uid[3:]) if uid.startswith("uid") else int(uid) for uid, _ in scores}
    if args.max_users is None and score_uids != test_uids:
        raise ValueError(f"Score users differ from test split: missing={len(test_uids - score_uids)}, extra={len(score_uids - test_uids)}")
    if args.max_users is not None:
        scores = scores[: args.max_users]
    per_user = []
    traces = []
    historical_predictions = []
    for uid, values in scores:
        uid_number = int(uid[3:]) if uid.startswith("uid") else int(uid)
        if uid_number not in test_uids:
            raise ValueError(f"Score file contains non-test student {uid}")
        if len(values) != raw.exercise_count:
            raise ValueError(f"{uid} has {len(values)} scores; expected {raw.exercise_count}")
        history = raw.interactions.loc[raw.interactions["uid"] == uid_number].tail(int(config.max_seq_len))
        if history.empty:
            continue
        historical_predictions.extend(validate_on_history(model, representations, history, device))
        initial_hidden = replay_history(model, representations, history, device)
        before = concept_state(model, initial_hidden, representations, q)
        ranked = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
        for mode in ("bernoulli", "threshold"):
            draws = args.draws if mode == "bernoulli" else 1
            for draw in range(draws):
                rng = np.random.default_rng(np.random.SeedSequence([args.simulation_seed, uid_number, draw])) if mode == "bernoulli" else None
                gains, steps = simulate_one(model, initial_hidden, representations, q, ranked, top_values, before, mode, rng)
                if args.save_traces:
                    traces.append({"uid": uid, "mode": mode, "draw": draw, "steps": steps})
                for top_n, gain in gains.items():
                    per_user.append({"uid": uid, "mode": mode, "draw": draw, "top_n": top_n, "ep": gain})
    summary = []
    for mode in ("bernoulli", "threshold"):
        for top_n in top_values:
            grouped = {}
            for row in per_user:
                if row["mode"] == mode and row["top_n"] == top_n and row["ep"] is not None:
                    grouped.setdefault(row["uid"], []).append(row["ep"])
            user_means = [float(np.mean(grouped[uid])) for uid in sorted(grouped)]
            summary.append({"mode": mode, "top_n": top_n, "users": len(user_means),
                            "draws": args.draws if mode == "bernoulli" else 1,
                            "mean": float(np.mean(user_means)) if user_means else None,
                            "std_across_users": float(np.std(user_means, ddof=1)) if len(user_means) > 1 else None})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if historical_predictions:
        predicted = np.asarray([p for p, _ in historical_predictions], dtype=np.float64)
        observed = np.asarray([y for _, y in historical_predictions], dtype=np.float64)
        validation = {"historical_answers": len(historical_predictions),
                      "brier_score": float(np.mean((predicted - observed) ** 2)),
                      "accuracy_at_0_5": float(np.mean((predicted >= 0.5) == observed)),
                      "mean_predicted_correct": float(np.mean(predicted)),
                      "observed_correct_rate": float(np.mean(observed)),
                      "protocol": "first 80% of each student's last max_seq_len historical answers warm up KT; remaining 20% evaluated sequentially"}
    else:
        validation = {"historical_answers": 0}
    metadata = {"protocol_version": 2, "dataset": args.dataset, "scores_file": str(args.scores_file), "checkpoint": str(checkpoint_path),
                "raw_name": raw_name, "mastery_loss_weight": float(config.mastery_loss_weight),
                "state_definition": "mean frozen-EKTM predicted correctness on fixed Q-linked exercises",
                "response_rules": {"bernoulli": "random < p_correct", "threshold": "p_correct >= 0.5"},
                "simulation_seed": args.simulation_seed, "top_n": top_values, "bernoulli_draws": args.draws,
                "exploratory_max_users": args.max_users,
                "kt_response_validation": validation}
    outputs = [("ep_kt_summary.json", {"metadata": metadata, "results": summary}),
               ("ep_kt_per_draw.json", per_user)]
    if args.save_traces:
        outputs.append(("ep_kt_trace.json", traces))
    for filename, payload in outputs:
        (args.output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
