"""Evaluate Ep from a trained full-history concept readout and paper score files."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core import (DATASETS, PROJECT, ConceptReadout, initial_bias_from_model,
                  load_source, sha256_file)

sys.path.insert(0, str(PROJECT / "ep_kt_evaluation"))
from evaluate_ep_kt import answer_probability, load_scores, normalized_gain  # noqa: E402
DEFAULT_TOP_N = tuple(range(5, 51, 5))


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(not 0 <= item <= 1 for item in values):
        raise ValueError("Thresholds must be probabilities in [0, 1]")
    return values


def parse_top_n(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values) or len(values) != len(set(values)):
        raise ValueError("Top-N values must be unique positive integers")
    return sorted(values)
DEFAULT_THRESHOLDS = {
    "Eedi": "0.4,0.5,0.6,0.7",
    "algebra2005": "0.4,0.5,0.6,0.7",
    "XES3G5M-sub-small": "0.4,0.5,0.6,0.7",
}


def numeric_uid(value: str) -> int:
    return int(value[3:]) if value.startswith("uid") else int(value)


@dataclass
class Context:
    dataset: str
    raw: object
    base: object
    readout: ConceptReadout
    representations: torch.Tensor
    hidden: np.ndarray
    mastery: np.ndarray
    report: dict
    device: torch.device
    checkpoint_path: Path


@torch.inference_mode()
def load_context(dataset: str, data_root: Path, calibration_root: Path,
                 device: torch.device, allow_smoke: bool = False) -> Context:
    raw, base, _, checkpoint_path, reps = load_source(dataset, data_root, device)
    folder = calibration_root / dataset
    report = json.loads((folder / "training_report.json").read_text(encoding="utf-8"))
    if report["dataset"] != dataset:
        raise ValueError("Training report dataset mismatch")
    if report["smoke_only"] and not allow_smoke:
        raise ValueError("Smoke-trained readout cannot be used for complete Ep evaluation")
    sources = (
        (checkpoint_path, "source_checkpoint_sha256"),
        (raw.root / "interactions_all.csv", "source_interactions_sha256"),
        (raw.root / "Q.txt", "source_q_sha256"),
    )
    for path, key in sources:
        if sha256_file(path) != report[key]:
            raise ValueError(f"Source changed since readout training: {path}")
    try:
        payload = torch.load(folder / "readout.pt", map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(folder / "readout.pt", map_location="cpu")
    if payload["concept_count"] != raw.concept_count or payload["exercise_count"] != raw.exercise_count:
        raise ValueError("Readout dimensions do not match current dataset")
    readout = ConceptReadout(base.hidden_size, raw.concept_count,
                             initial_bias_from_model(base)).to(device)
    readout.load_state_dict(payload["model_state_dict"], strict=True)
    readout.eval()
    hidden = np.load(folder / "full_history_hidden.npy")
    mastery = np.load(folder / "stu2know_mastery.npy")
    if hidden.shape != (raw.student_count, base.hidden_size) or mastery.shape != (raw.student_count, raw.concept_count):
        raise ValueError("Exported hidden/mastery shape mismatch")
    available = np.isfinite(hidden).all(axis=1)
    if not report["smoke_only"] and not available.all():
        raise ValueError("Complete evaluation requires full-history state for every student")
    for uids in np.array_split(np.flatnonzero(available), max(1, int(np.ceil(available.sum() / 512)))):
        states = torch.as_tensor(hidden[uids], device=device)
        difference = float(np.max(np.abs(readout.mastery(states).cpu().numpy() - mastery[uids])))
        if difference > 1e-6:
            raise ValueError(f"Exported mastery does not match readout: {difference}")
    return Context(dataset, raw, base, readout, reps, hidden, mastery,
                   report, device, checkpoint_path)


@torch.inference_mode()
def simulate(context: Context, uid: int, ranked: list[int],
             top_values: list[int], threshold: float, trace: bool):
    hidden = torch.as_tensor(context.hidden[uid], device=context.device).reshape(1, -1).clone()
    before = context.mastery[uid].astype(np.float64)
    gains = {}
    correct_rates = {}
    steps = []
    correct = 0
    snapshots = set(top_values)
    for position, exercise in enumerate(ranked[:max(top_values)], start=1):
        rep = context.representations[exercise]
        p = answer_probability(context.base, hidden, rep)
        if not np.isfinite(p):
            raise ValueError(f"Non-finite predicted response for uid{uid} ex{exercise}")
        answer = int(p >= threshold)
        step = torch.cat((rep.unsqueeze(0),
                          torch.tensor([[float(answer)]], device=context.device)), dim=1)
        hidden = context.base.gru_cell(step, hidden)
        correct += answer
        if trace:
            steps.append({"position": position, "exercise": exercise,
                          "prob_correct": p, "simulated_response": answer})
        if position in snapshots:
            after = context.readout.mastery(hidden).squeeze(0).cpu().numpy().astype(np.float64)
            gains[position] = normalized_gain(before, after)
            correct_rates[position] = correct / position
    return gains, correct_rates, steps


def evaluate_entry(context: Context, scores_file: Path, output_dir: Path,
                   top_values: list[int], thresholds: list[float],
                   max_users: int | None = None, save_traces: bool = False) -> dict:
    scores = load_scores(scores_file)
    expected = set(context.raw.test_uids)
    observed = {numeric_uid(uid) for uid, _ in scores}
    if max_users is None and observed != expected:
        raise ValueError(f"Score file test cohort differs: missing={len(expected-observed)} extra={len(observed-expected)}")
    if max_users is not None:
        scores = [(uid, values) for uid, values in scores
                  if numeric_uid(uid) in expected and np.isfinite(context.hidden[numeric_uid(uid)]).all()][:max_users]
    if not scores:
        raise ValueError("No eligible test student in score file")
    per_user = []
    traces = []
    for label, values in scores:
        uid = numeric_uid(label)
        if uid not in expected or len(values) != context.raw.exercise_count:
            raise ValueError(f"Invalid score row for {label}")
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError(f"Non-finite scores for {label}")
        if not np.isfinite(context.hidden[uid]).all():
            raise ValueError(f"Missing full-history state for {label}")
        ranked = sorted(range(len(values)), key=lambda item: (-float(values[item]), item))
        for threshold in thresholds:
            gains, correct_rates, steps = simulate(context, uid, ranked,
                                                   top_values, threshold, save_traces)
            if save_traces:
                traces.append({"uid": label, "threshold": threshold, "steps": steps})
            for top_n in top_values:
                per_user.append({"uid": label, "threshold": threshold, "top_n": top_n,
                                 "ep": gains[top_n], "simulated_correct_rate": correct_rates[top_n]})
    results = []
    for threshold in thresholds:
        for top_n in top_values:
            rows = [row for row in per_user if row["threshold"] == threshold and row["top_n"] == top_n]
            values = np.asarray([row["ep"] for row in rows], dtype=np.float64)
            results.append({"threshold": threshold, "top_n": top_n, "users": len(rows),
                            "mean": float(values.mean()),
                            "std_across_users": float(values.std(ddof=1)) if len(values) > 1 else None,
                            "positive_gain_fraction": float(np.mean(values > 0)),
                            "simulated_correct_rate": float(np.mean([row["simulated_correct_rate"] for row in rows]))})
    metadata = {
        "dataset": context.dataset, "scores_file": str(scores_file),
        "base_checkpoint": str(context.checkpoint_path),
        "state_definition": "trained pre-answer concept readout after complete historical replay",
        "response_rule": "frozen EKTM response probability >= fixed threshold",
        "thresholds": thresholds, "top_n": top_values,
        "history_attempts": context.report["processed_attempts"],
        "original_max_seq_len": context.report["original_max_seq_len"],
        "exploratory_max_users": max_users,
        "test_trailing_20pct": context.report["test_trailing_20pct"],
        "interpretation_limit": "Simulated model-estimated learning gain; no observed post-recommendation answers",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    (output_dir / "ep_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "ep_per_user.json").write_text(json.dumps(per_user, ensure_ascii=False), encoding="utf-8")
    if save_traces:
        (output_dir / "ep_traces.json").write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--scores-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-fin-root", type=Path, default=PROJECT / "Data_Fin_New")
    parser.add_argument("--calibration-root", type=Path, default=PROJECT / "Data_Fin_New" / "_ep_readouts")
    parser.add_argument("--top-n", default=",".join(map(str, DEFAULT_TOP_N)))
    parser.add_argument("--thresholds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    args = parser.parse_args()
    if args.max_users is not None and args.max_users < 1:
        parser.error("--max-users must be positive")
    top_values = parse_top_n(args.top_n)
    thresholds = parse_float_list(args.thresholds or DEFAULT_THRESHOLDS[args.dataset])
    context = load_context(args.dataset, args.data_fin_root, args.calibration_root,
                           torch.device(args.device), args.allow_smoke)
    if max(top_values) > context.raw.exercise_count:
        raise ValueError("Top-N exceeds exercise count")
    payload = evaluate_entry(context, args.scores_file, args.output_dir,
                             top_values, thresholds, args.max_users, args.save_traces)
    print(json.dumps({"dataset": args.dataset, "output": str(args.output_dir),
                      "rows": len(payload["results"]),
                      "users": payload["results"][0]["users"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
