from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from common import data_fin_root, front_dir, load_raw_dataset, matrix_stats, write_json, write_matrix_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw-data knowledge and exercise forgetting features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument(
        "--theta",
        default="auto",
        help="Positive decay denominator, or 'auto' to calibrate per dataset from positive elapsed intervals.",
    )
    parser.add_argument("--timestamp-unit", choices=["auto", "seconds", "milliseconds", "minutes", "days"], default="auto")
    parser.add_argument(
        "--time-basis",
        choices=["auto", "timestamp_seconds", "interaction_steps"],
        default="auto",
        help="Use timestamp-derived seconds, sequence positions, or infer Eedi as interaction steps.",
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.90,
        help="Positive elapsed-time quantile mapped to --calibration-target when --theta=auto.",
    )
    parser.add_argument(
        "--calibration-target",
        type=float,
        default=0.80,
        help="Forgetting value assigned to the calibration quantile when --theta=auto.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_time_basis(dataset: str, requested: str) -> Literal["timestamp_seconds", "interaction_steps"]:
    if requested == "timestamp_seconds":
        return "timestamp_seconds"
    if requested == "interaction_steps":
        return "interaction_steps"
    # Eedi-sub preserves interaction order but does not supply elapsed wall-clock time.
    return "interaction_steps" if dataset.lower() == "eedi" else "timestamp_seconds"


def elapsed_axis(frame, basis: Literal["timestamp_seconds", "interaction_steps"], timestamp_unit: str) -> np.ndarray:
    if basis == "interaction_steps":
        return np.arange(len(frame), dtype=np.float64)
    values = frame["timestamp"]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        result = numeric.to_numpy(dtype=np.float64)
        if timestamp_unit == "milliseconds" or (timestamp_unit == "auto" and float(np.max(np.abs(result))) >= 1e12):
            return result / 1000.0
        if timestamp_unit == "minutes":
            return result * 60.0
        if timestamp_unit == "days":
            return result * 86400.0
        if timestamp_unit in {"seconds", "auto"}:
            return result
        raise ValueError(f"Unknown timestamp unit: {timestamp_unit}")
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad_value = values.loc[parsed.isna()].iloc[0]
        raise ValueError(f"Cannot parse timestamp: {bad_value!r}")
    return parsed.astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0


def parse_theta(value: str, deltas: np.ndarray, quantile: float, target: float) -> tuple[float, str]:
    if value.strip().lower() != "auto":
        theta = float(value)
        if not math.isfinite(theta) or theta <= 0.0:
            raise ValueError("--theta must be a positive number or 'auto'")
        return theta, "manual"

    if not 0.0 < quantile <= 1.0:
        raise ValueError("--calibration-quantile must be in (0, 1]")
    if not 0.0 < target < 1.0:
        raise ValueError("--calibration-target must be in (0, 1)")
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot auto-calibrate forgetting: no positive elapsed intervals were found")
    anchor = float(np.quantile(positive, quantile))
    return anchor / -math.log1p(-target), "auto_quantile"


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    output = front_dir(args.dataset, root) / "forgetting"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "stu2know_forget.json"
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} exists. Use --force to regenerate.")

    all_deltas: list[float] = []
    last_seen_by_uid: list[np.ndarray] = []
    time_basis = resolve_time_basis(raw.name, args.time_basis)
    item_concepts = [np.flatnonzero(row > 0.0) for row in raw.q_matrix]
    frames_by_uid = {
        int(uid): frame
        for uid, frame in raw.interactions.groupby("uid", sort=False)
    }
    for uid in range(raw.student_count):
        frame = frames_by_uid.get(uid)
        last_seen = np.full(raw.concept_count, np.nan, dtype=np.float64)
        if frame is None or frame.empty:
            last_seen_by_uid.append(last_seen)
            continue
        times = elapsed_axis(frame, time_basis, args.timestamp_unit)
        last_time = float(times[-1])
        for item_id, timestamp in zip(frame["question"].to_numpy(dtype=np.int64), times):
            last_seen[item_concepts[item_id]] = timestamp
        observed = np.isfinite(last_seen)
        deltas = np.abs(last_time - last_seen[observed])
        all_deltas.extend(deltas.tolist())
        last_seen_by_uid.append(last_seen)

    delta_arr = np.asarray(all_deltas, dtype=np.float64)
    theta, theta_source = parse_theta(args.theta, delta_arr, args.calibration_quantile, args.calibration_target)
    # Forgetting is defined only after an observed exposure. An unseen concept
    # therefore starts with zero temporal forgetting rather than being treated
    # as fully forgotten.
    know_forget = np.zeros((raw.student_count, raw.concept_count), dtype=np.float64)
    for uid, last_seen in enumerate(last_seen_by_uid):
        observed = np.isfinite(last_seen)
        if not observed.any():
            continue
        frame = frames_by_uid[uid]
        times = elapsed_axis(frame, time_basis, args.timestamp_unit)
        deltas = np.abs(float(times[-1]) - last_seen[observed])
        know_forget[uid, observed] = 1.0 - np.exp(-deltas / theta)
    know_forget = np.clip(know_forget, 0.0, 1.0)
    q_counts = raw.q_matrix.sum(axis=1).clip(min=1.0)
    exercise_forget = (know_forget @ raw.q_matrix.T) / q_counts[None, :]
    exercise_forget = np.clip(exercise_forget, 0.0, 1.0)
    write_matrix_json(output / "stu2know_forget.json", know_forget)
    write_matrix_json(output / "stu2ex_forget.json", exercise_forget)
    write_matrix_json(front_dir(args.dataset, root) / "stu2know_forget.json", know_forget)
    write_matrix_json(front_dir(args.dataset, root) / "stu2ex_forget.json", exercise_forget)
    positive_deltas = delta_arr[delta_arr > 0.0]
    write_json(
        output / "forgetting_manifest.json",
        {
            "dataset": raw.name,
            "formula": "1 - exp(-elapsed_time / theta)",
            "time_basis": time_basis,
            "theta": theta,
            "theta_source": theta_source,
            "timestamp_unit": args.timestamp_unit,
            "calibration": {
                "quantile": args.calibration_quantile if theta_source == "auto_quantile" else None,
                "target_forgetting": args.calibration_target if theta_source == "auto_quantile" else None,
                "positive_interval_quantile": float(np.quantile(positive_deltas, args.calibration_quantile)) if theta_source == "auto_quantile" else None,
            },
            "unseen_concept_value": 0.0,
            "exercise_aggregation": "mean forgetting over Q-linked concepts",
            "elapsed_intervals": matrix_stats(delta_arr),
            "positive_elapsed_intervals": matrix_stats(positive_deltas),
            "knowledge_forgetting": matrix_stats(know_forget),
            "exercise_forgetting": matrix_stats(exercise_forget),
        },
    )
    print(f"saved raw forgetting outputs: {output}")


if __name__ == "__main__":
    main()
