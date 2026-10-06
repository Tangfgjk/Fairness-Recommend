"""Evaluate both Ep response rules for all model/seed runs under one dataset root."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run frozen-KT Ep evaluation and summarize recommendation seeds")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--models", default=None, help="Optional comma-separated model directory names")
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--top-n", default="10")
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = {int(value) for value in args.seeds.split(",") if value.strip()}
    models = {value.strip() for value in args.models.split(",") if value.strip()} if args.models else None
    score_files = sorted(args.run_root.glob("*/seed*/eval/uid_ex_scores.json"))
    selected = []
    for score_file in score_files:
        model_name = score_file.parents[2].name
        seed_name = score_file.parents[1].name
        if not seed_name.startswith("seed"):
            continue
        seed = int(seed_name[4:])
        if seed in seeds and (models is None or model_name in models):
            selected.append((model_name, seed, score_file))
    if not selected:
        raise FileNotFoundError("No requested model/seed score files found beneath run root")
    if len({(model, seed) for model, seed, _ in selected}) != len(selected):
        raise ValueError("Duplicate model/seed pairs in run root")
    rows = []
    evaluator = Path(__file__).with_name("evaluate_ep_kt.py")
    for model_name, seed, scores in selected:
        output_dir = args.output_root / model_name / f"seed{seed}"
        summary_file = output_dir / "ep_kt_summary.json"
        cached = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else None
        requested_n = {int(value) for value in args.top_n.split(",") if value.strip()}
        cache_matches = bool(cached) and (
            cached["metadata"]["dataset"] == args.dataset
            and cached["metadata"]["scores_file"] == str(scores)
            and cached["metadata"]["simulation_seed"] == args.simulation_seed
            and {int(row["top_n"]) for row in cached["results"]} == requested_n
            and all(int(row["draws"]) == args.draws for row in cached["results"] if row["mode"] == "bernoulli")
        )
        if args.force or not cache_matches:
            command = [sys.executable, str(evaluator), "--dataset", args.dataset,
                       "--scores-file", str(scores), "--output-dir", str(output_dir),
                       "--top-n", args.top_n, "--draws", str(args.draws),
                       "--simulation-seed", str(args.simulation_seed), "--device", args.device]
            if args.data_fin_root:
                command += ["--data-fin-root", str(args.data_fin_root)]
            subprocess.run(command, check=True)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary["metadata"]["exploratory_max_users"] is not None:
            raise ValueError(f"Exploratory output cannot be aggregated: {summary_file}")
        for result in summary["results"]:
            rows.append({"dataset": args.dataset, "model": model_name, "seed": seed,
                         "mode": result["mode"], "top_n": result["top_n"],
                         "users": result["users"], "draws": result["draws"],
                         "ep_mean": result["mean"], "summary_file": str(summary_file)})
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"], row["mode"], row["top_n"]), []).append(row)
    aggregates = []
    for (dataset, model, mode, top_n), group in sorted(groups.items()):
        values = [float(item["ep_mean"]) for item in group if item["ep_mean"] is not None]
        aggregates.append({"dataset": dataset, "model": model, "mode": mode, "top_n": top_n,
                           "recommendation_seeds": len(values),
                           "ep_mean": statistics.mean(values) if values else None,
                           "ep_std_across_seeds": statistics.stdev(values) if len(values) > 1 else None,
                           "seed_values": ";".join(f"{item['seed']}:{float(item['ep_mean']):.6f}" for item in group if item["ep_mean"] is not None)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, data in (("ep_kt_per_seed.csv", rows), ("ep_kt_across_seeds.csv", aggregates)):
        with (args.output_root / filename).open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"runs": len(selected), "aggregated_rows": len(aggregates),
                      "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
