"""One-command Ep evaluation of all three paper datasets and nine paper models."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT.parents[2] / "结果汇总" / "KG4ER论文最终结果" / "消融结果" / "runs" / "algebra2005"
DATASETS = {
    "Eedi": PROJECT / "runs" / "Eedi",
    "algebra2005": PAPER_ROOT,
    "XES3G5M-sub-small": PROJECT / "runs" / "XES3G5M-sub-small",
}
KG_BASELINES = ("RotatE", "DistMult", "ComplEx", "TransE", "TransE-adv")
TRADITIONAL = ("EB-CF", "SB-CF", "CBF")
SEEDS = (2024, 2025, 2026)


def discover() -> list[tuple[str, str, str, Path]]:
    """Require the exact 3 + 15 + 3 paper score files per dataset."""
    entries = []
    for dataset, root in DATASETS.items():
        stem = "Eedi" if dataset == "Eedi" else dataset
        semantic = root / f"{stem}_vfin8_semantic_3seeds"
        comparison = root / f"{stem}_vfin8_comparison_3seeds"
        for seed in SEEDS:
            entries.append((dataset, "SemanticConvE_feature_only", str(seed),
                            semantic / "SemanticConvE_feature_only" / f"seed{seed}" / "eval" / "uid_ex_scores.json"))
        for model in KG_BASELINES:
            for seed in SEEDS:
                entries.append((dataset, model, str(seed),
                                comparison / model / f"seed{seed}" / "eval" / "uid_ex_scores.json"))
        for model in TRADITIONAL:
            entries.append((dataset, model, "single", comparison / "traditional_baselines" / model / "eval" / "uid_ex_scores.json"))
    missing = [str(path) for _, _, _, path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paper score files:\n" + "\n".join(missing))
    if len(entries) != 63 or len({(d, m, s) for d, m, s, _ in entries}) != 63:
        raise AssertionError("Expected 63 unique paper model runs")
    return entries


def evaluate(entry, args):
    dataset, model, seed, scores = entry
    output_dir = args.output_root / dataset / model / f"seed{seed}"
    summary_file = output_dir / "ep_kt_summary.json"
    command = [sys.executable, str(Path(__file__).with_name("evaluate_ep_kt.py")),
               "--dataset", dataset, "--scores-file", str(scores), "--output-dir", str(output_dir),
               "--top-n", ",".join(map(str, args.top_n)), "--draws", str(args.draws),
               "--simulation-seed", str(args.simulation_seed), "--device", args.device]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"{dataset}/{model}/{seed} failed:\n{completed.stderr}")
    return entry, json.loads(summary_file.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ep@N for all 3 datasets and all 9 paper models")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "ep_kt_evaluation" / "results")
    parser.add_argument("--top-n", default="10")
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.top_n = sorted({int(x.strip()) for x in args.top_n.split(",") if x.strip()})
    if not args.top_n or args.top_n[0] < 1 or args.draws < 1 or args.workers < 1:
        parser.error("top-n, draws, and workers must be positive")
    entries = discover()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(entries)} complete paper runs across three datasets; using {args.workers} workers.", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, entry, args): entry for entry in entries}
        try:
            for i, future in enumerate(as_completed(futures), 1):
                (dataset, model, seed, scores), payload = future.result()
                summary_file = args.output_root / dataset / model / f"seed{seed}" / "ep_kt_summary.json"
                for result in payload["results"]:
                    rows.append({"dataset": dataset, "model": model, "seed": seed,
                                 "mode": result["mode"], "top_n": result["top_n"],
                                 "users": result["users"], "draws": result["draws"],
                                 "ep_mean": result["mean"], "summary_file": str(summary_file),
                                 "scores_file": str(scores)})
                print(f"[{i}/{len(entries)}] {dataset} {model} seed={seed}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda row: (row["dataset"], row["model"], row["seed"], row["mode"], row["top_n"]))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["model"], row["mode"], row["top_n"]), []).append(row)
    aggregate = []
    for (dataset, model, mode, top_n), group in sorted(grouped.items()):
        expected = 1 if model in TRADITIONAL else 3
        if len(group) != expected:
            raise AssertionError(f"Incomplete results for {dataset}/{model}/{mode}: {len(group)} vs {expected}")
        if len({row["users"] for row in group}) != 1:
            raise AssertionError(f"Different student counts across {dataset}/{model} seeds")
        values = [float(row["ep_mean"]) for row in group]
        aggregate.append({"dataset": dataset, "model": model, "mode": mode, "top_n": top_n,
                          "users": group[0]["users"], "recommendation_seeds": expected,
                          "ep_mean": statistics.mean(values),
                          "ep_std_across_seeds": statistics.stdev(values) if expected > 1 else "",
                          "seed_values": ";".join(f"{row['seed']}:{row['ep_mean']:.8f}" for row in group)})
    for name, data in (("ep_kt_per_run.csv", rows), ("ep_kt_paper_models.csv", aggregate)):
        with (args.output_root / name).open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"runs": len(entries), "paper_rows": len(aggregate),
                      "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
