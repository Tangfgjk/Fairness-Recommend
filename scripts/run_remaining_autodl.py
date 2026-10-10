"""Run only the unfinished final comparison model/seed jobs on AutoDL.

The experiment uses (theta_m, theta_f, rec) = (0.8, 0.8, 50) and the
drop-exfr0.00 graph. Algebra2005 uses the original split; Eedi uses the
20261007 student resplit. Data_Fin_New must be copied separately from Git.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_final_dropzero_three_seeds import (
    ROOT,
    THRESHOLDS,
    TOP_KS,
    dataset_config,
    metric_complete,
    validate_inputs,
)


JOBS = {
    "algebra2005": {
        "RotatE": (2025, 2026),
        "DistMult": (2024, 2025, 2026),
        "ComplEx": (2024, 2025, 2026),
    },
    "Eedi": {
        "RotatE": (2024, 2025, 2026),
        "DistMult": (2024, 2025, 2026),
        "ComplEx": (2024, 2025, 2026),
    },
}


def command_for(config, model: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "comparison_models" / "run_v9_comparison_experiments.py"),
        "--dataset", config.name,
        "--data-root", str(config.data_root),
        "--graph-subdir", config.graph_subdir,
        "--run-root", str(config.comparison_root),
        "--run-id", config.run_id,
        "--seeds", str(seed),
        "--models", model,
        "--top-ks", ",".join(map(str, TOP_KS)),
        "--ep-thresholds", ",".join(map(str, THRESHOLDS)),
        "--ep-calibration-root", str(config.data_root / "_ep_readouts"),
        "--cuda", "true",
    ]
    if config.comparison_run.exists():
        command.append("--resume")
    return command


def result_path(config, model: str, seed: int) -> Path:
    return config.comparison_run / model / f"seed{seed}" / "eval" / "metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *JOBS), default="all")
    parser.add_argument("--plan", action="store_true", help="Validate inputs and print jobs without training")
    args = parser.parse_args()
    datasets = JOBS if args.dataset == "all" else {args.dataset: JOBS[args.dataset]}

    if not args.plan:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable in {sys.executable}")

    pending = 0
    completed = 0
    for dataset, jobs in datasets.items():
        config = dataset_config(dataset)
        validate_inputs(config)
        print(f"\nDataset: {dataset}\nGraph: {config.graph}\nResults: {config.comparison_run}", flush=True)
        for model, seeds in jobs.items():
            for seed in seeds:
                target = result_path(config, model, seed)
                if metric_complete(target):
                    completed += 1
                    print(f"SKIP complete: {dataset}/{model}/seed{seed}", flush=True)
                    continue
                pending += 1
                command = command_for(config, model, seed)
                print(f"{'PLAN' if args.plan else 'RUN'}: {dataset}/{model}/seed{seed}", flush=True)
                print(subprocess.list2cmdline(command), flush=True)
                if args.plan:
                    continue
                subprocess.run(command, cwd=ROOT, check=True)
                if not metric_complete(target):
                    raise RuntimeError(f"Training returned successfully but evaluated metrics are missing: {target}")
    print(f"\nSelected jobs: {pending + completed}; completed: {completed}; "
          f"{'to run' if args.plan else 'processed'}: {pending}", flush=True)


if __name__ == "__main__":
    main()
