from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATASETS, canonical_raw_name, data_fin_root


STAGES = [
    "prepare",
    "mirt",
    "ektm",
    "forgetting",
    "semantic",
    "graph",
    "validate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Data_Fin/raw front-feature pipeline and construct an ER graph."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--stages", default="all", help="Comma-separated stages or all")
    parser.add_argument("--mirt-epochs", type=int, default=70)
    parser.add_argument("--mirt-batch-size", type=int, default=1024)
    parser.add_argument("--ektm-epochs", type=int, default=30)
    parser.add_argument("--ektm-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--theta", default="auto", help="Positive decay denominator or 'auto' for per-dataset calibration.")
    parser.add_argument("--timestamp-unit", default="auto", choices=["auto", "seconds", "milliseconds", "minutes", "days"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_stages(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(STAGES)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {','.join(unknown)}")
    return selected


def script_command(script: str, dataset: str, root: Path, raw_name: str, extra: list[str], force: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / script),
        "--dataset",
        dataset,
        "--data-fin-root",
        str(root),
        "--raw-name",
        raw_name,
        *extra,
    ]
    if force:
        command.append("--force")
    return command


def commands_for_dataset(args: argparse.Namespace, dataset: str, root: Path, stages: list[str], raw_name: str) -> list[list[str]]:
    by_stage = {
        "prepare": script_command("prepare_front_protocol.py", dataset, root, raw_name, [], args.force),
        "mirt": script_command(
            "train_q_mirt.py",
            dataset,
            root, raw_name,
            ["--epoch", str(args.mirt_epochs), "--batch-size", str(args.mirt_batch_size), "--lr", "0.001", "--device", args.device],
            args.force,
        ),
        "ektm": script_command(
            "train_ektm_mirt.py",
            dataset,
            root, raw_name,
            ["--epochs", str(args.ektm_epochs), "--batch-size", str(args.ektm_batch_size), "--learning-rate", "0.001", "--device", args.device],
            args.force,
        ),
        "forgetting": script_command(
            "generate_forgetting.py",
            dataset,
            root, raw_name,
            ["--theta", str(args.theta), "--timestamp-unit", args.timestamp_unit],
            args.force,
        ),
        "semantic": script_command("build_semantic_features.py", dataset, root, raw_name, [], args.force),
        "graph": script_command("build_er_graph.py", dataset, root, raw_name, [], args.force),
        "validate": [
            sys.executable,
            str(Path(__file__).resolve().parent / "validate_front_pipeline.py"),
            "--datasets",
            dataset,
            "--data-fin-root",
            str(root),
            "--raw-name",
            raw_name,
            "--require-graph",
            "--output-file",
            str(root / dataset / "front_features" / "front_validation_report.json"),
        ],
    }
    return [by_stage[stage] for stage in stages]


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    stages = selected_stages(args.stages)
    for dataset in datasets:
        raw_name = canonical_raw_name(dataset)
        print(f"\n===== [{dataset}] Data_Fin front pipeline ({raw_name}) =====")
        for command in commands_for_dataset(args, dataset, root, stages, raw_name):
            print(" ".join(command))
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
            if completed.returncode != 0:
                raise SystemExit(f"Stage failed for {dataset}: {' '.join(command)}")
    print("Data_Fin front pipeline completed.")


if __name__ == "__main__":
    main()
