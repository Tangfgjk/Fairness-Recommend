from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from experiment_utils import write_json
from run_dataset_experiments import (
    KGE_EXPERIMENTS,
    TRADITIONAL_BASELINES,
    collect_metrics,
    cuda_enabled,
    experiment_filter,
    resolve_batch_dir,
    run_traditional_baselines,
    train_kge,
    validate_data_dir,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "Data_Fin"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ID-only comparison models on an existing ER graph directory, "
            "usually Data_Fin/<dataset>/er_graph."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--graph-subdir", default="er_graph")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--cuda", default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "all or comma-separated names from "
            "TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF"
        ),
    )
    parser.add_argument("--kge-max-steps", type=int, default=30000)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--negative-sample-size", type=int, default=256)
    parser.add_argument("--kge-hidden-dim", type=int, default=1000)
    parser.add_argument("--kge-gamma", type=float, default=12.0)
    parser.add_argument("--kge-learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu-num", type=int, default=10)
    parser.add_argument("--top-ks", default="10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--ep-top-k", type=int, default=10)
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def graph_dir_for_dataset(data_root: Path, dataset: str, graph_subdir: str) -> Path:
    graph_dir = data_root / dataset / graph_subdir
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    return graph_dir.resolve()


def validate_requested_models(args: argparse.Namespace) -> None:
    if args.models == "all":
        return
    allowed = set(KGE_EXPERIMENTS) | set(TRADITIONAL_BASELINES)
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = sorted(requested - allowed)
    if unknown:
        raise ValueError(f"Unknown comparison models: {','.join(unknown)}")


def write_v9_manifest(args: argparse.Namespace, batch_dir: Path, data_dir: Path, use_cuda: bool, seeds: list[int]) -> None:
    write_json(
        {
            "dataset": args.dataset,
            "data_dir": str(data_dir),
            "graph_subdir": args.graph_subdir,
            "run_dir": str(batch_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "resume": args.resume,
            "cuda_enabled": use_cuda,
            "seeds": seeds,
            "kge_experiments": list(KGE_EXPERIMENTS),
            "traditional_baselines": TRADITIONAL_BASELINES,
            "command": " ".join(sys.argv),
            "notes": "The comparison runner uses ID-only KGE and traditional baselines on the selected ER graph.",
        },
        batch_dir / "manifest.json",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_requested_models(args)
    data_dir = graph_dir_for_dataset(args.data_root, args.dataset, args.graph_subdir)
    validate_data_dir(data_dir)
    seeds = parse_seeds(args.seeds)
    use_cuda = cuda_enabled(args.cuda)
    selected = experiment_filter(args)
    batch_dir = resolve_batch_dir(args).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    write_v9_manifest(args, batch_dir, data_dir, use_cuda, seeds)

    for experiment_name in KGE_EXPERIMENTS:
        for seed in seeds:
            train_kge(args, batch_dir, data_dir, data_dir, experiment_name, seed, use_cuda, selected)

    run_traditional_baselines(args, batch_dir, data_dir, selected)
    collect_metrics(batch_dir)
    print(f"run_dir -> {batch_dir}")
    print(f"summary -> {batch_dir / 'summaries' / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
