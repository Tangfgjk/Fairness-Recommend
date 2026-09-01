"""Run stage-2 alpha/gamma sensitivity experiments for fair 2CKG4ER models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from experiment_utils import load_json, write_json
from run_fairness_rerank_experiments import command_text


DEFAULT_METHODS = "fairreg_item,fairreg_kc,fairreg_item_kc,fairreg_item_kc_hybrid_rerank"
DEFAULT_TOP_KS = "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"


@dataclass(frozen=True)
class FairModelSensitivityExperiment:
    run_id: str
    alpha: float
    gamma: float


def parse_float_list(value: str | Sequence[float]) -> List[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def value_tag(value: float) -> str:
    scaled = int(round(float(value) * 100))
    if scaled % 10 == 0:
        return f"{scaled // 10:02d}"
    return f"{scaled:03d}"


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def run_dir_for(args: argparse.Namespace, run_id: str) -> Path:
    return args.runs_root / args.dataset / run_id


def run_completed(run_dir: Path) -> bool:
    status = load_json(run_dir / "run_status.json", default={}) or {}
    return status.get("status") == "completed"


def generate_experiments(dataset: str, run_prefix: str, alphas: Sequence[float], gammas: Sequence[float], seed_tag: str) -> List[FairModelSensitivityExperiment]:
    experiments: List[FairModelSensitivityExperiment] = []
    for gamma in gammas:
        for alpha in alphas:
            run_id = f"{run_prefix}_alpha{value_tag(alpha)}_gamma{value_tag(gamma)}_{seed_tag}"
            if not run_id.startswith(dataset):
                run_id = f"{dataset}_{run_id}"
            experiments.append(FairModelSensitivityExperiment(run_id=run_id, alpha=float(alpha), gamma=float(gamma)))
    return experiments


def build_command(args: argparse.Namespace, experiment: FairModelSensitivityExperiment) -> List[str]:
    return [
        args.python_executable,
        str(code_dir() / "run_fair_model_experiments.py"),
        "--dataset",
        args.dataset,
        "--run-id",
        experiment.run_id,
        "--methods",
        args.methods,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--bs",
        str(args.bs),
        "--learning-rate",
        str(args.learning_rate),
        "--negative-ratio",
        str(args.negative_ratio),
        "--cuda",
        args.cuda,
        "--data-root",
        str(args.data_root),
        "--runs-root",
        str(args.runs_root),
        "--source-graph-subdir",
        args.source_graph_subdir,
        "--fair-graph-subdir",
        args.fair_graph_subdir,
        "--top-ks",
        args.top_ks,
        "--alpha-item",
        str(experiment.alpha),
        "--alpha-kc",
        str(experiment.alpha),
        "--fairness-target-gamma",
        str(experiment.gamma),
        "--fairness-loss-scale",
        str(args.fairness_loss_scale),
        "--fairness-candidate-size",
        str(args.fairness_candidate_size),
        "--fairness-candidate-mode",
        args.fairness_candidate_mode,
        "--fairness-temperature",
        str(args.fairness_temperature),
    ]


def run_pipeline(args: argparse.Namespace, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Dict[str, object]:
    seed_tag = "seed" + "_".join(item.strip() for item in args.seeds.split(",") if item.strip())
    experiments = generate_experiments(
        dataset=args.dataset,
        run_prefix=args.run_prefix,
        alphas=parse_float_list(args.alphas),
        gammas=parse_float_list(args.gammas),
        seed_tag=seed_tag,
    )
    manifest: Dict[str, object] = {
        "dataset": args.dataset,
        "methods": args.methods,
        "seeds": args.seeds,
        "alphas": parse_float_list(args.alphas),
        "gammas": parse_float_list(args.gammas),
        "runs_root": str(args.runs_root),
        "experiments": [],
    }

    for experiment in experiments:
        target_run_dir = run_dir_for(args, experiment.run_id)
        record: Dict[str, object] = {
            "run_id": experiment.run_id,
            "run_dir": str(target_run_dir),
            "alpha_item": experiment.alpha,
            "alpha_kc": experiment.alpha,
            "target_gamma": experiment.gamma,
        }
        command = build_command(args, experiment)
        record["command"] = command_text(command)

        if run_completed(target_run_dir) and not args.force:
            print(f"[skip completed] {experiment.run_id}")
            record["status"] = "skipped_completed"
        else:
            print(command_text(command))
            if args.dry_run:
                record["status"] = "dry_run"
            else:
                runner(command, check=True)
                record["status"] = "completed"
        manifest["experiments"].append(record)

    manifest_path = args.manifest_file or args.runs_root / args.dataset / f"{args.run_prefix}_sensitivity_manifest.json"
    if not args.dry_run:
        write_json(manifest, manifest_path)
        print(f"fair model sensitivity manifest written to {manifest_path}")
    else:
        print("[dry-run] fair model sensitivity manifest was not written")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 alpha/gamma fair model sensitivity experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-prefix", default="stage2")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--gammas", default="0.5,0.25")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--source-graph-subdir", default="er_graph")
    parser.add_argument("--fair-graph-subdir", default="fair_kg_graph")
    parser.add_argument("--top-ks", default=DEFAULT_TOP_KS)
    parser.add_argument("--fairness-loss-scale", type=float, default=1.0)
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-candidate-mode", choices=["random", "popular", "top_score", "mixed"], default="random")
    parser.add_argument("--fairness-temperature", type=float, default=1.0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
