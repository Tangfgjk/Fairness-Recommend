"""Run stage-4 fairness loss scale and candidate-mode sensitivity experiments."""

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
DEFAULT_EXPERIMENTS = "random:1,random:10,random:50,random:100,mixed:50,mixed:100"
VALID_CANDIDATE_MODES = {"random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"}


@dataclass(frozen=True)
class FairLossSensitivityExperiment:
    run_id: str
    candidate_mode: str
    loss_scale: float


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def scale_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value)).zfill(3)
    return str(value).replace(".", "p")


def fairness_config_tag(args: argparse.Namespace) -> str:
    """Return a stable path token for fairness settings not already in the sweep axes."""
    def token(value: object) -> str:
        return str(value).replace(".", "p").replace("/", "-")

    return "_".join(
        [
            f"proxy-{args.fairness_exposure_proxy}",
            f"k{args.fairness_surrogate_k}",
            f"dist-{args.fairness_distance}",
            f"cand-{args.fairness_candidate_size}",
            f"temp-{token(args.fairness_temperature)}",
            f"mix-{token(args.fairness_top_score_ratio)}-{token(args.fairness_popular_ratio)}",
            f"pop-{args.fairness_popularity_source}-{args.fairness_popularity_aggregation}",
        ]
    )


def parse_experiments(value: str | Sequence[str]) -> List[tuple[str, float]]:
    raw_items = value.split(",") if isinstance(value, str) else list(value)
    experiments: List[tuple[str, float]] = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Each experiment must use candidate_mode:loss_scale, e.g. mixed:50")
        mode, scale = [part.strip() for part in item.split(":", 1)]
        if mode not in VALID_CANDIDATE_MODES:
            raise ValueError(f"candidate_mode must be one of: {sorted(VALID_CANDIDATE_MODES)}")
        scale_value = float(scale)
        if scale_value <= 0:
            raise ValueError("loss_scale must be positive")
        experiments.append((mode, scale_value))
    if not experiments:
        raise ValueError("At least one experiment is required")
    return experiments


def generate_experiments(
    dataset: str,
    run_prefix: str,
    experiment_specs: Sequence[tuple[str, float]],
    seed_tag: str,
) -> List[FairLossSensitivityExperiment]:
    experiments: List[FairLossSensitivityExperiment] = []
    for candidate_mode, loss_scale in experiment_specs:
        run_id = f"{run_prefix}_{candidate_mode}_scale{scale_tag(loss_scale)}_{seed_tag}"
        if not run_id.startswith(dataset):
            run_id = f"{dataset}_{run_id}"
        experiments.append(
            FairLossSensitivityExperiment(
                run_id=run_id,
                candidate_mode=candidate_mode,
                loss_scale=float(loss_scale),
            )
        )
    return experiments


def run_dir_for(args: argparse.Namespace, run_id: str) -> Path:
    return args.runs_root / args.dataset / run_id


def run_completed(run_dir: Path) -> bool:
    status = load_json(run_dir / "run_status.json", default={}) or {}
    return status.get("status") == "completed"


def build_command(args: argparse.Namespace, experiment: FairLossSensitivityExperiment) -> List[str]:
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
        str(args.alpha),
        "--alpha-kc",
        str(args.alpha),
        "--fairness-target-gamma",
        str(args.gamma),
        "--fairness-loss-scale",
        str(experiment.loss_scale),
        "--fairness-candidate-size",
        str(args.fairness_candidate_size),
        "--fairness-candidate-mode",
        experiment.candidate_mode,
        "--fairness-temperature",
        str(args.fairness_temperature),
        "--fairness-exposure-proxy",
        args.fairness_exposure_proxy,
        "--fairness-surrogate-k",
        str(args.fairness_surrogate_k),
        "--fairness-distance",
        args.fairness_distance,
        "--fairness-top-score-ratio",
        str(args.fairness_top_score_ratio),
        "--fairness-popular-ratio",
        str(args.fairness_popular_ratio),
        "--fairness-popularity-source",
        args.fairness_popularity_source,
        "--fairness-popularity-aggregation",
        args.fairness_popularity_aggregation,
    ]


def run_pipeline(args: argparse.Namespace, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Dict[str, object]:
    seed_tag = "seed" + "_".join(item.strip() for item in args.seeds.split(",") if item.strip())
    specs = parse_experiments(args.experiments)
    experiments = generate_experiments(args.dataset, args.run_prefix, specs, seed_tag)
    manifest: Dict[str, object] = {
        "dataset": args.dataset,
        "methods": args.methods,
        "seeds": args.seeds,
        "alpha_item": float(args.alpha),
        "alpha_kc": float(args.alpha),
        "target_gamma": float(args.gamma),
        "runs_root": str(args.runs_root),
        "fairness_exposure_proxy": args.fairness_exposure_proxy,
        "fairness_distance": args.fairness_distance,
        "fairness_popularity_source": args.fairness_popularity_source,
        "fairness_candidate_size": args.fairness_candidate_size,
        "fairness_temperature": args.fairness_temperature,
        "fairness_surrogate_k": args.fairness_surrogate_k,
        "fairness_top_score_ratio": args.fairness_top_score_ratio,
        "fairness_popular_ratio": args.fairness_popular_ratio,
        "fairness_popularity_aggregation": args.fairness_popularity_aggregation,
        "experiments": [],
    }
    config_tag = fairness_config_tag(args)

    for experiment in experiments:
        run_id = f"{experiment.run_id}_{config_tag}"
        target_run_dir = run_dir_for(args, run_id)
        record: Dict[str, object] = {
            "run_id": run_id,
            "run_dir": str(target_run_dir),
            "candidate_mode": experiment.candidate_mode,
            "fairness_loss_scale": experiment.loss_scale,
        }
        command = build_command(args, experiment)
        command[command.index("--run-id") + 1] = run_id
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

    manifest_path = args.manifest_file or args.runs_root / args.dataset / f"{args.run_prefix}_loss_sensitivity_manifest.json"
    if not args.dry_run:
        write_json(manifest, manifest_path)
        print(f"fair loss sensitivity manifest written to {manifest_path}")
    else:
        print("[dry-run] fair loss sensitivity manifest was not written")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-4 fairness loss scale/candidate-mode experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-prefix", default="stage4_loss")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.25)
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
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-temperature", type=float, default=1.0)
    parser.add_argument("--fairness-exposure-proxy", choices=["softmax", "sigmoid_topk"], default="softmax")
    parser.add_argument("--fairness-surrogate-k", type=int, default=10)
    parser.add_argument("--fairness-distance", choices=["mse", "l1", "kl_target_model", "js"], default="mse")
    parser.add_argument("--fairness-top-score-ratio", type=float, default=0.5)
    parser.add_argument("--fairness-popular-ratio", type=float, default=0.25)
    parser.add_argument("--fairness-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="rec_triples")
    parser.add_argument("--fairness-popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
