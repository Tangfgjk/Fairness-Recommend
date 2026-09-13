"""Run reproducible post-processing fairness rerank experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from experiment_utils import write_json


DEFAULT_METHODS = (
    "item_only",
    "kc_only",
    "item_kc_rerank",
    "stronger_item",
    "stronger_kc",
    "quota_ratio_10pct",
    "quota_ratio_15pct",
    "quota_ratio_hybrid",
)


@dataclass(frozen=True)
class RerankPreset:
    name: str
    rerank_method: str
    output_suffix: str
    eval_dir: str
    model_name: str
    options: Dict[str, str | int | float] = field(default_factory=dict)


PRESETS: Dict[str, RerankPreset] = {
    "item_only": RerankPreset(
        name="item_only",
        rerank_method="item",
        output_suffix="item_only",
        eval_dir="eval_item_only",
        model_name="2CKG4ER_item_only",
        options={
            "lambda-item": 0.8,
            "lambda-kc": 0.0,
            "beta-item": 0.3,
            "beta-kc": 0.0,
        },
    ),
    "kc_only": RerankPreset(
        name="kc_only",
        rerank_method="kc",
        output_suffix="kc_rerank",
        eval_dir="eval_kc_rerank",
        model_name="2CKG4ER_kc_rerank",
        options={
            "lambda-item": 0.0,
            "lambda-kc": 0.3,
            "beta-item": 0.0,
            "beta-kc": 0.1,
        },
    ),
    "item_kc_rerank": RerankPreset(
        name="item_kc_rerank",
        rerank_method="item_kc",
        output_suffix="item_kc_rerank",
        eval_dir="eval_item_kc_rerank",
        model_name="2CKG4ER_item_kc_rerank",
        options={
            "lambda-item": 0.3,
            "lambda-kc": 0.3,
            "beta-item": 0.1,
            "beta-kc": 0.1,
        },
    ),
    "stronger_item": RerankPreset(
        name="stronger_item",
        rerank_method="item_kc",
        output_suffix="item_kc_stronger_item",
        eval_dir="eval_item_kc_stronger_item",
        model_name="2CKG4ER_item_kc_stronger_item",
        options={
            "lambda-item": 0.8,
            "lambda-kc": 0.3,
            "beta-item": 0.3,
            "beta-kc": 0.1,
        },
    ),
    "stronger_kc": RerankPreset(
        name="stronger_kc",
        rerank_method="item_kc",
        output_suffix="stronger_kc",
        eval_dir="eval_stronger_kc",
        model_name="2CKG4ER_stronger_kc",
        options={
            "lambda-item": 0.3,
            "lambda-kc": 0.8,
            "beta-item": 0.1,
            "beta-kc": 0.3,
        },
    ),
    "quota_mid": RerankPreset(
        name="quota_mid",
        rerank_method="quota_strict",
        output_suffix="quota_mid",
        eval_dir="eval_quota_mid",
        model_name="2CKG4ER_quota_mid",
        options={
            "quota-top-k": 10,
            "min-long-tail-items": 1,
            "min-kc-coverage": 10,
        },
    ),
    "quota_ratio_10pct": RerankPreset(
        name="quota_ratio_10pct",
        rerank_method="quota_ratio",
        output_suffix="quota_ratio_10pct",
        eval_dir="eval_quota_ratio_10pct",
        model_name="2CKG4ER_quota_ratio_10pct",
        options={
            "quota-prefixes": "10,20,50,100",
            "long-tail-item-ratio": 0.1,
            "kc-coverage-ratio": 0.2,
            "lambda-item": 0.3,
            "lambda-kc": 0.3,
            "beta-item": 0.1,
            "beta-kc": 0.1,
        },
    ),
    "quota_ratio_15pct": RerankPreset(
        name="quota_ratio_15pct",
        rerank_method="quota_ratio",
        output_suffix="quota_ratio_15pct",
        eval_dir="eval_quota_ratio_15pct",
        model_name="2CKG4ER_quota_ratio_15pct",
        options={
            "quota-prefixes": "10,20,50,100",
            "long-tail-item-ratio": 0.15,
            "kc-coverage-ratio": 0.25,
            "lambda-item": 0.3,
            "lambda-kc": 0.3,
            "beta-item": 0.1,
            "beta-kc": 0.1,
        },
    ),
    "quota_ratio_hybrid": RerankPreset(
        name="quota_ratio_hybrid",
        rerank_method="quota_ratio_hybrid",
        output_suffix="quota_ratio_hybrid",
        eval_dir="eval_quota_ratio_hybrid",
        model_name="2CKG4ER_quota_ratio_hybrid",
        options={
            "quota-prefixes": "10,20,50,100",
            "long-tail-item-ratio": 0.1,
            "kc-coverage-ratio": 0.2,
            "lambda-item": 0.8,
            "lambda-kc": 0.3,
            "beta-item": 0.3,
            "beta-kc": 0.1,
        },
    ),
}


def parse_method_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        methods = [item.strip() for item in value.split(",") if item.strip()]
    else:
        methods = [str(item).strip() for item in value if str(item).strip()]
    unknown = sorted(set(methods) - set(PRESETS))
    if unknown:
        raise ValueError(f"Unknown methods: {','.join(unknown)}. Valid methods: {','.join(sorted(PRESETS))}")
    return methods


def option_args(options: Dict[str, str | int | float]) -> List[str]:
    args: List[str] = []
    for name, value in options.items():
        args.extend([f"--{name}", str(value)])
    return args


def script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def output_file_for(run_dir: Path, model_prefix: str, preset: RerankPreset) -> Path:
    return run_dir / f"{model_prefix}_uid_ex_scores_{preset.output_suffix}.pkl"


def eval_ready(eval_dir: Path) -> bool:
    return (eval_dir / "metrics.json").exists() and (eval_dir / "fairness_metrics.json").exists()


def build_rerank_command(
    python_executable: str,
    data_dir: Path,
    base_scores_file: Path,
    output_file: Path,
    preset: RerankPreset,
    top_k: int,
    candidate_multiplier: float,
    min_candidates: int,
    popularity_source: str,
    popularity_aggregation: str,
) -> List[str]:
    return [
        python_executable,
        str(script_path("fairness_rerank.py")),
        "--data-dir",
        str(data_dir),
        "--scores-file",
        str(base_scores_file),
        "--output-file",
        str(output_file),
        "--method",
        preset.rerank_method,
        "--top-k",
        str(top_k),
        "--candidate-multiplier",
        str(candidate_multiplier),
        "--min-candidates",
        str(min_candidates),
        "--popularity-source",
        popularity_source,
        "--popularity-aggregation",
        popularity_aggregation,
        *option_args(preset.options),
    ]


def build_eval_command(
    python_executable: str,
    data_dir: Path,
    scores_file: Path,
    eval_dir: Path,
    dataset_name: str,
    model_name: str,
    seed: int | None,
    top_ks: str,
    popularity_source: str,
    popularity_aggregation: str,
) -> List[str]:
    command = [
        python_executable,
        str(script_path("evaluate_recommendations.py")),
        "--data-dir",
        str(data_dir),
        "--scores-file",
        str(scores_file),
        "--output-dir",
        str(eval_dir),
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--top-ks",
        top_ks,
        "--fairness-popularity-source",
        popularity_source,
        "--fairness-popularity-aggregation",
        popularity_aggregation,
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def build_compare_command(
    python_executable: str,
    run_dir: Path,
    dataset_name: str,
    seed: int | None,
    compare_ks: str,
    methods: Sequence[str],
) -> List[str]:
    command = [
        python_executable,
        str(script_path("compare_fairness_results.py")),
        "--run-dir",
        str(run_dir),
        "--dataset-name",
        dataset_name,
        "--ks",
        compare_ks,
        "--methods",
        ",".join(["baseline", *methods]),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: Sequence[str], dry_run: bool = False, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    print(command_text(command))
    if dry_run:
        return
    runner(list(command), check=True)


def run_pipeline(args: argparse.Namespace, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Dict[str, object]:
    run_dir = args.run_dir
    data_dir = args.data_dir
    base_scores_file = args.scores_file or run_dir / f"{args.model_prefix}_uid_ex_scores.pkl"
    methods = parse_method_list(args.methods)
    manifest: Dict[str, object] = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "base_scores_file": str(base_scores_file),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "popularity_source": args.popularity_source,
        "popularity_aggregation": args.popularity_aggregation,
        "methods": [],
    }

    if not base_scores_file.exists() and not args.dry_run:
        raise FileNotFoundError(f"Base scores file not found: {base_scores_file}")

    for method in methods:
        preset = PRESETS[method]
        output_file = output_file_for(run_dir, args.model_prefix, preset)
        eval_dir = run_dir / preset.eval_dir
        method_record: Dict[str, object] = {
            "method": method,
            "rerank_method": preset.rerank_method,
            "output_file": str(output_file),
            "eval_dir": str(eval_dir),
            "model_name": preset.model_name,
            "options": dict(preset.options),
        }

        rerank_command = build_rerank_command(
            python_executable=args.python_executable,
            data_dir=data_dir,
            base_scores_file=base_scores_file,
            output_file=output_file,
            preset=preset,
            top_k=args.top_k,
            candidate_multiplier=args.candidate_multiplier,
            min_candidates=args.min_candidates,
            popularity_source=args.popularity_source,
            popularity_aggregation=args.popularity_aggregation,
        )
        if output_file.exists() and not args.force_rerank:
            print(f"[skip rerank] {method}: {output_file}")
            method_record["rerank_status"] = "skipped_existing"
        else:
            run_command(rerank_command, dry_run=args.dry_run, runner=runner)
            method_record["rerank_status"] = "dry_run" if args.dry_run else "completed"

        eval_command = build_eval_command(
            python_executable=args.python_executable,
            data_dir=data_dir,
            scores_file=output_file,
            eval_dir=eval_dir,
            dataset_name=args.dataset_name,
            model_name=preset.model_name,
            seed=args.seed,
            top_ks=args.top_ks,
            popularity_source=args.popularity_source,
            popularity_aggregation=args.popularity_aggregation,
        )
        if eval_ready(eval_dir) and not args.force_eval:
            print(f"[skip eval] {method}: {eval_dir}")
            method_record["eval_status"] = "skipped_existing"
        else:
            if not output_file.exists() and not args.dry_run:
                raise FileNotFoundError(f"Reranked scores file not found for evaluation: {output_file}")
            run_command(eval_command, dry_run=args.dry_run, runner=runner)
            method_record["eval_status"] = "dry_run" if args.dry_run else "completed"

        manifest["methods"].append(method_record)

    if not args.no_compare:
        compare_command = build_compare_command(
            python_executable=args.python_executable,
            run_dir=run_dir,
            dataset_name=args.dataset_name,
            seed=args.seed,
            compare_ks=args.compare_ks,
            methods=methods,
        )
        run_command(compare_command, dry_run=args.dry_run, runner=runner)
        manifest["comparison_status"] = "dry_run" if args.dry_run else "completed"
        manifest["comparison_dir"] = str(run_dir / "comparison")
    else:
        manifest["comparison_status"] = "skipped"

    manifest_path = args.manifest_file or run_dir / "fairness_rerank_pipeline.json"
    if not args.dry_run:
        write_json(manifest, manifest_path)
        print(f"pipeline manifest written to {manifest_path}")
    else:
        print("[dry-run] pipeline manifest was not written")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness rerank, evaluation, and comparison steps for 2CKG4ER outputs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--model-prefix", default="2CKG4ER")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-ks", default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    parser.add_argument("--compare-ks", default="10,20,50,100")
    parser.add_argument("--candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--min-candidates", type=int, default=150)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--manifest-file", type=Path, default=None)
    parser.add_argument("--force-rerank", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
