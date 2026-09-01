"""Run post-processing fairness parameter sensitivity experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from experiment_utils import write_json
from run_fairness_rerank_experiments import (
    RerankPreset,
    build_eval_command,
    build_rerank_command,
    command_text,
    eval_ready,
    run_command,
)


DEFAULT_SUITES = ("lt", "kc", "candidate", "soft_item", "soft_kc")
BASE_OPTIONS: Dict[str, str | int | float] = {
    "quota-prefixes": "10,20,50,100",
    "long-tail-item-ratio": 0.1,
    "kc-coverage-ratio": 0.2,
    "lambda-item": 0.8,
    "lambda-kc": 0.3,
    "beta-item": 0.3,
    "beta-kc": 0.1,
}
SOFT_BASE_OPTIONS: Dict[str, str | int | float] = {
    "lambda-item": 0.3,
    "lambda-kc": 0.3,
    "beta-item": 0.1,
    "beta-kc": 0.1,
}


@dataclass(frozen=True)
class SensitivityExperiment:
    name: str
    suite: str
    preset: RerankPreset
    candidate_multiplier: float = 1.5
    min_candidates: int = 150
    notes: Dict[str, str | int | float] = field(default_factory=dict)


def ratio_tag(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def multiplier_tag(value: float) -> str:
    return f"{int(round(float(value) * 10)):02d}"


def parse_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_float_list(value: str | Sequence[float]) -> List[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_weight_pairs(value: str | Sequence[str]) -> List[tuple[float, float]]:
    items = parse_list(value)
    pairs: List[tuple[float, float]] = []
    for item in items:
        if ":" in item:
            first, second = item.split(":", maxsplit=1)
        elif "/" in item:
            first, second = item.split("/", maxsplit=1)
        else:
            raise ValueError(f"Weight pair must use lambda:beta format: {item}")
        pairs.append((float(first.strip()), float(second.strip())))
    return pairs


def weight_tag(lambda_value: float, beta_value: float) -> str:
    return f"l{int(round(lambda_value * 100)):03d}_b{int(round(beta_value * 100)):03d}"


def with_option(options: Dict[str, str | int | float], name: str, value: str | int | float) -> Dict[str, str | int | float]:
    merged = dict(options)
    merged[name] = value
    return merged


def make_preset(
    name: str,
    output_suffix: str,
    eval_dir: str,
    options: Dict[str, str | int | float],
    rerank_method: str = "quota_ratio_hybrid",
) -> RerankPreset:
    return RerankPreset(
        name=name,
        rerank_method=rerank_method,
        output_suffix=output_suffix,
        eval_dir=eval_dir,
        model_name=f"2CKG4ER_{name}",
        options=options,
    )


def generate_experiments(
    suites: Sequence[str],
    long_tail_ratios: Sequence[float],
    kc_coverage_ratios: Sequence[float],
    candidate_multipliers: Sequence[float],
    soft_item_weights: Sequence[tuple[float, float]],
    soft_kc_weights: Sequence[tuple[float, float]],
    min_candidates: int,
) -> List[SensitivityExperiment]:
    unknown = sorted(set(suites) - set(DEFAULT_SUITES))
    if unknown:
        raise ValueError(f"Unknown sensitivity suites: {','.join(unknown)}. Valid suites: {','.join(DEFAULT_SUITES)}")

    experiments: List[SensitivityExperiment] = []
    if "lt" in suites:
        for ratio in long_tail_ratios:
            name = f"quota_ratio_hybrid_lt{ratio_tag(ratio)}"
            experiments.append(
                SensitivityExperiment(
                    name=name,
                    suite="lt",
                    preset=make_preset(
                        name=name,
                        output_suffix=name,
                        eval_dir=f"eval_{name}",
                        options=with_option(BASE_OPTIONS, "long-tail-item-ratio", ratio),
                    ),
                    min_candidates=min_candidates,
                    notes={"long_tail_item_ratio": ratio},
                )
            )

    if "kc" in suites:
        for ratio in kc_coverage_ratios:
            name = f"quota_ratio_hybrid_kc{ratio_tag(ratio)}"
            experiments.append(
                SensitivityExperiment(
                    name=name,
                    suite="kc",
                    preset=make_preset(
                        name=name,
                        output_suffix=name,
                        eval_dir=f"eval_{name}",
                        options=with_option(BASE_OPTIONS, "kc-coverage-ratio", ratio),
                    ),
                    min_candidates=min_candidates,
                    notes={"kc_coverage_ratio": ratio},
                )
            )

    if "candidate" in suites:
        for multiplier in candidate_multipliers:
            name = f"quota_ratio_hybrid_cand{multiplier_tag(multiplier)}"
            experiments.append(
                SensitivityExperiment(
                    name=name,
                    suite="candidate",
                    preset=make_preset(
                        name=name,
                        output_suffix=name,
                        eval_dir=f"eval_{name}",
                        options=dict(BASE_OPTIONS),
                    ),
                    candidate_multiplier=multiplier,
                    min_candidates=min_candidates,
                    notes={"candidate_multiplier": multiplier},
                )
            )

    if "soft_item" in suites:
        for lambda_item, beta_item in soft_item_weights:
            name = f"soft_item_{weight_tag(lambda_item, beta_item)}"
            options = dict(SOFT_BASE_OPTIONS)
            options["lambda-item"] = lambda_item
            options["beta-item"] = beta_item
            experiments.append(
                SensitivityExperiment(
                    name=name,
                    suite="soft_item",
                    preset=make_preset(
                        name=name,
                        output_suffix=name,
                        eval_dir=f"eval_{name}",
                        options=options,
                        rerank_method="item_kc",
                    ),
                    min_candidates=min_candidates,
                    notes={"lambda_item": lambda_item, "beta_item": beta_item},
                )
            )

    if "soft_kc" in suites:
        for lambda_kc, beta_kc in soft_kc_weights:
            name = f"soft_kc_{weight_tag(lambda_kc, beta_kc)}"
            options = dict(SOFT_BASE_OPTIONS)
            options["lambda-kc"] = lambda_kc
            options["beta-kc"] = beta_kc
            experiments.append(
                SensitivityExperiment(
                    name=name,
                    suite="soft_kc",
                    preset=make_preset(
                        name=name,
                        output_suffix=name,
                        eval_dir=f"eval_{name}",
                        options=options,
                        rerank_method="item_kc",
                    ),
                    min_candidates=min_candidates,
                    notes={"lambda_kc": lambda_kc, "beta_kc": beta_kc},
                )
            )
    return experiments


def output_file_for(run_dir: Path, model_prefix: str, experiment: SensitivityExperiment) -> Path:
    return run_dir / f"{model_prefix}_uid_ex_scores_{experiment.preset.output_suffix}.pkl"


def build_compare_command(
    python_executable: str,
    run_dir: Path,
    output_dir: Path,
    dataset_name: str,
    seed: int | None,
    compare_ks: str,
    methods: Sequence[str],
) -> List[str]:
    command = [
        python_executable,
        str(Path(__file__).resolve().parent / "compare_fairness_results.py"),
        "--run-dir",
        str(run_dir),
        "--dataset-name",
        dataset_name,
        "--ks",
        compare_ks,
        "--methods",
        ",".join(["baseline", *methods]),
        "--output-dir",
        str(output_dir),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def run_pipeline(args: argparse.Namespace, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Dict[str, object]:
    suites = parse_list(args.suites)
    experiments = generate_experiments(
        suites=suites,
        long_tail_ratios=parse_float_list(args.long_tail_ratios),
        kc_coverage_ratios=parse_float_list(args.kc_coverage_ratios),
        candidate_multipliers=parse_float_list(args.candidate_multipliers),
        soft_item_weights=parse_weight_pairs(args.soft_item_weights),
        soft_kc_weights=parse_weight_pairs(args.soft_kc_weights),
        min_candidates=args.min_candidates,
    )
    base_scores_file = args.scores_file or args.run_dir / f"{args.model_prefix}_uid_ex_scores.pkl"
    output_dir = args.output_dir or args.run_dir / "comparison_sensitivity"
    manifest: Dict[str, object] = {
        "run_dir": str(args.run_dir),
        "data_dir": str(args.data_dir),
        "base_scores_file": str(base_scores_file),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "suites": suites,
        "experiments": [],
        "comparison_dir": str(output_dir),
    }

    if not base_scores_file.exists() and not args.dry_run:
        raise FileNotFoundError(f"Base scores file not found: {base_scores_file}")

    for experiment in experiments:
        preset = experiment.preset
        scores_file = output_file_for(args.run_dir, args.model_prefix, experiment)
        eval_dir = args.run_dir / preset.eval_dir
        record: Dict[str, object] = {
            "name": experiment.name,
            "suite": experiment.suite,
            "scores_file": str(scores_file),
            "eval_dir": str(eval_dir),
            "model_name": preset.model_name,
            "candidate_multiplier": experiment.candidate_multiplier,
            "min_candidates": experiment.min_candidates,
            "options": dict(preset.options),
            "notes": dict(experiment.notes),
        }

        rerank_command = build_rerank_command(
            python_executable=args.python_executable,
            data_dir=args.data_dir,
            base_scores_file=base_scores_file,
            output_file=scores_file,
            preset=preset,
            top_k=args.top_k,
            candidate_multiplier=experiment.candidate_multiplier,
            min_candidates=experiment.min_candidates,
        )
        if scores_file.exists() and not args.force_rerank:
            print(f"[skip rerank] {experiment.name}: {scores_file}")
            record["rerank_status"] = "skipped_existing"
        else:
            run_command(rerank_command, dry_run=args.dry_run, runner=runner)
            record["rerank_status"] = "dry_run" if args.dry_run else "completed"

        eval_command = build_eval_command(
            python_executable=args.python_executable,
            data_dir=args.data_dir,
            scores_file=scores_file,
            eval_dir=eval_dir,
            dataset_name=args.dataset_name,
            model_name=preset.model_name,
            seed=args.seed,
            top_ks=args.top_ks,
        )
        if eval_ready(eval_dir) and not args.force_eval:
            print(f"[skip eval] {experiment.name}: {eval_dir}")
            record["eval_status"] = "skipped_existing"
        else:
            if not scores_file.exists() and not args.dry_run:
                raise FileNotFoundError(f"Reranked scores file not found for evaluation: {scores_file}")
            run_command(eval_command, dry_run=args.dry_run, runner=runner)
            record["eval_status"] = "dry_run" if args.dry_run else "completed"

        manifest["experiments"].append(record)

    if not args.no_compare:
        compare_command = build_compare_command(
            python_executable=args.python_executable,
            run_dir=args.run_dir,
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            seed=args.seed,
            compare_ks=args.compare_ks,
            methods=[experiment.name for experiment in experiments],
        )
        print(command_text(compare_command))
        if not args.dry_run:
            runner(compare_command, check=True)
            manifest["comparison_status"] = "completed"
        else:
            manifest["comparison_status"] = "dry_run"
    else:
        manifest["comparison_status"] = "skipped"

    manifest_path = args.manifest_file or args.run_dir / "fairness_sensitivity_pipeline.json"
    if not args.dry_run:
        write_json(manifest, manifest_path)
        print(f"sensitivity manifest written to {manifest_path}")
    else:
        print("[dry-run] sensitivity manifest was not written")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-processing fairness parameter sensitivity experiments.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--model-prefix", default="2CKG4ER")
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES), help="Comma-separated: lt,kc,candidate,soft_item,soft_kc")
    parser.add_argument("--long-tail-ratios", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--kc-coverage-ratios", default="0.10,0.20,0.25,0.30")
    parser.add_argument("--candidate-multipliers", default="1.0,1.5,2.0,3.0")
    parser.add_argument("--soft-item-weights", default="0.3:0.1,0.5:0.2,0.8:0.3")
    parser.add_argument("--soft-kc-weights", default="0.3:0.1,0.5:0.2,0.8:0.3")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-ks", default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    parser.add_argument("--compare-ks", default="10,20,50,100")
    parser.add_argument("--min-candidates", type=int, default=150)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=None)
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
