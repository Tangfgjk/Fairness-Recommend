"""Composable Pre/In/Post fairness experiment runner for 2CKG4ER."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from fairness_preprocess_rec_graph import prepare_fair_preprocess_graph
from run_fair_model_experiments import FAIRNESS_METRICS, QUALITY_METRICS, fairness_regularizer_flags, rerank_command
from semantic_experiment_utils import (
    DEFAULT_TOP_KS,
    MODEL_NAME,
    MODEL_VERSION,
    code_dir,
    command_to_markdown,
    default_data_root,
    default_runs_root,
    graph_path_for_dataset,
    mark_stage,
    parse_csv_list,
    parse_seed_list,
    parse_top_ks,
    python_env_info,
    read_json,
    run_logged_command,
    stage_done,
    timestamp,
    write_json,
)


@dataclass(frozen=True)
class PipelineExperiment:
    name: str
    pre: str = "none"
    in_method: str = "none"
    post: str = "none"


PRESETS: Dict[str, PipelineExperiment] = {
    "baseline": PipelineExperiment("baseline"),
    "pre_baseline_rebuild": PipelineExperiment("pre_baseline_rebuild", pre="baseline_rebuild"),
    "pre_item": PipelineExperiment("pre_item", pre="item"),
    "pre_kc": PipelineExperiment("pre_kc", pre="kc"),
    "pre_item_kc": PipelineExperiment("pre_item_kc", pre="item_kc"),
    "in_post": PipelineExperiment("in_post", in_method="fairreg_item_kc", post="quota_ratio_hybrid"),
    "pre_in_post": PipelineExperiment("pre_in_post", pre="item_kc", in_method="fairreg_item_kc", post="quota_ratio_hybrid"),
}

GROUPS: Dict[str, List[str]] = {
    "pre_only": ["pre_baseline_rebuild", "pre_item", "pre_kc", "pre_item_kc"],
    "requested": ["pre_baseline_rebuild", "pre_item", "pre_kc", "pre_item_kc", "in_post", "pre_in_post"],
    "all": ["baseline", "pre_baseline_rebuild", "pre_item", "pre_kc", "pre_item_kc", "in_post", "pre_in_post"],
}


def parse_experiments(value: str | Sequence[str]) -> List[str]:
    requested = parse_csv_list(value)
    expanded: List[str] = []
    for item in requested:
        if item in GROUPS:
            expanded.extend(GROUPS[item])
        elif item in PRESETS:
            expanded.append(item)
        else:
            raise ValueError(f"Unknown experiment: {item}. Valid: {','.join(sorted([*PRESETS, *GROUPS]))}")
    deduped: List[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def pre_lambdas(args: argparse.Namespace, pre: str) -> tuple[float, float]:
    if pre == "none":
        return 0.0, 0.0
    if pre == "baseline_rebuild":
        return 0.0, 0.0
    if pre == "item":
        return float(args.pre_lambda_item), 0.0
    if pre == "kc":
        return 0.0, float(args.pre_lambda_kc)
    if pre == "item_kc":
        return float(args.pre_lambda_item), float(args.pre_lambda_kc)
    raise ValueError(f"Unsupported pre method: {pre}")


def pre_graph_subdir(args: argparse.Namespace, experiment: PipelineExperiment) -> str:
    lambda_item, lambda_kc = pre_lambdas(args, experiment.pre)
    li = str(lambda_item).replace(".", "p")
    lk = str(lambda_kc).replace(".", "p")
    return f"{args.pre_graph_root_subdir}/{experiment.name}_M{args.pre_candidate_pool_size}_li{li}_lk{lk}"


def train_command(args: argparse.Namespace, graph_path: Path, seed_dir: Path, experiment: PipelineExperiment, seed: int) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "run_semantic_conve.py"),
        "--data-path",
        str(graph_path),
        "--dataset-name",
        args.dataset,
        "--save-path",
        str(seed_dir),
        "--epochs",
        str(args.epochs),
        "--bs",
        str(args.bs),
        "--learning-rate",
        str(args.learning_rate),
        "--negative-ratio",
        str(args.negative_ratio),
        "--seed",
        str(seed),
        "--cuda",
        args.cuda,
        "--ablation",
        MODEL_NAME,
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.resume:
        command.append("--resume")
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if experiment.in_method == "fairreg_item_kc":
        command.extend(["--relation-loss-weights", "balanced"])
        command.extend(fairness_regularizer_flags(args, str(args.alpha_item), str(args.alpha_kc)))
    return command


def test_command(args: argparse.Namespace, graph_path: Path, seed_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "test_semantic_conve.py"),
        "--data-path",
        str(graph_path),
        "--dataset-name",
        args.dataset,
        "--save-path",
        str(seed_dir),
        "--cuda",
        args.cuda,
        "--ablation",
        MODEL_NAME,
        "--forgetting-score-weight",
        str(args.forgetting_score_weight),
        "--forgetting-exercise-batch-size",
        str(args.forgetting_exercise_batch_size),
    ]
    if args.max_test_users > 0:
        command.extend(["--max-test-users", str(args.max_test_users)])
    return command


def eval_command(args: argparse.Namespace, graph_path: Path, scores: Path, output_dir: Path, seed: int, method: str) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "evaluate_recommendations.py"),
        "--data-dir",
        str(graph_path),
        "--scores-file",
        str(scores),
        "--output-dir",
        str(output_dir),
        "--dataset-name",
        args.dataset,
        "--model-name",
        method,
        "--top-ks",
        ",".join(str(k) for k in args.top_ks),
        "--seed",
        str(seed),
    ]


def score_file(seed_dir: Path) -> Path:
    return seed_dir / "2CKG4ER_uid_ex_scores.pkl"


def run_stage(stage_name: str, command: List[str], cwd: Path, log_path: Path, status_path: Path, dry_run: bool) -> None:
    mark_stage(status_path, "running", command=command)
    rc = run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        mark_stage(status_path, "failed", command=command, exit_code=rc)
        raise RuntimeError(f"{stage_name} failed with exit code {rc}. See {log_path}")
    mark_stage(status_path, "completed", command=command, exit_code=rc)


def prepare_pre_graph(args: argparse.Namespace, source_graph: Path, experiment: PipelineExperiment) -> tuple[Path, Dict[str, Any] | None]:
    if experiment.pre == "none":
        return source_graph, None
    target = args.data_root / args.dataset / pre_graph_subdir(args, experiment)
    lambda_item, lambda_kc = pre_lambdas(args, experiment.pre)
    if args.dry_run:
        manifest = {"dry_run": True, "target_dir": str(target), "lambda_item": lambda_item, "lambda_kc": lambda_kc}
    else:
        manifest = prepare_fair_preprocess_graph(
            source_graph_dir=source_graph,
            output_graph_dir=target,
            candidate_pool_size=args.pre_candidate_pool_size,
            top_k_rec=args.pre_top_k_rec,
            lambda_item=lambda_item,
            lambda_kc=lambda_kc,
            popularity_source=args.pre_popularity_source,
            popularity_aggregation=args.pre_popularity_aggregation,
            head_ratio=args.head_ratio,
            long_tail_ratio=args.long_tail_ratio,
            epsilon=args.pre_epsilon,
            force=args.force_preprocess,
        )
        if experiment.pre == "baseline_rebuild" and not manifest.get("baseline_rebuild_exact_match"):
            raise RuntimeError(f"pre_baseline_rebuild did not exactly reproduce source rec edges: {target}")
    return target, manifest


def run_one(args: argparse.Namespace, experiment: PipelineExperiment, seed: int, graph_path: Path, run_dir: Path) -> Dict[str, Any]:
    seed_dir = run_dir / experiment.name / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    commands: List[Dict[str, Any]] = []

    command = train_command(args, graph_path, seed_dir, experiment, seed)
    commands.append({"experiment": experiment.name, "stage": "train", "seed": seed, "command": command})
    if not stage_done(seed_dir / "train_stage.json") or not (seed_dir / "best.pt").exists():
        run_stage("train", command, code_dir(), seed_dir / "runner_train.log", seed_dir / "train_stage.json", args.dry_run)

    command = test_command(args, graph_path, seed_dir)
    commands.append({"experiment": experiment.name, "stage": "test", "seed": seed, "command": command})
    if not stage_done(seed_dir / "test_stage.json") or not score_file(seed_dir).exists():
        run_stage("test", command, code_dir(), seed_dir / "runner_test.log", seed_dir / "test_stage.json", args.dry_run)

    scores_for_eval = score_file(seed_dir)
    eval_dir = seed_dir / "eval"
    if experiment.post == "quota_ratio_hybrid":
        reranked_scores = seed_dir / "2CKG4ER_uid_ex_scores_quota_ratio_hybrid.pkl"
        command = rerank_command(args, graph_path, score_file(seed_dir), reranked_scores)
        commands.append({"experiment": experiment.name, "stage": "post", "seed": seed, "command": command})
        if not stage_done(seed_dir / "post_stage.json") or not reranked_scores.exists():
            run_stage("post", command, code_dir(), seed_dir / "runner_post.log", seed_dir / "post_stage.json", args.dry_run)
        scores_for_eval = reranked_scores
        eval_dir = seed_dir / "eval_quota_ratio_hybrid"
    elif experiment.post != "none":
        raise ValueError(f"Unsupported post method: {experiment.post}")

    command = eval_command(args, graph_path, scores_for_eval, eval_dir, seed, experiment.name)
    commands.append({"experiment": experiment.name, "stage": "eval", "seed": seed, "command": command})
    if not stage_done(seed_dir / "eval_stage.json") or not (eval_dir / "metrics.json").exists():
        run_stage("eval", command, code_dir(), seed_dir / "runner_eval.log", seed_dir / "eval_stage.json", args.dry_run)

    return {
        "experiment": experiment.name,
        "pre": experiment.pre,
        "in": experiment.in_method,
        "post": experiment.post,
        "seed": seed,
        "graph_path": graph_path,
        "seed_dir": seed_dir,
        "eval_dir": eval_dir,
        "status": "completed",
        "commands": commands,
    }


def metric_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    try:
        return float(metrics[metric][str(top_k)]["mean"])
    except Exception:
        return None


def fairness_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    try:
        return float(metrics["top_k"][str(top_k)][metric])
    except Exception:
        return None


def collect_rows(statuses: Sequence[Dict[str, Any]], top_ks: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for status in statuses:
        eval_dir = Path(status["eval_dir"])
        metrics_path = eval_dir / "metrics.json"
        fairness_path = eval_dir / "fairness_metrics.json"
        if not metrics_path.exists() or not fairness_path.exists():
            continue
        metrics = read_json(metrics_path)
        fairness = read_json(fairness_path)
        for top_k in top_ks:
            row: Dict[str, Any] = {
                "experiment": status["experiment"],
                "pre": status["pre"],
                "in": status["in"],
                "post": status["post"],
                "seed": status["seed"],
                "K": int(top_k),
                "eval_dir": str(eval_dir),
            }
            for metric in QUALITY_METRICS:
                row[metric] = metric_value(metrics, metric, top_k)
            for metric in FAIRNESS_METRICS:
                row[metric] = fairness_value(fairness, metric, top_k)
            rows.append(row)
    return rows


def write_comparison(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    out_dir = run_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "pre", "in", "post", "seed", "K", "Ada", "NOV", *FAIRNESS_METRICS, "eval_dir"]
    with (out_dir / "fairness_pipeline_comparison.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for top_k in sorted({int(row["K"]) for row in rows}):
        with (out_dir / f"fairness_pipeline_K{top_k}.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([row for row in rows if int(row["K"]) == top_k])
    lines = ["# Fairness Pipeline Comparison", ""]
    for top_k in sorted({int(row["K"]) for row in rows}):
        lines.extend([f"## K={top_k}", ""])
        lines.append("| experiment | Pre | In | Post | Ada | NOV | ItemGini | KCGini | ItemCov | KCCov | LTItem | LTKC |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in [item for item in rows if int(item["K"]) == top_k]:
            values = {key: "" if value is None else f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()}
            lines.append(
                "| {experiment} | {pre} | {in} | {post} | {Ada} | {NOV} | {ItemExposureGini} | {KCExposureGini} | "
                "{ItemCoverage} | {KCCoverage} | {LongTailItemExposureShare} | {LongTailKCExposureShare} |".format(**values)
            )
        lines.append("")
    (out_dir / "fairness_pipeline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_commands(run_dir: Path, commands: Sequence[Dict[str, Any]]) -> None:
    lines = ["# Fairness Pipeline Commands", ""]
    for item in commands:
        lines.append(f"## {item['experiment']} / {item['stage']} / seed {item['seed']}")
        lines.append("")
        lines.append("```powershell")
        lines.append(command_to_markdown(item["command"]))
        lines.append("```")
        lines.append("")
    (run_dir / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composable Pre/In/Post fairness experiments for 2CKG4ER.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiments", default="requested")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--source-graph-subdir", default="er_graph")
    parser.add_argument("--pre-graph-root-subdir", default="fair_pre_rec_graph")
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--forgetting-score-weight", type=float, default=0.0)
    parser.add_argument("--forgetting-exercise-batch-size", type=int, default=256)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)
    parser.add_argument("--pre-candidate-pool-size", type=int, default=50)
    parser.add_argument("--pre-top-k-rec", type=int, default=None)
    parser.add_argument("--pre-lambda-item", type=float, default=0.1)
    parser.add_argument("--pre-lambda-kc", type=float, default=0.1)
    parser.add_argument("--pre-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--pre-popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--pre-epsilon", type=float, default=1e-8)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--long-tail-ratio", type=float, default=0.8)
    parser.add_argument("--alpha-item", type=float, default=1.0)
    parser.add_argument("--alpha-kc", type=float, default=1.0)
    parser.add_argument("--fairness-loss-scale", type=float, default=50.0)
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-candidate-mode", choices=["random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"], default="mixed_user")
    parser.add_argument("--fairness-temperature", type=float, default=1.0)
    parser.add_argument("--fairness-target-gamma", type=float, default=0.25)
    parser.add_argument("--fairness-exposure-proxy", choices=["softmax", "sigmoid_topk"], default="sigmoid_topk")
    parser.add_argument("--fairness-surrogate-k", type=int, default=10)
    parser.add_argument("--fairness-distance", choices=["mse", "l1", "kl_target_model", "js"], default="js")
    parser.add_argument("--fairness-top-score-ratio", type=float, default=0.5)
    parser.add_argument("--fairness-popular-ratio", type=float, default=0.25)
    parser.add_argument("--fairness-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--fairness-popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument("--rerank-candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--rerank-min-candidates", type=int, default=150)
    parser.add_argument("--rerank-quota-prefixes", default="10,20,50,100")
    parser.add_argument("--rerank-long-tail-ratio-target", type=float, default=0.10)
    parser.add_argument("--rerank-kc-coverage-ratio-target", type=float, default=0.20)
    parser.add_argument("--rerank-lambda-item", type=float, default=0.8)
    parser.add_argument("--rerank-lambda-kc", type=float, default=0.3)
    parser.add_argument("--rerank-beta-item", type=float, default=0.3)
    parser.add_argument("--rerank-beta-kc", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.experiments = parse_experiments(args.experiments)
    args.seeds = parse_seed_list(args.seeds)
    args.top_ks = parse_top_ks(args.top_ks)
    run_id = args.run_id or f"{args.dataset}_fairness_pipeline_{timestamp()}"
    run_dir = args.runs_root / args.dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_graph = graph_path_for_dataset(args.dataset, args.data_root, graph_subdir=args.source_graph_subdir)

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "run_id": run_id,
            "run_dir": run_dir,
            "source_graph": source_graph,
            "experiments": args.experiments,
            "seeds": args.seeds,
            "top_ks": args.top_ks,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "preprocessing": {
                "candidate_pool_size": args.pre_candidate_pool_size,
                "top_k_rec": args.pre_top_k_rec,
                "lambda_item": args.pre_lambda_item,
                "lambda_kc": args.pre_lambda_kc,
                "popularity_source": args.pre_popularity_source,
                "popularity_aggregation": args.pre_popularity_aggregation,
            },
            "in_processing": {
                "alpha_item": args.alpha_item,
                "alpha_kc": args.alpha_kc,
                "loss_scale": args.fairness_loss_scale,
                "candidate_size": args.fairness_candidate_size,
                "candidate_mode": args.fairness_candidate_mode,
                "target_gamma": args.fairness_target_gamma,
                "exposure_proxy": args.fairness_exposure_proxy,
                "surrogate_top_k": args.fairness_surrogate_k,
                "distance": args.fairness_distance,
            },
            "post_processing": {
                "method": "quota_ratio_hybrid",
                "top_k": args.rerank_top_k,
                "candidate_multiplier": args.rerank_candidate_multiplier,
                "min_candidates": args.rerank_min_candidates,
                "long_tail_ratio_target": args.rerank_long_tail_ratio_target,
                "kc_coverage_ratio_target": args.rerank_kc_coverage_ratio_target,
            },
            "env": python_env_info(),
        },
    )

    statuses: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    pre_manifests: Dict[str, Any] = {}
    try:
        graph_by_experiment: Dict[str, Path] = {}
        for name in args.experiments:
            experiment = PRESETS[name]
            graph_path, manifest = prepare_pre_graph(args, source_graph, experiment)
            graph_by_experiment[name] = graph_path
            if manifest is not None:
                pre_manifests[name] = manifest
        write_json(run_dir / "preprocess_manifests.json", pre_manifests)

        for seed in args.seeds:
            for name in args.experiments:
                experiment = PRESETS[name]
                result = run_one(args, experiment, seed, graph_by_experiment[name], run_dir)
                statuses.append(result)
                commands.extend(result["commands"])
                write_json(run_dir / "run_status.json", {"status": "running", "runs": statuses})
    except Exception as exc:
        write_json(run_dir / "run_status.json", {"status": "failed", "runs": statuses, "error": repr(exc)})
        write_commands(run_dir, commands)
        raise

    write_json(run_dir / "run_status.json", {"status": "completed", "runs": statuses})
    write_commands(run_dir, commands)
    if not args.dry_run:
        rows = collect_rows(statuses, args.top_ks)
        write_comparison(run_dir, rows)
    print(f"Fairness pipeline run completed: {run_dir}")


if __name__ == "__main__":
    main()
