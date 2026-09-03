"""One-command runner for KG-aware and in-processing fair 2CKG4ER experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from prepare_fair_kg_graph import prepare_fair_kg_graph
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
    parse_seed_list,
    parse_top_ks,
    python_env_info,
    read_json,
    run_logged_command,
    stage_done,
    timestamp,
    write_json,
)


MODEL_METHODS = [
    "baseline",
    "fairkg_edges",
    "fairkg_weighted",
    "fairreg_item",
    "fairreg_kc",
    "fairreg_item_kc",
    "fairreg_item_kc_hybrid_rerank",
]
QUALITY_METRICS = ["Ada", "NOV"]
FAIRNESS_METRICS = [
    "ItemExposureGini",
    "KCExposureGini",
    "ItemCoverage",
    "KCCoverage",
    "LongTailItemExposureShare",
    "LongTailKCExposureShare",
]


def fairness_regularizer_flags(args: argparse.Namespace, alpha_item: str, alpha_kc: str) -> List[str]:
    return [
        "--fairness-loss",
        "expected_exposure",
        "--fairness-alpha-item",
        alpha_item,
        "--fairness-alpha-kc",
        alpha_kc,
        "--fairness-loss-scale",
        str(args.fairness_loss_scale),
        "--fairness-candidate-size",
        str(args.fairness_candidate_size),
        "--fairness-candidate-mode",
        args.fairness_candidate_mode,
        "--fairness-temperature",
        str(args.fairness_temperature),
        "--fairness-target-gamma",
        str(args.fairness_target_gamma),
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


def parse_methods(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if items == ["all"]:
        return list(MODEL_METHODS)
    invalid = sorted(set(items) - set(MODEL_METHODS))
    if invalid:
        raise ValueError(f"Unknown fair model methods: {','.join(invalid)}")
    return items


def method_uses_fair_graph(method: str) -> bool:
    return method in {"fairkg_edges", "fairkg_weighted", "fairreg_kc", "fairreg_item_kc", "fairreg_item_kc_hybrid_rerank"}


def method_is_rerank(method: str) -> bool:
    return method == "fairreg_item_kc_hybrid_rerank"


def method_seed_dir(run_dir: Path, method: str, seed: int) -> Path:
    return run_dir / method / f"seed{seed}"


def score_file(seed_dir: Path) -> Path:
    return seed_dir / "2CKG4ER_uid_ex_scores.pkl"


def train_command(args: argparse.Namespace, graph_path: Path, seed_dir: Path, method: str, seed: int) -> List[str]:
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
    if method in {"fairkg_weighted"}:
        command.extend(["--relation-loss-weights", "balanced"])
    if method == "fairreg_item":
        command.extend(fairness_regularizer_flags(args, str(args.alpha_item), "0.0"))
    if method == "fairreg_kc":
        command.extend(fairness_regularizer_flags(args, "0.0", str(args.alpha_kc)))
    if method == "fairreg_item_kc":
        command.extend(
            [
                "--relation-loss-weights",
                "balanced",
                *fairness_regularizer_flags(args, str(args.alpha_item), str(args.alpha_kc)),
            ]
        )
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


def rerank_command(args: argparse.Namespace, graph_path: Path, source_scores: Path, output_scores: Path) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "fairness_rerank.py"),
        "--data-dir",
        str(graph_path),
        "--scores-file",
        str(source_scores),
        "--output-file",
        str(output_scores),
        "--method",
        "quota_ratio_hybrid",
        "--top-k",
        str(args.rerank_top_k),
        "--candidate-multiplier",
        str(args.rerank_candidate_multiplier),
        "--min-candidates",
        str(args.rerank_min_candidates),
        "--quota-prefixes",
        args.rerank_quota_prefixes,
        "--long-tail-item-ratio",
        str(args.rerank_long_tail_ratio_target),
        "--kc-coverage-ratio",
        str(args.rerank_kc_coverage_ratio_target),
        "--lambda-item",
        str(args.rerank_lambda_item),
        "--lambda-kc",
        str(args.rerank_lambda_kc),
        "--beta-item",
        str(args.rerank_beta_item),
        "--beta-kc",
        str(args.rerank_beta_kc),
    ]


def run_stage(stage_name: str, command: List[str], cwd: Path, log_path: Path, status_path: Path, dry_run: bool) -> None:
    mark_stage(status_path, "running", command=command)
    rc = run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        mark_stage(status_path, "failed", command=command, exit_code=rc)
        raise RuntimeError(f"{stage_name} failed with exit code {rc}. See {log_path}")
    mark_stage(status_path, "completed", command=command, exit_code=rc)


def run_train_test_eval(
    args: argparse.Namespace,
    method: str,
    seed: int,
    graph_path: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    seed_dir = method_seed_dir(run_dir, method, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    commands: List[Dict[str, Any]] = []

    command = train_command(args, graph_path, seed_dir, method, seed)
    commands.append({"method": method, "seed": seed, "stage": "train", "command": command})
    if not stage_done(seed_dir / "train_stage.json") or not (seed_dir / "best.pt").exists():
        run_stage("train", command, code_dir(), seed_dir / "runner_train.log", seed_dir / "train_stage.json", args.dry_run)

    command = test_command(args, graph_path, seed_dir)
    commands.append({"method": method, "seed": seed, "stage": "test", "command": command})
    if not stage_done(seed_dir / "test_stage.json") or not score_file(seed_dir).exists():
        run_stage("test", command, code_dir(), seed_dir / "runner_test.log", seed_dir / "test_stage.json", args.dry_run)

    command = eval_command(args, graph_path, score_file(seed_dir), seed_dir / "eval", seed, method)
    commands.append({"method": method, "seed": seed, "stage": "eval", "command": command})
    if not stage_done(seed_dir / "eval_stage.json") or not (seed_dir / "eval" / "metrics.json").exists():
        run_stage("eval", command, code_dir(), seed_dir / "runner_eval.log", seed_dir / "eval_stage.json", args.dry_run)

    return {"method": method, "seed": seed, "status": "completed", "seed_dir": seed_dir, "commands": commands}


def run_hybrid_rerank(
    args: argparse.Namespace,
    seed: int,
    graph_path: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    source_dir = method_seed_dir(run_dir, "fairreg_item_kc", seed)
    source_scores = score_file(source_dir)
    if not args.dry_run and not source_scores.exists():
        raise FileNotFoundError(f"fairreg_item_kc scores are required before hybrid rerank: {source_scores}")
    seed_dir = method_seed_dir(run_dir, "fairreg_item_kc_hybrid_rerank", seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_scores = score_file(seed_dir)
    commands: List[Dict[str, Any]] = []

    command = rerank_command(args, graph_path, source_scores, output_scores)
    commands.append({"method": "fairreg_item_kc_hybrid_rerank", "seed": seed, "stage": "rerank", "command": command})
    if not stage_done(seed_dir / "rerank_stage.json") or not output_scores.exists():
        run_stage("rerank", command, code_dir(), seed_dir / "runner_rerank.log", seed_dir / "rerank_stage.json", args.dry_run)

    command = eval_command(args, graph_path, output_scores, seed_dir / "eval", seed, "fairreg_item_kc_hybrid_rerank")
    commands.append({"method": "fairreg_item_kc_hybrid_rerank", "seed": seed, "stage": "eval", "command": command})
    if not stage_done(seed_dir / "eval_stage.json") or not (seed_dir / "eval" / "metrics.json").exists():
        run_stage("eval", command, code_dir(), seed_dir / "runner_eval.log", seed_dir / "eval_stage.json", args.dry_run)

    return {
        "method": "fairreg_item_kc_hybrid_rerank",
        "seed": seed,
        "status": "completed",
        "seed_dir": seed_dir,
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


def collect_comparison(run_dir: Path, methods: Sequence[str], seeds: Sequence[int], top_ks: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            eval_dir = method_seed_dir(run_dir, method, seed) / "eval"
            metrics_path = eval_dir / "metrics.json"
            fairness_path = eval_dir / "fairness_metrics.json"
            if not metrics_path.exists() or not fairness_path.exists():
                continue
            metrics = read_json(metrics_path)
            fairness = read_json(fairness_path)
            for top_k in top_ks:
                row: Dict[str, Any] = {
                    "method": method,
                    "seed": seed,
                    "K": int(top_k),
                    "eval_dir": str(eval_dir),
                }
                for metric in QUALITY_METRICS:
                    row[metric] = metric_value(metrics, metric, top_k)
                for metric in FAIRNESS_METRICS:
                    row[metric] = fairness_value(fairness, metric, top_k)
                rows.append(row)
    return rows


def add_deltas(rows: Sequence[Dict[str, Any]], baseline_method: str = "baseline") -> List[Dict[str, Any]]:
    baseline = {(int(row["seed"]), int(row["K"])): row for row in rows if row["method"] == baseline_method}
    output: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        base = baseline.get((int(row["seed"]), int(row["K"])))
        for metric in [*QUALITY_METRICS, *FAIRNESS_METRICS]:
            value = row.get(metric)
            base_value = base.get(metric) if base else None
            enriched[f"Delta_{metric}"] = None if value is None or base_value is None else float(value) - float(base_value)
        output.append(enriched)
    return output


def write_comparison(run_dir: Path, rows: Sequence[Dict[str, Any]], delta_rows: Sequence[Dict[str, Any]]) -> None:
    out_dir = run_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["method", "seed", "K", "Ada", "NOV", *FAIRNESS_METRICS, "eval_dir"]
    delta_fields = [*fields[:-1], *[f"Delta_{metric}" for metric in [*QUALITY_METRICS, *FAIRNESS_METRICS]], "eval_dir"]
    with (out_dir / "fair_model_comparison.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "fair_model_comparison_delta.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=delta_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(delta_rows)

    lines = ["# Fair Model Comparison", ""]
    for top_k in sorted({int(row["K"]) for row in rows}):
        lines.extend([f"## K={top_k}", ""])
        lines.append("| method | seed | Ada | NOV | ItemGini | KCGini | ItemCov | KCCov | LTItem | LTKC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in [item for item in rows if int(item["K"]) == top_k]:
            lines.append(
                "| {method} | {seed} | {Ada} | {NOV} | {ItemExposureGini} | {KCExposureGini} | "
                "{ItemCoverage} | {KCCoverage} | {LongTailItemExposureShare} | {LongTailKCExposureShare} |".format(
                    **{key: "" if value is None else f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()}
                )
            )
        lines.append("")
    (out_dir / "fair_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def write_commands(run_dir: Path, commands: Sequence[Dict[str, Any]]) -> None:
    lines = ["# Fair Model Experiment Commands", ""]
    for item in commands:
        lines.append(f"## {item['method']} / {item['stage']} / seed {item['seed']}")
        lines.append("")
        lines.append("```powershell")
        lines.append(command_to_markdown(item["command"]))
        lines.append("```")
        lines.append("")
    (run_dir / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-aware fair 2CKG4ER model experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--methods", default="all")
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
    parser.add_argument("--fair-graph-subdir", default="fair_kg_graph")
    parser.add_argument("--overwrite-fair-graph", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--forgetting-score-weight", type=float, default=0.0)
    parser.add_argument("--forgetting-exercise-batch-size", type=int, default=256)
    parser.add_argument("--alpha-item", type=float, default=0.05)
    parser.add_argument("--alpha-kc", type=float, default=0.05)
    parser.add_argument("--fairness-loss-scale", type=float, default=1.0)
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-candidate-mode", choices=["random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"], default="random")
    parser.add_argument("--fairness-temperature", type=float, default=1.0)
    parser.add_argument("--fairness-target-gamma", type=float, default=0.5)
    parser.add_argument("--fairness-exposure-proxy", choices=["softmax", "sigmoid_topk"], default="softmax")
    parser.add_argument("--fairness-surrogate-k", type=int, default=10)
    parser.add_argument("--fairness-distance", choices=["mse", "l1", "kl_target_model", "js"], default="mse")
    parser.add_argument("--fairness-top-score-ratio", type=float, default=0.5)
    parser.add_argument("--fairness-popular-ratio", type=float, default=0.25)
    parser.add_argument("--fairness-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="rec_triples")
    parser.add_argument("--fairness-popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug smoke-test limit passed to training; 0 means no limit.")
    parser.add_argument("--max-test-users", type=int, default=0, help="Debug smoke-test limit passed to testing; 0 means no limit.")
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument("--rerank-candidate-multiplier", type=float, default=1.5)
    parser.add_argument("--rerank-min-candidates", type=int, default=150)
    parser.add_argument("--rerank-quota-prefixes", default="10,20,50,100")
    parser.add_argument("--rerank-long-tail-ratio-target", type=float, default=0.10)
    parser.add_argument("--rerank-kc-coverage-ratio-target", type=float, default=0.20)
    parser.add_argument("--rerank-lambda-item", type=float, default=0.3)
    parser.add_argument("--rerank-lambda-kc", type=float, default=0.3)
    parser.add_argument("--rerank-beta-item", type=float, default=0.1)
    parser.add_argument("--rerank-beta-kc", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.methods = parse_methods(args.methods)
    args.seeds = parse_seed_list(args.seeds)
    args.top_ks = parse_top_ks(args.top_ks)
    run_id = args.run_id or f"{args.dataset}_fair_model_{timestamp()}"
    run_dir = args.runs_root / args.dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_graph = graph_path_for_dataset(args.dataset, args.data_root, graph_subdir=args.source_graph_subdir)
    fair_graph = args.data_root / args.dataset / args.fair_graph_subdir
    if any(method_uses_fair_graph(method) for method in args.methods):
        if args.dry_run:
            fair_manifest = {"dry_run": True, "target_dir": str(fair_graph)}
        elif args.overwrite_fair_graph or not (fair_graph / "fair_kg_graph_manifest.json").exists():
            fair_manifest = prepare_fair_kg_graph(source_graph, fair_graph, overwrite=args.overwrite_fair_graph)
        else:
            fair_manifest = read_json(fair_graph / "fair_kg_graph_manifest.json")
    else:
        fair_manifest = None

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "run_id": run_id,
            "run_dir": run_dir,
            "source_graph": source_graph,
            "fair_graph": fair_graph,
            "fair_graph_manifest": fair_manifest,
            "methods": args.methods,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "bs": args.bs,
            "learning_rate": args.learning_rate,
            "negative_ratio": args.negative_ratio,
            "max_train_batches": args.max_train_batches,
            "max_test_users": args.max_test_users,
            "cuda": args.cuda,
            "top_ks": args.top_ks,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "fairness_regularizer": {
                "alpha_item": args.alpha_item,
                "alpha_kc": args.alpha_kc,
                "loss_scale": args.fairness_loss_scale,
                "candidate_size": args.fairness_candidate_size,
                "candidate_mode": args.fairness_candidate_mode,
                "temperature": args.fairness_temperature,
                "target_gamma": args.fairness_target_gamma,
                "exposure_proxy": args.fairness_exposure_proxy,
                "surrogate_top_k": args.fairness_surrogate_k,
                "distance": args.fairness_distance,
                "top_score_ratio": args.fairness_top_score_ratio,
                "popular_ratio": args.fairness_popular_ratio,
                "popularity_source": args.fairness_popularity_source,
                "popularity_aggregation": args.fairness_popularity_aggregation,
            },
            "env": python_env_info(),
        },
    )

    statuses: List[Dict[str, Any]] = []
    commands: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for method in args.methods:
            try:
                if method_is_rerank(method):
                    if "fairreg_item_kc" not in args.methods:
                        result = run_train_test_eval(args, "fairreg_item_kc", seed, fair_graph, run_dir)
                        statuses.append(result)
                        commands.extend(result.get("commands", []))
                    result = run_hybrid_rerank(args, seed, fair_graph, run_dir)
                else:
                    graph = fair_graph if method_uses_fair_graph(method) else source_graph
                    result = run_train_test_eval(args, method, seed, graph, run_dir)
                statuses.append(result)
                commands.extend(result.get("commands", []))
                write_json(run_dir / "run_status.json", {"status": "running", "runs": statuses})
            except Exception as exc:
                statuses.append({"method": method, "seed": seed, "status": "failed", "error": repr(exc)})
                write_json(run_dir / "run_status.json", {"status": "failed", "runs": statuses})
                write_commands(run_dir, commands)
                raise

    final_status = "completed" if all(item.get("status") == "completed" for item in statuses) else "partial"
    write_json(run_dir / "run_status.json", {"status": final_status, "runs": statuses})
    write_commands(run_dir, commands)
    if not args.dry_run:
        rows = collect_comparison(run_dir, args.methods, args.seeds, args.top_ks)
        write_comparison(run_dir, rows, add_deltas(rows))
    print(f"Fair model run {final_status}: {run_dir}")


if __name__ == "__main__":
    main()
