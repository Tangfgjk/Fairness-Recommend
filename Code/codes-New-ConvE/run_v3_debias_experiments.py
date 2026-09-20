"""One-command V3 debiasing runner for E0 bias-chain and E1 Pre/In/Post experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from semantic_experiment_utils import (
    code_dir,
    command_to_markdown,
    default_data_root,
    default_runs_root,
    graph_path_for_dataset,
    parse_csv_list,
    parse_seed_list,
    read_json,
    run_logged_command,
    write_json,
)


DEFAULT_V3_DATASETS = ["Eedi", "algebra2005", "XES3G5M-sub-small"]


def status_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return read_json(path).get("status") == "completed"
    except Exception:
        return False


def dataset_run_id(dataset: str, batch_id: str) -> str:
    return f"{dataset}_{batch_id}"


def dataset_run_dir(runs_root: Path, dataset: str, batch_id: str) -> Path:
    return runs_root / dataset / dataset_run_id(dataset, batch_id)


def score_path(run_dir: Path, experiment: str, seed: int, filename: str = "2CKG4ER_uid_ex_scores.pkl") -> Path:
    return run_dir / experiment / f"seed{seed}" / filename


def pipeline_command(args: argparse.Namespace, dataset: str) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "run_fairness_pipeline.py"),
        "--dataset",
        dataset,
        "--run-id",
        dataset_run_id(dataset, args.batch_id),
        "--experiments",
        args.experiments,
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
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
        "--top-ks",
        args.top_ks,
        "--pre-candidate-pool-size",
        str(args.pre_candidate_pool_size),
        "--pre-lambda-item",
        str(args.pre_lambda_item),
        "--pre-lambda-kc",
        str(args.pre_lambda_kc),
        "--fairness-loss-scale",
        str(args.fairness_loss_scale),
        "--fairness-candidate-size",
        str(args.fairness_candidate_size),
        "--fairness-candidate-mode",
        args.fairness_candidate_mode,
        "--fairness-target-gamma",
        str(args.fairness_target_gamma),
        "--fairness-exposure-proxy",
        args.fairness_exposure_proxy,
        "--fairness-surrogate-k",
        str(args.fairness_surrogate_k),
        "--fairness-distance",
        args.fairness_distance,
        "--rerank-top-k",
        str(args.rerank_top_k),
        "--rerank-candidate-multiplier",
        str(args.rerank_candidate_multiplier),
        "--rerank-min-candidates",
        str(args.rerank_min_candidates),
        "--rerank-quota-prefixes",
        args.rerank_quota_prefixes,
        "--rerank-long-tail-ratio-target",
        str(args.rerank_long_tail_ratio_target),
        "--rerank-kc-coverage-ratio-target",
        str(args.rerank_kc_coverage_ratio_target),
        "--rerank-lambda-item",
        str(args.rerank_lambda_item),
        "--rerank-lambda-kc",
        str(args.rerank_lambda_kc),
        "--rerank-beta-item",
        str(args.rerank_beta_item),
        "--rerank-beta-kc",
        str(args.rerank_beta_kc),
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.resume:
        command.append("--resume")
    if args.force_preprocess:
        command.append("--force-preprocess")
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_test_users > 0:
        command.extend(["--max-test-users", str(args.max_test_users)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def e0_command(args: argparse.Namespace, dataset: str, run_dir: Path, seed: int) -> List[str]:
    base_scores = score_path(run_dir, "baseline", seed)
    debiased_scores = score_path(run_dir, "post_only", seed, "2CKG4ER_uid_ex_scores_quota_ratio_hybrid.pkl")
    output_dir = run_dir / "E0_bias_chain_post_only" / f"seed{seed}"
    return [
        sys.executable,
        str(code_dir() / "bias_chain_diagnostics.py"),
        "--data-dir",
        str(args.data_root / dataset / args.source_graph_subdir),
        "--output-dir",
        str(output_dir),
        "--base-scores-file",
        str(base_scores),
        "--debiased-scores-file",
        str(debiased_scores),
        "--top-ks",
        args.bias_top_ks,
        "--popularity-aggregation",
        args.popularity_aggregation,
    ]


def preflight_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    graph_dir = graph_path_for_dataset(dataset, args.data_root, graph_subdir=args.source_graph_subdir)
    required = ["entities.dict", "relations.dict", "Q.txt", "triples.txt", "test_triples.txt", "stu2ex_recommend_full_precision.json"]
    missing = [name for name in required if not (graph_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} graph is missing required files: {missing}")
    raw_candidates = [args.data_root / dataset / "raw" / "interactions_all.csv", args.data_root / dataset / "raw_compact" / "interactions_all.csv"]
    if not any(path.exists() for path in raw_candidates):
        raise FileNotFoundError(f"{dataset} has no raw/raw_compact interactions_all.csv")
    return {"dataset": dataset, "graph_dir": str(graph_dir), "missing": missing}


def run_command_recorded(command: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    print(command_to_markdown(command))
    return run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)


def collect_csv(path: Path, dataset: str, extra: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    prefix = {"dataset": dataset}
    if extra:
        prefix.update(extra)
    return [{**prefix, **row} for row in rows]


def write_rows(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outputs(args: argparse.Namespace, summary_dir: Path) -> None:
    e0_rows: List[Dict[str, Any]] = []
    e1_rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
        for seed in args.seeds:
            e0_rows.extend(
                collect_csv(
                    run_dir / "E0_bias_chain_post_only" / f"seed{seed}" / "bias_chain_metrics.csv",
                    dataset,
                    {"seed": seed},
                )
            )
        e1_rows.extend(collect_csv(run_dir / "comparison" / "fairness_pipeline_comparison.csv", dataset))
    write_rows(e0_rows, summary_dir / "E0_bias_chain_all.csv")
    write_rows(e1_rows, summary_dir / "E1_fairness_pipeline_all.csv")


def run_dataset(args: argparse.Namespace, dataset: str, summary_dir: Path) -> Dict[str, Any]:
    run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"dataset": dataset, "run_dir": str(run_dir), "status": "running"}
    try:
        record["preflight"] = preflight_dataset(args, dataset)
        command = pipeline_command(args, dataset)
        record["pipeline_command"] = command
        rc = run_command_recorded(command, code_dir(), run_dir / "v3_pipeline.log", args.dry_run)
        record["pipeline_exit_code"] = rc
        if rc != 0:
            raise RuntimeError(f"{dataset} E1 pipeline failed with exit code {rc}")

        e0_records: List[Dict[str, Any]] = []
        for seed in args.seeds:
            command = e0_command(args, dataset, run_dir, seed)
            e0_records.append({"seed": seed, "command": command})
            rc = run_command_recorded(command, code_dir(), run_dir / f"v3_e0_seed{seed}.log", args.dry_run)
            e0_records[-1]["exit_code"] = rc
            if rc != 0:
                raise RuntimeError(f"{dataset} E0 diagnostics failed for seed {seed} with exit code {rc}")
        record["e0"] = e0_records
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        write_json(summary_dir / "batch_status.json", {"status": "failed", "last_error": repr(exc), "current": record})
        if not args.continue_on_error:
            raise
    return record


def write_batch_commands(args: argparse.Namespace, summary_dir: Path) -> None:
    lines = ["# V3 Debias E0/E1 Batch Commands", ""]
    for dataset in args.datasets:
        run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
        lines.extend([f"## {dataset} E1 pipeline", "", "```powershell"])
        lines.append(command_to_markdown(pipeline_command(args, dataset)))
        lines.extend(["```", ""])
        for seed in args.seeds:
            lines.extend([f"## {dataset} E0 seed {seed}", "", "```powershell"])
            lines.append(command_to_markdown(e0_command(args, dataset, run_dir, seed)))
            lines.extend(["```", ""])
    (summary_dir / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 debias E0 and E1 experiments across datasets.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_V3_DATASETS))
    parser.add_argument("--batch-id", default="v3_debias_e0_e1_seed2024")
    parser.add_argument("--experiments", default="v3_e1")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--source-graph-subdir", default="er_graph")
    parser.add_argument("--top-ks", default="10,20,50,100")
    parser.add_argument("--bias-top-ks", default="10,20,50,100")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)
    parser.add_argument("--pre-candidate-pool-size", type=int, default=50)
    parser.add_argument("--pre-lambda-item", type=float, default=0.1)
    parser.add_argument("--pre-lambda-kc", type=float, default=0.1)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--fairness-loss-scale", type=float, default=50.0)
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-candidate-mode", choices=["random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"], default="mixed_user")
    parser.add_argument("--fairness-target-gamma", type=float, default=0.25)
    parser.add_argument("--fairness-exposure-proxy", choices=["softmax", "sigmoid_topk"], default="sigmoid_topk")
    parser.add_argument("--fairness-surrogate-k", type=int, default=10)
    parser.add_argument("--fairness-distance", choices=["mse", "l1", "kl_target_model", "js"], default="js")
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
    args = parser.parse_args()
    args.datasets = parse_csv_list(args.datasets)
    args.seeds = parse_seed_list(args.seeds)
    args.data_root = Path(args.data_root)
    args.runs_root = Path(args.runs_root)
    return args


def main() -> None:
    args = parse_args()
    summary_dir = args.runs_root / args.batch_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_batch_commands(args, summary_dir)
    write_json(
        summary_dir / "batch_config.json",
        {
            "datasets": args.datasets,
            "batch_id": args.batch_id,
            "experiments": args.experiments,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "top_ks": args.top_ks,
            "bias_top_ks": args.bias_top_ks,
            "continue_on_error": args.continue_on_error,
            "dry_run": args.dry_run,
        },
    )
    records: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        record = run_dataset(args, dataset, summary_dir)
        records.append(record)
        write_json(summary_dir / "batch_status.json", {"status": "running", "runs": records})
    aggregate_outputs(args, summary_dir)
    final_status = "completed" if all(record.get("status") == "completed" for record in records) else "partial"
    write_json(summary_dir / "batch_status.json", {"status": final_status, "runs": records})
    print(f"V3 debias batch {final_status}: {summary_dir}")


if __name__ == "__main__":
    main()
