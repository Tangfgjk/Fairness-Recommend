"""One-command V3 ablation and external comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from semantic_experiment_utils import (
    code_dir,
    command_to_markdown,
    default_data_root,
    default_runs_root,
    mark_stage,
    parse_csv_list,
    parse_seed_list,
    read_json,
    run_logged_command,
    stage_done,
    write_json,
)


DEFAULT_DATASETS = ["Eedi", "algebra2005", "XES3G5M-sub-small"]
DEFAULT_V3_EXPERIMENTS = "main"
DEFAULT_COMPARISON_MODELS = "TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF"
FAIRNESS_METRICS = (
    "ItemExposureGini",
    "KCExposureGini",
    "ItemCoverage",
    "KCCoverage",
    "LongTailItemExposureShare",
    "LongTailKCExposureShare",
    "HeadItemExposureShare",
    "HeadKCExposureShare",
)


def comparison_code_dir() -> Path:
    return code_dir().parent / "comparison_models"


def v3_batch_id(args: argparse.Namespace) -> str:
    return args.v3_batch_id or f"{args.batch_id}_v3"


def comparison_run_id(args: argparse.Namespace, dataset: str) -> str:
    prefix = args.comparison_run_id_prefix or args.batch_id
    return f"{dataset}_{prefix}_comparison"


def comparison_run_dir(args: argparse.Namespace, dataset: str) -> Path:
    return args.runs_root / dataset / comparison_run_id(args, dataset)


def total_summary_dir(args: argparse.Namespace) -> Path:
    return args.runs_root / args.batch_id


def run_stage(
    stage_name: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    status_path: Path,
    dry_run: bool,
) -> bool:
    if stage_done(status_path):
        print(f"skip completed stage: {stage_name}")
        return True
    mark_stage(status_path, "running", command=list(command))
    print(command_to_markdown(command))
    rc = run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        mark_stage(status_path, "failed", command=list(command), exit_code=rc)
        return False
    mark_stage(status_path, "completed", command=list(command), exit_code=rc)
    return True


def build_v3_command(args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "run_v3_kppd_pipeline.py"),
        "--datasets",
        ",".join(args.datasets),
        "--experiments",
        args.v3_experiments,
        "--batch-id",
        v3_batch_id(args),
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
        args.graph_subdir,
        "--top-ks",
        args.top_ks,
        "--bias-top-ks",
        args.bias_top_ks,
        "--popularity-aggregation",
        args.popularity_aggregation,
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    if args.no_run_e0:
        command.append("--no-run-e0")
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_test_users > 0:
        command.extend(["--max-test-users", str(args.max_test_users)])
    return command


def build_comparison_command(args: argparse.Namespace, dataset: str) -> List[str]:
    command = [
        sys.executable,
        str(comparison_code_dir() / "run_v9_comparison_experiments.py"),
        "--dataset",
        dataset,
        "--data-root",
        str(args.data_root),
        "--graph-subdir",
        args.graph_subdir,
        "--run-root",
        str(args.runs_root),
        "--run-id",
        comparison_run_id(args, dataset),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--cuda",
        args.cuda,
        "--models",
        args.comparison_models,
        "--kge-max-steps",
        str(args.kge_max_steps),
        "--kge-batch-size",
        str(args.kge_batch_size),
        "--negative-sample-size",
        str(args.negative_sample_size),
        "--kge-hidden-dim",
        str(args.kge_hidden_dim),
        "--kge-gamma",
        str(args.kge_gamma),
        "--kge-learning-rate",
        str(args.kge_learning_rate),
        "--cpu-num",
        str(args.cpu_num),
        "--top-ks",
        args.top_ks,
        "--ep-top-k",
        str(args.ep_top_k),
        "--target-mastery",
        str(args.target_mastery),
        "--fairness-popularity-source",
        args.fairness_popularity_source,
        "--fairness-popularity-aggregation",
        args.popularity_aggregation,
    ]
    if args.resume and comparison_run_dir(args, dataset).exists():
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


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


def iter_comparison_metrics(run_dir: Path) -> Iterable[tuple[Path, Dict[str, Any]]]:
    for metrics_path in sorted(run_dir.glob("**/eval/metrics.json")):
        yield metrics_path, read_json(metrics_path)


def comparison_metric_rows(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dir = comparison_run_dir(args, dataset)
    for metrics_path, payload in iter_comparison_metrics(run_dir):
        model = payload.get("model")
        seed = payload.get("seed")
        ada = payload.get("Ada") or payload.get("ACC") or {}
        nov = payload.get("NOV") or {}
        ep = payload.get("Ep_sim") or {}
        for top_k in args.top_ks_list:
            k = str(top_k)
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "family": "comparison",
                        "method": model,
                        "seed": seed,
                        "metric": "Ada",
                        "top_k": k,
                        "mean": (ada.get(k) or {}).get("mean"),
                        "std": (ada.get(k) or {}).get("std"),
                        "metrics_file": str(metrics_path),
                    },
                    {
                        "dataset": dataset,
                        "family": "comparison",
                        "method": model,
                        "seed": seed,
                        "metric": "NOV",
                        "top_k": k,
                        "mean": (nov.get(k) or {}).get("mean"),
                        "std": (nov.get(k) or {}).get("std"),
                        "metrics_file": str(metrics_path),
                    },
                ]
            )
        rows.append(
            {
                "dataset": dataset,
                "family": "comparison",
                "method": model,
                "seed": seed,
                "metric": "Ep_sim",
                "top_k": ep.get("top_k"),
                "mean": ep.get("mean"),
                "std": ep.get("std"),
                "metrics_file": str(metrics_path),
            }
        )
    return rows


def comparison_fairness_rows(args: argparse.Namespace, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dir = comparison_run_dir(args, dataset)
    for metrics_path, payload in iter_comparison_metrics(run_dir):
        model = payload.get("model")
        seed = payload.get("seed")
        fairness = payload.get("Fairness") or {}
        for top_k in args.top_ks_list:
            payload_at_k = fairness.get(str(top_k)) or {}
            row = {
                "dataset": dataset,
                "family": "comparison",
                "method": model,
                "seed": seed,
                "top_k": str(top_k),
                "metrics_file": str(metrics_path),
            }
            for metric_name in FAIRNESS_METRICS:
                row[metric_name] = payload_at_k.get(metric_name)
            rows.append(row)
    return rows


def key_for_metric_row(row: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("family", "")),
        str(row.get("method", row.get("experiment", ""))),
        str(row.get("seed", "")),
    )


def build_main_table(
    metric_rows: Sequence[Dict[str, Any]],
    fairness_rows: Sequence[Dict[str, Any]],
    top_ks: Sequence[int],
) -> List[Dict[str, Any]]:
    table: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for row in metric_rows:
        metric = row.get("metric")
        top_k = str(row.get("top_k", ""))
        if metric not in {"Ada", "NOV"} or top_k not in {str(k) for k in top_ks}:
            continue
        dataset, family, method, seed = key_for_metric_row(row)
        key = (dataset, family, method, seed, top_k)
        item = table.setdefault(
            key,
            {"dataset": dataset, "family": family, "method": method, "seed": seed, "top_k": top_k},
        )
        item[metric] = row.get("mean")
    for row in metric_rows:
        if row.get("metric") != "Ep_sim":
            continue
        dataset, family, method, seed = key_for_metric_row(row)
        for top_k in top_ks:
            key = (dataset, family, method, seed, str(top_k))
            item = table.setdefault(
                key,
                {"dataset": dataset, "family": family, "method": method, "seed": seed, "top_k": str(top_k)},
            )
            item["Ep_sim@10"] = row.get("mean")
    for row in fairness_rows:
        dataset, family, method, seed = key_for_metric_row(row)
        top_k = str(row.get("top_k", ""))
        if top_k not in {str(k) for k in top_ks}:
            continue
        key = (dataset, family, method, seed, top_k)
        item = table.setdefault(
            key,
            {"dataset": dataset, "family": family, "method": method, "seed": seed, "top_k": top_k},
        )
        for metric_name in FAIRNESS_METRICS:
            item[metric_name] = row.get(metric_name)
    return sorted(table.values(), key=lambda item: (item["dataset"], item["family"], item["method"], str(item["seed"]), int(item["top_k"])))


def load_v3_rows(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_dir = args.runs_root / v3_batch_id(args)
    metric_rows = []
    for row in read_csv_rows(summary_dir / "V3_main_metrics_all.csv"):
        metric_rows.append(
            {
                "dataset": row.get("dataset"),
                "family": "v3_ablation",
                "method": row.get("experiment"),
                "seed": row.get("seed"),
                "metric": row.get("metric"),
                "top_k": row.get("top_k"),
                "mean": row.get("mean"),
                "std": row.get("std"),
                "metrics_file": str(summary_dir / "V3_main_metrics_all.csv"),
            }
        )
    fairness_rows = []
    for row in read_csv_rows(summary_dir / "V3_main_fairness_all.csv"):
        item = {
            "dataset": row.get("dataset"),
            "family": "v3_ablation",
            "method": row.get("experiment"),
            "seed": row.get("seed"),
            "top_k": row.get("top_k"),
            "metrics_file": str(summary_dir / "V3_main_fairness_all.csv"),
        }
        for metric_name in FAIRNESS_METRICS:
            item[metric_name] = row.get(metric_name)
        fairness_rows.append(item)
    e0_rows = read_csv_rows(summary_dir / "V3_main_E0_bias_chain_all.csv")
    for row in e0_rows:
        row.setdefault("family", "v3_ablation")
        if "method" not in row:
            row["method"] = row.get("experiment")
    return metric_rows, fairness_rows, e0_rows


def aggregate_outputs(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = total_summary_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    fairness_rows: List[Dict[str, Any]] = []
    e0_rows: List[Dict[str, Any]] = []
    if not args.skip_v3:
        v3_metrics, v3_fairness, e0_rows = load_v3_rows(args)
        metric_rows.extend(v3_metrics)
        fairness_rows.extend(v3_fairness)
    if not args.skip_comparison:
        for dataset in args.datasets:
            metric_rows.extend(comparison_metric_rows(args, dataset))
            fairness_rows.extend(comparison_fairness_rows(args, dataset))

    main_rows = build_main_table(metric_rows, fairness_rows, args.top_ks_list)
    write_rows(metric_rows, output_dir / "all_metric_rows.csv")
    write_rows(fairness_rows, output_dir / "all_fairness_rows.csv")
    write_rows(e0_rows, output_dir / "all_v3_e0_bias_chain_rows.csv")
    write_rows(main_rows, output_dir / "all_methods_at_k.csv")
    return {
        "summary_dir": str(output_dir),
        "metric_rows": len(metric_rows),
        "fairness_rows": len(fairness_rows),
        "e0_rows": len(e0_rows),
        "main_rows": len(main_rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 ablations and comparison baselines for selected datasets.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--batch-id", default="v3_ablation_comparison_seed2024")
    parser.add_argument("--v3-batch-id", default=None)
    parser.add_argument("--comparison-run-id-prefix", default=None)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--graph-subdir", default="er_graph")
    parser.add_argument("--cuda", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")

    parser.add_argument("--v3-experiments", default=DEFAULT_V3_EXPERIMENTS)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-run-e0", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)

    parser.add_argument("--comparison-models", default=DEFAULT_COMPARISON_MODELS)
    parser.add_argument("--kge-max-steps", type=int, default=30000)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--negative-sample-size", type=int, default=256)
    parser.add_argument("--kge-hidden-dim", type=int, default=1000)
    parser.add_argument("--kge-gamma", type=float, default=12.0)
    parser.add_argument("--kge-learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu-num", type=int, default=10)

    parser.add_argument("--top-ks", default="10,20,50,100")
    parser.add_argument("--bias-top-ks", default="10,20,50,100")
    parser.add_argument("--ep-top-k", type=int, default=10)
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--fairness-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")

    args = parser.parse_args(argv)
    args.datasets = parse_csv_list(args.datasets)
    args.seeds = parse_seed_list(args.seeds)
    args.top_ks_list = [int(item) for item in parse_csv_list(args.top_ks)]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_dir = total_summary_dir(args)
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "run_config.json", vars(args))

    stages = []
    if not args.skip_v3:
        command = build_v3_command(args)
        ok = run_stage(
            "v3_ablation",
            command,
            code_dir(),
            summary_dir / "logs" / "v3_ablation.log",
            summary_dir / "stages" / "v3_ablation.json",
            args.dry_run,
        )
        stages.append({"stage": "v3_ablation", "ok": ok, "command": command})
        if not ok and not args.continue_on_error:
            raise SystemExit("V3 ablation stage failed")

    if not args.skip_comparison:
        for dataset in args.datasets:
            command = build_comparison_command(args, dataset)
            ok = run_stage(
                f"comparison_{dataset}",
                command,
                comparison_code_dir(),
                summary_dir / "logs" / f"comparison_{dataset}.log",
                summary_dir / "stages" / f"comparison_{dataset}.json",
                args.dry_run,
            )
            stages.append({"stage": f"comparison_{dataset}", "ok": ok, "command": command})
            if not ok and not args.continue_on_error:
                raise SystemExit(f"Comparison stage failed for {dataset}")

    aggregate_info = aggregate_outputs(args) if not args.dry_run else {"summary_dir": str(summary_dir), "dry_run": True}
    write_json(summary_dir / "run_status.json", {"stages": stages, "aggregate": aggregate_info})
    print(json.dumps(aggregate_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
