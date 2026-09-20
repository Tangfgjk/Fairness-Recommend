"""One-command E2 KPPD runner for ConvE-Debias experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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


DEFAULT_E2_DATASETS = ["Eedi", "algebra2005", "XES3G5M-sub-small"]


@dataclass(frozen=True)
class E2Method:
    name: str
    training_objective: str = "bce"
    negative_sampling_mode: str = "random"
    tail_bias_mode: str = "legacy"
    popularity_debias: str = "none"
    debiased_eval: bool = False


METHODS: Dict[str, E2Method] = {
    "B0": E2Method("B0", training_objective="bce", tail_bias_mode="legacy"),
    "B1": E2Method("B1", training_objective="bce", tail_bias_mode="none"),
    "N1": E2Method("N1", training_objective="bce_bpr", negative_sampling_mode="mixed", tail_bias_mode="none"),
    "D1": E2Method("D1", training_objective="bce_bpr", negative_sampling_mode="mixed", tail_bias_mode="pop_branch", popularity_debias="global", debiased_eval=True),
    "D2": E2Method("D2", training_objective="bce_bpr", negative_sampling_mode="mixed", tail_bias_mode="pop_branch", popularity_debias="global_personal", debiased_eval=True),
    "D3": E2Method("D3", training_objective="bce_bpr", negative_sampling_mode="mixed", tail_bias_mode="pop_branch", popularity_debias="global_personal_need", debiased_eval=True),
}


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


def method_seed_dir(run_dir: Path, method: str, seed: int) -> Path:
    return run_dir / method / f"seed{seed}"


def train_command(args: argparse.Namespace, dataset: str, graph_dir: Path, seed_dir: Path, method: E2Method, seed: int) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "run_semantic_conve.py"),
        "--data-path",
        str(graph_dir),
        "--dataset-name",
        dataset,
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
        "2CKG4ER",
        "--training-objective",
        method.training_objective,
        "--bpr-weight",
        str(args.bpr_weight),
        "--negative-sampling-mode",
        method.negative_sampling_mode,
        "--bpr-distance-margin",
        str(args.bpr_distance_margin),
        "--bpr-pair-weighting",
        args.bpr_pair_weighting,
        "--tail-bias-mode",
        method.tail_bias_mode,
        "--popularity-debias",
        method.popularity_debias,
        "--beta-global-pop",
        str(args.beta_global_pop),
        "--beta-personal-pop",
        str(args.beta_personal_pop),
        "--beta-need",
        str(args.beta_need),
        "--lambda-global-pop",
        str(args.lambda_global_pop),
        "--lambda-personal-pop",
        str(args.lambda_personal_pop),
        "--global-pop-aux-weight",
        str(args.global_pop_aux_weight),
        "--personal-pop-aux-weight",
        str(args.personal_pop_aux_weight),
        "--need-aux-weight",
        str(args.need_aux_weight),
        "--decorr-weight",
        str(args.decorr_weight),
        "--fairness-loss",
        "none",
        "--fairness-popularity-source",
        "train_interactions",
        "--fairness-popularity-aggregation",
        args.popularity_aggregation,
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.resume:
        command.append("--resume")
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    return command


def test_command(args: argparse.Namespace, dataset: str, graph_dir: Path, seed_dir: Path, score_mode: str, output_file: Path, save_components: bool) -> List[str]:
    command = [
        sys.executable,
        str(code_dir() / "test_semantic_conve.py"),
        "--data-path",
        str(graph_dir),
        "--dataset-name",
        dataset,
        "--save-path",
        str(seed_dir),
        "--cuda",
        args.cuda,
        "--ablation",
        "2CKG4ER",
        "--score-mode",
        score_mode,
        "--output-file",
        str(output_file),
        "--forgetting-score-weight",
        "0.0",
    ]
    if args.max_test_users > 0:
        command.extend(["--max-test-users", str(args.max_test_users)])
    if save_components:
        command.append("--save-logit-components")
    return command


def eval_command(args: argparse.Namespace, dataset: str, graph_dir: Path, scores_file: Path, output_dir: Path, model_name: str, seed: int) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "evaluate_recommendations.py"),
        "--data-dir",
        str(graph_dir),
        "--scores-file",
        str(scores_file),
        "--output-dir",
        str(output_dir),
        "--dataset-name",
        dataset,
        "--model-name",
        model_name,
        "--top-ks",
        args.top_ks,
        "--fairness-popularity-source",
        "train_interactions",
        "--fairness-popularity-aggregation",
        args.popularity_aggregation,
        "--seed",
        str(seed),
    ]


def e0_command(args: argparse.Namespace, graph_dir: Path, raw_scores: Path, debiased_scores: Path, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "bias_chain_diagnostics.py"),
        "--data-dir",
        str(graph_dir),
        "--output-dir",
        str(output_dir),
        "--base-scores-file",
        str(raw_scores),
        "--debiased-scores-file",
        str(debiased_scores),
        "--top-ks",
        args.bias_top_ks,
        "--popularity-aggregation",
        args.popularity_aggregation,
    ]


def run_logged(command: Sequence[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    print(command_to_markdown(command))
    return run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)


def run_stage(name: str, command: Sequence[str], cwd: Path, log_path: Path, status_path: Path, dry_run: bool) -> None:
    if status_completed(status_path):
        return
    rc = run_logged(command, cwd, log_path, dry_run)
    write_json(status_path, {"status": "completed" if rc == 0 else "failed", "stage": name, "command": list(command), "exit_code": rc})
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}. See {log_path}")


def collect_csv(path: Path, dataset: str, method: str, seed: int, variant: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    return [{"dataset": dataset, "method": method, "seed": seed, "variant": variant, **row} for row in rows]


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


def run_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    graph_dir = graph_path_for_dataset(dataset, args.data_root, graph_subdir=args.source_graph_subdir)
    run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    statuses: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for method_name in args.methods:
            method = METHODS[method_name]
            seed_dir = method_seed_dir(run_dir, method.name, seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            train_cmd = train_command(args, dataset, graph_dir, seed_dir, method, seed)
            run_stage("train", train_cmd, code_dir(), seed_dir / "runner_train.log", seed_dir / "train_stage.json", args.dry_run)
            raw_scores = seed_dir / "2CKG4ER_uid_ex_scores_raw.pkl"
            test_cmd = test_command(args, dataset, graph_dir, seed_dir, "raw", raw_scores, save_components=method.popularity_debias != "none")
            run_stage("test_raw", test_cmd, code_dir(), seed_dir / "runner_test_raw.log", seed_dir / "test_raw_stage.json", args.dry_run)
            raw_eval_dir = seed_dir / "eval_raw"
            eval_cmd = eval_command(args, dataset, graph_dir, raw_scores, raw_eval_dir, f"{method.name}-Raw", seed)
            run_stage("eval_raw", eval_cmd, code_dir(), seed_dir / "runner_eval_raw.log", seed_dir / "eval_raw_stage.json", args.dry_run)
            statuses.append({"dataset": dataset, "method": method.name, "seed": seed, "variant": "raw", "eval_dir": str(raw_eval_dir)})
            if method.debiased_eval:
                debiased_scores = seed_dir / "2CKG4ER_uid_ex_scores_debiased.pkl"
                debias_cmd = test_command(args, dataset, graph_dir, seed_dir, "debiased", debiased_scores, save_components=False)
                run_stage("test_debiased", debias_cmd, code_dir(), seed_dir / "runner_test_debiased.log", seed_dir / "test_debiased_stage.json", args.dry_run)
                debias_eval_dir = seed_dir / "eval_debiased"
                eval_deb_cmd = eval_command(args, dataset, graph_dir, debiased_scores, debias_eval_dir, f"{method.name}-Debias", seed)
                run_stage("eval_debiased", eval_deb_cmd, code_dir(), seed_dir / "runner_eval_debiased.log", seed_dir / "eval_debiased_stage.json", args.dry_run)
                e0_cmd = e0_command(args, graph_dir, raw_scores, debiased_scores, seed_dir / "E0_raw_to_debiased")
                run_stage("e0_raw_to_debiased", e0_cmd, code_dir(), seed_dir / "runner_e0.log", seed_dir / "e0_stage.json", args.dry_run)
                statuses.append({"dataset": dataset, "method": method.name, "seed": seed, "variant": "debiased", "eval_dir": str(debias_eval_dir)})
    write_json(run_dir / "run_status.json", {"status": "completed", "runs": statuses})
    return {"dataset": dataset, "run_dir": str(run_dir), "runs": statuses}


def aggregate(args: argparse.Namespace, summary_dir: Path) -> None:
    metric_rows: List[Dict[str, Any]] = []
    fairness_rows: List[Dict[str, Any]] = []
    e0_rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
        for seed in args.seeds:
            for method_name in args.methods:
                seed_dir = method_seed_dir(run_dir, method_name, seed)
                metric_rows.extend(collect_csv(seed_dir / "eval_raw" / "metrics.csv", dataset, method_name, seed, "raw"))
                fairness_rows.extend(collect_csv(seed_dir / "eval_raw" / "fairness_metrics.csv", dataset, method_name, seed, "raw"))
                if METHODS[method_name].debiased_eval:
                    metric_rows.extend(collect_csv(seed_dir / "eval_debiased" / "metrics.csv", dataset, method_name, seed, "debiased"))
                    fairness_rows.extend(collect_csv(seed_dir / "eval_debiased" / "fairness_metrics.csv", dataset, method_name, seed, "debiased"))
                    e0_rows.extend(collect_csv(seed_dir / "E0_raw_to_debiased" / "bias_chain_metrics.csv", dataset, method_name, seed, "raw_to_debiased"))
    write_rows(metric_rows, summary_dir / "E2_metrics_all.csv")
    write_rows(fairness_rows, summary_dir / "E2_fairness_all.csv")
    write_rows(e0_rows, summary_dir / "E2_E0_bias_chain_all.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 KPPD ConvE-Debias experiments.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_E2_DATASETS))
    parser.add_argument("--methods", default="B0,B1,N1,D1,D2,D3")
    parser.add_argument("--batch-id", default="e2_kppd_seed2024")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--source-graph-subdir", default="er_graph")
    parser.add_argument("--top-ks", default="10,20,50,100")
    parser.add_argument("--bias-top-ks", default="10,20,50,100")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--bpr-weight", type=float, default=1.0)
    parser.add_argument("--bpr-distance-margin", type=float, default=0.01)
    parser.add_argument("--bpr-pair-weighting", choices=["none", "distance_gap"], default="distance_gap")
    parser.add_argument("--beta-global-pop", type=float, default=1.0)
    parser.add_argument("--beta-personal-pop", type=float, default=1.0)
    parser.add_argument("--beta-need", type=float, default=1.0)
    parser.add_argument("--lambda-global-pop", type=float, default=1.0)
    parser.add_argument("--lambda-personal-pop", type=float, default=1.0)
    parser.add_argument("--global-pop-aux-weight", type=float, default=0.1)
    parser.add_argument("--personal-pop-aux-weight", type=float, default=0.1)
    parser.add_argument("--need-aux-weight", type=float, default=0.0)
    parser.add_argument("--decorr-weight", type=float, default=0.01)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)
    args = parser.parse_args()
    args.datasets = parse_csv_list(args.datasets)
    args.methods = parse_csv_list(args.methods)
    args.seeds = parse_seed_list(args.seeds)
    unknown = [method for method in args.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown E2 methods: {unknown}. Valid: {sorted(METHODS)}")
    return args


def main() -> None:
    args = parse_args()
    summary_dir = args.runs_root / args.batch_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in args.datasets:
        results.append(run_dataset(args, dataset))
    aggregate(args, summary_dir)
    write_json(summary_dir / "e2_kppd_status.json", {"status": "completed", "datasets": results})
    print(f"E2 KPPD completed: {summary_dir}")


if __name__ == "__main__":
    main()

