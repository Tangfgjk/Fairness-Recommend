"""One-command V3 Pre/KPPD/Post debiasing experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from fairness_preprocess_rec_graph import prepare_fair_preprocess_graph
from run_fair_model_experiments import FAIRNESS_METRICS, QUALITY_METRICS
from semantic_experiment_utils import (
    code_dir,
    command_to_markdown,
    default_data_root,
    default_runs_root,
    graph_path_for_dataset,
    mark_stage,
    parse_csv_list,
    parse_seed_list,
    read_json,
    run_logged_command,
    stage_done,
    write_json,
)


DEFAULT_DATASETS = ["Eedi", "algebra2005", "XES3G5M-sub-small"]


@dataclass(frozen=True)
class V3Experiment:
    name: str
    pre: str = "none"
    in_method: str = "none"
    post: str = "none"


EXPERIMENTS: Dict[str, V3Experiment] = {
    "baseline": V3Experiment("baseline"),
    "pre_item": V3Experiment("pre_item", pre="item"),
    "pre_kc": V3Experiment("pre_kc", pre="kc"),
    "pre_item_kc": V3Experiment("pre_item_kc", pre="item_kc"),
    "in_kppd": V3Experiment("in_kppd", in_method="kppd"),
    "post_only": V3Experiment("post_only", post="quota_ratio_hybrid"),
    "in_kppd_post": V3Experiment("in_kppd_post", in_method="kppd", post="quota_ratio_hybrid"),
    "pre_in_kppd_post": V3Experiment("pre_in_kppd_post", pre="item_kc", in_method="kppd", post="quota_ratio_hybrid"),
}

ALIASES = {
    "in_only": "in_kppd",
    "in_post": "in_kppd_post",
    "pre_in_post": "pre_in_kppd_post",
}

GROUPS = {
    "main": ["baseline", "pre_item", "pre_kc", "pre_item_kc", "in_kppd", "post_only", "in_kppd_post", "pre_in_kppd_post"],
    "pre": ["baseline", "pre_item", "pre_kc", "pre_item_kc"],
    "kppd": ["baseline", "in_kppd", "in_kppd_post"],
}


def dataset_run_id(dataset: str, batch_id: str) -> str:
    return f"{dataset}_{batch_id}"


def dataset_run_dir(runs_root: Path, dataset: str, batch_id: str) -> Path:
    return runs_root / dataset / dataset_run_id(dataset, batch_id)


def method_seed_dir(run_dir: Path, experiment: str, seed: int) -> Path:
    return run_dir / experiment / f"seed{seed}"


def parse_experiments(value: str | Sequence[str]) -> List[str]:
    requested = parse_csv_list(value)
    expanded: List[str] = []
    for item in requested:
        item = ALIASES.get(item, item)
        if item in GROUPS:
            expanded.extend(GROUPS[item])
        elif item in EXPERIMENTS:
            expanded.append(item)
        else:
            valid = sorted([*EXPERIMENTS, *GROUPS, *ALIASES])
            raise ValueError(f"Unknown experiment: {item}. Valid: {valid}")
    deduped: List[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def pre_lambdas(args: argparse.Namespace, pre: str) -> tuple[float, float]:
    if pre == "none":
        return 0.0, 0.0
    if pre == "item":
        return float(args.pre_lambda_item), 0.0
    if pre == "kc":
        return 0.0, float(args.pre_lambda_kc)
    if pre == "item_kc":
        return float(args.pre_lambda_item), float(args.pre_lambda_kc)
    raise ValueError(f"Unsupported pre method: {pre}")


def pre_graph_dir(args: argparse.Namespace, dataset: str, pre: str) -> Path:
    lambda_item, lambda_kc = pre_lambdas(args, pre)
    li = str(lambda_item).replace(".", "p")
    lk = str(lambda_kc).replace(".", "p")
    return args.data_root / dataset / args.pre_graph_root_subdir / args.batch_id / f"{pre}_M{args.pre_candidate_pool_size}_li{li}_lk{lk}"


def score_file(seed_dir: Path, variant: str = "final") -> Path:
    if variant == "raw":
        return seed_dir / "2CKG4ER_uid_ex_scores_raw.pkl"
    if variant == "debiased":
        return seed_dir / "2CKG4ER_uid_ex_scores_debiased.pkl"
    if variant == "post":
        return seed_dir / "2CKG4ER_uid_ex_scores_quota_ratio_hybrid.pkl"
    return seed_dir / "2CKG4ER_uid_ex_scores.pkl"


def run_stage(stage_name: str, command: Sequence[str], cwd: Path, log_path: Path, status_path: Path, dry_run: bool) -> None:
    if stage_done(status_path):
        return
    mark_stage(status_path, "running", command=list(command))
    print(command_to_markdown(command))
    rc = run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        mark_stage(status_path, "failed", command=list(command), exit_code=rc)
        raise RuntimeError(f"{stage_name} failed with exit code {rc}. See {log_path}")
    mark_stage(status_path, "completed", command=list(command), exit_code=rc)


def train_command(args: argparse.Namespace, dataset: str, graph_dir: Path, seed_dir: Path, experiment: V3Experiment, seed: int) -> List[str]:
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
        "--fairness-loss",
        "none",
        "--fairness-popularity-source",
        "train_interactions",
        "--fairness-popularity-aggregation",
        args.popularity_aggregation,
    ]
    if experiment.in_method == "kppd":
        command.extend(
            [
                "--training-objective",
                "bce_bpr",
                "--bpr-weight",
                str(args.bpr_weight),
                "--negative-sampling-mode",
                "mixed",
                "--bpr-distance-margin",
                str(args.bpr_distance_margin),
                "--bpr-pair-weighting",
                args.bpr_pair_weighting,
                "--tail-bias-mode",
                "pop_branch",
                "--popularity-debias",
                "global_personal_need",
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
            ]
        )
    else:
        command.extend(["--training-objective", "bce", "--tail-bias-mode", "legacy", "--popularity-debias", "none"])
    if args.deterministic:
        command.append("--deterministic")
    if args.resume:
        command.append("--resume")
    if args.max_train_batches > 0:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    return command


def test_command(
    args: argparse.Namespace,
    dataset: str,
    graph_dir: Path,
    seed_dir: Path,
    experiment: V3Experiment,
    score_mode: str,
    output_file: Path,
    save_components: bool = False,
) -> List[str]:
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
    if save_components:
        command.append("--save-logit-components")
    if args.max_test_users > 0:
        command.extend(["--max-test-users", str(args.max_test_users)])
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


def rerank_command(args: argparse.Namespace, graph_dir: Path, source_scores: Path, output_scores: Path) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "fairness_rerank.py"),
        "--data-dir",
        str(graph_dir),
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
        "--popularity-source",
        "train_interactions",
        "--popularity-aggregation",
        args.popularity_aggregation,
    ]


def e0_command(args: argparse.Namespace, graph_dir: Path, baseline_scores: Path, method_scores: Path, output_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "bias_chain_diagnostics.py"),
        "--data-dir",
        str(graph_dir),
        "--output-dir",
        str(output_dir),
        "--base-scores-file",
        str(baseline_scores),
        "--debiased-scores-file",
        str(method_scores),
        "--top-ks",
        args.bias_top_ks,
        "--popularity-aggregation",
        args.popularity_aggregation,
    ]


def prepare_pre_graph(args: argparse.Namespace, dataset: str, source_graph: Path, pre: str, run_dir: Path) -> Path:
    if pre == "none":
        return source_graph
    target = pre_graph_dir(args, dataset, pre)
    status_path = run_dir / "preprocess" / f"{pre}_stage.json"
    manifest_path = target / "fairness_preprocess_manifest.json"
    if stage_done(status_path) and manifest_path.exists():
        return target
    lambda_item, lambda_kc = pre_lambdas(args, pre)
    if args.dry_run:
        print(f"[dry-run] preprocess {dataset}/{pre}: {source_graph} -> {target}")
        mark_stage(status_path, "completed", command=["prepare_fair_preprocess_graph", str(target)], exit_code=0)
        return target
    manifest = prepare_fair_preprocess_graph(
        source_graph_dir=source_graph,
        output_graph_dir=target,
        candidate_pool_size=args.pre_candidate_pool_size,
        top_k_rec=args.pre_top_k_rec,
        lambda_item=lambda_item,
        lambda_kc=lambda_kc,
        popularity_source=args.pre_popularity_source,
        popularity_aggregation=args.popularity_aggregation,
        head_ratio=args.head_ratio,
        long_tail_ratio=args.long_tail_ratio,
        epsilon=args.pre_epsilon,
        force=args.force_preprocess,
    )
    mark_stage(status_path, "completed", command=["prepare_fair_preprocess_graph", str(target)], exit_code=0)
    write_json(run_dir / "preprocess" / f"{pre}_manifest.json", manifest)
    return target


def final_scores_for(seed_dir: Path, experiment: V3Experiment) -> tuple[Path, str]:
    if experiment.post == "quota_ratio_hybrid":
        return score_file(seed_dir, "post"), "eval_quota_ratio_hybrid"
    if experiment.in_method == "kppd":
        return score_file(seed_dir, "debiased"), "eval_debiased"
    return score_file(seed_dir), "eval"


def run_model_experiment(args: argparse.Namespace, dataset: str, graph_dir: Path, run_dir: Path, experiment: V3Experiment, seed: int) -> Dict[str, Any]:
    seed_dir = method_seed_dir(run_dir, experiment.name, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = train_command(args, dataset, graph_dir, seed_dir, experiment, seed)
    run_stage("train", command, code_dir(), seed_dir / "runner_train.log", seed_dir / "train_stage.json", args.dry_run)

    if experiment.in_method == "kppd":
        raw_scores = score_file(seed_dir, "raw")
        command = test_command(args, dataset, graph_dir, seed_dir, experiment, "raw", raw_scores, save_components=args.save_kppd_components)
        run_stage("test_raw", command, code_dir(), seed_dir / "runner_test_raw.log", seed_dir / "test_raw_stage.json", args.dry_run)
        debiased_scores = score_file(seed_dir, "debiased")
        command = test_command(args, dataset, graph_dir, seed_dir, experiment, "debiased", debiased_scores)
        run_stage("test_debiased", command, code_dir(), seed_dir / "runner_test_debiased.log", seed_dir / "test_debiased_stage.json", args.dry_run)
        eval_source = debiased_scores
        eval_dir = seed_dir / "eval_debiased"
    else:
        scores = score_file(seed_dir)
        command = test_command(args, dataset, graph_dir, seed_dir, experiment, "raw", scores)
        run_stage("test", command, code_dir(), seed_dir / "runner_test.log", seed_dir / "test_stage.json", args.dry_run)
        eval_source = scores
        eval_dir = seed_dir / "eval"

    if experiment.post == "quota_ratio_hybrid":
        post_scores = score_file(seed_dir, "post")
        command = rerank_command(args, graph_dir, eval_source, post_scores)
        run_stage("post", command, code_dir(), seed_dir / "runner_post.log", seed_dir / "post_stage.json", args.dry_run)
        eval_source = post_scores
        eval_dir = seed_dir / "eval_quota_ratio_hybrid"

    command = eval_command(args, dataset, graph_dir, eval_source, eval_dir, experiment.name, seed)
    run_stage("eval", command, code_dir(), seed_dir / "runner_eval.log", seed_dir / "eval_stage.json", args.dry_run)

    return {
        "dataset": dataset,
        "experiment": experiment.name,
        "pre": experiment.pre,
        "in": experiment.in_method,
        "post": experiment.post,
        "seed": seed,
        "graph_dir": str(graph_dir),
        "seed_dir": str(seed_dir),
        "scores_file": str(eval_source),
        "eval_dir": str(eval_dir),
    }


def run_post_only(args: argparse.Namespace, dataset: str, graph_dir: Path, run_dir: Path, seed: int) -> Dict[str, Any]:
    baseline_dir = method_seed_dir(run_dir, "baseline", seed)
    baseline_scores = score_file(baseline_dir)
    if not args.dry_run and not baseline_scores.exists():
        raise FileNotFoundError(f"baseline scores are required for post_only: {baseline_scores}")
    experiment = EXPERIMENTS["post_only"]
    seed_dir = method_seed_dir(run_dir, "post_only", seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    post_scores = score_file(seed_dir, "post")
    command = rerank_command(args, graph_dir, baseline_scores, post_scores)
    run_stage("post_only", command, code_dir(), seed_dir / "runner_post.log", seed_dir / "post_stage.json", args.dry_run)
    eval_dir = seed_dir / "eval_quota_ratio_hybrid"
    command = eval_command(args, dataset, graph_dir, post_scores, eval_dir, "post_only", seed)
    run_stage("eval_post_only", command, code_dir(), seed_dir / "runner_eval.log", seed_dir / "eval_stage.json", args.dry_run)
    return {
        "dataset": dataset,
        "experiment": "post_only",
        "pre": experiment.pre,
        "in": experiment.in_method,
        "post": experiment.post,
        "seed": seed,
        "graph_dir": str(graph_dir),
        "seed_dir": str(seed_dir),
        "scores_file": str(post_scores),
        "eval_dir": str(eval_dir),
    }


def collect_csv(path: Path, dataset: str, experiment: str, seed: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    return [{"dataset": dataset, "experiment": experiment, "seed": seed, **row} for row in rows]


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


def aggregate(args: argparse.Namespace, summary_dir: Path, statuses: Sequence[Dict[str, Any]]) -> None:
    metric_rows: List[Dict[str, Any]] = []
    fairness_rows: List[Dict[str, Any]] = []
    e0_rows: List[Dict[str, Any]] = []
    for status in statuses:
        eval_dir = Path(status["eval_dir"])
        metric_rows.extend(collect_csv(eval_dir / "metrics.csv", status["dataset"], status["experiment"], int(status["seed"])))
        fairness_rows.extend(collect_csv(eval_dir / "fairness_metrics.csv", status["dataset"], status["experiment"], int(status["seed"])))
        e0_dir = Path(status["seed_dir"]) / "E0_vs_baseline"
        e0_rows.extend(collect_csv(e0_dir / "bias_chain_metrics.csv", status["dataset"], status["experiment"], int(status["seed"])))
    write_rows(metric_rows, summary_dir / "V3_main_metrics_all.csv")
    write_rows(fairness_rows, summary_dir / "V3_main_fairness_all.csv")
    write_rows(e0_rows, summary_dir / "V3_main_E0_bias_chain_all.csv")


def run_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    source_graph = graph_path_for_dataset(dataset, args.data_root, graph_subdir=args.source_graph_subdir)
    run_dir = dataset_run_dir(args.runs_root, dataset, args.batch_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_by_pre = {"none": source_graph}
    for pre in sorted({EXPERIMENTS[name].pre for name in args.experiments if EXPERIMENTS[name].pre != "none"}):
        graph_by_pre[pre] = prepare_pre_graph(args, dataset, source_graph, pre, run_dir)
    statuses: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for name in args.experiments:
            if name == "post_only":
                status = run_post_only(args, dataset, source_graph, run_dir, seed)
            else:
                experiment = EXPERIMENTS[name]
                graph_dir = graph_by_pre[experiment.pre]
                status = run_model_experiment(args, dataset, graph_dir, run_dir, experiment, seed)
            statuses.append(status)
            write_json(run_dir / "run_status.json", {"status": "running", "runs": statuses})

        if args.run_e0:
            baseline_scores = score_file(method_seed_dir(run_dir, "baseline", seed))
            for status in statuses:
                if int(status["seed"]) != seed or status["experiment"] == "baseline":
                    continue
                out_dir = Path(status["seed_dir"]) / "E0_vs_baseline"
                command = e0_command(args, Path(status["graph_dir"]), baseline_scores, Path(status["scores_file"]), out_dir)
                run_stage("e0", command, code_dir(), Path(status["seed_dir"]) / "runner_e0.log", Path(status["seed_dir"]) / "e0_stage.json", args.dry_run)

    write_json(run_dir / "run_status.json", {"status": "completed", "runs": statuses})
    return {"dataset": dataset, "run_dir": str(run_dir), "runs": statuses}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3 main experiments where In-processing is upgraded to KPPD.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--experiments", default="main")
    parser.add_argument("--batch-id", default="v3_kppd_main_seed2024")
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
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--source-graph-subdir", default="er_graph")
    parser.add_argument("--top-ks", default="10,20,50,100")
    parser.add_argument("--bias-top-ks", default="10,20,50,100")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--pre-graph-root-subdir", default="fair_pre_rec_graph_v3_kppd")
    parser.add_argument("--pre-candidate-pool-size", type=int, default=50)
    parser.add_argument("--pre-top-k-rec", type=int, default=None)
    parser.add_argument("--pre-lambda-item", type=float, default=0.1)
    parser.add_argument("--pre-lambda-kc", type=float, default=0.1)
    parser.add_argument("--pre-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--pre-epsilon", type=float, default=1e-8)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--long-tail-ratio", type=float, default=0.8)
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
    parser.add_argument("--run-e0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-kppd-components", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)
    args = parser.parse_args()
    args.datasets = parse_csv_list(args.datasets)
    args.experiments = parse_experiments(args.experiments)
    args.seeds = parse_seed_list(args.seeds)
    return args


def main() -> None:
    args = parse_args()
    summary_dir = args.runs_root / args.batch_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_statuses: List[Dict[str, Any]] = []
    write_json(summary_dir / "run_config.json", vars(args))
    for dataset in args.datasets:
        result = run_dataset(args, dataset)
        results.append(result)
        all_statuses.extend(result["runs"])
    aggregate(args, summary_dir, all_statuses)
    write_json(summary_dir / "v3_kppd_main_status.json", {"status": "completed", "datasets": results})
    print(f"V3 KPPD main experiments completed: {summary_dir}")


if __name__ == "__main__":
    main()
