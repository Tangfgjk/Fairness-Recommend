import argparse
import json
import time
from pathlib import Path

from baseline_recommenders import (
    SUPPORTED_BASELINES,
    build_all_baseline_scores,
    load_cf_protocol,
    load_json_matrix,
    load_q_matrix,
    load_sequence_interactions,
    save_uid_ex_scores,
)
from ep_sim import calculate_ep_sim
from experiment_utils import resolve_run_dir, update_timing, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run recommendation baselines and Ep_sim evaluation.")
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "Eedi"
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Canonical raw dataset directory containing student_split.csv and "
            "interactions_all.csv. Enables split-safe ItemKNN/UserKNN baselines."
        ),
    )
    parser.add_argument("--cf-neighbor-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--timing-file", type=Path, default=None)
    parser.add_argument("--methods", type=str, default=",".join(SUPPORTED_BASELINES))
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--top-k-ep", type=int, default=10)
    parser.add_argument("--learning-gain", type=float, default=0.1)
    return parser.parse_args()


def _load_required_inputs(data_dir):
    q_matrix = load_q_matrix(data_dir / "Q.txt")
    mastery = load_json_matrix(data_dir / "stu2know_mastery.json")
    forgetting_path = data_dir / "stu2ex_forget.json"
    forgetting = load_json_matrix(forgetting_path) if forgetting_path.exists() else None
    return q_matrix, mastery, None, forgetting


def main():
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    dataset_name = args.dataset_name or args.data_dir.name
    output_dir = args.output_dir
    if output_dir is None and args.run_root:
        output_dir = resolve_run_dir(args.run_root, dataset_name, "baselines", run_id=args.run_id) / "outputs"
    output_dir = output_dir or args.data_dir / "baseline_outputs"
    timing_file = args.timing_file or output_dir.parent / "timing.json"
    q_matrix, mastery, sequence, forgetting = _load_required_inputs(args.data_dir)
    if args.raw_dir is not None:
        interactions, query_interactions, user_ids = load_cf_protocol(args.raw_dir)
        cf_protocol = {
            "mode": "split_safe_knn",
            "raw_dir": str(args.raw_dir),
            "library": "outer_train_learners_only",
            "queries": "outer_test_learners_observed_history_only",
            "neighbor_count": args.cf_neighbor_count,
        }
    else:
        interactions = load_sequence_interactions(args.sequence_file)
        query_interactions = interactions
        user_ids = [f"uid{idx}" for idx in range(len(mastery))]
        cf_protocol = {
            "mode": "legacy_sequence_file",
            "sequence_file": str(args.sequence_file) if args.sequence_file else None,
        }

    inference_start = time.perf_counter()
    all_scores = build_all_baseline_scores(
        q_matrix=q_matrix,
        mastery=mastery,
        sequence=sequence,
        forgetting=forgetting,
        interactions=interactions,
        query_interactions=query_interactions,
        methods=methods,
        user_ids=user_ids,
        neighbor_count=args.cf_neighbor_count,
    )
    update_timing(timing_file, "inference_without_cache", time.perf_counter() - inference_start, extra={"methods": methods})

    metric_start = time.perf_counter()
    metrics = {}
    for method, uid_ex_scores in all_scores.items():
        score_path = output_dir / f"{method}_uid_ex_scores.pkl"
        json_score_path = output_dir / f"{method}_uid_ex_scores.json"
        ep_path = output_dir / f"{method}_ep_sim.json"
        save_uid_ex_scores(uid_ex_scores, score_path)
        save_uid_ex_scores(uid_ex_scores, json_score_path)
        ep_result = calculate_ep_sim(
            mastery=mastery,
            q_matrix=q_matrix,
            uid_ex_scores=uid_ex_scores,
            top_k=args.top_k_ep,
            learning_gain=args.learning_gain,
        )
        write_json(ep_result, ep_path)
        metrics[f"{method}_ep_sim_mean"] = ep_result["mean"]
        metrics[f"{method}_ep_sim_std"] = ep_result["std"]
        print(f"{method}: scores -> {score_path}; Ep_sim -> {ep_path}")
    update_timing(timing_file, "evaluation_metric", time.perf_counter() - metric_start, extra={"methods": methods})
    metrics["cf_protocol"] = cf_protocol
    write_json(metrics, output_dir.parent / "metrics.json")
    write_json(cf_protocol, output_dir.parent / "cf_protocol.json")


if __name__ == "__main__":
    main()
