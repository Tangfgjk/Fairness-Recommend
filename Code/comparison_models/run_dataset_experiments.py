import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from experiment_utils import load_json, write_json
from conve_ablation_data import CONVE_VARIANTS, prepare_conve_variant_data_dir


CODE_DIR = Path(__file__).resolve().parent
KG4ER_ROOT = CODE_DIR.parent
PROJECT_ROOT = KG4ER_ROOT.parents[1]

DATASET_DIRS = {
    "Eedi": KG4ER_ROOT / "data" / "Eedi",
    "algebra2005": KG4ER_ROOT / "data" / "algebra2005" / "prepared_for_kt",
    "assist2009": KG4ER_ROOT / "data" / "assist2009" / "prepared_for_kt",
    "assist2009-sub": KG4ER_ROOT / "data" / "assist2009-sub" / "prepared_for_kt",
    "statics2011": KG4ER_ROOT / "data" / "statics2011" / "prepared_for_kt",
    "XES3G5M-sub": KG4ER_ROOT / "data" / "XES3G5M-sub" / "prepared_for_kt",
    "XES3G5M-sub-small": KG4ER_ROOT / "data" / "XES3G5M-sub-small" / "prepared_for_kt",
}

REQUIRED_DATA_FILES = [
    "Q.txt",
    "stu2know_mastery.json",
    "stu2know_forget.json",
    "stu2ex_forget.json",
    "stu2ex_recommend.json",
    "triples.txt",
    "test_triples.txt",
    "entities.dict",
    "relations.dict",
]

CONVE_EXPERIMENTS = CONVE_VARIANTS

KGE_EXPERIMENTS = {
    "TransE": {"model": "TransE", "args": []},
    "TransE-adv": {
        "model": "TransE",
        "args": ["--negative_adversarial_sampling", "--adversarial_temperature", "1.0"],
    },
    "RotatE": {"model": "RotatE", "args": ["--double_entity_embedding"]},
    "DistMult": {"model": "DistMult", "args": []},
    "ComplEx": {"model": "ComplEx", "args": ["--double_entity_embedding", "--double_relation_embedding"]},
}

TRADITIONAL_BASELINES = ["EB-CF", "SB-CF", "CBF"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all ER experiments for one dataset with resume support.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_DIRS))
    parser.add_argument("--run-root", type=Path, default=KG4ER_ROOT / "runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume latest run for this dataset unless --run-id is given.")
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--cuda", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--conve-batch-size", type=int, default=1024)
    parser.add_argument("--conve-learning-rate", type=float, default=0.001)
    parser.add_argument("--conve-input-drop", type=float, default=0.2)
    parser.add_argument("--conve-hidden-drop", type=float, default=0.2)
    parser.add_argument("--conve-feat-drop", type=float, default=0.3)
    parser.add_argument("--conve-negative-ratio", type=int, default=5)
    parser.add_argument(
        "--conve-rec-top-k",
        type=int,
        default=10,
        help="Number of independently generated positive rec triples per training learner for each ConvE variant.",
    )
    parser.add_argument("--kge-max-steps", type=int, default=30000)
    parser.add_argument("--kge-batch-size", type=int, default=1024)
    parser.add_argument("--negative-sample-size", type=int, default=256)
    parser.add_argument("--kge-hidden-dim", type=int, default=1000)
    parser.add_argument("--kge-gamma", type=float, default=12.0)
    parser.add_argument("--kge-learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu-num", type=int, default=10)
    parser.add_argument("--top-ks", default="10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--ep-top-k", type=int, default=10)
    parser.add_argument("--target-mastery", type=float, default=0.8)
    parser.add_argument("--models", default="all", help="all or comma-separated experiment names.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def latest_run_dir(run_root, dataset):
    dataset_root = run_root / dataset
    if not dataset_root.exists():
        return None
    candidates = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and "dry" not in path.name.lower()
        and "smoke" not in path.name.lower()
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1] if candidates else None


def resolve_batch_dir(args):
    dataset_root = args.run_root / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if args.run_id:
            run_dir = dataset_root / args.run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"--resume requested but run directory does not exist: {run_dir}")
            return run_dir
        run_dir = latest_run_dir(args.run_root, args.dataset)
        if run_dir is None:
            raise FileNotFoundError(f"--resume requested but no previous run exists under {dataset_root}")
        return run_dir
    run_id = args.run_id or f"{args.dataset}_full_{timestamp()}"
    return dataset_root / run_id


def cuda_enabled(mode):
    if mode == "true":
        return True
    if mode == "false":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def line_count(path):
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fp:
        return sum(1 for _ in fp)


def validate_data_dir(data_dir):
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {missing}")
    relation_count = line_count(data_dir / "relations.dict")
    if relation_count not in {203, 304}:
        raise ValueError(
            f"{data_dir}/relations.dict must contain either 203 no-sequence relations "
            f"or 304 legacy sequence relations, got {relation_count}"
        )

    relation_names = set()
    with (data_dir / "relations.dict").open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                relation_names.add(parts[1])
    if "rec" not in relation_names:
        raise ValueError(f"{data_dir}/relations.dict must contain the recommendation relation 'rec'")
    if not any(name.startswith("mlkc") for name in relation_names):
        raise ValueError(f"{data_dir}/relations.dict must contain mastery relations with prefix 'mlkc'")
    if not any(name.startswith("exfr") for name in relation_names):
        raise ValueError(f"{data_dir}/relations.dict must contain forgetting relations with prefix 'exfr'")


def sequence_interaction_file(data_dir):
    for name in ["sequence_interactions.csv", "interactions.csv"]:
        path = data_dir / name
        if path.exists():
            return path
    return None


def resolve_raw_dir_for_graph(data_dir):
    """Return the raw folder that was used to build this ER graph."""
    manifest_names = [
        "vfin_graph_manifest.json",
        "er_graph_manifest.json",
        "v11_graph_manifest.json",
        "v10_graph_manifest.json",
    ]
    repo_root = CODE_DIR.parent
    for manifest_name in manifest_names:
        manifest_path = data_dir / manifest_name
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path, default={}) or {}
        raw_value = manifest.get("raw_dir") or manifest.get("source_raw_dir")
        raw_name = manifest.get("raw_name")
        candidates = []
        if raw_value:
            raw_path = Path(raw_value)
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                candidates.extend([
                    repo_root / raw_path,
                    Path.cwd() / raw_path,
                    data_dir.parent / raw_path.name,
                ])
        if raw_name:
            candidates.append(data_dir.parent / raw_name)
        for candidate in candidates:
            if candidate.exists():
                return candidate

    compact_dir = data_dir.parent / "raw_compact"
    if compact_dir.exists():
        return compact_dir
    raw_dir = data_dir.parent / "raw"
    if raw_dir.exists():
        return raw_dir
    return None


def load_status(batch_dir):
    return load_json(batch_dir / "status.json", default={"tasks": {}}) or {"tasks": {}}


def save_status(batch_dir, status):
    write_json(status, batch_dir / "status.json")


def task_completed(batch_dir, task_id):
    status = load_status(batch_dir)
    return status.get("tasks", {}).get(task_id, {}).get("status") == "completed"


def update_task(batch_dir, task_id, state, **extra):
    status = load_status(batch_dir)
    payload = status.setdefault("tasks", {}).setdefault(task_id, {})
    payload.update(extra)
    payload["status"] = state
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_status(batch_dir, status)


def run_command(command, cwd, log_path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(str(item) for item in command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {command_text}\n")
        log.flush()
        print(command_text)
        if dry_run:
            return 0
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nreturncode={completed.returncode}\n")
        return completed.returncode


def experiment_filter(args):
    if args.models == "all":
        return None
    return {item.strip() for item in args.models.split(",") if item.strip()}


def wanted(name, selected):
    return selected is None or name in selected


def train_conve(args, batch_dir, data_dir, eval_data_dir, experiment_name, seed, use_cuda, selected):
    if not wanted(experiment_name, selected):
        return
    task_id = f"{experiment_name}/seed{seed}"
    run_dir = batch_dir / experiment_name / f"seed{seed}"
    metrics_path = run_dir / "eval" / "metrics.json"
    if task_completed(batch_dir, task_id) and metrics_path.exists():
        print(f"skip completed {task_id}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    update_task(batch_dir, task_id, "running", run_dir=str(run_dir), seed=seed, experiment=experiment_name)
    command = [
        sys.executable,
        "run_ConvE.py",
        "--data_path",
        str(data_dir),
        "--dataset_name",
        args.dataset,
        "--save_path",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--bs",
        str(args.conve_batch_size),
        "--learning_rate",
        str(args.conve_learning_rate),
        "--input_drop",
        str(args.conve_input_drop),
        "--hidden_drop",
        str(args.conve_hidden_drop),
        "--feat_drop",
        str(args.conve_feat_drop),
        "--negative-ratio",
        str(args.conve_negative_ratio),
        "--cuda",
        "true" if use_cuda else "false",
        "--seed",
        str(seed),
    ]
    if args.resume and (run_dir / "last.pt").exists():
        command.append("--resume")
    rc = run_command(command, CODE_DIR, run_dir / "train.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed during training")
    if args.dry_run:
        update_task(batch_dir, task_id, "dry_run")
        return

    test_command = [
        sys.executable,
        "test_ConvE.py",
        "--dataset",
        args.dataset,
        "--model-type",
        experiment_name,
        "--data-path",
        str(eval_data_dir),
        "--embedding-path",
        str(run_dir),
        "--explain-top-k",
        "3",
        "--explain-user-count",
        "2",
        "--timing-file",
        str(run_dir / "timing.json"),
        "--scores-only",
    ]
    rc = run_command(test_command, CODE_DIR, run_dir / "test.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed during ConvE test")

    score_file = run_dir / f"{experiment_name}_uid_ex_scores.pkl"
    if not score_file.exists():
        fallback = run_dir / "ConvE_uid_ex_scores.pkl"
        score_file = fallback if fallback.exists() else score_file
    evaluate(args, batch_dir, task_id, experiment_name, seed, eval_data_dir, score_file, run_dir)


def train_kge(args, batch_dir, data_dir, eval_data_dir, experiment_name, seed, use_cuda, selected):
    if not wanted(experiment_name, selected):
        return
    task_id = f"{experiment_name}/seed{seed}"
    run_dir = batch_dir / experiment_name / f"seed{seed}"
    metrics_path = run_dir / "eval" / "metrics.json"
    if task_completed(batch_dir, task_id) and metrics_path.exists():
        print(f"skip completed {task_id}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    update_task(batch_dir, task_id, "running", run_dir=str(run_dir), seed=seed, experiment=experiment_name)
    experiment_config = KGE_EXPERIMENTS[experiment_name]
    command = [
        sys.executable,
        "run.py",
        "--do_train",
        "--data_path",
        str(data_dir),
        "--dataset_name",
        args.dataset,
        "--model",
        experiment_config["model"],
        "--save_path",
        str(run_dir),
        "--max_steps",
        str(args.kge_max_steps),
        "--batch_size",
        str(args.kge_batch_size),
        "--negative_sample_size",
        str(args.negative_sample_size),
        "--hidden_dim",
        str(args.kge_hidden_dim),
        "--gamma",
        str(args.kge_gamma),
        "--learning_rate",
        str(args.kge_learning_rate),
        "--cpu_num",
        str(args.cpu_num),
        "--seed",
        str(seed),
    ]
    command.extend(experiment_config["args"])
    if use_cuda:
        command.append("--cuda")
    if args.resume and (run_dir / "checkpoint").exists():
        command.extend(["--init_checkpoint", str(run_dir)])
    rc = run_command(command, CODE_DIR, run_dir / "train.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed during training")
    if args.dry_run:
        update_task(batch_dir, task_id, "dry_run")
        return

    score_dir = run_dir / "scores"
    score_command = [
        sys.executable,
        "score_kge_recommendations.py",
        "--model",
        experiment_name,
        "--data-dir",
        str(eval_data_dir),
        "--embedding-dir",
        str(run_dir),
        "--output-dir",
        str(score_dir),
        "--timing-file",
        str(run_dir / "timing.json"),
    ]
    rc = run_command(score_command, CODE_DIR, run_dir / "test.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed during scoring")
    evaluate(args, batch_dir, task_id, experiment_name, seed, eval_data_dir, score_dir / f"{experiment_name}_uid_ex_scores.pkl", run_dir)


def run_traditional_baselines(args, batch_dir, data_dir, selected):
    if not any(wanted(method, selected) for method in TRADITIONAL_BASELINES):
        return
    task_id = "traditional_baselines"
    run_dir = batch_dir / "traditional_baselines"
    metrics_path = run_dir / "metrics.json"
    if task_completed(batch_dir, task_id) and metrics_path.exists():
        print(f"skip completed {task_id}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in TRADITIONAL_BASELINES if wanted(method, selected)]
    update_task(batch_dir, task_id, "running", run_dir=str(run_dir), experiment=task_id, methods=methods)
    raw_dir = resolve_raw_dir_for_graph(data_dir)
    sequence_file = sequence_interaction_file(data_dir)
    command = [
        sys.executable,
        "run_baselines.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir / "outputs"),
        "--dataset-name",
        args.dataset,
        "--methods",
        ",".join(methods),
        "--top-k-ep",
        str(args.ep_top_k),
        "--timing-file",
        str(run_dir / "timing.json"),
    ]
    if raw_dir is not None:
        command.extend(["--raw-dir", str(raw_dir)])
    elif sequence_file is not None:
        command.extend(["--sequence-file", str(sequence_file)])
    rc = run_command(command, CODE_DIR, run_dir / "baseline.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed")
    if args.dry_run:
        update_task(batch_dir, task_id, "dry_run")
        return
    for method in methods:
        score_file = run_dir / "outputs" / f"{method}_uid_ex_scores.pkl"
        evaluate(args, batch_dir, f"{task_id}/{method}", method, None, data_dir, score_file, run_dir / method)
    update_task(batch_dir, task_id, "completed")


def evaluate(args, batch_dir, task_id, model_name, seed, data_dir, score_file, run_dir):
    command = [
        sys.executable,
        "evaluate_recommendations.py",
        "--data-dir",
        str(data_dir),
        "--scores-file",
        str(score_file),
        "--output-dir",
        str(run_dir / "eval"),
        "--dataset-name",
        args.dataset,
        "--model-name",
        model_name,
        "--top-ks",
        args.top_ks,
        "--ep-top-k",
        str(args.ep_top_k),
        "--target-mastery",
        str(args.target_mastery),
        "--timing-file",
        str(run_dir / "timing.json"),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    rc = run_command(command, CODE_DIR, run_dir / "eval.log", args.dry_run)
    if rc != 0:
        update_task(batch_dir, task_id, "failed", returncode=rc)
        raise RuntimeError(f"{task_id} failed during evaluation")
    update_task(batch_dir, task_id, "completed", metrics=str(run_dir / "eval" / "metrics.json"))


def collect_metrics(batch_dir):
    rows = []
    for metrics_file in batch_dir.glob("**/eval/metrics.json"):
        payload = load_json(metrics_file, default={}) or {}
        ada10 = (payload.get("Ada") or payload.get("ACC") or {}).get("10", {})
        nov10 = payload.get("NOV", {}).get("10", {})
        ep = payload.get("Ep_sim", {})
        rows.append(
            {
                "dataset": payload.get("dataset"),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "Ada@10": ada10.get("mean"),
                "NOV@10": nov10.get("mean"),
                "Ep_sim@10": ep.get("mean"),
                "metrics_file": str(metrics_file),
            }
        )
    summary_dir = batch_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "dataset_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["dataset", "model", "seed", "Ada@10", "NOV@10", "Ep_sim@10", "metrics_file"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md_path = summary_dir / "dataset_summary.md"
    lines = ["# Dataset Experiment Summary", "", "| Dataset | Model | Seed | Ada@10 | NOV@10 | Ep_sim@10 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['seed']} | {row['Ada@10']} | {row['NOV@10']} | {row['Ep_sim@10']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_rows = aggregate_metric_rows(rows)
    stats_csv_path = summary_dir / "dataset_summary_stats.csv"
    with stats_csv_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "dataset",
            "model",
            "run_count",
            "Ada@10_mean",
            "Ada@10_std",
            "NOV@10_mean",
            "NOV@10_std",
            "Ep_sim@10_mean",
            "Ep_sim@10_std",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    stats_md_path = summary_dir / "dataset_summary_stats.md"
    stats_lines = [
        "# Dataset Experiment Mean Std Summary",
        "",
        "| Dataset | Model | Runs | Ada@10 mean | Ada@10 std | NOV@10 mean | NOV@10 std | Ep_sim@10 mean | Ep_sim@10 std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats_rows:
        stats_lines.append(
            f"| {row['dataset']} | {row['model']} | {row['run_count']} | "
            f"{row['Ada@10_mean']} | {row['Ada@10_std']} | "
            f"{row['NOV@10_mean']} | {row['NOV@10_std']} | "
            f"{row['Ep_sim@10_mean']} | {row['Ep_sim@10_std']} |"
        )
    stats_md_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    write_json({"runs": rows, "mean_std": stats_rows}, summary_dir / "dataset_summary.json")


def aggregate_metric_rows(rows):
    grouped = {}
    for row in rows:
        key = (row.get("dataset"), row.get("model"))
        grouped.setdefault(key, []).append(row)

    stats_rows = []
    for (dataset, model), items in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        stats_row = {
            "dataset": dataset,
            "model": model,
            "run_count": len(items),
        }
        for metric in ["Ada@10", "NOV@10", "Ep_sim@10"]:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            stats_row[f"{metric}_mean"] = round(mean(values), 6) if values else None
            stats_row[f"{metric}_std"] = round(stdev(values), 6) if len(values) > 1 else None
        stats_rows.append(stats_row)
    return stats_rows


def write_manifest(args, batch_dir, data_dir, use_cuda, seeds):
    manifest = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "run_dir": str(batch_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "resume": args.resume,
        "cuda_enabled": use_cuda,
        "seeds": seeds,
        "conve_experiments": list(CONVE_EXPERIMENTS.keys()),
        "conve_training": {
            "epochs": args.epochs,
            "batch_size": args.conve_batch_size,
            "learning_rate": args.conve_learning_rate,
            "input_drop": args.conve_input_drop,
            "hidden_drop": args.conve_hidden_drop,
            "feat_drop": args.conve_feat_drop,
            "negative_ratio": args.conve_negative_ratio,
            "rec_top_k": args.conve_rec_top_k,
            "variant_formulas": CONVE_EXPERIMENTS,
            "train_triple_files": ["triples.txt"],
        },
        "kge_experiments": list(KGE_EXPERIMENTS.keys()),
        "kge_training": {
            "max_steps": args.kge_max_steps,
            "batch_size": args.kge_batch_size,
            "negative_sample_size": args.negative_sample_size,
            "hidden_dim": args.kge_hidden_dim,
            "gamma": args.kge_gamma,
            "learning_rate": args.kge_learning_rate,
        },
        "traditional_baselines": TRADITIONAL_BASELINES,
        "command": " ".join(sys.argv),
    }
    write_json(manifest, batch_dir / "manifest.json")


def main():
    args = parse_args()
    data_dir = DATASET_DIRS[args.dataset].resolve()
    validate_data_dir(data_dir)
    batch_dir = resolve_batch_dir(args).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    use_cuda = cuda_enabled(args.cuda)
    selected = experiment_filter(args)
    write_manifest(args, batch_dir, data_dir, use_cuda, seeds)

    variant_root = batch_dir / "data_variants"
    data_variants = {}
    for experiment_name in CONVE_EXPERIMENTS:
        variant_dir = variant_root / experiment_name
        if wanted(experiment_name, selected):
            prepare_conve_variant_data_dir(
                data_dir,
                variant_dir,
                experiment_name,
                top_k_rec=args.conve_rec_top_k,
                resume=args.resume,
            )
        data_variants[experiment_name] = variant_dir

    for experiment_name in CONVE_EXPERIMENTS:
        for seed in seeds:
            train_conve(args, batch_dir, data_variants[experiment_name], data_dir, experiment_name, seed, use_cuda, selected)

    for experiment_name in KGE_EXPERIMENTS:
        for seed in seeds:
            train_kge(args, batch_dir, data_dir, data_dir, experiment_name, seed, use_cuda, selected)

    run_traditional_baselines(args, batch_dir, data_dir, selected)
    collect_metrics(batch_dir)
    print(f"run_dir -> {batch_dir}")
    print(f"summary -> {batch_dir / 'summaries' / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
