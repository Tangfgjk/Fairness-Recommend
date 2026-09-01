"""One-command runner for 2CKG4ER experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from semantic_experiment_utils import (
    DEFAULT_TOP_KS,
    GRAPH_ABLATIONS,
    MODEL_NAME,
    MODEL_VERSION,
    VALID_ABLATIONS,
    ablation_model_dir,
    code_dir,
    command_to_markdown,
    default_ablation_data_root,
    default_data_root,
    default_runs_root,
    graph_path_for_dataset,
    mark_stage,
    old_codes_root,
    parse_ablation_list,
    parse_seed_list,
    parse_top_ks,
    python_env_info,
    read_json,
    run_logged_command,
    stage_done,
    timestamp,
    write_json,
)
from semantic_ablation_data import prepare_semantic_ablation_graph
from validate_semantic_ready import validate_dataset


def run_dir_for(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"{args.dataset}_2ckg4er_{timestamp()}"
    return args.runs_root / args.dataset / run_id


def seed_dir_for(run_dir: Path, ablation: str, seed: int) -> Path:
    return run_dir / ablation_model_dir(ablation) / f"seed{seed}"


def write_commands(run_dir: Path, commands: List[Dict[str, Any]]) -> None:
    lines = ["# 2CKG4ER Experiment Commands", ""]
    for item in commands:
        lines.append(f"## {item['ablation']} / {item['stage']} / seed {item['seed']}")
        lines.append("")
        lines.append("```powershell")
        lines.append(command_to_markdown(item["command"]))
        lines.append("```")
        lines.append("")
    (run_dir / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def command_train(
    args: argparse.Namespace,
    graph_path: Path,
    seed_dir: Path,
    seed: int,
    resume_train: bool,
    ablation: str,
) -> List[str]:
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
        ablation,
    ]
    if args.deterministic:
        command.append("--deterministic")
    if resume_train:
        command.append("--resume")
    return command


def command_test(args: argparse.Namespace, eval_graph_path: Path, seed_dir: Path, ablation: str) -> List[str]:
    return [
        sys.executable,
        str(code_dir() / "test_semantic_conve.py"),
        "--data-path",
        str(eval_graph_path),
        "--dataset-name",
        args.dataset,
        "--save-path",
        str(seed_dir),
        "--cuda",
        args.cuda,
        "--ablation",
        ablation,
        "--forgetting-score-weight",
        str(args.forgetting_score_weight),
        "--forgetting-exercise-batch-size",
        str(args.forgetting_exercise_batch_size),
    ]


def command_eval(args: argparse.Namespace, eval_graph_path: Path, seed_dir: Path, seed: int, ablation: str) -> List[str]:
    scores_file = seed_dir / "2CKG4ER_uid_ex_scores.pkl"
    return [
        sys.executable,
        str(old_codes_root() / "evaluate_recommendations.py"),
        "--data-dir",
        str(eval_graph_path),
        "--scores-file",
        str(scores_file),
        "--output-dir",
        str(seed_dir / "eval"),
        "--dataset-name",
        args.dataset,
        "--model-name",
        ablation_model_dir(ablation),
        "--top-ks",
        ",".join(str(k) for k in args.top_ks),
        "--seed",
        str(seed),
    ]


def run_stage(
    stage_name: str,
    command: List[str],
    cwd: Path,
    log_path: Path,
    status_path: Path,
    dry_run: bool,
) -> None:
    mark_stage(status_path, "running", command=command)
    rc = run_logged_command(command, cwd=cwd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        mark_stage(status_path, "failed", command=command, exit_code=rc)
        raise RuntimeError(f"{stage_name} failed with exit code {rc}. See {log_path}")
    mark_stage(status_path, "completed", command=command, exit_code=rc)


def seed_completed(seed_dir: Path) -> bool:
    return (
        stage_done(seed_dir / "train_stage.json")
        and stage_done(seed_dir / "test_stage.json")
        and stage_done(seed_dir / "eval_stage.json")
        and (seed_dir / "eval" / "metrics.json").exists()
    )


def graph_path_for_ablation(args: argparse.Namespace, base_graph_path: Path, run_dir: Path, ablation: str) -> Path:
    if ablation not in GRAPH_ABLATIONS:
        return base_graph_path
    target = args.ablation_data_root / args.dataset / f"{ablation}_top{args.top_k_rec}"
    manifest = prepare_semantic_ablation_graph(
        source_dir=base_graph_path,
        target_dir=target,
        ablation=ablation,
        top_k_rec=args.top_k_rec,
        delta_1=args.delta_1,
        delta_2=args.delta_2,
        resume=args.resume,
    )
    write_json(run_dir / "ablation_graphs" / f"{ablation}.json", manifest)
    return target


def run_seed(
    args: argparse.Namespace,
    run_dir: Path,
    train_graph_path: Path,
    eval_graph_path: Path,
    seed: int,
    ablation: str,
) -> Dict[str, Any]:
    seed_dir = seed_dir_for(run_dir, ablation, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    commands: List[Dict[str, Any]] = []

    if seed_completed(seed_dir):
        return {"seed": seed, "ablation": ablation, "status": "skipped_completed", "seed_dir": seed_dir}

    resume_train = bool(args.resume and (seed_dir / "last.pt").exists())
    train_command = command_train(args, train_graph_path, seed_dir, seed, resume_train, ablation)
    commands.append({"stage": "train", "ablation": ablation, "seed": seed, "command": train_command})
    train_outputs_ready = (seed_dir / "best.pt").exists() and (seed_dir / "last.pt").exists() and (seed_dir / "metrics.json").exists()
    if not stage_done(seed_dir / "train_stage.json") or not train_outputs_ready:
        run_stage(
            "train",
            train_command,
            cwd=code_dir(),
            log_path=seed_dir / "runner_train.log",
            status_path=seed_dir / "train_stage.json",
            dry_run=args.dry_run,
        )

    test_command = command_test(args, eval_graph_path, seed_dir, ablation)
    commands.append({"stage": "test", "ablation": ablation, "seed": seed, "command": test_command})
    test_outputs_ready = (seed_dir / "2CKG4ER_uid_ex_scores.pkl").exists() and (seed_dir / "2ckg4er_inference.json").exists()
    if not stage_done(seed_dir / "test_stage.json") or not test_outputs_ready:
        run_stage(
            "test",
            test_command,
            cwd=code_dir(),
            log_path=seed_dir / "runner_test.log",
            status_path=seed_dir / "test_stage.json",
            dry_run=args.dry_run,
        )

    eval_command = command_eval(args, eval_graph_path, seed_dir, seed, ablation)
    commands.append({"stage": "eval", "ablation": ablation, "seed": seed, "command": eval_command})
    eval_outputs_ready = (seed_dir / "eval" / "metrics.json").exists() and (seed_dir / "eval" / "metrics.csv").exists()
    if not stage_done(seed_dir / "eval_stage.json") or not eval_outputs_ready:
        run_stage(
            "eval",
            eval_command,
            cwd=old_codes_root(),
            log_path=seed_dir / "runner_eval.log",
            status_path=seed_dir / "eval_stage.json",
            dry_run=args.dry_run,
        )

    return {"seed": seed, "ablation": ablation, "status": "completed", "seed_dir": seed_dir, "commands": commands}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train/test/evaluation for 2CKG4ER.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--ablations", default="all", help="Comma-separated formal ablations or all.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--graph-subdir", default=None, help="Optional dataset subdirectory containing graph files, e.g. v10.")
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--ablation-data-root", type=Path, default=default_ablation_data_root())
    parser.add_argument("--top-k-rec", type=int, default=10)
    parser.add_argument("--delta-1", type=float, default=0.8)
    parser.add_argument("--delta-2", type=float, default=0.8)
    parser.add_argument("--forgetting-score-weight", type=float, default=0.0)
    parser.add_argument("--forgetting-exercise-batch-size", type=int, default=256)
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = parse_seed_list(args.seeds)
    args.ablations = parse_ablation_list(args.ablations)
    invalid_ablations = sorted(set(args.ablations) - set(VALID_ABLATIONS))
    if invalid_ablations:
        raise ValueError(f"Unknown ablations: {','.join(invalid_ablations)}")
    args.top_ks = parse_top_ks(args.top_ks)
    if args.forgetting_score_weight < 0:
        raise ValueError("--forgetting-score-weight must be non-negative")
    if args.forgetting_exercise_batch_size <= 0:
        raise ValueError("--forgetting-exercise-batch-size must be positive")
    run_dir = run_dir_for(args)
    base_graph_path = graph_path_for_dataset(args.dataset, args.data_root, graph_subdir=args.graph_subdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    validation = None
    if not args.skip_validation:
        validation = validate_dataset(args.dataset, args.data_root, graph_subdir=args.graph_subdir)
        write_json(run_dir / "semantic_ready_validation.json", validation)
        if validation["status"] != "passed":
            raise RuntimeError(f"Dataset validation failed: {validation['errors']}")

    write_json(
        run_dir / "run_config.json",
        {
            "dataset": args.dataset,
            "run_id": run_dir.name,
            "run_dir": run_dir,
            "base_graph_path": base_graph_path,
            "graph_subdir": args.graph_subdir,
            "ablations": args.ablations,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "bs": args.bs,
            "learning_rate": args.learning_rate,
            "negative_ratio": args.negative_ratio,
            "cuda": args.cuda,
            "deterministic": args.deterministic,
            "resume": args.resume,
            "dry_run": args.dry_run,
            "top_ks": args.top_ks,
            "top_k_rec": args.top_k_rec,
            "delta_1": args.delta_1,
            "delta_2": args.delta_2,
            "forgetting_score_weight": args.forgetting_score_weight,
            "forgetting_exercise_batch_size": args.forgetting_exercise_batch_size,
            "train_triple_files": ["triples.txt"],
            "train_eval_split": "train uses triples.txt only; test_triples.txt is evaluation-only.",
            "model_version": MODEL_VERSION,
            "model": MODEL_NAME,
            "env": python_env_info(),
            "validation": validation,
        },
    )

    statuses = []
    all_commands: List[Dict[str, Any]] = []
    for ablation in args.ablations:
        ablation_graph_path = graph_path_for_ablation(args, base_graph_path, run_dir, ablation)
        for seed in args.seeds:
            try:
                result = run_seed(args, run_dir, ablation_graph_path, base_graph_path, seed, ablation)
                statuses.append(result)
                all_commands.extend(result.get("commands", []))
                write_json(run_dir / "run_status.json", {"status": "running", "runs": statuses})
            except Exception as exc:
                statuses.append(
                    {
                        "seed": seed,
                        "ablation": ablation,
                        "status": "failed",
                        "error": repr(exc),
                        "seed_dir": seed_dir_for(run_dir, ablation, seed),
                    }
                )
                write_json(run_dir / "run_status.json", {"status": "failed", "runs": statuses})
                write_commands(run_dir, all_commands)
                raise

    final_status = "completed" if all(item["status"] in {"completed", "skipped_completed"} for item in statuses) else "partial"
    write_json(run_dir / "run_status.json", {"status": final_status, "runs": statuses})
    write_commands(run_dir, all_commands)
    print(f"2CKG4ER run {final_status}: {run_dir}")


if __name__ == "__main__":
    main()
