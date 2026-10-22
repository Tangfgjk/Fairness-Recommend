"""Run V6 backbone/objective controls before the nine-system ablation.

Plan mode only validates inputs and prints commands. Training is explicit.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from final_dataset import DEFAULT_DATA_ROOT, select_dataset
from semantic_experiment_utils import code_dir, default_runs_root, command_to_markdown, parse_csv_list, parse_seed_list


def planned_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    controls = [
        ("conve_reference", "conve", "direct", "c", "baseline"),
        ("graph_student_reference", "compgcn", "graph", "g", "baseline"),
        ("direct_student_controls", "compgcn", "direct", "d",
         "baseline,bpr_only,bpr_matched,in_kppd,in_kppd_no_decorr"),
    ]
    main = [
        ("direct_eight_groups", "compgcn", "direct", "d", "main"),
        ("conve_full_system", "conve", "direct", "c", "pre_in_kppd_post"),
    ]
    stages = controls if args.phase == "controls" else main if args.phase == "main" else controls + main
    commands = []
    for dataset in parse_csv_list(args.datasets):
        selection = select_dataset(dataset, args.data_root)
        if not (selection.graph / "educational_kg_v2.npz").is_file():
            raise FileNotFoundError(f"Missing message graph: {selection.graph}")
        # Validate the longest derived Pre feature path before launching a job.
        if sys.platform == "win32" and args.phase != "controls":
            for tag in ("c", "d"):
                pre = selection.root / "fair_pre_rec_graph_v3_kppd" / f"{args.batch_id}_{tag}" / "item_kc_M100_li0p1_lk0p1"
                for source in selection.graph.rglob("*"):
                    if source.is_file() and len(str(pre / source.relative_to(selection.graph))) >= 260:
                        raise ValueError("Pre graph path exceeds Windows MAX_PATH; shorten --batch-id")
        for label, backbone, mode, tag, experiments in stages:
            cmd = [sys.executable, str(code_dir() / "run_v3_kppd_pipeline.py"),
                   "--datasets", dataset, "--experiments", experiments,
                   "--batch-id", f"{args.batch_id}_{tag}", "--seeds", args.seeds,
                   "--backbone", backbone, "--compgcn-student-mode", mode,
                   "--compgcn-layers", str(args.compgcn_layers), "--compgcn-op", args.compgcn_op,
                   "--compgcn-dropout", str(args.compgcn_dropout), "--kppd-mode", args.kppd_mode,
                   "--rerank-method", "quality_swap", "--data-root", str(args.data_root),
                   "--runs-root", str(args.runs_root), "--cuda", args.cuda,
                   "--epochs", str(args.epochs), "--ep-thresholds", "0.5",
                   "--no-run-e0", "--save-kppd-components", "--audit-personalization"]
            if args.resume:
                cmd.append("--resume")
            if args.max_train_batches:
                cmd.extend(["--max-train-batches", str(args.max_train_batches)])
            commands.append((f"{dataset}/{label}", cmd))
        if args.include_comparisons and args.phase in {"main", "all"}:
            from run_v3_ablation_and_comparison import build_comparison_command, parse_args as unified_args
            options = ["--datasets", dataset, "--batch-id", args.batch_id,
                       "--seeds", args.seeds, "--data-root", str(args.data_root),
                       "--runs-root", str(args.runs_root), "--cuda", args.cuda,
                       "--comparison-models", args.comparison_models]
            if args.resume:
                options.append("--resume")
            commands.append((f"{dataset}/comparisons", build_comparison_command(unified_args(options), dataset)))
    return commands


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="Eedi,algebra2005,XES3G5M-sub-small")
    parser.add_argument("--phase", choices=["controls", "main", "all"], default="controls")
    parser.add_argument("--batch-id", default="V6_2024")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--cuda", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--compgcn-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--compgcn-op", choices=["corr", "sub", "mult"], default="corr")
    parser.add_argument("--compgcn-dropout", type=float, default=0.1)
    parser.add_argument("--kppd-mode", choices=["global", "global_personal", "global_personal_need"], default="global_personal_need")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--include-comparisons", action="store_true")
    parser.add_argument("--comparison-models", default="EB-CF,SB-CF,CBF,TransE,TransE-adv")
    parser.add_argument("--max-train-batches", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.batch_id.startswith("V6_") or len(args.batch_id) > 24:
        parser.error("--batch-id must start with V6_ and be at most 24 characters")
    if not parse_csv_list(args.datasets) or not parse_seed_list(args.seeds) or args.epochs <= 0:
        parser.error("datasets/seeds must not be empty and epochs must be positive")
    if args.max_train_batches < 0:
        parser.error("smoke limits must be nonnegative")
    if not 0 <= args.compgcn_dropout < 1:
        parser.error("compgcn dropout must lie in [0, 1)")
    args.data_root = args.data_root.resolve()
    args.runs_root = args.runs_root.resolve()
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    commands = planned_commands(args)
    for name, command in commands:
        print(f"[{name}] {command_to_markdown(command)}", flush=True)
        if not args.plan:
            subprocess.run(command, cwd=code_dir(), check=True)


if __name__ == "__main__":
    main()
