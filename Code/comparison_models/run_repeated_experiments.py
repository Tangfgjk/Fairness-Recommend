import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from experiment_utils import make_run_id, summarize_seed_metrics, write_json


DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command template for multiple random seeds.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--metrics-file-name", default="metrics.json")
    parser.add_argument("--summary-file-name", default="seed_summary.json")
    parser.add_argument(
        "--command-template",
        required=True,
        help="Command template. Available fields: {seed}, {run_dir}, {dataset}, {model}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    metric_files = []
    run_records = []

    for seed in seeds:
        run_id = make_run_id(args.dataset, args.model, seed=seed)
        run_dir = args.run_root / args.dataset / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command_text = args.command_template.format(
            seed=seed,
            run_dir=str(run_dir),
            dataset=args.dataset,
            model=args.model,
        )
        command = shlex.split(command_text)
        record = {"seed": seed, "run_dir": str(run_dir), "command": command_text}
        run_records.append(record)
        write_json(record, run_dir / "run_command.json")
        print(command_text)
        if not args.dry_run:
            completed = subprocess.run(command)
            record["returncode"] = completed.returncode
            if completed.returncode != 0:
                write_json({"runs": run_records}, args.run_root / args.dataset / f"{args.model}_failed_runs.json")
                return completed.returncode
        metric_files.append(run_dir / args.metrics_file_name)

    summary_path = args.run_root / args.dataset / f"{args.model}_{args.summary_file_name}"
    summarize_seed_metrics(metric_files, summary_path)
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
