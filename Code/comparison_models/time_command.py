import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import update_timing


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command and record elapsed time into timing.json.")
    parser.add_argument("--timing-file", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["training", "inference_without_cache", "evaluation_metric", "other"])
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command was provided.")

    start = time.perf_counter()
    completed = subprocess.run(command, cwd=args.cwd)
    elapsed = time.perf_counter() - start
    update_timing(
        args.timing_file,
        args.stage,
        elapsed,
        extra={
            "returncode": completed.returncode,
            "command": " ".join(shlex.quote(part) for part in command),
        },
    )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
