"""Run the original 2CKG4ER ConvE baseline on the final three datasets.

This control uses the final (0.8, 0.8, 50) drop-zero graphs and has no
Pre, KPPD, or Post debiasing. It is separate from the ConvE full-system
ablation, which retains all three debiasing stages.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Sequence

from final_dataset import select_dataset
from run_v3_ablation_and_comparison import (
    build_original_conve_command,
    parse_args as parse_all_args,
)
from semantic_experiment_utils import code_dir, command_to_markdown, parse_csv_list


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="Eedi,algebra2005,XES3G5M-sub-small")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--batch-id", default="V5_2024")
    parser.add_argument("--cuda", choices=["auto", "true", "false"], default="true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)

    runner = parse_all_args([
        "--datasets", args.datasets,
        "--seeds", args.seeds,
        "--batch-id", args.batch_id,
        "--cuda", args.cuda,
        "--resume",
        "--no-run-e0",
    ])
    for dataset in parse_csv_list(args.datasets):
        selection = select_dataset(dataset, runner.data_root)
        if not (selection.graph / "Q.txt").is_file():
            raise FileNotFoundError(f"Recommendation graph is incomplete: {selection.graph}")
        command = build_original_conve_command(runner, dataset)
        print(f"[{dataset}] {command_to_markdown(command)}", flush=True)
        if not args.plan:
            subprocess.run(command, cwd=code_dir(), check=True)


if __name__ == "__main__":
    main()
