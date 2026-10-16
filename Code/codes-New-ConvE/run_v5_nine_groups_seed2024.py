"""Run the V5 nine-group ablation, then five comparison models on three datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from final_dataset import select_dataset
from run_v3_ablation_and_comparison import main as run_all
from run_v3_ablation_and_comparison import parse_args as parse_all_args
from run_v3_ablation_and_comparison import planned_stages
from semantic_experiment_utils import command_to_markdown


DATASETS = ("Eedi", "algebra2005", "XES3G5M-sub-small")
COMPARISON_MODELS = "EB-CF,SB-CF,CBF,TransE,TransE-adv"
DEFAULT_BATCH_ID = "V5_2024"


def build_run_arguments(batch_id: str, cuda: str) -> list[str]:
    return [
        "--datasets", ",".join(DATASETS),
        "--batch-id", batch_id,
        "--seeds", "2024",
        "--backbone", "compgcn",
        "--stage-order", "ablation-first",
        "--comparison-models", COMPARISON_MODELS,
        "--cuda", cuda,
        "--no-run-e0",
        "--resume",
    ]


def validate_data(data_root: Path, batch_id: str) -> None:
    for dataset in DATASETS:
        selection = select_dataset(dataset, data_root)
        message_graph = selection.graph / "educational_kg_v2.npz"
        if not message_graph.is_file():
            raise FileNotFoundError(f"CompGCN message graph is missing: {message_graph}")
        if sys.platform == "win32":
            # Eedi's nested text-feature file exceeds MAX_PATH with a long
            # batch ID. Check both derived Pre graphs before any stage starts.
            for suffix in ("_v3", "_v3_no_compgcn"):
                target = (selection.root / "fair_pre_rec_graph_v3_kppd" /
                          f"{batch_id}{suffix}" / "item_kc_M100_li0p1_lk0p1")
                for source in selection.graph.rglob("*"):
                    if source.is_file():
                        destination = target / source.relative_to(selection.graph)
                        if len(str(destination.resolve())) >= 260:
                            raise ValueError(
                                f"Windows path limit would block the Pre graph copy ({len(str(destination.resolve()))} chars): "
                                f"{destination}. Choose a shorter --batch-id."
                            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--cuda", choices=["auto", "true", "false"], default="true")
    parser.add_argument("--plan", action="store_true", help="Validate data and print all stages without creating run files.")
    args = parser.parse_args(argv)

    run_arguments = build_run_arguments(args.batch_id, args.cuda)
    runner_args = parse_all_args(run_arguments)
    validate_data(runner_args.data_root, args.batch_id)
    if args.plan:
        for name, command, _, _, _ in planned_stages(runner_args):
            print(f"[{name}] {command_to_markdown(command)}")
        return
    run_all(run_arguments)


if __name__ == "__main__":
    main()
