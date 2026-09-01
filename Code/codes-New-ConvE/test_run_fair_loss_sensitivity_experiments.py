"""Unit checks for stage-4 fair loss sensitivity orchestration."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from run_fair_loss_sensitivity_experiments import (
    build_command,
    generate_experiments,
    parse_experiments,
    run_pipeline,
)


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], check: bool = False):
        self.commands.append(command)
        return None


def make_args(root: Path, **overrides):
    values = {
        "dataset": "Eedi",
        "run_prefix": "stage4_loss",
        "methods": "fairreg_item,fairreg_kc",
        "seeds": "2024",
        "experiments": "random:1,random:10,mixed:50",
        "alpha": 1.0,
        "gamma": 0.25,
        "epochs": 25,
        "bs": 1024,
        "learning_rate": 0.001,
        "negative_ratio": 5,
        "cuda": "auto",
        "data_root": root / "data",
        "runs_root": root / "runs",
        "source_graph_subdir": "er_graph",
        "fair_graph_subdir": "fair_kg_graph",
        "top_ks": "10,20,50",
        "fairness_candidate_size": 150,
        "fairness_temperature": 1.0,
        "python_executable": "python",
        "manifest_file": None,
        "force": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RunFairLossSensitivityExperimentsTest(unittest.TestCase):
    def test_parse_experiments_reads_mode_and_scale_pairs(self) -> None:
        self.assertEqual(parse_experiments("random:1,mixed:50"), [("random", 1.0), ("mixed", 50.0)])

    def test_generate_experiments_names_runs_with_mode_and_scale(self) -> None:
        experiments = generate_experiments("Eedi", "stage4", [("random", 1.0), ("mixed", 50.0)], "seed2024")

        self.assertEqual(experiments[0].run_id, "Eedi_stage4_random_scale001_seed2024")
        self.assertEqual(experiments[1].run_id, "Eedi_stage4_mixed_scale050_seed2024")

    def test_build_command_passes_loss_scale_and_candidate_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root)
            experiment = generate_experiments("Eedi", "stage4", [("mixed", 50.0)], "seed2024")[0]
            command = build_command(args, experiment)

            self.assertEqual(command[command.index("--fairness-loss-scale") + 1], "50.0")
            self.assertEqual(command[command.index("--fairness-candidate-mode") + 1], "mixed")
            self.assertEqual(command[command.index("--fairness-target-gamma") + 1], "0.25")

    def test_run_pipeline_records_dry_run_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, dry_run=True)
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(fake_runner.commands, [])
            self.assertEqual(len(manifest["experiments"]), 3)
            self.assertEqual(manifest["experiments"][0]["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
