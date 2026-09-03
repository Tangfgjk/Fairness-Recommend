"""Unit checks for stage-2 fair model sensitivity orchestration."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from run_fair_model_sensitivity_experiments import build_command, fairness_config_tag, generate_experiments, run_pipeline


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], check: bool = False):
        self.commands.append(command)
        return None


def make_args(root: Path, **overrides):
    values = {
        "dataset": "Eedi",
        "run_prefix": "stage2_sens",
        "methods": "fairreg_item,fairreg_kc",
        "seeds": "2024",
        "alphas": "0.5,1.0",
        "gammas": "0.5,0.25",
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
        "fairness_loss_scale": 1.0,
        "fairness_candidate_size": 150,
        "fairness_candidate_mode": "random",
        "fairness_temperature": 1.0,
        "fairness_exposure_proxy": "softmax",
        "fairness_surrogate_k": 10,
        "fairness_distance": "mse",
        "fairness_top_score_ratio": 0.5,
        "fairness_popular_ratio": 0.25,
        "fairness_popularity_source": "rec_triples",
        "fairness_popularity_aggregation": "unique_users",
        "python_executable": "python",
        "manifest_file": None,
        "force": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RunFairModelSensitivityExperimentsTest(unittest.TestCase):
    def test_generate_experiments_crosses_alpha_and_gamma(self) -> None:
        experiments = generate_experiments("Eedi", "stage2", [0.5, 1.0], [0.5, 0.25], "seed2024")

        self.assertEqual(len(experiments), 4)
        self.assertEqual(experiments[0].run_id, "Eedi_stage2_alpha05_gamma05_seed2024")
        self.assertEqual(experiments[-1].run_id, "Eedi_stage2_alpha10_gamma025_seed2024")

    def test_build_command_passes_alpha_and_gamma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root)
            experiment = generate_experiments("Eedi", "stage2", [0.5], [0.25], "seed2024")[0]
            command = build_command(args, experiment)

            self.assertIn("--alpha-item", command)
            self.assertEqual(command[command.index("--alpha-item") + 1], "0.5")
            self.assertEqual(command[command.index("--alpha-kc") + 1], "0.5")
            self.assertEqual(command[command.index("--fairness-target-gamma") + 1], "0.25")
            self.assertEqual(command[command.index("--fairness-loss-scale") + 1], "1.0")
            self.assertEqual(command[command.index("--fairness-candidate-mode") + 1], "random")
            self.assertEqual(command[command.index("--fairness-exposure-proxy") + 1], "softmax")
            self.assertEqual(command[command.index("--fairness-distance") + 1], "mse")

    def test_run_pipeline_records_dry_run_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, dry_run=True)
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(fake_runner.commands, [])
            self.assertEqual(len(manifest["experiments"]), 4)
            self.assertEqual(manifest["experiments"][0]["status"], "dry_run")
            self.assertIn("proxy-softmax", manifest["experiments"][0]["run_id"])
            self.assertIn("pop-rec_triples-unique_users", manifest["experiments"][0]["run_id"])

    def test_fairness_config_tag_changes_with_non_sweep_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            original = fairness_config_tag(args)
            args.fairness_distance = "js"
            self.assertNotEqual(original, fairness_config_tag(args))


if __name__ == "__main__":
    unittest.main()
