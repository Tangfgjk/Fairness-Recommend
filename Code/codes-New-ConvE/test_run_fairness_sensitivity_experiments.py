"""Unit checks for quota_ratio_hybrid sensitivity experiment pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_fairness_sensitivity_experiments import generate_experiments, output_file_for, run_pipeline


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], check: bool = False):
        self.commands.append(command)
        return None


def make_args(root: Path, **overrides):
    values = {
        "data_dir": root / "data",
        "run_dir": root / "run",
        "dataset_name": "Tiny",
        "seed": 2024,
        "scores_file": None,
        "model_prefix": "2CKG4ER",
        "suites": "lt",
        "long_tail_ratios": "0.05,0.10",
        "kc_coverage_ratios": "0.10",
        "candidate_multipliers": "1.0",
        "soft_item_weights": "0.3:0.1,0.5:0.2",
        "soft_kc_weights": "0.3:0.1,0.5:0.2",
        "top_k": 100,
        "top_ks": "10",
        "compare_ks": "10",
        "min_candidates": 150,
        "python_executable": "python",
        "output_dir": None,
        "manifest_file": None,
        "force_rerank": False,
        "force_eval": False,
        "no_compare": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class RunFairnessSensitivityExperimentsTest(unittest.TestCase):
    def test_generate_lt_experiments_names_and_options(self) -> None:
        experiments = generate_experiments(
            suites=["lt"],
            long_tail_ratios=[0.05, 0.1],
            kc_coverage_ratios=[0.2],
            candidate_multipliers=[1.5],
            soft_item_weights=[(0.3, 0.1)],
            soft_kc_weights=[(0.3, 0.1)],
            min_candidates=150,
        )

        self.assertEqual([experiment.name for experiment in experiments], ["quota_ratio_hybrid_lt005", "quota_ratio_hybrid_lt010"])
        self.assertEqual(experiments[0].preset.options["long-tail-item-ratio"], 0.05)
        self.assertEqual(experiments[0].preset.options["kc-coverage-ratio"], 0.2)

    def test_generate_rejects_unknown_suite(self) -> None:
        with self.assertRaises(ValueError):
            generate_experiments(
                suites=["missing"],
                long_tail_ratios=[0.1],
                kc_coverage_ratios=[0.2],
                candidate_multipliers=[1.5],
                soft_item_weights=[(0.3, 0.1)],
                soft_kc_weights=[(0.3, 0.1)],
                min_candidates=150,
            )

    def test_generate_soft_weight_experiments(self) -> None:
        experiments = generate_experiments(
            suites=["soft_item", "soft_kc"],
            long_tail_ratios=[0.1],
            kc_coverage_ratios=[0.2],
            candidate_multipliers=[1.5],
            soft_item_weights=[(0.5, 0.2)],
            soft_kc_weights=[(0.8, 0.3)],
            min_candidates=150,
        )

        self.assertEqual([experiment.name for experiment in experiments], ["soft_item_l050_b020", "soft_kc_l080_b030"])
        self.assertEqual(experiments[0].preset.rerank_method, "item_kc")
        self.assertEqual(experiments[0].preset.options["lambda-item"], 0.5)
        self.assertEqual(experiments[0].preset.options["lambda-kc"], 0.3)
        self.assertEqual(experiments[1].preset.options["lambda-item"], 0.3)
        self.assertEqual(experiments[1].preset.options["lambda-kc"], 0.8)

    def test_run_pipeline_skips_existing_outputs_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, suites="lt", long_tail_ratios="0.05")
            args.data_dir.mkdir(parents=True)
            args.run_dir.mkdir(parents=True)
            (args.run_dir / "2CKG4ER_uid_ex_scores.pkl").write_bytes(b"fake")
            experiment = generate_experiments(["lt"], [0.05], [0.2], [1.5], [(0.3, 0.1)], [(0.3, 0.1)], args.min_candidates)[0]
            output_file_for(args.run_dir, args.model_prefix, experiment).write_bytes(b"fake")
            write_json(args.run_dir / "eval_quota_ratio_hybrid_lt005" / "metrics.json", {})
            write_json(args.run_dir / "eval_quota_ratio_hybrid_lt005" / "fairness_metrics.json", {})
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(len(fake_runner.commands), 1)
            self.assertIn("compare_fairness_results.py", fake_runner.commands[0][1])
            self.assertEqual(manifest["experiments"][0]["rerank_status"], "skipped_existing")
            self.assertEqual(manifest["experiments"][0]["eval_status"], "skipped_existing")
            self.assertTrue((args.run_dir / "fairness_sensitivity_pipeline.json").exists())

    def test_run_pipeline_dry_run_does_not_require_base_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, dry_run=True)
            args.data_dir.mkdir(parents=True)
            args.run_dir.mkdir(parents=True)
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(fake_runner.commands, [])
            self.assertEqual(manifest["experiments"][0]["rerank_status"], "dry_run")
            self.assertFalse((args.run_dir / "fairness_sensitivity_pipeline.json").exists())


if __name__ == "__main__":
    unittest.main()
