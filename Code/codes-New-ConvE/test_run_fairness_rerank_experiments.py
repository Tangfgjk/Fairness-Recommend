"""Unit checks for the fairness rerank experiment pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_fairness_rerank_experiments import (
    PRESETS,
    build_rerank_command,
    output_file_for,
    parse_method_list,
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
        "data_dir": root / "data",
        "run_dir": root / "run",
        "dataset_name": "Tiny",
        "seed": 2024,
        "scores_file": None,
        "model_prefix": "2CKG4ER",
        "methods": "item_only,kc_only",
        "top_k": 100,
        "top_ks": "10",
        "compare_ks": "10",
        "candidate_multiplier": 1.5,
        "min_candidates": 150,
        "python_executable": "python",
        "popularity_source": "train_interactions",
        "popularity_aggregation": "unique_users",
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


class RunFairnessRerankExperimentsTest(unittest.TestCase):
    def test_parse_method_list_validates_names(self) -> None:
        self.assertEqual(parse_method_list("item_only,kc_only"), ["item_only", "kc_only"])
        with self.assertRaises(ValueError):
            parse_method_list("missing_method")

    def test_build_rerank_command_uses_preset_options(self) -> None:
        command = build_rerank_command(
            python_executable="python",
            data_dir=Path("data"),
            base_scores_file=Path("scores.pkl"),
            output_file=Path("out.pkl"),
            preset=PRESETS["quota_ratio_hybrid"],
            top_k=100,
            candidate_multiplier=1.5,
            min_candidates=150,
            popularity_source="train_interactions",
            popularity_aggregation="unique_users",
        )

        self.assertIn("--method", command)
        self.assertIn("quota_ratio_hybrid", command)
        self.assertIn("--long-tail-item-ratio", command)
        self.assertIn("0.1", command)
        self.assertIn("--lambda-item", command)
        self.assertIn("0.8", command)
        self.assertEqual(command[command.index("--popularity-source") + 1], "train_interactions")

    def test_run_pipeline_skips_existing_outputs_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, methods="item_only")
            args.data_dir.mkdir(parents=True)
            args.run_dir.mkdir(parents=True)
            base_scores = args.run_dir / "2CKG4ER_uid_ex_scores.pkl"
            base_scores.write_bytes(b"fake")
            output_file = output_file_for(args.run_dir, args.model_prefix, PRESETS["item_only"])
            output_file.write_bytes(b"fake")
            write_json(args.run_dir / "eval_item_only" / "metrics.json", {})
            write_json(args.run_dir / "eval_item_only" / "fairness_metrics.json", {})
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(len(fake_runner.commands), 1)
            self.assertIn("compare_fairness_results.py", fake_runner.commands[0][1])
            self.assertEqual(manifest["methods"][0]["rerank_status"], "skipped_existing")
            self.assertEqual(manifest["methods"][0]["eval_status"], "skipped_existing")
            self.assertTrue((args.run_dir / "fairness_rerank_pipeline.json").exists())

    def test_run_pipeline_dry_run_does_not_require_base_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, methods="item_only", dry_run=True)
            args.data_dir.mkdir(parents=True)
            args.run_dir.mkdir(parents=True)
            fake_runner = FakeRunner()

            manifest = run_pipeline(args, runner=fake_runner)

            self.assertEqual(fake_runner.commands, [])
            self.assertEqual(manifest["methods"][0]["rerank_status"], "dry_run")
            self.assertFalse((args.run_dir / "fairness_rerank_pipeline.json").exists())


if __name__ == "__main__":
    unittest.main()
