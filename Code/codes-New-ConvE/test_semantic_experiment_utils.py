"""Tests for 2CKG4ER experiment utilities."""

from __future__ import annotations

import unittest
import tempfile
import sys
from argparse import Namespace
from pathlib import Path

from run_semantic_experiments import command_test, parse_args
from semantic_experiment_utils import (
    DEFAULT_ALL_ABLATIONS,
    MODEL_VERSION,
    ablation_model_dir,
    graph_path_for_dataset,
    parse_ablation_list,
)


class SemanticExperimentUtilsTest(unittest.TestCase):
    def test_graph_path_prefers_prepared_for_formal_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            prepared = data_root / "statics2011" / "prepared_for_kt"
            prepared.mkdir(parents=True)
            (prepared / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")

            graph_path = graph_path_for_dataset("statics2011", data_root)

            self.assertEqual(graph_path.name, "prepared_for_kt")
            self.assertTrue((graph_path / "entities.dict").exists())

    def test_graph_path_uses_eedi_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            eedi = data_root / "Eedi"
            eedi.mkdir(parents=True)
            (eedi / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")

            graph_path = graph_path_for_dataset("Eedi", data_root)

            self.assertEqual(graph_path.name, "Eedi")
            self.assertTrue((graph_path / "entities.dict").exists())

    def test_graph_path_can_target_v8_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            er_v8 = data_root / "Eedi" / "er_v8"
            er_v8.mkdir(parents=True)
            (er_v8 / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")

            graph_path = graph_path_for_dataset("Eedi", data_root, graph_subdir="er_v8")

            self.assertEqual(graph_path, er_v8.resolve())

    def test_graph_path_resolves_relative_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                temp_root = Path(tmp)
                (temp_root / "data" / "Eedi" / "er_v8").mkdir(parents=True)
                (temp_root / "data" / "Eedi" / "er_v8" / "entities.dict").write_text("0\tuid0\n", encoding="utf-8")
                import os

                os.chdir(temp_root)
                graph_path = graph_path_for_dataset("Eedi", Path("data"), graph_subdir="er_v8")

                self.assertTrue(graph_path.is_absolute())
                self.assertEqual(graph_path, (temp_root / "data" / "Eedi" / "er_v8").resolve())
            finally:
                import os

                os.chdir(cwd)

    def test_model_version_marks_2ckg4er_workflow(self) -> None:
        self.assertIn("vfin8", MODEL_VERSION)
        self.assertIn("2ckg4er", MODEL_VERSION)
        self.assertIn("no_sequence", MODEL_VERSION)

    def test_ablation_model_dir_uses_explicit_formal_name(self) -> None:
        self.assertEqual(ablation_model_dir("2CKG4ER"), "2CKG4ER")
        self.assertEqual(ablation_model_dir("2CKG4ER_no_forgetting"), "2CKG4ER_no_forgetting")
        self.assertEqual(ablation_model_dir("feature_only"), "2CKG4ER")

    def test_parse_ablation_list_expands_all_keyword(self) -> None:
        ablations = parse_ablation_list("all")

        self.assertEqual(ablations, DEFAULT_ALL_ABLATIONS)
        self.assertEqual(
            ablations,
            [
                "id_only",
                "2CKG4ER",
                "2CKG4ER_relation_id",
                "2CKG4ER_learner_id",
                "2CKG4ER_exercise_id",
                "2CKG4ER_no_mastery",
                "2CKG4ER_no_forgetting",
            ],
        )

    def test_2ckg4er_cognitive_graph_ablations_are_registered(self) -> None:
        self.assertEqual(
            parse_ablation_list(
                "2CKG4ER_no_mastery,2CKG4ER_no_forgetting"
            ),
            [
                "2CKG4ER_no_mastery",
                "2CKG4ER_no_forgetting",
            ],
        )

    def test_parse_ablation_list_keeps_explicit_order(self) -> None:
        self.assertEqual(
            parse_ablation_list("2CKG4ER,id_only"),
            ["2CKG4ER", "id_only"],
        )

    def test_parse_args_defaults_to_raw_conve_score(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = ["run_semantic_experiments.py", "--dataset", "Eedi"]
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.forgetting_score_weight, 0.0)

    def test_test_command_passes_forgetting_score_weight(self) -> None:
        args = Namespace(dataset="Eedi", cuda="auto", forgetting_score_weight=0.0, forgetting_exercise_batch_size=128)

        command = command_test(args, Path("graph"), Path("seed"), "2CKG4ER")

        self.assertIn("--forgetting-score-weight", command)
        self.assertIn("0.0", command)
        self.assertIn("--forgetting-exercise-batch-size", command)
        self.assertIn("128", command)


if __name__ == "__main__":
    unittest.main()
