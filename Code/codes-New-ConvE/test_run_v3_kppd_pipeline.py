from __future__ import annotations

import unittest
import tempfile
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from run_v3_kppd_pipeline import EXPERIMENTS, aggregate, parse_args, parse_experiments, run_dataset, run_post_experiment, train_command


class RunV3KPPDPipelineTest(unittest.TestCase):
    def make_args(self) -> Namespace:
        return Namespace(
            epochs=1,
            bs=16,
            learning_rate=0.001,
            negative_ratio=5,
            cuda="auto",
            backbone="conve",
            compgcn_layers=1,
            compgcn_op="corr",
            compgcn_dropout=0.1,
            popularity_aggregation="unique_users",
            bpr_weight=1.0,
            bpr_distance_margin=0.01,
            bpr_pair_weighting="distance_gap",
            beta_global_pop=1.0,
            beta_personal_pop=1.0,
            beta_need=1.0,
            lambda_global_pop=1.0,
            lambda_personal_pop=1.0,
            global_pop_aux_weight=0.1,
            personal_pop_aux_weight=0.1,
            need_aux_weight=0.0,
            decorr_weight=0.01,
            deterministic=True,
            resume=False,
            max_train_batches=1,
        )

    def test_main_group_expands_to_v3_pre_kppd_post_experiments(self) -> None:
        self.assertEqual(
            parse_experiments("main"),
            [
                "baseline",
                "pre_item_kc",
                "in_kppd",
                "pre_in_kppd",
                "post_only",
                "pre_item_kc_post",
                "in_kppd_post",
                "pre_in_kppd_post",
            ],
        )

    def test_old_in_aliases_map_to_kppd_experiments(self) -> None:
        self.assertEqual(parse_experiments("in_only,in_post,pre_in_post"),
                         ["in_kppd", "in_kppd_post", "pre_in_kppd", "pre_in_kppd_post"])

    def test_post_group_adds_its_training_parent(self) -> None:
        self.assertEqual(parse_experiments("pre_item_kc_post"), ["pre_item_kc", "pre_item_kc_post"])
        self.assertEqual(parse_experiments("pre_in_kppd_post"), ["pre_in_kppd", "pre_in_kppd_post"])

    def test_post_group_reuses_parent_scores_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "pre_in_kppd" / "seed2024"
            parent.mkdir(parents=True)
            (parent / "2CKG4ER_uid_ex_scores_debiased.pkl").write_bytes(b"scores")
            with patch("sys.argv", ["run_v3_kppd_pipeline.py", "--data-root", str(root), "--no-run-e0"]):
                args = parse_args()
            with patch("run_v3_kppd_pipeline.run_stage") as stage:
                status = run_post_experiment(args, "algebra2005", root / "graph", root,
                                             EXPERIMENTS["pre_in_kppd_post"], 2024)
            self.assertEqual(status["reused_from"], "pre_in_kppd")
            self.assertEqual(stage.call_count, 2)
            self.assertEqual([call.args[0] for call in stage.call_args_list], ["post", "eval"])

    def test_kppd_train_command_disables_old_fairness_regularizer(self) -> None:
        command = train_command(
            self.make_args(),
            "Eedi",
            Path("graph"),
            Path("run") / "in_kppd" / "seed2024",
            EXPERIMENTS["in_kppd"],
            2024,
        )

        self.assertIn("--popularity-debias", command)
        self.assertIn("global_personal_need", command)
        self.assertIn("--tail-bias-mode", command)
        self.assertIn("pop_branch", command)
        self.assertIn("--fairness-loss", command)
        self.assertIn("none", command)
        self.assertNotIn("expected_exposure", command)

    def test_aggregate_keeps_earlier_phases_when_invoked_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(runs_root=root, batch_id="batch")
            run_dir = root / "algebra2005" / "algebra2005_batch"
            baseline_eval = run_dir / "baseline" / "seed2024" / "eval"
            baseline_eval.mkdir(parents=True)
            (baseline_eval / "metrics.csv").write_text("metric,top_k,mean,std\nAda,5,0.1,0\n", encoding="utf-8")
            summary = root / "batch"
            aggregate(args, summary)
            kppd_eval = run_dir / "in_kppd" / "seed2024" / "eval_debiased"
            kppd_eval.mkdir(parents=True)
            (kppd_eval / "metrics.csv").write_text("metric,top_k,mean,std\nAda,5,0.2,0\n", encoding="utf-8")
            aggregate(args, summary)
            content = (summary / "V3_main_metrics_all.csv").read_text(encoding="utf-8-sig")
            self.assertIn("baseline", content)
            self.assertIn("in_kppd", content)

    def test_run_status_keeps_earlier_phase_when_invoked_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(runs_root=root, data_root=root, source_graph_subdir="er_graph", batch_id="batch", experiments=["baseline"],
                             seeds=[2024], run_e0=False, dry_run=False)
            def fake_run(_args, dataset, _graph_dir, _run_dir, experiment, seed):
                return {"dataset": dataset, "experiment": experiment.name, "seed": seed}
            with patch("run_v3_kppd_pipeline.graph_path_for_dataset", return_value=root / "graph"), \
                 patch("run_v3_kppd_pipeline.run_model_experiment", side_effect=fake_run):
                run_dataset(args, "algebra2005")
                args.experiments = ["in_kppd"]
                run_dataset(args, "algebra2005")
            status_path = root / "algebra2005" / "algebra2005_batch" / "run_status.json"
            names = [item["experiment"] for item in json.loads(status_path.read_text(encoding="utf-8"))["runs"]]
            self.assertEqual(names, ["baseline", "in_kppd"])


if __name__ == "__main__":
    unittest.main()
