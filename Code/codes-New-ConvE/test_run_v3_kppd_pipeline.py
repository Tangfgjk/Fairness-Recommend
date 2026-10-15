from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from run_v3_kppd_pipeline import EXPERIMENTS, parse_experiments, train_command


class RunV3KPPDPipelineTest(unittest.TestCase):
    def make_args(self) -> Namespace:
        return Namespace(
            epochs=1,
            bs=16,
            learning_rate=0.001,
            negative_ratio=5,
            cuda="auto",
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
                "pre_item",
                "pre_kc",
                "pre_item_kc",
                "in_kppd",
                "post_only",
                "in_kppd_post",
                "pre_in_kppd_post",
            ],
        )

    def test_old_in_aliases_map_to_kppd_experiments(self) -> None:
        self.assertEqual(parse_experiments("in_only,in_post,pre_in_post"), ["in_kppd", "in_kppd_post", "pre_in_kppd_post"])

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


if __name__ == "__main__":
    unittest.main()
