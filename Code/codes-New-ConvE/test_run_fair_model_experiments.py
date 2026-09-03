from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from run_fair_model_experiments import parse_methods, rerank_command, train_command


class RunFairModelExperimentsTest(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            dataset="Eedi",
            epochs=1,
            bs=16,
            learning_rate=0.001,
            negative_ratio=1,
            seed=2024,
            cuda="cpu",
            deterministic=False,
            resume=False,
            max_train_batches=0,
            max_test_users=0,
            alpha_item=0.05,
            alpha_kc=0.07,
            fairness_loss_scale=10.0,
            fairness_candidate_size=12,
            fairness_candidate_mode="mixed",
            fairness_temperature=0.8,
            fairness_target_gamma=0.5,
            fairness_exposure_proxy="sigmoid_topk",
            fairness_surrogate_k=10,
            fairness_distance="js",
            fairness_top_score_ratio=0.5,
            fairness_popular_ratio=0.25,
            fairness_popularity_source="train_interactions",
            fairness_popularity_aggregation="unique_users",
            rerank_top_k=100,
            rerank_candidate_multiplier=1.5,
            rerank_min_candidates=150,
            rerank_quota_prefixes="10,20",
            rerank_long_tail_ratio_target=0.1,
            rerank_kc_coverage_ratio_target=0.2,
            rerank_lambda_item=0.3,
            rerank_lambda_kc=0.3,
            rerank_beta_item=0.1,
            rerank_beta_kc=0.1,
        )

    def test_parse_methods_all(self) -> None:
        self.assertIn("fairreg_item_kc", parse_methods("all"))
        self.assertEqual(parse_methods("baseline,fairkg_edges"), ["baseline", "fairkg_edges"])

    def test_train_command_adds_fairness_flags_for_joint_regularizer(self) -> None:
        command = train_command(self.args(), Path("graph"), Path("run"), "fairreg_item_kc", 2024)
        self.assertIn("--relation-loss-weights", command)
        self.assertIn("balanced", command)
        self.assertIn("--fairness-loss", command)
        self.assertIn("expected_exposure", command)
        self.assertIn("--fairness-alpha-kc", command)
        self.assertIn("0.07", command)
        self.assertEqual(command[command.index("--fairness-loss-scale") + 1], "10.0")
        self.assertEqual(command[command.index("--fairness-candidate-mode") + 1], "mixed")
        self.assertEqual(command[command.index("--fairness-exposure-proxy") + 1], "sigmoid_topk")
        self.assertEqual(command[command.index("--fairness-distance") + 1], "js")
        self.assertEqual(command[command.index("--fairness-popularity-source") + 1], "train_interactions")

    def test_rerank_command_uses_quota_ratio_hybrid(self) -> None:
        command = rerank_command(self.args(), Path("graph"), Path("scores.pkl"), Path("out.pkl"))
        self.assertIn("--method", command)
        self.assertIn("quota_ratio_hybrid", command)
        self.assertIn("--quota-prefixes", command)
        self.assertIn("10,20", command)


if __name__ == "__main__":
    unittest.main()
