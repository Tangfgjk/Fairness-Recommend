from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from run_v3_debias_experiments import dataset_run_id, e0_command, pipeline_command


class RunV3DebiasExperimentsTest(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            batch_id="v3_test",
            experiments="v3_e1",
            seeds=[2024],
            epochs=1,
            bs=16,
            learning_rate=0.001,
            negative_ratio=1,
            cuda="cpu",
            data_root=Path("data"),
            runs_root=Path("runs"),
            source_graph_subdir="er_graph",
            top_ks="10,20",
            bias_top_ks="10,20",
            popularity_aggregation="unique_users",
            pre_candidate_pool_size=20,
            pre_lambda_item=0.1,
            pre_lambda_kc=0.2,
            force_preprocess=False,
            fairness_loss_scale=10.0,
            fairness_candidate_size=12,
            fairness_candidate_mode="mixed_user",
            fairness_target_gamma=0.25,
            fairness_exposure_proxy="sigmoid_topk",
            fairness_surrogate_k=10,
            fairness_distance="js",
            rerank_top_k=100,
            rerank_candidate_multiplier=1.5,
            rerank_min_candidates=150,
            rerank_quota_prefixes="10,20",
            rerank_long_tail_ratio_target=0.1,
            rerank_kc_coverage_ratio_target=0.2,
            rerank_lambda_item=0.8,
            rerank_lambda_kc=0.3,
            rerank_beta_item=0.3,
            rerank_beta_kc=0.1,
            deterministic=True,
            resume=False,
            dry_run=False,
            max_train_batches=0,
            max_test_users=0,
        )

    def test_dataset_run_id_is_stable(self) -> None:
        self.assertEqual(dataset_run_id("Eedi", "v3_test"), "Eedi_v3_test")

    def test_pipeline_command_uses_v3_group(self) -> None:
        command = pipeline_command(self.args(), "algebra2005")

        self.assertIn("run_fairness_pipeline.py", command[1])
        self.assertEqual(command[command.index("--experiments") + 1], "v3_e1")
        self.assertEqual(command[command.index("--run-id") + 1], "algebra2005_v3_test")
        self.assertIn("--deterministic", command)

    def test_e0_command_uses_post_only_debiased_scores(self) -> None:
        args = self.args()
        command = e0_command(args, "algebra2005", Path("runs/algebra2005/algebra2005_v3_test"), 2024)

        self.assertIn("bias_chain_diagnostics.py", command[1])
        self.assertEqual(command[command.index("--base-scores-file") + 1], str(Path("runs/algebra2005/algebra2005_v3_test/baseline/seed2024/2CKG4ER_uid_ex_scores.pkl")))
        self.assertEqual(
            command[command.index("--debiased-scores-file") + 1],
            str(Path("runs/algebra2005/algebra2005_v3_test/post_only/seed2024/2CKG4ER_uid_ex_scores_quota_ratio_hybrid.pkl")),
        )


if __name__ == "__main__":
    unittest.main()
