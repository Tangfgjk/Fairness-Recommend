from __future__ import annotations

import unittest

from run_v3_ablation_and_comparison import (
    DEFAULT_COMPARISON_MODELS,
    build_comparison_command,
    build_v3_command,
    comparison_run_id,
    parse_args,
    planned_stages,
    v3_batch_id,
)


class RunV3AblationAndComparisonTest(unittest.TestCase):
    def test_defaults_target_three_datasets_and_external_comparison_models(self) -> None:
        args = parse_args([])

        self.assertEqual(args.datasets, ["Eedi", "algebra2005", "XES3G5M-sub-small"])
        self.assertEqual(args.seeds, [2024])
        self.assertEqual(DEFAULT_COMPARISON_MODELS, "TransE,TransE-adv,EB-CF,SB-CF,CBF")
        self.assertEqual(args.comparison_models, DEFAULT_COMPARISON_MODELS)
        self.assertEqual(args.top_ks_list, [10, 20, 50, 100])

    def test_v3_command_calls_kppd_pipeline_with_requested_ablation_group(self) -> None:
        args = parse_args(["--batch-id", "demo", "--dry-run", "--deterministic"])
        command = build_v3_command(args, datasets=["Eedi"], experiments=["baseline", "post_only"])

        self.assertIn("run_v3_kppd_pipeline.py", " ".join(command))
        self.assertIn("--experiments", command)
        self.assertIn("baseline,post_only", command)
        self.assertIn("Eedi", command)
        self.assertIn("--batch-id", command)
        self.assertIn(v3_batch_id(args), command)
        self.assertIn("--dry-run", command)
        self.assertIn("--deterministic", command)

    def test_comparison_command_excludes_old_conve_ablations(self) -> None:
        args = parse_args(["--batch-id", "demo", "--dry-run"])
        command = build_comparison_command(args, "Eedi")
        joined = " ".join(command)

        self.assertIn("run_v9_comparison_experiments.py", joined)
        self.assertIn("--models", command)
        self.assertIn(DEFAULT_COMPARISON_MODELS, command)
        self.assertNotIn("ConvE_full", joined)
        self.assertIn(comparison_run_id(args, "Eedi"), command)
        self.assertIn("--fairness-popularity-source", command)
        self.assertIn("train_interactions", command)

    def test_planned_stages_run_comparison_then_non_kppd_before_any_kppd(self) -> None:
        args = parse_args(["--datasets", "Eedi,algebra2005", "--batch-id", "demo", "--dry-run"])
        stages = planned_stages(args)

        self.assertEqual(
            [stage[0] for stage in stages],
            [
                "comparison_Eedi",
                "v3_non_kppd_Eedi",
                "comparison_algebra2005",
                "v3_non_kppd_algebra2005",
                "v3_kppd_Eedi",
                "v3_kppd_algebra2005",
            ],
        )
        non_kppd_command = stages[1][1]
        kppd_command = stages[4][1]
        self.assertIn("baseline,pre_item,pre_kc,pre_item_kc,post_only", non_kppd_command)
        self.assertNotIn("in_kppd", " ".join(non_kppd_command))
        self.assertIn("in_kppd,in_kppd_post,pre_in_kppd_post", kppd_command)


if __name__ == "__main__":
    unittest.main()
