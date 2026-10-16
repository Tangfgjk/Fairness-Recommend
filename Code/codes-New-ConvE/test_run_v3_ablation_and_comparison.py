from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from run_v3_ablation_and_comparison import (
    DEFAULT_COMPARISON_MODELS,
    build_comparison_command,
    build_no_compgcn_command,
    build_v3_command,
    comparison_run_id,
    parse_args,
    planned_stages,
    run_stage,
    v3_batch_id,
)


class RunV3AblationAndComparisonTest(unittest.TestCase):
    def test_dry_run_does_not_mark_unexecuted_stage_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "stage.json"
            self.assertTrue(run_stage("example", ["python", "missing.py"], root, root / "stage.log", status, True))
            self.assertFalse(status.exists())

    def test_defaults_target_three_datasets_and_external_comparison_models(self) -> None:
        args = parse_args([])

        self.assertEqual(args.datasets, ["Eedi", "algebra2005", "XES3G5M-sub-small"])
        self.assertEqual(args.seeds, [2024, 2025, 2026])
        self.assertEqual(DEFAULT_COMPARISON_MODELS, "EB-CF,SB-CF,CBF,TransE,TransE-adv,RotatE,DistMult,ComplEx")
        self.assertEqual(args.comparison_models, DEFAULT_COMPARISON_MODELS)
        self.assertEqual(args.top_ks_list, list(range(5, 51, 5)))
        self.assertEqual(args.backbone, "compgcn")
        self.assertEqual(args.v3_non_kppd_experiments, "baseline,pre_item_kc,post_only,pre_item_kc_post")
        self.assertEqual(args.v3_kppd_experiments, "in_kppd,pre_in_kppd,in_kppd_post,pre_in_kppd_post")

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

    def test_compgcn_settings_reach_v3_pipeline(self) -> None:
        args = parse_args(["--batch-id", "v5_demo", "--backbone", "compgcn", "--compgcn-layers", "2", "--compgcn-op", "mult"])
        command = build_v3_command(args, datasets=["algebra2005"], experiments=["baseline", "in_kppd"])
        self.assertEqual(command[command.index("--backbone") + 1], "compgcn")
        self.assertEqual(command[command.index("--compgcn-layers") + 1], "2")
        self.assertEqual(command[command.index("--compgcn-op") + 1], "mult")

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

    def test_ninth_group_uses_conve_without_e0(self) -> None:
        args = parse_args(["--datasets", "Eedi", "--batch-id", "v5_test", "--dry-run"])
        command = build_no_compgcn_command(args, "Eedi")
        self.assertEqual(command[command.index("--backbone") + 1], "conve")
        self.assertEqual(command[command.index("--experiments") + 1], "pre_in_kppd_post")
        self.assertIn("--no-run-e0", command)
        self.assertIn("v5_test_v3_no_compgcn", command)

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
                "no_compgcn_Eedi",
                "no_compgcn_algebra2005",
            ],
        )
        non_kppd_command = stages[1][1]
        kppd_command = stages[4][1]
        self.assertIn("baseline,pre_item_kc,post_only,pre_item_kc_post", non_kppd_command)
        self.assertNotIn("in_kppd", " ".join(non_kppd_command))
        self.assertIn("in_kppd,pre_in_kppd,in_kppd_post,pre_in_kppd_post", kppd_command)

    def test_ablation_first_places_every_comparison_after_nine_group_stages(self) -> None:
        args = parse_args(["--datasets", "Eedi,algebra2005", "--stage-order", "ablation-first"])
        names = [stage[0] for stage in planned_stages(args)]
        self.assertEqual(names[:6], [
            "v3_non_kppd_Eedi", "v3_non_kppd_algebra2005",
            "v3_kppd_Eedi", "v3_kppd_algebra2005",
            "no_compgcn_Eedi", "no_compgcn_algebra2005",
        ])
        self.assertEqual(names[6:], ["comparison_Eedi", "comparison_algebra2005"])


if __name__ == "__main__":
    unittest.main()
