from __future__ import annotations

import unittest

from run_v3_ablation_and_comparison import parse_args, planned_stages
from final_dataset import DEFAULT_DATA_ROOT
from run_v5_nine_groups_seed2024 import (
    COMPARISON_MODELS,
    DATASETS,
    DEFAULT_BATCH_ID,
    build_run_arguments,
    validate_data,
)


class RunV5NineGroupsSeed2024Test(unittest.TestCase):
    def test_default_batch_can_copy_eedi_pre_graph_on_windows(self) -> None:
        validate_data(DEFAULT_DATA_ROOT, DEFAULT_BATCH_ID)

    def test_long_batch_id_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "Windows path limit"):
            validate_data(DEFAULT_DATA_ROOT, "V5_nine_groups_seed2024_20261016")

    def test_one_click_plan_has_three_datasets_one_seed_and_requested_models(self) -> None:
        args = parse_args(build_run_arguments("v5_test", "true"))
        stages = planned_stages(args)
        names = [stage[0] for stage in stages]

        self.assertEqual(args.datasets, list(DATASETS))
        self.assertEqual(args.seeds, [2024])
        self.assertEqual(args.comparison_models, COMPARISON_MODELS)
        self.assertEqual(args.backbone, "compgcn")
        self.assertTrue(args.no_run_e0)
        self.assertTrue(args.resume)
        self.assertEqual(args.max_train_batches, 0)
        self.assertEqual(args.max_test_users, 0)
        self.assertEqual(len(names), 12)
        self.assertTrue(all(not name.startswith("comparison_") for name in names[:9]))
        self.assertEqual(names[9:], [f"comparison_{name}" for name in DATASETS])


if __name__ == "__main__":
    unittest.main()
