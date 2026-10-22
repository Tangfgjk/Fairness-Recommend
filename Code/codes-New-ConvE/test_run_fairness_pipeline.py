import unittest

from run_fairness_pipeline import parse_experiments, pre_lambdas


class Namespace:
    pre_lambda_item = 0.1
    pre_lambda_kc = 0.2


class RunFairnessPipelineTest(unittest.TestCase):
    def test_requested_group_expands_to_pre_in_post_experiments(self) -> None:
        self.assertEqual(
            parse_experiments("requested"),
            ["pre_baseline_rebuild", "pre_item", "pre_kc", "pre_item_kc", "in_post", "pre_in_post"],
        )

    def test_v3_group_expands_to_full_debias_ablation(self) -> None:
        self.assertEqual(
            parse_experiments("v3_e1"),
            ["baseline", "pre_item", "pre_kc", "pre_item_kc", "in_only", "post_only", "in_post", "pre_in_post"],
        )

    def test_pre_lambdas_are_component_specific(self) -> None:
        args = Namespace()
        self.assertEqual(pre_lambdas(args, "baseline_rebuild"), (0.0, 0.0))
        self.assertEqual(pre_lambdas(args, "item"), (0.1, 0.0))
        self.assertEqual(pre_lambdas(args, "kc"), (0.0, 0.2))
        self.assertEqual(pre_lambdas(args, "item_kc"), (0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
