from __future__ import annotations

import unittest

import torch

from popularity_debias_head import PopularityDebiasConfig, PopularityDebiasHead


class PopularityDebiasHeadTest(unittest.TestCase):
    def build_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "item_popularity_z": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
            "user_profiles": torch.zeros((2, 7), dtype=torch.float32),
            "pedagogical_relevance": torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=torch.float32),
            "exercise_entity_to_index": torch.tensor([-1, -1, 0, 1, 2], dtype=torch.long),
            "uid_entity_to_index": torch.tensor([0, 1, -1, -1, -1], dtype=torch.long),
        }

    def test_global_mode_debiases_global_popularity_term(self) -> None:
        head = PopularityDebiasHead(
            PopularityDebiasConfig(mode="global", beta_global=1.0, lambda_global=1.0),
            self.build_tensors(),
        )
        core = torch.tensor([0.25], dtype=torch.float32)
        h = torch.tensor([0], dtype=torch.long)
        t = torch.tensor([4], dtype=torch.long)

        raw, details = head.combine(core, h, t, score_mode="raw")
        debiased, _ = head.combine(core, h, t, score_mode="debiased")

        self.assertTrue(torch.allclose(raw, core + details["global_term"], atol=1e-6))
        self.assertTrue(torch.allclose(debiased, core, atol=1e-6))

    def test_need_mode_keeps_pedagogical_guardrail_in_debiased_score(self) -> None:
        head = PopularityDebiasHead(
            PopularityDebiasConfig(
                mode="global_personal_need",
                beta_global=1.0,
                beta_personal=0.0,
                beta_need=2.0,
                lambda_global=1.0,
                lambda_personal=1.0,
            ),
            self.build_tensors(),
        )
        core = torch.tensor([0.25], dtype=torch.float32)
        h = torch.tensor([0], dtype=torch.long)
        t = torch.tensor([4], dtype=torch.long)

        debiased, details = head.combine(core, h, t, score_mode="debiased")

        self.assertTrue(torch.allclose(debiased, core + details["need_term"], atol=1e-6))
        self.assertGreater(float(details["need_term"].item()), 0.0)

    def test_unknown_or_out_of_range_ids_return_zero_components(self) -> None:
        head = PopularityDebiasHead(PopularityDebiasConfig(mode="global_personal_need"), self.build_tensors())
        components = head.components(torch.tensor([4], dtype=torch.long), torch.tensor([0], dtype=torch.long))

        self.assertEqual(float(components["global_pop"].item()), 0.0)
        self.assertEqual(float(components["personal_pop"].item()), 0.0)
        self.assertEqual(float(components["need"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
