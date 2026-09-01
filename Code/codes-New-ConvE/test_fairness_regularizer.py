from __future__ import annotations

import unittest

import torch

from fairness_regularizer import FairnessRegularizer, FairnessRegularizerConfig


class TinyScoreModel:
    def score_tails(self, h: torch.Tensor, r: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        user_signal = h.float().unsqueeze(1) * 0.01
        tail_signal = tail_ids.float().unsqueeze(0) * 0.001
        return torch.sigmoid(user_signal + tail_signal)


class FairnessRegularizerTest(unittest.TestCase):
    def test_disabled_regularizer_returns_zero(self) -> None:
        tensors = {
            "item_target_distribution": torch.tensor([0.5, 0.5]),
            "kc_target_distribution": torch.tensor([1.0]),
            "ex_kc_matrix": torch.tensor([[1.0], [1.0]]),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(loss_type="none"),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11]),
            tensors=tensors,
        )
        loss, details = regularizer(TinyScoreModel(), torch.tensor([0, 1]))
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(details["fair_loss"], 0.0)

    def test_expected_exposure_regularizer_returns_positive_loss(self) -> None:
        tensors = {
            "item_target_distribution": torch.tensor([0.8, 0.1, 0.1]),
            "kc_target_distribution": torch.tensor([0.5, 0.5]),
            "ex_kc_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(
                loss_type="expected_exposure",
                alpha_item=0.5,
                alpha_kc=0.5,
                candidate_size=3,
            ),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12]),
            tensors=tensors,
        )
        torch.manual_seed(1)
        loss, details = regularizer(TinyScoreModel(), torch.tensor([0, 1, 1]))
        self.assertGreaterEqual(float(loss.item()), 0.0)
        self.assertIn("item_fair_loss", details)
        self.assertIn("kc_fair_loss", details)

    def test_loss_scale_multiplies_fairness_loss(self) -> None:
        tensors = {
            "item_popularity": torch.tensor([3.0, 2.0, 1.0]),
            "item_target_distribution": torch.tensor([0.8, 0.1, 0.1]),
            "kc_target_distribution": torch.tensor([0.5, 0.5]),
            "ex_kc_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        }
        base_config = dict(
            loss_type="expected_exposure",
            alpha_item=0.5,
            alpha_kc=0.5,
            candidate_size=3,
        )
        regularizer_1 = FairnessRegularizer(
            FairnessRegularizerConfig(**base_config, loss_scale=1.0),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12]),
            tensors=tensors,
        )
        regularizer_10 = FairnessRegularizer(
            FairnessRegularizerConfig(**base_config, loss_scale=10.0),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12]),
            tensors=tensors,
        )

        torch.manual_seed(1)
        loss_1, _ = regularizer_1(TinyScoreModel(), torch.tensor([0, 1, 1]))
        torch.manual_seed(1)
        loss_10, details = regularizer_10(TinyScoreModel(), torch.tensor([0, 1, 1]))

        self.assertAlmostEqual(float(loss_10.item()), float(loss_1.item()) * 10.0, places=6)
        self.assertEqual(details["fairness_loss_scale"], 10.0)

    def test_mixed_candidate_mode_returns_requested_candidate_count(self) -> None:
        tensors = {
            "item_popularity": torch.tensor([9.0, 8.0, 1.0, 0.0, 0.0]),
            "item_target_distribution": torch.tensor([0.4, 0.3, 0.1, 0.1, 0.1]),
            "kc_target_distribution": torch.tensor([0.5, 0.5]),
            "ex_kc_matrix": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
            ),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(
                loss_type="expected_exposure",
                alpha_item=1.0,
                alpha_kc=1.0,
                candidate_size=4,
                candidate_mode="mixed",
            ),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12, 13, 14]),
            tensors=tensors,
        )

        torch.manual_seed(1)
        candidates = regularizer.select_candidate_indices(TinyScoreModel(), torch.tensor([0, 1, 1]))

        self.assertEqual(int(candidates.numel()), 4)
        self.assertEqual(len(set(candidates.detach().cpu().tolist())), 4)


if __name__ == "__main__":
    unittest.main()
