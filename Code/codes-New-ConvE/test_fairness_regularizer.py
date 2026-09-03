from __future__ import annotations

import unittest

import torch

from fairness_regularizer import FairnessRegularizer, FairnessRegularizerConfig


class TinyScoreModel:
    def score_tails_logits(self, h: torch.Tensor, r: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        user_signal = h.float().unsqueeze(1) * 0.01
        tail_signal = tail_ids.float().unsqueeze(0) * 0.001
        return user_signal + tail_signal

    def score_tails(self, h: torch.Tensor, r: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score_tails_logits(h, r, tail_ids))

    def score_tail_matrix_logits(self, h: torch.Tensor, r: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        user_signal = h.float().unsqueeze(1) * 0.01
        tail_signal = tail_ids.float() * 0.001
        return user_signal + tail_signal


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

    def test_top_score_user_candidate_mode_returns_per_user_matrix(self) -> None:
        tensors = {
            "item_popularity": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "item_target_distribution": torch.tensor([0.25, 0.25, 0.25, 0.25]),
            "kc_target_distribution": torch.tensor([0.5, 0.5]),
            "ex_kc_matrix": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(
                loss_type="expected_exposure",
                alpha_item=1.0,
                candidate_size=2,
                candidate_mode="top_score_user",
            ),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12, 13]),
            tensors=tensors,
        )

        candidates = regularizer.select_candidate_indices(TinyScoreModel(), torch.tensor([0, 1, 2]))

        self.assertEqual(tuple(candidates.shape), (3, 2))

    def test_per_user_candidate_exposure_is_scattered_by_item_id(self) -> None:
        tensors = {
            "item_target_distribution": torch.tensor([0.25, 0.25, 0.25, 0.25]),
            "kc_target_distribution": torch.tensor([0.5, 0.5]),
            "ex_kc_matrix": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(loss_type="expected_exposure", alpha_item=1.0),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12, 13]),
            tensors=tensors,
        )
        candidate_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        weights = torch.tensor([[0.25, 0.75], [0.40, 0.60]], dtype=torch.float32)

        exposure, distribution, target = regularizer.item_exposure_distribution(candidate_indices, weights)

        self.assertTrue(torch.allclose(exposure, torch.tensor([0.125, 0.575, 0.300, 0.000]), atol=1e-6))
        self.assertEqual(tuple(distribution.shape), (3,))
        self.assertEqual(tuple(target.shape), (3,))

    def test_sigmoid_topk_exposure_has_k_mass_per_user(self) -> None:
        tensors = {
            "item_target_distribution": torch.tensor([0.25, 0.25, 0.25, 0.25]),
            "kc_target_distribution": torch.tensor([1.0]),
            "ex_kc_matrix": torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        }
        regularizer = FairnessRegularizer(
            FairnessRegularizerConfig(
                loss_type="expected_exposure",
                alpha_item=1.0,
                exposure_proxy="sigmoid_topk",
                surrogate_top_k=2,
            ),
            rec_relation_id=0,
            exercise_entity_ids=torch.tensor([10, 11, 12, 13]),
            tensors=tensors,
        )
        weights = regularizer.exposure_weights(torch.tensor([[4.0, 3.0, 1.0, 0.0], [2.0, 1.0, 0.5, -1.0]]))

        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.tensor([2.0, 2.0]), atol=1e-6))

    def test_distribution_distances_are_finite(self) -> None:
        tensors = {
            "item_target_distribution": torch.tensor([0.5, 0.5]),
            "kc_target_distribution": torch.tensor([1.0]),
            "ex_kc_matrix": torch.tensor([[1.0], [1.0]]),
        }
        for distance in ("mse", "l1", "kl_target_model", "js"):
            regularizer = FairnessRegularizer(
                FairnessRegularizerConfig(
                    loss_type="expected_exposure",
                    alpha_item=1.0,
                    distance=distance,
                ),
                rec_relation_id=0,
                exercise_entity_ids=torch.tensor([10, 11]),
                tensors=tensors,
            )
            loss = regularizer.distribution_loss(torch.tensor([0.99, 0.01]), torch.tensor([0.5, 0.5]))
            self.assertTrue(torch.isfinite(loss), distance)


if __name__ == "__main__":
    unittest.main()
