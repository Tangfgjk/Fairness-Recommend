from __future__ import annotations

import unittest

import torch

from training_objectives import (
    RelationWeightConfig,
    build_relation_weight_tensor,
    canonical_relation_group,
    weighted_bce_loss,
)


class TrainingObjectivesTest(unittest.TestCase):
    def test_relation_groups_include_ex_kc_edges(self) -> None:
        self.assertEqual(canonical_relation_group("rec"), "rec")
        self.assertEqual(canonical_relation_group("mlkc0.50"), "mlkc")
        self.assertEqual(canonical_relation_group("exfr0.50"), "exfr")
        self.assertEqual(canonical_relation_group("ex_has_kc"), "exkc")
        self.assertEqual(canonical_relation_group("kc_has_ex"), "exkc")

    def test_balanced_weights_upweight_sparse_rec_group(self) -> None:
        id2relation = {0: "rec", 1: "exfr0.20", 2: "ex_has_kc"}
        triples = [(0, 0, 1), (1, 1, 0), (2, 1, 0), (3, 1, 0), (1, 2, 4)]
        weights, metadata = build_relation_weight_tensor(
            id2relation,
            triples,
            RelationWeightConfig(mode="balanced"),
        )

        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertEqual(metadata["relation_group_counts"]["rec"], 1)
        self.assertEqual(metadata["relation_group_counts"]["exfr"], 3)

    def test_weighted_bce_loss_returns_scalar(self) -> None:
        predictions = torch.tensor([0.8, 0.2, 0.6])
        labels = torch.tensor([1.0, 0.0, 1.0])
        relation_ids = torch.tensor([0, 1, 0])
        weights = torch.tensor([2.0, 1.0])
        loss = weighted_bce_loss(predictions, labels, relation_ids, weights)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()

