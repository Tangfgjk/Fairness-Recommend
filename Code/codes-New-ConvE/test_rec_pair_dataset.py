from __future__ import annotations

import unittest

import numpy as np

from rec_pair_dataset import RecPairDataset


class RecPairDatasetTest(unittest.TestCase):
    def test_negative_sampling_excludes_positives_and_respects_distance_margin(self) -> None:
        triples = [(0, 0, 2), (0, 0, 3)]
        id2relation = {0: "rec"}
        id2entity = {0: "uid0", 1: "kc0", 2: "ex0", 3: "ex1", 4: "ex2"}
        distances = np.array([[0.10, 0.20, 0.80]], dtype=np.float32)
        dataset = RecPairDataset(
            triples,
            id2relation=id2relation,
            id2entity=id2entity,
            exercise_entity_ids=[2, 3, 4],
            rec_relation_id=0,
            distances=distances,
            item_popularity=[5.0, 4.0, 3.0],
            margin=0.05,
            negative_sampling_mode="mixed",
            pair_weighting="distance_gap",
            seed=2024,
        )

        uid, pos, neg, weight = dataset[0]

        self.assertEqual(int(uid), 0)
        self.assertEqual(int(pos), 2)
        self.assertEqual(int(neg), 4)
        self.assertGreater(float(weight), 0.0)

    def test_raises_when_user_has_all_exercises_positive(self) -> None:
        triples = [(0, 0, 1), (0, 0, 2)]
        dataset = RecPairDataset(
            triples,
            id2relation={0: "rec"},
            id2entity={0: "uid0", 1: "ex0", 2: "ex1"},
            exercise_entity_ids=[1, 2],
            rec_relation_id=0,
            distances=None,
            item_popularity=[1.0, 1.0],
            seed=2024,
        )

        with self.assertRaises(ValueError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
