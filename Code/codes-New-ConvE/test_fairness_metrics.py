"""Unit checks for resource-side fairness metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fairness_metrics import (
    calculate_fairness_metrics,
    coverage,
    exposure_counts,
    gini,
    head_and_long_tail,
)


class FairnessMetricsTest(unittest.TestCase):
    def test_gini_handles_uniform_and_skewed_exposure(self) -> None:
        self.assertAlmostEqual(gini([1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 3]), 2.0 / 3.0)

    def test_head_and_long_tail_use_popularity_order(self) -> None:
        head, long_tail = head_and_long_tail([10, 4, 0, 1, 2], head_ratio=0.4, long_tail_ratio=0.4)

        self.assertEqual(head, {0, 1})
        self.assertEqual(long_tail, {2, 3})

    def test_head_and_long_tail_do_not_overlap_after_rounding(self) -> None:
        head, long_tail = head_and_long_tail([10, 9, 8, 7], head_ratio=0.5, long_tail_ratio=0.75)

        self.assertEqual(head, {0, 1})
        self.assertEqual(long_tail, {2, 3})
        self.assertFalse(head.intersection(long_tail))

    def test_exposure_counts_fractional_knowledge_concepts(self) -> None:
        q_matrix = [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
        uid_ex_scores = [
            ("uid0", [0.9, 0.1, 0.8]),
            ("uid1", [0.2, 0.7, 0.6]),
        ]

        item_exposure, kc_exposure, recommendation_count = exposure_counts(uid_ex_scores, q_matrix, top_k=2)

        self.assertEqual(item_exposure, [1.0, 1.0, 2.0])
        self.assertEqual(kc_exposure, [2.0, 2.0])
        self.assertEqual(recommendation_count, 4)
        self.assertAlmostEqual(coverage(item_exposure), 1.0)

    def test_calculate_fairness_metrics_from_train_popularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            triples_path = Path(tmp) / "triples.txt"
            triples_path.write_text(
                "\n".join(
                    [
                        "uid0\trec\tex0",
                        "uid1\trec\tex0",
                        "uid2\trec\tex1",
                    ]
                ),
                encoding="utf-8",
            )
            q_matrix = [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
            uid_ex_scores = [
                ("uid0", [0.9, 0.1, 0.8]),
                ("uid1", [0.2, 0.7, 0.6]),
            ]

            metrics = calculate_fairness_metrics(
                uid_ex_scores=uid_ex_scores,
                q_matrix=q_matrix,
                train_triples_path=triples_path,
                top_ks=[1, 2],
                head_ratio=1 / 3,
                long_tail_ratio=1 / 3,
            )

            top_1 = metrics["top_k"]["1"]
            top_2 = metrics["top_k"]["2"]
            self.assertEqual(top_1["recommendations"], 2)
            self.assertEqual(top_2["recommendations"], 4)
            self.assertAlmostEqual(top_1["ItemCoverage"], 2 / 3, places=6)
            self.assertAlmostEqual(top_2["ItemCoverage"], 1.0)
            self.assertIn("LongTailItemExposureShare", top_1)
            self.assertEqual(metrics["definition"]["item_popularity_source"]["relation"], "rec")


if __name__ == "__main__":
    unittest.main()
