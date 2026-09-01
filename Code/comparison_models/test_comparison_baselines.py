import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_recommenders import (  # noqa: E402
    build_all_baseline_scores,
    content_based_scores,
    exercise_based_cf_scores,
    load_cf_protocol,
    student_based_cf_scores,
)


class ComparisonBaselineTests(unittest.TestCase):
    def test_cf_protocol_keeps_test_learners_out_of_library(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "student_split.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["uid", "split"])
                writer.writeheader()
                writer.writerows([{"uid": 0, "split": "train"}, {"uid": 1, "split": "test"}])
            with (root / "interactions_all.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["uid", "question", "response"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"uid": 0, "question": 0, "response": 1},
                        {"uid": 1, "question": 1, "response": 1},
                    ]
                )

            train, query, test_users = load_cf_protocol(root)
            self.assertEqual(set(train), {"uid0"})
            self.assertEqual(set(query), {"uid1"})
            self.assertEqual(test_users, ["uid1"])

    def test_knn_scores_use_query_user_without_mixing_query_into_library(self):
        q_matrix = [[1, 0], [0, 1], [1, 1]]
        train = {"uid0": [(0, 1), (1, 0)], "uid1": [(0, 1), (2, 1)]}
        query = {"uid2": [(0, 1)]}
        mastery = [[0.0, 0.0], [0.0, 0.0], [0.2, 0.8]]

        eb = exercise_based_cf_scores(q_matrix, train, user_ids=["uid2"], query_interactions=query)
        sb = student_based_cf_scores(q_matrix, mastery, train, user_ids=["uid2"], query_interactions=query)
        self.assertEqual(eb[0][0], "uid2")
        self.assertEqual(sb[0][0], "uid2")
        self.assertGreater(len(set(eb[0][1])), 1)
        self.assertGreater(len(set(sb[0][1])), 1)

    def test_rule_baseline_uses_canonical_uid_index(self):
        q_matrix = [[1, 0], [0, 1]]
        mastery = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]
        expected = content_based_scores(q_matrix, mastery[2])
        actual = build_all_baseline_scores(
            q_matrix=q_matrix,
            mastery=mastery,
            sequence=None,
            forgetting=None,
            methods=["CBF"],
            user_ids=["uid2"],
        )["CBF"][0][1]
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
