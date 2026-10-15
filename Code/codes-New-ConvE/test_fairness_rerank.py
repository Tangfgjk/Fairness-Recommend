"""Unit checks for fairness reranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fairness_rerank import (
    RerankConfig,
    load_uid_ex_scores,
    main,
    rerank_uid_ex_scores,
    rewrite_scores_by_order,
    save_uid_ex_scores,
)
from fairness_metrics import top_exercises


class FairnessRerankTest(unittest.TestCase):
    def test_rewrite_scores_preserves_requested_order(self) -> None:
        scores = rewrite_scores_by_order([0.1, 0.9, 0.8], [2, 0, 1])

        self.assertEqual(top_exercises(scores, 3), [2, 0, 1])

    def test_item_rerank_can_promote_long_tail_exercise(self) -> None:
        q_matrix = [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
        uid_ex_scores = [("uid0", [1.0, 0.95, 0.94])]
        item_popularity = [10.0, 0.0, 0.0]
        config = RerankConfig(
            method="item",
            top_k=1,
            candidate_multiplier=3.0,
            min_candidates=3,
            lambda_item=1.0,
            lambda_kc=0.0,
            beta_item=0.0,
            beta_kc=0.0,
        )

        reranked, metadata = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)

        self.assertEqual(top_exercises(reranked[0][1], 1), [1])
        self.assertEqual(metadata["changed_users"], 1)

    def test_kc_rerank_improves_within_user_concept_coverage(self) -> None:
        q_matrix = [
            [1, 0],
            [1, 0],
            [0, 1],
        ]
        uid_ex_scores = [("uid0", [1.0, 0.99, 0.98])]
        item_popularity = [1.0, 1.0, 1.0]
        config = RerankConfig(
            method="kc",
            top_k=2,
            candidate_multiplier=2.0,
            min_candidates=3,
            lambda_item=0.0,
            lambda_kc=1.0,
            beta_item=0.0,
            beta_kc=0.0,
        )

        reranked, _ = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)

        self.assertEqual(set(top_exercises(reranked[0][1], 2)), {0, 2})

    def test_quota_strict_enforces_long_tail_and_kc_prefix(self) -> None:
        q_matrix = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        uid_ex_scores = [("uid0", [1.0, 0.99, 0.98, 0.2])]
        item_popularity = [10.0, 9.0, 0.0, 0.0]
        config = RerankConfig(
            method="quota_strict",
            top_k=3,
            candidate_multiplier=1.0,
            min_candidates=4,
            quota_top_k=3,
            min_long_tail_items=1,
            min_kc_coverage=3,
            head_ratio=0.5,
            long_tail_ratio=0.5,
        )

        reranked, metadata = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)
        top_3 = top_exercises(reranked[0][1], 3)

        self.assertEqual(set(top_3), {0, 2, 3})
        self.assertTrue({2, 3}.intersection(top_3))
        self.assertEqual(metadata["quota_satisfied_users"], 1)
        self.assertEqual(metadata["average_quota_long_tail_items"], 2.0)
        self.assertEqual(metadata["average_quota_kc_coverage"], 3.0)
        self.assertEqual(metadata["average_effective_quota_long_tail_target"], 1.0)
        self.assertEqual(metadata["average_effective_quota_kc_target"], 3.0)

    def test_quota_ratio_enforces_multiple_prefixes(self) -> None:
        q_matrix = [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
        ]
        uid_ex_scores = [("uid0", [1.0, 0.99, 0.98, 0.97, 0.1])]
        item_popularity = [10.0, 9.0, 8.0, 0.0, 0.0]
        config = RerankConfig(
            method="quota_ratio",
            top_k=4,
            candidate_multiplier=1.0,
            min_candidates=5,
            quota_prefixes=(2, 4),
            long_tail_item_ratio=0.5,
            kc_coverage_ratio=1.0,
            head_ratio=0.4,
            long_tail_ratio=0.4,
        )

        reranked, metadata = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)
        top_2 = top_exercises(reranked[0][1], 2)
        top_4 = top_exercises(reranked[0][1], 4)

        self.assertEqual(sum(1 for idx in top_2 if idx in {3, 4}), 1)
        self.assertEqual(sum(1 for idx in top_4 if idx in {3, 4}), 2)
        self.assertEqual(len({kc for idx in top_4 for kc, value in enumerate(q_matrix[idx]) if value}), 3)
        self.assertEqual(metadata["quota_satisfied_users"], 1)
        self.assertEqual(metadata["quota_prefix_summary"]["2"]["average_long_tail_items"], 1.0)
        self.assertEqual(metadata["quota_prefix_summary"]["4"]["average_long_tail_items"], 2.0)

    def test_quota_ratio_hybrid_uses_soft_score_after_quota(self) -> None:
        q_matrix = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 1],
        ]
        uid_ex_scores = [("uid0", [1.0, 0.99, 0.98, 0.7, 0.6])]
        item_popularity = [10.0, 9.0, 8.0, 0.0, 0.0]
        config = RerankConfig(
            method="quota_ratio_hybrid",
            top_k=4,
            candidate_multiplier=1.0,
            min_candidates=5,
            quota_prefixes=(2,),
            long_tail_item_ratio=0.5,
            kc_coverage_ratio=0.0,
            lambda_item=2.0,
            lambda_kc=0.0,
            beta_item=0.0,
            beta_kc=0.0,
            head_ratio=0.4,
            long_tail_ratio=0.4,
        )

        reranked, metadata = rerank_uid_ex_scores(uid_ex_scores, q_matrix, item_popularity, config)
        top_4 = top_exercises(reranked[0][1], 4)

        self.assertIn(3, top_4)
        self.assertIn(4, top_4)
        self.assertEqual(metadata["quota_satisfied_users"], 1)

    def test_score_file_round_trip_supports_pickle_and_json(self) -> None:
        rows = [("uid0", [0.1, 0.2])]
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = Path(tmp) / "scores.pkl"
            json_path = Path(tmp) / "scores.json"

            save_uid_ex_scores(rows, pkl_path)
            save_uid_ex_scores(rows, json_path)

            self.assertEqual(load_uid_ex_scores(pkl_path), rows)
            self.assertEqual(load_uid_ex_scores(json_path), rows)

    def test_cli_writes_metadata_file(self) -> None:
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "Q.txt").write_text("1,0\n0,1\n", encoding="utf-8")
            (data_dir / "triples.txt").write_text("uid0\trec\tex0\n", encoding="utf-8")
            scores_file = root / "scores.pkl"
            output_file = root / "reranked.pkl"
            metadata_file = root / "metadata.json"
            save_uid_ex_scores([("uid0", [0.8, 0.7])], scores_file)
            old_argv = sys.argv
            try:
                sys.argv = [
                    "fairness_rerank.py",
                    "--data-dir",
                    str(data_dir),
                    "--scores-file",
                    str(scores_file),
                    "--output-file",
                    str(output_file),
                    "--metadata-file",
                    str(metadata_file),
                    "--method",
                    "baseline",
                    "--top-k",
                    "1",
                    "--popularity-source",
                    "rec_triples",
                ]
                main()
            finally:
                sys.argv = old_argv

            self.assertTrue(output_file.exists())
            self.assertTrue(metadata_file.exists())


if __name__ == "__main__":
    unittest.main()
