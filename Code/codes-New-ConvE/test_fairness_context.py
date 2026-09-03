from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fairness_context import build_fairness_context, context_stats, read_q_matrix


class FairnessContextTest(unittest.TestCase):
    def test_build_context_from_q_and_train_triples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "Q.txt").write_text("1,0\n1,1\n0,1\n", encoding="utf-8")
            (data_dir / "triples.txt").write_text(
                "\n".join(
                    [
                        "uid0\trec\tex0",
                        "uid1\trec\tex0",
                        "uid0\trec\tex1",
                        "ex2\texfr0.10\tuid0",
                    ]
                ),
                encoding="utf-8",
            )

            context = build_fairness_context(data_dir, head_ratio=0.34, long_tail_ratio=0.34, target_gamma=0.5)

            self.assertEqual(context.exercise_to_kc[0], [0])
            self.assertEqual(context.exercise_to_kc[1], [0, 1])
            self.assertEqual(context.item_popularity, [2.0, 1.0, 0.0])
            self.assertEqual(len(context.kc_popularity), 2)
            self.assertEqual(context.ex_kc_matrix.shape, (3, 2))
            self.assertAlmostEqual(float(context.ex_kc_matrix[1].sum()), 1.0)
            self.assertAlmostEqual(sum(context.item_target_distribution), 1.0)
            self.assertEqual(context_stats(context)["q_nonzero_edges"], 4)

    def test_read_q_matrix_accepts_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Q.txt"
            path.write_text("1\t0\n0\t1\n", encoding="utf-8")
            self.assertEqual(read_q_matrix(path), [[1, 0], [0, 1]])

    def test_build_context_can_use_train_interaction_unique_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "er_graph"
            raw_dir = root / "raw"
            data_dir.mkdir()
            raw_dir.mkdir()
            (data_dir / "Q.txt").write_text("1,0\n1,1\n0,1\n", encoding="utf-8")
            (data_dir / "triples.txt").write_text("uid0\trec\tex2\n", encoding="utf-8")
            (raw_dir / "interactions_all.csv").write_text(
                "\n".join(
                    [
                        "uid,entity_id,question,entity_exercise_id,split",
                        "0,uid0,0,ex0,train",
                        "0,uid0,0,ex0,train",
                        "1,uid1,0,ex0,train",
                        "2,uid2,1,ex1,test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            context = build_fairness_context(
                data_dir,
                popularity_source="train_interactions",
                popularity_aggregation="unique_users",
            )

            self.assertEqual(context.item_popularity, [2.0, 0.0, 0.0])
            self.assertEqual(context.metadata["item_popularity_source"]["source"], "train_interactions")
            self.assertEqual(context_stats(context)["item_popularity_source"]["aggregation"], "unique_users")


if __name__ == "__main__":
    unittest.main()
