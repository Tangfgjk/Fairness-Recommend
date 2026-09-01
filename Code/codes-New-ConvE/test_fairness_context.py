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


if __name__ == "__main__":
    unittest.main()
