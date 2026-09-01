"""Unit checks for SemanticConvE inference helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import test_semantic_conve as inference


class SemanticConvEInferenceHelperTest(unittest.TestCase):
    def test_test_users_can_be_read_without_mlkc_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "test_triples.txt").write_text(
                "\n".join(
                    [
                        "kc0\tmlkc0.50\tuid3",
                        "ex0\texfr0.20\tuid1",
                        "ex1\texfr0.80\tuid3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            users = inference.test_users_from_triples(data_dir)

        self.assertEqual(users, ["uid1", "uid3"])

    def test_read_exfr_relation_names_aligns_by_user_and_exercise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "test_triples.txt").write_text(
                "\n".join(
                    [
                        "kc0\tmlkc0.70\tuid0",
                        "ex0\texfr0.20\tuid0",
                        "ex1\texfr0.80\tuid0",
                        "ex0\texfr0.30\tuid2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            relation_names = inference.read_exfr_relation_names(data_dir, ["uid0", "uid2"], q_count=2)

        self.assertEqual(relation_names["uid0"], ["exfr0.20", "exfr0.80"])
        self.assertEqual(relation_names["uid2"], ["exfr0.30", None])

    def test_non_type_aware_scoring_scores_all_entities_then_selects_exercises(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.tail_ids_seen = []

            def score_tails(self, h, r, tail_ids=None):
                self.tail_ids_seen.append(tail_ids)
                if tail_ids is None:
                    return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
                return torch.tensor([[0.3, 0.5]], dtype=torch.float32)

        model = FakeModel()
        h = torch.tensor([0], dtype=torch.long)
        r = torch.tensor([1], dtype=torch.long)
        exercise_tail_ids = torch.tensor([2, 4], dtype=torch.long)

        scores = inference.score_exercise_candidates(
            model,
            h,
            r,
            exercise_tail_ids,
            type_aware=False,
        )

        self.assertIsNone(model.tail_ids_seen[0])
        self.assertTrue(torch.allclose(scores, torch.tensor([[0.3, 0.5]], dtype=torch.float32)))

    def test_type_aware_scoring_scores_only_exercise_entities(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.tail_ids_seen = []

            def score_tails(self, h, r, tail_ids=None):
                self.tail_ids_seen.append(tail_ids)
                return torch.tensor([[0.3, 0.5]], dtype=torch.float32)

        model = FakeModel()
        h = torch.tensor([0], dtype=torch.long)
        r = torch.tensor([1], dtype=torch.long)
        exercise_tail_ids = torch.tensor([2, 4], dtype=torch.long)

        scores = inference.score_exercise_candidates(
            model,
            h,
            r,
            exercise_tail_ids,
            type_aware=True,
        )

        self.assertTrue(torch.equal(model.tail_ids_seen[0], exercise_tail_ids))
        self.assertTrue(torch.allclose(scores, torch.tensor([[0.3, 0.5]], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
