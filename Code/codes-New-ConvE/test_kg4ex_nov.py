from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kg4ex_nov import calculate_kg4ex_nov, calculate_kg4ex_nov_set, read_correct_history


class KG4ExNoveltyTest(unittest.TestCase):
    def test_averages_per_exercise_then_per_test_student(self) -> None:
        q_matrix = [[1, 0], [0, 1], [1, 1]]
        history = {"uid1": frozenset({0}), "uid2": frozenset()}
        scores = [("uid1", [0.9, 0.8, 0.7]), ("uid2", [0.9, 0.8, 0.7])]

        result = calculate_kg4ex_nov(scores, q_matrix, history, [1, 2, 3])

        self.assertEqual(result["1"]["mean"], 0.5)
        self.assertEqual(result["2"]["mean"], 0.75)
        self.assertEqual(result["3"]["mean"], 0.75)
        self.assertEqual(result["1"]["std"], 0.707107)

    def test_history_uses_only_correct_test_responses(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / "tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            raw = Path(tmp)
            (raw / "interactions_all.csv").write_text(
                "entity_id,question,response,split\n"
                "uid1,0,1,test\n"
                "uid1,1,0,test\n"
                "uid1,2,1,train\n"
                "uid2,1,0,test\n", encoding="utf-8"
            )
            with patch("kg4ex_nov.select_dataset", return_value=SimpleNamespace(raw=raw)):
                history, source = read_correct_history("Tiny", [[1, 0], [0, 1], [1, 1]])
        self.assertEqual(history, {"uid1": frozenset({0}), "uid2": frozenset()})
        self.assertEqual(source.name, "interactions_all.csv")

    def test_set_novelty_uses_union_before_jaccard(self) -> None:
        q_matrix = [[1, 0], [0, 1], [1, 1]]
        history = {"uid1": frozenset({0}), "uid2": frozenset()}
        scores = [("uid1", [0.9, 0.8, 0.7]), ("uid2", [0.9, 0.8, 0.7])]
        result = calculate_kg4ex_nov_set(scores, q_matrix, history, [1, 2, 3])
        self.assertEqual(result["1"]["mean"], 0.5)
        self.assertEqual(result["2"]["mean"], 0.75)
        self.assertEqual(result["3"]["mean"], 0.75)

        # Two repeated concepts distinguish union novelty from per-item averaging.
        single = calculate_kg4ex_nov_set([("uid1", [3, 2, 1])],
                                           [[1, 0], [1, 0], [0, 1]],
                                           {"uid1": frozenset({0})}, [2, 3])
        self.assertEqual(single["2"]["mean"], 0.0)
        self.assertEqual(single["3"]["mean"], 0.5)

    def test_missing_test_student_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ from test split"):
            calculate_kg4ex_nov([("uid1", [1.0])], [[1]],
                                 {"uid1": frozenset(), "uid2": frozenset()}, [1])
