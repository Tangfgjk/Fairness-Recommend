"""Unit checks for semantic feature loading helpers."""

from __future__ import annotations

import unittest

from feature_loader import _log_count, infer_semantic_quality


class SemanticQualityInferenceTest(unittest.TestCase):
    def test_explicit_semantic_quality_is_clamped(self) -> None:
        self.assertEqual(infer_semantic_quality({"semantic_quality": 1.5}, "ex0"), 1.0)
        self.assertEqual(infer_semantic_quality({"semantic_quality": -0.2}, "kc0"), 0.0)

    def test_source_based_quality_priors(self) -> None:
        cases = [
            ({"text_source": "raw_question_text"}, "ex0", 1.0),
            ({"definition_source": "deepseek_llm"}, "kc0", 0.8),
            ({"text_source": "structured_problem_hierarchy"}, "ex1", 0.5),
            ({"text_source": "answer_type_problem_id"}, "ex2", 0.3),
            ({"definition_source": "template", "text_for_embedding": "template text"}, "kc1", 0.0),
            ({}, "uid0", 0.0),
        ]
        for feature, entity_name, expected in cases:
            with self.subTest(feature=feature):
                self.assertEqual(infer_semantic_quality(feature, entity_name), expected)

    def test_non_empty_text_without_source_gets_weak_default(self) -> None:
        self.assertEqual(
            infer_semantic_quality({"text_for_embedding": "Problem: linear equation"}, "ex3"),
            0.5,
        )

    def test_count_feature_is_scaled_to_unit_range(self) -> None:
        self.assertEqual(_log_count(0), 0.0)
        self.assertGreater(_log_count(10), 0.0)
        self.assertLessEqual(_log_count(1_000_000), 1.0)


if __name__ == "__main__":
    unittest.main()
