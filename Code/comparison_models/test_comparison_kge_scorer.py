import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_kge_recommendations import (  # noqa: E402
    native_tail_scores,
    score_native_er_candidates,
    tail_estimate,
)


class ComparisonKgeScorerTests(unittest.TestCase):
    def test_distmult_and_complex_keep_matched_tail_above_opposite_tail(self):
        distmult_state = tail_estimate("DistMult", np.array([1.0, 2.0]), np.array([1.0, 1.0]), 12.0)
        distmult_scores = native_tail_scores(
            "DistMult",
            distmult_state[None, :],
            np.array([1.0, 1.0]),
            np.array([[1.0, 2.0], [-1.0, -2.0]]),
            12.0,
        )
        self.assertGreater(distmult_scores[0], distmult_scores[1])

        complex_state = tail_estimate(
            "ComplEx", np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 0.0, 0.0]), 12.0
        )
        complex_scores = native_tail_scores(
            "ComplEx",
            complex_state[None, :],
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]),
            12.0,
        )
        self.assertGreater(complex_scores[0], complex_scores[1])

    def test_rotate_zero_phase_preserves_tail_state(self):
        head = np.array([1.0, 2.0, 3.0, 4.0])
        state = tail_estimate("RotatE", head, np.array([0.0, 0.0]), 12.0)
        np.testing.assert_allclose(state, head)

    def test_native_er_scoring_does_not_form_cognitive_to_exercise_triples(self):
        entity2id = {"kc0": 0, "ex0": 1, "ex1": 2}
        relation2id = {"mlkc0": 0, "rec": 1}
        entities = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        relations = np.array([[1.0, 1.0], [1.0, 1.0]])
        scores = score_native_er_candidates(
            "DistMult",
            entities,
            relations,
            entity2id,
            relation2id,
            ["ex0", "ex1"],
            [("kc0", "mlkc0")],
            {},
            12.0,
        )
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
