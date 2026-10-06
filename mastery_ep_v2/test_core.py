"""Small scientific-invariant checks; no dataset training is run."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from core import concept_tags, extract_history, gru_from_cell, verify_gru_equivalence


class FullHistoryTests(unittest.TestCase):
    def test_gru_chunking_and_preanswer_state(self):
        torch.manual_seed(8)
        model = SimpleNamespace(gru_cell=torch.nn.GRUCell(5, 7), hidden_size=7)
        device = torch.device("cpu")
        gru = gru_from_cell(model.gru_cell, device)
        self.assertLess(verify_gru_equivalence(model.gru_cell, gru, device), 2e-5)
        reps = torch.randn(3, 4)
        labels = [1, 0, 1, 0, 0, 1, 1, 0, 1]

        def raw_with(values):
            return SimpleNamespace(
                student_count=1, q_matrix=np.array([[1, 0], [0, 1], [1, 0]]),
                interactions=pd.DataFrame({"uid": [0] * len(values),
                                           "question": [0, 1, 2] * 3,
                                           "response": values,
                                           "concepts": [np.nan] * len(values)}),
            )

        whole = extract_history(raw_with(labels), model, reps, device, chunk_len=20)
        chunked = extract_history(raw_with(labels), model, reps, device, chunk_len=3)
        np.testing.assert_allclose(whole.before, chunked.before, atol=2e-6)
        np.testing.assert_allclose(whole.final_hidden, chunked.final_hidden, atol=2e-6)
        changed = labels.copy()
        changed[4] = 1
        altered = extract_history(raw_with(changed), model, reps, device, chunk_len=3)
        np.testing.assert_allclose(chunked.before[4], altered.before[4], atol=2e-6)
        self.assertGreater(float(np.max(np.abs(chunked.before[5] - altered.before[5]))), 1e-6)

    def test_interaction_tags_are_not_broad_q_union(self):
        frame = pd.DataFrame({"question": [0, 0], "concepts": ["1", "1_2"]})
        q = np.array([[1, 1, 1]], dtype=np.float32)
        tags, mask = concept_tags(frame, q)
        self.assertEqual(mask.sum(axis=1).tolist(), [1, 2])
        self.assertEqual(tags[0, 0], 1)
        self.assertEqual(tags[1, :2].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
