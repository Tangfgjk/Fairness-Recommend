from __future__ import annotations

import unittest
from argparse import Namespace

import torch

from run_semantic_conve import kppd_auxiliary_loss


class _FakeDebiasHead:
    pass


class _FailingScoreModel:
    popularity_debias_head = _FakeDebiasHead()

    def score_triples_components(self, h, r, t):  # pragma: no cover - should not be called
        raise AssertionError("single-rec KPPD batches should be skipped before scoring")


class RunSemanticConveKPPDTest(unittest.TestCase):
    def test_kppd_auxiliary_loss_skips_single_rec_sample(self) -> None:
        loss, details = kppd_auxiliary_loss(
            _FailingScoreModel(),
            h=torch.tensor([1], dtype=torch.long),
            r=torch.tensor([2], dtype=torch.long),
            t=torch.tensor([3], dtype=torch.long),
            args=Namespace(
                global_pop_aux_weight=0.1,
                personal_pop_aux_weight=0.1,
                need_aux_weight=0.0,
                decorr_weight=0.01,
            ),
            rec_relation_id=2,
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(details["global_pop_aux"], 0.0)


if __name__ == "__main__":
    unittest.main()
