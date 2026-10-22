"""Regressions reproduced in the V6 debias-only audit."""
import math
import unittest

import numpy as np
import torch

from fairness_rerank import RerankConfig, rerank_one_user_quality_swap
from popularity_debias_context import PopularityDebiasContext
from popularity_debias_head import PopularityDebiasConfig, PopularityDebiasHead, repair_legacy_context_state
from evaluate_kppd_score_modes import make_mode_logits


class DebiasFixTest(unittest.TestCase):
    def swap(self, scores, need, popularity, **kwargs):
        config = RerankConfig(method="quality_swap", top_k=50, min_candidates=100,
                              lambda_item=1., lambda_kc=0., beta_item=0., beta_kc=0.,
                              swap_fraction=.1, swap_score_margin=.03, swap_need_margin=.03, **kwargs)
        return rerank_one_user_quality_swap("uid0", scores, [[1.-v for v in need]],
                                            [[1]] * 100, popularity, [1.], [0.]*100, [0.], config)

    def test_test_student_identity_is_independent_of_training_profile_length(self):
        context = PopularityDebiasContext(1, [1.], [0.],
                    np.array([[.2, .1, .3, .4, .3, 1., 1.]], dtype=np.float32),
                    np.array([[.1], [.2], [.3], [.7]], dtype=np.float32), {})
        tensors = context.to_tensors(entity2id={"uid0": 0, "uid3": 1, "ex0": 2},
                                     exercise_entity_ids=torch.tensor([2]))
        self.assertEqual(int(tensors["uid_entity_to_index"][1]), 3)
        self.assertTrue(torch.equal(tensors["user_profiles"][0], torch.tensor(context.user_profiles[0])))
        self.assertEqual(float(tensors["user_profiles"][3, 6]), 0.)
        head = PopularityDebiasHead(PopularityDebiasConfig(mode="global_personal_need"), tensors)
        self.assertAlmostEqual(float(head.components(torch.tensor([1]), torch.tensor([2]))["need"][0]), .7, places=6)

    def test_zero_fairness_gain_never_changes_order(self):
        order, record = self.swap([1.-i/100 for i in range(100)], [1.]*100, [1.]*100)
        self.assertEqual(record["swaps"], 0)
        self.assertEqual(order, list(range(100)))

    def test_final_quality_loss_is_bounded_against_original_positions(self):
        scores = [1.]*20 + [.8] + [.79]*29 + [.78, .76] + [0.]*48
        need = [1.]*100
        need[20], need[50], need[51], need[99] = .9, .88, .86, 0.
        pop = [1.]*100
        pop[50], pop[51] = .5, 0.
        order, _ = self.swap(scores, need, pop)
        for slot, exercise in enumerate(order[:50]):
            self.assertLessEqual(scores[slot]-scores[exercise], .03+1e-12)
            self.assertLessEqual(need[slot]-need[exercise], .03+1e-12)

    def test_every_reported_top_n_obeys_replacement_budget(self):
        scores = [1.]*20 + [.8]*5 + [.79]*25 + [.78]*5 + [0.]*45
        need, pop = [1.]*100, [1.]*100
        for i in range(20, 25):
            need[i] = .9
        for i in range(50, 55):
            need[i], pop[i] = .89, 0.
        need[99] = 0.
        order, _ = self.swap(scores, need, pop)
        for n in range(5, 51, 5):
            self.assertLessEqual(len(set(order[:n])-set(range(n))), math.ceil(.1*n), f"Top-{n}")

    def test_repeated_swap_cannot_bypass_baseline_when_only_one_prefix_is_requested(self):
        scores = [1.]*20 + [.8] + [.79]*29 + [.78, .76] + [0.]*48
        need, pop = [1.]*100, [1.]*100
        need[20], need[50], need[51], need[99] = .9, .88, .86, 0.
        pop[50], pop[51] = .5, 0.
        order, record = self.swap(scores, need, pop, swap_prefixes=(50,))
        self.assertEqual(order[20], 50)  # beneficial first replacement survives
        self.assertLessEqual(record["max_need_loss"], .03+1e-12)
        self.assertLessEqual(record["max_score_loss"], .03+1e-12)

    def test_demoted_item_cannot_break_quality_at_a_later_protected_position(self):
        scores = [1.-i/100 for i in range(100)]
        need, pop = [1.]*100, [1.]*100
        need[4], pop[5] = .8, 0.
        need[99] = 0.
        order, _ = self.swap(scores, need, pop)
        # Moving item4 to position5 would decrease that position's need by .2.
        for pos, item in enumerate(order[:50]):
            self.assertLessEqual(need[pos]-need[item], .03+1e-12)

    def test_legacy_context_repair_changes_only_identity_and_fallback_buffers(self):
        context = PopularityDebiasContext(1, [1.], [0.], np.array([[.2]*6+[1.]], dtype=np.float32),
                                          np.array([[.1], [.2], [.3], [.7]], dtype=np.float32), {})
        tensors = context.to_tensors(entity2id={"uid0":0,"uid3":1,"ex0":2}, exercise_entity_ids=torch.tensor([2]))
        model = torch.nn.Module()
        model.popularity_debias_head = PopularityDebiasHead(PopularityDebiasConfig(mode="global_personal_need"), tensors)
        state = model.state_dict()
        state["popularity_debias_head.user_profiles"] = state["popularity_debias_head.user_profiles"][:1].clone()
        state["popularity_debias_head.uid_entity_to_index"] = torch.tensor([0,-1,-1])
        parameters = {n:p.detach().clone() for n,p in model.named_parameters()}
        fixed, changed = repair_legacy_context_state(model, state, 1)
        self.assertTrue(changed)
        self.assertEqual(int(state["popularity_debias_head.uid_entity_to_index"][1]), -1)
        model.load_state_dict(fixed)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(parameters[name], value))
        self.assertEqual(int(model.popularity_debias_head.uid_entity_to_index[1]), 3)
        state["popularity_debias_head.user_profiles"][0,0] += 1.
        with self.assertRaises(ValueError):
            repair_legacy_context_state(model, state, 1)

    def test_score_modes_separate_prior_and_validate_student_alignment(self):
        components = {"core":[("uid0",[.2,.4])], "need":[("uid0",[.5,.1])],
                      "raw":[("uid0",[1.2,1.4])], "debiased":[("uid0",[.7,.5])]}
        uids, modes = make_mode_logits(components, 1.)
        self.assertEqual(uids,["uid0"])
        self.assertTrue(np.allclose(modes["core_need"], modes["debiased"]))
        components["need"] = [("uid1",[.5,.1])]
        with self.assertRaises(ValueError):
            make_mode_logits(components, 1.)

    def test_swaps_remain_a_bounded_permutation_across_varied_popularity(self):
        changes = 0
        for seed in range(5):
            scores = np.linspace(1., 0., 100).tolist()
            pop = np.random.default_rng(seed).random(100).tolist()
            order, record = self.swap(scores, [1.]*100, pop)
            self.assertEqual(set(order), set(range(100)))
            self.assertEqual(len(order), 100)
            self.assertLessEqual(record["max_score_loss"], .03+1e-12)
            for n in range(5, 51, 5):
                self.assertLessEqual(len(set(order[:n])-set(range(n))), math.ceil(.1*n))
                self.assertLessEqual(record["prefix_swaps"][str(n)], math.ceil(.1*n))
            changes += record["swaps"]
        self.assertGreater(changes, 0)  # safety must not simply disable Post


if __name__ == "__main__":
    unittest.main()
