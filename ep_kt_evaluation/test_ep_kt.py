"""Checks for the independent, frozen-KT Ep calculation."""

import unittest
from pathlib import Path

import numpy as np
import torch

import evaluate_ep_kt as ep
import run_all_ep


class FormulaAndProtocolTests(unittest.TestCase):
    def test_reference_global_mastery_gain(self):
        before = np.array([0.2, 0.4, 0.9])
        after = np.array([0.3, 0.5, 0.8])
        expected = (after.sum() - before.sum()) / (3 - before.sum())
        self.assertAlmostEqual(ep.normalized_gain(before, after), expected)
        self.assertEqual(ep.normalized_gain(np.ones(3), np.ones(3)), 0.0)

    def test_both_response_rules(self):
        self.assertEqual(ep.choose_response(0.5, "threshold", None), 1)
        self.assertEqual(ep.choose_response(0.49999, "threshold", None), 0)
        rng = np.random.default_rng(42)
        draws = [ep.choose_response(0.7, "bernoulli", rng) for _ in range(10000)]
        self.assertAlmostEqual(np.mean(draws), 0.7, delta=0.02)

    def test_exact_paper_manifest(self):
        entries = run_all_ep.discover()
        self.assertEqual(len(entries), 63)
        for dataset in run_all_ep.DATASETS:
            subset = [entry for entry in entries if entry[0] == dataset]
            self.assertEqual(len(subset), 21)
            self.assertEqual(len({model for _, model, _, _ in subset}), 9)

    def test_all_paper_scores_match_their_test_population(self):
        raw_by_dataset = {}
        for dataset in run_all_ep.DATASETS:
            raw_name = "raw_compact" if dataset == "XES3G5M-sub-small" else "raw"
            raw_by_dataset[dataset] = ep.load_raw_dataset(dataset, ep.PROJECT / "Data_Fin", raw_name)
        for dataset, model, seed, score_path in run_all_ep.discover():
            with self.subTest(dataset=dataset, model=model, seed=seed):
                raw = raw_by_dataset[dataset]
                scores = ep.load_scores(score_path)
                uids = {int(uid[3:]) if uid.startswith("uid") else int(uid) for uid, _ in scores}
                self.assertEqual(uids, set(raw.test_uids))
                self.assertTrue(all(len(values) == raw.exercise_count for _, values in scores))
                self.assertTrue(all(np.isfinite(np.asarray(values, dtype=np.float64)).all() for _, values in scores))

    def test_history_replay_matches_ektm_forward(self):
        for dataset in run_all_ep.DATASETS:
            with self.subTest(dataset=dataset):
                raw_name = "raw_compact" if dataset == "XES3G5M-sub-small" else "raw"
                raw = ep.load_raw_dataset(dataset, ep.PROJECT / "Data_Fin", raw_name)
                model, config, _ = ep.load_frozen_kt(dataset, ep.PROJECT / "Data_Fin", raw, torch.device("cpu"))
                self.assertEqual(float(config.mastery_loss_weight), 0.0)
                reps = ep.exercise_representations(model, raw.exercise_count, torch.device("cpu"))
                uid = next(iter(raw.test_uids))
                history = raw.interactions.loc[raw.interactions["uid"] == uid].tail(int(config.max_seq_len))
                self.assertFalse(history.empty)
                hidden = ep.replay_history(model, reps, history, torch.device("cpu"))
                items = torch.tensor([[int(v) for v in history.question]], dtype=torch.long)
                answers = torch.tensor([[float(v) for v in history.response]], dtype=torch.float32)
                mask = torch.ones_like(items, dtype=torch.bool)
                with torch.inference_mode():
                    _, mastery_sequence = model(items, answers, mask)
                    replay_mastery = torch.sigmoid(model.mastery_head(hidden))
                    self.assertTrue(torch.allclose(replay_mastery, mastery_sequence[:, -1, :], atol=1e-6))

    def test_one_simulated_step_updates_the_same_frozen_kt(self):
        dataset = "Eedi"
        raw = ep.load_raw_dataset(dataset, ep.PROJECT / "Data_Fin", "raw")
        model, config, _ = ep.load_frozen_kt(dataset, ep.PROJECT / "Data_Fin", raw, torch.device("cpu"))
        reps = ep.exercise_representations(model, raw.exercise_count, torch.device("cpu"))
        uid = next(iter(raw.test_uids))
        history = raw.interactions.loc[raw.interactions["uid"] == uid].tail(int(config.max_seq_len))
        hidden = ep.replay_history(model, reps, history, torch.device("cpu"))
        q = raw.q_matrix.astype(np.float64)
        before = ep.concept_state(model, hidden, reps, q)
        first = 0
        gains, steps = ep.simulate_one(model, hidden, reps, q, [first], [1], before, "threshold", None)
        probability = ep.answer_probability(model, hidden, reps[first])
        response = int(probability >= 0.5)
        with torch.inference_mode():
            next_hidden = model.gru_cell(
                torch.cat((reps[first:first + 1], torch.tensor([[float(response)]])), dim=1), hidden)
        expected = ep.normalized_gain(before, ep.concept_state(model, next_hidden, reps, q))
        self.assertEqual(steps[0]["simulated_response"], response)
        self.assertAlmostEqual(steps[0]["prob_correct"], probability)
        self.assertAlmostEqual(gains[1], expected)
        self.assertFalse(torch.allclose(hidden, next_hidden))


if __name__ == "__main__":
    unittest.main()
