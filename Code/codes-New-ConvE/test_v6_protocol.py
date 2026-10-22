"""Regression checks for student routing, Post identity and objective controls."""
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_compgcn import SemanticCompGCN, checkpoint_student_mode
from semantic_conve_model import TwoCKG4ER
from run_v3_kppd_pipeline import EXPERIMENTS, final_scores_for, parse_args, rerank_command, run_stage, train_command, validate_batch_config
from audit_recommendation_personalization import pearson, summarize
from semantic_experiment_utils import write_json
from run_v6_controlled_experiments import parse_args as v6_args, planned_commands
from run_semantic_conve import kppd_auxiliary_loss
from training_objectives import correlation_penalty


class V6ProtocolTest(unittest.TestCase):
    def arguments(self, *extra):
        with patch("sys.argv", ["pipeline", "--no-run-e0", *extra]):
            return parse_args()

    def test_students_bypass_graph_but_items_and_gradients_use_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            # uid3 is held out and has no graph edge.
            np.savez(path, source=[0, 1, 2, 1], target=[1, 0, 1, 2],
                     relation=[0, 1, 4, 5], strength=[0.8, 0.8, 1., 1.], nentity=4)
            model = SemanticCompGCN(
                nentity=4, nrelation=1, text_dim=4, numeric_dim=3,
                relation_type_ids=torch.tensor([RELATION_TYPE_TO_ID["rec"]]),
                relation_strengths=torch.ones((1, 1)),
                entity_type_ids=torch.tensor([ENTITY_TYPE_TO_ID[t] for t in ("uid", "kc", "ex", "uid")]),
                cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID, 0]),
                text_features=torch.zeros((4, 4)), numeric_features=torch.tensor([[.1, 1., 1.], [1., 1., 1.], [1., 1., 1.], [.9, 1., 1.]]),
                semantic_quality=torch.ones((4, 1)), embedding_dim=20, embedding_shape1=4,
                hidden_size=576, numeric_feature_slices={"learner_irt": (0, 1), "exercise_irt": (1, 3)},
                graph_path=path, input_drop=0., hidden_drop=0., feat_drop=0., compgcn_dropout=0.,
            ).eval()
            ids = torch.arange(4)
            initial = TwoCKG4ER.entity_embedding(model, ids)
            encoded = model.entity_embedding(ids)
            self.assertTrue(torch.allclose(encoded[[0, 3]], initial[[0, 3]]))
            self.assertFalse(torch.allclose(encoded[2], initial[2]))
            h, r, t = torch.tensor([0, 3]), torch.tensor([0, 0]), torch.tensor([2, 2])
            triple = model.score_triples_logits(h, r, t)
            matrix = model.score_tail_matrix_logits(h, r, t[:, None]).flatten()
            self.assertTrue(torch.allclose(triple, matrix, atol=1e-5))
            triple.square().sum().backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.uid_feature_only_fusion.parameters() if p.grad is not None), 0)
            # Covers inverse edges deliver KC messages to exercise tails.
            self.assertGreater(model.comp_layers[0].inverse.weight.grad.abs().sum().item(), 0)
            model.compgcn_student_mode = "graph"
            model.eval()  # clear cache after changing the routing in this test
            self.assertFalse(torch.allclose(model.entity_embedding(ids)[[0, 3]], initial[[0, 3]]))

    def test_checkpoint_routing_preserves_v5(self):
        self.assertEqual(checkpoint_student_mode({"compgcn_impl_version": 2}), "graph")
        self.assertEqual(checkpoint_student_mode({"compgcn_impl_version": 3, "compgcn_student_mode": "direct"}), "direct")
        for config in ({"compgcn_impl_version": 1}, {"compgcn_impl_version": 3},
                       {"compgcn_impl_version": 2, "compgcn_student_mode": "direct"}):
            with self.assertRaises(ValueError):
                checkpoint_student_mode(config)

    def test_quality_swap_is_forwarded_and_named_consistently(self):
        args = self.arguments()
        cmd = rerank_command(args, Path("g"), Path("s"), Path("o"))
        self.assertEqual(cmd[cmd.index("--method") + 1], "quality_swap")
        self.assertEqual(cmd[cmd.index("--lambda-item") + 1], "0.2")
        self.assertIn("--swap-need-margin", cmd)
        scores, evaluation = final_scores_for(Path("run"), EXPERIMENTS["post_only"], args.rerank_method)
        self.assertIn("quality_swap", scores.name)
        self.assertEqual(evaluation, "eval_quality_swap")
        legacy = self.arguments("--rerank-method", "quota_ratio_hybrid")
        self.assertEqual(final_scores_for(Path("run"), EXPERIMENTS["post_only"], legacy.rerank_method)[1], "eval_quota_ratio_hybrid")

    def test_bpr_controls_isolate_bias_and_decomposition(self):
        args = self.arguments()
        for name in ("bpr_only", "bpr_matched", "in_kppd_no_decorr", "in_kppd"):
            cmd = train_command(args, "Eedi", Path("graph"), Path("run"), EXPERIMENTS[name], 2024)
            option = lambda key: cmd[cmd.index(key) + 1]
            self.assertEqual(option("--training-objective"), "bce_bpr")
            self.assertEqual(option("--negative-sampling-mode"), "mixed")
            self.assertEqual(option("--popularity-debias"), "none" if name.startswith("bpr") else "global_personal_need")
            self.assertEqual(option("--tail-bias-mode"), "legacy" if name == "bpr_only" else "pop_branch")
            self.assertEqual(float(option("--decorr-weight")), .01 if name == "in_kppd" else 0.)

    def test_existing_batch_cannot_silently_change_post_or_student_path(self):
        args = self.arguments()
        for old in ({"compgcn_student_mode": "graph"}, {"rerank_method": "quota_ratio_hybrid"}):
            with self.assertRaises(ValueError):
                validate_batch_config(old, args)
        validate_batch_config({"compgcn_student_mode": "direct", "rerank_method": "quality_swap", "experiments": ["baseline"]}, args)

    def test_completed_stage_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "stage.json"
            write_json(status, {"status": "completed", "command": ["python", "train", "--seed", "2024"]})
            with self.assertRaises(ValueError):
                run_stage("train", ["python", "train", "--seed", "2025"], root, root / "log", status, False)

    def test_diagnostics_detect_collapse_and_undefined_correlation(self):
        rows = {"uid0": np.array([.9, .8, .1]), "uid1": np.array([.8, .7, .1])}
        result = summarize(rows, [1, 2, 3], [2])
        self.assertEqual(result["top_n"]["2"]["unique_ordered_lists"], 1)
        self.assertEqual(result["top_n"]["2"]["largest_identical_list_fraction"], 1.)
        self.assertIsNone(pearson([1, 1], [0, 1]))
        with self.assertRaises(ValueError):
            summarize(rows, [1, 2, 3], [2], {"uid2": np.array([1, 2, 3])})

    def test_v6_controls_and_main_reuse_the_same_direct_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "educational_kg_v2.npz").touch()
            with patch("run_v6_controlled_experiments.select_dataset", return_value=Namespace(root=root, graph=root)):
                commands = planned_commands(v6_args(["--datasets", "Eedi", "--phase", "all", "--plan"]))
            self.assertEqual(len(commands), 5)
            controls, main = commands[2][1], commands[3][1]
            self.assertEqual(controls[controls.index("--batch-id") + 1], main[main.index("--batch-id") + 1])
            self.assertIn("--audit-personalization", controls)
            self.assertEqual(controls[controls.index("--ep-thresholds") + 1], "0.5")
            self.assertEqual(list(root.iterdir()), [root / "educational_kg_v2.npz"])

    def test_auxiliary_loss_does_not_require_a_fake_need_target(self):
        value = torch.tensor([.2, .4], requires_grad=True)
        head = Namespace(exercise_entity_to_index=torch.tensor([-1, -1, 0, 1]),
                         uid_entity_to_index=torch.tensor([0, 1, -1, -1]),
                         item_popularity_z=torch.tensor([1., 2.]), user_profiles=torch.ones((2, 7)))
        class Model:
            popularity_debias_head = head
            def score_triples_components(self, h, r, t):
                return {"core": value, "global_pop": value, "personal_pop": value}
        loss, details = kppd_auxiliary_loss(Model(), torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([2, 3]),
                                          Namespace(global_pop_aux_weight=.1, personal_pop_aux_weight=.1, decorr_weight=0.), 0)
        loss.backward()
        self.assertGreater(value.grad.abs().sum().item(), 0.)
        self.assertEqual(details["need_aux"], 0.)

    def test_constant_core_or_popularity_has_finite_gradients(self):
        for core, popularity in (([1., 1., 1.], [1., 2., 3.]), ([1., 2., 3.], [1., 1., 1.])):
            values = torch.tensor(core, requires_grad=True)
            loss = correlation_penalty(values, torch.tensor(popularity))
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == "__main__":
    unittest.main()
