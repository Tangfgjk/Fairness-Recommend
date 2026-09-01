"""Unit checks for compact raw-concat SemanticConvE representations."""

from __future__ import annotations

import unittest

import torch

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_conve_model import SemanticConvE


class SemanticConvECompactConcatTest(unittest.TestCase):
    def build_model(
        self,
        entity_type_ids: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        numeric_features: torch.Tensor | None = None,
    ) -> SemanticConvE:
        torch.manual_seed(2024)
        nentity = 3
        nrelation = 3
        text_dim = 4
        numeric_dim = 3
        if entity_type_ids is None:
            entity_type_ids = torch.tensor(
                [ENTITY_TYPE_TO_ID["uid"], ENTITY_TYPE_TO_ID["kc"], ENTITY_TYPE_TO_ID["ex"]],
                dtype=torch.long,
            )
        if cluster_ids is None:
            cluster_ids = torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID], dtype=torch.long)
        if numeric_features is None:
            numeric_features = torch.zeros((nentity, numeric_dim), dtype=torch.float32)
        return SemanticConvE(
            nentity=nentity,
            nrelation=nrelation,
            text_dim=text_dim,
            numeric_dim=numeric_dim,
            relation_type_ids=torch.tensor(
                [RELATION_TYPE_TO_ID["mlkc"], RELATION_TYPE_TO_ID["mlkc"], RELATION_TYPE_TO_ID["exfr"]],
                dtype=torch.long,
            ),
            relation_strengths=torch.tensor([[0.50], [0.80], [0.50]], dtype=torch.float32),
            entity_type_ids=entity_type_ids,
            cluster_ids=cluster_ids,
            text_features=torch.zeros((nentity, text_dim), dtype=torch.float32),
            numeric_features=numeric_features,
            semantic_quality=torch.ones((nentity, 1), dtype=torch.float32),
            embedding_dim=20,
            embedding_shape1=4,
            hidden_size=576,
            numeric_feature_slices={"learner_irt": (0, 1), "exercise_irt": (1, 3)},
        )

    def test_id_only_entity_is_independent_from_side_features(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "id_only"
        model_b.ablation_mode = "id_only"
        with torch.no_grad():
            model_b.text_features.fill_(9.0)
            model_b.numeric_features.fill_(7.0)
            model_b.cluster_ids.fill_(1)

        entity_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids), atol=1e-6))

    def test_feature_only_ignores_uid_exercise_and_relation_id_embeddings(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "feature_only"
        model_b.ablation_mode = "feature_only"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_a.numeric_features[0, 0] = 0.25
            model_b.numeric_features[0, 0] = 0.25
            model_a.text_features[2, 0] = 0.75
            model_b.text_features[2, 0] = 0.75
            model_a.numeric_features[2, 1:] = torch.tensor([0.2, 0.8])
            model_b.numeric_features[2, 1:] = torch.tensor([0.2, 0.8])
            model_b.emb_e.weight[0].fill_(8.0)
            model_b.emb_e.weight[2].fill_(7.0)
            model_b.relation_id_emb.weight.fill_(6.0)

        entity_ids = torch.tensor([0, 2], dtype=torch.long)
        relation_ids = torch.tensor([0, 1], dtype=torch.long)
        self.assertTrue(torch.allclose(model_a.entity_embedding(entity_ids), model_b.entity_embedding(entity_ids), atol=1e-6))
        self.assertTrue(torch.allclose(model_a.relation_embedding(relation_ids), model_b.relation_embedding(relation_ids), atol=1e-6))

    def test_feature_only_cognitive_graph_variants_ignore_ids(self) -> None:
        for ablation in ("feature_only_no_mastery", "feature_only_no_forgetting"):
            model_a = self.build_model()
            model_b = self.build_model()
            model_a.ablation_mode = ablation
            model_b.ablation_mode = ablation
            model_a.eval()
            model_b.eval()
            with torch.no_grad():
                model_a.numeric_features[0, 0] = model_b.numeric_features[0, 0] = 0.25
                model_a.text_features[2, 0] = model_b.text_features[2, 0] = 0.75
                model_a.numeric_features[2, 1:] = model_b.numeric_features[2, 1:] = torch.tensor([0.2, 0.8])
                model_b.emb_e.weight[[0, 2]].fill_(8.0)
                model_b.relation_id_emb.weight.fill_(8.0)

            self.assertTrue(
                torch.allclose(
                    model_a.entity_embedding(torch.tensor([0, 2])),
                    model_b.entity_embedding(torch.tensor([0, 2])),
                    atol=1e-6,
                ),
                ablation,
            )
            self.assertTrue(
                torch.allclose(
                    model_a.relation_embedding(torch.tensor([0, 1])),
                    model_b.relation_embedding(torch.tensor([0, 1])),
                    atol=1e-6,
                ),
                ablation,
            )

    def test_feature_only_gate_values_report_removed_id_features(self) -> None:
        model = self.build_model()
        model.ablation_mode = "feature_only"

        values = model.gate_values()

        self.assertNotIn("entity_id_200", values["entity_features"]["uid"])
        self.assertNotIn("entity_id_200", values["entity_features"]["ex"])
        self.assertNotIn("relation_id_200", values["relation_features"])

    def test_feature_only_relation_id_uses_only_relation_id(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "feature_only_relation_id"
        model_b.ablation_mode = "feature_only_relation_id"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.relation_type_ids.fill_(RELATION_TYPE_TO_ID["exfr"])
            model_b.relation_strengths.fill_(0.01)

        relation_ids = torch.tensor([0, 1], dtype=torch.long)
        self.assertTrue(torch.allclose(model_a.relation_embedding(relation_ids), model_b.relation_embedding(relation_ids), atol=1e-6))

    def test_feature_only_learner_id_uses_only_learner_id(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "feature_only_learner_id"
        model_b.ablation_mode = "feature_only_learner_id"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.numeric_features[0, 0] = 0.99

        learner_id = torch.tensor([0], dtype=torch.long)
        self.assertTrue(torch.allclose(model_a.entity_embedding(learner_id), model_b.entity_embedding(learner_id), atol=1e-6))

    def test_feature_only_exercise_id_uses_only_exercise_id(self) -> None:
        model_a = self.build_model()
        model_b = self.build_model()
        model_a.ablation_mode = "feature_only_exercise_id"
        model_b.ablation_mode = "feature_only_exercise_id"
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            model_b.text_features[2].fill_(0.95)
            model_b.numeric_features[2, 1:].fill_(0.99)

        exercise_id = torch.tensor([2], dtype=torch.long)
        self.assertTrue(torch.allclose(model_a.entity_embedding(exercise_id), model_b.entity_embedding(exercise_id), atol=1e-6))

    def test_can_score_tail_pairs_from_external_head_embeddings(self) -> None:
        model = self.build_model(
            entity_type_ids=torch.tensor(
                [ENTITY_TYPE_TO_ID["ex"], ENTITY_TYPE_TO_ID["ex"], ENTITY_TYPE_TO_ID["ex"]],
                dtype=torch.long,
            )
        )
        model.eval()
        tail_ids = torch.tensor([1, 2], dtype=torch.long)
        relation_ids = torch.tensor([0, 0], dtype=torch.long)
        base_head = model.entity_embedding(torch.tensor([0, 0], dtype=torch.long))
        shifted_head = base_head + 0.25

        base_scores = model.score_tail_pairs_from_head_embeddings(base_head, relation_ids, tail_ids)
        shifted_scores = model.score_tail_pairs_from_head_embeddings(shifted_head, relation_ids, tail_ids)

        self.assertEqual(tuple(base_scores.shape), (2,))
        self.assertFalse(torch.allclose(base_scores, shifted_scores, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
