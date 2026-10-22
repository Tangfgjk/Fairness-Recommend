"""Checks for train-only educational edges and differentiable CompGCN scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from educational_kg_v2 import build_graph
from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID
from semantic_compgcn import CompGCNLayer, SemanticCompGCN, compose


class EducationalGraphTest(unittest.TestCase):
    def test_composition_and_self_loop_relation(self) -> None:
        entity = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        relation = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        self.assertTrue(torch.equal(compose(entity, relation, "mult"), entity * relation))
        self.assertTrue(torch.equal(compose(entity, relation, "sub"), entity - relation))
        self.assertEqual(tuple(compose(entity, relation, "corr").shape), (1, 4))
        layer = CompGCNLayer(4, op="corr", dropout=0.0)
        layer.eval()
        no_edges = torch.empty(0, dtype=torch.long)
        output, updated_relations = layer(
            entity, relation, no_edges, no_edges, no_edges, torch.empty((0, 4)),
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(updated_relations.shape), (1, 4))
        output.sum().backward()
        self.assertIsNotNone(layer.loop_relation.grad)

    def test_directed_bipartite_edges_reach_destination(self) -> None:
        layer = CompGCNLayer(2, op="mult", dropout=0.0)
        with torch.no_grad():
            layer.original.weight.copy_(torch.eye(2))
        entities = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
        relations = torch.tensor([[1.0, 1.0]])
        source = torch.tensor([0, 1])
        target = torch.tensor([2, 2])
        kind = torch.tensor([0, 0])
        actual = layer._aggregate(
            entities, relations, source, target, kind,
            torch.zeros((2, 2)), layer.original,
        )
        expected = (entities[0] + entities[1]) / (2 ** 0.5)
        self.assertTrue(torch.allclose(actual[2], expected))

    def test_train_only_edges_and_no_rec_leakage(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "entities.dict").write_text("0\tuid0\n1\tuid1\n2\tkc0\n3\tex0\n", encoding="utf-8")
            (root / "Q.txt").write_text("1\n", encoding="utf-8")
            (root / "stu2know_mastery.json").write_text(json.dumps([[0.8], [0.3]]), encoding="utf-8")
            (root / "stu2know_forget.json").write_text(json.dumps([[0.0], [0.9]]), encoding="utf-8")
            (root / "triples.txt").write_text("kc0\tmlkc0.80\tuid0\nuid0\trec\tex0\n", encoding="utf-8")
            path = root / "educational_kg_v2.npz"
            stats = build_graph(root, path)
            self.assertEqual(stats["train_learners"], 1)
            with np.load(path) as graph:
                self.assertEqual(len(graph["source"]), 4)  # mastery pair + cover pair
                self.assertFalse(np.isin(1, graph["source"]).any())  # held-out uid1
                self.assertFalse(np.isin(1, graph["target"]).any())
                self.assertEqual(set(graph["relation"].tolist()), {0, 1, 4, 5})

    def test_compgcn_scores_and_backpropagates(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "graph.npz"
            np.savez_compressed(
                path, source=np.array([0, 1, 2, 1]), target=np.array([1, 0, 1, 2]),
                relation=np.array([0, 1, 4, 5]), strength=np.array([0.8, 0.8, 1.0, 1.0]), nentity=np.array(3),
            )
            model = SemanticCompGCN(
                nentity=3, nrelation=1, text_dim=4, numeric_dim=3,
                relation_type_ids=torch.tensor([RELATION_TYPE_TO_ID["rec"]]),
                relation_strengths=torch.ones((1, 1)),
                entity_type_ids=torch.tensor([ENTITY_TYPE_TO_ID["uid"], ENTITY_TYPE_TO_ID["kc"], ENTITY_TYPE_TO_ID["ex"]]),
                cluster_ids=torch.tensor([0, NO_CLUSTER_ID, NO_CLUSTER_ID]),
                text_features=torch.zeros((3, 4)), numeric_features=torch.ones((3, 3)),
                semantic_quality=torch.ones((3, 1)), embedding_dim=20, embedding_shape1=4,
                hidden_size=576, numeric_feature_slices={"learner_irt": (0, 1), "exercise_irt": (1, 3)},
                graph_path=path, compgcn_layers=1,
            )
            model.eval()
            h = torch.tensor([0, 0])
            r = torch.tensor([0, 0])
            t = torch.tensor([1, 2])
            logits = model.score_triples_logits(h, r, t)
            self.assertEqual(tuple(logits.shape), (2,))
            logits.sum().backward()
            self.assertIsNotNone(model.graph_relations.grad)
            self.assertTrue(torch.isfinite(model.graph_relations.grad).all())
            # rec is absent from the message graph but its updated embedding
            # must still affect recommendation scoring.
            self.assertGreater(model.graph_relations.grad[-1].abs().sum().item(), 0)
            model.train()
            model.begin_graph_batch()
            self.assertIs(model._all_graph_embeddings()[0], model._all_graph_embeddings()[0])
            positive = model.score_triples_logits(h, r, t)
            negative = model.score_triples_logits(h, r, t.flip(0))
            (positive - negative).square().sum().backward()
            model.end_graph_batch()
            self.assertIsNone(model._train_embedding_cache)


if __name__ == "__main__":
    unittest.main()
