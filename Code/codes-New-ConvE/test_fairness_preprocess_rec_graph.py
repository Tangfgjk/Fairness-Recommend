import json
import tempfile
import unittest
from pathlib import Path

from fairness_preprocess_rec_graph import prepare_fair_preprocess_graph, split_triples


class FairnessPreprocessRecGraphTest(unittest.TestCase):
    def make_graph(self, root: Path) -> Path:
        graph = root / "er_graph"
        graph.mkdir()
        (graph / "Q.txt").write_text("1\t0\n0\t1\n1\t0\n", encoding="utf-8")
        (graph / "entities.dict").write_text(
            "0\tuid0\n1\tuid1\n2\tkc0\n3\tkc1\n4\tex0\n5\tex1\n6\tex2\n",
            encoding="utf-8",
        )
        (graph / "relations.dict").write_text("0\trec\n1\tmlkc0.50\n", encoding="utf-8")
        (graph / "triples.txt").write_text(
            "\n".join(
                [
                    "kc0\tmlkc0.50\tuid0",
                    "kc1\tmlkc0.50\tuid1",
                    "uid0\trec\tex0",
                    "uid1\trec\tex0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (graph / "test_triples.txt").write_text("kc0\tmlkc0.50\tuid0\n", encoding="utf-8")
        (graph / "stu2ex_recommend_full_precision.json").write_text(
            json.dumps([[0.10, 0.11, 0.90], [0.10, 0.12, 0.90]]),
            encoding="utf-8",
        )
        return graph

    def rec_edges(self, graph: Path) -> set[tuple[int, int]]:
        _non_rec, rec_by_user, _edges = split_triples(graph / "triples.txt")
        return {(user, ex) for user, exercises in rec_by_user.items() for ex in exercises}

    def test_zero_lambdas_exactly_rebuild_original_rec_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            target = Path(tmp) / "fair_rebuild"

            manifest = prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=0.0,
                lambda_kc=0.0,
                popularity_source="rec_triples",
            )

            self.assertTrue(manifest["baseline_rebuild_exact_match"])
            self.assertEqual(manifest["changed_rec_edge_ratio"], 0.0)
            self.assertEqual(manifest["education_cost"]["mean_top10_overlap"], 1.0)
            self.assertEqual(self.rec_edges(source), self.rec_edges(target))

    def test_item_prior_can_promote_underexposed_candidate_within_top_m(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            target = Path(tmp) / "fair_item"

            manifest = prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=1.0,
                lambda_kc=0.0,
                popularity_source="rec_triples",
            )

            self.assertFalse(manifest["baseline_rebuild_exact_match"])
            self.assertGreater(manifest["changed_rec_edge_ratio"], 0.0)
            self.assertEqual(self.rec_edges(target), {(0, 1), (1, 1)})
            self.assertEqual(manifest["rec_degree_preserved"]["min"], 1)
            self.assertEqual(manifest["rec_degree_preserved"]["max"], 1)


if __name__ == "__main__":
    unittest.main()
