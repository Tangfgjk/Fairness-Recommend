import json
import tempfile
import unittest
from pathlib import Path

from fairness_preprocess_rec_graph import prepare_fair_preprocess_graph, split_triples


class FairnessPreprocessRecGraphTest(unittest.TestCase):
    def make_graph(self, root: Path) -> Path:
        graph = root / "er_graph"
        graph.mkdir(parents=True)
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
        (graph / "er_graph_manifest.json").write_text(
            json.dumps({"generator": "build_er_graph.py", "raw_dir": "../raw"}),
            encoding="utf-8",
        )
        (graph / "stu2ex_recommend_full_precision.json").write_text(
            json.dumps([[0.10, 0.11, 0.90], [0.10, 0.12, 0.90]]),
            encoding="utf-8",
        )
        return graph

    def rec_edges(self, graph: Path) -> set[tuple[int, int]]:
        _non_rec, rec_by_user, _edges = split_triples(graph / "triples.txt")
        return {(user, ex) for user, exercises in rec_by_user.items() for ex in exercises}

    def non_rec_lines(self, graph: Path) -> list[str]:
        non_rec, _rec_by_user, _edges = split_triples(graph / "triples.txt")
        return non_rec

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
            self.assertEqual(manifest["source_rec_edge_hash"], manifest["output_rec_edge_hash"])
            self.assertEqual(manifest["education_cost"]["mean_top10_overlap"], 1.0)
            self.assertEqual(self.rec_edges(source), self.rec_edges(target))
            self.assertTrue(manifest["rec_degree_preserved"])
            self.assertEqual(manifest["source_rec_degree"]["min"], 1)
            self.assertEqual(manifest["output_rec_degree"]["max"], 1)

    def test_zero_lambda_tie_breaking_matches_original_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            (source / "stu2ex_recommend_full_precision.json").write_text(
                json.dumps([[0.10, 0.10, 0.90], [0.10, 0.10, 0.90]]),
                encoding="utf-8",
            )
            target = Path(tmp) / "fair_tie"

            manifest = prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=0.0,
                lambda_kc=0.0,
                popularity_source="rec_triples",
            )

            self.assertTrue(manifest["baseline_rebuild_exact_match"])
            self.assertEqual(self.rec_edges(target), {(0, 0), (1, 0)})

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
            self.assertTrue(manifest["rec_degree_preserved"])
            self.assertEqual(manifest["replaced_rec_edge_count"], 2)
            self.assertEqual(manifest["rec_edge_symmetric_difference_count"], 4)
            self.assertEqual(manifest["changed_user_count"], 2)
            self.assertEqual(manifest["education_cost"]["mean_promoted_original_rank"], 2.0)
            self.assertEqual(manifest["education_cost"]["mean_promotion_depth"], 1.0)

    def test_kc_prior_can_promote_underexposed_kc_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            target = Path(tmp) / "fair_kc"

            prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=0.0,
                lambda_kc=1.0,
                popularity_source="rec_triples",
            )

            self.assertEqual(self.rec_edges(target), {(0, 1), (1, 1)})

    def test_non_rec_relations_and_manifest_are_verified_from_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            target = Path(tmp) / "fair_invariant"

            manifest = prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=1.0,
                lambda_kc=0.0,
                popularity_source="rec_triples",
            )

            self.assertTrue(manifest["non_rec_relations_unchanged"])
            self.assertEqual(manifest["source_non_rec_hash"], manifest["output_non_rec_hash"])
            self.assertEqual(self.non_rec_lines(source), self.non_rec_lines(target))
            output_er_manifest = json.loads((target / "er_graph_manifest.json").read_text(encoding="utf-8"))
            source_er_manifest = json.loads((target / "source_er_graph_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(output_er_manifest["graph_variant"], "fairness_preprocessed")
            self.assertEqual(output_er_manifest["rec_edge_preprocessing"]["manifest"], "fairness_preprocess_manifest.json")
            self.assertEqual(source_er_manifest["generator"], "build_er_graph.py")

    def test_train_interaction_popularity_ignores_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_graph(root / "dataset")
            raw = root / "dataset" / "raw"
            raw.mkdir()
            (raw / "interactions_all.csv").write_text(
                "\n".join(
                    [
                        "split,entity_id,entity_exercise_id",
                        "train,uid0,ex0",
                        "test,uid1,ex1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            target = root / "fair_train_pop"

            manifest = prepare_fair_preprocess_graph(
                source,
                target,
                candidate_pool_size=2,
                lambda_item=1.0,
                lambda_kc=0.0,
                popularity_source="train_interactions",
            )

            self.assertEqual(self.rec_edges(target), {(0, 1), (1, 1)})
            popularity_meta = manifest["fairness_context"]["item_popularity_source"]
            self.assertEqual(popularity_meta["source"], "train_interactions")
            self.assertEqual(popularity_meta["rows_seen"], 2)
            self.assertEqual(popularity_meta["train_rows"], 1)

    def test_invalid_parameters_and_shapes_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_graph(Path(tmp))
            target = Path(tmp) / "fair_invalid"

            with self.assertRaises(ValueError):
                prepare_fair_preprocess_graph(source, target, lambda_item=-0.1, popularity_source="rec_triples")

            with self.assertRaises(ValueError):
                prepare_fair_preprocess_graph(source, target, epsilon=0.0, popularity_source="rec_triples")

            (source / "stu2ex_recommend_full_precision.json").write_text(
                json.dumps([[0.10, 0.11], [0.10, 0.12]]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                prepare_fair_preprocess_graph(source, target, popularity_source="rec_triples")


if __name__ == "__main__":
    unittest.main()
