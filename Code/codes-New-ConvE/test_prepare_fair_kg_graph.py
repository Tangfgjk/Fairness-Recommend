from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prepare_fair_kg_graph import EX_HAS_KC, KC_HAS_EX, prepare_fair_kg_graph


class PrepareFairKgGraphTest(unittest.TestCase):
    def test_prepare_graph_adds_ex_kc_edges_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "er_graph"
            target = root / "fair_kg_graph"
            source.mkdir()
            (source / "entities.dict").write_text(
                "0\tuid0\n1\tex0\n2\tex1\n3\tkc0\n4\tkc1\n",
                encoding="utf-8",
            )
            (source / "relations.dict").write_text("0\trec\n", encoding="utf-8")
            (source / "triples.txt").write_text("uid0\trec\tex0\n", encoding="utf-8")
            (source / "Q.txt").write_text("1,0\n1,1\n", encoding="utf-8")
            (source / "semantic_kg_features").mkdir()

            manifest = prepare_fair_kg_graph(source, target)

            self.assertEqual(manifest["original_triples"], 1)
            self.assertEqual(manifest["candidate_ex_kc_triples"], 6)
            source_triples = (source / "triples.txt").read_text(encoding="utf-8")
            target_triples = (target / "triples.txt").read_text(encoding="utf-8")
            relations = (target / "relations.dict").read_text(encoding="utf-8")
            self.assertEqual(source_triples.strip(), "uid0\trec\tex0")
            self.assertIn(f"ex0\t{EX_HAS_KC}\tkc0", target_triples)
            self.assertIn(f"kc1\t{KC_HAS_EX}\tex1", target_triples)
            self.assertIn(EX_HAS_KC, relations)
            self.assertTrue((target / "fair_kg_graph_manifest.json").exists())

    def test_existing_target_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "er_graph"
            target = root / "fair_kg_graph"
            source.mkdir()
            target.mkdir()
            (source / "entities.dict").write_text("0\tex0\n0\tkc0\n", encoding="utf-8")
            (source / "relations.dict").write_text("0\trec\n", encoding="utf-8")
            (source / "triples.txt").write_text("", encoding="utf-8")
            (source / "Q.txt").write_text("1\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_fair_kg_graph(source, target)


if __name__ == "__main__":
    unittest.main()

