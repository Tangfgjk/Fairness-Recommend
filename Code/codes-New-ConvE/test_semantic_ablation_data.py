"""Unit checks for SemanticConvE graph ablation data generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from semantic_ablation_data import prepare_semantic_ablation_graph


class SemanticAblationDataTest(unittest.TestCase):
    def write_json(self, path: Path, data: object) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def write_text(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def test_no_forgetting_rebuilds_rec_and_removes_exfr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            self.write_text(source / "Q.txt", "1,0\n0,1\n")
            self.write_text(source / "entities.dict", "0\tuid0\n1\tuid1\n2\tkc0\n3\tkc1\n4\tex0\n5\tex1\n")
            self.write_text(source / "relations.dict", "0\trec\n1\tmlkc0.80\n2\texfr0.10\n")
            self.write_json(source / "stu2know_mastery.json", [[0.8, 0.2], [0.3, 0.9]])
            self.write_json(source / "stu2know_forget.json", [[0.0, 0.0], [0.0, 0.0]])
            self.write_json(source / "stu2ex_forget.json", [[0.1, 0.9], [0.8, 0.2]])
            self.write_text(
                source / "triples.txt",
                "kc0\tmlkc0.80\tuid0\n"
                "ex0\texfr0.10\tuid0\n"
                "uid0\trec\tex1\n",
            )
            self.write_text(
                source / "test_triples.txt",
                "kc1\tmlkc0.80\tuid1\n"
                "ex1\texfr0.10\tuid1\n",
            )
            feature_dir = source / "semantic_kg_features"
            feature_dir.mkdir()
            self.write_text(feature_dir / "marker.txt", "features")
            self.write_text(source / "statics2011_uid_kc_response.txt", "uid0\tkc0\t1\n")

            manifest = prepare_semantic_ablation_graph(
                source_dir=source,
                target_dir=target,
                ablation="feature_only_no_forgetting",
                top_k_rec=1,
                resume=False,
            )

            train_triples = (target / "triples.txt").read_text(encoding="utf-8")
            test_triples = (target / "test_triples.txt").read_text(encoding="utf-8")
            self.assertEqual(manifest["active_terms"], ["mastery"])
            self.assertNotIn("exfr", train_triples)
            self.assertNotIn("exfr", test_triples)
            self.assertIn("uid0\trec\tex0", train_triples)
            self.assertNotIn("uid0\trec\tex1", train_triples)
            self.assertTrue((target / "semantic_kg_features" / "marker.txt").exists())
            self.assertTrue((target / "statics2011_uid_kc_response.txt").exists())
            self.assertTrue((target / "stu2ex_recommend_full_precision.json").exists())


if __name__ == "__main__":
    unittest.main()
