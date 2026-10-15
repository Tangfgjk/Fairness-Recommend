from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from bias_chain_diagnostics import build_diagnostics


class BiasChainDiagnosticsTest(unittest.TestCase):
    def test_build_diagnostics_compares_data_label_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "er_graph"
            raw_dir = root / "raw"
            output_dir = root / "diagnostics"
            data_dir.mkdir()
            raw_dir.mkdir()
            (data_dir / "Q.txt").write_text("1,0\n0,1\n1,1\n", encoding="utf-8")
            (data_dir / "triples.txt").write_text(
                "\n".join(
                    [
                        "uid0\trec\tex0",
                        "uid1\trec\tex0",
                        "uid2\trec\tex2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (raw_dir / "interactions_all.csv").write_text(
                "\n".join(
                    [
                        "uid,entity_id,question,entity_exercise_id,split",
                        "0,uid0,0,ex0,train",
                        "1,uid1,1,ex1,train",
                        "2,uid2,1,ex1,train",
                        "3,uid3,2,ex2,test",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            scores_file = root / "scores.json"
            scores_file.write_text(
                json.dumps(
                    [
                        {"uid": "uid0", "scores": [0.9, 0.1, 0.8]},
                        {"uid": "uid1", "scores": [0.3, 0.7, 0.6]},
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_dir=data_dir,
                output_dir=output_dir,
                base_scores_file=scores_file,
                debiased_scores_file=None,
                top_ks=[1, 2],
                popularity_aggregation="unique_users",
                head_ratio=1 / 3,
                tail_ratio=1 / 3,
            )

            diagnostics = build_diagnostics(args)

            rows = diagnostics["rows"]
            stages = {(row["target"], row["stage"], row["top_k"]) for row in rows}
            self.assertIn(("item", "p_data", ""), stages)
            self.assertIn(("item", "p_label", ""), stages)
            self.assertIn(("item", "p_base", 1), stages)
            self.assertIn(("kc", "p_base", 2), stages)
            first_item = next(row for row in rows if row["target"] == "item" and row["stage"] == "p_label")
            self.assertIn("JS_vs_p_data", first_item)
            self.assertIn("MidExposureShare", first_item)
            self.assertEqual(diagnostics["metadata"]["p_data_source"]["source"], "train_interactions")


if __name__ == "__main__":
    unittest.main()
