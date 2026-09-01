"""Unit checks for 2CKG4ER summary helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_semantic_results import collect_gate_rows, collect_seed, summarize_gate_rows


class SemanticSummaryTest(unittest.TestCase):
    def write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_collect_seed_reads_ablation_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            seed_dir = run_dir / "2CKG4ER" / "seed2024"
            self.write_json(
                seed_dir / "eval" / "metrics.json",
                {
                    "Ada": {"10": {"mean": 0.7}},
                    "NOV": {"10": {"mean": 0.9}},
                    "Fairness": {"10": {"ItemCoverage": 0.5, "KCExposureGini": 0.25}},
                },
            )
            self.write_json(seed_dir / "metrics.json", {"training_seconds": 12.5})
            self.write_json(seed_dir / "2ckg4er_inference.json", {"inference_seconds": 1.25})

            row = collect_seed(run_dir, 2024, [10], "2CKG4ER")

            self.assertIsNotNone(row)
            self.assertEqual(row["ablation"], "2CKG4ER")
            self.assertEqual(row["Ada@10"], 0.7)
            self.assertEqual(row["NOV@10"], 0.9)
            self.assertEqual(row["ItemCoverage@10"], 0.5)
            self.assertEqual(row["KCExposureGini@10"], 0.25)
            self.assertEqual(row["training_seconds"], 12.5)
            self.assertNotIn("Ada-Avg", row)
            self.assertNotIn("NOV-Avg", row)

    def test_collects_gate_values_per_seed_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            seed2024 = run_dir / "2CKG4ER" / "seed2024"
            seed2025 = run_dir / "2CKG4ER" / "seed2025"
            self.write_json(
                seed2024 / "gate_values.json",
                {"gate_values": {"uid": {"semantic": 0.4, "pedagogical": 0.6, "cluster": 0.8}}},
            )
            self.write_json(
                seed2025 / "gate_values.json",
                {"gate_values": {"uid": {"semantic": 0.6, "pedagogical": 0.8, "cluster": 1.0}}},
            )

            rows = collect_gate_rows(run_dir, [2024, 2025], ["2CKG4ER"])
            summaries = summarize_gate_rows(rows)

            self.assertEqual(len(rows), 6)
            uid_semantic = next(
                item for item in summaries if item["entity_type"] == "uid" and item["feature_type"] == "semantic"
            )
            self.assertAlmostEqual(uid_semantic["mean"], 0.5)
            self.assertAlmostEqual(uid_semantic["std"], 0.14142135623730948)


if __name__ == "__main__":
    unittest.main()
