"""Unit checks for experiment fairness comparison tables."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compare_fairness_results import add_baseline_deltas, collect_rows, main


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_eval_dir(root: Path, dirname: str, ada: float, item_coverage: float) -> None:
    write_json(
        root / dirname / "metrics.json",
        {
            "dataset": "Tiny",
            "model": dirname,
            "seed": 2024,
            "Ada": {"10": {"mean": ada, "std": 0.1}},
            "NOV": {"10": {"mean": 0.9, "std": 0.2}},
        },
    )
    write_json(
        root / dirname / "fairness_metrics.json",
        {
            "top_k": {
                "10": {
                    "ItemExposureGini": 0.8,
                    "KCExposureGini": 0.7,
                    "ItemCoverage": item_coverage,
                    "KCCoverage": 0.4,
                    "LongTailItemExposureShare": 0.2,
                    "LongTailKCExposureShare": 0.3,
                    "HeadItemExposureShare": 0.8,
                    "HeadKCExposureShare": 0.7,
                    "recommendations": 10,
                }
            }
        },
    )


class CompareFairnessResultsTest(unittest.TestCase):
    def test_collect_rows_scans_eval_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_eval_dir(root, "eval", ada=0.7, item_coverage=0.1)
            write_eval_dir(root, "eval_quota_ratio_hybrid", ada=0.68, item_coverage=0.25)

            rows = collect_rows(root, top_ks=[10])

            self.assertEqual([row["method"] for row in rows], ["baseline", "quota_ratio_hybrid"])
            self.assertEqual(rows[0]["Ada"], 0.7)
            self.assertEqual(rows[1]["ItemCoverage"], 0.25)

    def test_add_baseline_deltas(self) -> None:
        rows = [
            {"method": "baseline", "K": 10, "Ada": 0.7, "NOV": 0.9, "ItemCoverage": 0.1},
            {"method": "rerank", "K": 10, "Ada": 0.68, "NOV": 0.91, "ItemCoverage": 0.25},
        ]

        delta_rows = add_baseline_deltas(rows)
        rerank = delta_rows[1]

        self.assertAlmostEqual(rerank["Delta_Ada"], -0.02)
        self.assertAlmostEqual(rerank["Delta_NOV"], 0.01)
        self.assertAlmostEqual(rerank["Delta_ItemCoverage"], 0.15)

    def test_cli_writes_comparison_files(self) -> None:
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "comparison"
            write_eval_dir(root, "eval", ada=0.7, item_coverage=0.1)
            write_eval_dir(root, "eval_item_only", ada=0.69, item_coverage=0.2)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "compare_fairness_results.py",
                    "--run-dir",
                    str(root),
                    "--dataset-name",
                    "Tiny",
                    "--seed",
                    "2024",
                    "--ks",
                    "10",
                    "--output-dir",
                    str(output_dir),
                ]
                main()
            finally:
                sys.argv = old_argv

            self.assertTrue((output_dir / "fairness_comparison.csv").exists())
            self.assertTrue((output_dir / "fairness_comparison_delta.csv").exists())
            self.assertTrue((output_dir / "fairness_comparison.md").exists())
            markdown = (output_dir / "fairness_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Tiny seed2024 Fairness Comparison", markdown)
            self.assertIn("item_only", markdown)


if __name__ == "__main__":
    unittest.main()
