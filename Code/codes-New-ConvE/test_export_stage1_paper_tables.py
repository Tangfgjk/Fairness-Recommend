"""Unit checks for stage-1 paper table exports."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from export_stage1_paper_tables import export_tables


class ExportStage1PaperTablesTest(unittest.TestCase):
    def test_export_filters_and_renames_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "comparison.csv"
            with source.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "method",
                        "K",
                        "Ada",
                        "NOV",
                        "ItemExposureGini",
                        "KCExposureGini",
                        "ItemCoverage",
                        "KCCoverage",
                        "LongTailItemExposureShare",
                        "LongTailKCExposureShare",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "baseline",
                        "K": "10",
                        "Ada": "0.7",
                        "NOV": "0.9",
                        "ItemExposureGini": "0.8",
                        "KCExposureGini": "0.6",
                        "ItemCoverage": "0.1",
                        "KCCoverage": "0.2",
                        "LongTailItemExposureShare": "0.3",
                        "LongTailKCExposureShare": "0.4",
                    }
                )

            paths = export_tables(source, root / "out", methods=["baseline"], top_ks=[10])

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["markdown"].exists())
            with paths["csv"].open("r", encoding="utf-8-sig") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(rows[0]["ItemGini"], "0.800000")
            self.assertEqual(rows[0]["KCGini"], "0.600000")
            self.assertIn("Top-10", paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
