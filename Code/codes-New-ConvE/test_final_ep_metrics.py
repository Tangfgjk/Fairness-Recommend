from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from final_ep_metrics import is_selected_graph


class FinalEpMetricGraphSelectionTest(unittest.TestCase):
    def test_selected_ablation_uses_formal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected"
            graph = selected / "er_graph"
            ablation = root / "ablation"
            graph.mkdir(parents=True)
            ablation.mkdir()
            (ablation / "ablation_manifest.json").write_text(json.dumps({
                "ablation": "2CKG4ER_no_forgetting", "source_dir": str(graph)
            }), encoding="utf-8")
            selection = SimpleNamespace(root=selected, graph=graph)
            with patch("final_ep_metrics.select_dataset", return_value=selection):
                self.assertTrue(is_selected_graph(graph, "algebra2005"))
                self.assertTrue(is_selected_graph(ablation, "algebra2005"))
                (ablation / "ablation_manifest.json").write_text(json.dumps({
                    "ablation": "2CKG4ER_no_forgetting", "source_dir": str(root / "old_graph")
                }), encoding="utf-8")
                self.assertFalse(is_selected_graph(ablation, "algebra2005"))
                self.assertFalse(is_selected_graph(root / "unrelated", "algebra2005"))


if __name__ == "__main__":
    unittest.main()
