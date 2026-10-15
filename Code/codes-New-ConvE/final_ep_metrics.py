"""Paper Ep readout for recommendations on the selected final datasets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from final_dataset import DEFAULT_DATA_ROOT, select_dataset


THRESHOLDS = (0.4, 0.5, 0.6, 0.7)


def is_selected_graph(data_dir: Path, dataset: str) -> bool:
    """Include Pre graphs below the selected dataset, but exclude old Data_Fin."""
    if dataset not in {"Eedi", "algebra2005", "XES3G5M-sub-small"}:
        return False
    selection = select_dataset(dataset, DEFAULT_DATA_ROOT, validate=False)
    return Path(data_dir).resolve().is_relative_to(selection.root.resolve())


def evaluate_final_ep(dataset: str, scores_file: Path, output_dir: Path,
                      top_ks: Sequence[int], device: str) -> dict:
    selection = select_dataset(dataset)
    module_dir = Path(__file__).resolve().parents[1] / "mastery_ep_v2"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import torch
    from evaluate import evaluate_entry, load_context

    context = load_context(dataset, selection.root.parent, selection.ep_readouts,
                           torch.device(device))
    result = evaluate_entry(context, Path(scores_file), Path(output_dir) / "ep",
                            list(top_ks), list(THRESHOLDS))
    ep = {f"Ep_t{threshold:.1f}": {} for threshold in THRESHOLDS}
    for row in result["results"]:
        ep[f"Ep_t{row['threshold']:.1f}"][str(row['top_n'])] = {
            "mean": round(row["mean"], 6),
            "std": round(row["std_across_users"], 6) if row["std_across_users"] is not None else None,
        }
    return {**ep, "ep_thresholds": list(THRESHOLDS),
            "ep_calibration_root": str(selection.ep_readouts)}
