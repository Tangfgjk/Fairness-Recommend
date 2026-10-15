"""Selected 2026-10-15 data and graph locations for the fairness experiments.

The source datasets live in this repository's Code/Data_Fin_New directory.
Earlier Code/Data_Fin runs remain available only through an explicit data root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "Data_Fin_New"


@dataclass(frozen=True)
class DatasetSelection:
    name: str
    root: Path
    graph: Path
    raw: Path
    ep_readouts: Path
    train_students: int
    test_students: int
    variant: str


_SELECTED = {
    "Eedi": ("_resplit_seed20261007/Eedi", "forgetting_variants/drop_exfr0p00_rec50/er_graph", "raw", 701, 234, "drop_exfr0p00_rec50"),
    "algebra2005": ("algebra2005", "forgetting_variants_rec50/drop_exfr0p00/er_graph", "raw", 430, 144, "drop_exfr0p00"),
    "XES3G5M-sub-small": ("XES3G5M-sub-small", "forgetting_variants_rec50/drop_exfr0p00/er_graph", "raw_compact", 727, 241, "drop_exfr0p00"),
}


def is_final_root(data_root: Path) -> bool:
    return Path(data_root).resolve() == DEFAULT_DATA_ROOT.resolve()


def dataset_root(dataset: str, data_root: Path) -> Path:
    data_root = Path(data_root)
    return select_dataset(dataset, data_root, validate=False).root if is_final_root(data_root) else data_root / dataset


def select_dataset(dataset: str, data_root: Path = DEFAULT_DATA_ROOT, *, validate: bool = True) -> DatasetSelection:
    if dataset not in _SELECTED:
        raise ValueError(f"No final dataset selection for {dataset!r}")
    relative_root, relative_graph, raw_name, train_count, test_count, variant = _SELECTED[dataset]
    data_root = Path(data_root)
    root = data_root / relative_root
    calibration_root = data_root / "_resplit_seed20261007" / "_ep_readouts" if dataset == "Eedi" else data_root / "_ep_readouts"
    selection = DatasetSelection(dataset, root, root / relative_graph, root / raw_name,
                                 calibration_root, train_count, test_count, variant)
    if validate:
        validate_selection(selection)
    return selection


def validate_selection(selection: DatasetSelection) -> None:
    required_graph = ("triples.txt", "test_triples.txt", "Q.txt", "entities.dict", "relations.dict",
                      "stu2know_mastery.json", "stu2know_forget.json", "stu2ex_forget.json",
                      "stu2ex_recommend_full_precision.json", "er_graph_manifest.json")
    missing = [str(selection.graph / name) for name in required_graph if not (selection.graph / name).is_file()]
    required_other = (selection.raw / "interactions_all.csv", selection.raw / "student_split.csv",
                      selection.root / "front_features" / "ektm_mirt" / "best.pt",
                      selection.ep_readouts / selection.name / "readout.pt",
                      selection.ep_readouts / selection.name / "training_report.json")
    missing.extend(str(path) for path in required_other if not path.is_file())
    if missing:
        raise FileNotFoundError("Final dataset is incomplete:\n" + "\n".join(missing))
    manifest = json.loads((selection.graph / "er_graph_manifest.json").read_text(encoding="utf-8"))
    rec = manifest.get("recommendation", {})
    counts = manifest.get("student_counts", {})
    variant = manifest.get("forgetting_edge_variant", {})
    if (manifest.get("dataset") != selection.name
            or (rec.get("delta_1"), rec.get("delta_2"), rec.get("top_k_rec")) != (0.8, 0.8, 50)
            or counts != {"train": selection.train_students, "test": selection.test_students}
            or variant.get("name") != selection.variant
            or manifest.get("triple_counts", {}).get("train", {}).get("exfr") != variant.get("retained_exfr_edges")):
        raise ValueError(f"Final graph manifest does not match selected dataset: {selection.graph}")
