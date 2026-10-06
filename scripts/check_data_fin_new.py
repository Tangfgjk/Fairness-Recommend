"""Verify the three rebuilt datasets before training new recommendation models."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes-New-ConvE"))
sys.path.insert(0, str(ROOT / "comparison_models"))
sys.path.insert(0, str(ROOT / "mastery_ep_v2"))
from validate_semantic_ready import validate_dataset  # noqa: E402
from run_dataset_experiments import validate_data_dir, resolve_raw_dir_for_graph  # noqa: E402
from evaluate import load_context  # noqa: E402
DATASETS = ("Eedi", "algebra2005", "XES3G5M-sub-small")
CHANGED_FILES = (
    "front_features/stu2know_mastery.json",
    "er_graph/stu2know_mastery.json",
    "er_graph/stu2ex_recommend.json",
    "er_graph/stu2ex_recommend_full_precision.json",
    "er_graph/triples.txt",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(dataset: str) -> dict:
    old = ROOT / "Data_Fin" / dataset
    new = ROOT / "Data_Fin_New" / dataset
    graph = new / "er_graph"
    front = new / "front_features"
    changed = {name: digest(old / name) != digest(new / name) for name in CHANGED_FILES}
    if not all(changed.values()):
        raise AssertionError(f"{dataset}: expected rebuilt files to differ: {changed}")
    graph_mastery = np.asarray(read_json(graph / "stu2know_mastery.json"), dtype=np.float64)
    front_mastery = np.asarray(read_json(front / "stu2know_mastery.json"), dtype=np.float64)
    if graph_mastery.shape != front_mastery.shape or not np.allclose(graph_mastery, front_mastery, atol=5e-7):
        raise AssertionError(f"{dataset}: graph/front mastery mismatch")
    full = np.asarray(read_json(graph / "stu2ex_recommend_full_precision.json"), dtype=np.float64)
    rounded = np.asarray(read_json(graph / "stu2ex_recommend.json"), dtype=np.float64)
    if full.shape != rounded.shape or not np.allclose(full, rounded, atol=5e-7):
        raise AssertionError(f"{dataset}: full/rounded recommendation scores mismatch")
    manifest = read_json(graph / "er_graph_manifest.json")
    top_k = int(manifest["recommendation"]["top_k_rec"])
    if top_k != 10:
        raise AssertionError(f"{dataset}: unexpected recommendation edge count {top_k}")
    rec_edges: dict[int, list[int]] = {}
    for line in (graph / "triples.txt").read_text(encoding="utf-8").splitlines():
        head, relation, tail = line.split("\t")
        if relation == "rec":
            rec_edges.setdefault(int(head[3:]), []).append(int(tail[2:]))
    if not rec_edges or len(rec_edges) != manifest["student_counts"]["train"]:
        raise AssertionError(f"{dataset}: recommendation edge cohort mismatch")
    for uid, actual in rec_edges.items():
        expected = np.lexsort((np.arange(full.shape[1]), full[uid]))[:top_k].tolist()
        if actual != expected:
            raise AssertionError(f"{dataset}: recommendation edges differ for uid{uid}")
    front_semantic = front / "semantic_kg_features"
    graph_semantic = graph / "semantic_kg_features"
    feature_files = [path.relative_to(front_semantic) for path in front_semantic.rglob("*") if path.is_file()]
    if not feature_files or any(digest(front_semantic / rel) != digest(graph_semantic / rel) for rel in feature_files):
        raise AssertionError(f"{dataset}: graph semantic features differ from rebuilt front features")
    readout = ROOT / "Data_Fin_New" / "_ep_readouts" / dataset
    if any(not (readout / name).is_file() for name in
           ("readout.pt", "full_history_hidden.npy", "stu2know_mastery.npy", "training_report.json")):
        raise AssertionError(f"{dataset}: missing calibrated Ep readout")
    exported = np.load(readout / "stu2know_mastery.npy")
    if exported.shape != front_mastery.shape or not np.allclose(exported, front_mastery, atol=5e-7):
        raise AssertionError(f"{dataset}: graph mastery differs from calibrated Ep readout")
    validation = validate_dataset(dataset, ROOT / "Data_Fin_New", graph_subdir="er_graph")
    if validation["status"] != "passed":
        raise AssertionError(f"{dataset}: main-model input validation failed: {validation['errors']}")
    validate_data_dir(graph)
    raw = resolve_raw_dir_for_graph(graph)
    if raw is None or not raw.resolve().is_relative_to(new.resolve()):
        raise AssertionError(f"{dataset}: baseline raw input does not come from Data_Fin_New")
    for name in ("interactions_all.csv", "student_split.csv", "Q.txt"):
        if not (raw / name).is_file():
            raise AssertionError(f"{dataset}: missing baseline raw input {name}")
    context = load_context(dataset, ROOT / "Data_Fin_New",
                           ROOT / "Data_Fin_New" / "_ep_readouts", torch.device("cpu"))
    if context.mastery.shape != graph_mastery.shape:
        raise AssertionError(f"{dataset}: Ep readout/student dimension mismatch")
    return {"dataset": dataset, "mastery_shape": list(graph_mastery.shape),
            "recommendation_score_shape": list(full.shape),
            "train_users_with_recommendation_edges": len(rec_edges),
            "semantic_files_checked": len(feature_files), "all_rebuilt_files_changed": True,
            "main_inputs": "passed", "comparison_inputs": "passed", "ep_readout": "passed"}


def main() -> None:
    print(json.dumps([check(name) for name in DATASETS], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
