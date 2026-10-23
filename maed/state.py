"""Model-side educational inputs: deliberately no evaluation labels/history sets."""
from dataclasses import dataclass
import csv
import numpy as np

from debias.io import read_json, sha256, digest
from scripts.run_final_dropzero_three_seeds import dataset_config, validate_inputs


@dataclass
class EducationState:
    dataset: str
    graph: object
    data_root: object
    raw: object
    q: np.ndarray
    mastery: np.ndarray
    forget: np.ndarray
    distance: np.ndarray
    train: np.ndarray
    test: np.ndarray
    provenance: dict

    def training_distance(self, uids):
        uids = np.asarray(uids, dtype=int)
        if uids.ndim != 1 or len(uids) == 0 or not np.isin(uids, self.train).all():
            raise ValueError("Exposure targets must contain only nonempty training cohorts")
        if len(np.unique(uids)) != len(uids):
            raise ValueError("Repeated students change the exposure weighting")
        return self.distance[uids]


def education_distances(mastery, forget, q):
    if mastery.ndim != 2 or forget.shape != (len(mastery), len(q)):
        raise ValueError("Educational state dimensions differ")
    product = np.column_stack([np.prod(mastery[:, row > 0], axis=1) for row in q])
    return np.sqrt((.8 - product) ** 2 + (.8 - forget) ** 2)


def load_state(dataset):
    cfg = dataset_config(dataset)
    validate_inputs(cfg)
    raw = cfg.data_root / dataset / ("raw_compact" if dataset == "XES3G5M-sub-small" else "raw")
    q = np.loadtxt(cfg.graph / "Q.txt", delimiter=",", dtype=np.int8)
    if q.ndim != 2 or not np.isin(q, [0, 1]).all() or (q.sum(1) == 0).any():
        raise ValueError("Invalid Q matrix")
    paths = [cfg.graph / name for name in ("Q.txt", "stu2know_mastery.json", "stu2ex_forget.json",
                                          "stu2ex_recommend_full_precision.json")]
    mastery, forget, distance = [np.asarray(read_json(p), dtype=float) for p in paths[1:]]
    if mastery.shape[1] != q.shape[1]:
        raise ValueError("Mastery KC dimension differs")
    if any(not np.isfinite(x).all() for x in (mastery, forget, distance)):
        raise ValueError("Nonfinite state")
    if any((x < 0).any() or (x > 1).any() for x in (mastery, forget)):
        raise ValueError("Cognitive state outside [0,1]")
    calculated = education_distances(mastery, forget, q)
    # Original graph manifests distinguish full-precision distances from rounded storage states.
    if distance.shape != calculated.shape or not np.allclose(distance, calculated, atol=1e-6, rtol=0):
        raise ValueError("Distance is not the formal (0.8,0.8) educational distance")
    split_path = raw / "student_split.csv"
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {s: np.array(sorted(int(r["entity_id"][3:]) for r in rows if r["split"] == s), dtype=int)
              for s in ("train", "test")}
    if set(groups["train"]) & set(groups["test"]) or sorted(np.r_[groups['train'], groups['test']]) != list(range(len(mastery))):
        raise ValueError("Invalid student split")
    paths.append(split_path)
    files = {str(p): sha256(p) for p in paths}
    return EducationState(dataset, cfg.graph, cfg.data_root, raw, q, mastery, forget, distance,
                          groups['train'], groups['test'], {"files": files, "state_hash": digest(files),
                          "distance_reconstruction_max_error": float(np.max(np.abs(distance-calculated))),
                          "distance_reconstruction_tolerance": 1e-6,
                          "history_policy": "all_observed_history; endpoint snapshot, not a future-outcome protocol"})
