from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import ROOT, read_json, sha256, digest


@dataclass
class Context:
    name: str
    data_root: Path
    graph: Path
    raw: Path
    q: np.ndarray
    mastery: np.ndarray
    forget: np.ndarray
    distance: np.ndarray
    train: np.ndarray
    test: np.ndarray
    positives: dict
    frequency: np.ndarray
    correct: dict
    ada_mastery: dict
    fingerprint: str
    files: dict
    difficulty: np.ndarray
    discrimination: np.ndarray

    def rule(self, uid, n=50):
        return np.argsort(self.distance[uid], kind="stable")[:n]

    def product(self, uid, exercises):
        return np.array([np.prod(self.mastery[uid, row > 0]) if row.any() else 0.0
                         for row in self.q[exercises]])


def load_context(name):
    from scripts.run_final_dropzero_three_seeds import dataset_config, validate_inputs
    cfg = dataset_config(name)
    validate_inputs(cfg)
    raw = cfg.data_root / name / ("raw_compact" if name == "XES3G5M-sub-small" else "raw")
    graph = cfg.graph
    q = np.loadtxt(graph / "Q.txt", delimiter=",", dtype=np.int8)
    if q.ndim != 2 or not np.isin(q, [0, 1]).all() or not (q.sum(1) > 0).all():
        raise ValueError("Q must be binary with at least one concept per exercise")
    mastery, forget, distance = [np.asarray(read_json(graph / f), dtype=np.float64) for f in
        ("stu2know_mastery.json", "stu2ex_forget.json", "stu2ex_recommend_full_precision.json")]
    with (raw / "student_split.csv").open(encoding="utf-8-sig", newline="") as f:
        split = list(csv.DictReader(f))
    groups = {s: np.array(sorted(int(r["entity_id"][3:]) for r in split if r["split"] == s), dtype=int)
              for s in ("train", "test")}
    if set(groups["train"]) & set(groups["test"]):
        raise ValueError("Overlapping train/test users")
    if any(not np.isfinite(x).all() for x in (mastery, forget, distance)):
        raise ValueError("Nonfinite educational context")
    if distance.shape != forget.shape or distance.shape != (mastery.shape[0], len(q)):
        raise ValueError("Educational matrix shape mismatch")
    positives = {int(u): [] for u in groups["train"]}
    frequency = np.zeros(len(q), dtype=np.int64)
    with (graph / "triples.txt").open(encoding="utf-8") as f:
        for line in f:
            h, r, t = line.split()
            if r == "rec":
                u, e = int(h[3:]), int(t[2:])
                if u not in positives or e in positives[u] or not 0 <= e < len(q):
                    raise ValueError("Invalid, duplicate or test rec edge")
                positives[u].append(e)
                frequency[e] += 1
    for u, es in positives.items():
        if len(es) != 50 or set(es) != set(np.argsort(distance[u], kind="stable")[:50]):
            raise ValueError(f"rec50 mismatch for uid{u}")
    correct = {int(u): set() for u in groups["test"]}
    with (raw / "interactions_all.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"].lower() == "test":
                u, e = int(r["entity_id"][3:]), int(r["question"])
                if u not in correct or not 0 <= e < len(q):
                    raise ValueError("History mapping mismatch")
                if int(float(r["response"])) == 1:
                    correct[u].update(np.flatnonzero(q[e]).tolist())
    ada = {int(u): {} for u in groups["test"]}
    with (graph / "test_triples.txt").open(encoding="utf-8") as f:
        for line in f:
            h, r, t = line.split()
            if r.startswith("mlkc"):
                ada[int(t[3:])][int(h[2:])] = float(r[4:])
    if any(len(v) != q.shape[1] for v in ada.values()):
        raise ValueError("Incomplete original Ada mastery relations")
    # Use the same normalized exercise IRT inputs as the original feature loader.
    from .io import model_modules
    import torch
    original = model_modules()
    bundle = original.load_semantic_feature_bundle(graph, None, device=torch.device("cpu"))
    sl = bundle.numeric_feature_slices["exercise_irt"]
    exids = [bundle.entity2id[f"ex{e}"] for e in range(len(q))]
    irt = bundle.numeric_features[exids, sl[0]:sl[1]].cpu().numpy()
    paths = [p for p in graph.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [raw / n for n in ("student_split.csv", "interactions_all.csv", "Q.txt")]
    paths += [p for p in (cfg.data_root / "_ep_readouts" / name).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    paths += [cfg.data_root / name / "front_features/ektm_mirt/best.pt"]
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}
    return Context(name, cfg.data_root, graph, raw, q, mastery, forget, distance,
                   groups["train"], groups["test"], positives, frequency, correct, ada,
                   digest(files), files, irt[:, 0], irt[:, 1])


def load_scores(path, context):
    entries = read_json(path)
    lookup = {}
    for row in entries:
        u = int(str(row["uid"]).removeprefix("uid"))
        if u in lookup:
            raise ValueError("Duplicate score UID")
        lookup[u] = row["scores"]
    if set(lookup) != set(context.test):
        raise ValueError("Score cohort differs from formal test cohort")
    scores = np.asarray([lookup[int(u)] for u in context.test], dtype=np.float64)
    if scores.shape != (len(context.test), len(context.q)) or not np.isfinite(scores).all():
        raise ValueError("Bad score matrix")
    return scores
