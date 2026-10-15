"""Build a leakage-safe, KC-centric message graph from an existing ER graph.

The original graph remains the source of recommendation labels and evaluation
files. This sidecar contains only edges available before a recommendation is
predicted; in particular, rec labels never enter message passing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_loader import read_dict


RELATIONS = (
    "mastery", "inv_mastery", "forgetting", "inv_forgetting",
    "covers", "covered_by",
)


def build_graph(graph_dir: Path, output: Path, *, keep_zero_forgetting: bool = False) -> dict[str, int]:
    entities = read_dict(graph_dir / "entities.dict")
    q = np.loadtxt(graph_dir / "Q.txt", delimiter=",", dtype=np.float32, ndmin=2)
    mastery = np.asarray(json.loads((graph_dir / "stu2know_mastery.json").read_text(encoding="utf-8")), dtype=np.float32)
    forgetting = np.asarray(json.loads((graph_dir / "stu2know_forget.json").read_text(encoding="utf-8")), dtype=np.float32)
    if mastery.shape != forgetting.shape or q.shape[1] != mastery.shape[1]:
        raise ValueError("Q, mastery and forgetting dimensions are inconsistent")
    if not all(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in (mastery, forgetting)):
        raise ValueError("mastery and forgetting must contain finite values in [0, 1]")
    if not np.isfinite(q).all():
        raise ValueError("Q contains non-finite values")

    # The original triples file contains train learners only. Never infer the
    # propagation cohort from the all-student state matrices.
    train_users: set[int] = set()
    with (graph_dir / "triples.txt").open(encoding="utf-8") as stream:
        for line in stream:
            h, _r, t = line.rstrip("\n").split("\t")
            if h.startswith("uid"):
                train_users.add(int(h[3:]))
            if t.startswith("uid"):
                train_users.add(int(t[3:]))
    if not train_users:
        raise ValueError("training triples contain no learner")
    if max(train_users) >= mastery.shape[0]:
        raise ValueError("train learner id exceeds state matrix")

    sources: list[int] = []
    targets: list[int] = []
    kinds: list[int] = []
    strengths: list[float] = []

    def add_pair(src: int, dst: int, relation: int, strength: float) -> None:
        sources.extend((src, dst))
        targets.extend((dst, src))
        kinds.extend((relation, relation + 1))
        strengths.extend((strength, strength))

    for uid in sorted(train_users):
        learner = entities[f"uid{uid}"]
        for kc in range(mastery.shape[1]):
            concept = entities[f"kc{kc}"]
            add_pair(learner, concept, 0, float(mastery[uid, kc]))
            value = float(forgetting[uid, kc])
            if keep_zero_forgetting or value > 0:
                add_pair(learner, concept, 2, value)
    for ex, kc in np.argwhere(q > 0):
        add_pair(entities[f"ex{ex}"], entities[f"kc{kc}"], 4, 1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        relation=np.asarray(kinds, dtype=np.int64),
        strength=np.asarray(strengths, dtype=np.float32),
        nentity=np.asarray(len(entities), dtype=np.int64),
    )
    return {"train_learners": len(train_users), "entities": len(entities), "message_edges": len(sources)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KC-centric Educational KG v2 message graph")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-zero-forgetting", action="store_true")
    args = parser.parse_args()
    print(build_graph(args.graph_dir, args.output, keep_zero_forgetting=args.keep_zero_forgetting))


if __name__ == "__main__":
    main()
