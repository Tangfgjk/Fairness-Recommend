"""Validated state graphs. Edges are explicitly source -> receiver, never rec.

B is rebuilt from the clean training graph for ONE query. Stored matrices used
by MAED Q are checked against every state triple before a graph is exposed.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

import numpy as np
import torch

from debias.io import digest, sha256

POLICY = "train_nonrec_bidirectional_symmetric_degree__B_single_known_state_v1"
STATE_RELATION = re.compile(r"^(mlkc|exfr)(0\.\d{2}|1\.00)$")


def _array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class MessageGraph:
    """CPU construction, device view via to(); tensor contents must not be edited."""
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    direction: torch.Tensor
    family: torch.Tensor  # 0 mastery, 1 forgetting (same for inverse)
    degree: torch.Tensor
    graph_id: str
    base_id: str
    protocol: str = "train"
    query_uid: int | None = None
    query_entity: int | None = None
    audit: dict = field(default_factory=dict)

    @property
    def nentity(self):
        return self.degree.numel()

    def to(self, device):
        from dataclasses import replace
        return replace(self, **{k: getattr(self, k).to(device) for k in
                              ("source", "target", "relation", "direction", "family", "degree")})

    def tensor_versions(self):
        return tuple(getattr(self, k)._version for k in
                     ("source", "target", "relation", "direction", "family", "degree"))


def make_graph(rows, nentity, inverse_ids, *, base_id=None, protocol="train",
               query_uid=None, query_entity=None, audit=None):
    """Construct symmetric graph from validated ORIGINAL direction rows (s,r,t,f)."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    if len(rows):
        if np.any(rows[:, [0, 2]] < 0) or np.any(rows[:, [0, 2]] >= nentity):
            raise ValueError("Endpoint outside full entity ID space")
        if any(int(r) not in inverse_ids for r in rows[:, 1]):
            raise ValueError("rec/unknown relation cannot enter message graph")
        if np.any((rows[:, 3] < 0) | (rows[:, 3] > 1)):
            raise ValueError("Unknown state family")
        if len(np.unique(rows[:, :3], axis=0)) != len(rows):
            raise ValueError("Duplicate original state edges")
    s, r, t, f = rows.T
    src = np.r_[s, t]; dst = np.r_[t, s]
    rel = np.r_[r, [inverse_ids[int(v)] for v in r]].astype(np.int64)
    direction = np.r_[np.zeros(len(rows), dtype=np.int64), np.ones(len(rows), dtype=np.int64)]
    family = np.r_[f, f]
    degree = np.bincount(src, minlength=nentity)
    identity = digest({"policy": POLICY, "edges": _array_hash(rows), "entities": nentity,
                       "protocol": protocol, "query": query_uid,
                       "inverse_ids": sorted(inverse_ids.items()), "sources": audit or {}})
    return MessageGraph(*(torch.from_numpy(a) for a in (src, dst, rel, direction, family, degree)),
                        identity, base_id or identity, protocol, query_uid, query_entity, audit or {})


class GraphProtocol:
    """Read-once validator; input provenance and query isolation live behind one interface."""
    def __init__(self, state, entity2id, relation2id):
        self.state = state
        self.entity2id = dict(entity2id); self.relation2id = dict(relation2id)
        for mapping in (self.entity2id, self.relation2id):
            if sorted(mapping.values()) != list(range(len(mapping))):
                raise ValueError("Original IDs must be dense and unmodified")
        expected = {"rec"} | {f"{p}{i/100:.2f}" for p in ("mlkc", "exfr") for i in range(101)}
        if set(self.relation2id) != expected:
            raise ValueError("Expected original 203 relation dictionary")
        self.inverse_ids = {r: len(relation2id)+i for i, r in enumerate(sorted(
            r for name, r in relation2id.items() if name != "rec"))}
        self.inverse_original_ids = tuple(sorted(self.inverse_ids))
        self.train_uids = frozenset(map(int, state.train)); self.test_uids = frozenset(map(int, state.test))
        if self.train_uids & self.test_uids:
            raise ValueError("Student splits overlap")
        for u in self.train_uids | self.test_uids:
            if f"uid{u}" not in entity2id:
                raise ValueError("Student absent from entity dictionary")
        self.test_entities = frozenset(entity2id[f"uid{u}"] for u in self.test_uids)
        self._round_boundary_count = 0
        self._round_boundary_examples = []
        train_rows, train_audit = self._read(Path(state.graph)/"triples.txt", self.train_uids, True)
        queries, test_audit = self._read(Path(state.graph)/"test_triples.txt", self.test_uids, False)
        files = {str(Path(state.graph)/p): sha256(Path(state.graph)/p) for p in
                 ("triples.txt", "test_triples.txt", "entities.dict", "relations.dict")}
        self.manifest = {"policy": POLICY, "state_hash": state.provenance["state_hash"],
                         "state_sources": state.provenance["files"], "graph_sources": files,
                         "train": train_audit, "test": test_audit,
                         "rounding_boundary_compatible": self._round_boundary_count,
                         "rounding_boundary_examples": self._round_boundary_examples,
                         "input_contract": "known pre-recommendation endpoint state; inherited temporal limitations",
                         "inverse_mapping": {str(r): v for r, v in self.inverse_ids.items()}}
        self._train_rows = np.array(train_rows, dtype=np.int64).reshape(-1, 4)
        self._queries = {u: [] for u in self.test_uids}
        for row in queries:
            u = int(self._id2entity(row[2])[3:])
            self._queries[u].append(row)
        self.base = make_graph(self._train_rows, len(entity2id), self.inverse_ids,
                               audit=self.manifest)
        if any(int(self.base.degree[e]) != 0 for e in self.test_entities):
            raise ValueError("Test student entered training message graph")

    def _id2entity(self, idx):
        if not hasattr(self, "_names"):
            self._names = {v: k for k, v in self.entity2id.items()}
        return self._names[int(idx)]

    def _read(self, path, allowed, train):
        rows = []; seen = set(); observed = {u: {"mlkc": set(), "exfr": set()} for u in allowed}
        counts = {"mlkc": 0, "exfr": 0, "rec_supervision_excluded": 0, "exfr_zero_filtered": 0}
        with path.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                fields = line.strip().split()
                if len(fields) != 3:
                    raise ValueError(f"Invalid triple {path}:{line_no}")
                h, name, t = fields
                if h not in self.entity2id or t not in self.entity2id or name not in self.relation2id:
                    raise ValueError(f"Unknown entity/relation {path}:{line_no}")
                key = (h, name, t)
                if key in seen:
                    raise ValueError(f"Duplicate triple {path}:{line_no}")
                seen.add(key)
                if name == "rec":
                    if not train or not h.startswith("uid") or int(h[3:]) not in allowed or not t.startswith("ex"):
                        raise ValueError("rec in query file or wrong supervised split/endpoints")
                    counts["rec_supervision_excluded"] += 1
                    continue
                match = STATE_RELATION.fullmatch(name)
                if not match or not t.startswith("uid") or int(t[3:]) not in allowed:
                    raise ValueError(f"State relation/split violation {path}:{line_no}")
                kind, level = match.groups(); u = int(t[3:]); prefix = "kc" if kind == "mlkc" else "ex"
                if not h.startswith(prefix):
                    raise ValueError("State endpoint kind mismatch")
                j = int(h[len(prefix):]); matrix = self.state.mastery if kind == "mlkc" else self.state.forget
                if j >= matrix.shape[1] or j < 0 or j in observed[u][kind]:
                    raise ValueError("Duplicate/missing state coordinate")
                observed[u][kind].add(j)
                value = float(matrix[u, j]); quantized = float(level)
                # Stored Q matrices have 6 decimal places, triples were generated before
                # rounding. Only an actual half-cent boundary may disagree in formatting.
                if f"{value:.2f}" != level:
                    if abs(abs(value-quantized)-.005) > 5.1e-7:
                        raise ValueError(f"Q/state snapshot mismatch {path}:{line_no}")
                    self._round_boundary_count += 1
                    if len(self._round_boundary_examples) < 30:
                        self._round_boundary_examples.append({"file":str(path),"line":line_no,
                            "student":u,"source":h,"stored_Q_value":value,"relation":name,
                            "reason":"within original half-cent boundary + six-decimal storage rounding"})
                counts[kind] += 1
                if kind == "exfr" and level == "0.00":
                    counts["exfr_zero_filtered"] += 1
                    if train:
                        raise ValueError("Training graph must already have exfr0.00 removed")
                    continue
                rows.append((self.entity2id[h], self.relation2id[name], self.entity2id[t], int(kind == "exfr")))
        for u, kinds in observed.items():
            if len(kinds["mlkc"]) != self.state.mastery.shape[1]:
                raise ValueError("Incomplete mastery state")
            if not train and len(kinds["exfr"]) != self.state.forget.shape[1]:
                raise ValueError("Incomplete query forgetting state")
            if train:
                missing = set(range(self.state.forget.shape[1]))-kinds["exfr"]
                if any(float(self.state.forget[u, j]) > .00500051 for j in missing):
                    raise ValueError("Training nonzero forgetting edge missing")
        counts["retained_original"] = len(rows)
        return rows, counts

    def query(self, uid):
        uid = int(uid)
        if uid not in self.test_uids:
            raise ValueError("B query must be one declared test student")
        extra = np.asarray(self._queries[uid], dtype=np.int64).reshape(-1, 4)
        if (extra[:, 2] != self.entity2id[f"uid{uid}"]).any():
            raise ValueError("Other student's state entered current query delta")
        metadata = {"base_id": self.base.graph_id, "state_hash": self.state.provenance["state_hash"],
                    "query_state_hash": _array_hash(extra), "retained_query_edges": len(extra),
                    "mlkc": int((extra[:, 3] == 0).sum()), "exfr": int((extra[:, 3] == 1).sum()),
                    "zero_exfr_filtered": self.state.forget.shape[1]-int((extra[:, 3] == 1).sum())}
        # Query targets and training targets belong to disjoint validated splits.
        # Thus no base edge can duplicate a query edge; only the small delta needs
        # construction/duplicate checking, not a repeated O(E log E) base sort.
        delta = make_graph(extra, len(self.entity2id), self.inverse_ids, audit=metadata)
        identity = digest({"base_id": self.base.graph_id, "delta_id": delta.graph_id,
                           "protocol": "B", "query": uid})
        return MessageGraph(*(torch.cat((getattr(self.base,k),getattr(delta,k))) for k in
                             ("source","target","relation","direction","family")),
                            self.base.degree+delta.degree, identity, self.base.graph_id,
                            "B", uid, self.entity2id[f"uid{uid}"], metadata)
