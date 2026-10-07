"""Create train-only exfr edge variants for Data_Fin_New base graphs.

Only triples.txt changes. The test graph, state matrices, recommendation scores,
entity/relation dictionaries, and semantic features are copied verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATASETS = ("Eedi", "algebra2005", "XES3G5M-sub-small")
VARIANTS = ("drop_exfr0p00", "weighted_nonzero_400", "weighted_nonzero_200")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "Data_Fin_New"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_exfr(line: str) -> tuple[int, int, str] | None:
    head, relation, tail = line.rstrip("\n").split("\t")
    if not relation.startswith("exfr"):
        return None
    if not (head.startswith("ex") and tail.startswith("uid")):
        raise ValueError(f"Bad exfr triple: {line[:100]}")
    return int(tail[3:]), int(head[2:]), relation


def weighted_rank(items: list[tuple[int, float]], seed: int) -> list[int]:
    """Weighted random permutation; higher forgetting is more likely early."""
    if not items:
        return []
    exercises = np.asarray([ex for ex, _ in items], dtype=np.int64)
    values = np.asarray([value for _, value in items], dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("Weighted candidates must have positive forgetting")
    rng = np.random.default_rng(seed)
    u = np.clip(rng.random(len(items)), np.finfo(float).tiny, 1.0)
    keys = -np.log(u) / (values * values)
    return exercises[np.argsort(keys, kind="stable")].tolist()


def collect(source: Path, seed: int, dataset_index: int):
    matrix = np.asarray(json.loads((source / "stu2ex_forget.json").read_text(encoding="utf-8")))
    candidates = defaultdict(list)
    seen_pairs = set()
    train_users = set()
    source_counts = Counter()
    exact_zero, quantized_zero_nonzero = 0, 0
    with (source / "triples.txt").open(encoding="utf-8") as stream:
        for line in stream:
            parsed = parse_exfr(line)
            if parsed is None:
                source_counts["other"] += 1
                continue
            uid, ex, relation = parsed
            train_users.add(uid)
            if not (0 <= uid < matrix.shape[0] and 0 <= ex < matrix.shape[1]):
                raise ValueError(f"exfr index out of range: {uid}, {ex}")
            if (uid, ex) in seen_pairs:
                raise ValueError(f"duplicate exfr triple: {uid}, {ex}")
            seen_pairs.add((uid, ex))
            value = float(matrix[uid, ex])
            source_counts["exfr"] += 1
            if value == 0:
                exact_zero += 1
            if relation == "exfr0.00":
                source_counts["exfr0.00"] += 1
                if value > 0:
                    quantized_zero_nonzero += 1
            else:
                if value <= 0:
                    raise ValueError(f"Nonzero exfr label with zero stored value: {uid}, {ex}")
                candidates[uid].append((ex, value))
    rank = {
        uid: weighted_rank(candidates[uid], seed + dataset_index * 1_000_003 + uid)
        for uid in sorted(train_users)
    }
    top400 = {(uid, ex) for uid, values in rank.items() for ex in values[:400]}
    top200 = {(uid, ex) for uid, values in rank.items() for ex in values[:200]}
    eligible = {(uid, ex) for uid, values in rank.items() for ex in values}
    if not top200 <= top400 <= eligible:
        raise AssertionError("Expected nested weighted samples")
    return matrix, source_counts, exact_zero, quantized_zero_nonzero, eligible, top400, top200, rank


def copy_graph_without_train(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in {"triples.txt", "er_graph_manifest.json"}:
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)


def build_dataset(data_root: Path, dataset: str, dataset_index: int, seed: int, plan: bool = False) -> None:
    source = data_root / dataset / "er_graph"
    manifest = json.loads((source / "er_graph_manifest.json").read_text(encoding="utf-8"))
    params = manifest["recommendation"]
    if (params["delta_1"], params["delta_2"], params["top_k_rec"]) != (0.8, 0.8, 10):
        raise ValueError(f"Base graph params differ from (0.8,0.8,10): {dataset}")
    matrix, source_counts, exact_zero, quantized_zero_nonzero, eligible, top400, top200, rank = collect(source, seed, dataset_index)
    users = len(rank)
    if users != int(manifest["student_counts"]["train"]):
        raise ValueError(f"Expected {manifest['student_counts']['train']} train users, got {users}")
    if source_counts["exfr"] != manifest["triple_counts"]["train"]["exfr"]:
        raise ValueError(f"Source exfr count mismatch: {dataset}")
    selected = {
        "drop_exfr0p00": eligible,
        "weighted_nonzero_400": top400,
        "weighted_nonzero_200": top200,
    }
    if plan:
        print(dataset, "train_students", users, "source_exfr", source_counts["exfr"],
              "exact_zero", exact_zero, "exfr0.00", source_counts["exfr0.00"],
              "quantized_nonzero", quantized_zero_nonzero)
        for name in VARIANTS:
            mean_value = sum(float(matrix[uid, ex]) for uid, ex in selected[name]) / max(1, len(selected[name]))
            print("  ", name, "retained", len(selected[name]),
                  "retention_pct", round(100 * len(selected[name]) / source_counts["exfr"], 2),
                  "mean_forgetting", round(mean_value, 4),
                  "min_max", min(min(len(v), 200 if name.endswith("200") else 400 if name.endswith("400") else math.inf) for v in rank.values()),
                  max(min(len(v), 200 if name.endswith("200") else 400 if name.endswith("400") else math.inf) for v in rank.values()))
        return
    variant_root = data_root / dataset / "forgetting_variants"
    pending = [variant_root / name / "er_graph" for name in VARIANTS]
    if any(path.exists() for path in pending):
        raise FileExistsError(f"One or more variant graph directories already exist under {variant_root}")
    for target in pending:
        copy_graph_without_train(source, target)
    sinks = {name: (variant_root / name / "er_graph" / "triples.txt").open("w", encoding="utf-8", newline="")
             for name in VARIANTS}
    out_counts = {name: Counter() for name in VARIANTS}
    try:
        with (source / "triples.txt").open(encoding="utf-8", newline="") as stream:
            for line in stream:
                parsed = parse_exfr(line)
                if parsed is None:
                    for name, sink in sinks.items():
                        sink.write(line)
                        out_counts[name]["other"] += 1
                    continue
                uid, ex, _ = parsed
                for name, sink in sinks.items():
                    if (uid, ex) in selected[name]:
                        sink.write(line)
                        out_counts[name]["exfr"] += 1
    finally:
        for sink in sinks.values():
            sink.close()

    for name in VARIANTS:
        target = variant_root / name / "er_graph"
        if out_counts[name]["other"] != source_counts["other"] or out_counts[name]["exfr"] != len(selected[name]):
            raise AssertionError(f"Bad output counts: {dataset} {name}")
        output_manifest = dict(manifest)
        output_manifest["triple_counts"] = dict(manifest["triple_counts"])
        output_manifest["triple_counts"]["train"] = dict(manifest["triple_counts"]["train"])
        output_manifest["triple_counts"]["train"]["exfr"] = out_counts[name]["exfr"]
        output_manifest["forgetting_edge_variant"] = {
            "name": name,
            "scope": "Only training triples.txt exfr edges change; test_triples.txt and all other inputs are copied unchanged",
            "eligibility": "exfr relation is not exfr0.00 (two-decimal relation seen by training model)",
            "sampling": "deterministic weighted random permutation without replacement; weight=stored exercise forgetting squared; first K per train student; 200 is subset of 400" if name != "drop_exfr0p00" else "retain all eligible exfr edges",
            "seed": seed,
            "per_student_limit": 200 if name.endswith("200") else 400 if name.endswith("400") else None,
            "train_students": users,
            "source_exfr_edges": source_counts["exfr"],
            "source_exact_zero_values": exact_zero,
            "source_exfr0p00_edges": source_counts["exfr0.00"],
            "source_positive_values_quantized_to_exfr0p00": quantized_zero_nonzero,
            "retained_exfr_edges": out_counts[name]["exfr"],
            "retained_exfr_mean_forgetting": round(sum(float(matrix[uid, ex]) for uid, ex in selected[name]) / max(1, len(selected[name])), 6),
            "retained_exfr_per_student_min": min(min(len(values), 200 if name.endswith("200") else 400 if name.endswith("400") else math.inf) for values in rank.values()),
            "retained_exfr_per_student_max": max(min(len(values), 200 if name.endswith("200") else 400 if name.endswith("400") else math.inf) for values in rank.values()),
            "source_train_triples_sha256": sha256(source / "triples.txt"),
            "output_train_triples_sha256": sha256(target / "triples.txt"),
            "test_triples_sha256": sha256(target / "test_triples.txt"),
        }
        (target / "er_graph_manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(dataset, name, "source_exfr", source_counts["exfr"], "retained_exfr", out_counts[name]["exfr"],
              "retention_pct", round(100 * out_counts[name]["exfr"] / source_counts["exfr"], 2),
              "per_student_min_max", output_manifest["forgetting_edge_variant"]["retained_exfr_per_student_min"],
              output_manifest["forgetting_edge_variant"]["retained_exfr_per_student_max"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--plan", action="store_true", help="Read-only counts; do not write variants")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        build_dataset(data_root, dataset, DATASETS.index(dataset), args.seed, plan=args.plan)


if __name__ == "__main__":
    main()
