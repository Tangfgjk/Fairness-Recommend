"""Rebuild XES Q-matrix and concept metadata from exercise ``kc_routes``.

The XES source carries two knowledge labels for each exercise:

* ``Q.txt`` / ``concept_metadata.json`` inherited from the legacy pipeline.
* ``exercise_metadata.json[kc_routes]`` used by the exercise text metadata.

For XES these two sources can disagree.  This repair script makes the canonical
raw layer internally consistent by deriving Q, concept metadata, interaction
concept fields, and evaluation KC histories from ``kc_routes``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET = "XES3G5M-sub-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-fin-root", type=Path, default=Path("Data_Fin"))
    parser.add_argument("--raw-name", default="raw")
    parser.add_argument(
        "--concept-key",
        choices=["leaf", "route"],
        default="leaf",
        help="Use the route leaf name or the full route as the repaired concept identity.",
    )
    parser.add_argument("--backup", action="store_true", help="Keep one .before_route_q_repair backup copy of changed files.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing repaired raw directory.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_q(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        for row in matrix:
            fp.write(",".join(str(int(value)) for value in row) + "\n")


def route_leaf(route: str) -> str:
    parts = [part.strip() for part in str(route).split("----") if part.strip()]
    return parts[-1] if parts else str(route).strip()


def concept_identity(route: str, mode: str) -> str:
    return route_leaf(route) if mode == "leaf" else str(route).strip()


def load_old_definitions(concept_metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    definitions: dict[str, dict[str, str]] = {}
    for item in concept_metadata.get("concepts", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        definitions.setdefault(
            name,
            {
                "definition": str(item.get("definition", "")),
                "definition_source": str(item.get("definition_source", "")),
            },
        )
    return definitions


def ensure_backup(path: Path) -> None:
    backup = path.with_name(path.name + ".before_route_q_repair")
    if not backup.exists():
        shutil.copy2(path, backup)


def build_route_q(raw_dir: Path, concept_key: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[int, list[int]]]:
    exercise_metadata = read_json(raw_dir / "exercise_metadata.json")
    old_concepts = read_json(raw_dir / "concept_metadata.json")
    old_definitions = load_old_definitions(old_concepts)

    exercises = exercise_metadata.get("exercises", [])
    if not exercises:
        raise ValueError(f"No exercises in {raw_dir / 'exercise_metadata.json'}")

    concepts: OrderedDict[str, dict[str, Any]] = OrderedDict()
    exercise_to_kcs: dict[int, list[int]] = {}
    for exercise in exercises:
        ex_idx = int(exercise["exercise_index"])
        routes = [str(route).strip() for route in exercise.get("kc_routes", []) if str(route).strip()]
        if not routes:
            raise ValueError(f"Exercise ex{ex_idx} has no kc_routes; cannot rebuild route-based Q.")
        kc_indices: list[int] = []
        for route in routes:
            key = concept_identity(route, concept_key)
            if key not in concepts:
                leaf = route_leaf(route)
                old_definition = old_definitions.get(leaf, {})
                concept_index = len(concepts)
                concepts[key] = {
                    "concept_index": concept_index,
                    "entity_id": f"kc{concept_index}",
                    "raw_concept_id": str(concept_index),
                    "name": leaf if concept_key == "leaf" else key,
                    "route": route,
                    "definition": old_definition.get("definition", ""),
                    "definition_source": old_definition.get("definition_source", "route_repair:missing_definition"),
                }
            kc_indices.append(int(concepts[key]["concept_index"]))
        exercise_to_kcs[ex_idx] = sorted(set(kc_indices))

    q = np.zeros((len(exercises), len(concepts)), dtype=np.int8)
    for ex_idx, kc_indices in exercise_to_kcs.items():
        for kc_idx in kc_indices:
            q[ex_idx, kc_idx] = 1
    return q, list(concepts.values()), exercise_to_kcs


def rewrite_interactions(raw_dir: Path, exercise_to_kcs: dict[int, list[int]]) -> None:
    path = raw_dir / "interactions_all.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame["question"] = frame["question"].astype(int)
    missing = sorted(set(frame["question"].unique()) - set(exercise_to_kcs))
    if missing:
        raise ValueError(f"Interactions reference exercises without route concepts: {missing[:10]}")
    frame["concepts"] = frame["question"].map(lambda ex: "_".join(str(kc) for kc in exercise_to_kcs[int(ex)]))
    frame.to_csv(path, index=False)


def rewrite_concept_maps(raw_dir: Path, concepts: list[dict[str, Any]]) -> None:
    with (raw_dir / "concept_id_map.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["concept_index", "entity_id", "raw_concept_id", "name", "definition_source", "route"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in concepts:
            writer.writerow({field: item.get(field, "") for field in fieldnames})
    write_json(raw_dir / "concept_metadata.json", {"dataset": DATASET, "concepts": concepts})


def rewrite_evaluation(raw_dir: Path, exercise_to_kcs: dict[int, list[int]]) -> int:
    interactions = pd.read_csv(raw_dir / "interactions_all.csv", low_memory=False)
    interactions["uid"] = interactions["uid"].astype(int)
    interactions["question"] = interactions["question"].astype(int)
    if "sequence_order" in interactions.columns:
        interactions = interactions.sort_values(["uid", "sequence_order", "timestamp"])
    else:
        interactions = interactions.sort_values(["uid", "timestamp"])
    test_frame = interactions[interactions["split"].astype(str).str.lower() == "test"]
    rows: list[str] = []
    for uid, group in test_frame.groupby("uid", sort=True):
        history: list[str] = []
        for question in group["question"].tolist():
            history.extend(str(kc) for kc in exercise_to_kcs[int(question)])
        rows.append(f"uid{int(uid)}\t{','.join(history)}")
    (raw_dir / "evaluation_uid_kc_response.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def update_manifest(raw_dir: Path, q: np.ndarray, evaluation_rows: int, concept_key: str) -> None:
    manifest_path = raw_dir / ("raw_compact_manifest.json" if (raw_dir / "raw_compact_manifest.json").exists() else "raw_manifest.json")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {"dataset": DATASET}
    counts = manifest.setdefault("counts", manifest.setdefault("after", {}))
    counts["exercises"] = int(q.shape[0])
    counts["concepts"] = int(q.shape[1])
    counts["evaluation_rows"] = int(evaluation_rows)
    manifest["q_repair"] = {
        "method": "rebuilt_from_exercise_metadata_kc_routes",
        "concept_key": concept_key,
        "q_shape": list(q.shape),
        "updated_files": [
            "Q.txt",
            "concept_metadata.json",
            "concept_id_map.csv",
            "interactions_all.csv",
            "evaluation_uid_kc_response.txt",
        ],
    }
    write_json(manifest_path, manifest)
    raw_manifest_path = raw_dir / "raw_manifest.json"
    if raw_manifest_path.exists() and raw_manifest_path != manifest_path:
        raw_manifest = read_json(raw_manifest_path)
        raw_counts = raw_manifest.setdefault("counts", {})
        raw_counts["exercises"] = int(q.shape[0])
        raw_counts["concepts"] = int(q.shape[1])
        raw_counts["evaluation_rows"] = int(evaluation_rows)
        raw_manifest["q_repair"] = manifest["q_repair"]
        write_json(raw_manifest_path, raw_manifest)


def verify(raw_dir: Path, q: np.ndarray, concepts: list[dict[str, Any]], exercise_to_kcs: dict[int, list[int]]) -> dict[str, Any]:
    exercise_metadata = read_json(raw_dir / "exercise_metadata.json")
    comparable = 0
    matches = 0
    examples: list[dict[str, Any]] = []
    for exercise in exercise_metadata.get("exercises", []):
        ex_idx = int(exercise["exercise_index"])
        route_leaves = {route_leaf(route) for route in exercise.get("kc_routes", [])}
        q_names = {str(concepts[kc]["name"]) for kc in np.flatnonzero(q[ex_idx])}
        comparable += 1
        if route_leaves == q_names:
            matches += 1
        elif len(examples) < 5:
            examples.append(
                {
                    "exercise": f"ex{ex_idx}",
                    "route_leaves": sorted(route_leaves),
                    "q_names": sorted(q_names),
                    "text": str(exercise.get("question_text", ""))[:120],
                }
            )
    return {
        "raw_dir": str(raw_dir),
        "q_shape": list(q.shape),
        "exercises_with_kc_min": int(q.sum(axis=1).min()),
        "exercises_with_kc_max": int(q.sum(axis=1).max()),
        "metadata_q_exact_leaf_matches": f"{matches}/{comparable}",
        "metadata_q_exact_leaf_match_rate": round(matches / max(1, comparable), 6),
        "mismatch_examples": examples,
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.data_fin_root / DATASET / args.raw_name
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    report_path = raw_dir / "q_route_repair_report.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"{report_path} exists. Use --force to repair again.")
    changed = [
        raw_dir / "Q.txt",
        raw_dir / "concept_metadata.json",
        raw_dir / "concept_id_map.csv",
        raw_dir / "interactions_all.csv",
        raw_dir / "evaluation_uid_kc_response.txt",
    ]
    if args.backup:
        for path in changed:
            ensure_backup(path)
    q, concepts, exercise_to_kcs = build_route_q(raw_dir, args.concept_key)
    write_q(raw_dir / "Q.txt", q)
    rewrite_concept_maps(raw_dir, concepts)
    rewrite_interactions(raw_dir, exercise_to_kcs)
    evaluation_rows = rewrite_evaluation(raw_dir, exercise_to_kcs)
    update_manifest(raw_dir, q, evaluation_rows, args.concept_key)
    report = verify(raw_dir, q, concepts, exercise_to_kcs)
    report["evaluation_rows"] = evaluation_rows
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
