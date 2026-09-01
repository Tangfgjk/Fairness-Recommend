"""Build an auditable active-learner/exercise subset from a canonical raw dataset.

The source ``raw`` directory is never modified.  The compact cohort is written
to ``raw_compact`` beside it, with every ID-bearing artifact reindexed to the
new learner, exercise, and knowledge-concept spaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["assist2009-sub", "XES3G5M-sub-small"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-fin-root", type=Path, default=Path("Data_Fin"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--min-student-interactions", type=int, default=20)
    parser.add_argument("--min-exercise-interactions", type=int, default=10)
    parser.add_argument("--output-name", default="raw_compact")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(DATASETS))
    if invalid:
        raise ValueError(f"This compact builder supports only {DATASETS}; got {invalid}")
    return values


def read_q(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = [token for token in re.split(r"[\t,\s]+", line.strip()) if token]
        if tokens:
            rows.append([float(token) for token in tokens])
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Invalid Q matrix: {path}")
    return (matrix > 0).astype(np.int8)


def write_q(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        for row in matrix:
            fp.write(",".join(str(int(value)) for value in row) + "\n")


def stable_core(interactions: pd.DataFrame, min_u: int, min_e: int) -> tuple[pd.DataFrame, int]:
    current = interactions.copy()
    rounds = 0
    while True:
        rounds += 1
        valid_users = set(current["uid"].value_counts().loc[lambda counts: counts >= min_u].index.astype(int))
        current = current[current["uid"].isin(valid_users)]
        valid_exercises = set(current["question"].value_counts().loc[lambda counts: counts >= min_e].index.astype(int))
        next_frame = current[current["question"].isin(valid_exercises)].copy()
        if len(next_frame) == len(current):
            return next_frame, rounds
        current = next_frame


def filtered_evaluation(source: Path, uid_map: dict[int, int], kc_map: dict[int, int]) -> list[str]:
    lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        uid_token, _, kc_tokens = raw_line.partition("\t")
        match = re.fullmatch(r"uid(\d+)", uid_token.strip())
        if not match:
            raise ValueError(f"Unexpected evaluation uid token: {uid_token!r}")
        old_uid = int(match.group(1))
        if old_uid not in uid_map:
            continue
        concepts = [int(token) for token in kc_tokens.split(",") if token.strip()]
        remapped = [str(kc_map[kc]) for kc in concepts if kc in kc_map]
        # The raw validator requires one evaluation row for every retained
        # test learner.  An empty target is valid when all of that learner's
        # original target KCs were filtered out with inactive exercises.
        lines.append(f"uid{uid_map[old_uid]}\t{','.join(remapped)}")
    return lines


def build_dataset(root: Path, dataset: str, args: argparse.Namespace) -> dict:
    source = root / dataset / "raw"
    target = root / dataset / args.output_name
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        if not args.force:
            raise FileExistsError(f"{target} exists; use --force to replace it")
        shutil.rmtree(target)

    interactions = pd.read_csv(source / "interactions_all.csv", low_memory=False)
    interactions["uid"] = interactions["uid"].astype(int)
    interactions["question"] = interactions["question"].astype(int)
    q = read_q(source / "Q.txt")
    before = {
        "students": int(interactions["uid"].nunique()),
        "exercises": int(q.shape[0]),
        "concepts": int(q.shape[1]),
        "interactions": int(len(interactions)),
    }
    retained, rounds = stable_core(interactions, args.min_student_interactions, args.min_exercise_interactions)
    if retained.empty:
        raise ValueError(f"{dataset}: thresholds removed every interaction")

    old_uids = sorted(retained["uid"].unique().tolist())
    old_questions = sorted(retained["question"].unique().tolist())
    q_subset_before_kc = q[np.asarray(old_questions, dtype=int)]
    old_concepts = np.flatnonzero(q_subset_before_kc.sum(axis=0) > 0).astype(int).tolist()
    uid_map = {old: new for new, old in enumerate(old_uids)}
    question_map = {old: new for new, old in enumerate(old_questions)}
    kc_map = {old: new for new, old in enumerate(old_concepts)}

    retained = retained.copy()
    # Preserve the legacy ``original_*`` columns carried by canonical raw.
    # The previous canonical index is useful for auditing this extra filter,
    # but is deliberately kept in a separate column.
    retained["previous_uid"] = retained["uid"].astype(int)
    retained["previous_exercise_index"] = retained["question"].astype(int)
    retained["uid"] = retained["uid"].map(uid_map).astype(int)
    retained["question"] = retained["question"].map(question_map).astype(int)
    retained["entity_id"] = retained["uid"].map(lambda value: f"uid{value}")
    retained["entity_exercise_id"] = retained["question"].map(lambda value: f"ex{value}")
    retained["concepts"] = retained["previous_exercise_index"].map(
        lambda old_question: "_".join(
            str(kc_map[int(kc)]) for kc in np.flatnonzero(q[old_question]) if int(kc) in kc_map
        )
    )
    retained = retained.sort_values(["uid", "sequence_order", "timestamp"]).reset_index(drop=True)
    retained["sequence_order"] = retained.groupby("uid").cumcount().astype(int)

    split = pd.read_csv(source / "student_split.csv")
    split = split[split["uid"].astype(int).isin(old_uids)].copy()
    split["previous_uid"] = split["uid"].astype(int)
    split["uid"] = split["uid"].map(uid_map).astype(int)
    split["entity_id"] = split["uid"].map(lambda value: f"uid{value}")
    split = split[["uid", "entity_id", "original_uid", "previous_uid", "split"]].sort_values("uid")
    if set(split["split"]) != {"train", "test"}:
        raise ValueError(f"{dataset}: filtering removed an entire split")

    q_subset = q[np.asarray(old_questions, dtype=int)][:, np.asarray(old_concepts, dtype=int)]
    exercise_map = pd.read_csv(source / "exercise_id_map.csv")
    exercise_map = exercise_map[exercise_map["exercise_index"].astype(int).isin(old_questions)].copy()
    exercise_map["previous_exercise_index"] = exercise_map["exercise_index"].astype(int)
    exercise_map["exercise_index"] = exercise_map["exercise_index"].map(question_map).astype(int)
    exercise_map["entity_id"] = exercise_map["exercise_index"].map(lambda value: f"ex{value}")
    exercise_map = exercise_map.sort_values("exercise_index")
    concept_map = pd.read_csv(source / "concept_id_map.csv")
    concept_map = concept_map[concept_map["concept_index"].astype(int).isin(old_concepts)].copy()
    concept_map["previous_concept_index"] = concept_map["concept_index"].astype(int)
    concept_map["concept_index"] = concept_map["concept_index"].map(kc_map).astype(int)
    concept_map["entity_id"] = concept_map["concept_index"].map(lambda value: f"kc{value}")
    concept_map = concept_map.sort_values("concept_index")

    exercise_meta = json.loads((source / "exercise_metadata.json").read_text(encoding="utf-8"))
    exercise_entries = {int(item["exercise_index"]): dict(item) for item in exercise_meta["exercises"]}
    compact_exercises = []
    for old, new in question_map.items():
        item = exercise_entries[old]
        item["original_exercise_index"] = old
        item["exercise_index"] = new
        item["entity_id"] = f"ex{new}"
        compact_exercises.append(item)
    concept_meta = json.loads((source / "concept_metadata.json").read_text(encoding="utf-8"))
    concept_entries = {int(item["concept_index"]): dict(item) for item in concept_meta["concepts"]}
    compact_concepts = []
    for old, new in kc_map.items():
        item = concept_entries[old]
        item["original_concept_index"] = old
        item["concept_index"] = new
        item["entity_id"] = f"kc{new}"
        compact_concepts.append(item)

    evaluation = filtered_evaluation(source / "evaluation_uid_kc_response.txt", uid_map, kc_map)
    target.mkdir(parents=True)
    retained.to_csv(target / "interactions_all.csv", index=False)
    split.to_csv(target / "student_split.csv", index=False)
    split.to_csv(target / "student_id_map.csv", index=False)
    exercise_map.to_csv(target / "exercise_id_map.csv", index=False)
    concept_map.to_csv(target / "concept_id_map.csv", index=False)
    write_q(target / "Q.txt", q_subset)
    (target / "evaluation_uid_kc_response.txt").write_text("\n".join(evaluation) + ("\n" if evaluation else ""), encoding="utf-8")
    (target / "exercise_metadata.json").write_text(json.dumps({"dataset": dataset, "exercises": compact_exercises}, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "concept_metadata.json").write_text(json.dumps({"dataset": dataset, "concepts": compact_concepts}, ensure_ascii=False, indent=2), encoding="utf-8")
    after = {
        "students": int(len(split)),
        "train_students": int((split["split"] == "train").sum()),
        "test_students": int((split["split"] == "test").sum()),
        "exercises": int(len(old_questions)),
        "concepts": int(len(old_concepts)),
        "interactions": int(len(retained)),
        "train_interactions": int((retained["split"] == "train").sum()),
        "test_interactions": int((retained["split"] == "test").sum()),
        "estimated_dense_exfr_edges": int(len(split) * len(old_questions)),
        "evaluation_rows": int(len(evaluation)),
    }
    manifest = {
        "dataset": dataset,
        "protocol": "iterative_bipartite_k_core_active_subset",
        "source_raw_dir": str(source.resolve()),
        "thresholds": {"min_student_interactions": args.min_student_interactions, "min_exercise_interactions": args.min_exercise_interactions},
        "iterations_to_stability": rounds,
        "before": before,
        "after": after,
        "reindexing": "students, exercises, and concepts are contiguous and aligned with interactions/Q/metadata/evaluation",
    }
    (target / "raw_compact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep the standard raw manifest contract so the existing front pipeline
    # can consume this directory after selecting it as its raw input.
    raw_manifest = {
        "dataset": dataset,
        "raw_protocol": "single_cohort_shared_by_front_models_and_er_graph_compact_subset",
        "split_strategy": "filtered_legacy_student_split",
        "source_raw_dir": str(source.resolve()),
        "counts": {
            key: after[key]
            for key in ["students", "train_students", "test_students", "exercises", "concepts", "interactions", "train_interactions", "test_interactions"]
        },
        "files": {
            "interactions": "interactions_all.csv",
            "student_split": "student_split.csv",
            "q_matrix": "Q.txt",
            "exercise_metadata": "exercise_metadata.json",
            "concept_metadata": "concept_metadata.json",
            "evaluation_targets": "evaluation_uid_kc_response.txt",
        },
        "compact_subset_manifest": "raw_compact_manifest.json",
    }
    (target / "raw_manifest.json").write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    if args.min_student_interactions < 1 or args.min_exercise_interactions < 1:
        raise ValueError("minimum interaction thresholds must be positive")
    reports = [build_dataset(args.data_fin_root, dataset, args) for dataset in selected_datasets(args.datasets)]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
