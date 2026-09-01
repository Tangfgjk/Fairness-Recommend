"""Build canonical raw inputs for the unified KG4ER experiment protocol.

This script never modifies the legacy source data. It creates a compact,
auditable data layer containing the selected learner cohort, a fixed learner
split, interactions, Q-matrix, static text metadata, and remapped evaluation
targets. Front models and ER graph builders should consume this layer only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


csv.field_size_limit(2**31 - 1)

DATASETS = ["Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small"]
EXERCISE_FIELDS = {
    "raw_exercise_id",
    "text_source",
    "raw_question_id",
    "raw_question_index",
    "raw_problem_id",
    "raw_step_id",
    "question_text",
    "problem_name",
    "problem_hierarchy",
    "step_name",
    "step_names",
    "skills",
    "skill_names",
    "answer_types",
    "problem_types",
    "assistment_ids",
    "template_ids",
    "question_type",
    "kc_routes",
    "answer",
}
CONCEPT_FIELDS = {"raw_concept_id", "name", "definition", "definition_source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Immutable legacy KG4ER/data directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Target Data_Fin directory.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--eedi-seed", type=int, default=2024)
    parser.add_argument("--eedi-train-ratio", type=float, default=0.75)
    parser.add_argument("--force", action="store_true", help="Replace an existing raw directory.")
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return DATASETS
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    return datasets


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def parse_int(value: Any, field_name: str) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def q_shape(path: Path) -> tuple[int, int]:
    rows = 0
    width: int | None = None
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
            if not tokens:
                continue
            if width is None:
                width = len(tokens)
            elif len(tokens) != width:
                raise ValueError(f"Non-rectangular Q matrix at {path}: expected {width}, got {len(tokens)}")
            rows += 1
    if not rows or width is None:
        raise ValueError(f"Empty Q matrix: {path}")
    return rows, width


def find_first(paths: list[Path], required: bool = True) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    if required:
        joined = "\n".join(str(path) for path in paths)
        raise FileNotFoundError(f"No candidate file exists:\n{joined}")
    return None


def source_layout(source_root: Path, dataset: str) -> dict[str, Path | None]:
    base = source_root / dataset
    prepared = base / "prepared_for_kt"
    sequence_dir = prepared if (prepared / "sequence_interactions.csv").exists() else base
    eval_name = f"{dataset}_uid_kc_response.txt"
    semantic_dirs = [
        sequence_dir / "semantic_kg_features" / "entity_features",
        base / "semantic_kg_features" / "entity_features",
        base / "er_v8" / "semantic_kg_features" / "entity_features",
    ]
    return {
        "base": base,
        "sequence_dir": sequence_dir,
        "interactions": sequence_dir / "sequence_interactions.csv",
        "q": sequence_dir / "Q.txt",
        "train_sequences": sequence_dir / "train_sequences.csv",
        "test_sequences": sequence_dir / "test_sequences.csv",
        "exercise_metadata": find_first([directory / "exercise_semantics.json" for directory in semantic_dirs]),
        "concept_metadata": find_first([directory / "concept_semantics.json" for directory in semantic_dirs]),
        "evaluation": find_first([sequence_dir / eval_name, base / eval_name, base / "er_v8" / eval_name]),
    }


def sequence_uid_order(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            uid = str(row.get("uid", "")).strip()
            if not uid:
                raise ValueError(f"Blank uid in {path}")
            values.append(uid)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate learner rows in {path}")
    return values


def eedi_test_uid_order(interactions_path: Path) -> list[str]:
    test_uids: set[str] = set()
    with interactions_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if str(row.get("source_split", "")).strip().lower() == "test":
                uid = str(row.get("uid", "")).strip()
                if uid:
                    test_uids.add(uid)
    if not test_uids:
        raise ValueError(f"No Eedi source_split=test learners in {interactions_path}")
    return sorted(test_uids, key=natural_key)


def build_split(dataset: str, layout: dict[str, Path | None], eedi_seed: int, eedi_train_ratio: float) -> tuple[dict[str, str], list[str], str]:
    if dataset == "Eedi":
        test_order = eedi_test_uid_order(layout["interactions"])
        shuffled = list(test_order)
        random.Random(eedi_seed).shuffle(shuffled)
        train_count = int(len(shuffled) * eedi_train_ratio)
        split_by_raw = {uid: "train" for uid in shuffled[:train_count]}
        split_by_raw.update({uid: "test" for uid in shuffled[train_count:]})
        return split_by_raw, test_order, f"eedi_sub_random_{eedi_train_ratio:g}_seed_{eedi_seed}"

    train_order = sequence_uid_order(layout["train_sequences"])
    test_order = sequence_uid_order(layout["test_sequences"])
    overlap = sorted(set(train_order) & set(test_order), key=natural_key)
    if overlap:
        raise ValueError(f"{dataset} has train/test learner overlap: {overlap[:10]}")
    split_by_raw = {uid: "train" for uid in train_order}
    split_by_raw.update({uid: "test" for uid in test_order})
    return split_by_raw, test_order, "legacy_student_split"


def metadata_records(path: Path, key: str) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    records = payload.get(key, {})
    if not isinstance(records, dict):
        raise ValueError(f"Expected object '{key}' in {path}")
    return {str(record_id): dict(record) for record_id, record in records.items() if isinstance(record, dict)}


def clean_record(record: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
    return {field: record[field] for field in allowed_fields if field in record}


def build_static_metadata(
    raw_dir: Path,
    dataset: str,
    exercise_count: int,
    concept_count: int,
    exercise_path: Path,
    concept_path: Path,
    original_question_by_index: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exercise_source = metadata_records(exercise_path, "exercises")
    concept_source = metadata_records(concept_path, "concepts")

    exercises: list[dict[str, Any]] = []
    for index in range(exercise_count):
        source = exercise_source.get(f"ex{index}", {})
        record = {
            "exercise_index": index,
            "entity_id": f"ex{index}",
            "original_question_id": original_question_by_index.get(index, str(source.get("raw_exercise_id", ""))),
            **clean_record(source, EXERCISE_FIELDS),
        }
        exercises.append(record)

    concepts: list[dict[str, Any]] = []
    for index in range(concept_count):
        source = concept_source.get(f"kc{index}", {})
        record = {
            "concept_index": index,
            "entity_id": f"kc{index}",
            "raw_concept_id": str(source.get("raw_concept_id", index)),
            "name": str(source.get("name", f"kc{index}")),
            "definition": str(source.get("definition", "")),
            "definition_source": str(source.get("definition_source", "")),
        }
        concepts.append(record)

    write_json(raw_dir / "exercise_metadata.json", {"dataset": dataset, "exercises": exercises})
    write_json(raw_dir / "concept_metadata.json", {"dataset": dataset, "concepts": concepts})
    return exercises, concepts


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_interactions(
    interactions_path: Path,
    output_path: Path,
    canonical_by_raw: dict[str, int],
    split_by_raw: dict[str, str],
    exercise_count: int,
) -> tuple[dict[int, str], dict[str, int], int]:
    fieldnames = [
        "uid",
        "entity_id",
        "original_uid",
        "question",
        "entity_exercise_id",
        "original_question",
        "concepts",
        "response",
        "timestamp",
        "sequence_order",
        "split",
    ]
    original_question_by_index: dict[int, str] = {}
    sequence_orders: Counter[str] = Counter()
    seen_uids: set[str] = set()
    interaction_count = 0
    with interactions_path.open("r", encoding="utf-8", newline="") as source, output_path.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(reader):
            raw_uid = str(row.get("uid", "")).strip()
            if raw_uid not in canonical_by_raw:
                continue
            question = parse_int(row.get("question"), "question")
            if not 0 <= question < exercise_count:
                raise ValueError(f"Question {question} outside Q matrix at source row {row_index} in {interactions_path}")
            response = parse_int(row.get("response"), "response")
            original_question = str(row.get("original_question", "")).strip()
            if original_question:
                known = original_question_by_index.setdefault(question, original_question)
                if known != original_question:
                    raise ValueError(f"Question {question} maps to multiple original IDs: {known!r}, {original_question!r}")
            canonical_uid = canonical_by_raw[raw_uid]
            writer.writerow(
                {
                    "uid": canonical_uid,
                    "entity_id": f"uid{canonical_uid}",
                    "original_uid": str(row.get("original_uid", raw_uid)).strip() or raw_uid,
                    "question": question,
                    "entity_exercise_id": f"ex{question}",
                    "original_question": original_question,
                    "concepts": str(row.get("concepts", "")).strip(),
                    "response": 1 if response > 0 else 0,
                    "timestamp": str(row.get("timestamp", "")).strip(),
                    "sequence_order": sequence_orders[raw_uid],
                    "split": split_by_raw[raw_uid],
                }
            )
            sequence_orders[raw_uid] += 1
            seen_uids.add(raw_uid)
            interaction_count += 1
    missing = sorted(set(canonical_by_raw) - seen_uids, key=natural_key)
    if missing:
        raise ValueError(f"Selected learners have no interactions: {missing[:20]}")
    return original_question_by_index, {raw: sequence_orders[raw] for raw in canonical_by_raw}, interaction_count


def rewrite_evaluation(
    source_path: Path,
    output_path: Path,
    test_uid_order: list[str],
    canonical_by_raw: dict[str, int],
    split_by_raw: dict[str, str],
) -> int:
    converted = 0
    seen_indices: set[int] = set()
    with source_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="") as target:
        for line in source:
            # Preserve a trailing tab: `uid172\t` is a valid record with an
            # empty KC target and must remain aligned to its learner.
            value = line.rstrip("\r\n")
            if not value:
                continue
            pieces = value.split("\t", 1)
            if len(pieces) != 2:
                raise ValueError(f"Unexpected evaluation row: {value[:120]!r}")
            match = re.fullmatch(r"uid(\d+)", pieces[0].strip())
            if not match:
                raise ValueError(f"Unexpected evaluation uid: {pieces[0]!r}")
            local_index = int(match.group(1))
            if not 0 <= local_index < len(test_uid_order):
                raise ValueError(f"Evaluation uid index {local_index} exceeds test learner count {len(test_uid_order)}")
            raw_uid = test_uid_order[local_index]
            # Eedi retains the legacy 935-row evaluation order as the source
            # index, but only the new 25% student split is written as ER test
            # targets. The other datasets already have a source test cohort.
            if split_by_raw[raw_uid] == "test":
                target.write(f"uid{canonical_by_raw[raw_uid]}\t{pieces[1].strip()}\n")
                converted += 1
            seen_indices.add(local_index)
    if seen_indices != set(range(len(test_uid_order))):
        raise ValueError("Evaluation file does not contain exactly one row for every test learner.")
    return converted


def build_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    layout = source_layout(args.source_root, dataset)
    for key in ("interactions", "q", "exercise_metadata", "concept_metadata", "evaluation"):
        if not isinstance(layout[key], Path) or not layout[key].exists():
            raise FileNotFoundError(f"Missing required {key} source for {dataset}: {layout[key]}")

    raw_dir = args.output_root / dataset / "raw"
    if raw_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {raw_dir}. Use --force to replace it.")
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    exercise_count, concept_count = q_shape(layout["q"])
    split_by_raw, test_uid_order, split_strategy = build_split(
        dataset, layout, args.eedi_seed, args.eedi_train_ratio
    )
    raw_uids = sorted(split_by_raw, key=natural_key)
    canonical_by_raw = {raw_uid: index for index, raw_uid in enumerate(raw_uids)}

    student_rows = [
        {
            "uid": canonical_by_raw[raw_uid],
            "entity_id": f"uid{canonical_by_raw[raw_uid]}",
            "original_uid": raw_uid,
            "split": split_by_raw[raw_uid],
        }
        for raw_uid in raw_uids
    ]
    write_csv(raw_dir / "student_id_map.csv", ["uid", "entity_id", "original_uid", "split"], student_rows)
    write_csv(raw_dir / "student_split.csv", ["uid", "entity_id", "original_uid", "split"], student_rows)

    original_question_by_index, learner_events, interaction_count = build_interactions(
        layout["interactions"],
        raw_dir / "interactions_all.csv",
        canonical_by_raw,
        split_by_raw,
        exercise_count,
    )
    shutil.copy2(layout["q"], raw_dir / "Q.txt")
    exercises, concepts = build_static_metadata(
        raw_dir,
        dataset,
        exercise_count,
        concept_count,
        layout["exercise_metadata"],
        layout["concept_metadata"],
        original_question_by_index,
    )
    write_csv(
        raw_dir / "exercise_id_map.csv",
        ["exercise_index", "entity_id", "original_question_id"],
        [
            {
                "exercise_index": item["exercise_index"],
                "entity_id": item["entity_id"],
                "original_question_id": item["original_question_id"],
            }
            for item in exercises
        ],
    )
    write_csv(
        raw_dir / "concept_id_map.csv",
        ["concept_index", "entity_id", "raw_concept_id", "name", "definition_source"],
        [
            {
                "concept_index": item["concept_index"],
                "entity_id": item["entity_id"],
                "raw_concept_id": item["raw_concept_id"],
                "name": item["name"],
                "definition_source": item["definition_source"],
            }
            for item in concepts
        ],
    )
    evaluation_rows = rewrite_evaluation(
        layout["evaluation"],
        raw_dir / "evaluation_uid_kc_response.txt",
        test_uid_order,
        canonical_by_raw,
        split_by_raw,
    )

    split_counts = Counter(split_by_raw.values())
    interaction_split_counts = Counter()
    for raw_uid, count in learner_events.items():
        interaction_split_counts[split_by_raw[raw_uid]] += count
    manifest = {
        "dataset": dataset,
        "raw_protocol": "single_cohort_shared_by_front_models_and_er_graph",
        "split_strategy": split_strategy,
        "source_files": {key: str(value) for key, value in layout.items() if isinstance(value, Path)},
        "counts": {
            "students": len(raw_uids),
            "train_students": int(split_counts["train"]),
            "test_students": int(split_counts["test"]),
            "exercises": exercise_count,
            "concepts": concept_count,
            "interactions": interaction_count,
            "train_interactions": int(interaction_split_counts["train"]),
            "test_interactions": int(interaction_split_counts["test"]),
            "evaluation_rows": evaluation_rows,
        },
        "files": {
            "interactions": "interactions_all.csv",
            "student_split": "student_split.csv",
            "q_matrix": "Q.txt",
            "exercise_metadata": "exercise_metadata.json",
            "concept_metadata": "concept_metadata.json",
            "evaluation_targets": "evaluation_uid_kc_response.txt",
        },
        "notes": [
            "No graph triples, learned embeddings, IRT parameters, or cognitive-state files are stored in raw.",
            "Question and concept indices are aligned with Q.txt rows and columns.",
            "Evaluation uid rows are remapped from local test-row order to canonical uid IDs.",
        ],
    }
    write_json(raw_dir / "raw_manifest.json", manifest)
    return {"dataset": dataset, "raw_dir": str(raw_dir), **manifest["counts"]}


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {args.source_root}")
    results = [build_dataset(args, dataset) for dataset in selected_datasets(args.datasets)]
    print(json.dumps({"output_root": str(args.output_root), "datasets": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
