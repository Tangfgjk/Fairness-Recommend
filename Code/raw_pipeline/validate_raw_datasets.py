"""Validate canonical Data_Fin raw directories before feature generation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


csv.field_size_limit(2**31 - 1)
DATASETS = ["Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small"]
RAW_COMPACT_DATASETS = {"assist2009-sub", "XES3G5M-sub-small"}
REQUIRED = {
    "Q.txt",
    "interactions_all.csv",
    "student_id_map.csv",
    "student_split.csv",
    "exercise_id_map.csv",
    "concept_id_map.csv",
    "exercise_metadata.json",
    "concept_metadata.json",
    "evaluation_uid_kc_response.txt",
    "raw_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def selected(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return DATASETS
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return values


def q_shape(path: Path) -> tuple[int, int]:
    rows = 0
    width: int | None = None
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
            if not tokens:
                continue
            width = len(tokens) if width is None else width
            if len(tokens) != width:
                raise ValueError("Q matrix is not rectangular")
            rows += 1
    return rows, width or 0


def canonical_raw_name(dataset: str) -> str:
    return "raw_compact" if dataset in RAW_COMPACT_DATASETS else "raw"


def resolve_raw_name(dataset: str, raw_name: str) -> str:
    expected = canonical_raw_name(dataset)
    if raw_name in {"", "auto"}:
        return expected
    if raw_name != expected:
        raise ValueError(
            f"{dataset} must read {expected!r}; got {raw_name!r}. "
            "The formal workflow fixes assist2009-sub and XES3G5M-sub-small to raw_compact, all others to raw."
        )
    return raw_name


def q_concept_sets(path: Path) -> list[set[int]]:
    rows: list[set[int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
            if tokens:
                rows.append({index for index, value in enumerate(tokens) if float(value) > 0.0})
    return rows


def parse_interaction_concepts(value: str) -> set[int]:
    return {int(token) for token in re.split(r"[_;,\s]+", value.strip()) if token}


def validate_dataset(output_root: Path, dataset: str, raw_name: str = "auto") -> dict:
    raw_dir = output_root / dataset / resolve_raw_name(dataset, raw_name)
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(name for name in REQUIRED if not (raw_dir / name).exists())
    if missing:
        return {"dataset": dataset, "raw_dir": str(raw_dir), "status": "failed", "errors": [f"missing: {missing}"], "warnings": []}

    manifest = json.loads((raw_dir / "raw_manifest.json").read_text(encoding="utf-8"))
    q_exercises, q_concepts = q_shape(raw_dir / "Q.txt")
    q_sets = q_concept_sets(raw_dir / "Q.txt")
    students: dict[int, str] = {}
    splits: dict[int, str] = {}
    with (raw_dir / "student_split.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            uid = int(row["uid"])
            if uid in students:
                errors.append(f"duplicate canonical uid {uid}")
            students[uid] = row["original_uid"]
            splits[uid] = row["split"]
    if set(splits.values()) - {"train", "test"}:
        errors.append("unexpected split label")

    interaction_counts = Counter()
    interaction_split_counts = Counter()
    bad_questions = 0
    concept_q_mismatches = 0
    with (raw_dir / "interactions_all.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            uid = int(row["uid"])
            question = int(row["question"])
            if uid not in students:
                errors.append(f"interaction references unknown uid {uid}")
            if not 0 <= question < q_exercises:
                bad_questions += 1
            elif row["concepts"].strip():
                try:
                    observed_concepts = parse_interaction_concepts(row["concepts"])
                except ValueError:
                    errors.append(f"invalid interaction concept encoding for uid {uid}, question {question}")
                else:
                    # Algebra logs the currently practiced KC while Q.txt stores
                    # every KC attached to the item. Therefore subset, rather
                    # than equality, is the correct alignment invariant.
                    if not observed_concepts.issubset(q_sets[question]):
                        concept_q_mismatches += 1
            if row["split"] != splits.get(uid):
                errors.append(f"interaction split mismatch for uid {uid}")
            interaction_counts[uid] += 1
            interaction_split_counts[row["split"]] += 1
    if bad_questions:
        errors.append(f"{bad_questions} interactions use a question outside Q.txt")
    if concept_q_mismatches:
        errors.append(f"{concept_q_mismatches} interactions have KC values outside their Q.txt row")
    without_events = sorted(uid for uid in students if interaction_counts[uid] == 0)
    if without_events:
        errors.append(f"students without interactions: {without_events[:20]}")

    exercise_metadata = json.loads((raw_dir / "exercise_metadata.json").read_text(encoding="utf-8"))
    concept_metadata = json.loads((raw_dir / "concept_metadata.json").read_text(encoding="utf-8"))
    if len(exercise_metadata.get("exercises", [])) != q_exercises:
        errors.append("exercise metadata count does not match Q rows")
    if len(concept_metadata.get("concepts", [])) != q_concepts:
        errors.append("concept metadata count does not match Q columns")

    evaluation_uids: set[int] = set()
    with (raw_dir / "evaluation_uid_kc_response.txt").open("r", encoding="utf-8") as fp:
        for line in fp:
            # Do not strip a trailing tab: it may represent a valid empty KC
            # target for a test learner.
            line = line.rstrip("\r\n")
            if not line:
                continue
            token = line.split("\t", 1)[0].strip()
            match = re.fullmatch(r"uid(\d+)", token)
            if not match:
                errors.append(f"invalid evaluation uid token: {token}")
                continue
            evaluation_uids.add(int(match.group(1)))
    test_uids = {uid for uid, split in splits.items() if split == "test"}
    if evaluation_uids != test_uids:
        errors.append("evaluation uid set does not exactly match test student split")

    counts = {
        "students": len(students),
        "train_students": sum(1 for split in splits.values() if split == "train"),
        "test_students": sum(1 for split in splits.values() if split == "test"),
        "exercises": q_exercises,
        "concepts": q_concepts,
        "interactions": sum(interaction_counts.values()),
        "train_interactions": interaction_split_counts["train"],
        "test_interactions": interaction_split_counts["test"],
    }
    for key, value in counts.items():
        if manifest.get("counts", {}).get(key) != value:
            errors.append(f"manifest count mismatch for {key}: {manifest.get('counts', {}).get(key)} != {value}")
    if counts["test_students"] == 0 or counts["train_students"] == 0:
        errors.append("both train and test student groups are required")
    if dataset == "Eedi" and (counts["students"], counts["train_students"], counts["test_students"]) != (935, 701, 234):
        warnings.append("Eedi cohort differs from the expected 935/701/234 protocol")
    return {
        "dataset": dataset,
        "raw_dir": str(raw_dir),
        "status": "passed" if not errors else "failed",
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    reports = [validate_dataset(args.output_root, dataset, args.raw_name) for dataset in selected(args.datasets)]
    payload = {"output_root": str(args.output_root), "reports": reports, "status": "passed" if all(item["status"] == "passed" for item in reports) else "failed"}
    report_path = args.report or args.output_root / "raw_validation_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
