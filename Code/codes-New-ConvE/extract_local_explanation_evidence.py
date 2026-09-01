"""Build local recommendation evidence tables for 2CKG4ER.

The script does not retrain or reload the model. It reads already exported
uid-exercise scores and joins them with V11 front files to produce traceable
evidence for each top-ranked recommendation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_q_matrix(path: Path) -> list[list[int]]:
    matrix: list[list[int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                matrix.append([int(float(value)) for value in line.replace(",", " ").split()])
    return matrix


def load_uid_ex_scores(path: Path) -> list[tuple[str, list[float]]]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as fp:
            return pickle.load(fp)
    rows = read_json(path)
    if rows and isinstance(rows[0], dict):
        return [(row["uid"], row["scores"]) for row in rows]
    return rows


def entity_index(entity_id: str, prefix: str) -> int | None:
    if not entity_id.startswith(prefix):
        return None
    suffix = entity_id[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def concepts_for_exercise(q_matrix: list[list[int]], exercise_idx: int) -> list[int]:
    if exercise_idx < 0 or exercise_idx >= len(q_matrix):
        return []
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def matrix_value(matrix: list[list[Any]], row: int, col: int, default: float | None = None) -> float | None:
    try:
        return float(matrix[row][col])
    except Exception:
        return default


def mean_existing(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_escape(value: Any, limit: int | None = None) -> str:
    text = clean_text(value)
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 3)] + "..."
    return text.replace("|", "\\|")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("# Local Recommendation Evidence\n\n")
        fp.write(f"Rows: {len(rows)}\n\n")
        fp.write("| " + " | ".join(fieldnames) + " |\n")
        fp.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            cells = []
            for field in fieldnames:
                limit = 120 if field == "Exercise Text" else None
                value = row.get(field)
                cells.append(markdown_escape(fmt(value) if isinstance(value, float) else value, limit=limit))
            fp.write("| " + " | ".join(cells) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract local evidence for 2CKG4ER recommendations.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--students",
        default="",
        help="Optional comma-separated uid list. Empty means all students in the scores file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    feature_dir = args.data_dir / "semantic_kg_features"
    entity_feature_dir = feature_dir / "entity_features"

    q_matrix = load_q_matrix(args.data_dir / "Q.txt")
    mastery = read_json(args.data_dir / "stu2know_mastery.json")
    forgetting = read_json(args.data_dir / "stu2ex_forget.json")
    concept_semantics = read_json(entity_feature_dir / "concept_semantics.json").get("concepts", {})
    exercise_semantics = read_json(entity_feature_dir / "exercise_semantics.json").get("exercises", {})
    learner_pedagogy = read_json(entity_feature_dir / "learner_pedagogy.json").get("learners", {})
    exercise_irt = read_json(feature_dir / "irt_features" / "exercise_irt_features.json").get("exercises", {})
    uid_ex_scores = load_uid_ex_scores(args.scores_file)

    selected_students = {item.strip() for item in args.students.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    for uid, scores in uid_ex_scores:
        if selected_students and uid not in selected_students:
            continue
        uid_idx = entity_index(uid, "uid")
        if uid_idx is None:
            continue
        learner = learner_pedagogy.get(uid, {})
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[: args.top_k]
        for rank, (ex_idx, model_score) in enumerate(ranked, start=1):
            exercise_id = f"ex{ex_idx}"
            kc_indices = concepts_for_exercise(q_matrix, ex_idx)
            kc_ids = [f"kc{kc_idx}" for kc_idx in kc_indices]
            concept_names = [
                clean_text(concept_semantics.get(kc_id, {}).get("name", kc_id))
                for kc_id in kc_ids
            ]
            exercise = exercise_irt.get(exercise_id, {})
            ex_sem = exercise_semantics.get(exercise_id, {})
            mastery_value = mean_existing(matrix_value(mastery, uid_idx, kc_idx) for kc_idx in kc_indices)
            forgetting_value = matrix_value(forgetting, uid_idx, ex_idx)
            rows.append(
                {
                    "Student": uid,
                    "Rank": rank,
                    "Exercise": exercise_id,
                    "Exercise Text": clean_text(
                        ex_sem.get("question_text")
                        or ex_sem.get("text_for_embedding")
                        or ex_sem.get("name")
                        or exercise_id
                    ),
                    "KC": ";".join(kc_ids),
                    "Concept Name": ";".join(concept_names),
                    "Mastery": mastery_value,
                    "Forgetting": forgetting_value,
                    "Theta": learner.get("theta_norm", learner.get("theta_mirt_mean")),
                    "Difficulty": exercise.get(
                        "difficulty_mirt_norm",
                        exercise.get("difficulty_norm", exercise.get("difficulty_mirt", exercise.get("difficulty"))),
                    ),
                    "Discrimination": exercise.get(
                        "discrimination_mirt_norm",
                        exercise.get(
                            "discrimination_norm",
                            exercise.get("discrimination_mirt", exercise.get("discrimination")),
                        ),
                    ),
                    "2CKG4ER Score": float(model_score),
                }
            )

    fieldnames = [
        "Student",
        "Rank",
        "Exercise",
        "Exercise Text",
        "KC",
        "Concept Name",
        "Mastery",
        "Forgetting",
        "Theta",
        "Difficulty",
        "Discrimination",
        "2CKG4ER Score",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"local_evidence_top{args.top_k}.json"
    csv_path = args.output_dir / f"local_evidence_top{args.top_k}.csv"
    md_path = args.output_dir / f"local_evidence_top{args.top_k}.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows, fieldnames)
    write_markdown(md_path, rows, fieldnames)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
