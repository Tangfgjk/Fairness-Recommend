"""KG4Ex novelty metrics on the selected final test splits.

Formal NOV uses the union of concepts across each Top-N recommendation list.
The per-exercise mean remains available separately as KG4Ex-NOV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

from final_dataset import select_dataset


def read_correct_history(dataset: str, q_matrix: Sequence[Sequence[int]]) -> tuple[dict[str, frozenset[int]], Path]:
    """Read knowledge points from correct responses of held-out test students."""
    raw_file = select_dataset(dataset).raw / "interactions_all.csv"
    questions = [frozenset(i for i, value in enumerate(row) if int(value)) for row in q_matrix]
    history: dict[str, set[int]] = {}
    with raw_file.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"].lower() != "test":
                continue
            uid = row["entity_id"]
            correct = history.setdefault(uid, set())
            if int(float(row["response"])) == 1:
                question = int(row["question"])
                if question < 0 or question >= len(questions):
                    raise ValueError(f"Question index {question} outside Q.txt: {raw_file}")
                correct.update(questions[question])
    if not history:
        raise ValueError(f"No test students in {raw_file}")
    return {uid: frozenset(concepts) for uid, concepts in history.items()}, raw_file


def calculate_kg4ex_nov(
    uid_ex_scores: Sequence[tuple[str, Sequence[float]]],
    q_matrix: Sequence[Sequence[int]],
    correct_history: dict[str, frozenset[int]],
    top_ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Mean_u mean_e in Top-N [1 - |K(e)∩C(u)| / |K(e)∪C(u)|]."""
    requested = sorted(set(top_ks))
    if not requested or requested[0] < 1:
        raise ValueError("top_ks must contain positive integers")
    questions = [frozenset(i for i, value in enumerate(row) if int(value)) for row in q_matrix]
    if any(not concepts for concepts in questions):
        raise ValueError("Every exercise must have at least one knowledge point")
    if requested[-1] > len(questions):
        raise ValueError("Top-N exceeds the number of exercises")
    per_student = {n: [] for n in requested}
    seen: set[str] = set()
    for uid, scores in uid_ex_scores:
        if uid in seen or uid not in correct_history or len(scores) != len(questions):
            raise ValueError(f"Invalid student or score count: {uid}")
        seen.add(uid)
        correct = correct_history[uid]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:requested[-1]]
        total = 0.0
        for rank, exercise in enumerate(top, 1):
            concepts = questions[exercise]
            total += 1.0 - len(concepts & correct) / len(concepts | correct)
            if rank in per_student:
                per_student[rank].append(total / rank)
    if seen != set(correct_history):
        raise ValueError(f"Recommendation students differ from test split: {len(seen)} != {len(correct_history)}")
    return _summarize(per_student)


def calculate_kg4ex_nov_set(
    uid_ex_scores: Sequence[tuple[str, Sequence[float]]],
    q_matrix: Sequence[Sequence[int]],
    correct_history: dict[str, frozenset[int]],
    top_ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Mean_u [1 - |K(Top-N,u)∩C(u)| / |K(Top-N,u)∪C(u)|]."""
    requested = sorted(set(top_ks))
    if not requested or requested[0] < 1:
        raise ValueError("top_ks must contain positive integers")
    questions = [frozenset(i for i, value in enumerate(row) if int(value)) for row in q_matrix]
    if any(not concepts for concepts in questions):
        raise ValueError("Every exercise must have at least one knowledge point")
    if requested[-1] > len(questions):
        raise ValueError("Top-N exceeds the number of exercises")
    per_student = {n: [] for n in requested}
    seen: set[str] = set()
    for uid, scores in uid_ex_scores:
        if uid in seen or uid not in correct_history or len(scores) != len(questions):
            raise ValueError(f"Invalid student or score count: {uid}")
        seen.add(uid)
        correct = correct_history[uid]
        recommended: set[int] = set()
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:requested[-1]]
        for rank, exercise in enumerate(top, 1):
            recommended.update(questions[exercise])
            if rank in per_student:
                union = recommended | correct
                per_student[rank].append(1.0 - len(recommended & correct) / len(union))
    if seen != set(correct_history):
        raise ValueError(f"Recommendation students differ from test split: {len(seen)} != {len(correct_history)}")
    return _summarize(per_student)


def _summarize(per_student: dict[int, list[float]]) -> dict[str, dict[str, float]]:
    return {str(n): {"mean": round(mean(values), 6),
                     "std": round(stdev(values), 6) if len(values) > 1 else 0.0}
            for n, values in per_student.items()}
