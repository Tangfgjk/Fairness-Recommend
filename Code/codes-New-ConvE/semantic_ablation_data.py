"""Build graph-level ablation data for 2CKG4ER cognitive factors."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


MASTERY = "mastery"
FORGETTING = "forgetting"
VALID_TERMS = (MASTERY, FORGETTING)

GRAPH_ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "2CKG4ER_no_mastery": {
        "active_terms": (FORGETTING,),
        "remove_relation_prefix": "mlkc",
    },
    "2CKG4ER_no_forgetting": {
        "active_terms": (MASTERY,),
        "remove_relation_prefix": "exfr",
    },
}

GRAPH_ABLATIONS = tuple(GRAPH_ABLATION_CONFIGS)

GRAPH_ABLATION_ALIASES = {
    "feature_only_no_mastery": "2CKG4ER_no_mastery",
    "feature_only_no_forgetting": "2CKG4ER_no_forgetting",
}

SHARED_DATA_FILES = (
    "Q.txt",
    "entities.dict",
    "relations.dict",
    "stu2know_mastery.json",
    "stu2know_forget.json",
    "stu2ex_forget.json",
)

GENERATED_OR_REBUILT_FILES = {
    "triples.txt",
    "test_triples.txt",
    "stu2ex_recommend.json",
    "stu2ex_recommend_full_precision.json",
    "ablation_manifest.json",
    "variant_manifest.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_active_terms(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        terms = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    else:
        terms = tuple(str(item).strip().lower() for item in value if str(item).strip())
    if not terms:
        raise ValueError("At least one recommendation-distance term is required")
    unknown = sorted(set(terms) - set(VALID_TERMS))
    if unknown:
        raise ValueError(f"Unknown recommendation-distance terms: {','.join(unknown)}")
    if len(set(terms)) != len(terms):
        raise ValueError("Recommendation-distance terms must not contain duplicates")
    return tuple(term for term in VALID_TERMS if term in terms)


def load_q_matrix(path: Path) -> List[List[int]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append([int(value) for value in line.split(",")])
    if not rows:
        raise ValueError(f"Q matrix is empty: {path}")
    return rows


def calculate_recommendation_scores(
    mastery: Sequence[Sequence[float]],
    exercise_forgetting: Sequence[Sequence[float]],
    q_matrix: Sequence[Sequence[int]],
    active_terms: Sequence[str] = VALID_TERMS,
    delta_1: float = 0.8,
    delta_2: float = 0.8,
) -> List[List[float]]:
    active_terms = parse_active_terms(active_terms)
    student_count = len(mastery)
    if len(exercise_forgetting) != student_count:
        raise ValueError("Mastery and forgetting student counts must match")

    all_scores: List[List[float]] = []
    for student_idx in range(student_count):
        mastery_row = mastery[student_idx]
        forgetting_row = exercise_forgetting[student_idx]
        if len(forgetting_row) != len(q_matrix):
            raise ValueError(
                f"Student {student_idx} forgetting length {len(forgetting_row)} "
                f"does not match Q rows {len(q_matrix)}"
            )

        student_scores: List[float] = []
        for exercise_idx, q_values in enumerate(q_matrix):
            if len(q_values) != len(mastery_row):
                raise ValueError(
                    f"Q row {exercise_idx} width {len(q_values)} does not match "
                    f"knowledge count {len(mastery_row)}"
                )
            total = 0.0
            if MASTERY in active_terms:
                mastery_product = 1.0
                for knowledge_idx, is_linked in enumerate(q_values):
                    if int(is_linked) == 1:
                        mastery_product *= float(mastery_row[knowledge_idx])
                total += (float(delta_1) - mastery_product) ** 2

            if FORGETTING in active_terms:
                forgetting_value = float(forgetting_row[exercise_idx])
                total += (float(delta_2) - forgetting_value) ** 2

            # Keep full precision for ranking; rounding is only for human-facing storage.
            student_scores.append(float(np.sqrt(total)))
        all_scores.append(student_scores)
    return all_scores


def read_triples_text(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid triple at {path}:{line_number}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def write_triples_text(path: Path, triples: Iterable[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for triple in triples:
            fp.write("\t".join(triple) + "\n")


def student_sort_key(uid: str) -> Tuple[int, int | str]:
    if uid.startswith("uid") and uid[3:].isdigit():
        return (0, int(uid[3:]))
    return (1, uid)


def student_index(uid: str) -> int:
    if not uid.startswith("uid") or not uid[3:].isdigit():
        raise ValueError(f"Unsupported learner entity name: {uid}")
    return int(uid[3:])


def extract_students(triples: Sequence[Tuple[str, str, str]]) -> set[str]:
    students = set()
    for head, relation, tail in triples:
        if relation == "rec" and head.startswith("uid"):
            students.add(head)
        if head.startswith("uid"):
            students.add(head)
        if tail.startswith("uid"):
            students.add(tail)
    return students


def filter_state_triples(
    triples: Sequence[Tuple[str, str, str]],
    remove_relation_prefix: str | None,
) -> List[Tuple[str, str, str]]:
    output = []
    for head, relation, tail in triples:
        if relation == "rec":
            continue
        if remove_relation_prefix and relation.startswith(remove_relation_prefix):
            continue
        output.append((head, relation, tail))
    return output


def copy_shared_inputs(source_dir: Path, target_dir: Path) -> None:
    for name in SHARED_DATA_FILES:
        source_file = source_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Required SemanticConvE data file is missing: {source_file}")
        shutil.copy2(source_file, target_dir / name)
    feature_dir = source_dir / "semantic_kg_features"
    if not feature_dir.exists():
        raise FileNotFoundError(f"semantic_kg_features is missing: {feature_dir}")
    shutil.copytree(feature_dir, target_dir / "semantic_kg_features", dirs_exist_ok=True)
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name in GENERATED_OR_REBUILT_FILES or source_file.name in SHARED_DATA_FILES:
            continue
        if source_file.suffix.lower() not in {".txt", ".csv", ".json", ".pkl"}:
            continue
        shutil.copy2(source_file, target_dir / source_file.name)


def expected_manifest(
    source_dir: Path,
    ablation: str,
    top_k_rec: int,
    delta_1: float,
    delta_2: float,
) -> Dict[str, Any]:
    ablation = GRAPH_ABLATION_ALIASES.get(ablation, ablation)
    config = GRAPH_ABLATION_CONFIGS[ablation]
    return {
        "schema_version": 1,
        "ablation": ablation,
        "active_terms": list(config["active_terms"]),
        "remove_relation_prefix": config["remove_relation_prefix"],
        "formula": "sqrt(sum(active squared terms))",
        "selection": "smallest_distance_top_k",
        "delta_1": float(delta_1),
        "delta_2": float(delta_2),
        "top_k_rec": int(top_k_rec),
        "source_dir": str(source_dir.resolve()),
    }


def prepare_semantic_ablation_graph(
    source_dir: str | Path,
    target_dir: str | Path,
    ablation: str,
    top_k_rec: int = 10,
    delta_1: float = 0.8,
    delta_2: float = 0.8,
    resume: bool = False,
) -> Dict[str, Any]:
    ablation = GRAPH_ABLATION_ALIASES.get(ablation, ablation)
    if ablation not in GRAPH_ABLATION_CONFIGS:
        raise ValueError(f"Unknown graph ablation: {ablation}")
    if int(top_k_rec) <= 0:
        raise ValueError("top_k_rec must be positive")

    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    manifest_path = target_dir / "ablation_manifest.json"
    base_manifest = expected_manifest(source_dir, ablation, top_k_rec, delta_1, delta_2)

    if resume and manifest_path.exists():
        existing = read_json(manifest_path)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in base_manifest.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Ablation manifest mismatch at {target_dir}: {mismatches}")
        copy_shared_inputs(source_dir, target_dir)
        required_outputs = [target_dir / "triples.txt", target_dir / "test_triples.txt", target_dir / "stu2ex_recommend.json"]
        missing = [str(path) for path in required_outputs if not path.exists()]
        if missing:
            raise ValueError("Ablation manifest exists but generated files are missing: " + ", ".join(missing))
        return existing

    if target_dir.exists() and not resume:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_shared_inputs(source_dir, target_dir)

    config = GRAPH_ABLATION_CONFIGS[ablation]
    mastery = read_json(source_dir / "stu2know_mastery.json")
    exercise_forgetting = read_json(source_dir / "stu2ex_forget.json")
    q_matrix = load_q_matrix(source_dir / "Q.txt")
    scores = calculate_recommendation_scores(
        mastery=mastery,
        exercise_forgetting=exercise_forgetting,
        q_matrix=q_matrix,
        active_terms=config["active_terms"],
        delta_1=delta_1,
        delta_2=delta_2,
    )
    write_json(target_dir / "stu2ex_recommend_full_precision.json", scores)
    rounded_scores = [[round(value, 6) for value in row] for row in scores]
    write_json(target_dir / "stu2ex_recommend.json", rounded_scores)

    source_train = read_triples_text(source_dir / "triples.txt")
    source_test = read_triples_text(source_dir / "test_triples.txt")
    train_students = extract_students(source_train)
    test_students = extract_students(source_test)
    overlap = train_students & test_students
    if overlap:
        raise ValueError(
            "Train/test learner overlap found while preparing 2CKG4ER graph ablation: "
            + ",".join(sorted(overlap, key=student_sort_key))
        )

    train_triples = filter_state_triples(source_train, config["remove_relation_prefix"])
    test_triples = filter_state_triples(source_test, config["remove_relation_prefix"])

    rec_triples = []
    for uid in sorted(train_students, key=student_sort_key):
        idx = student_index(uid)
        if idx >= len(scores):
            raise ValueError(f"Learner {uid} is outside recommendation score matrix")
        top_exercises = sorted(enumerate(scores[idx]), key=lambda item: (item[1], item[0]))[: int(top_k_rec)]
        rec_triples.extend((uid, "rec", f"ex{exercise_idx}") for exercise_idx, _ in top_exercises)
    train_triples.extend(rec_triples)

    write_triples_text(target_dir / "triples.txt", train_triples)
    write_triples_text(target_dir / "test_triples.txt", test_triples)
    manifest = dict(base_manifest)
    manifest.update(
        {
            "train_students": len(train_students),
            "test_students": len(test_students),
            "rec_triples": len(rec_triples),
            "train_triples": len(train_triples),
            "test_triples": len(test_triples),
        }
    )
    write_json(manifest_path, manifest)
    return manifest
