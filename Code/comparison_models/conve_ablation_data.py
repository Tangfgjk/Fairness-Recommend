import json
import shutil
from pathlib import Path

import numpy as np


MASTERY = "mastery"
SEQUENCE = "sequence"
FORGETTING = "forgetting"
VALID_TERMS = (MASTERY, SEQUENCE, FORGETTING)

CONVE_VARIANTS = {
    "ConvE_full": {
        "active_terms": (MASTERY, SEQUENCE, FORGETTING),
        "remove_relation_prefix": None,
    },
    "ConvE_no_seq": {
        "active_terms": (MASTERY, FORGETTING),
        "remove_relation_prefix": "pkc",
    },
    "ConvE_no_forgetting": {
        "active_terms": (MASTERY, SEQUENCE),
        "remove_relation_prefix": "exfr",
    },
    "ConvE_no_mastery": {
        "active_terms": (SEQUENCE, FORGETTING),
        "remove_relation_prefix": "mlkc",
    },
}

SHARED_DATA_FILES = (
    "Q.txt",
    "entities.dict",
    "relations.dict",
    "stu2know_mastery.json",
    "stu2know_seq.json",
    "stu2know_forget.json",
    "stu2ex_forget.json",
)


def parse_active_terms(value):
    if isinstance(value, str):
        terms = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    else:
        terms = tuple(value)
    if not terms:
        raise ValueError("At least one recommendation-distance term is required")
    unknown = sorted(set(terms) - set(VALID_TERMS))
    if unknown:
        raise ValueError(f"Unknown recommendation-distance terms: {','.join(unknown)}")
    if len(set(terms)) != len(terms):
        raise ValueError("Recommendation-distance terms must not contain duplicates")
    return tuple(term for term in VALID_TERMS if term in terms)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_q_matrix(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return [
            [int(value) for value in line.strip().split(",")]
            for line in fp
            if line.strip()
        ]


def calculate_recommendation_scores(
    mastery,
    sequence,
    exercise_forgetting,
    q_matrix,
    active_terms=VALID_TERMS,
    delta_1=0.8,
    delta_2=0.8,
):
    active_terms = parse_active_terms(active_terms)
    student_count = len(mastery)
    if len(sequence) != student_count or len(exercise_forgetting) != student_count:
        raise ValueError("Mastery, sequence, and forgetting student counts must match")
    if not q_matrix:
        raise ValueError("Q matrix must not be empty")
    exercise_count = len(q_matrix)

    all_scores = []
    for student_idx in range(student_count):
        mastery_row = mastery[student_idx]
        sequence_row = np.asarray(sequence[student_idx], dtype=float)
        forgetting_row = exercise_forgetting[student_idx]
        if len(forgetting_row) != exercise_count:
            raise ValueError(
                f"Student {student_idx} forgetting length {len(forgetting_row)} "
                f"does not match Q rows {exercise_count}"
            )
        if len(mastery_row) != len(sequence_row):
            raise ValueError(f"Student {student_idx} mastery and sequence lengths must match")

        student_scores = []
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

            if SEQUENCE in active_terms:
                q_vector = np.asarray(q_values, dtype=float)
                denominator = np.linalg.norm(q_vector) * np.linalg.norm(sequence_row) + 1e-9
                cosine_similarity = float(np.dot(q_vector, sequence_row.T) / denominator)
                total += cosine_similarity**2

            if FORGETTING in active_terms:
                forgetting_value = float(forgetting_row[exercise_idx])
                total += (float(delta_2) - forgetting_value) ** 2

            student_scores.append(round(float(np.sqrt(total)), 2))
        all_scores.append(student_scores)
    return all_scores


def _read_triples(path):
    triples = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid triple at {path}:{line_number}")
            triples.append(tuple(parts))
    return triples


def _student_sort_key(uid):
    if uid.startswith("uid") and uid[3:].isdigit():
        return (0, int(uid[3:]))
    return (1, uid)


def _student_index(uid):
    if not uid.startswith("uid") or not uid[3:].isdigit():
        raise ValueError(f"Unsupported learner entity name: {uid}")
    return int(uid[3:])


def extract_students(triples):
    students = set()
    for head, relation, tail in triples:
        if relation == "rec" and head.startswith("uid"):
            students.add(head)
        if head.startswith("uid"):
            students.add(head)
        if tail.startswith("uid"):
            students.add(tail)
    return students


def _filter_state_triples(triples, remove_relation_prefix):
    output = []
    for head, relation, tail in triples:
        if relation == "rec":
            continue
        if remove_relation_prefix and relation.startswith(remove_relation_prefix):
            continue
        output.append((head, relation, tail))
    return output


def _write_triples(path, triples):
    with Path(path).open("w", encoding="utf-8", newline="\n") as fp:
        for triple in triples:
            fp.write("\t".join(triple) + "\n")


def prepare_conve_variant_data_dir(
    source_dir,
    target_dir,
    model,
    top_k_rec=10,
    delta_1=0.8,
    delta_2=0.8,
    resume=False,
):
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    if model not in CONVE_VARIANTS:
        raise ValueError(f"Unknown ConvE variant: {model}")
    if int(top_k_rec) <= 0:
        raise ValueError("top_k_rec must be positive")
    config = CONVE_VARIANTS[model]
    manifest_path = target_dir / "variant_manifest.json"
    if resume and target_dir.exists() and any(target_dir.iterdir()):
        if not manifest_path.exists():
            raise ValueError(
                f"Legacy ConvE variant data found at {target_dir}. "
                "Use a new run-id instead of resuming across the new ablation formulas."
            )
        existing_manifest = load_json(manifest_path)
        expected = {
            "model": model,
            "active_terms": list(config["active_terms"]),
            "remove_relation_prefix": config["remove_relation_prefix"],
            "delta_1": float(delta_1),
            "delta_2": float(delta_2),
            "top_k_rec": int(top_k_rec),
        }
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in expected.items()
            if existing_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"ConvE variant manifest mismatch at {target_dir}: {mismatches}. "
                "Use a new run-id."
            )
        required_outputs = [
            target_dir / "triples.txt",
            target_dir / "test_triples.txt",
            target_dir / "stu2ex_recommend.json",
        ]
        missing_outputs = [str(path) for path in required_outputs if not path.exists()]
        if missing_outputs:
            raise ValueError(
                "ConvE variant manifest exists but generated files are missing: "
                + ", ".join(missing_outputs)
            )
        return existing_manifest
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in SHARED_DATA_FILES:
        source_file = source_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Required ConvE data file is missing: {source_file}")
        shutil.copy2(source_file, target_dir / name)

    mastery = load_json(source_dir / "stu2know_mastery.json")
    sequence = load_json(source_dir / "stu2know_seq.json")
    exercise_forgetting = load_json(source_dir / "stu2ex_forget.json")
    q_matrix = load_q_matrix(source_dir / "Q.txt")
    scores = calculate_recommendation_scores(
        mastery,
        sequence,
        exercise_forgetting,
        q_matrix,
        active_terms=config["active_terms"],
        delta_1=delta_1,
        delta_2=delta_2,
    )
    with (target_dir / "stu2ex_recommend.json").open("w", encoding="utf-8") as fp:
        json.dump(scores, fp)

    source_train = _read_triples(source_dir / "triples.txt")
    source_test = _read_triples(source_dir / "test_triples.txt")
    train_students = extract_students(source_train)
    test_students = extract_students(source_test)
    overlap = train_students & test_students
    if overlap:
        raise ValueError(
            "Train/test learner overlap found while preparing ConvE variants: "
            + ",".join(sorted(overlap, key=_student_sort_key))
        )

    train_triples = _filter_state_triples(
        source_train, config["remove_relation_prefix"]
    )
    test_triples = _filter_state_triples(
        source_test, config["remove_relation_prefix"]
    )
    rec_triples = []
    for uid in sorted(train_students, key=_student_sort_key):
        student_idx = _student_index(uid)
        if student_idx >= len(scores):
            raise ValueError(f"Learner {uid} is outside recommendation score matrix")
        top_exercises = sorted(
            enumerate(scores[student_idx]), key=lambda item: (item[1], item[0])
        )[: int(top_k_rec)]
        rec_triples.extend((uid, "rec", f"ex{exercise_idx}") for exercise_idx, _ in top_exercises)
    train_triples.extend(rec_triples)

    _write_triples(target_dir / "triples.txt", train_triples)
    _write_triples(target_dir / "test_triples.txt", test_triples)
    manifest = {
        "schema_version": 1,
        "model": model,
        "active_terms": list(config["active_terms"]),
        "remove_relation_prefix": config["remove_relation_prefix"],
        "formula": "sqrt(sum(active squared terms))",
        "selection": "smallest_distance_top_k",
        "delta_1": float(delta_1),
        "delta_2": float(delta_2),
        "top_k_rec": int(top_k_rec),
        "train_students": len(train_students),
        "test_students": len(test_students),
        "rec_triples": len(rec_triples),
        "source_dir": str(source_dir),
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return manifest
