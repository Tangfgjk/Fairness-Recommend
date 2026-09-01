from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_DATASETS, data_fin_root, front_dir, graph_dir, jsonable, load_raw_dataset, matrix_stats, read_json, read_matrix_json, read_q_matrix, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw-front features and final ER graph coverage.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument("--require-graph", action="store_true")
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def validate_one(dataset: str, root: Path, require_graph: bool, raw_name: str = "auto") -> dict:
    raw = load_raw_dataset(dataset, root, raw_name)
    front = front_dir(dataset, root)
    errors: list[str] = []
    warnings: list[str] = []
    protocol = front / "protocol.json"
    if not protocol.exists():
        errors.append("missing front protocol")
    else:
        payload = read_json(protocol)
        if sorted(payload.get("outer_train_uids", []) + payload.get("outer_test_uids", [])) != list(range(raw.student_count)):
            errors.append("protocol outer cohort does not exactly cover raw students")
    state_paths = [front / "stu2know_mastery.json", front / "stu2know_forget.json", front / "stu2ex_forget.json"]
    states: dict[str, np.ndarray] = {}
    for path in state_paths:
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        try:
            states[path.stem] = read_matrix_json(path)
        except Exception as exc:
            errors.append(f"cannot read {path.name}: {exc}")
    for name in ["stu2know_mastery", "stu2know_forget"]:
        if name in states and states[name].shape != (raw.student_count, raw.concept_count):
            errors.append(f"{name} shape={states[name].shape}, expected={(raw.student_count, raw.concept_count)}")
    if "stu2ex_forget" in states and states["stu2ex_forget"].shape != (raw.student_count, raw.exercise_count):
        errors.append(f"stu2ex_forget shape={states['stu2ex_forget'].shape}, expected={(raw.student_count, raw.exercise_count)}")
    for name, matrix in states.items():
        if not np.all(np.isfinite(matrix)):
            errors.append(f"{name} contains non-finite values")
        if name != "stu2ex_recommend" and (np.min(matrix) < -1e-6 or np.max(matrix) > 1.0 + 1e-6):
            errors.append(f"{name} is outside [0, 1]")
    q_mirt = front / "mirt" / "a_param_q_constrained.npy"
    if not q_mirt.exists():
        errors.append("missing Q-constrained MIRT a parameters")
    else:
        a = np.load(q_mirt)
        q = read_q_matrix(raw.root / "Q.txt")
        if a.shape != q.shape:
            errors.append(f"Q-MIRT a shape={a.shape}, Q shape={q.shape}")
        elif not np.allclose(a[q == 0], 0.0, atol=1e-7):
            errors.append("Q-MIRT has non-zero discrimination outside Q-linked concepts")
    semantic_dir = front / "semantic_kg_features"
    for name in [
        "text_embeddings/text_embedding_manifest.json",
        "entity_features/learner_pedagogy.json",
        "entity_features/exercise_semantics.json",
        "entity_features/concept_semantics.json",
        "irt_features/exercise_irt_features.json",
    ]:
        if not (semantic_dir / name).exists():
            errors.append(f"missing semantic feature file: {name}")
    graph = graph_dir(dataset, root)
    if require_graph:
        for name in ["entities.dict", "relations.dict", "Q.txt", "triples.txt", "test_triples.txt", "semantic_kg_features"]:
            if not (graph / name).exists():
                errors.append(f"missing graph file: {name}")
    report = {
        "dataset": dataset,
        "raw_counts": {"students": raw.student_count, "exercises": raw.exercise_count, "concepts": raw.concept_count, "interactions": int(len(raw.interactions))},
        "state_stats": {name: matrix_stats(value) for name, value in states.items()},
        "warnings": warnings,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    return report


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    reports = [validate_one(dataset, root, args.require_graph, args.raw_name) for dataset in datasets]
    payload = {"data_fin_root": root, "reports": reports, "status": "passed" if all(report["status"] == "passed" for report in reports) else "failed"}
    if args.output_file:
        write_json(args.output_file, payload)
    print(__import__("json").dumps(jsonable(payload), ensure_ascii=False, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
