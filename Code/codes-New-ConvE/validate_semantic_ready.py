"""Validate that datasets are ready for SemanticConvE experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from semantic_experiment_utils import (
    DEFAULT_DATASETS,
    default_data_root,
    graph_path_for_dataset,
    jsonable,
    parse_csv_list,
    read_json,
    write_json,
)


REQUIRED_GRAPH_FILES = [
    "entities.dict",
    "relations.dict",
    "triples.txt",
    "test_triples.txt",
    "Q.txt",
]

REQUIRED_FEATURE_FILES = [
    "entity_features/concept_semantics.json",
    "entity_features/exercise_semantics.json",
    "entity_features/learner_pedagogy.json",
    "irt_features/exercise_irt_features.json",
    "text_embeddings/exercise_text_embeddings.npy",
    "text_embeddings/exercise_text_tokens.npy",
    "text_embeddings/relation_text_tokens.npy",
    "text_embeddings/relation_text_embeddings.npy",
    "text_embeddings/shared_text_encoder.pt",
    "text_embeddings/shared_text_encoder_config.json",
    "text_embeddings/relation_semantics.json",
    "text_embeddings/text_embedding_manifest.json",
]


def count_dict_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def count_q_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def validate_dataset(dataset: str, data_root: Path, allow_template: bool = False, graph_subdir: str | None = None) -> Dict[str, Any]:
    graph_path = graph_path_for_dataset(dataset, data_root, graph_subdir=graph_subdir)
    feature_dir = graph_path / "semantic_kg_features"
    errors: List[str] = []
    warnings: List[str] = []

    for file_name in REQUIRED_GRAPH_FILES:
        if not (graph_path / file_name).exists():
            errors.append(f"missing graph file: {file_name}")

    for file_name in REQUIRED_FEATURE_FILES:
        if not (feature_dir / file_name).exists():
            errors.append(f"missing feature file: semantic_kg_features/{file_name}")

    entity_count = count_dict_rows(graph_path / "entities.dict") if (graph_path / "entities.dict").exists() else 0
    relation_count = count_dict_rows(graph_path / "relations.dict") if (graph_path / "relations.dict").exists() else 0
    q_count = count_q_rows(graph_path / "Q.txt") if (graph_path / "Q.txt").exists() else 0

    definition_summary: Dict[str, int] = {}
    concept_count = 0
    template_count = 0
    if (feature_dir / "entity_features" / "concept_semantics.json").exists():
        concept_data = read_json(feature_dir / "entity_features" / "concept_semantics.json")
        concepts = concept_data.get("concepts", {})
        concept_count = len(concepts)
        source_counter = Counter(str(item.get("definition_source", "missing")) for item in concepts.values())
        definition_summary = dict(source_counter)
        template_count = sum(
            1
            for item in concepts.values()
            if item.get("definition_source") == "template"
            or "used to describe the skills or learning objectives" in str(item.get("definition", ""))
        )
        if template_count and not allow_template:
            errors.append(f"template concept definitions detected: {template_count}/{concept_count}")

    exercise_count = 0
    if (feature_dir / "entity_features" / "exercise_semantics.json").exists():
        exercise_count = len(read_json(feature_dir / "entity_features" / "exercise_semantics.json").get("exercises", {}))

    learner_count = 0
    if (feature_dir / "entity_features" / "learner_pedagogy.json").exists():
        learner_count = len(read_json(feature_dir / "entity_features" / "learner_pedagogy.json").get("learners", {}))

    concept_embedding_shape = None
    exercise_embedding_shape = None
    manifest_dim = None
    manifest_path = feature_dir / "text_embeddings" / "text_embedding_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest_dim = int(manifest.get("embedding_dim") or 0)
        if manifest_dim <= 0:
            errors.append(f"text embedding dim should be positive, got {manifest_dim}")
        model_name = str(
            manifest.get("model")
            or manifest.get("model_name")
            or manifest.get("source")
            or ""
        ).lower()
        if "bge" in model_name or "sentence" in model_name:
            errors.append("legacy BGE/SentenceTransformer text embeddings are not allowed in V10")
        if "ektm" not in model_name and "topic" not in model_name:
            errors.append("text embedding manifest must identify EKTM_mirt TopicRNNModel topic_v source")
    if (feature_dir / "text_embeddings" / "concept_text_embeddings.npy").exists():
        warnings.append("concept_text_embeddings.npy is legacy-only; V-Fin7 knowledge concepts use ID embeddings")
    if (feature_dir / "text_embeddings" / "exercise_text_embeddings.npy").exists():
        exercise_embeddings = np.load(feature_dir / "text_embeddings" / "exercise_text_embeddings.npy", mmap_mode="r")
        exercise_embedding_shape = list(exercise_embeddings.shape)
        if exercise_embeddings.ndim != 2 or (manifest_dim and exercise_embeddings.shape[1] != manifest_dim):
            errors.append(f"bad exercise embedding shape: {exercise_embedding_shape}")
        if exercise_count and exercise_embeddings.shape[0] != exercise_count:
            errors.append(f"exercise embedding rows {exercise_embeddings.shape[0]} != exercise count {exercise_count}")

    if q_count and exercise_count and q_count != exercise_count:
        errors.append(f"Q rows {q_count} != exercise semantics count {exercise_count}")

    expected_feature_dir = graph_path / "semantic_kg_features"
    if feature_dir != expected_feature_dir:
        errors.append(f"feature_dir mismatch: {feature_dir} != {expected_feature_dir}")

    return {
        "dataset": dataset,
        "graph_path": graph_path,
        "feature_dir": feature_dir,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "q_exercise_count": q_count,
        "concept_count": concept_count,
        "exercise_count": exercise_count,
        "learner_count": learner_count,
        "definition_source_summary": definition_summary,
        "template_definition_count": template_count,
        "concept_embedding_shape": concept_embedding_shape,
        "exercise_embedding_shape": exercise_embedding_shape,
        "text_embedding_dim": manifest_dim,
        "warnings": warnings,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SemanticConvE dataset readiness.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--allow-template", action="store_true")
    parser.add_argument("--graph-subdir", default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        validate_dataset(dataset, args.data_root, allow_template=args.allow_template, graph_subdir=args.graph_subdir)
        for dataset in parse_csv_list(args.datasets)
    ]
    summary = {
        "data_root": args.data_root,
        "reports": reports,
        "status": "passed" if all(report["status"] == "passed" for report in reports) else "failed",
    }
    if args.output_file:
        write_json(args.output_file, summary)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
