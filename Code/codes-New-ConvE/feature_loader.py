"""Feature loading utilities for SemanticConvE.

The loader aligns semantic and pedagogical side features with the entity and
relation ids used by the existing KG4ER graph files.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ENTITY_TYPE_TO_ID = {"uid": 0, "kc": 1, "ex": 2, "other": 3}
RELATION_TYPE_TO_ID = {"rec": 0, "mlkc": 1, "exfr": 2, "other": 3}
NO_CLUSTER_ID = 5


@dataclass
class SemanticFeatureBundle:
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]
    id2entity: Dict[int, str]
    id2relation: Dict[int, str]
    text_features: torch.Tensor
    numeric_features: torch.Tensor
    state_features: torch.Tensor
    entity_type_ids: torch.Tensor
    cluster_ids: torch.Tensor
    semantic_quality: torch.Tensor
    relation_type_ids: torch.Tensor
    relation_strengths: torch.Tensor
    exercise_entity_ids: torch.Tensor
    exercise_text_token_ids: torch.Tensor
    exercise_entity_to_text_index: torch.Tensor
    relation_text_token_ids: torch.Tensor
    shared_text_encoder_state: Dict[str, Any]
    shared_text_encoder_config: Dict[str, Any]
    numeric_feature_slices: Dict[str, Tuple[int, int]]
    state_feature_slices: Dict[str, Tuple[int, int]]
    metadata: Dict[str, Any]

    @property
    def nentity(self) -> int:
        return len(self.entity2id)

    @property
    def nrelation(self) -> int:
        return len(self.relation2id)

    @property
    def text_dim(self) -> int:
        return int(self.text_features.shape[1])

    @property
    def numeric_dim(self) -> int:
        return int(self.numeric_features.shape[1])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_dict(path: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split("\t")
            result[name] = int(idx)
    return result


def entity_kind(entity_name: str) -> str:
    if entity_name.startswith("uid"):
        return "uid"
    if entity_name.startswith("kc"):
        return "kc"
    if entity_name.startswith("ex"):
        return "ex"
    return "other"


def relation_kind(relation_name: str) -> str:
    if relation_name == "rec":
        return "rec"
    if relation_name.startswith("mlkc"):
        return "mlkc"
    if relation_name.startswith("exfr"):
        return "exfr"
    if relation_name == "ex_has_kc":
        return "ex_has_kc"
    if relation_name == "kc_has_ex":
        return "kc_has_ex"
    return "other"


def relation_strength(relation_name: str) -> float:
    if relation_name == "rec":
        return 1.0
    match = re.search(r"[-+]?\d*\.?\d+", relation_name)
    if not match:
        return 0.0
    value = float(match.group(0))
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def locate_feature_dir(data_path: Path, feature_dir: Optional[Path] = None) -> Path:
    if feature_dir is not None:
        return Path(feature_dir)
    candidate = data_path / "semantic_kg_features"
    if candidate.exists():
        return candidate
    prepared_candidate = data_path / "prepared_for_kt" / "semantic_kg_features"
    if prepared_candidate.exists():
        return prepared_candidate
    raise FileNotFoundError(f"semantic_kg_features not found under {data_path}")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def infer_semantic_quality(feature: Dict[str, Any], entity_name: str) -> float:
    """Infer a fixed prior for how reliable an entity's semantic text is.

    The value is a non-trainable quality prior used by SemanticConvE V3 gates.
    It does not measure correctness of a definition; it only encodes how much
    we trust the semantic source before training.
    """

    if "semantic_quality" in feature:
        return _clamp01(_as_float(feature.get("semantic_quality"), 0.0))

    source = str(
        feature.get("semantic_quality_source")
        or feature.get("definition_source")
        or feature.get("text_source")
        or feature.get("semantic_source")
        or ""
    ).lower()
    text = str(
        feature.get("text_for_embedding")
        or feature.get("question_text")
        or feature.get("definition")
        or feature.get("text")
        or ""
    ).strip()

    if "template" in source:
        return 0.0
    if any(token in source for token in ["raw_text", "question_text", "exercise_text"]):
        return 1.0
    if any(token in source for token in ["llm", "deepseek", "definition"]):
        return 0.8
    if any(token in source for token in ["answer_type", "problem_id"]):
        return 0.3
    if any(token in source for token in ["structured", "problem", "hierarchy", "step"]):
        return 0.5
    if text:
        return 0.5
    return 0.0


def _log_count(value: Any) -> float:
    # Interaction counts can be thousands or more, while the other pedagogical
    # features are already in [0, 1]. Keep the count signal without letting it
    # dominate the numeric projector.
    return _clamp01(math.log1p(max(0.0, _as_float(value, 0.0))) / 10.0)


def _load_text_embeddings(feature_dir: Path) -> tuple[Dict[str, np.ndarray], int, Dict[str, Any], Dict[str, Any]]:
    emb_dir = feature_dir / "text_embeddings"
    manifest_path = emb_dir / "text_embedding_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. SemanticConvE requires EKTM-MIRT Bi-GRU text embeddings. "
            "Run front_pipeline/run_front_pipeline.py through the semantic-feature stage first."
        )
    manifest = read_json(manifest_path)
    model_name = str(
        manifest.get("model")
        or manifest.get("model_name")
        or manifest.get("source")
        or ""
    ).lower()
    if "bge" in model_name or "sentence" in model_name:
        raise ValueError(
            f"Legacy BGE/SentenceTransformer text embeddings are not allowed: {manifest_path}. "
            "Use EKTM_mirt TopicRNNModel topic_v embeddings for V10."
        )
    if "ektm" not in model_name and "topic" not in model_name:
        raise ValueError(
            f"Text embedding manifest must identify EKTM_mirt topic_v source: {manifest_path}. "
            "Expected model/source to contain EKTM or topic."
        )
    files = manifest.get("files", {})
    exercise_embeddings = np.load(emb_dir / files["exercise_text_embeddings"])
    text_by_entity: Dict[str, np.ndarray] = {}
    for entity_id, emb in zip(manifest.get("exercise_entity_ids", []), exercise_embeddings):
        text_by_entity[entity_id] = np.asarray(emb, dtype=np.float32)
    text_dim = int(manifest.get("embedding_dim") or 0)
    if text_dim <= 0 and text_by_entity:
        text_dim = int(next(iter(text_by_entity.values())).shape[0])
    required_dynamic = ["exercise_text_tokens", "relation_text_tokens", "shared_text_encoder", "shared_text_encoder_config"]
    missing_dynamic = [name for name in required_dynamic if name not in files or not (emb_dir / files[name]).exists()]
    if missing_dynamic:
        raise FileNotFoundError(
            "V-Fin7 requires shared EKTM_mirt text artifacts: " + ", ".join(missing_dynamic)
        )
    try:
        encoder_state = torch.load(emb_dir / files["shared_text_encoder"], map_location="cpu", weights_only=True)
    except TypeError:
        encoder_state = torch.load(emb_dir / files["shared_text_encoder"], map_location="cpu")
    artifacts = {
        "exercise_tokens": np.load(emb_dir / files["exercise_text_tokens"]).astype(np.int64),
        "relation_tokens": np.load(emb_dir / files["relation_text_tokens"]).astype(np.int64),
        "encoder_state": encoder_state,
        "encoder_config": read_json(emb_dir / files["shared_text_encoder_config"]),
    }
    return text_by_entity, text_dim, manifest, artifacts


def _load_semantic_metadata(feature_dir: Path) -> Dict[str, Dict[str, Any]]:
    entity_dir = feature_dir / "entity_features"
    metadata: Dict[str, Dict[str, Any]] = {}
    for file_name, root_key in [
        ("concept_semantics.json", "concepts"),
        ("exercise_semantics.json", "exercises"),
        ("learner_pedagogy.json", "learners"),
    ]:
        path = entity_dir / file_name
        if not path.exists():
            continue
        payload = read_json(path)
        entries = payload.get(root_key, {})
        if isinstance(entries, dict):
            for entity_id, item in entries.items():
                if isinstance(item, dict):
                    metadata[str(entity_id)] = item
    return metadata


def _learner_numeric(item: Dict[str, Any]) -> List[float]:
    return [
        _as_float(item.get("theta_norm"), 0.5),
        _as_float(item.get("overall_mastery_irt"), 0.5),
        _as_float(item.get("overall_mastery_kt_mean"), 0.0),
        _as_float(item.get("correct_rate"), 0.0),
        _log_count(item.get("history_length")),
        _as_float(item.get("concept_mastery_mean"), 0.5),
        _as_float(item.get("concept_mastery_std"), 0.0),
    ]


def _learner_irt_numeric(item: Dict[str, Any]) -> List[float]:
    return [
        _as_float(item.get("theta_norm"), 0.5),
    ]


def _learner_stat_numeric(item: Dict[str, Any]) -> List[float]:
    concept_mean = _as_float(
        item.get("concept_mastery_mean_from_interactions", item.get("concept_mastery_mean")),
        0.5,
    )
    concept_std = _as_float(
        item.get("concept_mastery_std_from_interactions", item.get("concept_mastery_std")),
        0.0,
    )
    return [
        _as_float(item.get("overall_mastery_kt_mean"), 0.0),
        _as_float(item.get("correct_rate"), 0.0),
        _log_count(item.get("history_length")),
        concept_mean,
        concept_std,
    ]


def _exercise_numeric(item: Dict[str, Any]) -> List[float]:
    return [
        _as_float(item.get("difficulty_norm"), 0.5),
        _as_float(item.get("discrimination_norm"), 0.0),
        _as_float(item.get("correct_rate"), 0.0),
        _log_count(item.get("interaction_count")),
        0.0,
        0.0,
        0.0,
    ]


def _exercise_irt_numeric(item: Dict[str, Any]) -> List[float]:
    return [
        _as_float(item.get("difficulty_norm"), 0.5),
        _as_float(item.get("discrimination_norm"), 0.0),
    ]


def _exercise_stat_numeric(item: Dict[str, Any]) -> List[float]:
    correct_rate = _as_float(item.get("correct_rate"), 0.0)
    error_rate = _as_float(item.get("error_rate"), 1.0 - correct_rate)
    high_error = _as_float(item.get("high_group_error_rate"), 0.0)
    low_error = _as_float(item.get("low_group_error_rate"), 0.0)
    return [
        correct_rate,
        error_rate,
        _log_count(item.get("interaction_count")),
        _as_float(item.get("high_low_gap"), max(0.0, low_error - high_error)),
        high_error,
        low_error,
        _as_float(item.get("mastery_response_corr"), 0.0),
    ]


def _concept_numeric(width: int) -> List[float]:
    return [0.0] * width


def _read_state_matrix(data_path: Path, name: str) -> List[List[float]]:
    path = data_path / name
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    return [[_as_float(value, 0.0) for value in row] for row in payload if isinstance(row, list)]


def _safe_state_row(matrix: List[List[float]], index: int, width: int) -> List[float]:
    if 0 <= index < len(matrix):
        row = list(matrix[index])
        if len(row) >= width:
            return row[:width]
        return row + [0.0] * (width - len(row))
    return [0.0] * width


def _uid_index(entity_name: str) -> int | None:
    if entity_name.startswith("uid") and entity_name[3:].isdigit():
        return int(entity_name[3:])
    return None


def load_semantic_feature_bundle(
    data_path: str | Path,
    feature_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> SemanticFeatureBundle:
    data_path = Path(data_path)
    feature_dir_path = locate_feature_dir(data_path, Path(feature_dir) if feature_dir else None)
    entity2id = read_dict(data_path / "entities.dict")
    relation2id = read_dict(data_path / "relations.dict")
    id2entity = {idx: name for name, idx in entity2id.items()}
    id2relation = {idx: name for name, idx in relation2id.items()}

    text_by_entity, text_dim, text_manifest, text_artifacts = _load_text_embeddings(feature_dir_path)
    semantic_metadata = _load_semantic_metadata(feature_dir_path)
    entity_dir = feature_dir_path / "entity_features"
    irt_dir = feature_dir_path / "irt_features"
    learner_data = read_json(entity_dir / "learner_pedagogy.json").get("learners", {})
    exercise_irt = read_json(irt_dir / "exercise_irt_features.json").get("exercises", {})

    learner_irt_width = 1
    exercise_irt_width = 2
    irt_width = learner_irt_width + exercise_irt_width
    learner_stat_width = 5
    exercise_stat_width = 7
    stat_width = learner_stat_width + exercise_stat_width
    numeric_width = irt_width + stat_width
    numeric_slices = {
        "learner_irt": (0, learner_irt_width),
        "exercise_irt": (learner_irt_width, irt_width),
        "learner_stat": (irt_width, irt_width + learner_stat_width),
        "exercise_stat": (irt_width + learner_stat_width, numeric_width),
    }

    mastery_matrix = _read_state_matrix(data_path, "stu2know_mastery.json")
    forget_matrix = _read_state_matrix(data_path, "stu2know_forget.json")
    knowledge_width = 0
    for matrix in (mastery_matrix, forget_matrix):
        if matrix:
            knowledge_width = max(knowledge_width, len(matrix[0]))
    state_slices = {
        "mastery": (0, knowledge_width),
        "forgetting": (knowledge_width, knowledge_width * 2),
        "learner_irt": (knowledge_width * 2, knowledge_width * 2 + learner_irt_width),
        "learner_stat": (
            knowledge_width * 2 + learner_irt_width,
            knowledge_width * 2 + learner_irt_width + learner_stat_width,
        ),
        "cluster": (
            knowledge_width * 2 + learner_irt_width + learner_stat_width,
            knowledge_width * 2 + learner_irt_width + learner_stat_width + 1,
        ),
    }
    state_width = state_slices["cluster"][1]

    nentity = len(entity2id)
    text_array = np.zeros((nentity, text_dim), dtype=np.float32)
    numeric_array = np.zeros((nentity, numeric_width), dtype=np.float32)
    state_array = np.zeros((nentity, state_width), dtype=np.float32)
    type_array = np.zeros((nentity,), dtype=np.int64)
    cluster_array = np.full((nentity,), NO_CLUSTER_ID, dtype=np.int64)
    semantic_quality_array = np.zeros((nentity, 1), dtype=np.float32)

    for entity_name, entity_id in entity2id.items():
        kind = entity_kind(entity_name)
        type_array[entity_id] = ENTITY_TYPE_TO_ID.get(kind, ENTITY_TYPE_TO_ID["other"])
        feature_item = semantic_metadata.get(entity_name, {})
        if entity_name in text_by_entity:
            text_array[entity_id] = text_by_entity[entity_name]
            if not feature_item:
                feature_item = {"text_for_embedding": "available_text_embedding"}
        semantic_quality_array[entity_id, 0] = infer_semantic_quality(feature_item, entity_name)
        if kind == "uid":
            item = learner_data.get(entity_name, {})
            learner_irt = _learner_irt_numeric(item)
            learner_stat = _learner_stat_numeric(item)
            numeric_array[entity_id, numeric_slices["learner_irt"][0] : numeric_slices["learner_irt"][1]] = np.asarray(
                learner_irt,
                dtype=np.float32,
            )
            numeric_array[entity_id, numeric_slices["learner_stat"][0] : numeric_slices["learner_stat"][1]] = np.asarray(
                learner_stat,
                dtype=np.float32,
            )
            cluster_array[entity_id] = int(item.get("cluster_id", NO_CLUSTER_ID))
            uid_idx = _uid_index(entity_name)
            if uid_idx is not None:
                state_array[entity_id, state_slices["mastery"][0] : state_slices["mastery"][1]] = np.asarray(
                    _safe_state_row(mastery_matrix, uid_idx, knowledge_width),
                    dtype=np.float32,
                )
                state_array[entity_id, state_slices["forgetting"][0] : state_slices["forgetting"][1]] = np.asarray(
                    _safe_state_row(forget_matrix, uid_idx, knowledge_width),
                    dtype=np.float32,
                )
            state_array[entity_id, state_slices["learner_irt"][0] : state_slices["learner_irt"][1]] = np.asarray(
                learner_irt,
                dtype=np.float32,
            )
            state_array[entity_id, state_slices["learner_stat"][0] : state_slices["learner_stat"][1]] = np.asarray(
                learner_stat,
                dtype=np.float32,
            )
            state_array[entity_id, state_slices["cluster"][0]] = _clamp01(cluster_array[entity_id] / max(1, NO_CLUSTER_ID - 1))
        elif kind == "ex":
            item = exercise_irt.get(entity_name, {})
            numeric_array[entity_id, numeric_slices["exercise_irt"][0] : numeric_slices["exercise_irt"][1]] = np.asarray(
                _exercise_irt_numeric(item),
                dtype=np.float32,
            )
            numeric_array[entity_id, numeric_slices["exercise_stat"][0] : numeric_slices["exercise_stat"][1]] = np.asarray(
                _exercise_stat_numeric(item),
                dtype=np.float32,
            )
        elif kind == "kc":
            numeric_array[entity_id] = np.asarray(_concept_numeric(numeric_width), dtype=np.float32)

    nrelation = len(relation2id)
    relation_type_array = np.zeros((nrelation,), dtype=np.int64)
    relation_strength_array = np.zeros((nrelation, 1), dtype=np.float32)
    for relation_name, relation_id in relation2id.items():
        kind = relation_kind(relation_name)
        relation_type_array[relation_id] = RELATION_TYPE_TO_ID.get(kind, RELATION_TYPE_TO_ID["other"])
        relation_strength_array[relation_id, 0] = relation_strength(relation_name)

    exercise_ids = sorted(
        entity_id for name, entity_id in entity2id.items() if entity_kind(name) == "ex"
    )
    exercise_entity_to_text_index = np.full((nentity,), -1, dtype=np.int64)
    for name, entity_id in entity2id.items():
        if entity_kind(name) == "ex" and name[2:].isdigit():
            index = int(name[2:])
            if index < text_artifacts["exercise_tokens"].shape[0]:
                exercise_entity_to_text_index[entity_id] = index

    return SemanticFeatureBundle(
        entity2id=entity2id,
        relation2id=relation2id,
        id2entity=id2entity,
        id2relation=id2relation,
        text_features=torch.tensor(text_array, dtype=torch.float32, device=device),
        numeric_features=torch.tensor(numeric_array, dtype=torch.float32, device=device),
        state_features=torch.tensor(state_array, dtype=torch.float32, device=device),
        entity_type_ids=torch.tensor(type_array, dtype=torch.long, device=device),
        cluster_ids=torch.tensor(cluster_array, dtype=torch.long, device=device),
        semantic_quality=torch.tensor(semantic_quality_array, dtype=torch.float32, device=device),
        relation_type_ids=torch.tensor(relation_type_array, dtype=torch.long, device=device),
        relation_strengths=torch.tensor(relation_strength_array, dtype=torch.float32, device=device),
        exercise_entity_ids=torch.tensor(exercise_ids, dtype=torch.long, device=device),
        exercise_text_token_ids=torch.tensor(text_artifacts["exercise_tokens"], dtype=torch.long, device=device),
        exercise_entity_to_text_index=torch.tensor(exercise_entity_to_text_index, dtype=torch.long, device=device),
        relation_text_token_ids=torch.tensor(text_artifacts["relation_tokens"], dtype=torch.long, device=device),
        shared_text_encoder_state=text_artifacts["encoder_state"],
        shared_text_encoder_config=text_artifacts["encoder_config"],
        numeric_feature_slices=numeric_slices,
        state_feature_slices=state_slices,
        metadata={
            "data_path": str(data_path),
            "feature_dir": str(feature_dir_path),
            "text_manifest": text_manifest,
            "numeric_feature_names": [
                "learner_theta_norm",
                "exercise_difficulty_norm",
                "exercise_discrimination_norm",
                "learner_overall_mastery_kt_mean",
                "learner_correct_rate",
                "learner_log_history_length",
                "learner_concept_mastery_mean",
                "learner_concept_mastery_std",
                "exercise_correct_rate",
                "exercise_error_rate",
                "exercise_log_interaction_count",
                "exercise_high_low_gap",
                "exercise_high_group_error_rate",
                "exercise_low_group_error_rate",
                "exercise_mastery_response_corr",
            ],
            "numeric_feature_slices": numeric_slices,
            "state_feature_slices": state_slices,
            "state_feature_dim": int(state_width),
            "knowledge_state_width": int(knowledge_width),
            "entity_type_to_id": ENTITY_TYPE_TO_ID,
            "relation_type_to_id": RELATION_TYPE_TO_ID,
            "no_cluster_id": NO_CLUSTER_ID,
            "semantic_quality": {
                "enabled": True,
                "mean": float(np.mean(semantic_quality_array)) if semantic_quality_array.size else 0.0,
                "nonzero_count": int(np.count_nonzero(semantic_quality_array)),
            },
        },
    )


def read_triples(path: Path, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples
