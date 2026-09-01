from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from common import (
    data_fin_root,
    exercise_texts,
    front_dir,
    load_raw_dataset,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SemanticConvE-compatible metadata from raw-front model outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument("--cluster-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def index_entries(payload: dict, key: str, index_name: str) -> dict[int, dict]:
    return {
        int(item[index_name]): item
        for item in payload.get(key, [])
        if isinstance(item, dict) and index_name in item
    }


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    front = front_dir(args.dataset, root)
    feature_dir = front / "semantic_kg_features"
    entity_dir = feature_dir / "entity_features"
    irt_dir = feature_dir / "irt_features"
    marker = feature_dir / "feature_manifest.json"
    if marker.exists() and not args.force:
        raise FileExistsError(f"{marker} exists. Use --force to rebuild semantic feature metadata.")
    mirt_dir = front / "mirt"
    theta = np.load(mirt_dir / "theta_param.npy").astype(np.float32)
    a = np.load(mirt_dir / "a_param_q_constrained.npy").astype(np.float32)
    b = np.load(mirt_dir / "b_param.npy").astype(np.float32).reshape(-1)
    theta_norm = np.load(mirt_dir / "learner_theta_norm.npy").astype(np.float32)
    difficulty_norm = np.load(mirt_dir / "exercise_difficulty_norm.npy").astype(np.float32)
    discrimination_norm = np.load(mirt_dir / "exercise_discrimination_norm.npy").astype(np.float32)
    mastery = np.asarray(read_json(front / "stu2know_mastery.json"), dtype=np.float32)
    if theta.shape != (raw.student_count, raw.concept_count):
        raise ValueError(f"theta shape mismatch: {theta.shape}")
    if a.shape != (raw.exercise_count, raw.concept_count) or b.shape[0] != raw.exercise_count:
        raise ValueError(f"item MIRT shape mismatch: a={a.shape}, b={b.shape}")
    if mastery.shape != (raw.student_count, raw.concept_count):
        raise ValueError(f"mastery shape mismatch: {mastery.shape}")

    protocol = read_json(front / "protocol.json")
    train_uids = np.asarray(protocol["outer_train_uids"], dtype=np.int64)
    cluster_count = min(int(args.cluster_count), len(train_uids))
    if cluster_count < 1:
        raise ValueError("No outer-train learner is available for clustering")
    kmeans = KMeans(n_clusters=cluster_count, random_state=args.seed, n_init=10)
    kmeans.fit(theta[train_uids])
    clusters = kmeans.predict(theta).astype(int)

    train_interactions = raw.interactions[raw.interactions["uid"].isin(train_uids)]
    exercise_stats = train_interactions.groupby("question")["response"].agg(["mean", "count"]).reindex(range(raw.exercise_count), fill_value=0.0)
    exercise_entries = index_entries(raw.exercise_metadata, "exercises", "exercise_index")
    concept_entries = index_entries(raw.concept_metadata, "concepts", "concept_index")
    exercise_text_list = exercise_texts(raw)
    learner_entries: dict[str, dict] = {}
    for uid in range(raw.student_count):
        personal = raw.interactions[raw.interactions["uid"] == uid]
        learner_entries[f"uid{uid}"] = {
            "entity_id": f"uid{uid}",
            "entity_type": "learner",
            "theta_mirt": theta[uid].tolist(),
            "theta_norm": float(theta_norm[uid]),
            "cluster_id": int(clusters[uid]),
            "overall_mastery_irt": float(theta_norm[uid]),
            "overall_mastery_kt_mean": float(np.mean(mastery[uid])),
            "correct_rate": float(personal["response"].mean()) if len(personal) else 0.0,
            "history_length": int(len(personal)),
            "concept_mastery_mean": float(np.mean(mastery[uid])),
            "concept_mastery_std": float(np.std(mastery[uid])),
            "theta_source": "q_constrained_mirt; outer-test theta adapted with frozen item parameters",
        }

    exercise_semantics: dict[str, dict] = {}
    exercise_irt: dict[str, dict] = {}
    for ex in range(raw.exercise_count):
        item = exercise_entries.get(ex, {})
        text = exercise_text_list[ex]
        exercise_semantics[f"ex{ex}"] = {
            "entity_id": f"ex{ex}",
            "entity_type": "exercise",
            "question_text": text,
            "text_for_embedding": text,
            "text_source": str(item.get("text_source", "raw_structured_text")),
            "semantic_quality": 1.0 if text and not text.startswith("exercise ") else 0.3,
        }
        correct_rate = float(exercise_stats.loc[ex, "mean"])
        interaction_count = int(exercise_stats.loc[ex, "count"])
        exercise_irt[f"ex{ex}"] = {
            "entity_id": f"ex{ex}",
            "entity_type": "exercise",
            "difficulty": float(b[ex]),
            "difficulty_norm": float(difficulty_norm[ex]),
            "discrimination": float(np.linalg.norm(a[ex])),
            "discrimination_norm": float(discrimination_norm[ex]),
            "latent_discrimination_vector": a[ex].tolist(),
            "correct_rate": correct_rate,
            "error_rate": float(1.0 - correct_rate),
            "interaction_count": interaction_count,
            "parameter_source": "QConstrainedMIRT trained on outer-train learners only",
        }

    concept_semantics: dict[str, dict] = {}
    for kc in range(raw.concept_count):
        item = concept_entries.get(kc, {})
        name = str(item.get("name", f"knowledge concept {kc}"))
        definition = str(item.get("definition", ""))
        concept_semantics[f"kc{kc}"] = {
            "entity_id": f"kc{kc}",
            "entity_type": "knowledge_concept",
            "name": name,
            "definition": definition,
            "definition_source": str(item.get("definition_source", "raw_metadata")),
            "semantic_quality": 0.8 if definition else 0.5,
        }

    entity_dir.mkdir(parents=True, exist_ok=True)
    irt_dir.mkdir(parents=True, exist_ok=True)
    write_json(entity_dir / "learner_pedagogy.json", {"dataset": raw.name, "learners": learner_entries})
    write_json(entity_dir / "exercise_semantics.json", {"dataset": raw.name, "exercises": exercise_semantics})
    write_json(entity_dir / "concept_semantics.json", {"dataset": raw.name, "concepts": concept_semantics})
    write_json(irt_dir / "exercise_irt_features.json", {"dataset": raw.name, "exercises": exercise_irt})
    write_json(
        entity_dir / "entity_feature_index.json",
        {
            "dataset": raw.name,
            "entities": {"learners": raw.student_count, "exercises": raw.exercise_count, "concepts": raw.concept_count},
            "text_embedding_source": "EKTM_mirt-style trainable Bi-GRU",
            "pedagogy_source": "Q-constrained MIRT",
            "cluster_source": "KMeans fit on outer-train theta only",
        },
    )
    source_text_dir = front / "semantic_kg_features" / "text_embeddings"
    if not (source_text_dir / "text_embedding_manifest.json").exists():
        raise FileNotFoundError("Run front_pipeline/train_ektm_mirt.py before building semantic metadata")
    write_json(
        marker,
        {
            "dataset": raw.name,
            "version": "raw_front_semantic_features_v1",
            "text_embeddings": source_text_dir / "text_embedding_manifest.json",
            "mirt": mirt_dir / "mirt_manifest.json",
            "cluster_count": cluster_count,
            "exercise_statistics_source": "outer-train interactions only",
        },
    )
    print(f"saved raw semantic feature metadata: {feature_dir}")


if __name__ == "__main__":
    main()
