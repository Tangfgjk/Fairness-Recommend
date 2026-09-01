import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


SUPPORTED_BASELINES = ["EB-CF", "SB-CF", "CBF"]


def load_json_matrix(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_q_matrix(path):
    q_matrix = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                q_matrix.append([int(value) for value in line.strip().split(",")])
    return q_matrix


def parse_int_list(text):
    if text is None:
        return []
    return [int(value) for value in str(text).split(",") if value and int(value) >= 0]


def load_sequence_interactions(sequence_file):
    interactions = {}
    if not sequence_file:
        return interactions
    path = Path(sequence_file)
    if not path.exists():
        return interactions

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            uid = f"uid{row['uid']}" if not str(row["uid"]).startswith("uid") else str(row["uid"])
            if "questions" in row and "responses" in row:
                questions = parse_int_list(row.get("questions"))
                responses = parse_int_list(row.get("responses"))
            else:
                questions = parse_int_list(row.get("question"))
                responses = parse_int_list(row.get("response"))
            pairs = []
            for question, response in zip(questions, responses):
                if response in (0, 1):
                    pairs.append((question, response))
            interactions.setdefault(uid, []).extend(pairs)
    return interactions


def load_cf_protocol(raw_dir):
    """Load a split-safe collaborative-filtering library from canonical raw data.

    Training learners form the neighbour/item-similarity library.  Test learners
    are retained only as recommendation queries, so they never become candidate
    neighbours or contribute to item-item statistics.
    """
    raw_dir = Path(raw_dir)
    split_path = raw_dir / "student_split.csv"
    interactions_path = raw_dir / "interactions_all.csv"
    if not split_path.exists() or not interactions_path.exists():
        raise FileNotFoundError(
            "Split-safe CF requires raw/student_split.csv and raw/interactions_all.csv "
            f"under {raw_dir}"
        )

    split_by_uid = {}
    with split_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            split_by_uid[f"uid{int(row['uid'])}"] = str(row["split"]).strip().lower()

    train_interactions = {}
    query_interactions = {}
    with interactions_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            uid = f"uid{int(row['uid'])}"
            split = split_by_uid.get(uid, str(row.get("split", "")).strip().lower())
            question = int(row["question"])
            response = int(row["response"])
            if response not in (0, 1):
                continue
            if split == "train":
                train_interactions.setdefault(uid, []).append((question, response))
            elif split == "test":
                query_interactions.setdefault(uid, []).append((question, response))

    test_user_ids = sorted(
        [uid for uid, split in split_by_uid.items() if split == "test"],
        key=_uid_sort_key,
    )
    for uid in test_user_ids:
        query_interactions.setdefault(uid, [])
    if not train_interactions or not test_user_ids:
        raise ValueError(
            f"CF protocol is empty: train_users={len(train_interactions)}, test_users={len(test_user_ids)}"
        )
    return train_interactions, query_interactions, test_user_ids


def _cosine(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _uid_sort_key(uid):
    return int(uid[3:]) if str(uid).startswith("uid") and str(uid)[3:].isdigit() else str(uid)


def _row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _mean(values, default=0.0):
    values = [value for value in values if value is not None]
    if not values:
        return default
    return float(sum(values) / len(values))


def normalize_scores(scores):
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def exercise_concepts(q_matrix, exercise_idx):
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def content_based_scores(q_matrix, mastery, sequence=None, forgetting=None, sequence_weight=0.2, forgetting_weight=0.2):
    sequence = sequence or [0.0] * len(mastery)
    forgetting = forgetting or [0.0] * len(q_matrix)
    scores = []
    for exercise_idx in range(len(q_matrix)):
        concepts = exercise_concepts(q_matrix, exercise_idx)
        if not concepts:
            scores.append(0.0)
            continue
        weak_score = _mean([1.0 - float(mastery[kc]) for kc in concepts])
        progress_score = _mean([float(sequence[kc]) for kc in concepts])
        score = weak_score + sequence_weight * progress_score + forgetting_weight * float(forgetting[exercise_idx])
        scores.append(score)
    return normalize_scores(scores)


def kcp_er_scores(q_matrix, mastery, forgetting=None, target_mastery=0.8, forgetting_weight=0.2):
    forgetting = forgetting or [0.0] * len(q_matrix)
    scores = []
    for exercise_idx in range(len(q_matrix)):
        concepts = exercise_concepts(q_matrix, exercise_idx)
        if not concepts:
            scores.append(0.0)
            continue
        concept_mastery = _mean([float(mastery[kc]) for kc in concepts])
        target_fit = 1.0 - abs(float(target_mastery) - concept_mastery)
        score = target_fit + forgetting_weight * float(forgetting[exercise_idx])
        scores.append(score)
    return normalize_scores(scores)


def _interaction_matrices(interactions, exercise_count):
    """Return correctness, signed-rating, and observation matrices by learner."""
    user_ids = sorted(interactions, key=_uid_sort_key)
    correctness_sum = np.zeros((len(user_ids), exercise_count), dtype=float)
    observation_count = np.zeros((len(user_ids), exercise_count), dtype=float)
    for row_index, uid in enumerate(user_ids):
        for exercise_idx, response in interactions[uid]:
            if 0 <= exercise_idx < exercise_count:
                correctness_sum[row_index, exercise_idx] += float(response)
                observation_count[row_index, exercise_idx] += 1.0
    correctness = np.divide(
        correctness_sum,
        observation_count,
        out=np.zeros_like(correctness_sum),
        where=observation_count > 0,
    )
    signed = np.where(observation_count > 0, correctness * 2.0 - 1.0, 0.0)
    return user_ids, correctness, signed, observation_count


def _profile_vector(pairs, exercise_count):
    response_sum = np.zeros(exercise_count, dtype=float)
    count = np.zeros(exercise_count, dtype=float)
    for exercise_idx, response in pairs:
        if 0 <= exercise_idx < exercise_count:
            response_sum[exercise_idx] += float(response)
            count[exercise_idx] += 1.0
    correctness = np.divide(response_sum, count, out=np.zeros_like(response_sum), where=count > 0)
    signed = np.where(count > 0, correctness * 2.0 - 1.0, 0.0)
    return correctness, signed, count


def _exercise_popularity(correctness, observation_count):
    total_correct = correctness * observation_count
    return np.divide(
        total_correct.sum(axis=0),
        observation_count.sum(axis=0),
        out=np.zeros(correctness.shape[1], dtype=float),
        where=observation_count.sum(axis=0) > 0,
    )


def _top_k_similarity_matrix(vectors, neighbor_count):
    count = len(vectors)
    if count == 0:
        return np.zeros((0, 0), dtype=float)
    if count == 1:
        return np.zeros((1, 1), dtype=float)
    neighbor_count = min(max(1, neighbor_count), count - 1)
    knn = NearestNeighbors(n_neighbors=neighbor_count + 1, metric="cosine", algorithm="brute")
    knn.fit(vectors)
    distances, indices = knn.kneighbors(vectors)
    similarity = np.zeros((count, count), dtype=float)
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, column_index in zip(row_distances, row_indices):
            if row_index != column_index:
                similarity[row_index, column_index] = max(0.0, 1.0 - float(distance))
    return similarity


def exercise_based_cf_scores(q_matrix, interactions, user_ids=None, query_interactions=None, neighbor_count=20):
    """ItemKNN with train-only item statistics and test-user query histories."""
    exercise_count = len(q_matrix)
    query_interactions = query_interactions if query_interactions is not None else interactions
    user_ids = user_ids or sorted(query_interactions, key=_uid_sort_key)
    _, correctness, signed, observation_count = _interaction_matrices(interactions, exercise_count)
    popularity = _exercise_popularity(correctness, observation_count)
    item_similarity = _top_k_similarity_matrix(signed.T, neighbor_count)
    uid_ex_scores = []
    for uid in user_ids:
        _, profile, observed = _profile_vector(query_interactions.get(uid, []), exercise_count)
        if np.any(observed > 0):
            scores = np.matmul(profile, item_similarity)
        else:
            scores = popularity
        uid_ex_scores.append((uid, normalize_scores(scores.tolist())))
    return uid_ex_scores


def student_based_cf_scores(
    q_matrix,
    mastery,
    interactions,
    user_ids=None,
    neighbor_count=20,
    query_interactions=None,
):
    """UserKNN with train learners as the only neighbours for test queries."""
    exercise_count = len(q_matrix)
    _ = mastery  # Kept in the signature for backwards compatibility with existing callers.
    query_interactions = query_interactions if query_interactions is not None else interactions
    user_ids = user_ids or sorted(query_interactions, key=_uid_sort_key)
    _, correctness, signed, observation_count = _interaction_matrices(interactions, exercise_count)
    popularity = _exercise_popularity(correctness, observation_count)
    train_count = signed.shape[0]
    knn = NearestNeighbors(n_neighbors=min(max(1, neighbor_count), train_count), metric="cosine", algorithm="brute")
    knn.fit(signed)
    uid_ex_scores = []

    for uid in user_ids:
        _, query_signed, observed = _profile_vector(query_interactions.get(uid, []), exercise_count)
        if not np.any(observed > 0):
            uid_ex_scores.append((uid, normalize_scores(popularity.tolist())))
            continue
        distances, indices = knn.kneighbors(query_signed.reshape(1, -1))
        weights = np.maximum(0.0, 1.0 - distances[0])
        selected_correctness = correctness[indices[0]]
        selected_observed = observation_count[indices[0]]
        numerator = np.sum(weights[:, None] * selected_correctness * selected_observed, axis=0)
        denominator = np.sum(weights[:, None] * selected_observed, axis=0)
        scores = np.divide(numerator, denominator, out=popularity.copy(), where=denominator > 0)
        uid_ex_scores.append((uid, normalize_scores(scores.tolist())))
    return uid_ex_scores


def build_all_baseline_scores(
    q_matrix,
    mastery,
    sequence,
    forgetting,
    interactions=None,
    query_interactions=None,
    methods=None,
    user_ids=None,
    neighbor_count=20,
):
    methods = methods or SUPPORTED_BASELINES
    interactions = interactions or {}
    user_ids = user_ids or [f"uid{i}" for i in range(len(mastery))]
    all_scores = {}

    for method in methods:
        if method == "EB-CF":
            all_scores[method] = exercise_based_cf_scores(
                q_matrix,
                interactions,
                user_ids=user_ids,
                query_interactions=query_interactions,
                neighbor_count=neighbor_count,
            )
        elif method == "SB-CF":
            all_scores[method] = student_based_cf_scores(
                q_matrix,
                mastery,
                interactions,
                user_ids=user_ids,
                query_interactions=query_interactions,
                neighbor_count=neighbor_count,
            )
        elif method == "CBF":
            all_scores[method] = [
                (
                    uid,
                    content_based_scores(
                        q_matrix,
                        mastery[idx],
                        sequence[idx] if sequence else None,
                        forgetting[idx] if forgetting else None,
                    ),
                )
                for uid in user_ids
                for idx in [int(uid[3:])]
            ]
        elif method == "KCP-ER":
            all_scores[method] = [
                (
                    uid,
                    kcp_er_scores(
                        q_matrix,
                        mastery[idx],
                        forgetting[idx] if forgetting else None,
                    ),
                )
                for uid in user_ids
                for idx in [int(uid[3:])]
            ]
        else:
            raise ValueError(f"Unsupported baseline: {method}")
    return all_scores


def save_uid_ex_scores(uid_ex_scores, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        with path.open("wb") as fp:
            pickle.dump(uid_ex_scores, fp)
    else:
        with path.open("w", encoding="utf-8") as fp:
            json.dump(uid_ex_scores, fp, ensure_ascii=False)
