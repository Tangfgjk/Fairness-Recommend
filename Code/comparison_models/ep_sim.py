import numpy as np


def _exercise_concepts(q_matrix, exercise_idx):
    if exercise_idx < 0 or exercise_idx >= len(q_matrix):
        return []
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def simulate_mastery_after_recommendations(mastery, q_matrix, recommended_exercises, learning_gain=0.1):
    updated = [float(value) for value in mastery]
    for exercise_idx in recommended_exercises:
        for kc_idx in _exercise_concepts(q_matrix, exercise_idx):
            updated[kc_idx] = updated[kc_idx] + learning_gain * (1.0 - updated[kc_idx])
            updated[kc_idx] = min(1.0, max(0.0, updated[kc_idx]))
    return updated


def calculate_user_ep_sim(mastery, updated_mastery, target_concepts, e_sup=1.0):
    if not target_concepts:
        return None
    e_start = _mean([mastery[kc_idx] for kc_idx in target_concepts])
    e_end = _mean([updated_mastery[kc_idx] for kc_idx in target_concepts])
    if e_start is None or e_end is None:
        return None
    denominator = e_sup - e_start
    if denominator <= 1e-12:
        return 0.0
    return float((e_end - e_start) / denominator)


def calculate_ep_sim(uid_ex_scores, q_matrix, mastery, top_k=10, learning_gain=0.1, e_sup=1.0):
    per_user = []
    for uid, scores in uid_ex_scores:
        uid_idx = int(uid[3:]) if str(uid).startswith("uid") else int(uid)
        if uid_idx < 0 or uid_idx >= len(mastery):
            continue
        top_exercises = [
            exercise_idx
            for exercise_idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        ]
        target_concepts = sorted(
            {
                kc_idx
                for exercise_idx in top_exercises
                for kc_idx in _exercise_concepts(q_matrix, exercise_idx)
            }
        )
        start_mastery = [float(value) for value in mastery[uid_idx]]
        updated_mastery = simulate_mastery_after_recommendations(
            start_mastery,
            q_matrix,
            top_exercises,
            learning_gain=learning_gain,
        )
        ep_value = calculate_user_ep_sim(start_mastery, updated_mastery, target_concepts, e_sup=e_sup)
        if ep_value is not None:
            per_user.append(
                {
                    "uid": uid,
                    "top_exercises": top_exercises,
                    "target_concepts": target_concepts,
                    "ep_sim": round(ep_value, 6),
                }
            )

    values = [item["ep_sim"] for item in per_user]
    return {
        "mean": round(float(np.mean(values)), 6) if values else 0.0,
        "std": round(float(np.std(values)), 6) if values else 0.0,
        "per_user": per_user,
    }
