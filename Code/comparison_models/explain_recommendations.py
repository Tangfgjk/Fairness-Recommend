import json
from pathlib import Path


def parse_relation_value(value, prefix):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    text = str(value)
    if text.startswith(prefix):
        text = text[len(prefix):]
    try:
        return round(float(text), 6)
    except ValueError:
        return None


def get_exercise_kcs(q_matrix, exercise_idx):
    if exercise_idx < 0 or exercise_idx >= len(q_matrix):
        return []
    return [idx for idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def _mean(values):
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return round(sum(valid_values) / len(valid_values), 6)


def build_recommendation_explanation(
    uid,
    exercise_idx,
    score,
    q_matrix,
    uid_mlkc_dict,
    uid_pkc_dict,
    uid_exfr_dict,
):
    exercise_id = f"ex{exercise_idx}"
    kc_indices = get_exercise_kcs(q_matrix, exercise_idx)
    user_mastery = uid_mlkc_dict.get(uid, {})
    user_sequence = uid_pkc_dict.get(uid, {})
    user_forgetting = uid_exfr_dict.get(uid, {})

    knowledge_concepts = []
    for kc_idx in kc_indices:
        kc_id = f"kc{kc_idx}"
        knowledge_concepts.append(
            {
                "kc_id": kc_id,
                "mastery": parse_relation_value(user_mastery.get(kc_id), "mlkc"),
                "sequence_progress": parse_relation_value(user_sequence.get(kc_id), "pkc"),
            }
        )

    forgetting = parse_relation_value(user_forgetting.get(exercise_id), "exfr")
    avg_mastery = _mean([item["mastery"] for item in knowledge_concepts])
    avg_sequence = _mean([item["sequence_progress"] for item in knowledge_concepts])
    rounded_score = round(float(score), 6)
    kc_text = ", ".join(item["kc_id"] for item in knowledge_concepts) or "no mapped KC"

    explanation = (
        f"For {uid}, {exercise_id} is recommended because it covers {kc_text}. "
        f"The average mastery on these concepts is {avg_mastery}, the average "
        f"sequence/progress signal is {avg_sequence}, and the exercise forgetting "
        f"rate is {forgetting}. ConvE assigns this candidate a recommendation "
        f"score of {rounded_score}."
    )

    return {
        "exercise_id": exercise_id,
        "exercise_index": exercise_idx,
        "conve_score": rounded_score,
        "knowledge_concepts": knowledge_concepts,
        "average_mastery": avg_mastery,
        "average_sequence_progress": avg_sequence,
        "exercise_forgetting": forgetting,
        "explanation": explanation,
    }


def generate_explanation_cards(
    uid_ex_scores,
    q_matrix,
    uid_mlkc_dict,
    uid_pkc_dict,
    uid_exfr_dict,
    top_k=3,
    user_limit=2,
    selected_users=None,
    model_name="ConvE",
):
    score_by_uid = {uid: scores for uid, scores in uid_ex_scores}
    if selected_users:
        candidate_users = [uid for uid in selected_users if uid in score_by_uid]
    else:
        candidate_users = [uid for uid, _ in uid_ex_scores]
    if user_limit is not None:
        candidate_users = candidate_users[:user_limit]

    students = []
    for uid in candidate_users:
        scores = score_by_uid[uid]
        top_items = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        recommendations = [
            build_recommendation_explanation(
                uid=uid,
                exercise_idx=exercise_idx,
                score=score,
                q_matrix=q_matrix,
                uid_mlkc_dict=uid_mlkc_dict,
                uid_pkc_dict=uid_pkc_dict,
                uid_exfr_dict=uid_exfr_dict,
            )
            for exercise_idx, score in top_items
        ]
        students.append({"uid": uid, "recommendations": recommendations})

    return {"model": model_name, "top_k": top_k, "students": students}


def write_explanation_cards(cards, output_dir, file_prefix="recommendation_explanations"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{file_prefix}.json"
    md_path = output_path / f"{file_prefix}.md"

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(cards, fp, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# {cards.get('model', 'Model')} Recommendation Explanation Cards\n\n")
        fp.write(f"Top-K: {cards.get('top_k')}\n\n")
        for student in cards.get("students", []):
            fp.write(f"## Student {student['uid']}\n\n")
            for rank, recommendation in enumerate(student.get("recommendations", []), 1):
                fp.write(f"### Rank {rank}: {recommendation['exercise_id']}\n\n")
                fp.write(f"- ConvE score: {recommendation['conve_score']}\n")
                fp.write(f"- Exercise forgetting: {recommendation['exercise_forgetting']}\n")
                fp.write(f"- Average mastery: {recommendation['average_mastery']}\n")
                fp.write(
                    f"- Average sequence/progress: {recommendation['average_sequence_progress']}\n"
                )
                fp.write("- Knowledge concepts:\n")
                for concept in recommendation.get("knowledge_concepts", []):
                    fp.write(
                        f"  - {concept['kc_id']}: mastery={concept['mastery']}, "
                        f"sequence/progress={concept['sequence_progress']}\n"
                    )
                fp.write(f"- Explanation: {recommendation['explanation']}\n\n")

    return json_path, md_path
