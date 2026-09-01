import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from experiment_utils import update_timing, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Score ER recommendations with trained KGE embeddings.")
    parser.add_argument("--model", required=True, choices=["TransE", "TransE-adv", "RotatE", "DistMult", "ComplEx"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-file", type=Path, default=None)
    parser.add_argument("--gamma", type=float, default=12.0)
    return parser.parse_args()


def load_q_matrix(path):
    matrix = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                matrix.append([int(value) for value in line.split(",")])
    return matrix


def load_dict(path):
    rows = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            idx, name = line.strip().split("\t")
            rows[name] = int(idx)
    return rows


def load_test_relations(path):
    uid_mlkc = {}
    uid_exfr = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            head, relation, uid = line.strip().split("\t")
            if relation.startswith("mlkc"):
                uid_mlkc.setdefault(uid, {})[head] = relation
            elif relation.startswith("exfr"):
                uid_exfr.setdefault(uid, {})[head] = relation
    return uid_mlkc, uid_exfr


def transe(head, relation, tail, gamma):
    return gamma - np.linalg.norm((head + relation) - tail, ord=2)


def rotate(head, relation, tail, gamma):
    re_head, im_head = np.split(head, 2)
    re_tail, im_tail = np.split(tail, 2)
    phase = relation / ((gamma + 2.0) / (len(re_head) * np.pi))
    re_relation = np.cos(phase)
    im_relation = np.sin(phase)
    re_score = re_head * re_relation - im_head * im_relation - re_tail
    im_score = re_head * im_relation + im_head * re_relation - im_tail
    return gamma - np.linalg.norm(np.stack([re_score, im_score], axis=0), axis=0).sum()


def distmult(head, relation, tail, gamma):
    return float(np.sum(head * relation * tail))


def complex_score(head, relation, tail, gamma):
    re_head, im_head = np.split(head, 2)
    re_relation, im_relation = np.split(relation, 2)
    re_tail, im_tail = np.split(tail, 2)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    return float(np.sum(re_score * re_tail + im_score * im_tail))


def score_fn(model_name):
    return {
        "TransE": transe,
        "TransE-adv": transe,
        "RotatE": rotate,
        "DistMult": distmult,
        "ComplEx": complex_score,
    }[model_name]


def score_triple(scorer, entity_embedding, relation_embedding, entity2id, relation2id, head, relation, tail, gamma):
    return scorer(
        entity_embedding[entity2id[head]],
        relation_embedding[relation2id[relation]],
        entity_embedding[entity2id[tail]],
        gamma,
    )


def score_exercise_candidate(
    scorer,
    entity_embedding,
    relation_embedding,
    entity2id,
    relation2id,
    exercise,
    cognitive_items,
    exfr_items,
    gamma,
    model_name=None,
    mlkc_count=None,
):
    if model_name in {"TransE", "TransE-adv"}:
        return score_transe_er_candidate(
            entity_embedding,
            relation_embedding,
            entity2id,
            relation2id,
            exercise,
            cognitive_items,
            exfr_items,
            gamma,
            mlkc_count,
        )
    raise RuntimeError(
        "Non-TransE models must be scored with score_native_er_candidates so that "
        "mlkc/exfr first infer a learner-state prototype rather than form "
        "schema-invalid triples ending at an exercise."
    )


def score_transe_er_candidate(
    entity_embedding,
    relation_embedding,
    entity2id,
    relation2id,
    exercise,
    cognitive_items,
    exfr_items,
    gamma,
    mlkc_count=None,
):
    rec_relation = relation_embedding[relation2id["rec"]]
    exercise_embedding = entity_embedding[entity2id[exercise]]
    cognitive_scores = [
        transe(
            entity_embedding[entity2id[kc]] + relation_embedding[relation2id[relation]],
            rec_relation,
            exercise_embedding,
            gamma,
        )
        for kc, relation in cognitive_items
    ]
    denominator = mlkc_count or max(1, len(cognitive_scores))
    cognitive_score = sum(cognitive_scores) / max(1, denominator)

    forget_score = 0.0
    forget_relation = exfr_items.get(exercise)
    if forget_relation:
        forget_score = transe(
            exercise_embedding + relation_embedding[relation2id[forget_relation]],
            rec_relation,
            exercise_embedding,
            gamma,
        )
    return float(cognitive_score + forget_score)


def tail_estimate(model_name, head, relation, gamma):
    """Estimate the tail state implied by a model-native (head, relation) pair."""
    if model_name == "RotatE":
        re_head, im_head = np.split(head, 2)
        phase = relation / ((gamma + 2.0) / (len(re_head) * np.pi))
        re_relation = np.cos(phase)
        im_relation = np.sin(phase)
        return np.concatenate(
            [
                re_head * re_relation - im_head * im_relation,
                re_head * im_relation + im_head * re_relation,
            ]
        )
    if model_name == "DistMult":
        return head * relation
    if model_name == "ComplEx":
        re_head, im_head = np.split(head, 2)
        re_relation, im_relation = np.split(relation, 2)
        return np.concatenate(
            [
                re_head * re_relation - im_head * im_relation,
                re_head * im_relation + im_head * re_relation,
            ]
        )
    raise ValueError(f"Unsupported native state model: {model_name}")


def native_tail_scores(model_name, learner_states, rec_relation, exercise_embeddings, gamma):
    """Score candidate exercises with each model's original triple scoring rule."""
    if model_name == "RotatE":
        re_head, im_head = np.split(learner_states, 2, axis=1)
        re_tail, im_tail = np.split(exercise_embeddings, 2, axis=1)
        phase = rec_relation / ((gamma + 2.0) / (re_head.shape[1] * np.pi))
        re_relation = np.cos(phase)[None, :]
        im_relation = np.sin(phase)[None, :]
        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail
        return gamma - np.linalg.norm(np.stack([re_score, im_score], axis=0), axis=0).sum(axis=1)
    if model_name == "DistMult":
        return np.sum(learner_states * rec_relation[None, :] * exercise_embeddings, axis=1)
    if model_name == "ComplEx":
        re_head, im_head = np.split(learner_states, 2, axis=1)
        re_relation, im_relation = np.split(rec_relation, 2)
        re_tail, im_tail = np.split(exercise_embeddings, 2, axis=1)
        re_score = re_head * re_relation[None, :] - im_head * im_relation[None, :]
        im_score = re_head * im_relation[None, :] + im_head * re_relation[None, :]
        return np.sum(re_score * re_tail + im_score * im_tail, axis=1)
    raise ValueError(f"Unsupported native scorer: {model_name}")


def score_native_er_candidates(
    model_name,
    entity_embedding,
    relation_embedding,
    entity2id,
    relation2id,
    exercises,
    cognitive_items,
    exfr_items,
    gamma,
):
    """Use graph-valid cognitive triples to infer a learner state before rec scoring.

    The original graph defines (kc, mlkc, uid) and (ex, exfr, uid), not
    (kc, mlkc, ex) or (ex, exfr, ex). For non-TransE KGE models, each
    cognitive pair therefore produces a model-native estimate of the tail
    learner state.  The exercise-specific forgetting estimate is added to the
    shared KC-derived learner state before scoring (state, rec, exercise).
    """
    state_estimates = []
    for head, relation in cognitive_items:
        if head in entity2id and relation in relation2id:
            state_estimates.append(
                tail_estimate(
                    model_name,
                    entity_embedding[entity2id[head]],
                    relation_embedding[relation2id[relation]],
                    gamma,
                )
            )

    embedding_dim = entity_embedding.shape[1]
    base_state = (
        np.mean(np.asarray(state_estimates, dtype=float), axis=0)
        if state_estimates
        else np.zeros(embedding_dim, dtype=float)
    )
    exercise_embeddings = np.asarray([entity_embedding[entity2id[exercise]] for exercise in exercises], dtype=float)
    learner_states = np.repeat(base_state[None, :], len(exercises), axis=0)

    for index, exercise in enumerate(exercises):
        relation = exfr_items.get(exercise)
        if relation and relation in relation2id:
            learner_states[index] += tail_estimate(
                model_name,
                exercise_embeddings[index],
                relation_embedding[relation2id[relation]],
                gamma,
            )

    return native_tail_scores(
        model_name,
        learner_states,
        relation_embedding[relation2id["rec"]],
        exercise_embeddings,
        gamma,
    ).astype(float).tolist()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    q_matrix = load_q_matrix(args.data_dir / "Q.txt")
    entity2id = load_dict(args.data_dir / "entities.dict")
    relation2id = load_dict(args.data_dir / "relations.dict")
    entity_embedding = np.load(args.embedding_dir / "entity_embedding.npy")
    relation_embedding = np.load(args.embedding_dir / "relation_embedding.npy")
    uid_mlkc, uid_exfr = load_test_relations(args.data_dir / "test_triples.txt")

    scorer = score_fn(args.model)
    uid_ex_scores = []
    user_ids = set(uid_mlkc) | set(uid_exfr)
    for uid in sorted(user_ids, key=lambda value: int(value[3:]) if value.startswith("uid") else value):
        scores = []
        mlkc_items = list(uid_mlkc.get(uid, {}).items())
        exfr_items = uid_exfr.get(uid, {})
        cognitive_items = mlkc_items
        exercises = [f"ex{exercise_idx}" for exercise_idx in range(len(q_matrix))]
        if args.model in {"TransE", "TransE-adv"}:
            for exercise in exercises:
                scores.append(
                    score_exercise_candidate(
                        scorer,
                        entity_embedding,
                        relation_embedding,
                        entity2id,
                        relation2id,
                        exercise,
                        cognitive_items,
                        exfr_items,
                        args.gamma,
                        model_name=args.model,
                        mlkc_count=len(mlkc_items),
                    )
                )
        else:
            scores = score_native_er_candidates(
                args.model,
                entity_embedding,
                relation_embedding,
                entity2id,
                relation2id,
                exercises,
                cognitive_items,
                exfr_items,
                args.gamma,
            )
        uid_ex_scores.append((uid, scores))

    output_pkl = args.output_dir / f"{args.model}_uid_ex_scores.pkl"
    output_json = args.output_dir / f"{args.model}_uid_ex_scores.json"
    with output_pkl.open("wb") as fp:
        pickle.dump(uid_ex_scores, fp)
    write_json([{"uid": uid, "scores": scores} for uid, scores in uid_ex_scores], output_json)
    write_json(
        {
            "model": args.model,
            "transE_policy": "legacy_kg4ex_compatible" if args.model in {"TransE", "TransE-adv"} else None,
            "non_transE_policy": (
                "model_native_tail_state_then_rec" if args.model not in {"TransE", "TransE-adv"} else None
            ),
            "candidate_type": "exercise_only",
            "cognitive_relations": ["mlkc", "exfr"],
        },
        args.output_dir / "scoring_protocol.json",
    )
    if args.timing_file:
        update_timing(args.timing_file, "inference_without_cache", time.perf_counter() - start)
    print(json.dumps({"scores_pkl": str(output_pkl), "scores_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
