"""Export recommendation scores from a trained 2CKG4ER model."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from feature_loader import load_semantic_feature_bundle
from popularity_debias_context import build_popularity_debias_context
from popularity_debias_head import PopularityDebiasConfig
from semantic_conve_model import MODEL_DISPLAY_NAME, TwoCKG4ER, VALID_MODEL_ABLATIONS, normalize_ablation_mode


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_device(cuda: str) -> torch.device:
    if cuda == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda.lower() in {"true", "1", "yes", "cuda"}:
        return torch.device("cuda")
    return torch.device("cpu")


def read_q_count(data_path: Path) -> int:
    with (data_path / "Q.txt").open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def users_from_triples(data_path: Path) -> List[str]:
    users = set()
    with (data_path / "test_triples.txt").open("r", encoding="utf-8") as fp:
        for line in fp:
            head, relation, tail = line.strip().split("\t")
            if tail.startswith("uid") and (
                relation.startswith("mlkc") or relation.startswith("exfr")
            ):
                users.add(tail)
    return sorted(users, key=lambda uid: int(uid[3:]) if uid.startswith("uid") and uid[3:].isdigit() else uid)


test_users_from_triples = users_from_triples
test_users_from_triples.__test__ = False


def exercise_index(entity_name: str) -> int | None:
    if not entity_name.startswith("ex") or not entity_name[2:].isdigit():
        return None
    return int(entity_name[2:])


def read_exfr_relation_names(data_path: Path, users: List[str], q_count: int) -> Dict[str, List[str | None]]:
    relation_names: Dict[str, List[str | None]] = {uid: [None] * q_count for uid in users}
    with (data_path / "test_triples.txt").open("r", encoding="utf-8") as fp:
        for line in fp:
            head, relation, tail = line.strip().split("\t")
            if not relation.startswith("exfr") or tail not in relation_names:
                continue
            ex_idx = exercise_index(head)
            if ex_idx is None or ex_idx >= q_count:
                continue
            relation_names[tail][ex_idx] = relation
    return relation_names


def resolve_relation_id(relation2id: Dict[str, int], relation_name: str) -> int:
    if relation_name in relation2id:
        return relation2id[relation_name]
    if relation_name.startswith("exfr"):
        value = float(relation_name[4:])
        candidates = [
            f"exfr{value:.2f}",
            f"exfr{round(value, 2)}",
            f"exfr{value:.2f}".rstrip("0").rstrip("."),
        ]
        for candidate in candidates:
            if candidate in relation2id:
                return relation2id[candidate]
    raise KeyError(f"Cannot resolve relation id for {relation_name!r}")


def exfr_relation_tensors(
    relation_names: Dict[str, List[str | None]],
    users: List[str],
    q_count: int,
    relation2id: Dict[str, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fallback_relation_id = next(iter(relation2id.values()))
    relation_ids = torch.full((len(users), q_count), fallback_relation_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(users), q_count), dtype=torch.float32, device=device)
    for user_idx, uid in enumerate(users):
        for ex_idx, relation_name in enumerate(relation_names.get(uid, [])):
            if relation_name is None:
                continue
            relation_ids[user_idx, ex_idx] = resolve_relation_id(relation2id, relation_name)
            mask[user_idx, ex_idx] = 1.0
    return relation_ids, mask


def score_forgetting_branch(
    model: TwoCKG4ER,
    rec_relation_id: int,
    exercise_tail_ids: torch.Tensor,
    exfr_relation_ids: torch.Tensor,
    exfr_mask: torch.Tensor,
    exercise_batch_size: int,
) -> torch.Tensor:
    batch_size, exercise_count = exfr_relation_ids.shape
    scores = torch.zeros((batch_size, exercise_count), dtype=torch.float32, device=exercise_tail_ids.device)
    rec_relation = torch.empty(0, dtype=torch.long, device=exercise_tail_ids.device)
    for start in range(0, exercise_count, exercise_batch_size):
        end = min(start + exercise_batch_size, exercise_count)
        chunk_tail_ids = exercise_tail_ids[start:end]
        flat_tail_ids = chunk_tail_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        flat_exfr_ids = exfr_relation_ids[:, start:end].reshape(-1)
        if rec_relation.numel() != flat_tail_ids.numel():
            rec_relation = torch.full_like(flat_tail_ids, rec_relation_id)
        exercise_emb = model.entity_embedding(flat_tail_ids)
        exfr_emb = model.relation_embedding(flat_exfr_ids)
        forgetting_head = exercise_emb + exfr_emb
        chunk_scores = model.score_tail_pairs_from_head_embeddings(forgetting_head, rec_relation, flat_tail_ids)
        scores[:, start:end] = chunk_scores.view(batch_size, end - start) * exfr_mask[:, start:end]
    return scores


def score_exercise_candidates(
    model: TwoCKG4ER,
    h: torch.Tensor,
    r: torch.Tensor,
    exercise_tail_ids: torch.Tensor,
    type_aware: bool,
    score_mode: str = "raw",
) -> torch.Tensor:
    def call_score_tails(tail_ids: torch.Tensor | None = None) -> torch.Tensor:
        try:
            if tail_ids is None:
                return model.score_tails(h, r, score_mode=score_mode)
            return model.score_tails(h, r, tail_ids, score_mode=score_mode)
        except TypeError as exc:
            if "score_mode" not in str(exc):
                raise
            if tail_ids is None:
                return model.score_tails(h, r)
            return model.score_tails(h, r, tail_ids)

    if type_aware:
        return call_score_tails(exercise_tail_ids)
    all_scores = call_score_tails()
    return all_scores.index_select(1, exercise_tail_ids)


def score_exercise_logits(
    model: TwoCKG4ER,
    h: torch.Tensor,
    r: torch.Tensor,
    exercise_tail_ids: torch.Tensor,
    type_aware: bool,
    score_mode: str = "raw",
) -> torch.Tensor:
    def call_score_tails_logits(tail_ids: torch.Tensor | None = None) -> torch.Tensor:
        try:
            if tail_ids is None:
                return model.score_tails_logits(h, r, score_mode=score_mode)
            return model.score_tails_logits(h, r, tail_ids, score_mode=score_mode)
        except TypeError as exc:
            if "score_mode" not in str(exc):
                raise
            if tail_ids is None:
                return model.score_tails_logits(h, r)
            return model.score_tails_logits(h, r, tail_ids)

    if type_aware:
        return call_score_tails_logits(exercise_tail_ids)
    all_scores = call_score_tails_logits()
    return all_scores.index_select(1, exercise_tail_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 2CKG4ER uid-exercise recommendation scores.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, required=True)
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", required=True)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--ablation", choices=sorted(VALID_MODEL_ABLATIONS), default=MODEL_DISPLAY_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=200)
    parser.add_argument("--embedding-shape1", "--embedding_shape1", dest="embedding_shape1", type=int, default=20)
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=9728)
    parser.add_argument("--input-drop", "--input_drop", dest="input_drop", type=float, default=0.2)
    parser.add_argument("--hidden-drop", "--hidden_drop", dest="hidden_drop", type=float, default=0.2)
    parser.add_argument("--feat-drop", "--feat_drop", dest="feat_drop", type=float, default=0.3)
    parser.add_argument("--score-mode", choices=["raw", "debiased"], default="raw")
    parser.add_argument("--tail-bias-mode", default=None)
    parser.add_argument("--popularity-debias", default=None)
    parser.add_argument("--save-logit-components", action="store_true")
    parser.add_argument("--save-raw-and-debiased", action="store_true")
    parser.add_argument("--forgetting-score-weight", type=float, default=0.0)
    parser.add_argument("--forgetting-exercise-batch-size", type=int, default=256)
    parser.add_argument("--max-test-users", type=int, default=0, help="Debug smoke-test limit; 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ablation = normalize_ablation_mode(args.ablation)
    if args.forgetting_score_weight < 0:
        raise ValueError("--forgetting-score-weight must be non-negative")
    if args.forgetting_exercise_batch_size <= 0:
        raise ValueError("--forgetting-exercise-batch-size must be positive")
    device = resolve_device(str(args.cuda))
    bundle = load_semantic_feature_bundle(args.data_path, args.feature_dir, device=device)
    checkpoint_path = args.save_path / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {}) or {}
    tail_bias_mode = args.tail_bias_mode or checkpoint_config.get("tail_bias_mode", "legacy")
    popularity_debias_mode = args.popularity_debias or checkpoint_config.get("popularity_debias", "none")
    popularity_debias_tensors = None
    if popularity_debias_mode != "none":
        kppd_context = build_popularity_debias_context(
            args.data_path,
            popularity_source=checkpoint_config.get("fairness_popularity_source", "train_interactions"),
            popularity_aggregation=checkpoint_config.get("fairness_popularity_aggregation", "unique_users"),
        )
        popularity_debias_tensors = kppd_context.to_tensors(
            entity2id=bundle.entity2id,
            exercise_entity_ids=bundle.exercise_entity_ids,
            device=device,
        )
    model = TwoCKG4ER.from_feature_bundle(
        bundle,
        embedding_dim=args.embedding_dim,
        embedding_shape1=args.embedding_shape1,
        hidden_size=args.hidden_size,
        input_drop=args.input_drop,
        hidden_drop=args.hidden_drop,
        feat_drop=args.feat_drop,
        tail_bias_mode=tail_bias_mode,
        ablation_mode=args.ablation,
        popularity_debias_config=PopularityDebiasConfig(
            mode=popularity_debias_mode,
            beta_global=float(checkpoint_config.get("beta_global_pop", 1.0)),
            beta_personal=float(checkpoint_config.get("beta_personal_pop", 1.0)),
            beta_need=float(checkpoint_config.get("beta_need", 1.0)),
            lambda_global=float(checkpoint_config.get("lambda_global_pop", 1.0)),
            lambda_personal=float(checkpoint_config.get("lambda_personal_pop", 1.0)),
            personal_hidden_dim=int(checkpoint_config.get("kppd_personal_hidden_dim", 16)),
        ),
        popularity_debias_tensors=popularity_debias_tensors,
    ).to(device)

    checkpoint_ablation = normalize_ablation_mode(checkpoint.get("ablation", MODEL_DISPLAY_NAME))
    if checkpoint_ablation != args.ablation:
        raise RuntimeError(
            f"Checkpoint ablation is incompatible: {checkpoint_ablation!r} != {args.ablation!r}. "
            "Use the matching --ablation value for testing."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    q_count = read_q_count(args.data_path)
    exercise_ids = [bundle.entity2id[f"ex{i}"] for i in range(q_count)]
    exercise_tail_ids = torch.tensor(exercise_ids, dtype=torch.long, device=device)
    rec_id = bundle.relation2id["rec"]
    rec_relation = torch.tensor([rec_id], dtype=torch.long, device=device)
    users = users_from_triples(args.data_path)
    if args.max_test_users > 0:
        users = users[: args.max_test_users]
    type_aware_scoring = True
    use_forgetting_score = args.forgetting_score_weight > 0 and args.ablation != "2CKG4ER_no_forgetting"
    exfr_ids = None
    exfr_mask = None
    exfr_coverage = 0.0
    if use_forgetting_score and users:
        exfr_relation_names = read_exfr_relation_names(args.data_path, users, q_count)
        exfr_ids, exfr_mask = exfr_relation_tensors(exfr_relation_names, users, q_count, bundle.relation2id, device)
        exfr_coverage = float(exfr_mask.mean().detach().cpu().item()) if exfr_mask.numel() else 0.0
        use_forgetting_score = bool(exfr_mask.sum().item() > 0)
    effective_forgetting_score_weight = float(args.forgetting_score_weight) if use_forgetting_score else 0.0

    uid_ex_scores = []
    component_logits: Dict[str, List[Tuple[str, List[float]]]] = {
        "raw": [],
        "debiased": [],
        "global_pop": [],
        "personal_pop": [],
        "need": [],
    }
    inference_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(users), args.batch_size):
            batch_users = users[start : start + args.batch_size]
            h = torch.tensor([bundle.entity2id[uid] for uid in batch_users], dtype=torch.long, device=device)
            r = rec_relation.repeat(len(batch_users))
            logits = score_exercise_logits(model, h, r, exercise_tail_ids, type_aware=type_aware_scoring, score_mode=args.score_mode)
            if use_forgetting_score and exfr_ids is not None and exfr_mask is not None:
                forgetting_scores = score_forgetting_branch(
                    model,
                    rec_id,
                    exercise_tail_ids,
                    exfr_ids[start : start + len(batch_users)],
                    exfr_mask[start : start + len(batch_users)],
                    args.forgetting_exercise_batch_size,
                )
                logits = logits + effective_forgetting_score_weight * torch.logit(forgetting_scores.clamp(min=1e-6, max=1 - 1e-6))
            scores = torch.sigmoid(logits)
            if args.save_logit_components or args.save_raw_and_debiased:
                components = model.score_tail_matrix_components(h, r, exercise_tail_ids.unsqueeze(0).expand(len(batch_users), -1))
                for name in component_logits:
                    values = components[name if name in components else "raw"].detach().cpu().tolist()
                    component_logits[name].extend((uid, row) for uid, row in zip(batch_users, values))
            scores = scores.detach().cpu().tolist()
            uid_ex_scores.extend((uid, row) for uid, row in zip(batch_users, scores))
    inference_seconds = time.perf_counter() - inference_start

    default_name = "2CKG4ER_uid_ex_scores.pkl" if args.score_mode == "raw" else "2CKG4ER_uid_ex_scores_debiased.pkl"
    output_file = args.output_file or (args.save_path / default_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as fp:
        pickle.dump(uid_ex_scores, fp)
    if args.save_logit_components or args.save_raw_and_debiased:
        for name, rows in component_logits.items():
            with (args.save_path / f"2CKG4ER_uid_ex_logits_{name}.pkl").open("wb") as fp:
                pickle.dump(rows, fp)
        if args.save_raw_and_debiased:
            for name in ("raw", "debiased"):
                rows = [(uid, torch.sigmoid(torch.tensor(logits)).tolist()) for uid, logits in component_logits[name]]
                with (args.save_path / f"2CKG4ER_uid_ex_scores_{name}.pkl").open("wb") as fp:
                    pickle.dump(rows, fp)

    write_json(
        args.save_path / "2ckg4er_inference.json",
        {
            "model": MODEL_DISPLAY_NAME,
            "ablation": args.ablation,
            "dataset": args.dataset_name,
            "checkpoint": str(checkpoint_path),
            "output_file": str(output_file),
            "score_mode": args.score_mode,
            "tail_bias_mode": tail_bias_mode,
            "popularity_debias": popularity_debias_mode,
            "saved_logit_components": bool(args.save_logit_components or args.save_raw_and_debiased),
            "test_user_count": len(users),
            "exercise_count": q_count,
            "inference_seconds": round(inference_seconds, 6),
            "forgetting_score_weight": float(args.forgetting_score_weight),
            "effective_forgetting_score_weight": effective_forgetting_score_weight,
            "forgetting_exfr_coverage": round(exfr_coverage, 6),
            "type_aware_scoring": type_aware_scoring,
            "scoring": (
                "type-aware: rec scores are computed only over exercise entities ex0..exN; "
                "the optional forgetting branch adds weight * score(ex+exfr, rec, ex)"
            ),
        },
    )
    print(json.dumps({"output_file": str(output_file), "users": len(users), "exercises": q_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
