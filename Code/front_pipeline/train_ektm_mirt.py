from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import data_fin_root, exercise_texts, front_dir, load_raw_dataset, read_json, write_json, write_matrix_json


RELATION_TEXT_TYPES = ("rec", "mlkc", "exfr", "other")
RELATION_TEXT_TEMPLATES = {
    "mlkc": (
        "A cognitive mastery relation from a knowledge concept to a learner, "
        "where the relation strength denotes the learner's mastery level of the concept."
    ),
    "exfr": (
        "A cognitive forgetting relation from an exercise to a learner, "
        "where the relation strength denotes the learner's forgetting degree for the exercise."
    ),
    "rec": (
        "A recommendation relation from a learner to an exercise, "
        "where the exercise is recommended according to the learner's cognitive mastery "
        "and forgetting states."
    ),
    "other": "An unspecified educational relation between two graph entities.",
}


def legacy_components() -> dict[str, Any]:
    """Load the local EKTM implementation without invoking its old data loaders."""
    project = Path(__file__).resolve().parents[1]
    legacy_dir = project / "v11_pipeline"
    legacy_common_spec = importlib.util.spec_from_file_location("_legacy_v11_common", legacy_dir / "common.py")
    if legacy_common_spec is None or legacy_common_spec.loader is None:
        raise ImportError("Cannot load local V11 EKTM common module")
    legacy_common = importlib.util.module_from_spec(legacy_common_spec)
    legacy_common_spec.loader.exec_module(legacy_common)
    previous_common = sys.modules.get("common")
    inserted_legacy_dir = str(legacy_dir) not in sys.path
    if inserted_legacy_dir:
        sys.path.insert(0, str(legacy_dir))
    sys.modules["common"] = legacy_common
    try:
        spec = importlib.util.spec_from_file_location("_legacy_v11_ektm", legacy_dir / "train_ektm_mirt_v11.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load local V11 EKTM implementation")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_common is not None:
            sys.modules["common"] = previous_common
        else:
            sys.modules.pop("common", None)
        if inserted_legacy_dir:
            sys.path.remove(str(legacy_dir))
    return {
        "LearnerSequenceDataset": module.LearnerSequenceDataset,
        "V11EKTMmirt": module.V11EKTMmirt,
        "build_vocab_and_tokens": module.build_vocab_and_tokens,
        "collate_sequences": module.collate_sequences,
        "run_epoch": module.run_epoch,
        "export_mastery": module.export_mastery,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EKTM_mirt-style front model from raw cohorts and export mastery/text embeddings.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-emb-size", type=int, default=128)
    parser.add_argument("--text-hidden-size", type=int, default=100)
    parser.add_argument("--knowledge-emb-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--mirt-proj-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mastery-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(dataset, allowed_uids: set[int]) -> list[int]:
    return [index for index, sample in enumerate(dataset.samples) if int(sample["user_id"]) in allowed_uids]


def make_model(components, token_matrix, concept_token_matrix, raw, a_param, b_param, vocab_size, args):
    return components["V11EKTMmirt"](
        token_matrix=token_matrix,
        concept_token_matrix=concept_token_matrix,
        q_matrix=raw.q_matrix,
        a_param=a_param,
        b_param=b_param,
        vocab_size=vocab_size,
        text_emb_size=args.text_emb_size,
        text_hidden_size=args.text_hidden_size,
        knowledge_emb_size=args.knowledge_emb_size,
        hidden_size=args.hidden_size,
        mirt_proj_size=args.mirt_proj_size,
        dropout=args.dropout,
    )


def export_relation_topics(model, relation_token_matrix: np.ndarray) -> np.ndarray:
    """Encode fixed relation templates with EKTM_mirt's shared topic encoder."""
    device = next(model.parameters()).device
    with torch.no_grad():
        tokens = torch.as_tensor(relation_token_matrix, dtype=torch.long, device=device)
        return model.encode_text_tokens(tokens).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.text_hidden_size % 2 != 0:
        raise ValueError("--text-hidden-size must be even for the bidirectional GRU text encoder")
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    front = front_dir(args.dataset, root)
    protocol = read_json(front / "protocol.json")
    mirt_dir = front / "mirt"
    a_path, b_path = mirt_dir / "a_param_q_constrained.npy", mirt_dir / "b_param.npy"
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Run front_pipeline/train_q_mirt.py before EKTM_mirt training")
    a_param = np.load(a_path).astype(np.float32)
    b_param = np.load(b_path).astype(np.float32)
    if a_param.shape != (raw.exercise_count, raw.concept_count) or b_param.shape != (raw.exercise_count, 1):
        raise ValueError(f"Unexpected Q-MIRT parameter shapes: a={a_param.shape}, b={b_param.shape}")

    components = legacy_components()
    # Knowledge concepts are ID-only in V-Fin7. The shared text encoder is
    # trained on exercise text and retains explicit tokens for relation templates.
    relation_texts = [RELATION_TEXT_TEMPLATES[name] for name in RELATION_TEXT_TYPES]
    vocabulary, all_tokens = components["build_vocab_and_tokens"](
        exercise_texts(raw) + relation_texts, args.min_token_count, args.max_text_len
    )
    token_matrix = all_tokens[: raw.exercise_count]
    relation_token_matrix = all_tokens[raw.exercise_count :]
    # The legacy EKTM class checks this shape but does not consume concept text
    # in its response/mastery forward pass. Keep an all-padding placeholder.
    concept_token_matrix = np.zeros((raw.concept_count, args.max_text_len), dtype=np.int64)
    full_dataset = components["LearnerSequenceDataset"](
        raw.interactions.rename(columns={"uid": "user_id", "question": "item_id", "response": "score"}),
        raw.q_matrix,
        raw.student_count,
        args.max_seq_len,
    )
    inner_train = set(int(uid) for uid in protocol["inner_train_uids"])
    inner_valid = set(int(uid) for uid in protocol["inner_valid_uids"])
    outer_train = set(int(uid) for uid in protocol["outer_train_uids"])
    train_indices = subset_indices(full_dataset, inner_train)
    valid_indices = subset_indices(full_dataset, inner_valid)
    if not train_indices or not valid_indices:
        raise ValueError("EKTM inner train and validation cohorts both need at least one non-empty sequence")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    collate = components["collate_sequences"]
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(Subset(full_dataset, valid_indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = make_model(components, token_matrix, concept_token_matrix, raw, a_param, b_param, len(vocabulary), args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    run_epoch = components["run_epoch"]
    history: list[dict[str, float]] = []
    best_epoch, best_valid = 1, float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, optimizer, args.mastery_loss_weight)
        valid_stats = run_epoch(model, valid_loader, device, None, args.mastery_loss_weight)
        row = {"epoch": float(epoch), "train_loss": train_stats["loss"], "valid_loss": valid_stats["loss"]}
        history.append(row)
        if valid_stats["loss"] < best_valid:
            best_epoch, best_valid = epoch, valid_stats["loss"]
        print(f"[Epoch {epoch}] train_loss={train_stats['loss']:.6f} valid_loss={valid_stats['loss']:.6f} best_epoch={best_epoch}")

    # Retrain only on outer-train learners for the selected epoch count, then freeze for every export.
    set_seed(args.seed)
    final_indices = subset_indices(full_dataset, outer_train)
    final_loader = DataLoader(Subset(full_dataset, final_indices), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    final_model = make_model(components, token_matrix, concept_token_matrix, raw, a_param, b_param, len(vocabulary), args).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for _ in range(best_epoch):
        run_epoch(final_model, final_loader, device, final_optimizer, args.mastery_loss_weight)

    output = front / "ektm_mirt"
    export_dir = output / "exports"
    text_dir = front / "semantic_kg_features" / "text_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    topics = final_model.export_exercise_topics()
    relation_topics = export_relation_topics(final_model, relation_token_matrix)
    mastery = components["export_mastery"](final_model, full_dataset, raw.student_count, raw.concept_count, device)
    np.save(export_dir / "exercise_topic_v.npy", topics)
    np.save(export_dir / "know_output.npy", mastery)
    np.save(text_dir / "exercise_text_embeddings.npy", topics)
    np.save(text_dir / "exercise_text_tokens.npy", token_matrix)
    np.save(text_dir / "relation_text_tokens.npy", relation_token_matrix)
    np.save(text_dir / "relation_text_embeddings.npy", relation_topics)
    concept_embedding_path = text_dir / "concept_text_embeddings.npy"
    if concept_embedding_path.exists():
        concept_embedding_path.unlink()
    torch.save(
        {
            "word_embedding": final_model.word_embedding.state_dict(),
            "text_encoder": final_model.text_encoder.state_dict(),
        },
        text_dir / "shared_text_encoder.pt",
    )
    write_json(text_dir / "shared_text_vocab.json", {"token_to_id": vocabulary})
    write_json(
        text_dir / "shared_text_encoder_config.json",
        {
            "vocab_size": len(vocabulary),
            "text_emb_size": args.text_emb_size,
            "text_hidden_size": args.text_hidden_size,
            "padding_idx": 0,
            "relation_types": list(RELATION_TEXT_TYPES),
        },
    )
    write_json(
        text_dir / "relation_semantics.json",
        {
            "relation_types": list(RELATION_TEXT_TYPES),
            "templates": RELATION_TEXT_TEMPLATES,
            "source": "V-Fin7 shared EKTM_mirt Bi-GRU relation templates",
        },
    )
    write_matrix_json(export_dir / "stu2know_mastery.json", mastery)
    write_matrix_json(front / "stu2know_mastery.json", mastery)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "vocab": vocabulary,
            "config": vars(args),
            "selected_epoch": best_epoch,
            "selected_valid_loss": best_valid,
            "outer_train_uids": sorted(outer_train),
        },
        output / "best.pt",
    )
    write_json(
        text_dir / "text_embedding_manifest.json",
        {
            "dataset": raw.name,
            "model": "EKTM_mirt_shared_trainable_BiGRU_topic_relation_encoder",
            "source": "front_pipeline/train_ektm_mirt.py",
            "embedding_dim": int(topics.shape[1]),
            "exercise_entity_ids": [f"ex{index}" for index in range(raw.exercise_count)],
            "files": {
                "exercise_text_embeddings": "exercise_text_embeddings.npy",
                "exercise_text_tokens": "exercise_text_tokens.npy",
                "relation_text_tokens": "relation_text_tokens.npy",
                "relation_text_embeddings": "relation_text_embeddings.npy",
                "shared_text_encoder": "shared_text_encoder.pt",
                "shared_text_encoder_config": "shared_text_encoder_config.json",
                "shared_text_vocab": "shared_text_vocab.json",
                "relation_semantics": "relation_semantics.json",
            },
            "notes": "Exercise and relation text share the EKTM_mirt Bi-GRU. Knowledge concepts are ID-only. "
            "SemanticConvE loads the saved encoder and dynamically fine-tunes exercise/relation text representations.",
        },
    )
    write_json(
        output / "ektm_manifest.json",
        {
            "dataset": raw.name,
            "model": "EKTM_mirt_style",
            "mirt_source": mirt_dir / "mirt_manifest.json",
            "training_users": len(outer_train),
            "test_users_frozen_inference_only": len(raw.test_uids),
            "selected_epoch": best_epoch,
            "selected_valid_loss": best_valid,
            "mastery_source": "final-step know_output from the frozen EKTM_mirt-style model",
            "mastery_shape": list(mastery.shape),
            "topic_shape": list(topics.shape),
            "relation_text_shape": list(relation_topics.shape),
            "history": history,
        },
    )
    print(f"saved raw EKTM_mirt exports: {output}")


if __name__ == "__main__":
    main()
