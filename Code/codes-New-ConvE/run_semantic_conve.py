"""Train 2CKG4ER on KG4ER graph files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairness_context import build_fairness_context, context_stats, write_fairness_context_summary
from fairness_regularizer import FairnessRegularizer, FairnessRegularizerConfig
from feature_loader import entity_kind, load_semantic_feature_bundle, read_triples, relation_kind
from semantic_conve_model import MODEL_DISPLAY_NAME, TwoCKG4ER, VALID_MODEL_ABLATIONS, normalize_ablation_mode
from semantic_experiment_utils import MODEL_VERSION
from training_objectives import RelationWeightConfig, build_relation_weight_tensor, weighted_bce_loss


Triple = Tuple[int, int, int]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def resolve_device(cuda: str) -> torch.device:
    if cuda == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda.lower() in {"true", "1", "yes", "cuda"}:
        return torch.device("cuda")
    return torch.device("cpu")


def training_triple_files() -> List[str]:
    return ["triples.txt"]


def build_positive_tails_by_hr(triples: Sequence[Triple]) -> Dict[Tuple[int, int], set[int]]:
    result: Dict[Tuple[int, int], set[int]] = {}
    for h, r, t in triples:
        result.setdefault((h, r), set()).add(t)
    return result


class TripleDataset(Dataset):
    def __init__(
        self,
        triples: Sequence[Triple],
        id2relation: Dict[int, str],
        exercise_ids: Sequence[int],
        positive_tails_by_hr: Dict[Tuple[int, int], set[int]],
        negative_ratio: int,
        seed: int,
    ) -> None:
        self.triples = list(triples)
        self.id2relation = id2relation
        self.exercise_ids = list(exercise_ids)
        self.positive_tails_by_hr = positive_tails_by_hr
        self.negative_ratio = max(0, int(negative_ratio))
        self.seed = int(seed)
        self.sample_index: List[Tuple[int, int]] = []
        for triple_idx, (_, relation_id, _) in enumerate(self.triples):
            self.sample_index.append((triple_idx, 0))
            if self.should_sample_negative(relation_id):
                for negative_idx in range(1, self.negative_ratio + 1):
                    self.sample_index.append((triple_idx, negative_idx))

    def should_sample_negative(self, relation_id: int) -> bool:
        return relation_kind(self.id2relation.get(relation_id, "")) == "rec"

    def __len__(self) -> int:
        return len(self.sample_index)

    def sample_negative_tail(self, h: int, r: int, sample_idx: int) -> int:
        blocked = self.positive_tails_by_hr.get((h, r), set())
        rng = random.Random(self.seed + sample_idx * 1000003)
        for _ in range(100):
            candidate = self.exercise_ids[rng.randrange(len(self.exercise_ids))]
            if candidate not in blocked:
                return candidate
        eligible = [candidate for candidate in self.exercise_ids if candidate not in blocked]
        if not eligible:
            raise ValueError("No filtered recommendation negative tail available.")
        return eligible[rng.randrange(len(eligible))]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        triple_idx, offset = self.sample_index[idx]
        h, r, t = self.triples[triple_idx]
        label = 1.0
        if offset:
            t = self.sample_negative_tail(h, r, idx)
            label = 0.0
        return (
            torch.tensor(h, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def save_checkpoint(
    model: TwoCKG4ER,
    optimizer: torch.optim.Optimizer,
    path: Path,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "ablation": args.ablation,
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": jsonable(vars(args)),
        },
        path,
    )


def load_checkpoint(
    model: TwoCKG4ER,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
    expected_ablation: str,
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_version = checkpoint.get("model_version")
    if checkpoint_version != MODEL_VERSION:
        raise RuntimeError(
            f"Checkpoint model_version is incompatible: {checkpoint_version!r} != {MODEL_VERSION!r}. "
            "Please start a new run-id because relation encoding has changed."
        )
    checkpoint_ablation = normalize_ablation_mode(checkpoint.get("ablation", MODEL_DISPLAY_NAME))
    if checkpoint_ablation != expected_ablation:
        raise RuntimeError(
            f"Checkpoint ablation is incompatible: {checkpoint_ablation!r} != {expected_ablation!r}. "
            "Please use a separate run-id for each ablation."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch") or 0), checkpoint.get("loss")


def export_embeddings(model: TwoCKG4ER, save_path: Path, nentity: int, nrelation: int, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        entity_ids = torch.arange(nentity, device=device)
        relation_ids = torch.arange(nrelation, device=device)
        entity_emb = model.entity_embedding(entity_ids).detach().cpu().numpy()
        relation_emb = model.relation_embedding(relation_ids).detach().cpu().numpy()
    np.save(save_path / "final_entity_embedding.npy", entity_emb)
    np.save(save_path / "final_relation_embedding.npy", relation_emb)


def setup_logger(save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(save_path / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2CKG4ER semantic and pedagogical feature-aware ConvE.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, required=True)
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", required=True)
    parser.add_argument("--save-path", "--save_path", dest="save_path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", "--embedding_dim", dest="embedding_dim", type=int, default=200)
    parser.add_argument("--embedding-shape1", "--embedding_shape1", dest="embedding_shape1", type=int, default=20)
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=9728)
    parser.add_argument("--input-drop", "--input_drop", dest="input_drop", type=float, default=0.2)
    parser.add_argument("--hidden-drop", "--hidden_drop", dest="hidden_drop", type=float, default=0.2)
    parser.add_argument("--feat-drop", "--feat_drop", dest="feat_drop", type=float, default=0.3)
    parser.add_argument("--negative-ratio", "--negative_ratio", dest="negative_ratio", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cuda", default="auto")
    parser.add_argument("--ablation", choices=sorted(VALID_MODEL_ABLATIONS), default=MODEL_DISPLAY_NAME)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-bias", "--use_bias", dest="use_bias", action="store_true", default=True)
    parser.add_argument("--relation-loss-weights", choices=["none", "balanced", "custom"], default="none")
    parser.add_argument("--rec-loss-weight", type=float, default=1.0)
    parser.add_argument("--mlkc-loss-weight", type=float, default=1.0)
    parser.add_argument("--exfr-loss-weight", type=float, default=1.0)
    parser.add_argument("--exkc-loss-weight", type=float, default=1.0)
    parser.add_argument("--other-loss-weight", type=float, default=1.0)
    parser.add_argument("--fairness-loss", choices=["none", "expected_exposure"], default="none")
    parser.add_argument("--fairness-alpha-item", type=float, default=0.0)
    parser.add_argument("--fairness-alpha-kc", type=float, default=0.0)
    parser.add_argument("--fairness-loss-scale", type=float, default=1.0)
    parser.add_argument("--fairness-temperature", type=float, default=1.0)
    parser.add_argument("--fairness-candidate-size", type=int, default=150)
    parser.add_argument("--fairness-candidate-mode", choices=["random", "popular", "top_score", "mixed", "top_score_user", "mixed_user"], default="random")
    parser.add_argument("--fairness-target-gamma", type=float, default=0.5)
    parser.add_argument("--fairness-exposure-proxy", choices=["softmax", "sigmoid_topk"], default="softmax")
    parser.add_argument("--fairness-surrogate-k", type=int, default=10)
    parser.add_argument("--fairness-distance", choices=["mse", "l1", "kl_target_model", "js"], default="mse")
    parser.add_argument("--fairness-top-score-ratio", type=float, default=0.5)
    parser.add_argument("--fairness-popular-ratio", type=float, default=0.25)
    parser.add_argument("--fairness-popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--fairness-popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--fairness-head-ratio", type=float, default=0.2)
    parser.add_argument("--fairness-long-tail-ratio", type=float, default=0.8)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Debug smoke-test limit; 0 means no limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ablation = normalize_ablation_mode(args.ablation)
    args.model_version = MODEL_VERSION
    set_seed(args.seed, deterministic=args.deterministic)
    setup_logger(args.save_path)
    device = resolve_device(str(args.cuda))
    logging.info("dataset=%s", args.dataset_name)
    logging.info("device=%s", device)

    bundle = load_semantic_feature_bundle(args.data_path, args.feature_dir, device=device)
    train_files = training_triple_files()
    all_positive: List[Triple] = []
    file_triples: List[tuple[str, List[Triple]]] = []
    for file_name in train_files:
        triples = read_triples(args.data_path / file_name, bundle.entity2id, bundle.relation2id)
        file_triples.append((file_name, triples))
        all_positive.extend(triples)
    positive_tails_by_hr = build_positive_tails_by_hr(all_positive)
    relation_weight_tensor, relation_weight_metadata = build_relation_weight_tensor(
        bundle.id2relation,
        all_positive,
        RelationWeightConfig(
            mode=args.relation_loss_weights,
            rec=args.rec_loss_weight,
            mlkc=args.mlkc_loss_weight,
            exfr=args.exfr_loss_weight,
            exkc=args.exkc_loss_weight,
            other=args.other_loss_weight,
        ),
        device=device,
    )

    dataloaders = []
    total_samples = 0
    for file_name, triples in file_triples:
        dataset = TripleDataset(
            triples=triples,
            id2relation=bundle.id2relation,
            exercise_ids=bundle.exercise_entity_ids.detach().cpu().tolist(),
            positive_tails_by_hr=positive_tails_by_hr,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        dataloaders.append((file_name, DataLoader(dataset, batch_size=args.bs, shuffle=True), len(triples), len(dataset)))
        total_samples += len(dataset)

    model = TwoCKG4ER.from_feature_bundle(
        bundle,
        embedding_dim=args.embedding_dim,
        embedding_shape1=args.embedding_shape1,
        hidden_size=args.hidden_size,
        input_drop=args.input_drop,
        hidden_drop=args.hidden_drop,
        feat_drop=args.feat_drop,
        use_bias=args.use_bias,
        ablation_mode=args.ablation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    fairness_config = FairnessRegularizerConfig(
        loss_type=args.fairness_loss,
        alpha_item=args.fairness_alpha_item,
        alpha_kc=args.fairness_alpha_kc,
        loss_scale=args.fairness_loss_scale,
        temperature=args.fairness_temperature,
        candidate_size=args.fairness_candidate_size,
        candidate_mode=args.fairness_candidate_mode,
        target_gamma=args.fairness_target_gamma,
        exposure_proxy=args.fairness_exposure_proxy,
        surrogate_top_k=args.fairness_surrogate_k,
        distance=args.fairness_distance,
        top_score_ratio=args.fairness_top_score_ratio,
        popular_ratio=args.fairness_popular_ratio,
    )
    fairness_context = None
    fairness_regularizer = None
    fairness_context_summary = None
    if fairness_config.enabled:
        fairness_context = build_fairness_context(
            args.data_path,
            head_ratio=args.fairness_head_ratio,
            long_tail_ratio=args.fairness_long_tail_ratio,
            target_gamma=args.fairness_target_gamma,
            popularity_source=args.fairness_popularity_source,
            popularity_aggregation=args.fairness_popularity_aggregation,
        )
        fairness_context_summary = context_stats(fairness_context)
        write_fairness_context_summary(fairness_context, args.save_path / "fairness_context.json")
        fairness_regularizer = FairnessRegularizer(
            fairness_config,
            rec_relation_id=bundle.relation2id["rec"],
            exercise_entity_ids=bundle.exercise_entity_ids,
            tensors=fairness_context.to_tensors(device=device),
        )

    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    last_path = args.save_path / "last.pt"
    best_path = args.save_path / "best.pt"
    if args.resume and last_path.exists():
        start_epoch, last_loss = load_checkpoint(model, optimizer, last_path, device, args.ablation)
        if last_loss is not None:
            best_loss = float(last_loss)
            best_epoch = start_epoch
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device)
            if best_ckpt.get("loss") is not None:
                best_loss = float(best_ckpt["loss"])
                best_epoch = int(best_ckpt.get("epoch") or best_epoch)
        logging.info("resumed from %s start_epoch=%s best_loss=%s", last_path, start_epoch, best_loss)

    logging.info("train_files=%s", train_files)
    logging.info("positive_triples=%s total_samples=%s negative_ratio=%s", len(all_positive), total_samples, args.negative_ratio)
    logging.info("relation_loss_weights=%s", relation_weight_metadata)
    if fairness_regularizer is not None:
        logging.info("fairness_regularizer=%s", fairness_regularizer.metadata())
    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_fair_loss = 0.0
        batch_count = 0
        stop_epoch = False
        for file_name, dataloader, positive_count, sample_count in dataloaders:
            logging.info("epoch %s training %s positives=%s samples=%s", epoch + 1, file_name, positive_count, sample_count)
            for h, r, t, label in tqdm(dataloader):
                h = h.to(device)
                r = r.to(device)
                t = t.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                pred = model.score_triples(h, r, t)
                bce_loss = weighted_bce_loss(pred, label, r, relation_weight_tensor)
                fair_loss = torch.zeros((), dtype=bce_loss.dtype, device=device)
                if fairness_regularizer is not None:
                    rec_mask = r == bundle.relation2id["rec"]
                    fair_loss, _fair_details = fairness_regularizer(model, h[rec_mask])
                loss = bce_loss + fair_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                epoch_bce_loss += float(bce_loss.item())
                epoch_fair_loss += float(fair_loss.item())
                batch_count += 1
                if args.max_train_batches > 0 and batch_count >= args.max_train_batches:
                    logging.info("max_train_batches=%s reached; ending epoch early for smoke testing", args.max_train_batches)
                    stop_epoch = True
                    break
            if stop_epoch:
                break
        avg_loss = epoch_loss / max(1, batch_count)
        avg_bce_loss = epoch_bce_loss / max(1, batch_count)
        avg_fair_loss = epoch_fair_loss / max(1, batch_count)
        save_checkpoint(model, optimizer, last_path, epoch + 1, avg_loss, args)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, best_path, epoch + 1, avg_loss, args)
        logging.info(
            "epoch=%s/%s loss=%.8f bce_loss=%.8f fair_loss=%.8f best_epoch=%s best_loss=%.8f",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_bce_loss,
            avg_fair_loss,
            best_epoch,
            best_loss,
        )

    training_seconds = time.perf_counter() - training_start
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    export_embeddings(model, args.save_path, bundle.nentity, bundle.nrelation, device)
    write_json(
        args.save_path / "gate_values.json",
        {
            "model": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION,
            "dataset": args.dataset_name,
            "seed": args.seed,
            "ablation": args.ablation,
            "checkpoint": "best.pt" if best_path.exists() else "last.pt",
            "entity_fusion": "V10 attention fusion: ID embedding + residual self-attention over active side-feature tokens",
            "semantic_quality": "enabled as a fixed mask for text feature tokens",
            "state_encoder": "disabled in V10-attn; learner entities use ID embedding plus theta_mirt_norm and cluster_id feature tokens",
            "gate_values": model.gate_values(),
        },
    )
    write_json(
        args.save_path / "state_encoder_config.json",
        {
            "model": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION,
            "dataset": args.dataset_name,
            "seed": args.seed,
            "ablation": args.ablation,
            "enabled": False,
            "state_feature_slices": bundle.state_feature_slices,
            "state_feature_dim": int(bundle.state_features.shape[1]),
            "learner_representation": (
                "V10-attn does not replace learner ID with state-aware head embedding; "
                "learner side features are fused as residual tokens"
            ),
            "train_triple_files": train_files,
        },
    )
    write_json(
        args.save_path / "relation_encoder_config.json",
        {
            "model": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION,
            "dataset": args.dataset_name,
            "seed": args.seed,
            "ablation": args.ablation,
            "relation_encoding": "shared relation-text Bi-GRU embedding plus a projected continuous strength feature",
            "relation_id_embedding_used": args.ablation == "2CKG4ER_relation_id",
            "relation_type_to_id": bundle.metadata.get("relation_type_to_id"),
            "relation_loss_weights": relation_weight_metadata,
        },
    )
    if fairness_regularizer is not None:
        write_json(
            args.save_path / "fairness_regularizer_config.json",
            {
                "regularizer": fairness_regularizer.metadata(),
                "context": fairness_context_summary,
            },
        )
    write_json(
        args.save_path / "metrics.json",
        {
            "model": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION,
            "ablation": args.ablation,
            "dataset": args.dataset_name,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_loss": best_loss if best_loss < float("inf") else None,
            "training_seconds": round(training_seconds, 6),
            "positive_triples": len(all_positive),
            "total_training_samples": total_samples,
            "negative_ratio": args.negative_ratio,
            "negative_sampling": "rec-only type-constrained filtered tail replacement",
            "train_triple_files": train_files,
            "feature_dir": bundle.metadata["feature_dir"],
            "text_embedding_model": bundle.metadata["text_manifest"].get("model"),
            "embedding_dim": args.embedding_dim,
            "relation_encoding": "relation ID embedding with attention-fused relation type and continuous relation strength residual",
            "entity_fusion": "entity ID embedding with attention-fused semantic and pedagogical residual tokens",
            "state_feature_dim": int(bundle.state_features.shape[1]),
            "state_feature_slices": bundle.state_feature_slices,
            "numeric_feature_slices": bundle.numeric_feature_slices,
            "semantic_quality": bundle.metadata.get("semantic_quality"),
            "relation_loss_weights": relation_weight_metadata,
            "fairness_regularizer": fairness_regularizer.metadata() if fairness_regularizer is not None else {"enabled": False},
            "fairness_context": fairness_context_summary,
        },
    )
    write_json(args.save_path / "config.json", jsonable(vars(args)))
    logging.info("finished training in %.3fs", training_seconds)


if __name__ == "__main__":
    main()
