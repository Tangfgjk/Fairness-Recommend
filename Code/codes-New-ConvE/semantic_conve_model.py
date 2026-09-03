"""2CKG4ER and its six formal ablation variants.

The formal primary model is ``2CKG4ER``: learners use MIRT theta,
exercises use shared-text plus MIRT features, concepts retain only KC IDs, and
relations use relation text plus relation strength. The remaining variants
either restore one ID representation or remove one cognitive graph relation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from feature_loader import ENTITY_TYPE_TO_ID, NO_CLUSTER_ID, RELATION_TYPE_TO_ID, SemanticFeatureBundle


MODEL_DISPLAY_NAME = "2CKG4ER"

ABLATION_ALIASES = {
    "feature_only": "2CKG4ER",
    "feature_only_relation_id": "2CKG4ER_relation_id",
    "feature_only_learner_id": "2CKG4ER_learner_id",
    "feature_only_exercise_id": "2CKG4ER_exercise_id",
    "feature_only_no_mastery": "2CKG4ER_no_mastery",
    "feature_only_no_forgetting": "2CKG4ER_no_forgetting",
}

VALID_MODEL_ABLATIONS = {
    "id_only",
    "2CKG4ER",
    "2CKG4ER_relation_id",
    "2CKG4ER_learner_id",
    "2CKG4ER_exercise_id",
    "2CKG4ER_no_mastery",
    "2CKG4ER_no_forgetting",
}


def normalize_ablation_mode(ablation_mode: str) -> str:
    return ABLATION_ALIASES.get(str(ablation_mode), str(ablation_mode))


class RawConcatFusion(nn.Module):
    """Fuse raw feature vectors by concatenation and an MLP.

    Unlike the older token-mask implementation, this module does not create
    zero placeholder tokens. Each caller passes exactly the features that should
    be used by the current entity/relation type and ablation.
    """

    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        # Inputs already have feature-specific scales. In particular,
        # LayerNorm(1) would erase the learner theta scalar completely.
        self.input_norm = nn.Identity()
        hidden_dim = embedding_dim * 2
        self.output = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must be shaped [batch, feature_dim]")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected feature_dim={self.input_dim}, got {features.shape[1]}")
        return self.output_norm(self.output(self.input_norm(features)))


class SharedTextBiGRU(nn.Module):
    """The EKTM_mirt text encoder reused by exercises and relation templates."""

    def __init__(self, vocab_size: int, text_emb_size: int, text_hidden_size: int, padding_idx: int = 0) -> None:
        super().__init__()
        if text_hidden_size % 2 != 0:
            raise ValueError("shared text_hidden_size must be even for Bi-GRU encoding")
        self.word_embedding = nn.Embedding(vocab_size, text_emb_size, padding_idx=padding_idx)
        self.text_encoder = nn.GRU(
            text_emb_size,
            text_hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("shared text token_ids must be shaped [batch, sequence_length]")
        embedded = self.word_embedding(token_ids)
        _outputs, hidden = self.text_encoder(embedded)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)


class TwoCKG4ER(nn.Module):
    """ConvE with type-specific raw concat-MLP feature fusion."""

    def __init__(
        self,
        nentity: int,
        nrelation: int,
        text_dim: int,
        numeric_dim: int,
        relation_type_ids: torch.Tensor,
        relation_strengths: torch.Tensor,
        entity_type_ids: torch.Tensor,
        cluster_ids: torch.Tensor,
        text_features: torch.Tensor,
        numeric_features: torch.Tensor,
        semantic_quality: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        exercise_text_token_ids: Optional[torch.Tensor] = None,
        exercise_entity_to_text_index: Optional[torch.Tensor] = None,
        relation_text_token_ids: Optional[torch.Tensor] = None,
        shared_text_encoder_state: Optional[dict[str, Any]] = None,
        shared_text_encoder_config: Optional[dict[str, Any]] = None,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        hidden_size: int = 9728,
        input_drop: float = 0.2,
        hidden_drop: float = 0.2,
        feat_drop: float = 0.3,
        use_bias: bool = True,
        freeze_text_features: bool = True,
        ablation_mode: str = "2CKG4ER",
        numeric_feature_slices: Optional[dict[str, tuple[int, int]]] = None,
        state_feature_slices: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        ablation_mode = normalize_ablation_mode(ablation_mode)
        if ablation_mode not in VALID_MODEL_ABLATIONS:
            raise ValueError(f"Unknown 2CKG4ER ablation mode: {ablation_mode}")
        if embedding_dim % embedding_shape1 != 0:
            raise ValueError("embedding_dim must be divisible by embedding_shape1")

        self.nentity = nentity
        self.nrelation = nrelation
        self.embedding_dim = embedding_dim
        self.emb_dim1 = embedding_shape1
        self.emb_dim2 = embedding_dim // embedding_shape1
        self.freeze_text_features = freeze_text_features
        self.ablation_mode = ablation_mode
        self.numeric_feature_slices = numeric_feature_slices or {"learner_irt": (0, 0), "exercise_irt": (0, 0)}
        self.state_feature_slices = state_feature_slices or {}

        self.text_dim = max(1, int(text_dim))
        self.learner_irt_width = max(1, self._slice_width("learner_irt"))
        self.exercise_irt_width = max(1, self._slice_width("exercise_irt"))
        self.relation_type_dim = 16

        self.emb_e = nn.Embedding(nentity, embedding_dim)
        self.relation_id_emb = nn.Embedding(nrelation, embedding_dim)
        self.relation_type_emb = nn.Embedding(len(RELATION_TYPE_TO_ID), self.relation_type_dim)

        self.uid_feature_only_fusion = RawConcatFusion(self.learner_irt_width, embedding_dim)
        self.exercise_feature_only_fusion = RawConcatFusion(
            self.text_dim + self.exercise_irt_width,
            embedding_dim,
        )
        self.kc_fusion = RawConcatFusion(embedding_dim, embedding_dim)
        self.strength_projector = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16))
        self.relation_text_dim = self.text_dim
        self.relation_feature_only_fusion = RawConcatFusion(self.relation_text_dim + 16, embedding_dim)
        # Fallback for front features created before shared relation text exists.
        self.legacy_relation_feature_only_fusion = RawConcatFusion(self.relation_type_dim + 1, embedding_dim)
        self.entity_norm = nn.LayerNorm(embedding_dim)
        self.relation_norm = nn.LayerNorm(embedding_dim)

        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_map_drop = nn.Dropout2d(feat_drop)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.register_parameter("b", nn.Parameter(torch.zeros(nentity)))

        self.register_buffer("text_features", text_features.detach().clone())
        self.register_buffer("numeric_features", numeric_features.detach().clone())
        if state_features is None:
            state_features = torch.zeros((nentity, 1), dtype=torch.float32, device=numeric_features.device)
        self.register_buffer("state_features", state_features.detach().clone())
        self.register_buffer("entity_type_ids", entity_type_ids.detach().clone())
        self.register_buffer("cluster_ids", cluster_ids.detach().clone())
        self.register_buffer("semantic_quality", semantic_quality.detach().clone())
        self.register_buffer("relation_type_ids", relation_type_ids.detach().clone())
        self.register_buffer("relation_strengths", relation_strengths.detach().clone())
        self.register_buffer(
            "exercise_text_token_ids",
            (exercise_text_token_ids.detach().clone() if exercise_text_token_ids is not None else torch.zeros((0, 1), dtype=torch.long)),
        )
        self.register_buffer(
            "exercise_entity_to_text_index",
            (exercise_entity_to_text_index.detach().clone() if exercise_entity_to_text_index is not None else torch.full((nentity,), -1, dtype=torch.long)),
        )
        self.register_buffer(
            "relation_text_token_ids",
            (relation_text_token_ids.detach().clone() if relation_text_token_ids is not None else torch.zeros((0, 1), dtype=torch.long)),
        )
        self.shared_text_encoder: Optional[SharedTextBiGRU] = None
        if shared_text_encoder_config and shared_text_encoder_state is not None:
            self.shared_text_encoder = SharedTextBiGRU(
                vocab_size=int(shared_text_encoder_config["vocab_size"]),
                text_emb_size=int(shared_text_encoder_config["text_emb_size"]),
                text_hidden_size=int(shared_text_encoder_config["text_hidden_size"]),
                padding_idx=int(shared_text_encoder_config.get("padding_idx", 0)),
            )
            self.shared_text_encoder.word_embedding.load_state_dict(shared_text_encoder_state["word_embedding"])
            self.shared_text_encoder.text_encoder.load_state_dict(shared_text_encoder_state["text_encoder"])
            if int(shared_text_encoder_config["text_hidden_size"]) != self.text_dim:
                raise ValueError("shared text encoder output dimension must match text_dim")
        self.init()

    @classmethod
    def from_feature_bundle(cls, bundle: SemanticFeatureBundle, **kwargs) -> "TwoCKG4ER":
        return cls(
            nentity=bundle.nentity,
            nrelation=bundle.nrelation,
            text_dim=bundle.text_dim,
            numeric_dim=bundle.numeric_dim,
            relation_type_ids=bundle.relation_type_ids,
            relation_strengths=bundle.relation_strengths,
            entity_type_ids=bundle.entity_type_ids,
            cluster_ids=bundle.cluster_ids,
            text_features=bundle.text_features,
            numeric_features=bundle.numeric_features,
            state_features=bundle.state_features,
            semantic_quality=bundle.semantic_quality,
            exercise_text_token_ids=bundle.exercise_text_token_ids,
            exercise_entity_to_text_index=bundle.exercise_entity_to_text_index,
            relation_text_token_ids=bundle.relation_text_token_ids,
            shared_text_encoder_state=bundle.shared_text_encoder_state,
            shared_text_encoder_config=bundle.shared_text_encoder_config,
            numeric_feature_slices=bundle.numeric_feature_slices,
            state_feature_slices=bundle.state_feature_slices,
            **kwargs,
        )

    def init(self) -> None:
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.relation_id_emb.weight.data)
        xavier_normal_(self.relation_type_emb.weight.data)

    def _slice_width(self, name: str) -> int:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        return max(0, int(end - start))

    def _slice_numeric(self, entity_ids: torch.Tensor, name: str) -> torch.Tensor:
        start, end = self.numeric_feature_slices.get(name, (0, 0))
        if end <= start:
            return torch.zeros((entity_ids.numel(), 1), dtype=self.numeric_features.dtype, device=entity_ids.device)
        return self.numeric_features[entity_ids, start:end]

    def _entity_text(self, entity_ids: torch.Tensor) -> torch.Tensor:
        if self.shared_text_encoder is not None:
            text_indices = self.exercise_entity_to_text_index[entity_ids]
            valid = text_indices >= 0
            output = torch.zeros((entity_ids.numel(), self.text_dim), dtype=self.emb_e.weight.dtype, device=entity_ids.device)
            if valid.any():
                output[valid] = self.shared_text_encoder(self.exercise_text_token_ids[text_indices[valid]])
            return output
        text_features = self.text_features[entity_ids]
        if self.freeze_text_features:
            text_features = text_features.detach()
        if text_features.shape[1] == 0:
            return torch.zeros((entity_ids.numel(), self.text_dim), dtype=self.emb_e.weight.dtype, device=entity_ids.device)
        if text_features.shape[1] != self.text_dim:
            raise ValueError(f"expected text_dim={self.text_dim}, got {text_features.shape[1]}")
        return text_features

    def _uses_theta(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) != "2CKG4ER_learner_id"

    def _uses_exercise_text(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) != "2CKG4ER_exercise_id"

    def _uses_exercise_pedagogy(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) != "2CKG4ER_exercise_id"

    def _uses_relation_features(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) != "2CKG4ER_relation_id"

    def _removes_learner_id(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) in {
            "2CKG4ER",
            "2CKG4ER_relation_id",
            "2CKG4ER_exercise_id",
            "2CKG4ER_no_mastery",
            "2CKG4ER_no_forgetting",
        }

    def _uses_learner_id_only(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) == "2CKG4ER_learner_id"

    def _removes_relation_id(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) in {
            "2CKG4ER",
            "2CKG4ER_learner_id",
            "2CKG4ER_exercise_id",
            "2CKG4ER_no_mastery",
            "2CKG4ER_no_forgetting",
        }

    def _uses_relation_id_only(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) == "2CKG4ER_relation_id"

    def _removes_exercise_id(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) in {
            "2CKG4ER",
            "2CKG4ER_relation_id",
            "2CKG4ER_learner_id",
            "2CKG4ER_no_mastery",
            "2CKG4ER_no_forgetting",
        }

    def _uses_exercise_id_only(self) -> bool:
        return normalize_ablation_mode(self.ablation_mode) == "2CKG4ER_exercise_id"

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.emb_e(entity_ids)
        if self.ablation_mode == "id_only":
            return self.entity_norm(id_emb)

        type_ids = self.entity_type_ids[entity_ids]
        fused_emb = torch.empty_like(id_emb)

        uid_mask = type_ids == ENTITY_TYPE_TO_ID["uid"]
        if uid_mask.any():
            uid_ids = entity_ids[uid_mask]
            uid_id_emb = id_emb[uid_mask]
            if self._uses_learner_id_only():
                fused_emb[uid_mask] = uid_id_emb
            elif self._removes_learner_id():
                theta = self._slice_numeric(uid_ids, "learner_irt")
                fused_emb[uid_mask] = self.uid_feature_only_fusion(theta)
            else:
                fused_emb[uid_mask] = uid_id_emb

        ex_mask = type_ids == ENTITY_TYPE_TO_ID["ex"]
        if ex_mask.any():
            ex_ids = entity_ids[ex_mask]
            ex_id_emb = id_emb[ex_mask]
            if self._uses_exercise_id_only():
                fused_emb[ex_mask] = ex_id_emb
            else:
                fused_emb[ex_mask] = self.exercise_feature_only_fusion(
                    torch.cat([self._entity_text(ex_ids), self._slice_numeric(ex_ids, "exercise_irt")], dim=1)
                )

        kc_mask = type_ids == ENTITY_TYPE_TO_ID["kc"]
        if kc_mask.any():
            fused_emb[kc_mask] = self.kc_fusion(id_emb[kc_mask])

        other_mask = ~(uid_mask | ex_mask | kc_mask)
        if other_mask.any():
            fused_emb[other_mask] = id_emb[other_mask]

        return self.entity_norm(fused_emb)

    def gate_values(self) -> dict[str, object]:
        uid_features = (
            ["entity_id_200"]
            if self._uses_learner_id_only()
            else ["theta_mirt_norm_1"]
            if self._removes_learner_id()
            else ["entity_id_200", "theta_mirt_norm_1"]
        )
        exercise_features = (
            ["entity_id_200"]
            if self._uses_exercise_id_only()
            else ["shared_topic_text_100", "difficulty_mirt_norm_1", "discrimination_mirt_norm_1"]
            if self._removes_exercise_id()
            else ["entity_id_200", "shared_topic_text_100", "difficulty_mirt_norm_1", "discrimination_mirt_norm_1"]
        )
        relation_features = (
            ["relation_id_200"]
            if self._uses_relation_id_only()
            else ["relation_text_100", "strength_embedding_16"]
            if self._removes_relation_id()
            else ["relation_id_200", "relation_text_100", "strength_embedding_16"]
        )
        return {
            "fusion": "type-specific raw feature concatenation + MLP compression",
            "entity_features": {
                "uid": uid_features,
                "ex": exercise_features,
                "kc": ["entity_id_200"],
            },
            "relation_features": relation_features,
            "ablation": self.ablation_mode,
        }

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.relation_id_emb(relation_ids)
        if self._uses_relation_id_only() or not self._uses_relation_features():
            return self.relation_norm(id_emb)
        strength = self.relation_strengths[relation_ids]
        if self.shared_text_encoder is None or self.relation_text_token_ids.numel() == 0:
            type_emb = self.relation_type_emb(self.relation_type_ids[relation_ids])
            relation_emb = self.legacy_relation_feature_only_fusion(torch.cat([type_emb, strength], dim=1))
            return self.relation_norm(relation_emb)
        relation_type_ids = self.relation_type_ids[relation_ids].clamp(max=self.relation_text_token_ids.shape[0] - 1)
        relation_text = self.shared_text_encoder(self.relation_text_token_ids[relation_type_ids])
        strength_emb = self.strength_projector(strength)
        relation_emb = self.relation_feature_only_fusion(torch.cat([relation_text, strength_emb], dim=1))
        return self.relation_norm(relation_emb)

    def conve_transform(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        h_2d = h_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_2d = r_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([h_2d, r_2d], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

    def score_triples_logits(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)
        x = self.conve_transform(h_emb, r_emb)
        return torch.sum(x * t_emb, dim=1) + self.b[t]

    def score_triples(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score_triples_logits(h, r, t))

    def score_tail_pairs_from_head_embeddings_logits(
        self,
        h_emb: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if h_emb.dim() != 2:
            raise ValueError("h_emb must be a 2D tensor shaped [batch_size, embedding_dim]")
        if h_emb.shape[0] != r.shape[0] or h_emb.shape[0] != t.shape[0]:
            raise ValueError("h_emb, r, and t must contain the same number of samples")
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)
        x = self.conve_transform(h_emb, r_emb)
        return torch.sum(x * t_emb, dim=1) + self.b[t]

    def score_tail_pairs_from_head_embeddings(
        self,
        h_emb: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.score_tail_pairs_from_head_embeddings_logits(h_emb, r, t))

    def score_tails_logits(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        tail_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        x = self.conve_transform(h_emb, r_emb)
        if tail_ids is None:
            all_ids = torch.arange(self.nentity, device=h.device)
            tail_emb = self.entity_embedding(all_ids)
            bias = self.b
        else:
            tail_emb = self.entity_embedding(tail_ids)
            bias = self.b[tail_ids]
        scores = torch.mm(x, tail_emb.transpose(1, 0)) + bias.unsqueeze(0)
        return scores

    def score_tails(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        tail_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.score_tails_logits(h, r, tail_ids))

    def score_tail_matrix_logits(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tail_ids.dim() != 2:
            raise ValueError("tail_ids must be shaped [batch_size, candidate_count]")
        if h.dim() != 1 or r.dim() != 1:
            raise ValueError("h and r must be 1D tensors")
        if h.shape[0] != tail_ids.shape[0] or r.shape[0] != tail_ids.shape[0]:
            raise ValueError("h, r, and tail_ids must share the same batch size")
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        x = self.conve_transform(h_emb, r_emb)
        tail_emb = self.entity_embedding(tail_ids.reshape(-1)).view(
            tail_ids.shape[0],
            tail_ids.shape[1],
            self.embedding_dim,
        )
        scores = torch.sum(x.unsqueeze(1) * tail_emb, dim=2) + self.b[tail_ids]
        return scores

    def score_tail_matrix(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.score_tail_matrix_logits(h, r, tail_ids))


SemanticConvE = TwoCKG4ER
