"""Semantic CompGCN backbone with the existing ConvE decoder and KPPD head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from educational_kg_v2 import RELATIONS
from semantic_conve_model import TwoCKG4ER


class CompGCNLayer(nn.Module):
    """Relation-composition messages, direction-specific transforms and residuals."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.incoming = nn.Linear(dim, dim, bias=False)
        self.outgoing = nn.Linear(dim, dim, bias=False)
        self.self_loop = nn.Linear(dim, dim)
        self.relation_update = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        entities: torch.Tensor,
        relations: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        kind: torch.Tensor,
        strength_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        composed = entities[source] * (relations[kind] + strength_embedding)
        transformed = torch.empty_like(composed)
        forward = kind.remainder(2) == 0
        transformed[forward] = self.outgoing(composed[forward])
        transformed[~forward] = self.incoming(composed[~forward])
        aggregated = torch.zeros_like(entities).index_add(0, target, transformed)
        degree = torch.zeros((entities.shape[0], 1), dtype=entities.dtype, device=entities.device)
        degree.index_add_(0, target, torch.ones((target.numel(), 1), dtype=entities.dtype, device=entities.device))
        updated = self.norm(entities + F.relu(self.self_loop(entities) + aggregated / degree.clamp_min(1)))
        return updated, self.relation_update(relations)


class SemanticCompGCN(TwoCKG4ER):
    """Preserve semantic features, ConvE decoder and KPPD; add graph propagation."""

    def __init__(self, *args, graph_path: str | Path, compgcn_layers: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if compgcn_layers < 1:
            raise ValueError("compgcn_layers must be positive")
        with np.load(graph_path) as graph:
            if int(graph["nentity"]) != self.nentity:
                raise ValueError("Educational KG entity ids do not match semantic feature bundle")
            source = torch.from_numpy(graph["source"].astype(np.int64))
            target = torch.from_numpy(graph["target"].astype(np.int64))
            kind = torch.from_numpy(graph["relation"].astype(np.int64))
            strength = torch.from_numpy(graph["strength"].astype(np.float32))
        if source.numel() == 0 or source.numel() != target.numel() or source.numel() != kind.numel() or source.numel() != strength.numel():
            raise ValueError("Educational KG edge arrays must be nonempty and equal length")
        if int(kind.min()) < 0 or int(kind.max()) >= len(RELATIONS):
            raise ValueError("Educational KG contains an unknown relation type")
        if int(source.min()) < 0 or int(target.min()) < 0 or int(source.max()) >= self.nentity or int(target.max()) >= self.nentity:
            raise ValueError("Educational KG contains an unknown entity")
        self.register_buffer("graph_source", source, persistent=False)
        self.register_buffer("graph_target", target, persistent=False)
        self.register_buffer("graph_kind", kind, persistent=False)
        self.register_buffer("graph_strength", strength, persistent=False)
        self.graph_relations = nn.Parameter(torch.empty((len(RELATIONS), self.embedding_dim)))
        nn.init.xavier_uniform_(self.graph_relations)
        self.edge_strength_encoder = nn.Sequential(nn.Linear(1, self.embedding_dim), nn.Tanh())
        self.comp_layers = nn.ModuleList(CompGCNLayer(self.embedding_dim) for _ in range(compgcn_layers))
        self.graph_path = str(Path(graph_path).resolve())
        self._eval_embedding_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "SemanticCompGCN":
        self._eval_embedding_cache = None
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        self._eval_embedding_cache = None
        return super().load_state_dict(*args, **kwargs)

    def all_entity_embeddings(self) -> torch.Tensor:
        use_cache = not self.training and not torch.is_grad_enabled()
        if use_cache and self._eval_embedding_cache is not None:
            return self._eval_embedding_cache
        ids = torch.arange(self.nentity, device=self.emb_e.weight.device)
        entities = super().entity_embedding(ids)
        relations = self.graph_relations
        strength_embedding = self.edge_strength_encoder(self.graph_strength.unsqueeze(1))
        for layer in self.comp_layers:
            entities, relations = layer(
                entities, relations, self.graph_source, self.graph_target,
                self.graph_kind, strength_embedding,
            )
        if use_cache:
            self._eval_embedding_cache = entities
        return entities

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.all_entity_embeddings()[entity_ids]

    def gate_values(self) -> dict[str, object]:
        values = super().gate_values()
        values["backbone"] = "semantic_compgcn"
        values["compgcn_layers"] = len(self.comp_layers)
        values["message_relations"] = RELATIONS
        return values
