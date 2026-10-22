"""CompGCN encoder for Educational KG v2, adapted from Vashishth et al. ICLR 2020.

The official Apache-2.0 implementation informs composition, directional
transforms, relation updates and normalization. Continuous strengths, semantic
features and the ConvE/KPPD decoder are project-specific extensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from educational_kg_v2 import RELATIONS
from feature_loader import ENTITY_TYPE_TO_ID
from semantic_conve_model import TwoCKG4ER


def checkpoint_student_mode(config: dict) -> str:
    """Read legacy graph-only checkpoints without changing their predictions."""
    version = config.get("compgcn_impl_version")
    if version == 2:
        if config.get("compgcn_student_mode", "graph") != "graph":
            raise ValueError("CompGCN v2 checkpoints must use graph student encoding")
        return "graph"
    if version == 3 and config.get("compgcn_student_mode") in {"direct", "graph"}:
        return config["compgcn_student_mode"]
    raise ValueError("Unsupported CompGCN checkpoint configuration; use a new run directory")


def compose(entity: torch.Tensor, relation: torch.Tensor, op: str) -> torch.Tensor:
    if op == "mult":
        return entity * relation
    if op == "sub":
        return entity - relation
    if op == "corr":
        return torch.fft.irfft(torch.conj(torch.fft.rfft(entity, dim=-1)) * torch.fft.rfft(relation, dim=-1), n=entity.shape[-1], dim=-1)
    raise ValueError(f"Unknown composition: {op}")


class CompGCNLayer(nn.Module):
    """Original, inverse and self-loop messages with separate transforms."""

    def __init__(self, dim: int, *, op: str = "corr", dropout: float = 0.1) -> None:
        super().__init__()
        if op not in {"corr", "sub", "mult"}:
            raise ValueError(f"Unknown composition: {op}")
        self.op = op
        self.original = nn.Linear(dim, dim, bias=False)
        self.inverse = nn.Linear(dim, dim, bias=False)
        self.self_loop = nn.Linear(dim, dim, bias=False)
        self.loop_relation = nn.Parameter(torch.empty(dim))
        self.relation_update = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(dim)
        nn.init.xavier_normal_(self.loop_relation.unsqueeze(0))

    def _aggregate(self, entities, relations, source, target, kind, strength, transform):
        result = torch.zeros_like(entities)
        if source.numel() == 0:
            return result
        # Each direction in our typed graph can be bipartite (student -> KC).
        # Use source and destination degrees separately so destination-only
        # nodes still receive messages; the authors' single degree vector
        # would zero all messages on such a directional partition.
        source_degree = torch.bincount(source, minlength=entities.shape[0]).to(entities.dtype)
        target_degree = torch.bincount(target, minlength=entities.shape[0]).to(entities.dtype)
        message = transform(compose(entities[source], relations[kind] + strength, self.op))
        weight = (source_degree[source].rsqrt() * target_degree[target].rsqrt()).unsqueeze(1)
        return result.index_add(0, target, message * weight)

    def forward(self, entities, relations, source, target, kind, strength):
        forward = kind.remainder(2) == 0
        original = self._aggregate(entities, relations, source[forward], target[forward], kind[forward], strength[forward], self.original)
        inverse = self._aggregate(entities, relations, source[~forward], target[~forward], kind[~forward], strength[~forward], self.inverse)
        loop = self.self_loop(compose(entities, self.loop_relation, self.op))
        updated = self.norm((self.dropout(original) + self.dropout(inverse) + loop) / 3)
        return torch.tanh(updated), self.relation_update(relations)


class SemanticCompGCN(TwoCKG4ER):
    """CompGCN graph encoder with the established semantic ConvE/KPPD scorer."""

    def __init__(self, *args, graph_path: str | Path, compgcn_layers: int = 1, compgcn_op: str = "corr", compgcn_dropout: float = 0.1, compgcn_student_mode: str = "direct", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if compgcn_student_mode not in {"direct", "graph"}:
            raise ValueError("compgcn_student_mode must be direct or graph")
        self.compgcn_student_mode = compgcn_student_mode
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
        if not torch.isfinite(strength).all() or torch.any((strength < 0) | (strength > 1)):
            raise ValueError("Educational KG strengths must be finite in [0, 1]")
        self.register_buffer("graph_source", source, persistent=False)
        self.register_buffer("graph_target", target, persistent=False)
        self.register_buffer("graph_kind", kind, persistent=False)
        self.register_buffer("graph_strength", strength, persistent=False)
        # The rec relation participates in W_rel and decoding, never in message edges.
        self.graph_relations = nn.Parameter(torch.empty((len(RELATIONS) + 1, self.embedding_dim)))
        nn.init.xavier_uniform_(self.graph_relations)
        self.edge_strength_encoder = nn.Sequential(nn.Linear(1, self.embedding_dim), nn.Tanh())
        self.comp_layers = nn.ModuleList(CompGCNLayer(self.embedding_dim, op=compgcn_op, dropout=compgcn_dropout) for _ in range(compgcn_layers))
        self.compgcn_op = compgcn_op
        self.compgcn_dropout = compgcn_dropout
        # Original supervised relation types: rec, mlkc, exfr, other.
        self.register_buffer("type_to_graph_relation", torch.tensor([len(RELATIONS), 0, 2, 4]), persistent=False)
        self.graph_path = str(Path(graph_path).resolve())
        self._eval_embedding_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._train_embedding_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def train(self, mode: bool = True) -> "SemanticCompGCN":
        self._eval_embedding_cache = None
        self._train_embedding_cache = None
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        self._eval_embedding_cache = None
        self._train_embedding_cache = None
        return super().load_state_dict(*args, **kwargs)

    def begin_graph_batch(self) -> None:
        if self._train_embedding_cache is not None:
            raise RuntimeError("A CompGCN training batch is already active")
        self._train_embedding_cache = self._all_graph_embeddings()

    def end_graph_batch(self) -> None:
        self._train_embedding_cache = None

    def _all_graph_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._train_embedding_cache is not None:
            return self._train_embedding_cache
        use_cache = not self.training and not torch.is_grad_enabled()
        if use_cache and self._eval_embedding_cache is not None:
            return self._eval_embedding_cache
        ids = torch.arange(self.nentity, device=self.emb_e.weight.device)
        entities = super().entity_embedding(ids)
        initial_entities = entities
        relations = self.graph_relations
        strength = self.edge_strength_encoder(self.graph_strength.unsqueeze(1))
        for layer in self.comp_layers:
            entities, relations = layer(entities, relations, self.graph_source, self.graph_target, self.graph_kind, strength)
        if self.compgcn_student_mode == "direct":
            # Train and held-out learners use the same ConvE feature path.
            # Train-only learners may still send messages to KC/item nodes.
            student_mask = (self.entity_type_ids == ENTITY_TYPE_TO_ID["uid"]).unsqueeze(1)
            entities = torch.where(student_mask, initial_entities, entities)
        if use_cache:
            self._eval_embedding_cache = (entities, relations)
        return entities, relations

    def all_entity_embeddings(self) -> torch.Tensor:
        return self._all_graph_embeddings()[0]

    def entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.all_entity_embeddings()[entity_ids]

    def relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        semantic = super().relation_embedding(relation_ids)
        graph_relations = self._all_graph_embeddings()[1]
        types = self.relation_type_ids[relation_ids]
        return semantic + graph_relations[self.type_to_graph_relation[types]]

    def score_triples_core_logits(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        entities, graph_relations = self._all_graph_embeddings()
        types = self.relation_type_ids[r]
        relation = super().relation_embedding(r) + graph_relations[self.type_to_graph_relation[types]]
        transformed = self.conve_transform(entities[h], relation)
        return torch.sum(transformed * entities[t], dim=1) + self.tail_bias(t)

    def gate_values(self) -> dict[str, object]:
        values = super().gate_values()
        values["backbone"] = "semantic_compgcn"
        values["compgcn_layers"] = len(self.comp_layers)
        values["compgcn_op"] = self.compgcn_op
        values["compgcn_student_mode"] = self.compgcn_student_mode
        values["message_relations"] = RELATIONS
        values["query_relation"] = "rec"
        return values
