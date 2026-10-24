"""Explicit graph snapshots and the inherited ConvE decoder. No hidden graph cache."""
from dataclasses import dataclass

import torch
from torch import nn

from .compgcn_layer import SemanticCompGCNLayer


def stable_top_n(logits, n):
    """Catalog columns must be original exercise order; ties use exercise ID."""
    return torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :n]


@dataclass
class SemanticInitials:
    entities: torch.Tensor
    relations: torch.Tensor
    valid: torch.Tensor
    owner: int
    parameter_versions: tuple
    training: bool


@dataclass
class Encoding:
    entities: torch.Tensor
    relations: torch.Tensor
    valid: torch.Tensor
    graph: object
    owner: int
    parameter_versions: tuple
    training: bool
    graph_versions: tuple
    diagnostics: dict | None = None
    bypass: bool = False


class SemanticCompGCN(nn.Module):
    def __init__(self, base, inverse_original_ids, *, test_entities=(), chunk_size=16384,
                 recompute=True, graph_seed=2024):
        super().__init__()
        if base.ablation_mode != "feature_only":
            raise ValueError("Only the frozen feature_only definition is supported")
        self.base = base
        self.inverse_original_ids = tuple(inverse_original_ids)
        self.test_entities = tuple(sorted(test_entities))
        # Graph parameter initialization must not change original sampler/Dropout RNG.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also reseeds CUDA, which fork_rng(devices=[]) does
            # not restore. New matrices are initialized on CPU, so seed CPU only.
            torch.random.default_generator.manual_seed(graph_seed+240024)
            self.layer = SemanticCompGCNLayer(base.emb_e.embedding_dim, chunk_size, recompute)
        self.layer.to(device=base.b.device, dtype=base.b.dtype)

    def _versions(self):
        return tuple((p.data_ptr(), p._version, str(p.device), str(p.dtype)) for p in self.parameters())

    def _check_initials(self, initial):
        if initial.owner != id(self) or initial.parameter_versions != self._versions() or initial.training != self.training:
            raise ValueError("Stale/cross-model/mode semantic snapshot")

    def semantic_initials(self):
        device = self.base.b.device; count = self.base.nentity
        valid = torch.ones(count, dtype=torch.bool, device=device)
        # Test features do not consume training RNG or enter training computation.
        if self.training and self.test_entities:
            valid[torch.tensor(self.test_entities, device=device)] = False
        ids = torch.where(valid)[0]
        h = self.base.b.new_zeros((count, self.layer.dimension))
        h = h.index_copy(0, ids, self.base.entity_embedding(ids))
        r = self.base.relation_embedding(torch.arange(self.base.nrelation, device=device))
        return SemanticInitials(h, r, valid, id(self), self._versions(), self.training)

    def encode_graph(self, graph, mode="full", *, initial=None, output_ids=None,
                     neighbors=True, diagnostics=False):
        if mode not in ("full", "local", "bypass"):
            raise ValueError("Unknown encoding mode")
        if self.training and graph.protocol == "B":
            raise ValueError("Test query states cannot enter training")
        if graph.degree.device != self.base.b.device:
            raise ValueError("Graph must be explicitly moved to model device")
        if mode == "bypass":
            # No new encoder or semantic forward; scoring delegates original functions.
            valid = torch.ones(self.base.nentity, dtype=torch.bool, device=self.base.b.device)
            return Encoding(None, None, valid, graph, id(self), self._versions(), self.training,
                            graph.tensor_versions(), bypass=True)
        if mode == "local" and (self.training or torch.is_grad_enabled()):
            raise ValueError("Local inference requires eval and no_grad")
        if mode == "local" and output_ids is None:
            raise ValueError("Local inference must declare output IDs")
        if mode == "full" and output_ids is not None:
            raise ValueError("Full reference materializes all valid outputs")
        initial = self.semantic_initials() if initial is None else initial
        self._check_initials(initial)
        if not initial.valid[graph.source].all() or not initial.valid[graph.target].all():
            raise ValueError("Unencoded entity used by message graph")
        if output_ids is not None:
            if (output_ids.ndim!=1 or output_ids.dtype!=torch.long or (output_ids<0).any()
                or (output_ids>=self.base.nentity).any() or len(torch.unique(output_ids))!=len(output_ids)
                or not initial.valid[output_ids].all()):
                raise ValueError("Invalid/duplicate local output IDs")
        result = self.layer(initial.entities, initial.relations, graph, self.inverse_original_ids,
                            output_ids=output_ids, neighbors=neighbors, diagnostics=diagnostics)
        if output_ids is None:
            entities = result.entities; valid = initial.valid.clone()
        else:
            entities = torch.zeros_like(initial.entities).index_copy(0, output_ids, result.entities)
            valid = torch.zeros_like(initial.valid); valid[output_ids] = True
        return Encoding(entities, result.relations[:self.base.nrelation], valid, graph, id(self),
                        self._versions(), self.training, graph.tensor_versions(), result.diagnostics)

    def _check(self, encoded, ids, query_entity=None):
        if not isinstance(encoded, Encoding) or encoded.owner != id(self):
            raise ValueError("Explicit same-model encoding required")
        if encoded.parameter_versions != self._versions() or encoded.training != self.training:
            raise ValueError("Stale encoding after parameter/device/mode change")
        if encoded.graph.tensor_versions() != encoded.graph_versions:
            raise ValueError("Graph tensors changed after encoding")
        for group in ids:
            if group.dtype != torch.long or group.ndim != 1 or (group<0).any() or (group>=len(encoded.valid)).any():
                raise ValueError("Invalid original entity IDs")
            if not encoded.valid[group].all():
                raise ValueError("Requested row not encoded; invalid placeholder access")
        if encoded.graph.protocol == "B":
            if query_entity is None or not (query_entity == encoded.graph.query_entity).all():
                raise ValueError("Cross-query scoring rejected: B snapshot belongs to one student")

    def score_logits(self, h, r, t, encoded):
        self._check(encoded, (h, t), h)
        if r.shape != h.shape or t.shape != h.shape or r.dtype!=torch.long or (r<0).any() or (r>=self.base.nrelation).any():
            raise ValueError("Invalid aligned supervised relation IDs")
        if encoded.bypass:
            return self.base.score_logits(h, r, t)
        x = self.base.conve_transform(encoded.entities[h], encoded.relations[r])
        return (x*encoded.entities[t]).sum(1) + self.base.b[t]

    def score_rec_all(self, users, exercise_ids, encoded, rec_id):
        self._check(encoded, (users, exercise_ids), users)
        r = torch.full_like(users, rec_id)
        if encoded.bypass:
            h = self.base.entity_embedding(users); relation = self.base.relation_embedding(r)
            tails = self.base.entity_embedding(exercise_ids)
        else:
            h = encoded.entities[users]; relation = encoded.relations[r]; tails = encoded.entities[exercise_ids]
        return self.base.conve_transform(h, relation) @ tails.T + self.base.b[exercise_ids].unsqueeze(0)
