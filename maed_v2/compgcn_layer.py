"""One-layer mult CompGCN adapted from malllabiisc/CompGCN (Apache-2.0).

Modified implementation: shared semantic inverse transform, bidirectional degree,
node LayerNorm, semantic residual, exact isolated bypass, block recomputation.
See third_party/CompGCN/PROVENANCE.json and LICENSE for pinned upstream sources.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class LayerOutput:
    entities: torch.Tensor
    relations: torch.Tensor
    diagnostics: dict | None = None


class SemanticCompGCNLayer(nn.Module):
    def __init__(self, dimension=200, chunk_size=16384, recompute=True):
        super().__init__()
        if dimension < 1 or chunk_size < 1:
            raise ValueError("Positive dimension/chunk size required")
        self.dimension = dimension; self.chunk_size = chunk_size; self.recompute = recompute
        for name in ("w_forward", "w_inverse", "w_loop", "w_relation", "inverse_transform"):
            self.register_parameter(name, nn.Parameter(torch.empty(dimension, dimension)))
            nn.init.xavier_normal_(getattr(self, name))
        self.norm = nn.LayerNorm(dimension, eps=1e-5, elementwise_affine=True)

    def relation_initials(self, relations, inverse_original_ids):
        ids = torch.as_tensor(inverse_original_ids, device=relations.device, dtype=torch.long)
        return torch.cat((relations, relations[ids] @ self.inverse_transform))

    def aggregate(self, h0, r0, graph, direction, output_ids=None, family=None):
        """Pre-transform aggregate. Checkpoint only pure product/index-add blocks.

        output_ids restricts RECEIVERS, never the message sources or their degrees.
        This is a one-layer exact local reference, not a cached delta approximation.
        """
        if h0.device != graph.degree.device:
            raise ValueError("Graph/features on different devices")
        mask = graph.direction == direction
        if family is not None:
            mask = mask & (graph.family == family)
        if output_ids is None:
            size = len(h0); targets = graph.target
        else:
            lookup = torch.full((len(h0),), -1, device=h0.device, dtype=torch.long)
            lookup[output_ids] = torch.arange(len(output_ids), device=h0.device)
            targets = lookup[graph.target]
            mask = mask & (targets >= 0); size = len(output_ids)
        src = graph.source[mask]; dst = targets[mask]; rel = graph.relation[mask]
        receiver = graph.target[mask]
        inv_degree = graph.degree.to(h0.dtype).clamp_min(1).rsqrt()
        alpha = inv_degree[src] * inv_degree[receiver]
        result = h0.new_zeros((size, h0.shape[1]))
        for start in range(0, len(src), self.chunk_size):
            end = start+self.chunk_size
            # Defaults bind each block; backward cannot accidentally reuse last indices.
            def block(h, r, s=src[start:end], t=dst[start:end], rid=rel[start:end], a=alpha[start:end]):
                message = h[s] * r[rid] * a[:, None]
                return h.new_zeros((size, h.shape[1])).index_add(0, t, message)
            if self.recompute and torch.is_grad_enabled() and (h0.requires_grad or r0.requires_grad):
                piece = checkpoint(block, h0, r0, use_reentrant=False, preserve_rng_state=False)
            else:
                piece = block(h0, r0)
            result = result + piece
        return result

    def forward(self, h0, original_relations, graph, inverse_original_ids, *,
                output_ids=None, neighbors=True, diagnostics=False):
        if h0.ndim != 2 or h0.shape != (graph.nentity, self.dimension):
            raise ValueError("Full entity ID-space initialization required")
        r0 = self.relation_initials(original_relations, inverse_original_ids)
        # Inverse is formed BEFORE update; messages use the layer-zero relations.
        r1 = .5*r0 + .5*(r0 @ self.w_relation)
        ids = torch.arange(len(h0), device=h0.device) if output_ids is None else output_ids
        if neighbors:
            a = self.aggregate(h0, r0, graph, 0, output_ids)
            b = self.aggregate(h0, r0, graph, 1, output_ids)
            forward = a @ self.w_forward; inverse = b @ self.w_inverse
        else:
            forward = torch.zeros_like(h0[ids]); inverse = torch.zeros_like(forward)
        self_message = h0[ids] @ self.w_loop
        pre_norm = (forward + inverse + self_message)/3
        normalized = self.norm(pre_norm); candidate = torch.tanh(normalized)
        connected = graph.degree[ids] > 0
        entities = torch.where(connected[:, None], .5*h0[ids] + .5*candidate, h0[ids])
        extra = None
        if diagnostics:
            extra = {"h0": h0[ids], "candidate": candidate, "forward": forward,
                     "inverse": inverse, "self": self_message, "pre_norm": pre_norm,
                     "normalized": normalized, "connected": connected, "ids": ids}
            if neighbors:
                for family, name in ((0, "mlkc"), (1, "exfr")):
                    before = [self.aggregate(h0, r0, graph, d, output_ids, family) for d in (0, 1)]
                    extra[name+"_pre_forward"], extra[name+"_pre_inverse"] = before
                    extra[name+"_forward"] = before[0] @ self.w_forward
                    extra[name+"_inverse"] = before[1] @ self.w_inverse
        return LayerOutput(entities, r1, extra)
