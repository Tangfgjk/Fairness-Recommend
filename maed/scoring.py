from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass(frozen=True)
class EducationScale:
    center: float
    scale: float
    clip: float
    train_users: tuple
    raw_iqr: float
    floor: float = 1e-6

    @classmethod
    def fit(cls, state, clip=3.0, floor=1e-6):
        if not np.isfinite(clip) or clip <= 0 or not np.isfinite(floor) or floor <= 0:
            raise ValueError("Scale constants must be positive finite numbers")
        values = state.training_distance(state.train)
        lo, hi = np.quantile(values, [.25, .75])
        return cls(float(np.median(values)), max(float(hi-lo), floor), float(clip),
                   tuple(int(u) for u in state.train), float(hi-lo), floor)

    def transform(self, distance):
        if torch.is_tensor(distance):
            return -torch.clamp((distance-self.center)/self.scale, -self.clip, self.clip)
        return -np.clip((np.asarray(distance)-self.center)/self.scale, -self.clip, self.clip)

    def to_dict(self):
        return asdict(self)


def fuse_logits(logits, distance, scale, beta):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if logits.shape != distance.shape:
        raise ValueError("Q distances and logits must have identical shapes")
    if beta == 0:
        return logits  # Exact bypass: no new arithmetic or random-number draws.
    return logits + beta * scale.transform(distance)


def score_rec(model, bundle, uids, nitems):
    """Full catalog, real logits; also differentiable for model-direction probes."""
    device = next(model.parameters()).device
    heads = torch.tensor([bundle.entity2id[f"uid{int(u)}"] for u in uids], device=device)
    rel = torch.full_like(heads, bundle.relation2id['rec'])
    tails = torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(nitems)], device=device)
    h = model.entity_embedding(heads)
    r = model.relation_embedding(rel)
    t = model.entity_embedding(tails)
    return model.conve_transform(h, r) @ t.T + model.b[tails].unsqueeze(0)
