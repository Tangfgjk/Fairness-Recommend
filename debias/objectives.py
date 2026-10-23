import numpy as np
import torch
from .scoring import pair_logits


def matched_hinge(high, low, users, margin, direction=None):
    delta = high - low
    if direction is not None:
        delta = delta * direction
    values = torch.relu(delta - margin).square()
    return torch.stack([values[users == u].mean() for u in torch.unique(users)]).mean()


class FairObjective:
    def __init__(self, pairs, bundle, seed, margin=.1, users_per_step=32, pairs_per_user=4, directions=None):
        if len(pairs) == 0:
            raise ValueError("No eligible training pairs; cannot run a fairness treatment")
        self.pairs = pairs
        self.users = np.unique(pairs[:, 0])
        self.indices = {int(u): np.flatnonzero(pairs[:, 0] == u) for u in self.users}
        self.rng = np.random.default_rng(seed + 580001)
        self.margin, self.users_per_step, self.pairs_per_user = margin, users_per_step, pairs_per_user
        self.directions = directions
        self.entity2id, self.rec = bundle.entity2id, bundle.relation2id["rec"]

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state

    def __call__(self, model, device):
        selected = self.rng.choice(self.users, min(len(self.users), self.users_per_step), replace=False)
        ix = np.concatenate([self.rng.choice(self.indices[int(u)], min(len(self.indices[int(u)]), self.pairs_per_user), replace=False) for u in selected])
        p = self.pairs[ix]
        make = lambda values: torch.as_tensor(values, dtype=torch.long, device=device)
        heads = make([self.entity2id[f"uid{u}"] for u in p[:, 0]])
        high = make([self.entity2id[f"ex{e}"] for e in p[:, 1]])
        low = make([self.entity2id[f"ex{e}"] for e in p[:, 2]])
        hz, lz = pair_logits(model, heads, torch.full_like(heads, self.rec), high, low)
        direction = torch.as_tensor(self.directions[ix], device=device) if self.directions is not None else None
        loss = matched_hinge(hz, lz, make(p[:, 0]), self.margin, direction)
        signed = hz - lz if direction is None else (hz - lz) * direction
        return loss, {"pairs": len(ix), "users": len(selected),
                      "active_fraction": float((signed > self.margin).float().mean().detach()),
                      "high_low_logit_gap": float((hz - lz).mean().detach()),
                      "pair_logit_std": float(torch.cat((hz, lz)).std().detach())}


def combine_loss(model, bce, objective, weight, has_rec, device):
    # Strict baseline path: do not draw a sample or execute an auxiliary forward.
    if weight == 0 or not has_rec:
        return bce, None, {}
    fair, stats = objective(model, device)
    return bce + weight * fair, fair, stats
