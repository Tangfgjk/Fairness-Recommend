from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class MatchConfig:
    distance_gap: float = 0.02
    mastery_gap: float = 0.05
    forgetting_gap: float = 0.05
    min_log_frequency_gap: float = float(np.log(2))
    frequency_bin_width: float = 0.25
    pair_seed: int = 20261022

    def validate(self):
        if min(self.distance_gap, self.mastery_gap, self.forgetting_gap) < 0:
            raise ValueError("Matching tolerances must be nonnegative")
        if self.min_log_frequency_gap <= 0 or self.frequency_bin_width <= 0:
            raise ValueError("Frequency gaps/bin widths must be positive")


def candidates(ctx, uid, exercises, cfg):
    exs = np.asarray(exercises, dtype=int)
    i, j = np.triu_indices(len(exs), 1)
    a, b = exs[i], exs[j]
    logf = np.log1p(ctx.frequency)
    gap = np.abs(logf[a] - logf[b])
    keep = gap >= cfg.min_log_frequency_gap
    a, b, gap = a[keep], b[keep], gap[keep]
    high = ctx.frequency[a] > ctx.frequency[b]
    h, l = np.where(high, a, b), np.where(high, b, a)
    # Products only need to be computed for the candidate exercises.
    products = dict(zip(exs.tolist(), ctx.product(uid, exs).tolist()))
    mgap = np.array([abs(products[int(x)] - products[int(y)]) for x, y in zip(h, l)])
    mask = ((np.abs(ctx.distance[uid, h] - ctx.distance[uid, l]) <= cfg.distance_gap)
            & (mgap <= cfg.mastery_gap)
            & (np.abs(ctx.forget[uid, h] - ctx.forget[uid, l]) <= cfg.forgetting_gap)
            & (ctx.q[h].sum(1) == ctx.q[l].sum(1)))
    pairs = np.column_stack((np.full(len(h), uid), h, l)).astype(np.int64)
    bins = np.floor(gap / cfg.frequency_bin_width).astype(int)
    return pairs, mask, bins


def build_pairs(ctx, cfg):
    cfg.validate()
    rng = np.random.default_rng(cfg.pair_seed)
    matched, unconditioned, test, support = [], [], [], []
    for u in ctx.train:
        pairs, mask, bins = candidates(ctx, int(u), sorted(ctx.positives[int(u)]), cfg)
        selected = pairs[mask]
        matched.extend(selected.tolist())
        # E1 is a subset of the broad pool, so each stratum always has enough candidates.
        # E3 removes the educational restriction, not the original positive-label restriction.
        broad = []
        for b in np.unique(bins[mask]):
            n = int(np.sum(mask & (bins == b)))
            ix = rng.choice(np.flatnonzero(bins == b), n, replace=False)
            broad.extend(pairs[ix].tolist())
        unconditioned.extend(broad)
        support.append({"uid": int(u), "matched_pairs": len(selected), "E3_pairs": len(broad)})
    for u in ctx.test:
        pairs, mask, _ = candidates(ctx, int(u), ctx.rule(int(u)), cfg)
        test.extend(pairs[mask].tolist())
    def array(rows):
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    arrays = {"E1": array(matched), "E3": array(unconditioned), "evaluation": array(test)}
    arrays["E4_direction"] = np.random.default_rng(cfg.pair_seed + 1).choice(
        [-1, 1], len(arrays["E1"])).astype(np.int8)
    return arrays, {"matching": asdict(cfg), "train_support": support,
                    "evaluation_scope": "matched pairs within each test user's rule Top-50; no model scores used",
                    "audit": {k: audit_pairs(ctx, a, cfg) for k, a in arrays.items() if a.ndim == 2}}


def audit_pairs(ctx, rows, cfg):
    if not len(rows):
        return {"pairs": 0, "users": 0, "reason": "no eligible pairs"}
    u, h, l = rows.T
    loggap = np.log1p(ctx.frequency[h]) - np.log1p(ctx.frequency[l])
    union = np.logical_or(ctx.q[h], ctx.q[l]).sum(1)
    jc = 1 - np.logical_and(ctx.q[h], ctx.q[l]).sum(1) / union
    def stats(x):
        return {"mean": float(np.mean(x)), "p50": float(np.median(x)), "p95": float(np.quantile(x, .95)), "max": float(np.max(x))}
    return {"pairs": len(rows), "users": len(np.unique(u)),
            "log_frequency_gap": stats(loggap),
            "frequency_gap_bins": {str(int(b)): int(n) for b, n in zip(*np.unique(np.floor(loggap / cfg.frequency_bin_width), return_counts=True))},
            "distance_gap": stats(abs(ctx.distance[u, h] - ctx.distance[u, l])),
            "forgetting_gap": stats(abs(ctx.forget[u, h] - ctx.forget[u, l])),
            "difficulty_gap": stats(abs(ctx.difficulty[h] - ctx.difficulty[l])),
            "discrimination_gap": stats(abs(ctx.discrimination[h] - ctx.discrimination[l])),
            "KC_jaccard_distance": stats(jc), "same_KC_pairs": int(np.sum(jc == 0))}


def exposure_audit(ctx, scores, pairs):
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    counts = np.bincount(ranking.ravel(), minlength=len(ctx.q))
    high_counts = np.bincount(pairs[:, 1], minlength=len(ctx.q)) if len(pairs) else np.zeros(len(ctx.q), int)
    low_counts = np.bincount(pairs[:, 2], minlength=len(ctx.q)) if len(pairs) else np.zeros(len(ctx.q), int)
    top = np.argsort(-counts, kind="stable")[:20]
    return {"top_n": 10, "top20_exposed_exercises": [
        {"exercise": int(e), "exposure": int(counts[e]), "rec_frequency": int(ctx.frequency[e]),
         "high_side_pairs": int(high_counts[e]), "low_side_pairs": int(low_counts[e]),
         "high_side_train_users": len(np.unique(pairs[pairs[:, 1] == e, 0])) if len(pairs) else 0}
        for e in top if counts[e] > 0],
        "exposure_mass_on_any_matched_item": float(counts[(high_counts + low_counts) > 0].sum() / counts.sum()),
        "exposure_mass_on_high_side_item": float(counts[high_counts > 0].sum() / counts.sum())}
