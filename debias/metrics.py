from __future__ import annotations

import numpy as np
from .io import TOP_NS


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.dot(np.arange(1, len(x) + 1), x) / x.sum() - len(x) - 1) / len(x))


def total_variation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Exposure distributions must have positive mass")
    return float(np.abs(a / a.sum() - b / b.sum()).sum() / 2)


def novelty_set(recommended, correct):
    union = set(recommended) | set(correct)
    if not union:
        raise ValueError("Undefined Jaccard novelty: empty union")
    return 1 - len(set(recommended) & set(correct)) / len(union)


def conditional_preference(scores, uids, pairs):
    lookup = {int(u): i for i, u in enumerate(uids)}
    per_user = []
    for u in np.unique(pairs[:, 0]):
        p = pairs[pairs[:, 0] == u]
        delta = scores[lookup[int(u)], p[:, 1]] - scores[lookup[int(u)], p[:, 2]]
        per_user.append({"uid": int(u), "pairs": len(p),
                         "CFP": float(np.mean((delta > 0) + .5 * (delta == 0))),
                         "score_gap": float(delta.mean())})
    return {"value": float(np.mean([r["CFP"] for r in per_user])) if per_user else None,
            "mean_score_gap": float(np.mean([r["score_gap"] for r in per_user])) if per_user else None,
            "users": len(per_user), "pairs": len(pairs), "per_user": per_user}


def evaluate(ctx, scores, pairs, baseline=None, top_ns=TOP_NS):
    ranking = np.argsort(-scores, axis=1, kind="stable")
    rules = np.array([ctx.rule(int(u), max(top_ns)) for u in ctx.test])
    q = ctx.q.astype(float)
    frac_q = q / q.sum(1, keepdims=True)
    valid_kcs = q.sum(0) > 0
    metrics, users, items = {}, [], {}
    for n in top_ns:
        top = ranking[:, :n]
        expo = np.bincount(top.ravel(), minlength=len(q))
        ref = np.bincount(rules[:, :n].ravel(), minlength=len(q))
        kc_expo = expo @ frac_q
        values = []
        sets = [set(row.tolist()) for row in top]
        for row, uid in enumerate(ctx.test):
            selected = top[row]
            concepts = np.flatnonzero(q[selected].sum(0)).tolist()
            acc = np.mean([1 - abs(.8 - np.prod([ctx.ada_mastery[int(uid)][int(k)] for k in np.flatnonzero(q[e])])) for e in selected])
            record = {"uid": int(uid), "N": n, "Ada": float(acc),
                      "KG4Ex-NOV-Set": novelty_set(concepts, ctx.correct[int(uid)]),
                      "RuleOverlap": len(sets[row] & set(rules[row, :n])) / n}
            if baseline is not None:
                base = set(np.argsort(-baseline[row], kind="stable")[:n])
                record["BaselineOverlap"] = len(sets[row] & base) / n
                record["ChangedSet"] = float(sets[row] != base)
            values.append(record)
        keys = [k for k in values[0] if k not in ("uid", "N")]
        result = {k: float(np.mean([r[k] for r in values])) for k in keys}
        distances = [1 - len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                     for i in range(len(sets)) for j in range(i)]
        result.update(ItemGini=gini(expo), KCGini=gini(kc_expo[valid_kcs]),
                      ItemCoverage=float(np.mean(expo > 0)), KCCoverage=float(np.mean(kc_expo[valid_kcs] > 0)),
                      DED=total_variation(expo, ref),
                      UniqueOrderedRatio=len({tuple(r) for r in top}) / len(top),
                      UniqueSetRatio=len({frozenset(r) for r in top}) / len(top),
                      ListJaccardDistance=float(np.mean(distances)) if distances else 0.0)
        metrics[str(n)] = result
        items[str(n)] = {"model_exposure": expo.tolist(), "rule_exposure": ref.tolist(), "KC_exposure": kc_expo.tolist()}
        users.extend(values)
    return {"dataset": ctx.name, "context_hash": ctx.fingerprint, "test_users": len(ctx.test),
            "candidate_exercises": len(q), "candidate_KCs": int(valid_kcs.sum()),
            "NOV_definition": "KG4Ex-NOV-Set; union of Top-N KCs versus correct-history KCs; user macro average",
            "CFP": conditional_preference(scores, ctx.test, pairs), "top_n": metrics,
            "per_user": users, "exposure": items,
            "ranked_top50": [{"uid":int(u), "exercises":row[:50].tolist()} for u,row in zip(ctx.test,ranking)]}
