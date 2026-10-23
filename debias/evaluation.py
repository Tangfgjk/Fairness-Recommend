from pathlib import Path
import numpy as np
import torch

from .context import load_scores
from .io import TOP_NS, read_json, write_json, sha256, source_digest, ep_module
from .metrics import evaluate


def evaluate_file(ctx, arrays, manifest, scores_file, output, baseline=None):
    output = Path(output)
    contract = {"scores_sha256": sha256(scores_file), "context_id": manifest["context_id"],
                "pair_hash": manifest["pairs_sha256"], "source_hash": source_digest(),
                "baseline_sha256": sha256(baseline) if baseline is not None else None,
                "NOV": "KG4Ex-NOV-Set", "Ep_threshold": .5, "top_ns": list(TOP_NS)}
    completed = output / "evaluation_complete.json"
    if completed.exists():
        marker = read_json(completed)
        if marker["contract"] != contract:
            raise ValueError(f"Evaluation provenance changed: {output}; use a new run-id")
        if not (output / "metrics.json").exists():
            raise ValueError("Missing metrics despite completion marker")
        if marker["metrics_sha256"] != sha256(output / "metrics.json"):
            raise ValueError("Completed metrics changed")
        return read_json(output / "metrics.json")
    scores = load_scores(scores_file, ctx)
    base = load_scores(baseline, ctx) if baseline is not None else None
    result = evaluate(ctx, scores, arrays["evaluation"], baseline=base)
    ep = ep_module()
    context = ep.load_context(ctx.name, ctx.data_root, ctx.data_root / "_ep_readouts", torch.device("cpu"))
    ep_result = ep.evaluate_entry(context, Path(scores_file), output / "ep", list(TOP_NS), [.5])
    for row in ep_result["results"]:
        result["top_n"][str(row["top_n"])]["Ep_t0.5"] = row["mean"]
    ep_users = {(int(r["uid"].removeprefix("uid")), r["top_n"]): r["ep"]
                for r in read_json(output / "ep/ep_per_user.json")}
    for row in result["per_user"]:
        row["Ep_t0.5"] = ep_users[(row["uid"], row["N"])]
    if baseline is not None:
        result["paired_student_bootstrap"] = bootstrap_quality(result, Path(baseline).parent / "eval/metrics.json")
        result["exposure_bootstrap"] = bootstrap_exposure(ctx,scores,base)
    result["provenance"] = contract
    write_json(output / "metrics.json", result)
    write_json(completed, {"contract":contract,"metrics_sha256":sha256(output / "metrics.json")})
    return result


def bootstrap_quality(result, baseline_metrics, repeats=1000):
    """Paired students, NOT independent pairs or an across-seed significance claim."""
    if not baseline_metrics.exists():
        return {"status": "baseline evaluation not available"}
    reference = read_json(baseline_metrics)
    if "per_user" not in reference:
        return {"status":"archived baseline has no compatible per-user metrics; no quality CI reported"}
    output = {"status": "available", "replicates": repeats, "unit": "paired test student within seed", "difference_CI95": {}}
    rng = np.random.default_rng(20261022)
    for n in TOP_NS:
        a = {r["uid"]:r for r in result["per_user"] if r["N"] == n}
        b = {r["uid"]:r for r in reference["per_user"] if r["N"] == n}
        if set(a) != set(b):
            raise ValueError("Bootstrap cohorts differ")
        uids = sorted(a)
        ix = rng.integers(0,len(uids),(repeats,len(uids)))
        intervals = {}
        for metric in ("Ada", "KG4Ex-NOV-Set", "Ep_t0.5"):
            delta = np.array([a[u][metric]-b[u][metric] for u in uids])
            estimates = delta[ix].mean(1)
            intervals[metric] = {"mean_delta":float(delta.mean()),"lower":float(np.quantile(estimates,.025)),"upper":float(np.quantile(estimates,.975))}
        output["difference_CI95"][str(n)] = intervals
    return output


def write_score_matrix(path, ctx, scores):
    write_json(path, [{"uid": f"uid{u}", "scores": row.tolist()} for u, row in zip(ctx.test, scores)])


def frequency_correct(scores, frequency, strength):
    if strength < 0:
        raise ValueError("Score correction strength must be nonnegative")
    f = np.log1p(frequency)
    scale = f.max() - f.min()
    penalty = (f - f.min()) / scale if scale > 0 else np.zeros_like(f)
    return scores - strength * penalty[None, :]


def bootstrap_exposure(ctx, scores, baseline, repeats=200):
    from .metrics import gini, total_variation
    rank = np.argsort(-scores,axis=1,kind="stable")[:,:50]
    ref_rank = np.argsort(-baseline,axis=1,kind="stable")[:,:50]
    rule = np.array([ctx.rule(int(u)) for u in ctx.test])
    rng = np.random.default_rng(20261022)
    indices = rng.integers(0,len(rank),(repeats,len(rank)))
    fq = ctx.q / ctx.q.sum(1,keepdims=True)
    active = ctx.q.sum(0)>0
    output = {"unit":"paired test student; recompute exposure after each resample", "replicates":repeats,"difference_CI95":{}}
    def measures(selected,rule_counts):
        counts = np.bincount(selected.ravel(),minlength=len(ctx.q))
        kc = (counts @ fq)[active]
        return np.array([gini(counts),np.mean(counts>0),gini(kc),np.mean(kc>0),total_variation(counts,rule_counts)])
    for n in TOP_NS:
        deltas=[]
        for ix in indices:
            rule_counts=np.bincount(rule[ix,:n].ravel(),minlength=len(ctx.q))
            deltas.append(measures(rank[ix,:n],rule_counts)-measures(ref_rank[ix,:n],rule_counts))
        deltas=np.asarray(deltas)
        output["difference_CI95"][str(n)]={k:{"lower":float(np.quantile(deltas[:,i],.025)),"upper":float(np.quantile(deltas[:,i],.975))}
            for i,k in enumerate(("ItemGini","ItemCoverage","KCGini","KCCoverage","DED"))}
    return output
