"""Inspectable P0 diagnostics; distributions are not acceptance/retuning thresholds."""
import torch


def distribution(value):
    x = value.detach().double().flatten().cpu()
    if not len(x): return {"count": 0}
    if not torch.isfinite(x).all(): raise FloatingPointError("Nonfinite diagnostic")
    q = torch.quantile(x, torch.tensor([0.,.05,.5,.95,1.],dtype=x.dtype))
    return {"count":len(x), "mean":float(x.mean()), "std":float(x.std(unbiased=False)),
            **dict(zip(("min","p05","median","p95","max"),map(float,q)))}


def gradient_groups(model):
    result = {}
    for name, parameter in model.named_parameters():
        if name.startswith("layer."):
            group = name
        elif name.startswith("base.uid_feature_only_fusion"):
            group = "base.student_semantics"
        elif name.startswith("base.exercise_feature_only_fusion"):
            group = "base.exercise_semantics"
        elif name.startswith("base.shared_text_encoder"):
            group = "base.shared_text_encoder"
        elif any(name.startswith("base."+p) for p in
                 ("relation_feature_only_fusion","legacy_relation_feature_only_fusion","strength_projector")):
            group = "base.relation_semantics"
        else:
            group = "base.other"
        row = result.setdefault(group,{"registered":0,"with_gradient":0,"squared_norm":0.,"finite":True})
        row["registered"] += parameter.numel()
        if parameter.grad is not None:
            row["with_gradient"] += parameter.numel()
            row["finite"] = row["finite"] and bool(torch.isfinite(parameter.grad).all())
            row["squared_norm"] += float(parameter.grad.detach().double().square().sum())
    for row in result.values(): row["norm"] = row.pop("squared_norm")**.5
    return result


def encoding_diagnostics(encoded, entity_types):
    d = encoded.diagnostics
    if d is None: raise ValueError("Encoding must request diagnostics")
    h0 = d["h0"]; candidate = d["candidate"]; ids = d["ids"]
    h1 = encoded.entities[ids]
    out = {}
    for kind, type_id in (("uid",0),("ex",2),("kc",1)):
        select = (entity_types[ids] == type_id) & encoded.valid[ids]
        row = {"degree":distribution(encoded.graph.degree[ids][select]),
               "connected":int(d["connected"][select].sum())}
        for key, value in (("h0_norm",h0.norm(dim=1)),("h1_norm",h1.norm(dim=1)),
                           ("rho",candidate.norm(dim=1)/(h0.norm(dim=1)+1e-8)),
                           ("delta",(h1-h0).norm(dim=1))):
            row[key] = distribution(value[select])
        for key in ("forward","inverse","self","mlkc_pre_forward","mlkc_pre_inverse",
                    "exfr_pre_forward","exfr_pre_inverse","mlkc_forward","mlkc_inverse",
                    "exfr_forward","exfr_inverse"):
            if key in d: row[key+"_norm"] = distribution(d[key][select].norm(dim=1))
        for key in ("pre_norm","normalized","candidate"):
            row[key+"_values"] = distribution(d[key][select])
        if select.any():
            row["tanh_saturated_fraction"] = float((candidate[select].abs() >= .99).float().mean())
            neighbor = d["forward"]+d["inverse"]
            row["neighbor_self_ratio"] = distribution(neighbor[select].norm(dim=1)/(d["self"][select].norm(dim=1)+1e-8))
            row["forward_inverse_cosine"] = distribution(torch.nn.functional.cosine_similarity(
                d["forward"][select],d["inverse"][select]))
            row["neighbor_cancellation_ratio"] = distribution(neighbor[select].norm(dim=1)/(
                d["forward"][select].norm(dim=1)+d["inverse"][select].norm(dim=1)+1e-8))
            if "mlkc_forward" in d:
                ml = d["mlkc_forward"]+d["mlkc_inverse"]
                ex = d["exfr_forward"]+d["exfr_inverse"]
                row["mlkc_norm_share_not_causal"] = distribution(ml[select].norm(dim=1)/(
                    ml[select].norm(dim=1)+ex[select].norm(dim=1)+1e-8))
                row["family_cancellation_ratio"] = distribution((ml+ex)[select].norm(dim=1)/(
                    ml[select].norm(dim=1)+ex[select].norm(dim=1)+1e-8))
        out[kind] = row
    return out


def query_shift(base, query, item_ids):
    old = base.graph.degree[item_ids]; new = query.graph.degree[item_ids]
    delta = (query.entities[item_ids]-base.entities[item_ids]).norm(dim=1)
    positive = old>0; linked = new>old
    cosine = torch.nn.functional.cosine_similarity(query.entities[item_ids],base.entities[item_ids])
    return {"base_degree":distribution(old),"query_degree":distribution(new),
            "degree_increase":distribution(new-old),
            "relative_degree_increase_positive_base_only":distribution((new[positive]-old[positive])/old[positive]),
            "zero_to_connected":int(((old==0)&(new>0)).sum()),
            "changed_candidate_rows":int((delta>0).sum()),"candidate_delta":distribution(delta),
            "new_state_link_delta":distribution(delta[linked]),"no_new_state_link_delta":distribution(delta[~linked]),
            "candidate_cosine":distribution(cosine)}
