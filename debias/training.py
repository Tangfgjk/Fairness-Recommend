from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io import read_json, write_json, digest, source_digest, model_modules, sha256
from .objectives import FairObjective, combine_loss
from .prepare import baseline_folder

MODEL_ARGS = ("embedding_dim", "embedding_shape1", "hidden_size", "input_drop", "hidden_drop", "feat_drop", "use_bias")


def build_model(original, bundle, cfg):
    return original.SemanticConvE.from_feature_bundle(bundle, ablation_mode=cfg["ablation"],
                                                     **{k: cfg[k] for k in MODEL_ARGS})


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_checkpoint(path, payload):
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(ctx, arrays, manifest, out, method, seed, weight=.1, margin=.1, cuda="true"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    original = model_modules()
    cfg = read_json(baseline_folder(ctx.name, seed) / "config.json")
    cfg.update(data_path=str(ctx.graph), save_path=str(out), feature_dir=None, seed=seed, cuda=cuda)
    effective_weight = 0.0 if method == "E0" else weight
    if method not in ("E0", "E1", "E3", "E4") or weight < 0 or margin < 0:
        raise ValueError("Invalid training treatment")
    contract = {"method": method, "config": cfg, "weight": effective_weight, "margin": margin,
                "context_id": manifest["context_id"], "pairs_hash": manifest["pairs_sha256"],
                "source_hash": source_digest(), "aux_mode": "eval_behavior_with_autograd",
                "users_per_step": 32, "pairs_per_user": 4}
    contract_hash = digest(contract)
    config_path = out / "run_config.json"
    if config_path.exists() and read_json(config_path) != contract:
        raise ValueError(f"Resume configuration/source mismatch: {out}; choose a new run-id")
    write_json(config_path, contract)
    if (out / "training_complete.json").exists():
        complete = read_json(out / "training_complete.json")
        if complete["contract_hash"] != contract_hash:
            raise ValueError("Completion contract mismatch")
        if not (out / "best.pt").exists():
            raise ValueError("Completion marker without checkpoint")
        if complete["best_sha256"] != sha256(out / "best.pt"):
            raise ValueError("Completed checkpoint checksum mismatch")
        return
    device = original.resolve_device(cuda)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Follow the original initialization order exactly; lambda=0 draws no extra RNG.
    original.set_seed(seed, deterministic=cfg["deterministic"])
    bundle = original.load_semantic_feature_bundle(ctx.graph, None, device=device)
    triples = original.read_triples(ctx.graph / "triples.txt", bundle.entity2id, bundle.relation2id)
    dataset = original.TripleDataset(triples, bundle.id2relation, bundle.exercise_entity_ids.cpu().tolist(),
                                    original.build_positive_tails_by_hr(triples), cfg["negative_ratio"], seed)
    loader = DataLoader(dataset, batch_size=cfg["bs"], shuffle=True)
    model = build_model(original, bundle, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    objective = None
    if effective_weight:
        key = "E3" if method == "E3" else "E1"
        objective = FairObjective(arrays[key], bundle, seed, margin=margin,
                                  directions=arrays["E4_direction"] if method == "E4" else None)
    start, best, best_epoch, elapsed = 0, float("inf"), 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if ck.get("contract_hash") != contract_hash:
            raise ValueError("Checkpoint contract mismatch")
        model.load_state_dict(ck["model_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start, best, best_epoch, elapsed = ck["epoch"], ck["best_loss"], ck["best_epoch"], ck["training_seconds"]
        if objective is not None:
            objective.load_state_dict(ck["fair_rng"])
        restore_rng(ck["rng"])
    rec_id = bundle.relation2id["rec"]
    for epoch in range(start, cfg["epochs"]):
        model.train()
        began = time.perf_counter()
        totals = {"bce": 0., "fair": 0., "total": 0., "active_fraction": 0., "high_low_logit_gap": 0., "pair_logit_std": 0.}
        count, fair_steps = 0, 0
        for h, r, t, labels in tqdm(loader, desc=f"{ctx.name}/{method}/{seed} epoch {epoch+1}/{cfg['epochs']}"):
            h, r, t, labels = (x.to(device) for x in (h, r, t, labels))
            optimizer.zero_grad()
            pred = model.score_triples(h, r, t)
            bce = F.binary_cross_entropy(pred.view(-1), labels.view(-1))
            total, fair, stats = combine_loss(model, bce, objective, effective_weight,
                                             bool((r == rec_id).any()) if effective_weight else False, device)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            total.backward()
            # No clipping or new optimizer steps: preserve the original update rule.
            optimizer.step()
            totals["bce"] += float(bce.detach())
            totals["total"] += float(total.detach())
            if fair is not None:
                fair_steps += 1
                totals["fair"] += float(fair.detach())
                for key in ("active_fraction", "high_low_logit_gap", "pair_logit_std"):
                    totals[key] += stats[key]
            count += 1
        avg_bce = totals["bce"] / count
        is_best = avg_bce < best
        if is_best:
            best, best_epoch = avg_bce, epoch + 1
        elapsed += time.perf_counter() - began
        log = {k: v / (count if k in ("bce", "total") else max(1, fair_steps)) for k, v in totals.items()}
        log.update(epoch=epoch+1, batches=count, fair_steps=fair_steps, training_seconds=elapsed,
                   best_epoch=best_epoch, gradient_norm=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))))
        write_json(out / "epochs" / f"{epoch+1:03d}.json", log)
        print(json.dumps(log), flush=True)
        payload = {"model_version": cfg["model_version"], "ablation": cfg["ablation"], "config": cfg,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "epoch": epoch+1, "loss": avg_bce, "best_loss": best, "best_epoch": best_epoch,
                   "training_seconds": elapsed, "contract_hash": contract_hash, "rng": rng_state(),
                   "fair_rng": objective.state_dict() if objective is not None else None}
        if is_best:
            atomic_checkpoint(out / "best.pt", payload)
        atomic_checkpoint(out / "last.pt", payload)
    write_json(out / "config.json", cfg)
    write_json(out / "training_complete.json", {"contract_hash": contract_hash, "epochs": cfg["epochs"],
                                               "best_epoch": best_epoch, "training_seconds": elapsed,
                                               "best_sha256":sha256(out / "best.pt")})


def export_scores(ctx, checkpoint, output, cuda="true"):
    original = model_modules()
    device = original.resolve_device(cuda)
    bundle = original.load_semantic_feature_bundle(ctx.graph, None, device=device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(original, bundle, ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    # Preserve Q row order and the original score_tails matrix multiplication path.
    ids = torch.tensor([bundle.entity2id[f"ex{e}"] for e in range(len(ctx.q))], device=device)
    rows, logits = [], []
    with torch.no_grad():
        tails = model.entity_embedding(ids)
        for users in np.array_split(ctx.test, max(1, int(np.ceil(len(ctx.test)/64)))):
            heads = torch.tensor([bundle.entity2id[f"uid{u}"] for u in users], device=device)
            rel = torch.full_like(heads, bundle.relation2id["rec"])
            x = model.conve_transform(model.entity_embedding(heads), model.relation_embedding(rel))
            z = torch.mm(x, tails.T) + model.b[ids].unsqueeze(0)
            p = torch.sigmoid(z).cpu().numpy()
            logits.extend(z.cpu().numpy())
            rows.extend({"uid": f"uid{u}", "scores": s.tolist()} for u, s in zip(users, p))
    write_json(output, rows)
    np.save(Path(output).with_suffix(".logits.npy"), np.asarray(logits))
