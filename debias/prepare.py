from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import numpy as np

from .context import load_context, load_scores
from .io import ROOT, VERSION, read_json, write_json, sha256, digest
from .metrics import evaluate
from .pair_matching import MatchConfig, build_pairs, exposure_audit

ARTIFACTS = ROOT / "artifacts"


def import_baseline(source, dataset):
    directory = "Algebra2005" if dataset == "algebra2005" else dataset
    run_id = "resplit_seed20261007_modelseed2024" if dataset == "Eedi" else "forgetting_rec50_seed2024"
    for seed in (2024, 2025, 2026):
        src = Path(source) / directory / "ablation" / run_id / "SemanticConvE_feature_only" / f"seed{seed}"
        dst = ARTIFACTS / "baseline_original" / dataset / f"seed{seed}"
        files = ("config.json", "best.pt", "semantic_conve_inference.json", "eval/uid_ex_scores.json", "eval/metrics.json")
        manifest = {}
        for rel in files:
            original, target = src / rel, dst / rel
            if not original.is_file():
                raise FileNotFoundError(original)
            h = sha256(original)
            if target.exists() and sha256(target) != h:
                raise ValueError(f"Existing baseline differs; refusing overwrite: {target}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            if sha256(target) != h:
                raise ValueError("Baseline copy checksum mismatch")
            manifest[rel] = {"source": str(original), "sha256": h}
        write_json(dst / "source_manifest.json", manifest)


def baseline_folder(dataset, seed):
    path = ARTIFACTS / "baseline_original" / dataset / f"seed{seed}"
    if not (path / "source_manifest.json").is_file():
        raise FileNotFoundError(f"Missing independent baseline copy: {path}; use prepare --import-source once")
    for rel, info in read_json(path / "source_manifest.json").items():
        if sha256(path / rel) != info["sha256"]:
            raise ValueError(f"Baseline checksum changed: {path / rel}")
    return path


def prepare(dataset, cfg=MatchConfig(), import_source=None):
    if import_source:
        import_baseline(import_source, dataset)
    ctx = load_context(dataset)
    folder = ARTIFACTS / "debias_context" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    contract = {"version": VERSION, "data_hash": ctx.fingerprint, "matching": asdict(cfg)}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["contract"] != contract:
            raise ValueError(f"Locked context differs: {folder}. Use a new artifact version; never silently replace evaluation pairs.")
        if sha256(folder / "pairs.npz") != manifest["pairs_sha256"]:
            raise ValueError("Pair archive changed")
        return ctx, folder
    arrays, audit = build_pairs(ctx, cfg)
    if not len(arrays["E1"]):
        raise ValueError("No training pairs, treatment cannot be prepared")
    np.savez_compressed(folder / "pairs.npz", **arrays)
    write_json(folder / "matching_audit.json", audit)
    write_json(folder / "frequency.json", ctx.frequency.tolist())
    diagnostics = {}
    for seed in (2024, 2025, 2026):
        source = baseline_folder(dataset, seed)
        config = read_json(source / "config.json")
        if config["ablation"] != "feature_only" or config["include_test_triples"]:
            raise ValueError("Not the formal original baseline")
        scores = load_scores(source / "eval/uid_ex_scores.json", ctx)
        result = evaluate(ctx, scores, arrays["evaluation"])
        # Reuse unchanged archived Ep only for this exact, hashed score artifact.
        original_metrics = read_json(source / "eval/metrics.json")
        for n, values in result["top_n"].items():
            values["Ep_t0.5"] = float(original_metrics["Ep_t0.5"][n]["mean"])
            if abs(values["Ada"] - float(original_metrics["ACC"][n]["mean"])) > 2e-6:
                raise ValueError(f"Ada regression failed: {dataset} seed{seed} N={n}")
        result["Ep_provenance"] = "unchanged archived score and original Ep_t0.5, source hashes verified; no new simulation"
        write_json(folder / f"baseline_seed{seed}.json", result)
        diagnostics[str(seed)] = exposure_audit(ctx, scores, arrays["E1"])
    write_json(folder / "high_exposure_coverage.json", diagnostics)
    write_json(manifest_path, {"contract": contract, "context_id": digest(contract),
                             "pairs_sha256": sha256(folder / "pairs.npz"), "source_files": ctx.files})
    return ctx, folder


def prepared(dataset):
    ctx = load_context(dataset)
    folder = ARTIFACTS / "debias_context" / dataset
    manifest = read_json(folder / "manifest.json")
    if manifest["contract"]["version"] != VERSION or manifest["contract"]["data_hash"] != ctx.fingerprint:
        raise ValueError("Prepared data does not match current sources")
    if sha256(folder / "pairs.npz") != manifest["pairs_sha256"]:
        raise ValueError("Pair archive checksum mismatch")
    with np.load(folder / "pairs.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return ctx, arrays, manifest
