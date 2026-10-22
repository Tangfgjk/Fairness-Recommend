"""Evaluate Core/Core+Need/Raw/Debiased from one checkpoint, without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from audit_recommendation_personalization import summarize
from fairness_context import load_item_popularity, read_q_matrix
from fairness_rerank import load_uid_ex_scores
from semantic_experiment_utils import code_dir, write_json


def make_mode_logits(components: dict, beta_need: float) -> tuple[list[str], dict[str, np.ndarray]]:
    """Validate identity/order/shape before combining any score components."""
    names = ("core", "need", "raw", "debiased")
    uids = [u for u, _ in components["core"]]
    if not uids or len(set(uids)) != len(uids):
        raise ValueError("Missing or duplicate students in component files")
    arrays = {}
    for name in names:
        rows = components[name]
        if [u for u, _ in rows] != uids:
            raise ValueError(f"Mismatched student identities/order in {name}")
        arrays[name] = np.asarray([v for _, v in rows], dtype=np.float32)
        if arrays[name].ndim != 2 or not arrays[name].shape[1] or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid {name} components")
        if arrays[name].shape != arrays["core"].shape:
            raise ValueError(f"Mismatched candidate dimensions in {name}")
    return uids, {"core": arrays["core"], "core_need": arrays["core"] + float(beta_need)*arrays["need"],
                  "raw": arrays["raw"], "debiased": arrays["debiased"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--component-dir", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--top-ks", default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--ep-thresholds", default="0.5")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    if args.checkpoint_dir:
        components_dir = args.output_dir / "components"
        if components_dir.is_relative_to(args.checkpoint_dir.resolve()):
            raise ValueError("Use an output directory outside the read-only checkpoint directory")
        command = [sys.executable, str(code_dir()/"test_semantic_conve.py"),
                   "--data-path", str(args.data_dir.resolve()), "--dataset-name", args.dataset,
                   "--save-path", str(args.checkpoint_dir.resolve()), "--checkpoint", args.checkpoint,
                   "--artifact-dir", str(components_dir), "--cuda", args.cuda,
                   "--score-mode", "raw", "--save-logit-components", "--forgetting-score-weight", "0.0"]
        commands.append(command)
        with (args.output_dir/"inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=code_dir(), stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        components_dir = args.component_dir.resolve()
    meta = json.loads((components_dir/"2ckg4er_inference.json").read_text(encoding="utf-8"))
    if (meta.get("kppd_context_version") != 2 or meta.get("dataset") != args.dataset
            or Path(meta.get("data_dir", "")).resolve() != args.data_dir.resolve()
            or not meta.get("popularity_debias_config")):
        raise ValueError("Components must come from corrected KPPD inference on the same dataset/graph; use --checkpoint-dir to regenerate")
    paths = {name: components_dir/f"2CKG4ER_uid_ex_logits_{name}.pkl" for name in ("core", "need", "raw", "debiased")}
    components = {name: load_uid_ex_scores(path) for name, path in paths.items()}
    config = meta["popularity_debias_config"]
    uids, modes = make_mode_logits(components, config["beta_need"])
    expected_equal = config["lambda_global"] == 1. and config["lambda_personal"] == 1.
    equal = bool(np.allclose(modes["core_need"], modes["debiased"], rtol=1e-5, atol=1e-6))
    consistency = {"expected_equivalence": expected_equal, "within_tolerance": equal,
                   "max_logit_difference": float(np.max(np.abs(modes["core_need"]-modes["debiased"]))),
                   "rtol": 1e-5, "atol": 1e-6}
    write_json(args.output_dir/"score_consistency.json", consistency)
    if expected_equal and not equal:
        raise ValueError("Core+Need and Debiased disagree at default subtraction coefficients")
    top_ks = [int(n) for n in args.top_ks.split(",")]
    count = len(read_q_matrix(args.data_dir/"Q.txt"))
    if modes["core"].shape[1] != count:
        raise ValueError("Component candidates do not match Q.txt")
    popularity, pop_meta = load_item_popularity(args.data_dir, count, source="train_interactions", aggregation="unique_users")
    summary = []
    for mode, logits in modes.items():
        output = args.output_dir/mode
        output.mkdir(exist_ok=True)
        scores = torch.sigmoid(torch.from_numpy(logits)).tolist()
        path = output/"scores.pkl"
        with path.open("wb") as stream:
            pickle.dump(list(zip(uids, scores)), stream)
        command = [sys.executable, str(code_dir()/"evaluate_recommendations.py"),
                   "--data-dir", str(args.data_dir.resolve()), "--dataset-name", args.dataset,
                   "--scores-file", str(path), "--output-dir", str(output/"eval"),
                   "--model-name", f"KPPD_{mode}", "--top-ks", args.top_ks,
                   "--ep-thresholds", args.ep_thresholds, "--fairness-popularity-source", "train_interactions",
                   "--fairness-popularity-aggregation", "unique_users"]
        commands.append(command)
        if meta.get("seed") is not None:
            command.extend(["--seed", str(meta["seed"])])
        with (output/"eval.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=code_dir(), stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = summarize({u: np.asarray(v) for u, v in zip(uids, scores)}, popularity, top_ks)
        audit["popularity_source"] = pop_meta
        write_json(output/"personalization_audit.json", audit)
        metrics = json.loads((output/"eval/metrics.json").read_text(encoding="utf-8"))
        for metric in ("Ada", "NOV", *(f"Ep_t{float(t):.1f}" for t in args.ep_thresholds.split(","))):
            for n in top_ks:
                summary.append({"mode": mode, "metric": metric, "top_k": n, "mean": metrics[metric][str(n)]["mean"]})
        for n in top_ks:
            for metric, value in metrics["Fairness"][str(n)].items():
                if isinstance(value, (int, float)):
                    summary.append({"mode": mode, "metric": metric, "top_k": n, "mean": value})
    with (args.output_dir/"score_modes_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "metric", "top_k", "mean"])
        writer.writeheader()
        writer.writerows(summary)
    write_json(args.output_dir/"manifest.json", {"dataset": args.dataset, "seed": meta.get("seed"), "checkpoint": meta["checkpoint"],
               "checkpoint_sha256": meta.get("checkpoint_sha256"), "kppd_context_version": 2,
               "context_repaired": meta.get("kppd_context_repaired"), "score_consistency": consistency,
               "components_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()},
               "commands": commands, "training_performed": False})
    print(f"KPPD four-score evaluation -> {args.output_dir}")


if __name__ == "__main__":
    main()
