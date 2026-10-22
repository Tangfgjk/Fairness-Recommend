"""Re-evaluate quality-bounded Post variants from saved V5 seed-2024 scores.

This script does not train or alter checkpoints. It creates separate, inspectable
PDF/CSV results for each prespecified variant on Eedi and Algebra2005.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from recalculate_v5_post_methods_20261020 import (
    POST_ROWS, RERANK, argument, replace_argument, source_stage,
)
from report_v5_seed2024_paper_metric_tables import export_dataset
from report_v5_seed2024_two_datasets import DATASETS, RUNS, TOP_KS, load_results


OUTPUT = RUNS / "论文格式文档统计" / "V5_2024_Post质量约束重排_20261021"
VARIANTS = {
    "soft_weak": {
        "method": "item_kc", "lambda_item": 0.08, "lambda_kc": 0.04,
        "beta_item": 0.02, "beta_kc": 0.01,
    },
    "swap_guarded": {
        "method": "quality_swap", "lambda_item": 0.20, "lambda_kc": 0.10,
        "beta_item": 0.05, "beta_kc": 0.02, "swap_fraction": 0.10,
        "swap_score_margin": 0.03, "swap_need_margin": 0.03,
    },
    "swap_relaxed": {
        "method": "quality_swap", "lambda_item": 0.20, "lambda_kc": 0.10,
        "beta_item": 0.05, "beta_kc": 0.02, "swap_fraction": 0.10,
        "swap_score_margin": 0.08, "swap_need_margin": 0.05,
    },
}
PARENTS = {
    "post_only": "baseline",
    "pre_item_kc_post": "pre_item_kc",
    "in_kppd_post": "in_kppd",
    "pre_in_kppd_post": "pre_in_kppd",
    "no_compgcn": None,
}
METRICS = (
    "Ada", "NOV", "Ep_t0.5", "ItemExposureGini", "KCExposureGini",
    "ItemCoverage", "KCCoverage", "LongTailItemExposureShare",
    "LongTailKCExposureShare",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_one(dataset: str, row: str, variant: str) -> tuple[dict, dict]:
    source, post_stage, eval_stage = source_stage(dataset, row)
    config = VARIANTS[variant]
    work = OUTPUT / variant / "data" / dataset / row
    work.mkdir(parents=True, exist_ok=True)
    scores = work / "scores.pkl"
    metrics = work / "eval" / "metrics.json"
    manifest_path = work / "manifest.json"

    rerank = list(post_stage["command"])
    rerank[1] = str(RERANK)
    replace_argument(rerank, "--method", str(config["method"]))
    replace_argument(rerank, "--output-file", str(scores))
    for key in ("lambda_item", "lambda_kc", "beta_item", "beta_kc"):
        replace_argument(rerank, "--" + key.replace("_", "-"), str(config[key]))
    for key in ("swap_fraction", "swap_score_margin", "swap_need_margin"):
        if key in config:
            rerank.extend(("--" + key.replace("_", "-"), str(config[key])))

    evaluation = list(eval_stage["command"])
    replace_argument(evaluation, "--scores-file", str(scores))
    replace_argument(evaluation, "--output-dir", str(metrics.parent))
    replace_argument(evaluation, "--model-name", f"{row}_{variant}")
    evaluation.extend(("--ep-thresholds", "0.5"))
    manifest = {
        "dataset": dataset, "row": row, "variant": variant,
        "source_scores": str(source), "source_sha256": sha256(source),
        "rerank_code_sha256": sha256(RERANK),
        "rerank_command": rerank, "evaluation_command": evaluation,
    }
    complete = metrics.is_file() and manifest_path.is_file() and (
        json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    )
    if not complete:
        for name, command in (("rerank", rerank), ("eval", evaluation)):
            print(f"{variant} / {dataset} / {row}: {name}", flush=True)
            with (work / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=RERANK.parent, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"{variant} / {dataset} / {row}: reuse", flush=True)

    payload = json.loads(metrics.read_text(encoding="utf-8-sig"))
    if payload.get("NOV") != payload.get("KG4Ex-NOV-Set"):
        raise ValueError(f"NOV definition changed: {metrics}")
    for n in TOP_KS:
        if any(str(n) not in payload.get(key, {}) for key in ("Ada", "NOV", "Ep_t0.5", "Fairness")):
            raise ValueError(f"Incomplete Top-{n} evaluation: {metrics}")
    return payload, manifest


def value(result: dict, metric: str, n: int) -> float:
    if metric in ("Ada", "NOV", "Ep_t0.5"):
        return float(result[metric][str(n)]["mean"])
    return float(result["Fairness"][str(n)][metric])


def write_variant(variant: str, original: dict, results: dict, manifests: list[dict]) -> list[dict]:
    folder = OUTPUT / variant
    fields = ["dataset", "post_row", "parent", "N"]
    for metric in METRICS:
        fields.extend((f"{metric}_before", f"{metric}_after", f"{metric}_change"))
    rows = []
    for dataset in DATASETS:
        results[(dataset, "main")] = results[(dataset, "pre_in_kppd_post")]
        for row in POST_ROWS:
            parent = PARENTS[row]
            if parent is None:
                continue
            for n in TOP_KS:
                record = {"dataset": dataset, "post_row": row, "parent": parent, "N": n}
                for metric in METRICS:
                    before = value(original[(dataset, parent)], metric, n)
                    after = value(results[(dataset, row)], metric, n)
                    record.update({f"{metric}_before": before, f"{metric}_after": after,
                                   f"{metric}_change": round(after - before, 6)})
                rows.append(record)
        pdf = folder / f"V5_2024_{dataset}_one_metric_per_table.pdf"
        export_dataset(results, dataset, pdf, post_method=variant)
        print(pdf, flush=True)
    with (folder / "Post_impact_all_N.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "source_manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    return [dict(variant=variant, **row) for row in rows if row["post_row"] == "pre_in_kppd_post"]


def main() -> None:
    original, _ = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = []
    for variant in VARIANTS:
        results = dict(original)
        manifests = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = {executor.submit(evaluate_one, dataset, row, variant): (dataset, row)
                    for dataset in DATASETS for row in POST_ROWS}
            for job in as_completed(jobs):
                dataset, row = jobs[job]
                results[(dataset, row)], manifest = job.result()
                manifests.append(manifest)
        comparison.extend(write_variant(variant, original, results, manifests))
    fields = list(comparison[0])
    with (OUTPUT / "three_variants_main_all_N.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)


if __name__ == "__main__":
    main()
