"""Re-evaluate six Post rerank methods from saved V5 scores, then rebuild PDFs.

No model training occurs. Each method gets two paper tables under its own
subdirectory of the existing V5_2024_Eedi_Algebra2005_20261020 report folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from report_v5_seed2024_paper_metric_tables import export_dataset
from report_v5_seed2024_two_datasets import DATASETS, RUNS, TOP_KS, load_results, source_path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = RUNS / "论文格式文档统计" / "V5_2024_Eedi_Algebra2005_20261020"
RERANK = ROOT / "codes-New-ConvE" / "fairness_rerank.py"
POST_ROWS = ("post_only", "pre_item_kc_post", "in_kppd_post", "pre_in_kppd_post", "no_compgcn")
METHODS = ("item", "kc", "item_kc", "quota_strict", "quota_ratio", "quota_ratio_hybrid")
WEIGHTS = {
    "item": (0.8, 0.0, 0.3, 0.0),
    "kc": (0.0, 0.3, 0.0, 0.1),
    # Match the hybrid's soft weights so this isolates the hard quota effect.
    "item_kc": (0.8, 0.3, 0.3, 0.1),
    "quota_strict": (0.8, 0.3, 0.3, 0.1),
    "quota_ratio": (0.8, 0.3, 0.3, 0.1),
    "quota_ratio_hybrid": (0.8, 0.3, 0.3, 0.1),
}


def argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def replace_argument(command: list[str], name: str, value: str) -> None:
    command[command.index(name) + 1] = value


def source_stage(dataset: str, row: str) -> tuple[Path, dict, dict]:
    original_metrics = source_path(dataset, row)
    seed_dir = original_metrics.parent.parent
    post = json.loads((seed_dir / "post_stage.json").read_text(encoding="utf-8"))
    evaluation = json.loads((seed_dir / "eval_stage.json").read_text(encoding="utf-8"))
    if post["status"] != "completed" or evaluation["status"] != "completed":
        raise ValueError(f"Original Post is incomplete: {seed_dir}")
    source = Path(argument(post["command"], "--scores-file"))
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, post, evaluation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=RERANK.parent, check=True, stdout=stream, stderr=subprocess.STDOUT)


def recompute(dataset: str, row: str, method: str) -> tuple[dict, dict]:
    source, post, evaluation = source_stage(dataset, row)
    if method == "quota_ratio_hybrid":
        existing = source_path(dataset, row)
        return json.loads(existing.read_text(encoding="utf-8-sig")), {
            "dataset": dataset, "row": row, "method": method, "reused_existing_metrics": str(existing),
            "source_scores": str(source), "source_sha256": sha256(source),
        }

    work = OUTPUT / method / "data" / dataset / row
    work.mkdir(parents=True, exist_ok=True)
    scores = work / "scores.pkl"
    metrics = work / "eval" / "metrics.json"
    manifest_path = work / "manifest.json"
    rerank_command = list(post["command"])
    rerank_command[1] = str(RERANK)
    replace_argument(rerank_command, "--method", method)
    replace_argument(rerank_command, "--output-file", str(scores))
    lambda_item, lambda_kc, beta_item, beta_kc = WEIGHTS[method]
    for flag, value in (("--lambda-item", lambda_item), ("--lambda-kc", lambda_kc),
                        ("--beta-item", beta_item), ("--beta-kc", beta_kc)):
        replace_argument(rerank_command, flag, str(value))
    if method == "quota_strict":
        rerank_command += ["--quota-top-k", "10", "--min-long-tail-items", "1", "--min-kc-coverage", "10"]

    eval_command = list(evaluation["command"])
    replace_argument(eval_command, "--scores-file", str(scores))
    replace_argument(eval_command, "--output-dir", str(metrics.parent))
    replace_argument(eval_command, "--model-name", f"{row}_{method}")
    eval_command += ["--ep-thresholds", "0.5"]
    manifest = {
        "dataset": dataset, "row": row, "method": method,
        "source_scores": str(source), "source_sha256": sha256(source),
        "rerank_command": rerank_command, "eval_command": eval_command,
    }
    complete = metrics.is_file() and manifest_path.is_file() and (
        json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    )
    if not complete:
        print(f"  {dataset} / {row} / {method}: rerank", flush=True)
        run_logged(rerank_command, work / "rerank.log")
        print(f"  {dataset} / {row} / {method}: evaluate", flush=True)
        run_logged(eval_command, work / "eval.log")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"  {dataset} / {row} / {method}: reuse", flush=True)
    payload = json.loads(metrics.read_text(encoding="utf-8-sig"))
    if payload.get("NOV") != payload.get("KG4Ex-NOV-Set"):
        raise ValueError(f"Formal NOV definition changed: {metrics}")
    for top_k in TOP_KS:
        for key in ("Ada", "NOV", "Ep_t0.5"):
            if str(top_k) not in payload.get(key, {}):
                raise ValueError(f"Incomplete {key}@{top_k}: {metrics}")
    return payload, manifest


def write_summary(method: str, results: dict, manifests: list[dict]) -> None:
    folder = OUTPUT / method
    folder.mkdir(parents=True, exist_ok=True)
    fields = ("dataset", "method", "post_row", "N", "parent_Ada", "post_Ada", "Ada_change",
              "parent_NOV", "post_NOV", "parent_Ep_0.5", "post_Ep_0.5")
    with (folder / "Post_impact_all_N.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for row in POST_ROWS:
                parent = {
                    "post_only": "baseline", "pre_item_kc_post": "pre_item_kc",
                    "in_kppd_post": "in_kppd", "pre_in_kppd_post": "pre_in_kppd",
                    "no_compgcn": None,
                }[row]
                if parent is None:
                    continue
                old = results[(dataset, parent)]
                new = results[(dataset, row)]
                for n in TOP_KS:
                    key = str(n)
                    prior = old["Ada"][key]["mean"]
                    current = new["Ada"][key]["mean"]
                    writer.writerow({
                        "dataset": dataset, "method": method, "post_row": row, "N": n,
                        "parent_Ada": prior, "post_Ada": current,
                        "Ada_change": round(current - prior, 6),
                        "parent_NOV": old["NOV"][key]["mean"], "post_NOV": new["NOV"][key]["mean"],
                        "parent_Ep_0.5": old["Ep_t0.5"][key]["mean"],
                        "post_Ep_0.5": new["Ep_t0.5"][key]["mean"],
                    })
    (folder / "source_manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    for dataset in DATASETS:
        output = folder / f"V5_2024_{dataset}_one_metric_per_table.pdf"
        export_dataset(results, dataset, output, post_method=method)
        print(output, flush=True)


def main() -> None:
    original, _ = load_results()
    comparison = []
    for method in METHODS:
        print(f"Post method: {method}", flush=True)
        results = dict(original)
        manifests = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(recompute, dataset, row, method): (dataset, row)
                       for dataset in DATASETS for row in POST_ROWS}
            for future in as_completed(futures):
                dataset, row = futures[future]
                payload, manifest = future.result()
                results[(dataset, row)] = payload
                manifests.append(manifest)
        for dataset in DATASETS:
            results[(dataset, "main")] = results[(dataset, "pre_in_kppd_post")]
            for n in TOP_KS:
                key = str(n)
                parent = original[(dataset, "pre_in_kppd")]
                current = results[(dataset, "pre_in_kppd_post")]
                comparison.append({
                    "dataset": dataset, "post_method": method, "N": n,
                    "Ada_before_Post": parent["Ada"][key]["mean"],
                    "Ada_after_Post": current["Ada"][key]["mean"],
                    "NOV_after_Post": current["NOV"][key]["mean"],
                    "Ep_0.5_after_Post": current["Ep_t0.5"][key]["mean"],
                    "ItemExposureGini_after_Post": current["Fairness"][key]["ItemExposureGini"],
                    "KCCoverage_after_Post": current["Fairness"][key]["KCCoverage"],
                })
        write_summary(method, results, manifests)
    with (OUTPUT / "six_Post_methods_main_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


if __name__ == "__main__":
    main()
