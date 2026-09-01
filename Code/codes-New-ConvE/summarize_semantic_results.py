"""Summarize 2CKG4ER experiment outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List

from semantic_experiment_utils import (
    DEFAULT_TOP_KS,
    VALID_ABLATIONS,
    ablation_model_dir,
    default_runs_root,
    parse_ablation_list,
    parse_seed_list,
    parse_top_ks,
    read_json,
    write_json,
)

FAIRNESS_METRICS = [
    "ItemExposureGini",
    "KCExposureGini",
    "ItemCoverage",
    "KCCoverage",
    "LongTailItemExposureShare",
    "LongTailKCExposureShare",
    "HeadItemExposureShare",
    "HeadKCExposureShare",
]


def maybe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def sample_std(values: List[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def metric_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    if metric == "Ada" and "Ada" not in metrics and "ACC" in metrics:
        metric = "ACC"
    try:
        return float(metrics[metric][str(top_k)]["mean"])
    except Exception:
        return None


def fairness_metric_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    try:
        return float(metrics["Fairness"][str(top_k)][metric])
    except Exception:
        return None


def collect_seed(run_dir: Path, seed: int, top_ks: List[int], ablation: str = "2CKG4ER") -> Dict[str, Any] | None:
    seed_dir = run_dir / ablation_model_dir(ablation) / f"seed{seed}"
    eval_metrics = maybe_read_json(seed_dir / "eval" / "metrics.json")
    if not eval_metrics:
        return None
    train_metrics = maybe_read_json(seed_dir / "metrics.json")
    inference_metrics = maybe_read_json(seed_dir / "2ckg4er_inference.json") or maybe_read_json(seed_dir / "semantic_conve_inference.json")
    row: Dict[str, Any] = {
        "ablation": ablation,
        "model": ablation_model_dir(ablation),
        "seed": seed,
        "seed_dir": str(seed_dir),
        "training_seconds": train_metrics.get("training_seconds"),
        "inference_seconds": inference_metrics.get("inference_seconds"),
    }
    for top_k in top_ks:
        ada = metric_value(eval_metrics, "Ada", top_k)
        nov = metric_value(eval_metrics, "NOV", top_k)
        row[f"Ada@{top_k}"] = ada
        row[f"NOV@{top_k}"] = nov
        for metric in FAIRNESS_METRICS:
            row[f"{metric}@{top_k}"] = fairness_metric_value(eval_metrics, metric, top_k)
    return row


def collect_gate_rows(run_dir: Path, seeds: List[int], ablations: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ablation in ablations:
        model_name = ablation_model_dir(ablation)
        for seed in seeds:
            seed_dir = run_dir / model_name / f"seed{seed}"
            gate_payload = maybe_read_json(seed_dir / "gate_values.json")
            gate_values = gate_payload.get("gate_values", {})
            if not isinstance(gate_values, dict):
                continue
            for entity_type, feature_values in gate_values.items():
                if not isinstance(feature_values, dict):
                    continue
                for feature_type, value in feature_values.items():
                    try:
                        gate_value = float(value)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "ablation": ablation,
                            "model": model_name,
                            "seed": seed,
                            "seed_dir": str(seed_dir),
                            "entity_type": entity_type,
                            "feature_type": feature_type,
                            "gate_value": gate_value,
                        }
                    )
    return rows


def summarize_gate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], List[float]] = {}
    for row in rows:
        key = (
            str(row["ablation"]),
            str(row["model"]),
            str(row["entity_type"]),
            str(row["feature_type"]),
        )
        grouped.setdefault(key, []).append(float(row["gate_value"]))
    summaries = []
    for (ablation, model, entity_type, feature_type), values in sorted(grouped.items()):
        summaries.append(
            {
                "ablation": ablation,
                "model": model,
                "entity_type": entity_type,
                "feature_type": feature_type,
                "runs": len(values),
                "mean": sum(values) / len(values),
                "std": sample_std(values),
            }
        )
    return summaries


def summarize_rows(rows: List[Dict[str, Any]], top_ks: List[int]) -> Dict[str, Any]:
    fields = [f"Ada@{k}" for k in top_ks] + [f"NOV@{k}" for k in top_ks]
    fields.extend(f"{metric}@{k}" for metric in FAIRNESS_METRICS for k in top_ks)
    fields.extend(["training_seconds", "inference_seconds"])
    summary: Dict[str, Any] = {"runs": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_mean"] = sum(values) / len(values) if values else None
        summary[f"{field}_std"] = sample_std(values) if values else None
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_paper_table(path: Path, dataset: str, summaries: Dict[str, Dict[str, Any]], top_ks: List[int]) -> None:
    lines = [
        f"# {dataset} 2CKG4ER Summary",
        "",
    ]
    lines.extend(
        [
            "## Metrics By Recommendation Length",
            "",
        ]
    )
    metric_headers = ["Metric"] + [f"@{k} Mean" for k in top_ks] + [f"@{k} Std" for k in top_ks]
    for ablation, summary in summaries.items():
        lines.append(f"### {ablation_model_dir(ablation)}")
        lines.append("")
        lines.append("| " + " | ".join(metric_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(metric_headers)) + " |")
        for metric in ["Ada", "NOV", *FAIRNESS_METRICS]:
            metric_row = [metric]
            metric_row.extend(fmt(summary.get(f"{metric}@{k}_mean")) for k in top_ks)
            metric_row.extend(fmt(summary.get(f"{metric}@{k}_std")) for k in top_ks)
            lines.append("| " + " | ".join(metric_row) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_table_csv(path: Path, dataset: str, summaries: Dict[str, Dict[str, Any]], top_ks: List[int]) -> None:
    rows = []
    for ablation, summary in summaries.items():
        for metric in ("Ada", "NOV", *FAIRNESS_METRICS):
            row: Dict[str, Any] = {
                "Dataset": dataset,
                "Model": ablation_model_dir(ablation),
                "Ablation": ablation,
                "Runs": summary["runs"],
                "Metric": metric,
            }
            for top_k in top_ks:
                row[f"@{top_k} Mean"] = summary.get(f"{metric}@{top_k}_mean")
                row[f"@{top_k} Std"] = summary.get(f"{metric}@{top_k}_std")
            rows.append(row)
    if not rows:
        return
    write_csv(path, rows, list(rows[0].keys()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 2CKG4ER runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=default_runs_root())
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    parser.add_argument("--ablations", default="all")
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    ablations = parse_ablation_list(args.ablations)
    invalid_ablations = sorted(set(ablations) - set(VALID_ABLATIONS))
    if invalid_ablations:
        raise ValueError(f"Unknown ablations: {','.join(invalid_ablations)}")
    top_ks = parse_top_ks(args.top_ks)
    run_dir = args.runs_root / args.dataset / args.run_id
    output_dir = args.output_dir or (run_dir / "summaries")
    rows = []
    missing: Dict[str, List[int]] = {}
    for ablation in ablations:
        for seed in seeds:
            row = collect_seed(run_dir, seed, top_ks, ablation)
            if row is None:
                missing.setdefault(ablation, []).append(seed)
            else:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed 2CKG4ER metrics found under {run_dir}")

    summaries = {
        ablation: summarize_rows([row for row in rows if row["ablation"] == ablation], top_ks)
        for ablation in ablations
        if any(row["ablation"] == ablation for row in rows)
    }
    fields = ["ablation", "model", "seed", "seed_dir"]
    fields.extend(f"Ada@{k}" for k in top_ks)
    fields.extend(f"NOV@{k}" for k in top_ks)
    fields.extend(f"{metric}@{k}" for metric in FAIRNESS_METRICS for k in top_ks)
    fields.extend(["training_seconds", "inference_seconds"])
    write_csv(output_dir / "per_seed_metrics.csv", rows, fields)

    gate_rows = collect_gate_rows(run_dir, seeds, ablations)
    gate_summary_rows = summarize_gate_rows(gate_rows)
    if gate_rows:
        write_csv(
            output_dir / "gate_values_per_seed.csv",
            gate_rows,
            ["ablation", "model", "seed", "seed_dir", "entity_type", "feature_type", "gate_value"],
        )
        write_csv(
            output_dir / "gate_values_mean_std.csv",
            gate_summary_rows,
            ["ablation", "model", "entity_type", "feature_type", "runs", "mean", "std"],
        )

    summary_rows = []
    for ablation, summary in summaries.items():
        for key, value in summary.items():
            if key.endswith("_mean"):
                summary_rows.append(
                    {
                        "ablation": ablation,
                        "model": ablation_model_dir(ablation),
                        "metric": key.removesuffix("_mean"),
                        "mean": value,
                        "std": summary.get(key.replace("_mean", "_std")),
                    }
                )
    write_csv(output_dir / "mean_std_metrics.csv", summary_rows, ["ablation", "model", "metric", "mean", "std"])
    write_json(
        output_dir / "summary.json",
        {
            "dataset": args.dataset,
            "run_id": args.run_id,
            "missing_seeds": missing,
            "summaries": summaries,
            "rows": rows,
            "gate_summaries": gate_summary_rows,
        },
    )
    write_paper_table(output_dir / "paper_table.md", args.dataset, summaries, top_ks)
    write_paper_table_csv(output_dir / "paper_table.csv", args.dataset, summaries, top_ks)
    print(f"summary written to {output_dir}")


if __name__ == "__main__":
    main()
