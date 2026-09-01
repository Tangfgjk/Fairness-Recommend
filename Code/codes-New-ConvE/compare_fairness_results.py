"""Compare recommendation quality and fairness metrics across evaluated methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from experiment_utils import load_json


DEFAULT_KS = (10, 20, 50, 100)
QUALITY_METRICS = ("Ada", "NOV")
FAIRNESS_METRICS = (
    "ItemExposureGini",
    "KCExposureGini",
    "ItemCoverage",
    "KCCoverage",
    "LongTailItemExposureShare",
    "LongTailKCExposureShare",
    "HeadItemExposureShare",
    "HeadKCExposureShare",
)
LOWER_IS_BETTER = {"ItemExposureGini", "KCExposureGini", "HeadItemExposureShare", "HeadKCExposureShare"}
METHOD_ORDER = {
    "baseline": 0,
    "item_only": 10,
    "kc_only": 20,
    "item_kc_rerank": 30,
    "stronger_item": 40,
    "stronger_kc": 50,
    "quota_mid": 60,
    "quota_ratio_10pct": 70,
    "quota_ratio_15pct": 80,
    "quota_ratio_hybrid": 90,
}
EVAL_DIR_ALIASES = {
    "eval": "baseline",
    "eval_kc_rerank": "kc_only",
    "eval_item_kc_stronger_item": "stronger_item",
}
METHOD_DIR_ALIASES = {
    "baseline": "eval",
    "kc_only": "eval_kc_rerank",
    "stronger_item": "eval_item_kc_stronger_item",
}


def parse_int_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def method_name_from_eval_dir(eval_dir: Path) -> str:
    if eval_dir.name in EVAL_DIR_ALIASES:
        return EVAL_DIR_ALIASES[eval_dir.name]
    if eval_dir.name.startswith("eval_"):
        return eval_dir.name[len("eval_") :]
    return eval_dir.name


def method_sort_key(method: str) -> tuple[int, str]:
    return (METHOD_ORDER.get(method, 1000), method)


def discover_eval_dirs(run_dir: Path, methods: Sequence[str] | None = None) -> List[Path]:
    if methods:
        eval_dirs = []
        for method in methods:
            dirname = METHOD_DIR_ALIASES.get(method, "eval" if method == "baseline" else f"eval_{method}")
            eval_dir = run_dir / dirname
            if eval_dir.exists():
                eval_dirs.append(eval_dir)
        return sorted(eval_dirs, key=lambda path: method_sort_key(method_name_from_eval_dir(path)))

    eval_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("eval") and (path / "metrics.json").exists() and (path / "fairness_metrics.json").exists()
    ]
    return sorted(eval_dirs, key=lambda path: method_sort_key(method_name_from_eval_dir(path)))


def quality_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    source_metric = "ACC" if metric == "Ada" and "Ada" not in metrics and "ACC" in metrics else metric
    try:
        return float(metrics[source_metric][str(top_k)]["mean"])
    except Exception:
        return None


def quality_std(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    source_metric = "ACC" if metric == "Ada" and "Ada" not in metrics and "ACC" in metrics else metric
    try:
        return float(metrics[source_metric][str(top_k)]["std"])
    except Exception:
        return None


def fairness_value(metrics: Dict[str, Any], metric: str, top_k: int) -> float | None:
    try:
        return float(metrics["top_k"][str(top_k)][metric])
    except Exception:
        return None


def collect_rows(run_dir: Path, top_ks: Sequence[int], methods: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for eval_dir in discover_eval_dirs(run_dir, methods):
        metrics = load_json(eval_dir / "metrics.json", default={}) or {}
        fairness = load_json(eval_dir / "fairness_metrics.json", default={}) or {}
        method = method_name_from_eval_dir(eval_dir)
        for top_k in top_ks:
            row: Dict[str, Any] = {
                "method": method,
                "eval_dir": str(eval_dir),
                "dataset": metrics.get("dataset"),
                "model": metrics.get("model"),
                "seed": metrics.get("seed"),
                "K": int(top_k),
            }
            for metric in QUALITY_METRICS:
                row[metric] = quality_value(metrics, metric, top_k)
                row[f"{metric}_std"] = quality_std(metrics, metric, top_k)
            for metric in FAIRNESS_METRICS:
                row[metric] = fairness_value(fairness, metric, top_k)
            rows.append(row)
    return rows


def add_baseline_deltas(rows: Sequence[Dict[str, Any]], baseline_method: str = "baseline") -> List[Dict[str, Any]]:
    baseline_by_k = {int(row["K"]): row for row in rows if row.get("method") == baseline_method}
    output: List[Dict[str, Any]] = []
    delta_metrics = [*QUALITY_METRICS, *FAIRNESS_METRICS]
    for row in rows:
        enriched = dict(row)
        baseline = baseline_by_k.get(int(row["K"]))
        for metric in delta_metrics:
            value = row.get(metric)
            base_value = baseline.get(metric) if baseline else None
            delta_key = f"Delta_{metric}"
            if value is None or base_value is None:
                enriched[delta_key] = None
            else:
                enriched[delta_key] = float(value) - float(base_value)
        output.append(enriched)
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def delta_fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):+.6f}"


def method_best_note(rows: Sequence[Dict[str, Any]], top_k: int, metric: str) -> str:
    values = [(row["method"], row.get(metric)) for row in rows if int(row["K"]) == top_k and row.get(metric) is not None]
    if not values:
        return ""
    reverse = metric not in LOWER_IS_BETTER
    method, value = sorted(values, key=lambda item: float(item[1]), reverse=reverse)[0]
    direction = "lowest" if metric in LOWER_IS_BETTER else "highest"
    return f"- {metric}@{top_k}: {method} is {direction} ({float(value):.6f})"


def write_markdown(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    dataset_name: str | None,
    seed: int | None,
    top_ks: Sequence[int],
) -> None:
    title_parts = [dataset_name or "Experiment"]
    if seed is not None:
        title_parts.append(f"seed{seed}")
    title = " ".join(title_parts)

    lines = [
        f"# {title} Fairness Comparison",
        "",
        "## Compared Metrics",
        "",
        "- Quality: Ada, NOV",
        "- Exercise fairness: ItemExposureGini, ItemCoverage, LongTailItemExposureShare",
        "- Knowledge fairness: KCExposureGini, KCCoverage, LongTailKCExposureShare",
        "",
    ]

    by_k = {top_k: [row for row in rows if int(row["K"]) == top_k] for top_k in top_ks}
    for top_k, k_rows in by_k.items():
        if not k_rows:
            continue
        lines.append(f"## K={top_k}")
        lines.append("")
        headers = [
            "method",
            "Ada",
            "NOV",
            "ItemGini",
            "KCGini",
            "ItemCov",
            "KCCov",
            "LTItemShare",
            "LTKCShare",
        ]
        fields = [
            "method",
            "Ada",
            "NOV",
            "ItemExposureGini",
            "KCExposureGini",
            "ItemCoverage",
            "KCCoverage",
            "LongTailItemExposureShare",
            "LongTailKCExposureShare",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in k_rows:
            lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
        lines.append("")

    lines.extend(["## Delta vs Baseline", ""])
    delta_fields = [
        "method",
        "K",
        "Delta_Ada",
        "Delta_NOV",
        "Delta_ItemCoverage",
        "Delta_KCCoverage",
        "Delta_LongTailItemExposureShare",
        "Delta_LongTailKCExposureShare",
        "Delta_ItemExposureGini",
        "Delta_KCExposureGini",
    ]
    delta_headers = [
        "method",
        "K",
        "Ada",
        "NOV",
        "ItemCov",
        "KCCov",
        "LTItemShare",
        "LTKCShare",
        "ItemGini",
        "KCGini",
    ]
    lines.append("| " + " | ".join(delta_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(delta_headers)) + " |")
    for row in delta_rows:
        if row.get("method") == "baseline":
            continue
        values = [fmt(row.get("method")), fmt(row.get("K"))]
        values.extend(delta_fmt(row.get(field)) for field in delta_fields[2:])
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")

    lines.extend(["## Quick Best-Value Checks", ""])
    for top_k in top_ks:
        for metric in ("Ada", "ItemCoverage", "KCCoverage", "LongTailItemExposureShare", "LongTailKCExposureShare", "KCExposureGini"):
            note = method_best_note(rows, top_k, metric)
            if note:
                lines.append(note)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison_fields() -> List[str]:
    return [
        "method",
        "eval_dir",
        "dataset",
        "model",
        "seed",
        "K",
        "Ada",
        "Ada_std",
        "NOV",
        "NOV_std",
        *FAIRNESS_METRICS,
    ]


def delta_fields() -> List[str]:
    base_fields = comparison_fields()
    return [*base_fields, *(f"Delta_{metric}" for metric in [*QUALITY_METRICS, *FAIRNESS_METRICS])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluated 2CKG4ER quality and fairness results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ks", default=",".join(str(value) for value in DEFAULT_KS))
    parser.add_argument("--methods", default=None, help="Comma-separated method names. Use baseline for eval/.")
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_ks = parse_int_list(args.ks)
    methods = None
    if args.methods:
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    output_dir = args.output_dir or args.run_dir / "comparison"

    rows = collect_rows(args.run_dir, top_ks=top_ks, methods=methods)
    delta_rows = add_baseline_deltas(rows, baseline_method=args.baseline_method)

    write_csv(output_dir / "fairness_comparison.csv", rows, comparison_fields())
    write_csv(output_dir / "fairness_comparison_delta.csv", delta_rows, delta_fields())
    write_markdown(
        output_dir / "fairness_comparison.md",
        rows=rows,
        delta_rows=delta_rows,
        dataset_name=args.dataset_name,
        seed=args.seed,
        top_ks=top_ks,
    )
    print(f"comparison written to {output_dir}")


if __name__ == "__main__":
    main()
