"""Build paper-ready comparison and ablation tables from Data_Fin_New runs.

The ranking is based on means rounded to the same precision shown in the table.
All models tied at the largest displayed mean are bold; models tied at the
second-largest distinct displayed mean are underlined. Missing models and
seeds remain blank until later runs are added. Higher is better for ACC, NOV,
and all simulated Ep variants.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
TOP_KS = tuple(range(5, 51, 5))
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
METRICS = ("ACC", "NOV", "Ep_t0.4", "Ep_t0.5", "Ep_t0.6", "Ep_t0.7")
PANELS = (
    ("ACC", "Adaptability performance (Ada)"),
    ("NOV", "Novelty performance (NOV)"),
    ("Ep_t0.4", r"Simulated mastery gain (Ep), $\tau=0.4$"),
    ("Ep_t0.5", r"Simulated mastery gain (Ep), $\tau=0.5$"),
    ("Ep_t0.6", r"Simulated mastery gain (Ep), $\tau=0.6$"),
    ("Ep_t0.7", r"Simulated mastery gain (Ep), $\tau=0.7$"),
)
EXPECTED_SEEDS = (2024, 2025, 2026)
MAIN_MODELS = (
    "SemanticConvE_id_only",
    "SemanticConvE_feature_only",
    "SemanticConvE_feature_only_relation_id",
    "SemanticConvE_feature_only_learner_id",
    "SemanticConvE_feature_only_exercise_id",
    "SemanticConvE_feature_only_no_mastery",
    "SemanticConvE_feature_only_no_forgetting",
)
COMPARISON_MODELS = (
    "EB-CF", "SB-CF", "CBF", "RotatE", "DistMult", "ComplEx",
    "TransE", "TransE-adv", "SemanticConvE_feature_only",
)
ABLATION_MODELS = (
    "SemanticConvE_feature_only",
    "SemanticConvE_id_only",
    "SemanticConvE_feature_only_relation_id",
    "SemanticConvE_feature_only_learner_id",
    "SemanticConvE_feature_only_exercise_id",
    "SemanticConvE_feature_only_no_mastery",
    "SemanticConvE_feature_only_no_forgetting",
)
LABELS = {
    "SemanticConvE_feature_only": "2CKG4ER",
    "SemanticConvE_id_only": "ID only",
    "SemanticConvE_feature_only_relation_id": "w/o relation ID",
    "SemanticConvE_feature_only_learner_id": "w/o learner ID",
    "SemanticConvE_feature_only_exercise_id": "w/o exercise ID",
    "SemanticConvE_feature_only_no_mastery": "w/o mastery",
    "SemanticConvE_feature_only_no_forgetting": "w/o forgetting",
}
DATASET_LABELS = {
    "Eedi": "Eedi-sub",
    "algebra2005": "Algebra2005",
    "XES3G5M-sub-small": "XES3G5M-sub-small",
}


def checked_source(run_dir: Path, dataset: str, kind: str) -> None:
    name = "run_config.json" if kind == "main" else "manifest.json"
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind} run manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset:
        raise ValueError(f"Dataset mismatch in {path}")
    source = payload.get("base_graph_path") if kind == "main" else payload.get("data_dir")
    expected = (ROOT / "Data_Fin_New" / dataset / "er_graph").resolve()
    if source is None or Path(source).resolve() != expected:
        raise ValueError(f"Run did not use the expected Data_Fin_New graph: {path}")


def load_run(run_dir: Path, dataset: str, kind: str) -> dict[str, dict]:
    expected = set(MAIN_MODELS if kind == "main" else COMPARISON_MODELS[:-1])
    rows: dict[tuple[str, int | None], dict] = {}
    paths = sorted(run_dir.glob("**/eval/metrics.json"))
    if not paths:
        return rows
    checked_source(run_dir, dataset, kind)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if model not in expected:
            raise ValueError(f"Unexpected model {model!r} in {path}")
        if payload.get("dataset") != dataset:
            raise ValueError(f"Dataset mismatch in {path}")
        if payload.get("ep_thresholds") != list(THRESHOLDS):
            raise ValueError(f"Ep thresholds differ in {path}")
        seed = payload.get("seed")
        if model in ("EB-CF", "SB-CF", "CBF"):
            if seed is not None:
                raise ValueError(f"Deterministic baseline should have no seed: {path}")
        elif seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected or missing seed for {model}: {path}")
        key = (model, seed)
        if key in rows:
            raise ValueError(f"Duplicate result for {model}/seed{seed}")
        for metric in METRICS:
            values = payload.get(metric)
            if not isinstance(values, dict):
                raise ValueError(f"Missing {metric} in {path}")
            for top_k in TOP_KS:
                item = values.get(str(top_k))
                value = item.get("mean") if isinstance(item, dict) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Missing or invalid {metric}@{top_k} in {path}")
        rows[key] = payload
    return rows


def aggregate(main: dict, comparison: dict) -> dict[str, dict]:
    combined = dict(comparison)
    combined.update(main)
    result = {}
    for model in set(COMPARISON_MODELS) | set(ABLATION_MODELS):
        expected_seeds = (None,) if model in ("EB-CF", "SB-CF", "CBF") else EXPECTED_SEEDS
        available_seeds = [seed for seed in expected_seeds if (model, seed) in combined]
        items = [combined[(model, seed)] for seed in available_seeds]
        metrics = {}
        for metric in METRICS:
            metrics[metric] = {}
            for top_k in TOP_KS:
                values = [float(item[metric][str(top_k)]["mean"]) for item in items]
                metrics[metric][str(top_k)] = {
                    "mean": mean(values) if values else None,
                    "std": stdev(values) if len(values) > 1 else None,
                }
        result[model] = {"available_seeds": available_seeds,
                         "expected_seeds": list(expected_seeds),
                         "complete": len(available_seeds) == len(expected_seeds),
                         "metrics": metrics}
    return result


def ranked_cells(aggregate_data: dict, models: tuple[str, ...], metric: str,
                 top_k: int, decimals: int) -> list[dict]:
    values = [aggregate_data[model]["metrics"][metric][str(top_k)] for model in models]
    shown = [round(item["mean"], decimals) if item["mean"] is not None else None
             for item in values]
    distinct = sorted({value for value in shown if value is not None}, reverse=True)
    best = distinct[0] if distinct else None
    second = distinct[1] if len(distinct) > 1 else None
    return [{"model": model, "mean": value["mean"], "std": value["std"],
             "rank": "missing" if display is None else
                     "best" if display == best else "second" if display == second else "other"}
            for model, value, display in zip(models, values, shown)]


def formatted_cell(cell: dict, decimals: int, latex: bool) -> str:
    if cell["mean"] is None:
        return ""
    avg = f"{cell['mean']:.{decimals}f}"
    std = cell["std"]
    value = avg if std is None else (f"{avg} $\\pm$ {std:.{decimals}f}" if latex
                                     else f"{avg} ± {std:.{decimals}f}")
    if cell["rank"] == "best":
        return rf"\textbf{{{value}}}" if latex else f"**{value}**"
    if cell["rank"] == "second":
        return rf"\underline{{{value}}}" if latex else f"<u>{value}</u>"
    return value


def metric_title(metric: str, latex: bool) -> str:
    title = dict(PANELS)[metric]
    return title if latex else title.replace("$\\tau=", "τ=").replace("$", "")


def table_complete(data: dict, models: tuple[str, ...]) -> bool:
    return all(data[model]["complete"] for model in models)


def render_latex(dataset: str, kind: str, models: tuple[str, ...],
                 data: dict, decimals: int) -> str:
    caption = ("Performance comparison of different methods" if kind == "comparison"
               else "Ablation performance")
    caption += f" on {DATASET_LABELS[dataset]} under varying recommendation sizes $N$ (Mean$\\pm$Std)."
    if not table_complete(data, models):
        caption += " Partial results; rankings use available models only, and a single seed has no Std."
    label = f"tab:{dataset.lower().replace('-', '_')}_{kind}_new_ep"
    columns = "c" * (len(models) + 1)
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{{columns}}}", r"\hline"]
    for index, metric in enumerate(METRICS):
        title = metric_title(metric, True)
        lines.append(rf"\multicolumn{{{len(models) + 1}}}{{c}}{{({chr(97 + index)}) {title}.}} \\")
        lines.append(r"\hline")
        lines.append("$N$ & " + " & ".join(LABELS.get(model, model) for model in models) + r" \\")
        lines.append(r"\hline")
        for top_k in TOP_KS:
            cells = ranked_cells(data, models, metric, top_k, decimals)
            lines.append(str(top_k) + " & " + " & ".join(
                formatted_cell(cell, decimals, True) for cell in cells) + r" \\")
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_markdown(dataset: str, kind: str, models: tuple[str, ...],
                    data: dict, decimals: int) -> str:
    lines = [f"# {DATASET_LABELS[dataset]} {kind} tables", "",
             "Mean ± sample standard deviation across seeds; deterministic baselines show only the mean.",
             f"Best and second-best are ranked by means shown to {decimals} decimal places; ties share the same style.",
             "Incomplete: missing model/seed results are blank; highlighted ranks are provisional."
             if not table_complete(data, models) else "Complete: all expected model/seed results are present.", ""]
    for index, metric in enumerate(METRICS):
        lines.extend([f"## ({chr(97 + index)}) {metric_title(metric, False)}", "",
                      "| N | " + " | ".join(LABELS.get(model, model) for model in models) + " |",
                      "| ---: | " + " | ".join("---:" for _ in models) + " |"])
        for top_k in TOP_KS:
            cells = ranked_cells(data, models, metric, top_k, decimals)
            lines.append(f"| {top_k} | " + " | ".join(
                formatted_cell(cell, decimals, False) for cell in cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_LABELS))
    parser.add_argument("--main-run-dir", type=Path, required=True)
    parser.add_argument("--comparison-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.decimals <= 6:
        parser.error("--decimals must be between 1 and 6")
    main_rows = load_run(args.main_run_dir, args.dataset, "main")
    comparison_rows = load_run(args.comparison_run_dir, args.dataset, "comparison")
    data = aggregate(main_rows, comparison_rows)
    output = args.output_dir or args.main_run_dir / "summaries" / "paper_tables"
    output.mkdir(parents=True, exist_ok=True)
    for kind, models in (("comparison", COMPARISON_MODELS), ("ablation", ABLATION_MODELS)):
        (output / f"{kind}.tex").write_text(
            render_latex(args.dataset, kind, models, data, args.decimals), encoding="utf-8")
        (output / f"{kind}.md").write_text(
            render_markdown(args.dataset, kind, models, data, args.decimals), encoding="utf-8")
    coverage = {model: {"available_seeds": item["available_seeds"],
                        "expected_seeds": item["expected_seeds"],
                        "complete": item["complete"]} for model, item in data.items()}
    (output / "aggregate.json").write_text(
        json.dumps({"dataset": args.dataset, "top_ks": TOP_KS, "thresholds": THRESHOLDS,
                    "ranking_decimals": args.decimals, "models": data,
                    "coverage": coverage,
                    "comparison_complete": table_complete(data, COMPARISON_MODELS),
                    "ablation_complete": table_complete(data, ABLATION_MODELS)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "output_dir": str(output),
                      "main_runs": len(main_rows), "comparison_runs": len(comparison_rows),
                      "comparison_complete": table_complete(data, COMPARISON_MODELS),
                      "ablation_complete": table_complete(data, ABLATION_MODELS),
                      "tables": ["comparison.tex", "ablation.tex"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
