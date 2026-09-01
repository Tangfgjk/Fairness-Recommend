"""Create readable split top-K reports from completed ER runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


METRICS = ("Ada", "NOV")
TOP_KS = tuple(range(5, 101, 5))

# Fixed colors make the same model recognisable across datasets and figures.
MODEL_COLORS = {
    "TransE": "#1F2937",
    "TransE-adv": "#6B7280",
    "RotatE": "#A21CAF",
    "DistMult": "#0F766E",
    "ComplEx": "#C2410C",
    "CBF": "#854D0E",
    "SB-CF": "#BE123C",
    "EB-CF": "#4F46E5",
    "2CKG4ER_id_only": "#D55E00",
    "2CKG4ER": "#009E73",
    "2CKG4ER_relation_id": "#7C3AED",
    "2CKG4ER_learner_id": "#DC2626",
    "2CKG4ER_exercise_id": "#0891B2",
    "2CKG4ER_no_mastery": "#CA8A04",
    "2CKG4ER_no_forgetting": "#9333EA",
}
MODEL_MARKERS = {
    "TransE": "o", "TransE-adv": "s", "RotatE": "^", "DistMult": "D", "ComplEx": "P",
    "CBF": "X", "SB-CF": "v", "EB-CF": "<",
    "2CKG4ER_id_only": "s", "2CKG4ER": "^", "2CKG4ER_relation_id": "D",
    "2CKG4ER_learner_id": "P", "2CKG4ER_exercise_id": "X",
    "2CKG4ER_no_mastery": "v", "2CKG4ER_no_forgetting": "<",
}
REPRESENTATION_MODELS = (
    "2CKG4ER", "2CKG4ER_id_only", "2CKG4ER_relation_id",
    "2CKG4ER_learner_id", "2CKG4ER_exercise_id",
)
COGNITIVE_MODELS = (
    "2CKG4ER",
    "2CKG4ER_no_mastery",
    "2CKG4ER_no_forgetting",
)
COMPARISON_MODELS = ("2CKG4ER", "TransE", "TransE-adv", "RotatE", "DistMult", "ComplEx", "CBF", "SB-CF", "EB-CF")


def parse_run_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=RUN_DIR")
    return label, Path(raw_path)


def model_key(model: str) -> str:
    legacy = {
        "SemanticConvE": "2CKG4ER",
        "feature_only": "2CKG4ER",
        "id_only": "2CKG4ER_id_only",
        "feature_only_relation_id": "2CKG4ER_relation_id",
        "feature_only_learner_id": "2CKG4ER_learner_id",
        "feature_only_exercise_id": "2CKG4ER_exercise_id",
        "feature_only_no_mastery": "2CKG4ER_no_mastery",
        "feature_only_no_forgetting": "2CKG4ER_no_forgetting",
    }
    normalized = model.replace("SemanticConvE_", "")
    if normalized == "2CKG4ER_id_only":
        return normalized
    return legacy.get(normalized, normalized)


def load_rows(label: str, run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("**/eval/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model", path.parents[2].name))
        for metric in METRICS:
            for top_k in TOP_KS:
                source_metric = metric
                if metric == "Ada" and "Ada" not in payload and "ACC" in payload:
                    source_metric = "ACC"
                item = payload.get(source_metric, {}).get(str(top_k), {})
                value = item.get("mean") if isinstance(item, dict) else None
                if isinstance(value, (int, float)):
                    rows.append({"group": label, "model": model, "metric": metric, "top_k": top_k, "value": float(value)})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["model"], row["metric"], row["top_k"])].append(row["value"])
    return [
        {"group": group, "model": model, "metric": metric, "top_k": top_k, "mean": mean(values), "runs": len(values)}
        for (group, model, metric, top_k), values in sorted(grouped.items())
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "model", "metric", "top_k", "mean", "runs"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Top-K Result Comparison", "", "All models use the train graph only; `test_triples.txt` is evaluation-only.", ""]
    for metric in METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        names = sorted({(row["group"], row["model"]) for row in metric_rows})
        lookup = {(row["group"], row["model"], row["top_k"]): row for row in metric_rows}
        lines.extend([f"## {metric}", "", "| Group | Model | Runs | " + " | ".join(f"@{k}" for k in TOP_KS) + " |", "| --- | --- | ---: | " + " | ".join("---:" for _ in TOP_KS) + " |"])
        for group, model in names:
            values = [lookup.get((group, model, k)) for k in TOP_KS]
            runs = max((item["runs"] for item in values if item), default=0)
            lines.append(f"| {group} | {model} | {runs} | " + " | ".join(f"{item['mean']:.6f}" if item else "" for item in values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_category(rows: list[dict], models: tuple[str, ...], title: str, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=220)
    for axis, metric in zip(axes, METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        for key in models:
            candidates = [row for row in metric_rows if model_key(row["model"]) == key]
            if not candidates:
                continue
            values = {row["top_k"]: row["mean"] for row in candidates}
            xs = [k for k in TOP_KS if k in values]
            if not xs:
                continue
            axis.plot(
                xs, [values[k] for k in xs], label=key, color=MODEL_COLORS.get(key, "#111827"),
                marker=MODEL_MARKERS.get(key, "o"), linewidth=3.0 if key == "2CKG4ER" else 2.0,
                markersize=6.0 if key == "2CKG4ER" else 4.8,
            )
        axis.set_title(metric, fontsize=14, weight="bold")
        axis.set_xlabel("Recommendation list size N")
        axis.set_ylabel(metric)
        axis.set_xticks(TOP_KS)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Model", loc="center left", bbox_to_anchor=(0.84, 0.5), fontsize=8, title_fontsize=9)
    figure.suptitle(title, fontsize=16, weight="bold")
    figure.subplots_adjust(left=0.06, right=0.82, bottom=0.14, top=0.86, wspace=0.22)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    global TOP_KS
    parser = argparse.ArgumentParser(description="Create split tables and readable top-K figures for completed ER runs.")
    parser.add_argument("--run-dir", action="append", required=True, type=parse_run_dir, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-ks", default=",".join(str(value) for value in TOP_KS))
    args = parser.parse_args()
    TOP_KS = tuple(int(item.strip()) for item in args.top_ks.split(",") if item.strip())
    if not TOP_KS or any(value <= 0 for value in TOP_KS):
        raise ValueError("--top-ks must contain positive integers")
    rows: list[dict] = []
    for label, run_dir in args.run_dir:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        rows.extend(load_rows(label, run_dir))
    if not rows:
        raise ValueError("No eval/metrics.json files were found in the supplied run directories.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(summary, output_dir / "topk_comparison.csv")
    write_markdown(summary, output_dir / "topk_comparison.md")
    plot_category(summary, REPRESENTATION_MODELS, "Representation Ablations", output_dir / "topk_representation.png")
    plot_category(summary, COGNITIVE_MODELS, "Cognitive Relation Ablations", output_dir / "topk_cognitive.png")
    plot_category(summary, COMPARISON_MODELS, "2CKG4ER vs Baselines", output_dir / "topk_baselines.png")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "metric_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
