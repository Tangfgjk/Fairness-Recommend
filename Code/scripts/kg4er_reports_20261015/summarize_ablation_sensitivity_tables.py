"""Make paper-style tables from completed sensitivity ablation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "Data_Fin_New" / "_sensitivity_runs"
DATASETS = ("algebra2005", "XES3G5M-sub-small", "Eedi")
MODELS = (
    ("feature_only", "2CKG4ER"),
    ("feature_only_no_mastery", "No-mastery"),
    ("feature_only_no_forgetting", "No-forgetting"),
    ("feature_only_learner_id", "No-student-feature"),
    ("feature_only_exercise_id", "No-Exercise-feature"),
    ("id_only", "ID-based Only"),
)
METRICS = (
    ("ACC", "Ada", "Ada"),
    ("NOV", "NOV", "NOV"),
    ("Ep_t0.4", "Ep_t0p4", "Ep (threshold 0.4)"),
    ("Ep_t0.5", "Ep_t0p5", "Ep (threshold 0.5)"),
    ("Ep_t0.6", "Ep_t0p6", "Ep (threshold 0.6)"),
    ("Ep_t0.7", "Ep_t0p7", "Ep (threshold 0.7)"),
)
TOP_KS = range(5, 51, 5)


def read_results(run_dir: Path) -> dict[str, dict]:
    results = {}
    for code, _ in MODELS:
        path = run_dir / f"SemanticConvE_{code}" / "seed2024" / "eval" / "metrics.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # A model may be writing its result while this script runs.
        if data.get("seed") != 2024:
            raise ValueError(f"Unexpected seed in {path}")
        results[code] = data
    return results


def rank(values: dict[str, float]) -> dict[str, str]:
    if len(values) < 2:
        return {}
    levels = sorted(set(values.values()), reverse=True)
    return {code: "best" if value == levels[0] else "second" if len(levels) > 1 and value == levels[1] else ""
            for code, value in values.items()}


def formatted(value: float | None, place: str, style: str) -> str:
    if value is None:
        return "—" if style == "md" else "--"
    text = f"{value:.3f}"
    if place == "best":
        return f"**{text}**" if style == "md" else rf"\textbf{{{text}}}"
    if place == "second":
        return f"<u>{text}</u>" if style == "md" else rf"\underline{{{text}}}"
    return text


def table(dataset: str, variant: str, metric: str, title: str, results: dict[str, dict], style: str) -> str:
    headers = [label for _, label in MODELS]
    complete = len(results) == len(MODELS)
    if style == "md":
        lines = [f"## {title} — {dataset} — {variant}", "",
                 f"仅 seed 2024；已完成 {len(results)}/{len(MODELS)} 个模型。表内数值不是跨随机种子的 Mean±Std。"
                 + ("最优加粗，次优加下划线；并列同等标注。" if complete else "待全部模型完成后再标最优和次优。"), "",
                 "| N | " + " | ".join(headers) + " |",
                 "| ---: | " + " | ".join("---:" for _ in headers) + " |"]
    else:
        caption = f"{title} on {dataset} ({variant}), seed 2024."
        lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}",
                 r"\begin{tabular}{r" + "c" * len(MODELS) + "}", r"\hline",
                 "$N$ & " + " & ".join(headers) + " " + chr(92) * 2, r"\hline"]
    for n in TOP_KS:
        values = {}
        for code, _ in MODELS:
            payload = results.get(code, {}).get(metric, {}).get(str(n))
            if isinstance(payload, dict) and payload.get("mean") is not None:
                values[code] = float(payload["mean"])
        places = rank(values) if complete and len(values) == len(MODELS) else {}
        cells = [formatted(values.get(code), places.get(code, ""), style) for code, _ in MODELS]
        if style == "md":
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        else:
            lines.append(f"{n} & " + " & ".join(cells) + " " + chr(92) * 2)
    if style == "md":
        lines.append("")
    else:
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="sensitivity_seed2024")
    parser.add_argument("--output-root", type=Path, default=RUNS / "_paper_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = []
    for variant_dir in sorted(RUNS.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        for dataset in DATASETS:
            run_dir = variant_dir / dataset / args.run_id
            if not run_dir.is_dir():
                continue
            results = read_results(run_dir)
            if not results:
                continue
            output = args.output_root / variant_dir.name / dataset
            output.mkdir(parents=True, exist_ok=True)
            markdown = []
            latex = []
            for metric, filename, title in METRICS:
                md = table(dataset, variant_dir.name, metric, title, results, "md")
                tex = table(dataset, variant_dir.name, metric, title, results, "tex")
                (output / f"{filename}.md").write_text(md, encoding="utf-8")
                (output / f"{filename}.tex").write_text(tex, encoding="utf-8")
                markdown.append(md)
                latex.append(tex)
            (output / "all_metrics.md").write_text("\n".join(markdown), encoding="utf-8")
            (output / "all_metrics.tex").write_text("\n".join(latex), encoding="utf-8")
            generated.append((variant_dir.name, dataset, len(results), output))
    if not generated:
        print("No completed evaluations found.")
    for variant, dataset, models, output in generated:
        print(f"{dataset} / {variant}: {models}/{len(MODELS)} models -> {output}")


if __name__ == "__main__":
    main()
