"""Report novelty of the *union* of concepts in each student's Top-N list.

This is distinct from report_kg4ex_nov.py, which averages per-exercise
Jaccard novelty. Both reports reuse exactly the same saved rankings and history.

    python scripts/report_kg4ex_nov_set.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

import report_kg4ex_nov as base


OUTPUT = (
    base.ROOT / "runs" / "kg4er_reports_20261015"
    / "KG4Ex-NOV-Set_三个数据集_消融与对比_20261012.pdf"
)


def kg4ex_nov_set(
    entries: list[tuple[str, list[float]]],
    questions: list[frozenset[int]],
    history: dict[str, frozenset[int]],
) -> dict[int, float]:
    """Mean over students of 1 - Jaccard(KCs in Top-N questions, correct KCs)."""
    totals = {n: 0.0 for n in base.TOP_KS}
    for uid, scores in entries:
        # The original ranker uses a stable descending sort; ties keep Q.txt order.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: base.TOP_KS[-1]]
        recommended_concepts: set[int] = set()
        correct = history[uid]
        for rank, exercise in enumerate(top, 1):
            recommended_concepts.update(questions[exercise])
            if rank in totals:
                intersection = len(recommended_concepts & correct)
                union = len(recommended_concepts | correct)
                if union == 0:
                    raise ValueError(f"Undefined Jaccard score for {uid} at Top-{rank}")
                totals[rank] += 1.0 - intersection / union
    return {n: totals[n] / len(entries) for n in base.TOP_KS}


def collect_dataset(spec: tuple, repo: Path, results: Path) -> tuple[str, dict[str, dict[int, tuple[float, float | None]]]]:
    label, directory, run_id, raw_relative, graph_relative = spec
    raw = repo / "Data_Fin_New" / raw_relative
    graph = repo / "Data_Fin_New" / graph_relative
    questions = base.read_question_concepts(graph)
    history = base.read_correct_history(raw, questions)
    data: dict[str, dict[int, tuple[float, float | None]]] = {}
    jobs = [(name, "ablation", path, base.SEEDS) for name, path in base.ABLATIONS]
    jobs += [(name, "comparison", name, (None,) if name in base.COMPARISONS[:3] else base.SEEDS)
             for name in base.COMPARISONS]
    for name, section, model_dir, seeds in jobs:
        per_seed = []
        for seed in seeds:
            path = base.score_file(results / directory, section, run_id, model_dir, seed)
            scores = base.read_scores(path, len(questions), set(history))
            per_seed.append(kg4ex_nov_set(scores, questions, history))
        data[name] = {
            n: (mean(values := [v[n] for v in per_seed]), stdev(values) if len(values) > 1 else None)
            for n in base.TOP_KS
        }
        print(f"{label}: {name}: {len(per_seed)} score file(s), {len(history)} test students", flush=True)
    return label, data


def draw_table(c: canvas.Canvas, label: str, section: str,
               data: dict[str, dict[int, tuple[float, float | None]]]) -> None:
    page_width, page_height = landscape(A4)
    margin = 32
    width = page_width - 2 * margin
    n_width = 40
    model_names = ([name for name, _ in base.ABLATIONS] if section == "Ablation"
                   else [base.ABLATIONS[0][0], *base.COMPARISONS])
    model_width = (width - n_width) / len(model_names)

    c.setFont(base.font_name(True), 19)
    c.drawString(margin, page_height - 56, f"KG4Ex-NOV-Set on {label} | {section} experiments")
    c.setFont(base.font_name(), 9)
    c.drawString(margin, page_height - 76, "Final graph: theta_m=0.8, theta_f=0.8, rec=50; zero-valued exercise-forgetting edges removed")
    c.drawString(margin, page_height - 92, "Union of concepts in each student's Top-N list; N = 5, 10, ..., 50; higher is better")

    top = page_height - 135
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.line(margin, top, margin + width, top)
    base.draw_cell(c, "Six ablation models" if section == "Ablation" else "Main model and eight comparison models",
                   margin + n_width, top - 19, len(model_names) * model_width, bold=True, size=10)
    c.setLineWidth(0.6)
    c.line(margin, top - 29, margin + width, top - 29)

    header_y = top - 54
    base.draw_cell(c, "N", margin, header_y, n_width, bold=True, size=10)
    headers = {
        "No-student-feature": ("No-student", "feature"),
        "No-exercise-feature": ("No-exercise", "feature"),
        "No-forgetting": ("No-", "forgetting"),
        "No-mastery": ("No-", "mastery"),
        "ID-based Only": ("ID-based", "Only"),
    }
    for i, name in enumerate(model_names):
        x = margin + n_width + i * model_width
        for j, line in enumerate(headers.get(name, (name,))):
            base.draw_cell(c, line, x, header_y + 4 - j * 10, model_width, bold=True, size=9)
    c.line(margin, top - 72, margin + width, top - 72)

    row_height = 25.5
    for row, n in enumerate(base.TOP_KS):
        y = top - 96 - row * row_height
        base.draw_cell(c, str(n), margin, y, n_width, size=9)
        ranked = sorted(((data[name][n][0], name) for name in model_names), reverse=True)
        best, second = ranked[0][1], ranked[1][1]
        for i, name in enumerate(model_names):
            average, sd = data[name][n]
            shown = f"{average:.3f}" if sd is None else f"{average:.3f} ± {sd:.3f}"
            base.draw_cell(c, shown, margin + n_width + i * model_width, y, model_width,
                           bold=name == best, underline=name == second, size=9)
        c.setLineWidth(0.35)
        c.setStrokeColor(colors.Color(0.65, 0.65, 0.65))
        c.line(margin, y - 10.5, margin + width, y - 10.5)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.9)
    c.line(margin, top - 96 - (len(base.TOP_KS) - 1) * row_height - 10.5,
           margin + width, top - 96 - (len(base.TOP_KS) - 1) * row_height - 10.5)

    note_y = top - 96 - len(base.TOP_KS) * row_height - 20
    c.setFont(base.font_name(), 8)
    c.drawString(margin, note_y,
                 "K_TopN(u) = union of concepts across the Top-N exercises; NOV-Set(u) = 1 - |K_TopN(u) intersection C(u)| / |K_TopN(u) union C(u)|.")
    c.drawString(margin, note_y - 15,
                 "C(u) is the student's correct-response concept set. Average students first, then report mean ± sample SD across seeds 2024, 2025, 2026.")
    c.drawString(margin, note_y - 30,
                 ("EB-CF, SB-CF and CBF are deterministic single runs. " if section == "Comparison" else "")
                 + "Bold: best; underline: second best, ranked by unrounded mean within this table and N row.")
    c.showPage()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=base.ROOT)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    results = (args.results_root or base.RESULTS).resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base.load_fonts()
    c = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    c.setTitle("KG4Ex-NOV-Set: six tables for three datasets")
    for spec in base.DATASETS:
        label, data = collect_dataset(spec, repo, results)
        draw_table(c, label, "Ablation", data)
        draw_table(c, label, "Comparison", data)
    c.save()
    print(f"Wrote six-table KG4Ex-NOV-Set PDF: {output}")


if __name__ == "__main__":
    main()
