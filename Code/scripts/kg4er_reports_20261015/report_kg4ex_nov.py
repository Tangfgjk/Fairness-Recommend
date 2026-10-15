"""Recalculate KG4Ex novelty from saved recommendation scores.

This is a standalone reporting script. It does not modify training artifacts or
the project's existing evaluation code. Run it after the final three-seed runs:

    python scripts/report_kg4ex_nov.py

The output is one six-page PDF with separate ablation and comparison tables
for each of the three datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "historical_kg4er_results_20261015" / "完整运行结果20261011"
OUTPUT = (
    ROOT / "runs" / "kg4er_reports_20261015"
    / "KG4Ex-NOV_三个数据集_完整对照_20261012.pdf"
)
TOP_KS = tuple(range(5, 51, 5))
SEEDS = (2024, 2025, 2026)
ABLATIONS = (
    ("2CKG4ER", "SemanticConvE_feature_only"),
    ("No-mastery", "SemanticConvE_feature_only_no_mastery"),
    ("No-forgetting", "SemanticConvE_feature_only_no_forgetting"),
    ("No-student-feature", "SemanticConvE_feature_only_learner_id"),
    ("No-exercise-feature", "SemanticConvE_feature_only_exercise_id"),
    ("ID-based Only", "SemanticConvE_id_only"),
)
COMPARISONS = ("EB-CF", "SB-CF", "CBF", "TransE", "TransE-adv", "RotatE", "DistMult", "ComplEx")
DATASETS = (
    ("Eedi", "Eedi", "resplit_seed20261007_modelseed2024",
     Path("_resplit_seed20261007/Eedi/raw"),
     Path("_resplit_seed20261007/Eedi/forgetting_variants/drop_exfr0p00_rec50/er_graph")),
    ("Algebra2005", "algebra2005", "forgetting_rec50_seed2024",
     Path("algebra2005/raw"),
     Path("algebra2005/forgetting_variants_rec50/drop_exfr0p00/er_graph")),
    ("XES3G5M-sub-small", "XES3G5M-sub-small", "forgetting_rec50_seed2024",
     Path("XES3G5M-sub-small/raw_compact"),
     Path("XES3G5M-sub-small/forgetting_variants_rec50/drop_exfr0p00/er_graph")),
)


def read_question_concepts(graph: Path) -> list[frozenset[int]]:
    questions: list[frozenset[int]] = []
    with (graph / "Q.txt").open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            concepts = frozenset(i for i, value in enumerate(line.strip().split(",")) if int(value))
            if not concepts:
                raise ValueError(f"Question {line_number} has no concepts: {graph}")
            questions.append(concepts)
    if not questions:
        raise ValueError(f"No questions in {graph}")
    return questions


def read_correct_history(raw: Path, questions: list[frozenset[int]]) -> dict[str, frozenset[int]]:
    """Use test students' actual correct responses, including XES's corrected history."""
    history: dict[str, set[int]] = defaultdict(set)
    with (raw / "interactions_all.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"].lower() != "test":
                continue
            uid = row["entity_id"]
            history[uid]  # retain a student with zero correct responses
            if int(float(row["response"])) == 1:
                exercise = int(row["question"])
                history[uid].update(questions[exercise])
    return {uid: frozenset(concepts) for uid, concepts in history.items()}


def read_scores(path: Path, question_count: int, expected_uids: set[str]) -> list[tuple[str, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved recommendation scores: {path}")
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of student scores: {path}")
    result = []
    seen = set()
    for entry in entries:
        uid, scores = entry["uid"], entry["scores"]
        if uid in seen or uid not in expected_uids or len(scores) != question_count:
            raise ValueError(f"Invalid UID or score count for {uid}: {path}")
        seen.add(uid)
        result.append((uid, scores))
    if seen != expected_uids:
        raise ValueError(f"Score users differ from test users in {path}: {len(seen)} != {len(expected_uids)}")
    return result


def score_file(result_root: Path, section: str, run_id: str, directory: str, seed: int | None) -> Path:
    base = result_root / section / run_id
    if seed is None:
        return base / "traditional_baselines" / directory / "eval" / "uid_ex_scores.json"
    return base / directory / f"seed{seed}" / "eval" / "uid_ex_scores.json"


def kg4ex_nov(
    entries: list[tuple[str, list[float]]],
    questions: list[frozenset[int]],
    history: dict[str, frozenset[int]],
) -> dict[int, float]:
    """Per KG4Ex: mean_u mean_e in TopN [1 - |K_e ∩ C_u|/|K_e ∪ C_u|]."""
    totals = {n: 0.0 for n in TOP_KS}
    cache: dict[tuple[str, int], float] = {}
    for uid, scores in entries:
        # Python's stable sort matches the saved project's ranking evaluator.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: TOP_KS[-1]]
        cumulative = 0.0
        correct = history[uid]
        for rank, exercise in enumerate(top, 1):
            key = uid, exercise
            novelty = cache.get(key)
            if novelty is None:
                concepts = questions[exercise]
                intersection = len(concepts & correct)
                union = len(concepts | correct)
                if union == 0:
                    raise ValueError(f"Undefined Jaccard score for {uid}, question {exercise}")
                novelty = 1.0 - intersection / union
                cache[key] = novelty
            cumulative += novelty
            if rank in totals:
                totals[rank] += cumulative / rank
    return {n: totals[n] / len(entries) for n in TOP_KS}


def collect_dataset(spec: tuple, repo: Path, results: Path) -> tuple[str, dict[str, dict[int, tuple[float, float | None]]]]:
    label, directory, run_id, raw_relative, graph_relative = spec
    raw = repo / "Data_Fin_New" / raw_relative
    graph = repo / "Data_Fin_New" / graph_relative
    questions = read_question_concepts(graph)
    history = read_correct_history(raw, questions)
    data: dict[str, dict[int, tuple[float, float | None]]] = {}
    jobs = [(name, "ablation", path, SEEDS) for name, path in ABLATIONS]
    jobs += [(name, "comparison", name, (None,) if name in COMPARISONS[:3] else SEEDS)
             for name in COMPARISONS]
    for name, section, model_dir, seeds in jobs:
        per_seed = []
        for seed in seeds:
            path = score_file(results / directory, section, run_id, model_dir, seed)
            scores = read_scores(path, len(questions), set(history))
            per_seed.append(kg4ex_nov(scores, questions, history))
        data[name] = {
            n: (mean(values := [v[n] for v in per_seed]), stdev(values) if len(values) > 1 else None)
            for n in TOP_KS
        }
        print(f"{label}: {name}: {len(per_seed)} score file(s), {len(history)} test students", flush=True)
    return label, data


def load_fonts() -> None:
    # Windows Times New Roman provides compact, readable publication-style tables.
    fonts = Path("C:/Windows/Fonts")
    regular, bold = fonts / "times.ttf", fonts / "timesbd.ttf"
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("TableSerif", str(regular)))
        pdfmetrics.registerFont(TTFont("TableSerifBold", str(bold)))
    else:
        # The script also works on headless Linux, including AutoDL.
        pdfmetrics.registerFontFamily("TableSerif", normal="Times-Roman", bold="Times-Bold")


def font_name(bold: bool = False) -> str:
    if "TableSerif" in pdfmetrics.getRegisteredFontNames():
        return "TableSerifBold" if bold else "TableSerif"
    return "Times-Bold" if bold else "Times-Roman"


def draw_cell(c: canvas.Canvas, text: str, x: float, y: float, width: float, *,
              bold: bool = False, underline: bool = False, size: float = 8.0) -> None:
    name = font_name(bold)
    c.setFont(name, size)
    c.setFillColor(colors.black)
    c.drawCentredString(x + width / 2, y, text)
    if underline:
        text_width = pdfmetrics.stringWidth(text, name, size)
        c.setLineWidth(0.75)
        c.line(x + (width - text_width) / 2, y - 1.8, x + (width + text_width) / 2, y - 1.8)


def draw_table(c: canvas.Canvas, label: str, section: str,
               data: dict[str, dict[int, tuple[float, float | None]]]) -> None:
    page_width, page_height = landscape(A4)
    margin = 32
    width = page_width - 2 * margin
    n_width = 40
    model_names = ([name for name, _ in ABLATIONS] if section == "Ablation"
                   else [ABLATIONS[0][0], *COMPARISONS])
    model_width = (width - n_width) / len(model_names)

    c.setFont(font_name(True), 19)
    c.drawString(margin, page_height - 56, f"KG4Ex-NOV on {label} | {section} experiments")
    c.setFont(font_name(), 9)
    c.drawString(margin, page_height - 76, "Final graph: theta_m=0.8, theta_f=0.8, rec=50; zero-valued exercise-forgetting edges removed")
    c.drawString(margin, page_height - 92, "Correct-response history; N = 5, 10, ..., 50; higher is better")

    top = page_height - 135
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.line(margin, top, margin + width, top)
    group_y = top - 19
    draw_cell(c, "Six ablation models" if section == "Ablation" else "Main model and eight comparison models",
              margin + n_width, group_y, len(model_names) * model_width, bold=True, size=10)
    c.setLineWidth(0.6)
    c.line(margin, top - 29, margin + width, top - 29)

    header_y = top - 54
    draw_cell(c, "N", margin, header_y, n_width, bold=True, size=10)
    headers = {
        "No-student-feature": ("No-student", "feature"),
        "No-exercise-feature": ("No-exercise", "feature"),
        "No-forgetting": ("No-", "forgetting"),
        "No-mastery": ("No-", "mastery"),
        "ID-based Only": ("ID-based", "Only"),
    }
    for i, name in enumerate(model_names):
        lines = headers.get(name, (name,))
        x = margin + n_width + i * model_width
        for j, line in enumerate(lines):
            draw_cell(c, line, x, header_y + 4 - j * 10, model_width, bold=True, size=9)
    c.line(margin, top - 72, margin + width, top - 72)

    row_height = 25.5
    for row, n in enumerate(TOP_KS):
        y = top - 96 - row * row_height
        draw_cell(c, str(n), margin, y, n_width, size=9)
        ranking = sorted(((data[name][n][0], name) for name in model_names), reverse=True)
        best = ranking[0][1]
        second = ranking[1][1]
        for i, name in enumerate(model_names):
            average, sd = data[name][n]
            shown = f"{average:.3f}" if sd is None else f"{average:.3f} ± {sd:.3f}"
            draw_cell(c, shown, margin + n_width + i * model_width, y, model_width,
                      bold=name == best, underline=name == second, size=9)
        c.setLineWidth(0.35)
        c.setStrokeColor(colors.Color(0.65, 0.65, 0.65))
        c.line(margin, y - 10.5, margin + width, y - 10.5)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.9)
    c.line(margin, top - 96 - (len(TOP_KS) - 1) * row_height - 10.5,
           margin + width, top - 96 - (len(TOP_KS) - 1) * row_height - 10.5)

    note_y = top - 96 - len(TOP_KS) * row_height - 20
    c.setFont(font_name(), 8)
    c.drawString(margin, note_y,
                 "KG4Ex-NOV(u,e) = 1 - |K(e) intersection C(u)| / |K(e) union C(u)|; average over Top-N exercises, then test students.")
    c.drawString(margin, note_y - 15,
                 "C(u) contains knowledge concepts from correctly answered test-history exercises. Values are mean ± sample SD across seeds 2024, 2025, 2026.")
    c.drawString(margin, note_y - 30,
                 ("EB-CF, SB-CF and CBF are deterministic single runs. " if section == "Comparison" else "")
                 + "Bold: best; underline: second best, ranked by unrounded mean within this table and N row.")
    c.showPage()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    results = (args.results_root or RESULTS).resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    load_fonts()
    c = canvas.Canvas(str(output), pagesize=landscape(A4), pageCompression=1)
    c.setTitle("KG4Ex-NOV: ablation and comparison tables for three datasets")
    for spec in DATASETS:
        label, data = collect_dataset(spec, repo, results)
        draw_table(c, label, "Ablation", data)
        draw_table(c, label, "Comparison", data)
    c.save()
    print(f"Wrote six-table PDF: {output}")


if __name__ == "__main__":
    main()
