"""Make reference-style, one-metric tables for the completed V5 datasets."""

from __future__ import annotations

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from report_v5_seed2024_two_datasets import (
    ABLATIONS, COMPARISONS, DATASETS, FAIRNESS, LABELS, LOWER_BETTER,
    RUNS, TOP_KS, load_results, metric_values, rank_map,
)


OUTPUT = RUNS / "论文格式文档统计" / "V5_2024_Eedi_Algebra2005_20261020"
PAGE_SIZE = landscape(A3)
WIDTH, HEIGHT = PAGE_SIZE
MARGIN = 45
QUALITY_METRICS = ("Ada", "NOV", "Ep_t0.5")
ABLATION_METRICS = QUALITY_METRICS + FAIRNESS
COMPARISON_METRICS = QUALITY_METRICS


def metric_pairs(metrics: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(metrics[i:i + 2]) for i in range(0, len(metrics), 2))
DISPLAY = {
    "Ada": "Adaptability (Ada)",
    "NOV": "Novelty (NOV; KG4Ex-NOV-Set)",
    "Ep_t0.4": "Educational potential (Ep, threshold 0.4)",
    "Ep_t0.5": "Educational potential (Ep, threshold 0.5)",
    "Ep_t0.6": "Educational potential (Ep, threshold 0.6)",
    "Ep_t0.7": "Educational potential (Ep, threshold 0.7)",
    "ItemExposureGini": "Item exposure Gini",
    "KCExposureGini": "Knowledge-concept exposure Gini",
    "ItemCoverage": "Item coverage",
    "KCCoverage": "Knowledge-concept coverage",
    "LongTailItemExposureShare": "Long-tail item exposure share",
    "LongTailKCExposureShare": "Long-tail knowledge-concept exposure share",
    "HeadItemExposureShare": "Head-item exposure share",
    "HeadKCExposureShare": "Head knowledge-concept exposure share",
}
SHORT_LABELS = {
    "baseline": "CompGCN+ConvE",
    "pre_item_kc": "Pre",
    "post_only": "Post",
    "pre_item_kc_post": "Pre+Post",
    "in_kppd": "In (KPPD)",
    "pre_in_kppd": "Pre+In",
    "in_kppd_post": "In+Post",
    "pre_in_kppd_post": "Pre+In+Post (main)",
    "no_compgcn": "Pre+In+Post (ConvE)",
    "main": "Pre+In+Post (main)",
}


def paragraph(value: str, style: ParagraphStyle, rank: int = 0) -> Paragraph:
    markup = escape(value)
    if rank == 1:
        markup = f"<b>{markup}</b>"
    elif rank == 2:
        markup = f"<u>{markup}</u>"
    return Paragraph(markup, style)


def make_metric_table(results: dict, dataset: str, methods: tuple[str, ...],
                      metric: str, available_width: float) -> Table:
    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=8.8,
                             leading=10.2, alignment=TA_CENTER)
    number = ParagraphStyle("number", fontName="Helvetica", fontSize=9,
                            leading=10.6, alignment=TA_CENTER)
    names = ["N"] + [SHORT_LABELS.get(method, LABELS.get(method, method)) for method in methods]
    rows = [[paragraph(label, heading) for label in names]]
    for top_k in TOP_KS:
        ranks = rank_map(results, dataset, methods, metric, top_k)
        cells = [paragraph(str(top_k), number)]
        for method in methods:
            value, _ = metric_values(results[(dataset, method)], metric, top_k)
            cells.append(paragraph(f"{value:.4f}", number, ranks[method]))
        rows.append(cells)

    n_width = 48
    table = Table(rows, colWidths=[n_width] + [(available_width - n_width) / len(methods)] * len(methods),
                  rowHeights=[42] + [25] * len(TOP_KS))
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, colors.HexColor("#203044")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.HexColor("#203044")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.75, colors.HexColor("#203044")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def export_dataset(results: dict, dataset: str, output: Path, *, post_method: str | None = None) -> None:
    pdf = canvas.Canvas(str(output), pagesize=PAGE_SIZE)
    suffix = f" | Post: {post_method}" if post_method else ""
    pdf.setTitle(f"V5 seed 2024 | {dataset} | ablation and comparison metric tables{suffix}")
    page = 0
    section_letter = 0
    for family, methods, metrics in (
        ("Ablation", ABLATIONS, ABLATION_METRICS),
        ("Comparison", COMPARISONS, COMPARISON_METRICS),
    ):
        for metric_group in metric_pairs(metrics):
            page += 1
            pdf.setFont("Times-Bold", 12)
            pdf.drawString(MARGIN, HEIGHT - 34, "Fairness-Recommend  |  V5 experiment")
            pdf.setFont("Times-Roman", 10)
            pdf.drawRightString(WIDTH - MARGIN, HEIGHT - 34, f"{dataset}  |  {family}")
            pdf.setStrokeColor(colors.HexColor("#233448"))
            pdf.line(MARGIN, HEIGHT - 41, WIDTH - MARGIN, HEIGHT - 41)
            pdf.setFont("Times-Bold", 21)
            pdf.drawString(MARGIN, HEIGHT - 82, f"{family} performance")
            pdf.setFont("Times-Roman", 11)
            pdf.setFillColor(colors.HexColor("#526477"))
            subtitle = "Top-N results; seed 2024."
            if post_method:
                subtitle += f" Post method: {post_method}."
            pdf.drawString(MARGIN, HEIGHT - 102, subtitle)
            pdf.setFillColor(colors.black)

            for index, metric in enumerate(metric_group):
                top = HEIGHT - 130 - index * 331
                letter = chr(ord("a") + section_letter)
                section_letter += 1
                direction = "lower is better" if metric in LOWER_BETTER else "higher is better"
                pdf.setFont("Times-Bold", 13)
                pdf.drawString(MARGIN, top, f"({letter}) {DISPLAY[metric]} ({direction})")
                table = make_metric_table(results, dataset, methods, metric, WIDTH - 2 * MARGIN)
                _, table_height = table.wrapOn(pdf, WIDTH - 2 * MARGIN, HEIGHT)
                table.drawOn(pdf, MARGIN, top - 17 - table_height)

            pdf.setStrokeColor(colors.HexColor("#CDD7E0"))
            pdf.line(MARGIN, 56, WIDTH - MARGIN, 56)
            pdf.setFont("Times-Roman", 9)
            pdf.setFillColor(colors.HexColor("#526477"))
            pdf.drawString(MARGIN, 40,
                           "Bold = best; underline = second best within each N and experiment family (ranked before rounding).")
            pdf.drawRightString(WIDTH - MARGIN, 40, str(page))
            pdf.setFillColor(colors.black)
            pdf.showPage()
    pdf.save()


def main() -> None:
    results, _ = load_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        target = OUTPUT / f"V5_2024_{dataset}_one_metric_per_table.pdf"
        export_dataset(results, dataset, target)
        print(target)


if __name__ == "__main__":
    main()
