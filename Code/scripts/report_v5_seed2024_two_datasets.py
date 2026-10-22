"""Export checked V5 seed-2024 results for Eedi and Algebra2005.

The PDF shows Top-10/20/50 paper tables. The CSV keeps every Top-5..50
value, its source path, and ranks within each ablation/comparison table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATASETS = ("Eedi", "algebra2005")
TOP_KS = tuple(range(5, 51, 5))
PAPER_KS = (10, 20, 50)
ABLATIONS = (
    "baseline", "pre_item_kc", "post_only", "pre_item_kc_post",
    "in_kppd", "pre_in_kppd", "in_kppd_post", "pre_in_kppd_post",
    "no_compgcn",
)
COMPARISONS = ("EB-CF", "SB-CF", "CBF", "TransE", "TransE-adv", "main")
LABELS = {
    "baseline": "CompGCN + ConvE (base)",
    "pre_item_kc": "Pre",
    "post_only": "Post",
    "pre_item_kc_post": "Pre + Post",
    "in_kppd": "In (KPPD)",
    "pre_in_kppd": "Pre + In",
    "in_kppd_post": "In + Post",
    "pre_in_kppd_post": "Pre + In + Post (main)",
    "no_compgcn": "Pre + In + Post (ConvE only)",
    "main": "Pre + In + Post (main)",
}
QUALITY = ("Ada", "NOV", "Ep_t0.4", "Ep_t0.5", "Ep_t0.6", "Ep_t0.7")
FAIRNESS = (
    "ItemExposureGini", "KCExposureGini", "ItemCoverage", "KCCoverage",
    "LongTailItemExposureShare", "LongTailKCExposureShare",
    "HeadItemExposureShare", "HeadKCExposureShare",
)
METRICS = QUALITY + FAIRNESS
LOWER_BETTER = {
    "ItemExposureGini", "KCExposureGini",
    "HeadItemExposureShare", "HeadKCExposureShare",
}
HEADERS = {
    "Ada": "Ada ↑", "NOV": "NOV ↑", "Ep_t0.4": "Ep .4 ↑",
    "Ep_t0.5": "Ep .5 ↑", "Ep_t0.6": "Ep .6 ↑", "Ep_t0.7": "Ep .7 ↑",
    "ItemExposureGini": "Item Gini ↓", "KCExposureGini": "KC Gini ↓",
    "ItemCoverage": "Item cov. ↑", "KCCoverage": "KC cov. ↑",
    "LongTailItemExposureShare": "LT item ↑",
    "LongTailKCExposureShare": "LT KC ↑",
    "HeadItemExposureShare": "Head item ↓",
    "HeadKCExposureShare": "Head KC ↓",
}


def source_path(dataset: str, method: str) -> Path:
    base = RUNS / dataset
    if method in ABLATIONS:
        if method == "no_compgcn":
            run = base / f"{dataset}_V5_2024_v3_no_compgcn" / "pre_in_kppd_post" / "seed2024"
        else:
            run = base / f"{dataset}_V5_2024_v3" / method / "seed2024"
        stage = run / "eval_stage.json"
        if not stage.is_file() or json.loads(stage.read_text(encoding="utf-8"))["status"] != "completed":
            raise ValueError(f"Evaluation is not complete: {stage}")
        matches = list(run.glob("eval*/metrics.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one final evaluation in {run}, found {len(matches)}")
        return matches[0]
    if method == "main":
        return source_path(dataset, "pre_in_kppd_post")
    run = base / f"{dataset}_V5_2024_comparison"
    if method in ("EB-CF", "SB-CF", "CBF"):
        return run / "traditional_baselines" / method / "eval" / "metrics.json"
    return run / method / "seed2024" / "eval" / "metrics.json"


def load_results() -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    result = {}
    manifest = {}
    for dataset in DATASETS:
        for phase in ("v3_non_kppd", "v3_kppd", "no_compgcn", "comparison"):
            stage = RUNS / "V5_2024" / "stages" / f"{phase}_{dataset}.json"
            if not stage.is_file() or json.loads(stage.read_text(encoding="utf-8"))["status"] != "completed":
                raise ValueError(f"Batch phase is not complete: {stage}")
        for method in (*ABLATIONS, *COMPARISONS):
            path = source_path(dataset, method)
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("dataset") != dataset:
                raise ValueError(f"Wrong dataset in {path}")
            if method not in ("EB-CF", "SB-CF", "CBF") and payload.get("seed") != 2024:
                raise ValueError(f"Wrong seed in {path}")
            if payload.get("NOV") != payload.get("KG4Ex-NOV-Set"):
                raise ValueError(f"Formal NOV differs from KG4Ex-NOV-Set: {path}")
            for top_k in TOP_KS:
                key = str(top_k)
                for metric in METRICS:
                    values = payload["Fairness"][key] if metric in FAIRNESS else payload[metric][key]
                    value = values[metric] if metric in FAIRNESS else values["mean"]
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Nonfinite {metric} at Top-{top_k}: {path}")
                    if metric in QUALITY and not math.isfinite(float(values["std"])):
                        raise ValueError(f"Nonfinite std for {metric} at Top-{top_k}: {path}")
            result[(dataset, method)] = payload
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, manifest


def metric_values(payload: dict, metric: str, top_k: int) -> tuple[float, float | None]:
    if metric in FAIRNESS:
        return float(payload["Fairness"][str(top_k)][metric]), None
    data = payload[metric][str(top_k)]
    return float(data["mean"]), float(data["std"])


def rank_map(results: dict, dataset: str, methods: tuple[str, ...], metric: str, top_k: int) -> dict[str, int]:
    values = {method: metric_values(results[(dataset, method)], metric, top_k)[0] for method in methods}
    distinct = sorted(set(values.values()), reverse=metric not in LOWER_BETTER)
    return {method: distinct.index(value) + 1 for method, value in values.items()}


def export_csv(results: dict, output: Path) -> None:
    fields = ["dataset", "family", "method", "top_k", "source_metrics"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_rank"))
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for family, methods in (("ablation", ABLATIONS), ("comparison", COMPARISONS)):
                for top_k in TOP_KS:
                    ranks = {metric: rank_map(results, dataset, methods, metric, top_k) for metric in METRICS}
                    for method in methods:
                        row = {
                            "dataset": dataset, "family": family, "method": method,
                            "top_k": top_k, "source_metrics": str(source_path(dataset, method).relative_to(ROOT)),
                        }
                        for metric in METRICS:
                            mean, std = metric_values(results[(dataset, method)], metric, top_k)
                            row[f"{metric}_mean"] = f"{mean:.6f}"
                            row[f"{metric}_std"] = "" if std is None else f"{std:.6f}"
                            row[f"{metric}_rank"] = ranks[metric][method]
                        writer.writerow(row)


def cell(value: str, style: ParagraphStyle, rank: int = 0) -> Paragraph:
    escaped = escape(value)
    if rank == 1:
        escaped = f"<b>{escaped}</b>"
    elif rank == 2:
        escaped = f"<u>{escaped}</u>"
    return Paragraph(escaped, style)


def make_table(results: dict, dataset: str, methods: tuple[str, ...], metrics: tuple[str, ...],
               top_k: int, width: float) -> Table:
    numeric_style = ParagraphStyle("numeric", fontName="Helvetica", fontSize=8.1,
                                   leading=10.1, alignment=TA_CENTER)
    label_style = ParagraphStyle("label", fontName="Helvetica", fontSize=8.4,
                                 leading=10.4, alignment=TA_LEFT)
    header_style = ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=8.1,
                                  leading=10.2, alignment=TA_CENTER)
    names = ["Method"] + [HEADERS[metric] for metric in metrics]
    rows = [[cell(name, header_style) for name in names]]
    rankings = {metric: rank_map(results, dataset, methods, metric, top_k) for metric in metrics}
    for method in methods:
        line = [cell(LABELS.get(method, method), label_style)]
        for metric in metrics:
            mean, std = metric_values(results[(dataset, method)], metric, top_k)
            display = f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"
            line.append(cell(display, numeric_style, rankings[metric][method]))
        rows.append(line)
    label_width = 226
    table = Table(rows, colWidths=[label_width] + [(width - label_width) / len(metrics)] * len(metrics),
                  rowHeights=[27] + [26] * len(methods))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#63788A")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.2, colors.HexColor("#D7DFE6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def draw_page_header(pdf: canvas.Canvas, title: str, subtitle: str, page: int, size: tuple[float, float]) -> None:
    width, height = size
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(42, height - 52, title)
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#526477"))
    pdf.drawString(42, height - 73, subtitle)
    pdf.setStrokeColor(colors.HexColor("#34495E"))
    pdf.line(42, height - 84, width - 42, height - 84)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawRightString(width - 42, 29, f"V5 seed 2024 | page {page}")
    pdf.setFillColor(colors.black)


def draw_cover(pdf: canvas.Canvas, size: tuple[float, float]) -> None:
    width, height = size
    draw_page_header(pdf, "V5 results | Eedi and Algebra2005", "Completed datasets only | 2024 seed | Top-5 to Top-50", 1, size)
    y = height - 143
    lines = [
        ("Scope", "Nine ablation settings and five comparison models per dataset. The full method is repeated in comparison tables."),
        ("Data", "Eedi uses the new split. Both datasets use theta_m=0.8, theta_f=0.8, rec=50 and remove zero-valued exercise-forgetting training edges."),
        ("Ranking", "Within each table column, bold is best and underline is second best, using unrounded values. Ties share a rank; the next distinct value is second."),
        ("Directions", "Ada, NOV, Ep, coverage and long-tail exposure share: higher. Exposure Gini and head exposure share: lower."),
        ("NOV", "NOV is KG4Ex-NOV-Set: Jaccard novelty of the union of Top-N concepts against the student's correct-answer history."),
        ("Uncertainty", "Ada, NOV and Ep cells display mean ± student-level standard deviation. One model seed was run; these are not across-seed deviations."),
        ("Fairness", "Exposure metrics are dataset-level aggregates. Long-tail and head shares are complementary; inspect them alongside coverage and Gini."),
        ("Complete data", "The companion CSV contains all 10 N values (5, 10, ..., 50), all metrics, table ranks and source metric paths. Four curve pages show selected full-range trends."),
    ]
    label_style = ParagraphStyle("cover_label", fontName="Helvetica-Bold", fontSize=11, leading=15)
    body_style = ParagraphStyle("cover_body", fontName="Helvetica", fontSize=10.5, leading=15)
    for label, body in lines:
        label_p = Paragraph(escape(label), label_style)
        body_p = Paragraph(escape(body), body_style)
        label_p.wrapOn(pdf, 180, 30)
        label_p.drawOn(pdf, 50, y - 16)
        _, h = body_p.wrapOn(pdf, width - 270, 54)
        body_p.drawOn(pdf, 218, y - h)
        y -= max(64, h + 17)
    pdf.showPage()


CHART_COLORS = (
    "#24478F", "#CC5A28", "#399474", "#8864A7", "#B78A22",
    "#3284AE", "#A95779", "#333333", "#6D8C44",
)


def draw_chart(pdf: canvas.Canvas, results: dict, dataset: str, methods: tuple[str, ...],
               metric: str, x: float, y: float, width: float, height: float) -> None:
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(x, y + height + 18, HEADERS[metric].replace(" ↑", "").replace(" ↓", ""))
    left, right, bottom, top = x + 43, x + width - 12, y + 28, y + height - 12
    series = {method: [metric_values(results[(dataset, method)], metric, k)[0] for k in TOP_KS]
              for method in methods}
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    margin = max((high - low) * 0.08, 0.005)
    low -= margin
    high += margin
    pdf.setStrokeColor(colors.HexColor("#DCE3EA"))
    pdf.setFillColor(colors.HexColor("#5A6572"))
    pdf.setFont("Helvetica", 7.4)
    for i in range(5):
        yy = bottom + (top - bottom) * i / 4
        pdf.line(left, yy, right, yy)
        pdf.drawRightString(left - 5, yy - 2, f"{low + (high - low) * i / 4:.3f}")
    pdf.setStrokeColor(colors.HexColor("#7F8995"))
    pdf.line(left, bottom, right, bottom)
    pdf.line(left, bottom, left, top)
    for i, k in enumerate(TOP_KS):
        xx = left + (right - left) * i / (len(TOP_KS) - 1)
        pdf.drawCentredString(xx, bottom - 14, str(k))
    for index, method in enumerate(methods):
        pdf.setStrokeColor(colors.HexColor(CHART_COLORS[index]))
        pdf.setFillColor(colors.HexColor(CHART_COLORS[index]))
        pdf.setLineWidth(1.65 if method in ("pre_in_kppd_post", "main") else 1.05)
        points = [
            (left + (right - left) * i / (len(TOP_KS) - 1),
             bottom + (top - bottom) * (value - low) / (high - low))
            for i, value in enumerate(series[method])
        ]
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.drawPath(path)
        for xx, yy in points:
            pdf.circle(xx, yy, 1.9, stroke=0, fill=1)
    pdf.setFillColor(colors.black)


def draw_curves(pdf: canvas.Canvas, results: dict, dataset: str,
                family: str, methods: tuple[str, ...], page: int,
                size: tuple[float, float]) -> None:
    width, height = size
    draw_page_header(pdf, f"{dataset} | {family} | Top-5 to Top-50",
                     "Selected quality and exposure-fairness trends; all metrics are in the companion CSV", page, size)
    chart_width = (width - 110) / 2
    draw_chart(pdf, results, dataset, methods, "Ada", 42, 468, chart_width, 205)
    draw_chart(pdf, results, dataset, methods, "NOV", 68 + chart_width, 468, chart_width, 205)
    draw_chart(pdf, results, dataset, methods, "ItemExposureGini", 42, 192, chart_width, 205)
    draw_chart(pdf, results, dataset, methods, "KCCoverage", 68 + chart_width, 192, chart_width, 205)
    label_style = ParagraphStyle("legend", fontName="Helvetica", fontSize=8.2, leading=10)
    columns = 3
    for index, method in enumerate(methods):
        col, row = index % columns, index // columns
        xx = 52 + col * (width - 104) / columns
        yy = 152 - row * 21
        pdf.setStrokeColor(colors.HexColor(CHART_COLORS[index]))
        pdf.setLineWidth(2.2)
        pdf.line(xx, yy + 4, xx + 21, yy + 4)
        paragraph = Paragraph(escape(LABELS.get(method, method)), label_style)
        paragraph.wrapOn(pdf, (width - 104) / columns - 35, 15)
        paragraph.drawOn(pdf, xx + 29, yy)
    pdf.setFillColor(colors.HexColor("#526477"))
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(42, 46, "Each line uses the same method color across all four charts; the main method has a heavier line.")
    pdf.setFillColor(colors.black)
    pdf.showPage()


def export_pdf(results: dict, output: Path) -> None:
    size = landscape(A3)
    width, height = size
    pdf = canvas.Canvas(str(output), pagesize=size)
    pdf.setTitle("V5 2024 results: Eedi and Algebra2005")
    draw_cover(pdf, size)
    page = 2
    usable = width - 84
    for dataset in DATASETS:
        for family, methods in (("Ablation", ABLATIONS), ("Comparison", COMPARISONS)):
            for top_k in PAPER_KS:
                draw_page_header(pdf, f"{dataset} | {family} | Top-{top_k}",
                                 "Recommendation quality and exposure fairness", page, size)
                y = height - 118
                for heading, metrics in (("Recommendation quality", QUALITY), ("Exposure fairness", FAIRNESS)):
                    pdf.setFont("Helvetica-Bold", 12)
                    pdf.drawString(42, y, heading)
                    y -= 12
                    table = make_table(results, dataset, methods, metrics, top_k, usable)
                    _, table_height = table.wrapOn(pdf, usable, height)
                    table.drawOn(pdf, 42, y - table_height)
                    y -= table_height + 39
                if y < 48:
                    raise ValueError(f"Tables overflow page {page}: y={y}")
                pdf.setFont("Helvetica", 8.5)
                pdf.setFillColor(colors.HexColor("#526477"))
                pdf.drawString(42, 46, "Bold = best; underline = second distinct value within this table. Quality rankings use means.")
                pdf.setFillColor(colors.black)
                pdf.showPage()
                page += 1
    for dataset in DATASETS:
        for family, methods in (("Ablation", ABLATIONS), ("Comparison", COMPARISONS)):
            draw_curves(pdf, results, dataset, family, methods, page, size)
            page += 1
    pdf.save()


def main() -> None:
    results, manifest = load_results()
    output = RUNS / "V5_2024" / "paper_tables_Eedi_Algebra2005_20261019"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "V5_2024_Eedi_Algebra2005_all_topk.csv"
    pdf_path = output / "V5_2024_Eedi_Algebra2005_paper_tables.pdf"
    export_csv(results, csv_path)
    export_pdf(results, pdf_path)
    (output / "source_manifest.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "datasets": DATASETS,
        "top_ks": TOP_KS,
        "paper_top_ks": PAPER_KS,
        "seed": 2024,
        "sources_sha256": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pdf_path)
    print(csv_path)
    print(f"source files: {len(manifest)}")


if __name__ == "__main__":
    main()
