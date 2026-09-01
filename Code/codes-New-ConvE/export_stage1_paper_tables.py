"""Export fixed Top-K paper tables from stage-1 fairness comparison results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_METHODS = ("baseline", "item_only", "kc_only", "item_kc_rerank", "quota_ratio_hybrid")
DEFAULT_TOP_KS = (10, 20, 50)
OUTPUT_FIELDS = (
    "method",
    "K",
    "Ada",
    "NOV",
    "ItemGini",
    "KCGini",
    "ItemCoverage",
    "KCCoverage",
    "LongTailItemShare",
    "LongTailKCShare",
)
SOURCE_FIELDS = {
    "ItemGini": "ItemExposureGini",
    "KCGini": "KCExposureGini",
    "LongTailItemShare": "LongTailItemExposureShare",
    "LongTailKCShare": "LongTailKCExposureShare",
}


def parse_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_int_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return f"{float(text):.6f}"
    except ValueError:
        return text


def table_value(row: Dict[str, str], field: str) -> str:
    if field == "method":
        return str(row.get("method", "")).strip()
    if field == "K":
        return str(int(float(str(row.get("K", "0")).strip())))
    source_field = SOURCE_FIELDS.get(field, field)
    return normalize_value(row.get(source_field, ""))


def filter_rows(rows: Sequence[Dict[str, str]], methods: Sequence[str], top_ks: Sequence[int]) -> List[Dict[str, str]]:
    method_order = {method: idx for idx, method in enumerate(methods)}
    top_k_order = {int(top_k): idx for idx, top_k in enumerate(top_ks)}
    selected = [
        row
        for row in rows
        if row.get("method") in method_order and int(row.get("K", -1)) in top_k_order
    ]
    return sorted(selected, key=lambda row: (top_k_order[int(row["K"])], method_order[row["method"]]))


def output_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for row in rows:
        output.append({field: table_value(row, field) for field in OUTPUT_FIELDS})
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Dict[str, str]], top_ks: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage-1 Paper Tables",
        "",
        "Fields: Ada, NOV, ItemGini, KCGini, ItemCoverage, KCCoverage, LongTailItemShare, LongTailKCShare.",
        "",
    ]
    headers = [
        "方法",
        "Ada",
        "NOV",
        "ItemGini",
        "KCGini",
        "ItemCoverage",
        "KCCoverage",
        "LongTailItemShare",
        "LongTailKCShare",
    ]
    fields = [
        "method",
        "Ada",
        "NOV",
        "ItemGini",
        "KCGini",
        "ItemCoverage",
        "KCCoverage",
        "LongTailItemShare",
        "LongTailKCShare",
    ]
    for top_k in top_ks:
        k_rows = [row for row in rows if int(row["K"]) == int(top_k)]
        if not k_rows:
            continue
        lines.append(f"## Top-{top_k}")
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
        for row in k_rows:
            lines.append("| " + " | ".join(row[field] for field in fields) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_tables(comparison_csv: Path, output_dir: Path, methods: Sequence[str], top_ks: Sequence[int]) -> Dict[str, Path]:
    rows = output_rows(filter_rows(read_rows(comparison_csv), methods, top_ks))
    if not rows:
        raise ValueError(f"No matching rows found in {comparison_csv}")
    csv_path = output_dir / "stage1_topk_paper_tables.csv"
    md_path = output_dir / "stage1_topk_paper_tables.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, top_ks)
    return {"csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed stage-1 Top-K paper tables.")
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-ks", default=",".join(str(top_k) for top_k in DEFAULT_TOP_KS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_tables(
        comparison_csv=args.comparison_csv,
        output_dir=args.output_dir,
        methods=parse_list(args.methods),
        top_ks=parse_int_list(args.top_ks),
    )
    print(f"stage-1 paper tables written to {paths['markdown']} and {paths['csv']}")


if __name__ == "__main__":
    main()
