import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TOP_KS = tuple(range(5, 101, 5))
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
PREFERRED_MODEL_ORDER = (
    "ConvE_full",
    "ConvE_no_seq",
    "ConvE_no_forgetting",
    "ConvE_no_mastery",
    "TransE",
    "TransE-adv",
    "RotatE",
    "DistMult",
    "ComplEx",
    "EB-CF",
    "SB-CF",
    "CBF",
)


def _model_sort_key(model):
    try:
        return (0, PREFERRED_MODEL_ORDER.index(model))
    except ValueError:
        return (1, str(model))


def _round(value):
    return round(float(value), 6)


def _maybe_round(value):
    if value is None:
        return None
    return _round(value)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_complete_timing(metrics_path):
    timing_path = Path(metrics_path).parents[1] / "timing.json"
    empty = {
        "Training Seconds": None,
        "Inference Seconds": None,
        "Evaluation Seconds": None,
        "Total Seconds": None,
        "timing_file": None,
    }
    if not timing_path.is_file():
        return empty

    payload = _load_json(timing_path)
    values = []
    for section in ("training", "inference_without_cache", "evaluation_metric"):
        item = payload.get(section)
        seconds = item.get("seconds") if isinstance(item, dict) else None
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return empty
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            return empty
        values.append(seconds)

    return {
        "Training Seconds": _round(values[0]),
        "Inference Seconds": _round(values[1]),
        "Evaluation Seconds": _round(values[2]),
        "Total Seconds": _round(sum(values)),
        "timing_file": str(timing_path.resolve()),
    }


def discover_metrics(run_dir):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    metric_files = sorted(run_dir.glob("**/eval/metrics.json"))
    if not metric_files:
        raise ValueError(f"No eval/metrics.json files found under: {run_dir}")
    return metric_files


def validate_metric_payload(payload, dataset, path):
    actual_dataset = payload.get("dataset")
    if actual_dataset != dataset:
        raise ValueError(
            f"Dataset mismatch in {path}: expected {dataset!r}, got {actual_dataset!r}"
        )
    model = payload.get("model")
    if not model:
        raise ValueError(f"Missing model in {path}")

    for metric_name in ("Ada", "NOV"):
        metric = payload.get(metric_name)
        if metric_name == "Ada" and metric is None:
            metric = payload.get("ACC")
        if not isinstance(metric, dict):
            raise ValueError(f"Missing {metric_name} object in {path}")
        for top_k in TOP_KS:
            item = metric.get(str(top_k))
            if not isinstance(item, dict) or item.get("mean") is None:
                raise ValueError(f"Missing {metric_name}@{top_k} mean in {path}")

    ep_sim = payload.get("Ep_sim")
    if not isinstance(ep_sim, dict) or ep_sim.get("mean") is None:
        raise ValueError(f"Missing Ep_sim@10 mean in {path}")
    if int(ep_sim.get("top_k", -1)) != 10:
        raise ValueError(
            f"Expected Ep_sim@10 in {path}, got Ep_sim@{ep_sim.get('top_k')}"
        )

    fairness = payload.get("Fairness")
    if fairness is not None:
        if not isinstance(fairness, dict):
            raise ValueError(f"Invalid Fairness object in {path}")
        for top_k in TOP_KS:
            item = fairness.get(str(top_k))
            if item is None:
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Invalid Fairness@{top_k} object in {path}")


def _payload_to_row(payload, path, source):
    ada_payload = payload.get("Ada") or payload.get("ACC")
    fairness_payload = payload.get("Fairness") or {}
    row = {
        "dataset": payload["dataset"],
        "model": payload["model"],
        "seed": payload.get("seed"),
        "source": source,
        "metrics_file": str(Path(path).resolve()),
    }
    for top_k in TOP_KS:
        row[f"Ada@{top_k}"] = _round(ada_payload[str(top_k)]["mean"])
        row[f"NOV@{top_k}"] = _round(payload["NOV"][str(top_k)]["mean"])
        fairness_at_k = fairness_payload.get(str(top_k), {})
        for metric_name in FAIRNESS_METRICS:
            row[f"{metric_name}@{top_k}"] = _maybe_round(fairness_at_k.get(metric_name))
    row["Ep_sim@10"] = _round(payload["Ep_sim"]["mean"])
    row.update(_load_complete_timing(path))
    return row


def _load_rows(dataset, run_dir, source):
    rows = {}
    for path in discover_metrics(run_dir):
        payload = _load_json(path)
        validate_metric_payload(payload, dataset, path)
        row = _payload_to_row(payload, path, source)
        key = (row["model"], row["seed"])
        if key in rows:
            raise ValueError(
                f"Duplicate result for model={key[0]!r}, seed={key[1]!r} under {run_dir}"
            )
        rows[key] = row
    return rows


def validate_replacement_options(
    replacement_run_dir, replacement_models, replacement_seeds
):
    options = (
        replacement_run_dir is not None,
        bool(replacement_models),
        bool(replacement_seeds),
    )
    if any(options) and not all(options):
        raise ValueError(
            "--replacement-run-dir, --replacement-models, and --replacement-seeds "
            "must be provided together"
        )


def collect_dataset_results(
    dataset,
    run_dir,
    replacement_run_dir=None,
    replacement_models=(),
    replacement_seeds=(),
):
    run_dir = Path(run_dir).resolve()
    replacement_models = tuple(replacement_models)
    replacement_seeds = tuple(int(seed) for seed in replacement_seeds)
    validate_replacement_options(
        replacement_run_dir, replacement_models, replacement_seeds
    )

    rows_by_key = _load_rows(dataset, run_dir, "primary")
    primary_count = len(rows_by_key)
    replacement_count = 0
    replacement_dir = None

    if replacement_run_dir is not None:
        replacement_dir = Path(replacement_run_dir).resolve()
        replacement_rows = _load_rows(dataset, replacement_dir, "replacement")
        requested_keys = {
            (model, seed)
            for model in replacement_models
            for seed in replacement_seeds
        }
        missing_replacements = sorted(requested_keys - set(replacement_rows))
        if missing_replacements:
            raise ValueError(
                "Replacement results not found for: "
                + ", ".join(
                    f"{model}/seed{seed}" for model, seed in missing_replacements
                )
            )
        missing_primary = sorted(requested_keys - set(rows_by_key))
        if missing_primary:
            raise ValueError(
                "Replacement target is absent from the primary run: "
                + ", ".join(
                    f"{model}/seed{seed}" for model, seed in missing_primary
                )
            )
        for key in requested_keys:
            rows_by_key[key] = replacement_rows[key]
            replacement_count += 1

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            _model_sort_key(row["model"]),
            row["seed"] is None,
            row["seed"] if row["seed"] is not None else 0,
        ),
    )
    metadata = {
        "run_dir": str(run_dir),
        "primary_result_count": primary_count,
        "replacement_run_dir": str(replacement_dir) if replacement_dir else None,
        "replacement_models": list(replacement_models),
        "replacement_seeds": list(replacement_seeds),
        "replacement_count": replacement_count,
        "final_result_count": len(rows),
    }
    return rows, metadata


def aggregate_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    metric_names = [f"Ada@{top_k}" for top_k in TOP_KS]
    metric_names += [f"NOV@{top_k}" for top_k in TOP_KS]
    for fairness_metric in FAIRNESS_METRICS:
        metric_names += [f"{fairness_metric}@{top_k}" for top_k in TOP_KS]
    metric_names.append("Ep_sim@10")
    summary = []
    for model in sorted(grouped, key=_model_sort_key):
        items = grouped[model]
        seeds = [item["seed"] for item in items]
        if any(seed is None for seed in seeds) and any(seed is not None for seed in seeds):
            raise ValueError(
                f"Model {model!r} mixes seeded and deterministic results"
            )
        output = {
            "dataset": items[0]["dataset"],
            "model": model,
            "run_count": len(items),
            "seeds": [seed for seed in seeds if seed is not None],
        }
        for metric_name in metric_names:
            values = [
                float(item[metric_name])
                for item in items
                if item.get(metric_name) is not None
            ]
            output[f"{metric_name}_mean"] = _round(mean(values)) if values else None
            output[f"{metric_name}_std"] = (
                _round(stdev(values))
                if len(values) > 1 and all(seed is not None for seed in seeds)
                else None
            )
        timed_items = [item for item in items if item["Total Seconds"] is not None]
        if timed_items:
            max_time_item = max(timed_items, key=lambda item: item["Total Seconds"])
            output.update(
                {
                    "max_time_seed": max_time_item["seed"],
                    "max_training_seconds": max_time_item["Training Seconds"],
                    "max_inference_seconds": max_time_item["Inference Seconds"],
                    "max_evaluation_seconds": max_time_item["Evaluation Seconds"],
                    "max_total_seconds": max_time_item["Total Seconds"],
                }
            )
        else:
            output.update(
                {
                    "max_time_seed": None,
                    "max_training_seconds": None,
                    "max_inference_seconds": None,
                    "max_evaluation_seconds": None,
                    "max_total_seconds": None,
                }
            )
        summary.append(output)
    return summary


def _per_seed_fieldnames():
    fields = ["dataset", "model", "seed", "source", "metrics_file", "timing_file"]
    fields += [f"Ada@{top_k}" for top_k in TOP_KS]
    fields += [f"NOV@{top_k}" for top_k in TOP_KS]
    for metric_name in FAIRNESS_METRICS:
        fields += [f"{metric_name}@{top_k}" for top_k in TOP_KS]
    fields.append("Ep_sim@10")
    fields += [
        "Training Seconds",
        "Inference Seconds",
        "Evaluation Seconds",
        "Total Seconds",
    ]
    return fields


def _summary_fieldnames():
    fields = ["dataset", "model", "run_count", "seeds"]
    for prefix in ("Ada", "NOV", *FAIRNESS_METRICS):
        for top_k in TOP_KS:
            fields.extend([f"{prefix}@{top_k}_mean", f"{prefix}@{top_k}_std"])
    fields.extend(["Ep_sim@10_mean", "Ep_sim@10_std"])
    fields.extend(
        [
            "max_time_seed",
            "max_training_seconds",
            "max_inference_seconds",
            "max_evaluation_seconds",
            "max_total_seconds",
        ]
    )
    return fields


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if isinstance(output.get("seeds"), list):
                output["seeds"] = ",".join(str(seed) for seed in output["seeds"])
            writer.writerow(output)


def _format_summary(value, std):
    if value is None:
        return ""
    if std is None:
        return f"{value:.6f}"
    return f"{value:.6f} ± {std:.6f}"


def _format_optional(value):
    return "" if value is None else f"{float(value):.6f}"


def _markdown_table(headers, rows, right_aligned_columns=()):
    lines = ["| " + " | ".join(headers) + " |"]
    separator = []
    for index in range(len(headers)):
        separator.append("---:" if index in right_aligned_columns else "---")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_markdown_report(dataset, rows, summary, metadata, path):
    lines = [
        f"# {dataset} experiment result summary",
        "",
        "## Statistical scope",
        "",
        "- Metrics: `Ada/NOV/Fairness @ 5,10,15,...,100` and `Ep_sim@10`.",
        "- Per-seed tables retain the mean stored in each `eval/metrics.json`.",
        "- Model summaries report `mean ± sample std`; sample std uses `ddof=1`.",
        "- Deterministic single-run baselines have no cross-seed standard deviation.",
        "- Maximum complete runtime selects the seed with the largest sum of training, uncached inference, and metric-evaluation seconds; no runtime mean or std is calculated.",
        f"- Final result rows: {metadata['final_result_count']}; replaced rows: {metadata['replacement_count']}.",
        "",
    ]
    if metadata["replacement_run_dir"]:
        lines.extend(
            [
                "### Replacement source",
                "",
                f"- Primary run: `{metadata['run_dir']}`",
                f"- Replacement run: `{metadata['replacement_run_dir']}`",
                f"- Models: `{','.join(metadata['replacement_models'])}`",
                f"- Seeds: `{','.join(str(seed) for seed in metadata['replacement_seeds'])}`",
                "",
            ]
        )

    lines.extend(["## Model mean ± sample std", "", "### Ada", ""])
    acc_headers = ["Model", "Runs"] + [f"Ada@{top_k}" for top_k in TOP_KS]
    acc_rows = []
    for item in summary:
        acc_rows.append(
            [item["model"], item["run_count"]]
            + [
                _format_summary(
                    item[f"Ada@{top_k}_mean"], item[f"Ada@{top_k}_std"]
                )
                for top_k in TOP_KS
            ]
        )
    lines.extend(_markdown_table(acc_headers, acc_rows, range(1, len(acc_headers))))

    lines.extend(["", "### NOV and Ep_sim", ""])
    nov_headers = ["Model", "Runs"] + [f"NOV@{top_k}" for top_k in TOP_KS]
    nov_headers.append("Ep_sim@10")
    nov_rows = []
    for item in summary:
        nov_rows.append(
            [item["model"], item["run_count"]]
            + [
                _format_summary(
                    item[f"NOV@{top_k}_mean"], item[f"NOV@{top_k}_std"]
                )
                for top_k in TOP_KS
            ]
            + [
                _format_summary(
                    item["Ep_sim@10_mean"], item["Ep_sim@10_std"]
                )
            ]
        )
    lines.extend(_markdown_table(nov_headers, nov_rows, range(1, len(nov_headers))))

    for fairness_metric in FAIRNESS_METRICS:
        lines.extend(["", f"### {fairness_metric}", ""])
        headers = ["Model", "Runs"] + [f"{fairness_metric}@{top_k}" for top_k in TOP_KS]
        metric_rows = []
        for item in summary:
            metric_rows.append(
                [item["model"], item["run_count"]]
                + [
                    _format_summary(
                        item[f"{fairness_metric}@{top_k}_mean"],
                        item[f"{fairness_metric}@{top_k}_std"],
                    )
                    for top_k in TOP_KS
                ]
            )
        lines.extend(_markdown_table(headers, metric_rows, range(1, len(headers))))

    lines.extend(["", "### Maximum complete runtime", ""])
    timing_headers = [
        "Model",
        "Seed",
        "Training (s)",
        "Inference (s)",
        "Evaluation (s)",
        "Total (s)",
    ]
    timing_rows = [
        [
            item["model"],
            item["max_time_seed"] if item["max_time_seed"] is not None else "-",
            _format_optional(item["max_training_seconds"]),
            _format_optional(item["max_inference_seconds"]),
            _format_optional(item["max_evaluation_seconds"]),
            _format_optional(item["max_total_seconds"]),
        ]
        for item in summary
    ]
    lines.extend(
        _markdown_table(timing_headers, timing_rows, range(1, len(timing_headers)))
    )

    lines.extend(["", "## Per-seed raw means", "", "### Ada", ""])
    raw_acc_headers = ["Model", "Seed", "Source"] + [
        f"Ada@{top_k}" for top_k in TOP_KS
    ]
    raw_acc_rows = [
        [
            item["model"],
            item["seed"] if item["seed"] is not None else "-",
            item["source"],
        ]
        + [f"{item[f'Ada@{top_k}']:.6f}" for top_k in TOP_KS]
        for item in rows
    ]
    lines.extend(
        _markdown_table(raw_acc_headers, raw_acc_rows, range(1, len(raw_acc_headers)))
    )

    lines.extend(["", "### NOV and Ep_sim", ""])
    raw_nov_headers = ["Model", "Seed", "Source"] + [
        f"NOV@{top_k}" for top_k in TOP_KS
    ]
    raw_nov_headers.append("Ep_sim@10")
    raw_nov_rows = [
        [
            item["model"],
            item["seed"] if item["seed"] is not None else "-",
            item["source"],
        ]
        + [f"{item[f'NOV@{top_k}']:.6f}" for top_k in TOP_KS]
        + [f"{item['Ep_sim@10']:.6f}"]
        for item in rows
    ]
    lines.extend(
        _markdown_table(raw_nov_headers, raw_nov_rows, range(1, len(raw_nov_headers)))
    )
    for fairness_metric in FAIRNESS_METRICS:
        lines.extend(["", f"### {fairness_metric}", ""])
        raw_headers = ["Model", "Seed", "Source"] + [
            f"{fairness_metric}@{top_k}" for top_k in TOP_KS
        ]
        raw_rows = [
            [
                item["model"],
                item["seed"] if item["seed"] is not None else "-",
                item["source"],
            ]
            + [
                ""
                if item.get(f"{fairness_metric}@{top_k}") is None
                else f"{item[f'{fairness_metric}@{top_k}']:.6f}"
                for top_k in TOP_KS
            ]
            for item in rows
        ]
        lines.extend(_markdown_table(raw_headers, raw_rows, range(1, len(raw_headers))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_metrics_csv(summary, path):
    fieldnames = ["Model", "Runs", "Metric"]
    for top_k in TOP_KS:
        fieldnames.extend([f"@{top_k} Mean", f"@{top_k} Std"])
    fieldnames.extend(["Ep_sim@10 Mean", "Ep_sim@10 Std"])
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            for metric in ("Ada", "NOV", *FAIRNESS_METRICS):
                row = {"Model": item["model"], "Runs": item["run_count"], "Metric": metric}
                for top_k in TOP_KS:
                    row[f"@{top_k} Mean"] = item[f"{metric}@{top_k}_mean"]
                    row[f"@{top_k} Std"] = item[f"{metric}@{top_k}_std"]
                row["Ep_sim@10 Mean"] = item["Ep_sim@10_mean"] if metric == "NOV" else ""
                row["Ep_sim@10 Std"] = item["Ep_sim@10_std"] if metric == "NOV" else ""
                writer.writerow(row)


def write_paper_metrics_markdown(dataset, summary, metadata, path):
    lines = [
        f"# {dataset} paper-style per-K metrics",
        "",
        "- K: `5, 10, 15, ..., 100`",
        "- Seeded models: mean and sample Std over available seeds.",
        "- Deterministic baselines: mean is reported and Std is blank.",
        f"- Replaced rows: {metadata['replacement_count']}.",
        "",
    ]
    for metric in ("Ada", "NOV", *FAIRNESS_METRICS):
        headers = ["Model", "Runs"] + [f"@{top_k}" for top_k in TOP_KS]
        if metric == "NOV":
            headers.append("Ep_sim@10")
        rows = []
        for item in summary:
            row = [item["model"], item["run_count"]]
            row.extend(
                _format_summary(item[f"{metric}@{top_k}_mean"], item[f"{metric}@{top_k}_std"])
                for top_k in TOP_KS
            )
            if metric == "NOV":
                row.append(_format_summary(item["Ep_sim@10_mean"], item["Ep_sim@10_std"]))
            rows.append(row)
        lines.extend([f"## {metric}", ""])
        lines.extend(_markdown_table(headers, rows, range(1, len(headers))))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(dataset, rows, summary, metadata, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "per_seed_metrics.csv"
    summary_path = output_dir / "model_mean_std.csv"
    json_path = output_dir / "dataset_metrics.json"
    markdown_path = output_dir / "dataset_metrics.md"
    paper_csv_path = output_dir / "paper_metrics_table.csv"
    paper_markdown_path = output_dir / "paper_metrics_table.md"

    _write_csv(per_seed_path, rows, _per_seed_fieldnames())
    _write_csv(summary_path, summary, _summary_fieldnames())
    json_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "top_ks": list(TOP_KS),
                "metadata": metadata,
                "per_seed": rows,
                "model_mean_std": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown_report(dataset, rows, summary, metadata, markdown_path)
    write_paper_metrics_csv(summary, paper_csv_path)
    write_paper_metrics_markdown(dataset, summary, metadata, paper_markdown_path)
    return [
        per_seed_path,
        summary_path,
        json_path,
        markdown_path,
        paper_csv_path,
        paper_markdown_path,
    ]


def _split_csv(value):
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize one dataset's recommendation experiment results."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replacement-run-dir", type=Path)
    parser.add_argument("--replacement-models", default="")
    parser.add_argument("--replacement-seeds", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    replacement_models = _split_csv(args.replacement_models)
    try:
        replacement_seeds = tuple(
            int(seed) for seed in _split_csv(args.replacement_seeds)
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --replacement-seeds: {exc}") from exc

    try:
        rows, metadata = collect_dataset_results(
            args.dataset,
            args.run_dir,
            replacement_run_dir=args.replacement_run_dir,
            replacement_models=replacement_models,
            replacement_seeds=replacement_seeds,
        )
        summary = aggregate_results(rows)
        output_dir = args.output_dir or args.run_dir / "summaries" / "full_metrics"
        written = write_report(args.dataset, rows, summary, metadata, output_dir)
    except ValueError as exc:
        raise SystemExit(f"Result summarization failed: {exc}") from exc

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "result_rows": len(rows),
                "model_summaries": len(summary),
                "replaced_rows": metadata["replacement_count"],
                "output_dir": str(Path(output_dir).resolve()),
                "files": [str(path) for path in written],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
