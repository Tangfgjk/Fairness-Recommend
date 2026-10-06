import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TOP_KS = tuple(range(5, 51, 5))
METRICS = ("ACC", "NOV", "Ep_t0.4", "Ep_t0.5", "Ep_t0.6", "Ep_t0.7")
EP_THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
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
    if payload.get("ep_thresholds") != list(EP_THRESHOLDS):
        raise ValueError(f"Expected four Ep response thresholds in {path}")

    for metric_name in METRICS:
        metric = payload.get(metric_name)
        if not isinstance(metric, dict):
            raise ValueError(f"Missing {metric_name} object in {path}")
        for top_k in TOP_KS:
            item = metric.get(str(top_k))
            if not isinstance(item, dict) or item.get("mean") is None:
                raise ValueError(f"Missing {metric_name}@{top_k} mean in {path}")



def _payload_to_row(payload, path, source):
    row = {
        "dataset": payload["dataset"],
        "model": payload["model"],
        "seed": payload.get("seed"),
        "source": source,
        "metrics_file": str(Path(path).resolve()),
        "ep_thresholds": tuple(payload["ep_thresholds"]),
    }
    for metric in METRICS:
        for top_k in TOP_KS:
            row[f"{metric}@{top_k}"] = _round(payload[metric][str(top_k)]["mean"])
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
    thresholds = {row["ep_thresholds"] for row in rows}
    if thresholds != {EP_THRESHOLDS}:
        raise ValueError(f"Mixed Ep thresholds in {run_dir}: {sorted(thresholds)}")
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

    metric_names = [f"{metric}@{top_k}" for metric in METRICS for top_k in TOP_KS]
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
            values = [float(item[metric_name]) for item in items]
            output[f"{metric_name}_mean"] = _round(mean(values))
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
    fields = ["dataset", "model", "seed", "source", "metrics_file", "timing_file", "ep_thresholds"]
    fields += [f"{metric}@{top_k}" for metric in METRICS for top_k in TOP_KS]
    fields += [
        "Training Seconds",
        "Inference Seconds",
        "Evaluation Seconds",
        "Total Seconds",
    ]
    return fields


def _summary_fieldnames():
    fields = ["dataset", "model", "run_count", "seeds"]
    for prefix in METRICS:
        for top_k in TOP_KS:
            fields.extend([f"{prefix}@{top_k}_mean", f"{prefix}@{top_k}_std"])
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
            if isinstance(output.get("ep_thresholds"), tuple):
                output["ep_thresholds"] = ",".join(str(value) for value in output["ep_thresholds"])
            writer.writerow(output)


def _format_summary(value, std):
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
    lines = [f"# {dataset} experiment result summary", "",
             "ACC, NOV, and Ep are reported at N=5,10,...,50.",
             "Ep uses the same frozen readout and four response thresholds for every model in this dataset.",
             "Seeded models show mean ± sample standard deviation across seeds; single-run baselines show a mean.", ""]
    for metric in METRICS:
        headers = ["Model", "Runs"] + [f"@{top_k}" for top_k in TOP_KS]
        table_rows = []
        for item in summary:
            table_rows.append([item["model"], item["run_count"]] + [
                _format_summary(item[f"{metric}@{top_k}_mean"], item[f"{metric}@{top_k}_std"])
                for top_k in TOP_KS])
        lines.extend([f"## {metric}", ""])
        lines.extend(_markdown_table(headers, table_rows, range(1, len(headers))))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_metrics_csv(summary, path):
    fieldnames = ["Metric", "N"] + [item["model"] for item in summary]
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRICS:
            for top_k in TOP_KS:
                row = {"Metric": metric, "N": top_k}
                for item in summary:
                    row[item["model"]] = _format_summary(
                        item[f"{metric}@{top_k}_mean"], item[f"{metric}@{top_k}_std"])
                writer.writerow(row)


def write_paper_metrics_markdown(dataset, summary, metadata, path):
    lines = [f"# {dataset} paper-style per-N metrics", "",
             "Mean ± sample standard deviation across seeds; deterministic baselines show the mean.", ""]
    for metric in METRICS:
        headers = ["N"] + [item["model"] for item in summary]
        table_rows = []
        for top_k in TOP_KS:
            table_rows.append([top_k] + [
                _format_summary(item[f"{metric}@{top_k}_mean"], item[f"{metric}@{top_k}_std"])
                for item in summary])
        lines.extend([f"## {metric}", ""])
        lines.extend(_markdown_table(headers, table_rows, range(len(headers))))
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
