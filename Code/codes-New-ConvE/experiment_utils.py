import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


def make_run_id(dataset, model, seed=None, now=None):
    now = now or datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    parts = [str(dataset), str(model), stamp]
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(part.replace(" ", "-") for part in parts if part)


def resolve_run_dir(run_root, dataset, model, run_id=None, seed=None):
    run_root = Path(run_root)
    run_id = run_id or make_run_id(dataset, model, seed=seed)
    return run_root / str(dataset) / run_id


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def set_random_seed(seed, deterministic=False):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


@contextmanager
def timed_stage(timing_path, stage_name):
    timing_path = Path(timing_path)
    timings = load_json(timing_path, default={}) or {}
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings[stage_name] = round(elapsed, 6)
    write_json(timings, timing_path)


def update_timing(timing_path, stage_name, seconds, extra=None):
    timing_path = Path(timing_path)
    timings = load_json(timing_path, default={}) or {}
    payload = {"seconds": round(float(seconds), 6)}
    if extra:
        payload.update(extra)
    timings[stage_name] = payload
    write_json(timings, timing_path)


def summarize_seed_metrics(metric_files, output_path):
    rows = []
    for metric_file in metric_files:
        payload = load_json(metric_file, default={})
        if payload:
            rows.append(payload)

    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary = {"runs": rows, "summary": {}}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary["summary"][key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    write_json(summary, output_path)
    return summary
