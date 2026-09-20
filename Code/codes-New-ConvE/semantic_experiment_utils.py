"""Shared utilities for 2CKG4ER experiments."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence


MODEL_NAME = "2CKG4ER"
MODEL_VERSION = "2ckg4er_vfin9_kppd"

ABLATION_ALIASES = {
    "feature_only": "2CKG4ER",
    "feature_only_relation_id": "2CKG4ER_relation_id",
    "feature_only_learner_id": "2CKG4ER_learner_id",
    "feature_only_exercise_id": "2CKG4ER_exercise_id",
    "feature_only_no_mastery": "2CKG4ER_no_mastery",
    "feature_only_no_forgetting": "2CKG4ER_no_forgetting",
}

VALID_ABLATIONS = [
    "id_only",
    "2CKG4ER",
    "2CKG4ER_relation_id",
    "2CKG4ER_learner_id",
    "2CKG4ER_exercise_id",
    "2CKG4ER_no_mastery",
    "2CKG4ER_no_forgetting",
]

GRAPH_ABLATIONS = [
    "2CKG4ER_no_mastery",
    "2CKG4ER_no_forgetting",
]

DEFAULT_ALL_ABLATIONS = [
    "id_only",
    "2CKG4ER",
    "2CKG4ER_relation_id",
    "2CKG4ER_learner_id",
    "2CKG4ER_exercise_id",
    "2CKG4ER_no_mastery",
    "2CKG4ER_no_forgetting",
]

DEFAULT_DATASETS = [
    "Eedi",
    "algebra2005",
    "assist2009-sub",
    "statics2011",
    "XES3G5M-sub-small",
]

# Use a regular recommendation-length grid for every experiment and report.
DEFAULT_TOP_KS = list(range(5, 101, 5))


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def kg4er_new_root() -> Path:
    return code_dir().parent


def er_root() -> Path:
    return kg4er_new_root().parent


def project_root() -> Path:
    return er_root().parent


def default_data_root() -> Path:
    # The active workflow keeps raw data, front-model outputs, and ER graphs together.
    # Legacy KG4ER/data remains intentionally outside the execution path.
    return kg4er_new_root() / "Data_Fin"


def default_runs_root() -> Path:
    return kg4er_new_root() / "runs"


def default_ablation_data_root() -> Path:
    return kg4er_new_root() / "ablation_data"


def old_codes_root() -> Path:
    bundled_codes = code_dir()
    if (bundled_codes / "evaluate_recommendations.py").exists():
        return bundled_codes
    return er_root() / "KG4ER" / "codes"


def graph_path_for_dataset(dataset: str, data_root: Path | None = None, graph_subdir: str | None = None) -> Path:
    root = (data_root or default_data_root()) / dataset
    if graph_subdir:
        target = root / graph_subdir
        if (target / "entities.dict").exists():
            return target.resolve()
        raise FileNotFoundError(f"Cannot locate graph files for dataset {dataset} in subdir {graph_subdir}: {target}")
    prepared = root / "prepared_for_kt"
    if (prepared / "entities.dict").exists():
        return prepared.resolve()
    if (root / "entities.dict").exists():
        return root.resolve()
    raise FileNotFoundError(f"Cannot locate graph files for dataset {dataset}: {root}")


def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def parse_ablation_list(value: str | Sequence[str]) -> List[str]:
    items = parse_csv_list(value)
    if any(item.lower() == "all" for item in items):
        if len(items) != 1:
            raise ValueError("Use --ablations all by itself, or provide an explicit comma-separated list.")
        return list(DEFAULT_ALL_ABLATIONS)
    return [normalize_ablation(item) for item in items]


def parse_seed_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_top_ks(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def ablation_model_dir(ablation: str) -> str:
    ablation = normalize_ablation(ablation)
    if ablation == MODEL_NAME:
        return MODEL_NAME
    return f"{MODEL_NAME}_{ablation}" if ablation == "id_only" else ablation


def normalize_ablation(ablation: str) -> str:
    return ABLATION_ALIASES.get(str(ablation), str(ablation))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def python_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - only for environment reporting.
        info["torch_error"] = repr(exc)
    return info


def command_to_markdown(command: Sequence[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if not value:
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in ['"', "'", "`"]):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def run_logged_command(command: Sequence[str], cwd: Path, log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = command_to_markdown(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {command_line}\n")
        log.flush()
        if dry_run:
            log.write("[dry-run] command not executed\n")
            return 0
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"[exit_code] {completed.returncode}\n")
        return int(completed.returncode)


def stage_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    return data.get("status") == "completed"


def mark_stage(path: Path, status: str, **extra: Any) -> None:
    payload = {"status": status, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    payload.update(extra)
    write_json(path, payload)
