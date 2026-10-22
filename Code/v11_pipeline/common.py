from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASETS = ["Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small"]


def kg4er_new_root() -> Path:
    return Path(__file__).resolve().parents[1]


def er_root() -> Path:
    return kg4er_new_root().parent


def project_root() -> Path:
    return er_root().parent


def source_data_root() -> Path:
    return er_root() / "KG4ER" / "data"


def output_data_root() -> Path:
    return kg4er_new_root() / "data"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_large_csv_field_limit(min_limit: int = 1024 * 1024 * 1024) -> None:
    current = csv.field_size_limit()
    if current >= min_limit:
        return
    limit = min_limit
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def source_dataset_dir(dataset: str, data_root: Path | None = None) -> Path:
    root = data_root or source_data_root()
    path = root / dataset
    if not path.exists():
        raise FileNotFoundError(f"source dataset directory not found: {path}")
    return path


def v11_dataset_dir(dataset: str, data_root: Path | None = None) -> Path:
    return (data_root or output_data_root()) / dataset / "v11"


def resolve_source_data_root(dataset: str, explicit_root: Path | None = None, output_root: Path | None = None) -> Path:
    """Resolve immutable KG4ER source data without treating V11 outputs as source."""
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    env_root = os.environ.get("KG4ER_SOURCE_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(source_data_root())

    manifest_path = v11_dataset_dir(dataset, output_root) / "v11_graph_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            graph_path = manifest.get("source_graph_dir")
            if graph_path:
                graph_dir = Path(graph_path)
                dataset_dir = graph_dir.parent if graph_dir.name == "prepared_for_kt" else graph_dir
                candidates.append(dataset_dir.parent)
        except (json.JSONDecodeError, OSError):
            pass

    for root in candidates:
        if (root / dataset).exists():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot find immutable source data for {dataset}. Checked: {searched}. "
        "Pass --source-data-root or set KG4ER_SOURCE_DATA_ROOT."
    )


def locate_source_graph_dir(dataset_dir: Path) -> Path:
    prepared = dataset_dir / "prepared_for_kt"
    if (prepared / "entities.dict").exists():
        return prepared
    if (dataset_dir / "entities.dict").exists():
        return dataset_dir
    raise FileNotFoundError(f"cannot locate source graph under {dataset_dir}")


def locate_sequence_interactions(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "prepared_for_kt" / "sequence_interactions.csv",
        dataset_dir / "processed" / "sequence_interactions.csv",
        dataset_dir / "sequence_interactions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"cannot locate sequence_interactions.csv under {dataset_dir}")


def locate_test_sequences(dataset: str, dataset_dir: Path, graph_dir: Path) -> Path:
    candidates = [graph_dir / "test_sequences.csv", dataset_dir / "prepared_for_kt" / "test_sequences.csv"]
    if dataset == "Eedi":
        candidates.append(er_root() / "pykt-toolkit-main" / "data" / "Eedi" / "test_sequences.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"cannot locate test_sequences.csv for {dataset}")


def locate_source_feature_dir(dataset: str, dataset_dir: Path, graph_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "er_v8" / "semantic_kg_features",
        graph_dir / "semantic_kg_features",
        dataset_dir / "semantic_kg_features_v8",
        dataset_dir / "semantic_kg_features",
        kg4er_new_root() / "data" / dataset / "er_v8" / "semantic_kg_features",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def read_entity_dict(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split("\t")
            result[name] = int(idx)
    return result


def entity_count(entity2id: dict[str, int], prefix: str) -> int:
    return sum(1 for name in entity2id if name.startswith(prefix))


def read_q_matrix(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append([int(float(x)) for x in re.split(r"[,\s]+", line) if x != ""])
    if not rows:
        raise ValueError(f"empty Q matrix: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"ragged Q matrix: {path}")
    return np.asarray(rows, dtype=np.int64)


def parse_concepts(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [int(x) for x in re.findall(r"-?\d+", str(value))]


def numeric_suffix(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        raise ValueError(f"{name!r} does not start with {prefix!r}")
    return int(name[len(prefix) :])


def sorted_entities(entity2id: dict[str, int], prefix: str) -> list[str]:
    return sorted([name for name in entity2id if name.startswith(prefix)], key=lambda x: numeric_suffix(x, prefix))


def relation_label(prefix: str, value: float) -> str:
    clipped = max(0.0, min(1.0, float(value)))
    return f"{prefix}{clipped:.2f}"


def write_fixed_relations(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["rec"]
    for prefix in ["mlkc", "pkc", "exfr"]:
        names.extend(f"{prefix}{i / 100:.2f}" for i in range(101))
    with path.open("w", encoding="utf-8") as fp:
        for idx, name in enumerate(names):
            fp.write(f"{idx}\t{name}\n")


def read_triple_students(path: Path) -> set[int]:
    students: set[int] = set()
    if not path.exists():
        return students
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            for item in (parts[0], parts[2]):
                if item.startswith("uid") and item[3:].isdigit():
                    students.add(int(item[3:]))
    return students


def matrix_stats(matrix: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return {"shape": list(arr.shape), "min": None, "max": None, "mean": None, "std": None, "unique_rows": 0}
    if arr.ndim == 1:
        unique_rows = len(set(arr.tolist()))
    else:
        rounded = np.round(arr, 8)
        unique_rows = int(np.unique(rounded, axis=0).shape[0])
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "unique_rows": unique_rows,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_matrix_json(path: Path) -> np.ndarray:
    data = read_json(path)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D matrix")
    return arr


def write_matrix_json(path: Path, matrix: np.ndarray, decimals: int = 6) -> None:
    rounded = np.round(np.asarray(matrix, dtype=np.float64), decimals)
    write_json(path, rounded.tolist())


def minmax(values: np.ndarray, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    if maximum - minimum <= 1e-12:
        return np.full_like(arr, float(default), dtype=np.float64)
    return (arr - minimum) / (maximum - minimum)


def pick_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
