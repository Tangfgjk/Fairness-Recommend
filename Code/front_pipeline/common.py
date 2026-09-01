from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "Eedi",
    "algebra2005",
    "assist2009-sub",
    "statics2011",
    "XES3G5M-sub-small",
]

RAW_COMPACT_DATASETS = {"assist2009-sub", "XES3G5M-sub-small"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_fin_root(explicit_root: Path | None = None) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root)
    else:
        root = Path(os.environ.get("KG4ER_DATA_FIN_ROOT", project_root() / "Data_Fin"))
    if not root.exists():
        raise FileNotFoundError(
            f"Data_Fin root does not exist: {root}. "
            "Set KG4ER_DATA_FIN_ROOT or pass --data-fin-root."
        )
    return root


def dataset_root(dataset: str, explicit_root: Path | None = None) -> Path:
    path = data_fin_root(explicit_root) / dataset
    if not path.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {path}")
    return path


def canonical_raw_name(dataset: str) -> str:
    return "raw_compact" if dataset in RAW_COMPACT_DATASETS else "raw"


def resolve_raw_name(dataset: str, raw_name: str | None = None) -> str:
    expected = canonical_raw_name(dataset)
    if raw_name in {None, "", "auto"}:
        return expected
    if raw_name != expected:
        raise ValueError(
            f"{dataset} must read {expected!r}; got {raw_name!r}. "
            "The formal Data_Fin workflow fixes assist2009-sub and XES3G5M-sub-small to raw_compact, all others to raw."
        )
    return raw_name


def raw_dir(dataset: str, explicit_root: Path | None = None, raw_name: str | None = None) -> Path:
    path = dataset_root(dataset, explicit_root) / resolve_raw_name(dataset, raw_name)
    if not path.exists():
        raise FileNotFoundError(f"raw dataset directory does not exist: {path}")
    return path


def front_dir(dataset: str, explicit_root: Path | None = None) -> Path:
    return dataset_root(dataset, explicit_root) / "front_features"


def graph_dir(dataset: str, explicit_root: Path | None = None) -> Path:
    return dataset_root(dataset, explicit_root) / "er_graph"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_matrix_json(path: Path, matrix: np.ndarray, decimals: int = 6) -> None:
    write_json(path, np.round(np.asarray(matrix, dtype=np.float64), decimals).tolist())


def read_matrix_json(path: Path) -> np.ndarray:
    matrix = np.asarray(read_json(path), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix in {path}, got shape={matrix.shape}")
    return matrix


def read_q_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append([float(value) for value in line.replace("\t", ",").split(",") if value != ""])
    if not rows:
        raise ValueError(f"Q matrix is empty: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Q matrix is ragged: {path}")
    matrix = np.asarray(rows, dtype=np.float32)
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"Q matrix must contain finite non-negative values: {path}")
    return (matrix > 0).astype(np.float32)


@dataclass(frozen=True)
class RawDataset:
    name: str
    root: Path
    interactions: pd.DataFrame
    student_split: pd.DataFrame
    q_matrix: np.ndarray
    exercise_metadata: dict[str, Any]
    concept_metadata: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def student_count(self) -> int:
        return int(len(self.student_split))

    @property
    def exercise_count(self) -> int:
        return int(self.q_matrix.shape[0])

    @property
    def concept_count(self) -> int:
        return int(self.q_matrix.shape[1])

    @property
    def train_uids(self) -> list[int]:
        return self.student_split.loc[self.student_split["split"] == "train", "uid"].astype(int).tolist()

    @property
    def test_uids(self) -> list[int]:
        return self.student_split.loc[self.student_split["split"] == "test", "uid"].astype(int).tolist()


def load_raw_dataset(dataset: str, explicit_root: Path | None = None, raw_name: str | None = None) -> RawDataset:
    root = raw_dir(dataset, explicit_root, raw_name)
    required = [
        "interactions_all.csv",
        "student_split.csv",
        "Q.txt",
        "exercise_metadata.json",
        "concept_metadata.json",
        "raw_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"raw dataset {dataset} misses files: {missing}")

    interactions = pd.read_csv(root / "interactions_all.csv", low_memory=False)
    required_columns = {"uid", "question", "response", "timestamp", "sequence_order", "split"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        raise ValueError(f"{root / 'interactions_all.csv'} misses columns: {sorted(missing_columns)}")
    interactions = interactions.copy()
    for name in ["uid", "question", "sequence_order"]:
        interactions[name] = pd.to_numeric(interactions[name], errors="raise").astype(int)
    interactions["response"] = pd.to_numeric(interactions["response"], errors="raise").astype(float)
    if not set(interactions["response"].unique()).issubset({0.0, 1.0}):
        raise ValueError(f"{dataset} response values must be binary")

    split = pd.read_csv(root / "student_split.csv")
    required_split = {"uid", "entity_id", "split"}
    if required_split - set(split.columns):
        raise ValueError(f"{root / 'student_split.csv'} misses columns: {sorted(required_split - set(split.columns))}")
    split = split.copy()
    split["uid"] = pd.to_numeric(split["uid"], errors="raise").astype(int)
    if split["uid"].duplicated().any():
        raise ValueError(f"{dataset} student_split.csv contains duplicate uid values")
    if set(split["split"].unique()) != {"train", "test"}:
        raise ValueError(f"{dataset} split labels must contain exactly train and test")
    expected_uids = list(range(len(split)))
    if sorted(split["uid"].tolist()) != expected_uids:
        raise ValueError(f"{dataset} raw uid values must be contiguous 0..N-1")
    split_by_uid = split.set_index("uid")["split"]
    interactions["_expected_split"] = interactions["uid"].map(split_by_uid)
    if interactions["_expected_split"].isna().any() or (interactions["split"] != interactions["_expected_split"]).any():
        raise ValueError(f"{dataset} interaction split values do not match student_split.csv")
    interactions = interactions.drop(columns=["_expected_split"]).sort_values(["uid", "sequence_order"]).reset_index(drop=True)

    q_matrix = read_q_matrix(root / "Q.txt")
    if q_matrix.shape[0] <= int(interactions["question"].max()):
        raise ValueError(f"{dataset} Q matrix has fewer rows than the largest question id")
    if int(interactions["question"].min()) < 0:
        raise ValueError(f"{dataset} contains negative question ids")

    return RawDataset(
        name=dataset,
        root=root,
        interactions=interactions,
        student_split=split.sort_values("uid").reset_index(drop=True),
        q_matrix=q_matrix,
        exercise_metadata=read_json(root / "exercise_metadata.json"),
        concept_metadata=read_json(root / "concept_metadata.json"),
        manifest=read_json(root / "raw_manifest.json"),
    )


def exercise_texts(raw: RawDataset) -> list[str]:
    entries = raw.exercise_metadata.get("exercises", [])
    by_index = {int(item["exercise_index"]): item for item in entries if isinstance(item, dict) and "exercise_index" in item}
    texts: list[str] = []
    for index in range(raw.exercise_count):
        item = by_index.get(index, {})
        text = (
            item.get("question_text")
            or item.get("text_for_embedding")
            or item.get("problem_name")
            or item.get("step_name")
            or item.get("problem_hierarchy")
            or f"exercise {index}"
        )
        texts.append(str(text))
    return texts


def concept_texts(raw: RawDataset) -> list[str]:
    entries = raw.concept_metadata.get("concepts", [])
    by_index = {int(item["concept_index"]): item for item in entries if isinstance(item, dict) and "concept_index" in item}
    texts: list[str] = []
    for index in range(raw.concept_count):
        item = by_index.get(index, {})
        text = " ".join(str(value).strip() for value in [item.get("name", ""), item.get("definition", "")] if str(value).strip())
        texts.append(text or f"knowledge concept {index}")
    return texts


def split_inner_train_valid(train_uids: Iterable[int], valid_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    ids = np.asarray(sorted(int(uid) for uid in train_uids), dtype=np.int64)
    if len(ids) < 2:
        return ids.tolist(), ids.tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    valid_size = max(1, int(round(len(ids) * valid_ratio)))
    valid_size = min(valid_size, len(ids) - 1)
    valid = sorted(ids[:valid_size].tolist())
    train = sorted(ids[valid_size:].tolist())
    return train, valid


def minmax_from_train(values: np.ndarray, train_mask: np.ndarray, default: float = 0.5) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    train_values = values[np.asarray(train_mask, dtype=bool)]
    if train_values.size == 0:
        return np.full_like(values, default), {"min": float(default), "max": float(default), "default": float(default)}
    low = float(np.min(train_values))
    high = float(np.max(train_values))
    if high - low <= 1e-12:
        return np.full_like(values, default), {"min": low, "max": high, "default": float(default)}
    return np.clip((values - low) / (high - low), 0.0, 1.0), {"min": low, "max": high, "default": float(default)}


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def relation_label(prefix: str, value: float) -> str:
    return f"{prefix}{max(0.0, min(1.0, float(value))):.2f}"


def write_relations(path: Path) -> None:
    rows = ["rec"]
    for prefix in ["mlkc", "exfr"]:
        rows.extend(f"{prefix}{value / 100:.2f}" for value in range(101))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for index, name in enumerate(rows):
            fp.write(f"{index}\t{name}\n")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def matrix_stats(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.size == 0:
        return {"shape": list(value.shape), "min": None, "max": None, "mean": None, "std": None}
    return {
        "shape": list(value.shape),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
    }


def normalise_timestamp(value: Any, unit: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Cannot parse timestamp: {value!r}")
        return float(parsed.timestamp())
    if not math.isfinite(number):
        raise ValueError(f"Timestamp is not finite: {value!r}")
    if unit == "milliseconds" or (unit == "auto" and abs(number) >= 1e12):
        return number / 1000.0
    if unit == "minutes":
        return number * 60.0
    if unit == "days":
        return number * 86400.0
    if unit in {"seconds", "auto"}:
        return number
    raise ValueError(f"Unknown timestamp unit: {unit}")
