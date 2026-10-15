from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    copy_file,
    ensure_large_csv_field_limit,
    entity_count,
    locate_sequence_interactions,
    locate_source_graph_dir,
    locate_test_sequences,
    output_data_root,
    read_entity_dict,
    source_data_root,
    source_dataset_dir,
    v11_dataset_dir,
    write_json,
)


def eedi_valid_question_length(row: dict[str, str]) -> int:
    questions = [q for q in str(row.get("questions", "")).split(",") if q != ""]
    masks = [m for m in str(row.get("selectmasks", "")).split(",") if m != ""]
    if masks and len(masks) == len(questions):
        return sum(1 for q, m in zip(questions, masks) if q != "-1" and m == "1")
    return sum(1 for q in questions if q != "-1")


def graph_sequence_uids(dataset: str, source_dir: Path, graph_dir: Path, learner_count: int) -> tuple[list[str], dict[str, Any]]:
    test_sequences = locate_test_sequences(dataset, source_dir, graph_dir)
    ensure_large_csv_field_limit()
    if dataset == "Eedi":
        selected: list[dict[str, str]] = []
        with test_sequences.open("r", encoding="utf-8", newline="") as fp:
            for row in csv.DictReader(fp):
                valid_len = eedi_valid_question_length(row)
                if 10 <= valid_len <= 198:
                    selected.append(dict(row, _valid_question_length=str(valid_len)))
        selected = selected[:learner_count]
        if len(selected) != learner_count:
            raise ValueError(
                f"Eedi selected test learners do not match graph learner count: "
                f"selected={len(selected)} graph={learner_count}"
            )
        # Historical Eedi sequence_interactions.csv stores ER learners as 0..934.
        uids = [str(i) for i in range(learner_count)]
        return uids, {
            "source_file": test_sequences,
            "selection_rule": "Eedi pyKT test rows with 10 <= valid question length <= 198; sequence uid is uid index 0..N-1",
            "raw_uid_by_entity": {f"uid{i}": str(row.get("uid", "")) for i, row in enumerate(selected)},
            "valid_question_length_by_entity": {
                f"uid{i}": int(row.get("_valid_question_length", "0")) for i, row in enumerate(selected)
            },
        }

    test_df = pd.read_csv(test_sequences, low_memory=False)
    if "uid" not in test_df.columns:
        raise ValueError(f"{test_sequences} missing uid column")
    selected = test_df.head(learner_count)
    if len(selected) != learner_count:
        raise ValueError(
            f"{dataset} test_sequences row count does not match graph learner count: "
            f"selected={len(selected)} graph={learner_count}"
        )
    uids = [str(value) for value in selected["uid"].tolist()]
    return uids, {
        "source_file": test_sequences,
        "selection_rule": "ER uid order follows prepared_for_kt/test_sequences.csv row order",
        "sequence_uid_by_entity": {f"uid{i}": uids[i] for i in range(len(uids))},
    }


def normalise_filtered_interactions(
    seq_df: pd.DataFrame,
    graph_uids: list[str],
    exercise_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"uid", "question", "response"}
    missing = required - set(seq_df.columns)
    if missing:
        raise ValueError(f"sequence_interactions.csv missing columns: {sorted(missing)}")

    user_map = {uid: idx for idx, uid in enumerate(graph_uids)}
    work = seq_df.copy()
    work["uid_key"] = work["uid"].astype(str)
    work = work[work["uid_key"].isin(user_map)].copy()
    work["user_id"] = work["uid_key"].map(user_map).astype(int)
    work["item_id"] = pd.to_numeric(work["question"], errors="coerce")
    work["score"] = pd.to_numeric(work["response"], errors="coerce")
    work = work.dropna(subset=["item_id", "score"])
    work["item_id"] = work["item_id"].astype(int)
    work = work[(work["item_id"] >= 0) & (work["item_id"] < exercise_count)].copy()
    work["score"] = (work["score"] > 0).astype(float)
    if "source_split" not in work.columns:
        work["source_split"] = "all"
    else:
        work["source_split"] = work["source_split"].astype(str).str.lower()

    out = work[["user_id", "item_id", "score", "source_split"]].sort_values(["user_id", "item_id"]).reset_index(drop=True)
    observed_items = set(out["item_id"].astype(int).tolist())
    missing_items = [idx for idx in range(exercise_count) if idx not in observed_items]
    return out, {
        "graph_learner_count": len(graph_uids),
        "filtered_interaction_count": int(len(out)),
        "observed_item_count": int(len(observed_items)),
        "exercise_count": int(exercise_count),
        "missing_item_count": int(len(missing_items)),
        "missing_item_examples": missing_items[:20],
    }


def split_frames(df: pd.DataFrame, valid_ratio: float, test_ratio: float, seed: int) -> dict[str, pd.DataFrame]:
    if len(df) == 0:
        raise ValueError("filtered MIRT interactions are empty")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_size = max(1, int(len(shuffled) * test_ratio)) if len(shuffled) > 5 else 0
    valid_size = max(1, int(len(shuffled) * valid_ratio)) if len(shuffled) > 5 else 0
    test = shuffled.iloc[:test_size].copy()
    valid = shuffled.iloc[test_size : test_size + valid_size].copy()
    train = df.copy()
    cols = ["user_id", "item_id", "score"]
    return {
        "train": train[cols].sort_values(cols).reset_index(drop=True),
        "valid": valid[cols].sort_values(cols).reset_index(drop=True),
        "test": test[cols].sort_values(cols).reset_index(drop=True),
        "all": df[cols].sort_values(cols).reset_index(drop=True),
    }


def write_frames(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V11 MIRT inputs aligned to ER graph learners and exercise ids.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-data-root", type=Path, default=source_data_root())
    parser.add_argument("--output-data-root", type=Path, default=output_data_root())
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = source_dataset_dir(args.dataset, args.source_data_root)
    graph_dir = locate_source_graph_dir(source_dir)
    output_dir = v11_dataset_dir(args.dataset, args.output_data_root)
    input_dir = output_dir / "mirt" / "inputs"
    if input_dir.exists() and any(input_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{input_dir} already exists. Use --force to overwrite.")

    entity2id = read_entity_dict(graph_dir / "entities.dict")
    learner_count = entity_count(entity2id, "uid")
    exercise_count = entity_count(entity2id, "ex")
    graph_uids, uid_info = graph_sequence_uids(args.dataset, source_dir, graph_dir, learner_count)
    seq_path = locate_sequence_interactions(source_dir)
    seq_df = pd.read_csv(seq_path, low_memory=False)
    filtered, filter_info = normalise_filtered_interactions(seq_df, graph_uids, exercise_count)
    frames = split_frames(filtered, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio, seed=args.seed)
    write_frames(frames, input_dir)
    copy_file(graph_dir / "Q.txt", input_dir / "Q_matrix.csv")

    manifest = {
        "dataset": args.dataset,
        "version": "v11_graph_subset_mirt_inputs",
        "source_sequence_file": seq_path,
        "source_graph_dir": graph_dir,
        "output_dir": input_dir,
        "uid_alignment": uid_info,
        "filter_info": filter_info,
        "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "user_num": int(learner_count),
        "item_num": int(exercise_count),
        "item_id_policy": "preserve ER exercise id ex0..exN; do not remap observed item ids",
        "train_policy": "MIRT is a pedagogical feature estimator; train.csv uses all filtered graph-learner interactions.",
    }
    write_json(input_dir / "mirt_input_manifest.json", manifest)
    print(f"created V11 MIRT inputs: {input_dir}")
    print(f"user_num={learner_count} item_num={exercise_count} rows={len(filtered)} missing_items={filter_info['missing_item_count']}")


if __name__ == "__main__":
    main()
