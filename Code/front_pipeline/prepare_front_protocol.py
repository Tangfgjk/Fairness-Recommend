from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import data_fin_root, front_dir, load_raw_dataset, split_inner_train_valid, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a leakage-auditable front-feature protocol from Data_Fin/raw.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-fin-root", type=Path, default=None)
    parser.add_argument("--raw-name", default="auto", help="Validated raw cohort name; use auto for the formal dataset-specific choice.")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--test-history-policy",
        choices=["all_observed_history"],
        default="all_observed_history",
        help="The supplied raw data has no timestamped evaluation cutoff. The external evaluation labels are never read here.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_interactions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ["uid", "question", "response", "timestamp", "sequence_order", "split"]].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = data_fin_root(args.data_fin_root)
    raw = load_raw_dataset(args.dataset, root, args.raw_name)
    output = front_dir(args.dataset, root)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and not args.force:
        raise FileExistsError(f"{protocol_path} exists. Use --force to rebuild the protocol.")

    inner_train, inner_valid = split_inner_train_valid(raw.train_uids, args.valid_ratio, args.seed)
    input_dir = output / "mirt" / "inputs"
    train_all = raw.interactions[raw.interactions["uid"].isin(raw.train_uids)].copy()
    train_inner = raw.interactions[raw.interactions["uid"].isin(inner_train)].copy()
    valid_inner = raw.interactions[raw.interactions["uid"].isin(inner_valid)].copy()
    test_adaptation = raw.interactions[raw.interactions["uid"].isin(raw.test_uids)].copy()
    write_interactions(train_all, input_dir / "outer_train.csv")
    write_interactions(train_inner, input_dir / "inner_train.csv")
    write_interactions(valid_inner, input_dir / "inner_valid.csv")
    write_interactions(test_adaptation, input_dir / "outer_test_adaptation.csv")
    write_interactions(raw.interactions, input_dir / "all_observed.csv")
    (input_dir / "Q_matrix.csv").write_text((raw.root / "Q.txt").read_text(encoding="utf-8"), encoding="utf-8")

    write_json(
        protocol_path,
        {
            "dataset": raw.name,
            "version": "raw_front_protocol_v1",
            "raw_dir": raw.root,
            "front_dir": output,
            "outer_split_source": raw.root / "student_split.csv",
            "outer_train_uids": raw.train_uids,
            "outer_test_uids": raw.test_uids,
            "inner_train_uids": inner_train,
            "inner_valid_uids": inner_valid,
            "seed": args.seed,
            "test_history_policy": args.test_history_policy,
            "shared_parameter_policy": "Only outer_train learners may update shared front-model parameters.",
            "test_student_policy": "Test students are passed through the frozen EKTM model; Q-MIRT only adapts their personal theta with item parameters fixed.",
            "evaluation_label_policy": "evaluation_uid_kc_response.txt is not read by protocol or feature generation stages.",
            "counts": {
                "students": raw.student_count,
                "outer_train_students": len(raw.train_uids),
                "outer_test_students": len(raw.test_uids),
                "inner_train_students": len(inner_train),
                "inner_valid_students": len(inner_valid),
                "exercises": raw.exercise_count,
                "concepts": raw.concept_count,
                "interactions": int(len(raw.interactions)),
                "outer_train_interactions": int(len(train_all)),
                "outer_test_interactions": int(len(test_adaptation)),
            },
            "files": {
                "outer_train": "mirt/inputs/outer_train.csv",
                "inner_train": "mirt/inputs/inner_train.csv",
                "inner_valid": "mirt/inputs/inner_valid.csv",
                "outer_test_adaptation": "mirt/inputs/outer_test_adaptation.csv",
                "all_observed": "mirt/inputs/all_observed.csv",
                "q_matrix": "mirt/inputs/Q_matrix.csv",
            },
        },
    )
    print(f"prepared raw front protocol: {protocol_path}")


if __name__ == "__main__":
    main()
