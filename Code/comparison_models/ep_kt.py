import argparse
import json
import pickle
from pathlib import Path

from kt_state_tools import (
    calculate_ep_from_kt_states,
    load_json_matrix,
    load_q_matrix,
    target_concepts_from_recommendations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Ep from KT before/after mastery states.")
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "Eedi"
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--before-mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--after-mastery-file", required=True, help="KT simulator output after recommended exercises.")
    parser.add_argument("--q-file", default="Q.txt")
    parser.add_argument("--uid-ex-scores", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--e-sup", type=float, default=1.0)
    return parser.parse_args()


def load_uid_ex_scores(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as fp:
            return pickle.load(fp)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    args = parse_args()
    before = load_json_matrix(args.data_dir / args.before_mastery_file)
    after = load_json_matrix(args.data_dir / args.after_mastery_file)
    q_matrix = load_q_matrix(args.data_dir / args.q_file)
    uid_ex_scores = load_uid_ex_scores(args.uid_ex_scores)
    target_concepts = target_concepts_from_recommendations(uid_ex_scores, q_matrix, top_k=args.top_k)
    result = calculate_ep_from_kt_states(before, after, target_concepts, e_sup=args.e_sup)
    output_file = args.output_file or args.data_dir / f"ep_kt_top{args.top_k}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output_file), "mean": result["mean"], "std": result["std"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
