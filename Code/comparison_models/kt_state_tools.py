import argparse
import json
from pathlib import Path

import numpy as np


def load_json_matrix(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_q_matrix(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append([int(value) for value in line.strip().split(",")])
    return rows


def top_k_exercises(scores, top_k):
    return [
        exercise_idx
        for exercise_idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    ]


def exercise_concepts(q_matrix, exercise_idx):
    if exercise_idx < 0 or exercise_idx >= len(q_matrix):
        return []
    return [kc_idx for kc_idx, value in enumerate(q_matrix[exercise_idx]) if int(value) == 1]


def target_concepts_from_recommendations(uid_ex_scores, q_matrix, top_k=10):
    targets = {}
    for uid, scores in uid_ex_scores:
        concepts = set()
        for exercise_idx in top_k_exercises(scores, top_k):
            concepts.update(exercise_concepts(q_matrix, exercise_idx))
        targets[str(uid)] = sorted(concepts)
    return targets


def calculate_ep_from_kt_states(before_mastery, after_mastery, target_concepts, e_sup=1.0):
    per_user = []
    for uid, concepts in target_concepts.items():
        uid_idx = int(str(uid)[3:]) if str(uid).startswith("uid") else int(uid)
        if uid_idx >= len(before_mastery) or uid_idx >= len(after_mastery) or not concepts:
            continue
        before = np.asarray(before_mastery[uid_idx], dtype=float)
        after = np.asarray(after_mastery[uid_idx], dtype=float)
        valid_concepts = [kc for kc in concepts if kc < len(before) and kc < len(after)]
        if not valid_concepts:
            continue
        e_start = float(np.mean(before[valid_concepts]))
        e_end = float(np.mean(after[valid_concepts]))
        denom = e_sup - e_start
        ep = 0.0 if denom <= 1e-12 else (e_end - e_start) / denom
        per_user.append(
            {
                "uid": uid,
                "target_concepts": valid_concepts,
                "e_start": round(e_start, 6),
                "e_end": round(e_end, 6),
                "ep": round(float(ep), 6),
            }
        )
    values = [item["ep"] for item in per_user]
    return {
        "mean": round(float(np.mean(values)), 6) if values else 0.0,
        "std": round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0,
        "per_user": per_user,
    }


def write_heatmap(mastery, output_path, student_indices=None, concept_indices=None, title="ER KT Mastery Heatmap"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(mastery, dtype=float)
    if student_indices is not None:
        matrix = matrix[student_indices, :]
    if concept_indices is not None:
        matrix = matrix[:, concept_indices]

    csv_path = output_path.with_suffix(".csv")
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%.6f")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"image": None, "csv": str(csv_path), "reason": "matplotlib is not installed"}

    width = max(8, min(18, matrix.shape[1] * 0.18))
    height = max(4, min(12, matrix.shape[0] * 0.35))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Knowledge concept")
    ax.set_ylabel("Student")
    fig.colorbar(im, ax=ax, label="Mastery")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return {"image": str(output_path), "csv": str(csv_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ER KT mastery heatmaps from stu2know_mastery.json.")
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "Eedi"
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--mastery-file", default="stu2know_mastery.json")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--students", default="0,1", help="Comma-separated student indices.")
    parser.add_argument("--concepts", default=None, help="Comma-separated KC indices. Defaults to all concepts.")
    parser.add_argument("--title", default="ER KT Mastery Heatmap")
    return parser.parse_args()


def _parse_indices(text):
    if text is None or str(text).strip() == "":
        return None
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def main():
    args = parse_args()
    mastery = load_json_matrix(args.data_dir / args.mastery_file)
    output_file = args.output_file or args.data_dir / "kt_mastery_heatmap.png"
    result = write_heatmap(
        mastery,
        output_file,
        student_indices=_parse_indices(args.students),
        concept_indices=_parse_indices(args.concepts),
        title=args.title,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
