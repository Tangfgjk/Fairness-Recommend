"""Select the best 2CKG4ER seed and export local evidence.

The selection is based on the already evaluated Ada@K value.  It does not
retrain or rescore the model, so it is safe to call after a completed or
resumed 2CKG4ER batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PRIMARY_MODEL_DIR = "2CKG4ER"


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def ada_at(metrics_path: Path, top_k: int) -> float | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        metric = payload.get("Ada") or payload.get("ACC")
        return float(metric[str(top_k)]["mean"])
    except (KeyError, TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best 2CKG4ER seed and write local explanation evidence."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--semantic-run-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--selection-top-k", type=int, default=20)
    parser.add_argument("--evidence-top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selection_top_k <= 0 or args.evidence_top_k <= 0:
        raise ValueError("top-k values must be positive")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"ER graph directory not found: {args.data_dir}")

    candidates: list[tuple[float, int, Path]] = []
    for seed in parse_seed_list(args.seeds):
        seed_dir = args.semantic_run_dir / PRIMARY_MODEL_DIR / f"seed{seed}"
        value = ada_at(seed_dir / "eval" / "metrics.json", args.selection_top_k)
        score_file = seed_dir / "2CKG4ER_uid_ex_scores.pkl"
        if value is not None and score_file.exists():
            candidates.append((value, seed, seed_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No completed {PRIMARY_MODEL_DIR} seed with metrics and scores was found under "
            f"{args.semantic_run_dir}"
        )

    # Highest Ada wins; the smaller seed deterministically breaks an exact tie.
    best_ada, best_seed, best_dir = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    output_dir = args.semantic_run_dir / PRIMARY_MODEL_DIR / f"local_explanation_best_seed{best_seed}"
    extractor = Path(__file__).resolve().parents[1] / "codes-New-ConvE" / "extract_local_explanation_evidence.py"
    command = [
        sys.executable,
        str(extractor),
        "--data-dir",
        str(args.data_dir),
        "--scores-file",
        str(best_dir / "2CKG4ER_uid_ex_scores.pkl"),
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(args.evidence_top_k),
    ]
    subprocess.run(command, check=True)
    selection = {
        "primary_model": PRIMARY_MODEL_DIR,
        "selection_metric": f"Ada@{args.selection_top_k}",
        "best_seed": best_seed,
        "best_ada": best_ada,
        "candidates": [
            {"seed": seed, "Ada": value, "seed_dir": str(seed_dir)}
            for value, seed, seed_dir in sorted(candidates, key=lambda item: item[1])
        ],
        "scores_file": str(best_dir / "2CKG4ER_uid_ex_scores.pkl"),
        "evidence_dir": str(output_dir),
    }
    selection_path = output_dir / "primary_model_selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
