"""Create a KG-enhanced graph with explicit exercise-KC edges."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from fairness_context import read_q_matrix


EX_HAS_KC = "ex_has_kc"
KC_HAS_EX = "kc_has_ex"


def read_dict(path: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split("\t")
            result[name] = int(idx)
    return result


def write_dict(path: Path, name_to_id: Dict[str, int]) -> None:
    rows = sorted(name_to_id.items(), key=lambda item: item[1])
    with path.open("w", encoding="utf-8", newline="") as fp:
        for name, idx in rows:
            fp.write(f"{idx}\t{name}\n")


def read_triple_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def write_triple_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        for line in lines:
            fp.write(f"{line}\n")


def ex_kc_triples(q_matrix: List[List[int]], include_reverse: bool = True) -> List[str]:
    triples: List[str] = []
    for exercise_idx, row in enumerate(q_matrix):
        for kc_idx, value in enumerate(row):
            if int(value) != 1:
                continue
            triples.append(f"ex{exercise_idx}\t{EX_HAS_KC}\tkc{kc_idx}")
            if include_reverse:
                triples.append(f"kc{kc_idx}\t{KC_HAS_EX}\tex{exercise_idx}")
    return triples


def copy_graph(source_dir: Path, target_dir: Path, overwrite: bool = False) -> None:
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Target graph already exists: {target_dir}. Use --overwrite to replace it.")
        shutil.rmtree(target_dir)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source_dir, target_dir, ignore=ignore)


def add_relations(relation2id: Dict[str, int], include_reverse: bool = True) -> Dict[str, int]:
    result = dict(relation2id)
    next_id = max(result.values(), default=-1) + 1
    for name in [EX_HAS_KC, KC_HAS_EX] if include_reverse else [EX_HAS_KC]:
        if name not in result:
            result[name] = next_id
            next_id += 1
    return result


def prepare_fair_kg_graph(
    source_dir: str | Path,
    target_dir: str | Path,
    include_reverse: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not (source_dir / "entities.dict").exists():
        raise FileNotFoundError(f"Missing entities.dict in {source_dir}")
    if not (source_dir / "relations.dict").exists():
        raise FileNotFoundError(f"Missing relations.dict in {source_dir}")
    if not (source_dir / "triples.txt").exists():
        raise FileNotFoundError(f"Missing triples.txt in {source_dir}")
    if not (source_dir / "Q.txt").exists():
        raise FileNotFoundError(f"Missing Q.txt in {source_dir}")

    copy_graph(source_dir, target_dir, overwrite=overwrite)
    q_matrix = read_q_matrix(target_dir / "Q.txt")
    relation2id = read_dict(target_dir / "relations.dict")
    relation2id = add_relations(relation2id, include_reverse=include_reverse)
    write_dict(target_dir / "relations.dict", relation2id)

    original_triples = read_triple_lines(target_dir / "triples.txt")
    added_triples = ex_kc_triples(q_matrix, include_reverse=include_reverse)
    merged = list(dict.fromkeys([*original_triples, *added_triples]))
    write_triple_lines(target_dir / "triples.txt", merged)

    manifest = {
        "version": "fair_kg_graph_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "include_reverse": bool(include_reverse),
        "q_matrix_shape": [
            len(q_matrix),
            max((len(row) for row in q_matrix), default=0),
        ],
        "relations_added": [EX_HAS_KC, KC_HAS_EX] if include_reverse else [EX_HAS_KC],
        "relation_count": len(relation2id),
        "original_triples": len(original_triples),
        "candidate_ex_kc_triples": len(added_triples),
        "deduplicated_total_triples": len(merged),
        "added_after_dedup": len(merged) - len(original_triples),
        "files_copied_from_source": True,
        "original_graph_preserved": True,
    }
    (target_dir / "fair_kg_graph_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fair KG graph with explicit exercise-KC edges.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--no-reverse", action="store_true", help="Only add ex_has_kc and skip kc_has_ex.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_fair_kg_graph(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        include_reverse=not args.no_reverse,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

