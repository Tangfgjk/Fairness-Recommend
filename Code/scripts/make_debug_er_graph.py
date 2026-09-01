"""Create a tiny ER graph for debugging SemanticConvE.

The debug graph keeps all exercises, concepts, relations and semantic feature
artifacts, but keeps only a few learner entities. This makes the train/test
pipeline small enough for breakpoint debugging while preserving the same file
contracts as the full ER graph.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


STATE_JSON_FILES = [
    "stu2know_mastery.json",
    "stu2know_forget.json",
    "stu2ex_forget.json",
    "stu2ex_recommend.json",
    "stu2ex_recommend_full_precision.json",
]


def read_entity_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            _idx, name = line.split("\t", 1)
            names.append(name)
    return names


def write_entities(path: Path, names: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for idx, name in enumerate(names):
            fp.write(f"{idx}\t{name}\n")


def parse_uids(value: str) -> list[str]:
    uids = [item.strip() for item in value.split(",") if item.strip()]
    if not uids:
        raise ValueError("at least one uid is required")
    for uid in uids:
        if not uid.startswith("uid") or not uid[3:].isdigit():
            raise ValueError(f"invalid uid: {uid}")
    return uids


def uid_number(uid: str) -> int:
    return int(uid[3:])


def filter_uid_response(source: Path, target: Path, keep_uids: set[str]) -> None:
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            uid = line.split("\t", 1)[0].strip()
            if uid in keep_uids:
                dst.write(line)


def triple_key(line: str) -> tuple[str, str, str] | None:
    parts = line.strip().split("\t")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def write_filtered_triples(
    source_files: list[Path],
    target: Path,
    keep_uids: set[str],
    keep_entities: set[str],
    drop_rec: bool = False,
) -> int:
    seen: set[tuple[str, str, str]] = set()
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as dst:
        for source in source_files:
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as src:
                for line in src:
                    triple = triple_key(line)
                    if triple is None:
                        continue
                    head, relation, tail = triple
                    if tail not in keep_uids:
                        continue
                    if drop_rec and relation == "rec":
                        continue
                    if head not in keep_entities or tail not in keep_entities:
                        continue
                    if triple in seen:
                        continue
                    seen.add(triple)
                    dst.write(f"{head}\t{relation}\t{tail}\n")
                    count += 1
    return count


def copy_json_rows(source: Path, target: Path, row_count: int) -> None:
    if not source.exists():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        shutil.copy2(source, target)
        return
    if len(payload) < row_count:
        raise ValueError(f"{source} has {len(payload)} rows, but {row_count} rows are required")
    target.write_text(json.dumps(payload[:row_count], ensure_ascii=False), encoding="utf-8")


def copy_tree_or_file(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny 3-train-1-test ER graph for debugging.")
    parser.add_argument("--source", type=Path, default=Path("Data_Fin/Eedi/er_graph"))
    parser.add_argument("--output", type=Path, default=Path("Data_Fin/Eedi/er_graph_debug_3train_1test"))
    parser.add_argument("--train-uids", default="uid0,uid1,uid2")
    parser.add_argument("--test-uids", default="uid3")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output
    train_uids = parse_uids(args.train_uids)
    test_uids = parse_uids(args.test_uids)
    keep_uids = set(train_uids + test_uids)
    if set(train_uids).intersection(test_uids):
        raise ValueError("train-uids and test-uids must be disjoint")
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} already exists; pass --force to overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    entity_names = read_entity_names(source / "entities.dict")
    keep_entity_names = [name for name in entity_names if not name.startswith("uid") or name in keep_uids]
    keep_entities = set(keep_entity_names)
    write_entities(output / "entities.dict", keep_entity_names)

    for file_name in ["Q.txt", "relations.dict"]:
        copy_tree_or_file(source / file_name, output / file_name)
    feature_dir = source / "semantic_kg_features"
    if feature_dir.exists():
        copy_tree_or_file(feature_dir, output / "semantic_kg_features")

    uid_response_files = list(source.glob("*_uid_kc_response.txt"))
    for uid_response in uid_response_files:
        filter_uid_response(uid_response, output / uid_response.name, keep_uids)

    max_uid = max(uid_number(uid) for uid in keep_uids)
    row_count = max_uid + 1
    for file_name in STATE_JSON_FILES:
        copy_json_rows(source / file_name, output / file_name, row_count)

    train_count = write_filtered_triples(
        [source / "triples.txt", source / "test_triples.txt"],
        output / "triples.txt",
        set(train_uids),
        keep_entities,
        drop_rec=False,
    )
    test_count = write_filtered_triples(
        [source / "triples.txt", source / "test_triples.txt"],
        output / "test_triples.txt",
        set(test_uids),
        keep_entities,
        drop_rec=True,
    )

    manifest = {
        "source": str(source),
        "debug_graph": str(output),
        "train_uids": train_uids,
        "test_uids": test_uids,
        "entity_count": len(keep_entity_names),
        "relation_count": sum(1 for _ in (output / "relations.dict").open("r", encoding="utf-8")),
        "train_triples": train_count,
        "test_triples": test_count,
        "note": "Debug-only graph. Keeps all exercises and concepts, but only selected learners.",
    }
    (output / "debug_graph_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
