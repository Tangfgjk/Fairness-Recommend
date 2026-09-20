"""Build fairness-aware training graphs by reconstructing ``rec`` edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from fairness_context import build_fairness_context
from fairness_metrics import coverage, exposure_share, gini, head_and_long_tail, kc_popularity_from_items
from semantic_experiment_utils import jsonable, write_json


EPSILON = 1e-8


@dataclass(frozen=True)
class RecEdge:
    user: int
    exercise: int
    line: str


def parse_user(value: str) -> int | None:
    if not value.startswith("uid"):
        return None
    try:
        return int(value[3:])
    except ValueError:
        return None


def parse_exercise(value: str) -> int | None:
    if not value.startswith("ex"):
        return None
    try:
        return int(value[2:])
    except ValueError:
        return None


def read_distance_matrix(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"distance matrix must be two-dimensional: {path}")
    return matrix


def split_triples(path: Path) -> Tuple[List[str], Dict[int, List[int]], List[RecEdge]]:
    non_rec_lines: List[str] = []
    rec_by_user: Dict[int, List[int]] = {}
    rec_edges: List[RecEdge] = []
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 3 or parts[1] != "rec":
                non_rec_lines.append(line)
                continue
            user_idx = parse_user(parts[0])
            exercise_idx = parse_exercise(parts[2])
            if user_idx is None or exercise_idx is None:
                non_rec_lines.append(line)
                continue
            rec_by_user.setdefault(user_idx, []).append(exercise_idx)
            rec_edges.append(RecEdge(user_idx, exercise_idx, line))
    return non_rec_lines, rec_by_user, rec_edges


def hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def top_by_distance(distances: np.ndarray, count: int) -> np.ndarray:
    exercise_ids = np.arange(distances.shape[0])
    return np.lexsort((exercise_ids, distances))[:count]


def normalize_relevance(candidate_distances: np.ndarray, epsilon: float) -> np.ndarray:
    min_d = float(np.min(candidate_distances))
    max_d = float(np.max(candidate_distances))
    spread = max_d - min_d
    if spread < epsilon:
        return np.ones_like(candidate_distances, dtype=np.float64)
    return 1.0 - ((candidate_distances - min_d) / (spread + epsilon))


def normalized_log_inverse_prior(popularity: Sequence[float]) -> np.ndarray:
    values = np.asarray([max(0.0, float(value)) for value in popularity], dtype=np.float64)
    logged = np.log1p(values)
    max_value = float(np.max(logged)) if logged.size else 0.0
    if max_value <= 0.0:
        return np.ones_like(logged, dtype=np.float64)
    return 1.0 - (logged / max_value)


def kc_prior_by_exercise(q_matrix: Sequence[Sequence[int]], kc_prior: np.ndarray) -> np.ndarray:
    values = np.zeros(len(q_matrix), dtype=np.float64)
    for exercise_idx, row in enumerate(q_matrix):
        concepts = [idx for idx, value in enumerate(row) if int(value) == 1 and idx < len(kc_prior)]
        if concepts:
            values[exercise_idx] = float(np.mean(kc_prior[concepts]))
    return values


def train_rec_exposure(
    rec_by_user: Dict[int, Sequence[int]],
    q_matrix: Sequence[Sequence[int]],
) -> Tuple[List[float], List[float], int]:
    exercise_count = len(q_matrix)
    concept_count = max((len(row) for row in q_matrix), default=0)
    item_exposure = [0.0 for _ in range(exercise_count)]
    kc_exposure = [0.0 for _ in range(concept_count)]
    rec_count = 0
    for exercises in rec_by_user.values():
        for exercise_idx in exercises:
            if 0 <= int(exercise_idx) < exercise_count:
                item_exposure[int(exercise_idx)] += 1.0
                rec_count += 1
                concepts = [idx for idx, value in enumerate(q_matrix[int(exercise_idx)]) if int(value) == 1]
                if not concepts:
                    continue
                share = 1.0 / len(concepts)
                for concept_idx in concepts:
                    kc_exposure[concept_idx] += share
    return item_exposure, kc_exposure, rec_count


def graph_distribution_metrics(
    rec_by_user: Dict[int, Sequence[int]],
    q_matrix: Sequence[Sequence[int]],
    item_popularity: Sequence[float],
    head_ratio: float,
    long_tail_ratio: float,
) -> Dict[str, Any]:
    item_exposure, kc_exposure, rec_count = train_rec_exposure(rec_by_user, q_matrix)
    kc_popularity = kc_popularity_from_items(item_popularity, q_matrix)
    _head_items, long_tail_items = head_and_long_tail(item_popularity, head_ratio, long_tail_ratio)
    _head_kcs, long_tail_kcs = head_and_long_tail(kc_popularity, head_ratio, long_tail_ratio)
    return {
        "TrainRecItemGini": round(gini(item_exposure), 6),
        "TrainRecKCGini": round(gini(kc_exposure), 6),
        "TrainRecItemCoverage": round(coverage(item_exposure), 6),
        "TrainRecKCCoverage": round(coverage(kc_exposure), 6),
        "TrainRecLongTailItemShare": round(exposure_share(item_exposure, long_tail_items), 6),
        "TrainRecLongTailKCShare": round(exposure_share(kc_exposure, long_tail_kcs), 6),
        "train_rec_edges": int(rec_count),
    }


def rec_degree_stats(rec_by_user: Dict[int, Sequence[int]]) -> Dict[str, Any]:
    counts = [len(values) for values in rec_by_user.values()]
    if not counts:
        return {"users": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "users": len(counts),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "mean": float(np.mean(counts)),
    }


def rec_edge_set(rec_by_user: Dict[int, Sequence[int]]) -> set[Tuple[int, int]]:
    return {(int(user), int(exercise)) for user, exercises in rec_by_user.items() for exercise in exercises}


def canonical_rec_lines(rec_by_user: Dict[int, Sequence[int]]) -> List[str]:
    return [f"uid{user}\trec\tex{exercise}" for user, exercise in sorted(rec_edge_set(rec_by_user))]


def rec_degree_preserved(source: Dict[int, Sequence[int]], output: Dict[int, Sequence[int]]) -> bool:
    if set(source) != set(output):
        return False
    return all(len(source[user]) == len(output[user]) for user in source)


def update_output_graph_manifest(source_graph_dir: Path, output_graph_dir: Path) -> None:
    source_manifest_path = source_graph_dir / "er_graph_manifest.json"
    output_manifest_path = output_graph_dir / "er_graph_manifest.json"
    original_payload: Dict[str, Any] = {}
    if source_manifest_path.exists():
        try:
            original_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            original_payload = {"raw_manifest_text": source_manifest_path.read_text(encoding="utf-8")}
        write_json(output_graph_dir / "source_er_graph_manifest.json", original_payload)

    updated_payload = dict(original_payload)
    updated_payload.update(
        {
            "graph_variant": "fairness_preprocessed",
            "source_graph_dir": str(source_graph_dir),
            "rec_edge_preprocessing": {
                "method": "Fairness-Aware Rec Edge Construction",
                "manifest": "fairness_preprocess_manifest.json",
            },
        }
    )
    write_json(output_manifest_path, updated_payload)


def build_fair_rec_edges(
    distances: np.ndarray,
    original_rec_by_user: Dict[int, Sequence[int]],
    item_prior: np.ndarray,
    exercise_kc_prior: np.ndarray,
    candidate_pool_size: int,
    top_k_rec: int,
    lambda_item: float,
    lambda_kc: float,
    epsilon: float,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    if candidate_pool_size < top_k_rec:
        raise ValueError("candidate_pool_size must be greater than or equal to top_k_rec")
    fair_rec_by_user: Dict[int, List[int]] = {}
    selected_distances: List[float] = []
    original_distances: List[float] = []
    overlaps: List[float] = []
    original_ranks: List[int] = []
    promoted_ranks: List[int] = []
    promotion_depths: List[int] = []

    exercise_count = distances.shape[1]
    candidate_pool_size = min(int(candidate_pool_size), exercise_count)
    top_k_rec = min(int(top_k_rec), candidate_pool_size)

    for user_idx in sorted(original_rec_by_user):
        if user_idx < 0 or user_idx >= distances.shape[0]:
            raise ValueError(f"user uid{user_idx} is outside distance matrix with shape {distances.shape}")
        user_distances = distances[user_idx]
        candidates = top_by_distance(user_distances, candidate_pool_size)
        candidate_distances = user_distances[candidates]
        relevance = normalize_relevance(candidate_distances, epsilon)
        fair_scores = relevance + float(lambda_item) * item_prior[candidates] + float(lambda_kc) * exercise_kc_prior[candidates]
        ranked_positions = np.lexsort((candidates, -fair_scores))
        selected = candidates[ranked_positions[:top_k_rec]].astype(int).tolist()
        fair_rec_by_user[user_idx] = selected

        original = list(original_rec_by_user[user_idx])[:top_k_rec]
        original_set = set(int(item) for item in original)
        selected_set = set(selected)
        overlaps.append(len(original_set.intersection(selected_set)) / float(top_k_rec or 1))
        selected_distances.extend(float(user_distances[item]) for item in selected)
        original_distances.extend(float(user_distances[item]) for item in original)
        for rank, exercise_idx in enumerate(top_by_distance(user_distances, exercise_count), start=1):
            if int(exercise_idx) in selected_set:
                original_ranks.append(rank)
                if int(exercise_idx) not in original_set:
                    promoted_ranks.append(rank)
                    promotion_depths.append(max(0, rank - top_k_rec))

    mean_original_distance = float(np.mean(original_distances)) if original_distances else 0.0
    mean_fair_distance = float(np.mean(selected_distances)) if selected_distances else 0.0
    return fair_rec_by_user, {
        "mean_original_distance": mean_original_distance,
        "mean_fair_distance": mean_fair_distance,
        "edu_distance_increase": mean_fair_distance - mean_original_distance,
        "relative_edu_distance_increase": (mean_fair_distance - mean_original_distance) / (mean_original_distance + epsilon),
        "mean_top10_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "min_top10_overlap": float(np.min(overlaps)) if overlaps else 0.0,
        "mean_original_rank_of_selected_items": float(np.mean(original_ranks)) if original_ranks else 0.0,
        "max_original_rank_of_selected_items": int(max(original_ranks)) if original_ranks else 0,
        "mean_promoted_original_rank": float(np.mean(promoted_ranks)) if promoted_ranks else 0.0,
        "max_promoted_original_rank": int(max(promoted_ranks)) if promoted_ranks else 0,
        "mean_promotion_depth": float(np.mean(promotion_depths)) if promotion_depths else 0.0,
        "max_promotion_depth": int(max(promotion_depths)) if promotion_depths else 0,
        "promoted_edge_count": int(len(promoted_ranks)),
    }


def write_rebuilt_triples(path: Path, non_rec_lines: Sequence[str], rec_by_user: Dict[int, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for line in non_rec_lines:
            fp.write(f"{line}\n")
        for user_idx in sorted(rec_by_user):
            for exercise_idx in rec_by_user[user_idx]:
                fp.write(f"uid{user_idx}\trec\tex{int(exercise_idx)}\n")


def copy_graph_except_triples(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if source.name == "triples.txt":
            continue
        target = output_dir / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def prepare_fair_preprocess_graph(
    source_graph_dir: str | Path,
    output_graph_dir: str | Path,
    method: str = "fair_score",
    candidate_pool_size: int = 50,
    top_k_rec: int | None = None,
    lambda_item: float = 0.1,
    lambda_kc: float = 0.1,
    popularity_source: str = "train_interactions",
    popularity_aggregation: str = "unique_users",
    head_ratio: float = 0.2,
    long_tail_ratio: float = 0.8,
    epsilon: float = EPSILON,
    force: bool = False,
) -> Dict[str, Any]:
    source_graph_dir = Path(source_graph_dir)
    output_graph_dir = Path(output_graph_dir)
    if method != "fair_score":
        raise ValueError("Only method=fair_score is currently supported")
    if int(candidate_pool_size) <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if float(lambda_item) < 0.0 or float(lambda_kc) < 0.0:
        raise ValueError("lambda_item and lambda_kc must be non-negative fairness rewards")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if not (source_graph_dir / "triples.txt").exists():
        raise FileNotFoundError(f"source graph triples.txt not found: {source_graph_dir}")
    if output_graph_dir.exists() and any(output_graph_dir.iterdir()):
        if not force:
            manifest_path = output_graph_dir / "fairness_preprocess_manifest.json"
            if manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            raise FileExistsError(f"{output_graph_dir} already exists. Use force=True or --force.")
        shutil.rmtree(output_graph_dir)
    output_graph_dir.mkdir(parents=True, exist_ok=True)

    non_rec_lines, original_rec_by_user, _original_edges = split_triples(source_graph_dir / "triples.txt")
    if not original_rec_by_user:
        raise ValueError("source graph has no uid-rec-ex training edges")
    inferred_top_k = len(next(iter(original_rec_by_user.values())))
    top_k = int(top_k_rec or inferred_top_k)
    if any(len(values) != inferred_top_k for values in original_rec_by_user.values()):
        raise ValueError("source graph must have a fixed rec degree per user for this preprocessing method")
    if top_k != inferred_top_k:
        raise ValueError(f"top_k_rec={top_k} does not match source rec degree={inferred_top_k}")

    distances = read_distance_matrix(source_graph_dir / "stu2ex_recommend_full_precision.json")
    context = build_fairness_context(
        source_graph_dir,
        head_ratio=head_ratio,
        long_tail_ratio=long_tail_ratio,
        popularity_source=popularity_source,
        popularity_aggregation=popularity_aggregation,
    )
    if distances.shape[1] != len(context.q_matrix):
        raise ValueError(
            "distance exercise dimension does not match Q-matrix exercise count: "
            f"{distances.shape[1]} != {len(context.q_matrix)}"
        )
    if not np.isfinite(distances).all():
        raise ValueError("distance matrix contains NaN or infinite values")
    item_prior = normalized_log_inverse_prior(context.item_popularity)
    kc_prior = normalized_log_inverse_prior(context.kc_popularity)
    exercise_kc_prior = kc_prior_by_exercise(context.q_matrix, kc_prior)

    fair_rec_by_user, edu_metrics = build_fair_rec_edges(
        distances=distances,
        original_rec_by_user=original_rec_by_user,
        item_prior=item_prior,
        exercise_kc_prior=exercise_kc_prior,
        candidate_pool_size=int(candidate_pool_size),
        top_k_rec=top_k,
        lambda_item=float(lambda_item),
        lambda_kc=float(lambda_kc),
        epsilon=float(epsilon),
    )
    original_set = rec_edge_set(original_rec_by_user)
    fair_set = rec_edge_set(fair_rec_by_user)
    symmetric_difference_count = len(original_set.symmetric_difference(fair_set))
    replaced_edges = symmetric_difference_count // 2
    rec_edges = sum(len(values) for values in original_rec_by_user.values())
    baseline_rebuild = float(lambda_item) == 0.0 and float(lambda_kc) == 0.0
    exact_match = original_set == fair_set
    if baseline_rebuild and not exact_match:
        raise RuntimeError(
            "Zero-lambda baseline rebuild failed: reconstructed rec edges differ from source graph. "
            f"symmetric_difference_count={symmetric_difference_count}, "
            f"mean_top10_overlap={edu_metrics.get('mean_top10_overlap')}"
        )

    copy_graph_except_triples(source_graph_dir, output_graph_dir)
    write_rebuilt_triples(output_graph_dir / "triples.txt", non_rec_lines, fair_rec_by_user)
    update_output_graph_manifest(source_graph_dir, output_graph_dir)

    output_non_rec_lines, output_rec_by_user, _output_edges = split_triples(output_graph_dir / "triples.txt")
    output_set = rec_edge_set(output_rec_by_user)
    if output_set != fair_set:
        raise RuntimeError("Written triples.txt rec edges do not match the in-memory fairness rec graph")
    source_non_rec_hash = hash_lines(non_rec_lines)
    output_non_rec_hash = hash_lines(output_non_rec_lines)
    non_rec_unchanged = source_non_rec_hash == output_non_rec_hash
    if not non_rec_unchanged:
        raise RuntimeError("Output non-rec triples differ from the source graph")
    degree_preserved = rec_degree_preserved(original_rec_by_user, output_rec_by_user)
    if not degree_preserved:
        raise RuntimeError("Output rec degree differs from source rec degree for at least one user")

    changed_users = sum(
        1
        for user in original_rec_by_user
        if set(int(item) for item in original_rec_by_user.get(user, [])) != set(int(item) for item in output_rec_by_user.get(user, []))
    )
    zero_popularity = {idx for idx, value in enumerate(context.item_popularity) if float(value) <= 0.0}
    zero_selected = sum(1 for values in output_rec_by_user.values() for ex in values if int(ex) in zero_popularity)
    before_metrics = graph_distribution_metrics(original_rec_by_user, context.q_matrix, context.item_popularity, head_ratio, long_tail_ratio)
    after_metrics = graph_distribution_metrics(output_rec_by_user, context.q_matrix, context.item_popularity, head_ratio, long_tail_ratio)
    manifest: Dict[str, Any] = {
        "method": method,
        "preprocessing_name": "Fairness-Aware Rec Edge Construction",
        "source_graph_dir": source_graph_dir,
        "output_graph_dir": output_graph_dir,
        "candidate_pool_size": int(candidate_pool_size),
        "top_k_rec": top_k,
        "lambda_item": float(lambda_item),
        "lambda_kc": float(lambda_kc),
        "epsilon": float(epsilon),
        "candidate_tie_break_rule": "distance ascending, exercise id ascending; fair score descending, exercise id ascending",
        "popularity_source": popularity_source,
        "popularity_aggregation": popularity_aggregation,
        "popularity_log_transform": "1 - log1p(popularity) / max(log1p(popularity))",
        "pseudo_label_construction": True,
        "pseudo_label_scope": "reconstruct uid-rec-ex labels only from each user's educational Top-M candidate pool",
        "rec_degree_preserved": bool(degree_preserved),
        "source_rec_degree": rec_degree_stats(original_rec_by_user),
        "output_rec_degree": rec_degree_stats(output_rec_by_user),
        "source_rec_edge_count": rec_edges,
        "output_rec_edge_count": sum(len(values) for values in output_rec_by_user.values()),
        "source_rec_edge_hash": hash_lines(canonical_rec_lines(original_rec_by_user)),
        "output_rec_edge_hash": hash_lines(canonical_rec_lines(output_rec_by_user)),
        "source_non_rec_hash": source_non_rec_hash,
        "output_non_rec_hash": output_non_rec_hash,
        "non_rec_relations_unchanged": bool(non_rec_unchanged),
        "baseline_rebuild_exact_match": bool(baseline_rebuild and exact_match),
        "replaced_rec_edge_count": int(replaced_edges),
        "replaced_rec_edge_ratio": float(replaced_edges / (rec_edges or 1)),
        "rec_edge_symmetric_difference_count": int(symmetric_difference_count),
        "changed_rec_edges": int(replaced_edges),
        "changed_rec_edge_ratio": float(replaced_edges / (rec_edges or 1)),
        "changed_user_count": int(changed_users),
        "changed_user_ratio": float(changed_users / (len(original_rec_by_user) or 1)),
        "zero_popularity_item_count": int(len(zero_popularity)),
        "zero_popularity_selected_count": int(zero_selected),
        "zero_popularity_selected_ratio": float(zero_selected / (sum(len(values) for values in output_rec_by_user.values()) or 1)),
        "before_graph_metrics": before_metrics,
        "after_graph_metrics": after_metrics,
        "education_cost": edu_metrics,
        "fairness_context": context.metadata,
    }
    write_json(output_graph_dir / "fairness_preprocess_manifest.json", manifest)
    print(f"created fairness preprocessing graph: {output_graph_dir}")
    print(json.dumps(jsonable({"changed_rec_edge_ratio": manifest["changed_rec_edge_ratio"], "education_cost": edu_metrics}), ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fairness-aware rec-edge preprocessing graph.")
    parser.add_argument("--source-graph-dir", type=Path, required=True)
    parser.add_argument("--output-graph-dir", type=Path, required=True)
    parser.add_argument("--method", default="fair_score", choices=["fair_score"])
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--top-k-rec", type=int, default=None)
    parser.add_argument("--lambda-item", type=float, default=0.1)
    parser.add_argument("--lambda-kc", type=float, default=0.1)
    parser.add_argument("--popularity-source", choices=["rec_triples", "train_interactions", "auto"], default="train_interactions")
    parser.add_argument("--popularity-aggregation", choices=["unique_users", "interactions"], default="unique_users")
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--long-tail-ratio", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_fair_preprocess_graph(
        source_graph_dir=args.source_graph_dir,
        output_graph_dir=args.output_graph_dir,
        method=args.method,
        candidate_pool_size=args.candidate_pool_size,
        top_k_rec=args.top_k_rec,
        lambda_item=args.lambda_item,
        lambda_kc=args.lambda_kc,
        popularity_source=args.popularity_source,
        popularity_aggregation=args.popularity_aggregation,
        head_ratio=args.head_ratio,
        long_tail_ratio=args.long_tail_ratio,
        epsilon=args.epsilon,
        force=args.force,
    )


if __name__ == "__main__":
    main()
