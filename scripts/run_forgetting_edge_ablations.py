"""Run six ablations on the three train-only forgetting-edge variants.

The base graph is Data_Fin_New/<dataset>/er_graph with (theta_m, theta_f, rec)
= (0.8, 0.8, 10). Variant data must be copied beside that graph first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Data_Fin_New"
DATASETS = ("algebra2005", "XES3G5M-sub-small", "Eedi")
VARIANTS = ("drop_exfr0p00", "weighted_nonzero_400", "weighted_nonzero_200")
ABLATIONS = (
    "feature_only", "feature_only_no_mastery", "feature_only_no_forgetting",
    "feature_only_learner_id", "feature_only_exercise_id", "id_only",
)
TOP_KS = "5,10,15,20,25,30,35,40,45,50"
EP_THRESHOLDS = "0.4,0.5,0.6,0.7"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def selected_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError(f"Choose unique {label} from {allowed}")
    return names


def check_graph(data_root: Path, dataset: str, variant: str) -> Path:
    graph = data_root / dataset / "forgetting_variants" / variant / "er_graph"
    required = (
        "triples.txt", "test_triples.txt", "entities.dict", "relations.dict", "Q.txt",
        "stu2know_mastery.json", "stu2know_forget.json", "stu2ex_forget.json",
        "stu2ex_recommend.json", "stu2ex_recommend_full_precision.json",
        "er_graph_manifest.json", "semantic_kg_features/text_embeddings/shared_text_encoder.pt",
    )
    missing = [str(graph / name) for name in required if not (graph / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing variant graph files:\n" + "\n".join(missing))
    manifest = json.loads((graph / "er_graph_manifest.json").read_text(encoding="utf-8"))
    params = manifest["recommendation"]
    if manifest["dataset"] != dataset or (params["delta_1"], params["delta_2"], params["top_k_rec"]) != (0.8, 0.8, 10):
        raise ValueError(f"Wrong base graph parameters: {graph}")
    info = manifest.get("forgetting_edge_variant", {})
    if info.get("name") != variant or not 0 < info.get("retained_exfr_edges", 0) <= info.get("source_exfr_edges", 0):
        raise ValueError(f"Missing or inconsistent forgetting variant manifest: {graph}")
    if manifest["triple_counts"]["train"]["exfr"] != info["retained_exfr_edges"]:
        raise ValueError(f"Forgetting edge count mismatch: {graph}")
    base = data_root / dataset / "er_graph"
    if (base / "triples.txt").is_file():
        if sha256(base / "triples.txt") != info["source_train_triples_sha256"]:
            raise ValueError(f"Variant was built from a different base graph: {graph}")
    if sha256(graph / "test_triples.txt") != info["test_triples_sha256"]:
        raise ValueError(f"Evaluation triples changed: {graph}")
    return graph


def check_ep_inputs(data_root: Path, dataset: str) -> None:
    base = data_root / dataset
    readout = data_root / "_ep_readouts" / dataset
    required = (
        base / "front_features" / "stu2know_mastery.json",
        base / "front_features" / "stu2know_forget.json",
        base / "front_features" / "ektm_mirt" / "best.pt",
        readout / "training_report.json", readout / "readout.pt",
        readout / "full_history_hidden.npy", readout / "stu2know_mastery.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset or Ep calibration files:\n" + "\n".join(missing))


def check_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in {sys.executable}; select a CUDA-enabled Python environment")


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def group_completed(run_dir: Path, seed: int) -> bool:
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return False
    if json.loads(status_path.read_text(encoding="utf-8")).get("status") != "completed":
        return False
    return all((run_dir / f"SemanticConvE_{name}" / f"seed{seed}" / "eval" / "metrics.json").is_file()
               for name in ABLATIONS)


def commands(data_root: Path, dataset: str, variant: str, seed: int, run_id: str, resume: bool):
    graph_subdir = Path("forgetting_variants") / variant / "er_graph"
    runs_root = data_root / "_forgetting_edge_runs" / variant
    ablation_root = data_root / "_forgetting_edge_ablation_graphs" / variant
    train = [
        sys.executable, str(ROOT / "codes-New-ConvE" / "run_semantic_experiments.py"),
        "--dataset", dataset, "--data-root", str(data_root),
        "--graph-subdir", str(graph_subdir), "--runs-root", str(runs_root),
        "--ablation-data-root", str(ablation_root), "--run-id", run_id,
        "--seeds", str(seed), "--ablations", ",".join(ABLATIONS),
        "--delta-1", "0.8", "--delta-2", "0.8", "--top-k-rec", "10",
        "--top-ks", TOP_KS, "--ep-thresholds", EP_THRESHOLDS,
        "--ep-calibration-root", str(data_root / "_ep_readouts"),
        "--cuda", "true",
    ]
    if resume:
        train.append("--resume")
    summarize = [
        sys.executable, str(ROOT / "codes-New-ConvE" / "summarize_semantic_results.py"),
        "--dataset", dataset, "--run-id", run_id, "--runs-root", str(runs_root),
        "--seeds", str(seed), "--ablations", ",".join(ABLATIONS), "--top-ks", TOP_KS,
    ]
    return runs_root / dataset / run_id, train, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="One per computer: Eedi, algebra2005, or XES3G5M-sub-small")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Check inputs and print commands without training")
    args = parser.parse_args()
    datasets = selected_csv(args.datasets, DATASETS, "datasets")
    variants = selected_csv(args.variants, VARIANTS, "variants")
    data_root = args.data_root.resolve()
    run_id = args.run_id or f"forgetting_seed{args.seed}"
    check_cuda()
    for dataset in datasets:
        check_ep_inputs(data_root, dataset)
        for variant in variants:
            graph = check_graph(data_root, dataset, variant)
            run_dir, train, summarize = commands(data_root, dataset, variant, args.seed, run_id, args.resume)
            print(f"\n[{dataset}] {variant} | six ablations | seed {args.seed} | {graph}", flush=True)
            if args.plan:
                print("TRAIN:", subprocess.list2cmdline(train))
                print("SUMMARY:", subprocess.list2cmdline(summarize))
                continue
            if run_dir.exists() and not args.resume:
                raise RuntimeError(f"Run exists; pass --resume or change --run-id: {run_dir}")
            if args.resume and group_completed(run_dir, args.seed):
                print(f"SKIP completed group: {run_dir}", flush=True)
            else:
                run(train)
            run(summarize)
    print("All requested forgetting-edge ablation groups processed.", flush=True)


if __name__ == "__main__":
    main()
