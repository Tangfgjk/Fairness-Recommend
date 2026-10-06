"""Run six ablations for the selected dataset's graph parameter groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "Data_Fin_New"
DATASETS = ("algebra2005", "XES3G5M-sub-small", "Eedi")
ABLATIONS = (
    "feature_only",                  # main model
    "feature_only_no_mastery",
    "feature_only_no_forgetting",
    "feature_only_learner_id",       # No-student-feature
    "feature_only_exercise_id",      # No-Exercise-feature
    "id_only",                      # ID-based Only
)
TOP_KS = "5,10,15,20,25,30,35,40,45,50"


@dataclass(frozen=True)
class Variant:
    name: str
    theta_m: float
    theta_f: float
    rec_count: int


BASE_VARIANT = Variant("base_theta_m0p8_theta_f0p8_rec10", 0.8, 0.8, 10)
DATASET_SPECIAL_VARIANTS = {
    "Eedi": (
        Variant("theta_m0p8_theta_f0p6_rec10", 0.8, 0.6, 10),
        Variant("theta_m0p8_theta_f0p6_rec50", 0.8, 0.6, 50),
    ),
    "algebra2005": (
        Variant("theta_m0p8_theta_f0p5_rec50", 0.8, 0.5, 50),
        Variant("theta_m0p8_theta_f0p4_rec50", 0.8, 0.4, 50),
    ),
    "XES3G5M-sub-small": (
        Variant("theta_m0p8_theta_f0p4_rec50", 0.8, 0.4, 50),
        Variant("theta_m0p8_theta_f0p4_rec100", 0.8, 0.4, 100),
    ),
}


def variants_for_dataset(dataset: str) -> tuple[Variant, ...]:
    return (*DATASET_SPECIAL_VARIANTS[dataset], BASE_VARIANT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(dataset: str, data_root: Path) -> dict[str, str]:
    base = data_root / dataset
    raw_name = "raw_compact" if dataset == "XES3G5M-sub-small" else "raw"
    return {
        "mastery": sha256(base / "front_features" / "stu2know_mastery.json"),
        "forgetting": sha256(base / "front_features" / "stu2know_forget.json"),
        "q_matrix": sha256(base / raw_name / "Q.txt"),
        "split": sha256(base / raw_name / "student_split.csv"),
    }


def expected_spec(dataset: str, variant: Variant, data_root: Path) -> dict:
    return {
        "dataset": dataset,
        "theta_m": variant.theta_m,
        "theta_f": variant.theta_f,
        "rec_count": variant.rec_count,
        "source_sha256": source_fingerprint(dataset, data_root),
    }


def graph_path(dataset: str, variant: Variant, data_root: Path) -> Path:
    if variant == BASE_VARIANT:
        return data_root / dataset / "er_graph"
    return data_root / dataset / "variants" / variant.name / "er_graph"


def check_graph(dataset: str, variant: Variant, path: Path) -> None:
    manifest_path = path / "er_graph_manifest.json"
    required = ("entities.dict", "relations.dict", "Q.txt", "triples.txt",
                "test_triples.txt", "stu2ex_recommend_full_precision.json",
                "semantic_kg_features")
    if not manifest_path.is_file() or any(not (path / name).exists() for name in required):
        raise RuntimeError(f"Incomplete graph: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = manifest["recommendation"]
    observed = (float(rec["delta_1"]), float(rec["delta_2"]), int(rec["top_k_rec"]))
    expected = (variant.theta_m, variant.theta_f, variant.rec_count)
    if manifest["dataset"] != dataset or observed != expected:
        raise RuntimeError(f"Graph parameter mismatch: {path}: {observed} != {expected}")
    train = manifest["triple_counts"]["train"]
    students = manifest["student_counts"]["train"]
    if train["rec"] != students * variant.rec_count:
        raise RuntimeError(f"Unexpected number of recommendation edges: {path}")
    exercise_count = manifest["ranges"]["exercise_forgetting"]["shape"][1]
    if train["exfr"] != students * exercise_count:
        raise RuntimeError(f"Forgetting edges were reduced: {path}")


def run(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_inputs(dataset: str, data_root: Path) -> None:
    base = data_root / dataset
    readout = data_root / "_ep_readouts" / dataset
    required = (
        base / "front_features" / "stu2know_mastery.json",
        base / "front_features" / "stu2know_forget.json",
        base / "front_features" / "ektm_mirt" / "best.pt",
        base / "er_graph" / "semantic_kg_features" / "text_embeddings" / "shared_text_encoder.pt",
        readout / "training_report.json",
        readout / "readout.pt",
        readout / "full_history_hidden.npy",
        readout / "stu2know_mastery.npy",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing copied data or Ep readout files:\n" +
                                "\n".join(str(path) for path in missing))


def prepare_graph(dataset: str, variant: Variant, data_root: Path, plan: bool) -> Path:
    target = graph_path(dataset, variant, data_root)
    if variant == BASE_VARIANT:
        check_graph(dataset, variant, target)
        return target
    spec = expected_spec(dataset, variant, data_root)
    spec_path = target.parent / "variant_spec.json"
    if spec_path.is_file():
        old = json.loads(spec_path.read_text(encoding="utf-8"))
        if old != spec:
            raise RuntimeError(f"Variant source or parameters changed: {spec_path}")
        check_graph(dataset, variant, target)
        return target
    if target.exists() and any(target.iterdir()):
        # A prior run can finish graph construction but stop before writing the
        # fingerprint. Validate that graph, then adopt it for safe resumption.
        check_graph(dataset, variant, target)
        if not plan:
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return target
    raw_name = "raw_compact" if dataset == "XES3G5M-sub-small" else "raw"
    command = [sys.executable, str(ROOT / "front_pipeline" / "build_er_graph.py"),
               "--dataset", dataset, "--data-fin-root", str(data_root),
               "--raw-name", raw_name, "--output-graph", str(target),
               "--delta-1", str(variant.theta_m), "--delta-2", str(variant.theta_f),
               "--top-k-rec", str(variant.rec_count)]
    if plan:
        print("PREPARE:", subprocess.list2cmdline(command))
        return target
    run(command)
    check_graph(dataset, variant, target)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return target


def group_completed(run_dir: Path) -> bool:
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        return False
    return all((run_dir / f"SemanticConvE_{name}" / "seed2024" / "eval" / "metrics.json").is_file()
               for name in ABLATIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="sensitivity_seed2024")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same run-id after an interruption.")
    parser.add_argument("--plan", action="store_true",
                        help="Validate the base graphs and print the work plan; do not train.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing the copied Data_Fin_New datasets."
                             " Defaults to the repository's Data_Fin_New folder.")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--variants", default=None,
                        help="Optional comma-separated variant names; by default run two"
                             " dataset-specific groups, then the shared base group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets or any(item not in DATASETS for item in datasets):
        raise ValueError(f"Choose datasets from {DATASETS}")
    for dataset in datasets:
        if not (data_root / dataset).is_dir():
            raise FileNotFoundError(f"Copy {dataset} into {data_root} before running")
        check_inputs(dataset, data_root)
        by_name = {item.name: item for item in variants_for_dataset(dataset)}
        names = ([item.strip() for item in args.variants.split(",") if item.strip()]
                 if args.variants is not None else list(by_name))
        if not names or len(set(names)) != len(names) or any(name not in by_name for name in names):
            raise ValueError(f"Choose unique variants for {dataset} from {tuple(by_name)}")
        for name in names:
            variant = by_name[name]
            print(f"\n[{dataset}] {name}: six models, seed 2024", flush=True)
            graph = prepare_graph(dataset, variant, data_root, args.plan)
            runs_root = data_root / "_sensitivity_runs" / name
            run_dir = runs_root / dataset / args.run_id
            if run_dir.exists() and not args.resume and not args.plan:
                raise RuntimeError(f"Run exists; pass --resume or change --run-id: {run_dir}")
            if args.resume and not args.plan and group_completed(run_dir):
                print(f"SKIP completed group: {run_dir}", flush=True)
                continue
            command = [sys.executable, str(ROOT / "codes-New-ConvE" / "run_semantic_experiments.py"),
                       "--dataset", dataset, "--data-root", str(data_root),
                       "--graph-subdir", str(graph.relative_to(data_root / dataset)),
                       "--runs-root", str(runs_root),
                       "--ablation-data-root", str(data_root / "_sensitivity_ablation_graphs" / name),
                       "--run-id", args.run_id, "--seeds", "2024",
                       "--ablations", ",".join(ABLATIONS),
                       "--delta-1", str(variant.theta_m), "--delta-2", str(variant.theta_f),
                       "--top-k-rec", str(variant.rec_count),
                       "--top-ks", TOP_KS, "--ep-thresholds", "0.4,0.5,0.6,0.7",
                       "--ep-calibration-root", str(data_root / "_ep_readouts"),
                       "--cuda", "true"]
            if args.resume:
                command.append("--resume")
            if args.plan:
                print("TRAIN:", subprocess.list2cmdline(command))
                continue
            run(command)
            run([sys.executable, str(ROOT / "codes-New-ConvE" / "summarize_semantic_results.py"),
                 "--dataset", dataset, "--run-id", args.run_id,
                 "--runs-root", str(runs_root), "--seeds", "2024",
                 "--ablations", ",".join(ABLATIONS), "--top-ks", TOP_KS])


if __name__ == "__main__":
    main()
