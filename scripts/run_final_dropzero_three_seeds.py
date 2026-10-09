"""Run the selected rec=50, zero-forgetting-edge experiment on one computer.

Run all six ablations at seeds 2024/2025/2026 before any comparison model.
Data_Fin_New is intentionally excluded from Git and must be copied separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2024, 2025, 2026)
TOP_KS = tuple(range(5, 51, 5))
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
ABLATIONS = (
    "feature_only",
    "feature_only_no_mastery",
    "feature_only_no_forgetting",
    "feature_only_learner_id",
    "feature_only_exercise_id",
    "id_only",
)
MODEL_DIRS = tuple("SemanticConvE_" + model for model in ABLATIONS)
COMPARISONS = ("EB-CF", "SB-CF", "CBF", "TransE", "TransE-adv", "RotatE", "DistMult", "ComplEx")


@dataclass(frozen=True)
class Dataset:
    name: str
    data_root: Path
    graph_subdir: str
    runs_root: Path
    ablation_graph_root: Path
    comparison_root: Path
    run_id: str
    variant_name: str

    @property
    def graph(self) -> Path:
        return self.data_root / self.name / self.graph_subdir

    @property
    def ablation_run(self) -> Path:
        return self.runs_root / self.name / self.run_id

    @property
    def comparison_run(self) -> Path:
        return self.comparison_root / self.name / self.run_id


def dataset_config(name: str) -> Dataset:
    base = ROOT / "Data_Fin_New"
    if name == "Eedi":
        root = base / "_resplit_seed20261007"
        variant = "drop_exfr0p00_rec50"
        return Dataset(name, root, f"forgetting_variants/{variant}/er_graph",
                       root / "_forgetting_edge_runs" / variant,
                       root / "_forgetting_edge_ablation_graphs" / variant,
                       root / "_forgetting_edge_comparison_runs" / variant,
                       "resplit_seed20261007_modelseed2024", variant)
    if name in ("algebra2005", "XES3G5M-sub-small"):
        variant = "drop_exfr0p00"
        return Dataset(name, base, f"forgetting_variants_rec50/{variant}/er_graph",
                       base / "_forgetting_edge_runs_rec50" / variant,
                       base / "_forgetting_edge_ablation_graphs_rec50" / variant,
                       base / "_forgetting_edge_comparison_runs_rec50" / variant,
                       "forgetting_rec50_seed2024", variant)
    raise ValueError(f"Unsupported dataset: {name}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_inputs(config: Dataset) -> None:
    graph = config.graph
    required_graph = (
        "triples.txt", "test_triples.txt", "entities.dict", "relations.dict",
        "Q.txt", "stu2know_mastery.json", "stu2know_forget.json",
        "stu2ex_forget.json", "stu2ex_recommend.json",
        "stu2ex_recommend_full_precision.json",
        "semantic_kg_features/text_embeddings/shared_text_encoder.pt",
        "er_graph_manifest.json",
    )
    required_data = (
        config.data_root / config.name / "front_features" / "ektm_mirt" / "best.pt",
        config.data_root / "_ep_readouts" / config.name / "readout.pt",
        config.data_root / "_ep_readouts" / config.name / "full_history_hidden.npy",
        config.data_root / "_ep_readouts" / config.name / "stu2know_mastery.npy",
    )
    missing = [str(graph / file) for file in required_graph if not (graph / file).is_file()]
    missing += [str(path) for path in required_data if not path.exists()]
    dataset_dir = config.data_root / config.name
    if not any((dataset_dir / name).is_dir() for name in ("raw", "raw_compact")):
        missing.append(str(dataset_dir / "raw (or raw_compact)"))
    if missing:
        raise FileNotFoundError("Required dataset files are missing:\n" + "\n".join(missing))
    manifest = read_json(graph / "er_graph_manifest.json")
    rec = manifest.get("recommendation", {})
    variant = manifest.get("forgetting_edge_variant", {})
    if (manifest.get("dataset") != config.name
            or (rec.get("delta_1"), rec.get("delta_2"), rec.get("top_k_rec")) != (0.8, 0.8, 50)
            or variant.get("name") != config.variant_name):
        raise ValueError(f"Wrong dataset, graph variant or (theta_m,theta_f,rec): {graph}")
    train = manifest.get("triple_counts", {}).get("train", {})
    if train.get("exfr") != variant.get("retained_exfr_edges"):
        raise ValueError(f"Retained forgetting-edge count differs from manifest: {graph}")
    if config.name == "Eedi" and manifest.get("student_counts") != {"train": 701, "test": 234}:
        raise ValueError("Eedi graph does not use the selected 20261007 split")


def metric_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        metrics = read_json(path)
        if metrics.get("ep_thresholds") != list(THRESHOLDS):
            return False
        for metric in ("ACC", "NOV", "Ep_t0.4", "Ep_t0.5", "Ep_t0.6", "Ep_t0.7"):
            for k in TOP_KS:
                float(metrics[metric][str(k)]["mean"])
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def missing_ablations(config: Dataset) -> list[str]:
    return [
        f"{model}/seed{seed}"
        for model in MODEL_DIRS for seed in SEEDS
        if not metric_complete(config.ablation_run / model / f"seed{seed}" / "eval" / "metrics.json")
    ]


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(command: list[str], plan: bool) -> None:
    print(("PLAN: " if plan else "RUN: ") + command_text(command), flush=True)
    if not plan:
        subprocess.run(command, cwd=ROOT, check=True)


def ablation_command(config: Dataset) -> list[str]:
    return [
        sys.executable, str(ROOT / "codes-New-ConvE" / "run_semantic_experiments.py"),
        "--dataset", config.name, "--data-root", str(config.data_root),
        "--graph-subdir", config.graph_subdir,
        "--runs-root", str(config.runs_root),
        "--ablation-data-root", str(config.ablation_graph_root),
        "--run-id", config.run_id, "--seeds", ",".join(map(str, SEEDS)),
        "--ablations", ",".join(ABLATIONS),
        "--delta-1", "0.8", "--delta-2", "0.8", "--top-k-rec", "50",
        "--top-ks", ",".join(map(str, TOP_KS)),
        "--ep-thresholds", ",".join(map(str, THRESHOLDS)),
        "--ep-calibration-root", str(config.data_root / "_ep_readouts"),
        "--cuda", "true", "--resume",
    ]


def ablation_summary_command(config: Dataset) -> list[str]:
    return [
        sys.executable, str(ROOT / "codes-New-ConvE" / "summarize_semantic_results.py"),
        "--dataset", config.name, "--run-id", config.run_id,
        "--runs-root", str(config.runs_root),
        "--seeds", ",".join(map(str, SEEDS)), "--ablations", ",".join(ABLATIONS),
        "--top-ks", ",".join(map(str, TOP_KS)),
    ]


def comparison_command(config: Dataset, models: tuple[str, ...]) -> list[str]:
    command = [
        sys.executable, str(ROOT / "comparison_models" / "run_v9_comparison_experiments.py"),
        "--dataset", config.name, "--data-root", str(config.data_root),
        "--graph-subdir", config.graph_subdir,
        "--run-root", str(config.comparison_root), "--run-id", config.run_id,
        "--seeds", ",".join(map(str, SEEDS)), "--models", ",".join(models),
        "--top-ks", ",".join(map(str, TOP_KS)),
        "--ep-thresholds", ",".join(map(str, THRESHOLDS)),
        "--ep-calibration-root", str(config.data_root / "_ep_readouts"),
        "--cuda", "true",
    ]
    if config.comparison_run.exists():
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("Eedi", "algebra2005", "XES3G5M-sub-small"))
    parser.add_argument("--phase", choices=("all", "ablation", "comparison"), default="all")
    parser.add_argument("--plan", action="store_true", help="Validate inputs and show execution order only")
    args = parser.parse_args()
    config = dataset_config(args.dataset)
    validate_inputs(config)
    if not args.plan and args.phase != "comparison":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable in {sys.executable}")

    print(f"Dataset: {config.name} | graph: {config.graph}", flush=True)
    print(f"Existing ablations: {18 - len(missing_ablations(config))}/18", flush=True)
    if args.phase in ("all", "ablation"):
        run(ablation_command(config), args.plan)
        run(ablation_summary_command(config), args.plan)
    if args.phase in ("all", "comparison"):
        missing = missing_ablations(config)
        if missing and not args.plan:
            raise RuntimeError("Complete all 18 ablation model/seed metrics before comparison:\n" + "\n".join(missing))
        if missing:
            print(f"Comparison is gated on {len(missing)} unfinished ablation results.", flush=True)
        run(comparison_command(config, COMPARISONS[:3]), args.plan)
        for model in COMPARISONS[3:]:
            run(comparison_command(config, (model,)), args.plan)
    print(f"Finished requested phase: {args.phase} / {config.name}", flush=True)


if __name__ == "__main__":
    main()
