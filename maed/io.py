from pathlib import Path
import csv

from debias.io import ROOT, read_json, write_json, sha256, digest

VERSION = "maed_diagnostics_v1"


def source_hash():
    paths = list((ROOT / "maed").glob("*.py"))
    paths += [ROOT / "scripts/run_maed_suite.py", ROOT / "scripts/run_final_dropzero_three_seeds.py"]
    # Include reused implementation, without changing its original contract.
    for directory in ("debias", "front_pipeline", "mastery_ep_v2", "ep_kt_evaluation"):
        paths.extend((ROOT / directory).glob("*.py"))
    paths += [ROOT / "codes-New-ConvE" / name for name in
              ("semantic_conve_model.py", "run_semantic_conve.py", "feature_loader.py")]
    return digest({p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)})


def input_hashes(state):
    """Include histories, front checkpoints and Ep inputs, not only scoring matrices."""
    paths = list(state.graph.rglob('*')) + list(state.raw.rglob('*'))
    paths += list((state.data_root/state.dataset/'front_features').rglob('*'))
    paths += list((state.data_root/'_ep_readouts'/state.dataset).rglob('*'))
    paths = sorted({p for p in paths if p.is_file() and '__pycache__' not in p.parts})
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def lock_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Contract differs; choose another output directory: {path}")
    if not path.exists():
        write_json(path, value)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
