from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "merit_frequency_v1"
DATASETS = ("Eedi", "algebra2005", "XES3G5M-sub-small")
TOP_NS = tuple(range(5, 51, 5))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_digest():
    files = sorted((ROOT / "debias").glob("*.py"))
    files += [ROOT / "codes-New-ConvE" / n for n in ("semantic_conve_model.py", "feature_loader.py", "run_semantic_conve.py")]
    files += sorted((ROOT / "mastery_ep_v2").glob("*.py"))
    files += sorted((ROOT / "ep_kt_evaluation").glob("*.py"))
    files += sorted((ROOT / "front_pipeline").glob("*.py"))
    files += [ROOT / "scripts" / n for n in ("run_merit_debias_suite.py", "prepare_debias_context.py", "evaluate_merit_debias.py", "report_merit_debias.py")]
    return digest({p.relative_to(ROOT).as_posix(): sha256(p) for p in files})


def model_modules():
    path = str(ROOT / "codes-New-ConvE")
    if path not in sys.path:
        sys.path.insert(0, path)
    import run_semantic_conve as original
    return original


def ep_module():
    # The inherited modules use unqualified imports; keep their directories explicit.
    for folder in ("ep_kt_evaluation", "mastery_ep_v2"):
        path = str(ROOT / folder)
        if path not in sys.path:
            sys.path.insert(0, path)
    name = "merit_original_ep_evaluate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "mastery_ep_v2/evaluate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
