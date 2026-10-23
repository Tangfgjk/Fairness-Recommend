"""Copy baseline artifacts once, validate final datasets, lock pairs, re-evaluate baseline metrics."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debias.io import DATASETS
from debias.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--import-source", type=Path, help="Original 完整运行结果20261011 directory; only needed once")
    args = parser.parse_args()
    for name in args.datasets.split(","):
        if name not in DATASETS:
            parser.error(f"Unknown dataset: {name}")
        print(f"Preparing {name}", flush=True)
        ctx, folder = prepare(name, import_source=args.import_source)
        from debias.diagnostics import sampling_counts, prepare_summary
        if not (folder / "sampling_counts.json").exists():
            sampling_counts(ctx, folder / "sampling_counts.json")
        prepare_summary(folder, name)
        print(f"Ready: {folder}; train={len(ctx.train)}, test={len(ctx.test)}", flush=True)


if __name__ == "__main__":
    main()
