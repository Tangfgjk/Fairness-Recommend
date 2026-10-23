"""One-command E0-E4 + rule experiments. --plan never trains or evaluates models."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT, DATASETS, read_json, write_json, source_digest


def run_stage(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=environment)
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--methods", default="E0,E1,E2,E3,E4,R")
    parser.add_argument("--weight", type=float, default=.1)
    parser.add_argument("--margin", type=float, default=.1, help="Positive exploratory margin; zero is a separate mechanism probe")
    parser.add_argument("--score-penalty", type=float, default=.1)
    parser.add_argument("--cuda", default="true", choices=("true", "false"))
    parser.add_argument("--run-id", default="merit_frequency_20261022")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    datasets, seeds, methods = args.datasets.split(","), [int(s) for s in args.seeds.split(",")], args.methods.split(",")
    if not set(datasets) <= set(DATASETS) or not set(seeds) <= {2024,2025,2026} or not set(methods) <= {"E0","E1","E2","E3","E4","R"}:
        parser.error("Unsupported dataset, seed or method")
    if any(len(v) != len(set(v)) for v in (datasets, seeds, methods)):
        parser.error("Duplicate selections")
    if min(args.weight, args.margin, args.score_penalty) < 0:
        parser.error("Weights and margin must be nonnegative")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("run-id must be a single folder name")
    jobs = [(d, s, m) for d in datasets for s in seeds for m in methods]
    if args.plan:
        for d, s, m in jobs:
            print(f"{d} / seed{s} / {m}: " + ("25 epochs + evaluation" if m in ("E0","E1","E3","E4") else "scores + evaluation; no training"))
        print(f"Training jobs: {sum(m in ('E0','E1','E3','E4') for _,_,m in jobs)}; total stages: {len(jobs)}")
        return
    root = ROOT / "runs" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"weight": args.weight, "margin": args.margin, "score_penalty": args.score_penalty,
                "cuda": args.cuda, "source_hash": source_digest(), "NOV": "KG4Ex-NOV-Set",
                "Ep_threshold": .5, "baseline": "retrained E0, original protocol"}
    if (root / "protocol.json").exists() and read_json(root / "protocol.json") != protocol:
        raise ValueError("Run protocol or source changed; choose a different --run-id")
    write_json(root / "protocol.json", protocol)
    if args.worker:
        if len(jobs) != 1:
            parser.error("Worker needs exactly one job")
        work(*jobs[0], args, root)
        return
    # Each stage is isolated: Ep imports and model RNG cannot pollute the next training process.
    for i, (d, s, m) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {d}/seed{s}/{m}", flush=True)
        write_json(root / "status.json", {"stage": i, "total": len(jobs), "dataset": d, "seed": s, "method": m, "state": "running"})
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--datasets", d,
                   "--seeds", str(s), "--methods", m, "--run-id", args.run_id, "--cuda", args.cuda,
                   "--weight", str(args.weight), "--margin", str(args.margin), "--score-penalty", str(args.score_penalty)]
        try:
            run_stage(command, root / d / m / f"seed{s}" / "stage.log")
        except BaseException:
            write_json(root / "status.json", {"stage": i, "total": len(jobs), "dataset": d, "seed": s, "method": m, "state": "interrupted"})
            raise
    from debias.report import report
    path = report(root)
    write_json(root / "status.json", {"state": "complete", "stages": len(jobs), "report": str(path)})
    print(f"Completed. Tables: {path}", flush=True)


def work(dataset, seed, method, args, root):
    import numpy as np
    from debias.prepare import prepared, baseline_folder
    from debias.training import train, export_scores
    from debias.context import load_scores
    from debias.evaluation import evaluate_file, write_score_matrix, frequency_correct
    from debias.io import sha256
    ctx, arrays, manifest = prepared(dataset)
    out = root / dataset / method / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    score_file = out / "uid_ex_scores.json"
    baseline = root / dataset / "E0" / f"seed{seed}" / "uid_ex_scores.json"
    score_meta = out / "score_provenance.json"
    if method in ("E0", "E1", "E3", "E4"):
        train(ctx, arrays, manifest, out, method, seed, args.weight, args.margin, args.cuda)
        provenance = {"checkpoint_sha256": sha256(out / "best.pt"), "data": ctx.fingerprint}
        if not score_meta.exists():
            export_scores(ctx, out / "best.pt", score_file, args.cuda)
            provenance["score_sha256"] = sha256(score_file)
            write_json(score_meta, provenance)
        else:
            saved = read_json(score_meta)
            if any(saved[k] != v for k,v in provenance.items()) or saved["score_sha256"] != sha256(score_file):
                raise ValueError("Exported scores changed")
    else:
        if method == "E2":
            if not baseline.exists():
                raise FileNotFoundError("E2 requires completed E0 scores for this seed")
            scores = frequency_correct(load_scores(baseline, ctx), ctx.frequency, args.score_penalty)
        else:
            scores = -ctx.distance[ctx.test]
        provenance = {"data": ctx.fingerprint, "method": method, "score_penalty": args.score_penalty if method=="E2" else 0,
                      "baseline_sha256": sha256(baseline) if method=="E2" else None}
        if score_meta.exists() and read_json(score_meta) != provenance:
            raise ValueError("Correction score provenance changed")
        write_score_matrix(score_file, ctx, scores)
        write_json(score_meta, provenance)
    reference = baseline_folder(dataset,seed) / "eval/uid_ex_scores.json" if method == "E0" else baseline
    evaluate_file(ctx, arrays, manifest, score_file, out / "eval", baseline=reference if reference.exists() else None)
    from debias.report import report
    report(root)


if __name__ == "__main__":
    main()
