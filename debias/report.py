from pathlib import Path
import csv
import html
import numpy as np

from .io import DATASETS, TOP_NS, read_json, write_json, sha256


def report(run_root):
    run_root = Path(run_root)
    output = run_root.parent / "论文格式文档统计" / run_root.name
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for dataset in DATASETS:
        runs = {}
        for method in ("E0", "E1", "E2", "E3", "E4", "R"):
            paths = sorted((run_root / dataset / method).glob("seed*/eval/metrics.json"))
            # Only include evaluations with a successful completion marker.
            paths = [p for p in paths if (p.parent / "evaluation_complete.json").exists()]
            for p in paths:
                if read_json(p.parent / "evaluation_complete.json")["metrics_sha256"] != sha256(p):
                    raise ValueError(f"Completed metrics checksum changed: {p}")
            runs[method] = [(p.parent.parent.name, read_json(p)) for p in paths]
        if not any(runs.values()):
            continue
        methods = [m for m, values in runs.items() if values]
        metric_names = sorted({k for values in runs.values() for _, result in values for v in result["top_n"].values() for k in v})
        pages = ["<!doctype html><meta charset='utf-8'><style>body{font-family:Arial;margin:35px}table{border-collapse:collapse;width:100%;margin-bottom:40px}th,td{padding:8px;text-align:center;border-bottom:1px solid #ccc}th{border-top:2px solid}h2{page-break-before:always}small{color:#555}</style>",
                 f"<h1>{html.escape(dataset)} — educational-merit frequency debiasing</h1>",
                 "<p>E0: ConvE; E1: matched frequency regularizer; E2: score correction; E3: unconditioned, frequency-stratified; E4: random direction; R: rule reference.</p>",
                 "<p>NOV = KG4Ex-NOV-Set. Ep threshold = 0.5. Single-seed values have no standard deviation; multiple seeds show mean ± sample SD. R is excluded from best/second-best highlighting.</p>",
                 "<p>BaselineOverlap: E0 versus archived original; other methods versus retrained E0. CFP applies only to eligible matched test students; educational/exposure metrics use the full cohort.</p>",
                 "<p>Available seeds: " + html.escape(str({m: [s for s, _ in runs[m]] for m in methods})) + "</p>"]
        for metric in metric_names:
            pages += [f"<h2>{html.escape(metric)}</h2><table><tr><th>N</th>" + "".join(f"<th>{m}</th>" for m in methods) + "</tr>"]
            for n in TOP_NS:
                cells, means = {}, {}
                for method in methods:
                    values = [v["top_n"][str(n)][metric] for _, v in runs[method] if metric in v["top_n"][str(n)]]
                    if not values:
                        cells[method] = "NA"
                        continue
                    mean = float(np.mean(values))
                    means[method] = mean
                    cells[method] = f"{mean:.4f}" + (f" ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else "")
                    all_rows.append([dataset, method, metric, n, len(values), mean, float(np.std(values, ddof=1)) if len(values)>1 else ""])
                directional = metric in ("Ada", "KG4Ex-NOV-Set", "Ep_t0.5", "ItemGini", "KCGini", "ItemCoverage", "KCCoverage", "DED")
                if directional:
                    lower = metric in ("ItemGini", "KCGini", "DED")
                    eligible = {m: v for m, v in means.items() if m != "R"}
                    ranks = sorted(set(eligible.values()), reverse=not lower)
                    for m, v in eligible.items():
                        if v == ranks[0]:
                            cells[m] = f"<b>{cells[m]}</b>"
                        elif len(ranks)>1 and v == ranks[1]:
                            cells[m] = f"<u>{cells[m]}</u>"
                pages.append(f"<tr><td>{n}</td>" + "".join(f"<td>{cells[m]}</td>" for m in methods) + "</tr>")
            pages.append("</table>")
        pages.append("<h2>CFP (diagnostic; not simply lower is better)</h2><table><tr><th>Method</th><th>Seed</th><th>CFP</th><th>Users</th><th>Pairs</th></tr>")
        for method in methods:
            for seed, result in runs[method]:
                c = result["CFP"]
                pages.append(f"<tr><td>{method}</td><td>{seed}</td><td>{c['value']}</td><td>{c['users']}</td><td>{c['pairs']}</td></tr>")
        pages.append("</table>")
        (output / f"{dataset}_one_metric_per_table.html").write_text("\n".join(pages), encoding="utf-8")
    with (output / "all_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "metric", "N", "seeds", "mean", "sample_std_across_seeds"])
        writer.writerows(all_rows)
    write_json(output / "report_manifest.json", {"run_root": str(run_root), "rows": len(all_rows), "interpretation": "Partial reports show available seeds explicitly. No automatic claim of fairness improvement."})
    return output
