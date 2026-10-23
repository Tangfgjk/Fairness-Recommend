"""Inspectable single-seed paper tables; primary best checkpoints and last diagnostics separated."""
import html
from pathlib import Path
from debias.io import DATASETS,TOP_NS,read_json,sha256,write_json
from .io import write_csv
from .evaluation import save_comparison,compare_results


def checked_result(path):
    path=Path(path)
    if not path.exists():return None
    marker=read_json(path.parent/'evaluation_complete.json')
    if marker['metrics_sha256']!=sha256(path):raise ValueError(f'Metrics changed: {path}')
    return read_json(path)


def monitor_report(run_root,output):
    """Readable epoch diagnostics, including initialization; no selection by their values."""
    sources={};components=[];boundaries=[];gradients=[];initializations=[]
    for dataset in DATASETS:
        for method in ('M3','M4'):
            folder=Path(run_root)/dataset/method/'seed2024'
            initial=folder/'initialization_audit.json'
            entries=[]
            if initial.exists():
                row=read_json(initial);sources[str(initial)]=sha256(initial)
                initializations.append({'dataset':dataset,'method':method,'seed':row['seed'],
                                        'model_state_sha256':row['model_state_sha256']})
                entries.append((0,{'training_exposure_audit':row['training_audit']}))
            for path in sorted((folder/'epochs').glob('*.json')):
                row=read_json(path);sources[str(path)]=sha256(path);entries.append((row['epoch'],row))
            for epoch,log in entries:
                audit=log['training_exposure_audit']
                if 'Q_cancellation' not in audit:continue  # Historical v1 files stay unchanged.
                prefix={'dataset':dataset,'method':method,'epoch':epoch,'training_users':len(audit['training_users'])}
                cancel=audit['Q_cancellation']
                for name,values in audit['score_components'].items():
                    components.append({**prefix,'component':name,**values,
                        **{k:v for k,v in cancel.items() if k!='interpretation'}})
                boundaries.extend({**prefix,**row} for row in audit['boundary_per_N'])
                for row in log.get('gradient_balance',[]):
                    gradients.append({**prefix,**{k:','.join(v) if k=='roles' else v for k,v in row.items()}})
    output=Path(output)
    write_csv(output/'training_score_components.csv',components)
    write_csv(output/'training_boundary_alignment.csv',boundaries)
    write_csv(output/'training_gradient_balance.csv',gradients)
    write_csv(output/'training_initializations.csv',initializations)
    return sources


def report(run_root,baseline_root):
    run_root=Path(run_root)
    output=run_root.parent/'论文格式文档统计'/run_root.name
    output.mkdir(parents=True,exist_ok=True)
    provenance=monitor_report(run_root,output);all_rows=[]
    for dataset in DATASETS:
        results={};last={}
        paths={'E0':baseline_root/dataset/'E0/seed2024/eval/metrics.json',
               'Q-fixed':run_root/dataset/'Q-fixed/seed2024/eval/metrics.json',
               'M3':run_root/dataset/'M3/seed2024/best/eval/metrics.json',
               'M4':run_root/dataset/'M4/seed2024/best/eval/metrics.json',
               'E1-v1':baseline_root/dataset/'E1/seed2024/eval/metrics.json',
               'E2-prob':baseline_root/dataset/'E2/seed2024/eval/metrics.json'}
        for label,p in paths.items():
            value=checked_result(p)
            if value is not None:results[label]=value;provenance[str(p)]=sha256(p)
        for method in ('M3','M4'):
            p=run_root/dataset/method/'seed2024/last/eval/metrics.json'
            value=checked_result(p)
            if value is not None:last[method]=value;provenance[str(p)]=sha256(p)
        # Do not publish baseline-only reports for datasets not selected/completed yet.
        if not any(m in results for m in ('Q-fixed','M3','M4')):continue
        ds_output=output/dataset;ds_output.mkdir(parents=True,exist_ok=True)
        save_comparison(ds_output,results)
        q_risk=[];q_people=[]
        if 'Q-fixed' in results:
            for method in ('M3','M4'):
                if method in results:
                    a,b=compare_results(results[method],results['Q-fixed'],method,reference='Q-fixed')
                    q_risk.extend(a);q_people.extend(b)
        write_csv(ds_output/'quality_and_risk_vs_Q_fixed.csv',q_risk)
        write_csv(ds_output/'student_deltas_vs_Q_fixed.csv',q_people)
        if 'M3' in results and 'M4' in results:
            a,b=compare_results(results['M4'],results['M3'],'M4',reference='M3')
            write_csv(ds_output/'risk_M4_vs_M3.csv',a)
            write_csv(ds_output/'student_deltas_M4_vs_M3.csv',b)
            delta_rows=[]
            for n in TOP_NS:
                aa=results['M3']['top_n'][str(n)];bb=results['M4']['top_n'][str(n)]
                delta_rows.extend({'N':n,'metric':metric,'M3':aa[metric],'M4':bb[metric],
                    'delta_M4_minus_M3':bb[metric]-aa[metric]} for metric in sorted(set(aa)&set(bb)))
            write_csv(ds_output/'all_metrics_M4_vs_M3.csv',delta_rows)
        write_csv(ds_output/'last_epoch_metrics.csv',[{'method':method,'N':n,'metric':metric,'value':value}
            for method,result in last.items() for n,metrics_row in result['top_n'].items() for metric,value in metrics_row.items()])
        main=[m for m in ('E0','Q-fixed','M3','M4','E1-v1','E2-prob') if m in results]
        metrics=sorted({k for m in main for row in results[m]['top_n'].values() for k in row})
        page=['<!doctype html><meta charset="utf-8"><style>body{font-family:Arial;margin:32px}table{border-collapse:collapse;width:100%;margin:20px 0 40px}th,td{text-align:center;padding:8px;border-bottom:1px solid #ccc}th{border-top:2px solid}h2{break-before:page}</style>',
              f'<h1>{html.escape(dataset)} — MAED controlled comparison</h1>',
              '<p>Seed 2024; no across-seed standard deviation. M3: ConvE+Q; M4: ConvE+Q+boundary-calibrated exposure. Q-fixed reuses frozen E0. E1/E2 are archived same-round comparisons.</p>',
              '<p>NOV = KG4Ex-NOV-Set; Ep threshold 0.5. Per-N budgets versus E0: Ada decline ≤0.01, NOV absolute decline ≤0.03. These do not guarantee individual-student protection.</p>',
              '<p>Primary results use the minimum-supervised-BCE checkpoint. Last-epoch results are separate diagnostics. Snapshot protocol and existing preprocessing limitations remain.</p>',
              '<p>BaselineOverlap: E0 versus archived original; other methods versus same-round E0. CFP and list diversity are diagnostics without an automatic fairness direction. Only comparisons versus E0 use the formal quality budget; M4 versus M3 is reported separately.</p>',
              '<p>Available primary methods: '+html.escape(', '.join(main))+'</p>']
        for metric in metrics:
            page.append(f'<h2>{html.escape(metric)}</h2><table><tr><th>N</th>'+''.join(f'<th>{html.escape(m)}</th>' for m in main)+'</tr>')
            for n in TOP_NS:
                values={m:results[m]['top_n'][str(n)].get(metric) for m in main}
                directional=metric in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','DED','ItemGini','KCGini','ItemCoverage','KCCoverage')
                ranks=sorted({v for v in values.values() if v is not None},reverse=metric not in ('DED','ItemGini','KCGini'))
                cells=[]
                for m,v in values.items():
                    text='NA' if v is None else f'{v:.4f}'
                    if directional and v is not None:
                        if v==ranks[0]:text=f'<b>{text}</b>'
                        elif len(ranks)>1 and v==ranks[1]:text=f'<u>{text}</u>'
                    cells.append('<td>'+text+'</td>')
                    if v is not None:all_rows.append({'dataset':dataset,'method':m,'checkpoint':'best_or_frozen','N':n,'metric':metric,'value':v})
                page.append(f'<tr><td>{n}</td>'+''.join(cells)+'</tr>')
            page.append('</table>')
        if last:
            page.append('<h2>Last-epoch diagnostics (not used to select primary results)</h2>')
            for metric in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','DED'):
                page.append(f'<h3>{html.escape(metric)}</h3><table><tr><th>N</th>'+''.join(f'<th>{m} last</th>' for m in last)+'</tr>')
                for n in TOP_NS:
                    page.append(f'<tr><td>{n}</td>'+''.join(f'<td>{r["top_n"][str(n)][metric]:.4f}</td>' for r in last.values())+'</tr>')
                page.append('</table>')
        page.append('<h2>CFP (diagnostic, not simply lower is better)</h2><table><tr><th>Method</th><th>CFP</th><th>Eligible students</th></tr>')
        for m in main:
            c=results[m]['CFP'];page.append(f'<tr><td>{html.escape(m)}</td><td>{c["value"]}</td><td>{c["users"]}</td></tr>')
        page.append('</table>')
        (output/f'{dataset}_one_metric_per_table.html').write_text('\n'.join(page),encoding='utf-8')
    write_csv(output/'all_metrics.csv',all_rows)
    write_json(output/'report_manifest.json',{'sources':provenance,'rows':len(all_rows),'seed':2024,
               'primary_checkpoint':'minimum supervised BCE','last_checkpoint_is_diagnostic':True})
    return output
