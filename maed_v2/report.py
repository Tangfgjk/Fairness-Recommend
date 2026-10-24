"""Primary A/B paper tables, paired quality budgets and training mechanism audits."""
import html
from pathlib import Path

from debias.io import ROOT,DATASETS,TOP_NS,read_json,write_json,sha256
from maed.io import write_csv
from maed.training_report import checked_result
from maed.evaluation import save_comparison,compare_results


def report(run_root):
    root=Path(run_root);output=ROOT/'runs/论文格式文档统计'/root.name
    output.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'runs/merit_frequency_20261022'
    v1=ROOT/'runs/maed_controlled_monitored_seed2024_20261023'
    sources={};all_rows=[];components=[];boundaries=[];gradients=[];init=[]
    statuses=[]
    for dataset in DATASETS:
        paths={m:baseline/dataset/m/'seed2024/eval/metrics.json' for m in ('E0','R')}
        paths['E1-v1']=baseline/dataset/'E1/seed2024/eval/metrics.json'
        paths['E2']=baseline/dataset/'E2/seed2024/eval/metrics.json'
        for m in ('Q-fixed','M3','M4'):
            paths[m]=v1/dataset/m/('seed2024/eval/metrics.json' if m=='Q-fixed' else 'seed2024/best/eval/metrics.json')
        for m in ('C0','C3','C4'):
            for p in ('A','B'):paths[f'{m}-{p}']=root/dataset/m/f'seed2024/best/inference_{p}/eval/metrics.json'
        results={}
        for name,path in paths.items():
            if not path.exists():continue
            results[name]=checked_result(path);sources[str(path)]=sha256(path)
        if not any(m in results for m in ('C0-A','C0-B','C3-B','C4-B')):continue
        folder=output/dataset;folder.mkdir(exist_ok=True)
        save_comparison(folder,results)
        for new,reference in (('C0-A','E0'),('C0-B','C0-A'),('C3-B','C0-B'),('C4-B','C3-B'),('C4-B','M4')):
            if new not in results or reference not in results:continue
            rows,people=compare_results(results[new],results[reference],new,reference=reference)
            write_csv(folder/f'quality_risk_{new}_vs_{reference}.csv',rows)
            write_csv(folder/f'student_deltas_{new}_vs_{reference}.csv',people)
            write_csv(folder/f'all_metrics_{new}_vs_{reference}.csv',[
                {'N':n,'metric':metric,'new':v,'reference':results[reference]['top_n'][str(n)][metric],
                 'delta':v-results[reference]['top_n'][str(n)][metric]} for n in TOP_NS
                for metric,v in results[new]['top_n'][str(n)].items()
                if metric in results[reference]['top_n'][str(n)]])
        main=[m for m in ('E0','C0-A','C0-B','Q-fixed','M3','C3-B','M4','C4-B','E1-v1','E2','R') if m in results]
        metrics=sorted({k for name in main for row in results[name]['top_n'].values() for k in row})
        page=['<!doctype html><html lang="en"><head><meta charset="utf-8"><style>body{font-family:Arial;margin:28px}table{border-collapse:collapse;width:100%;margin-bottom:30px}td,th{text-align:center;padding:7px;border-bottom:1px solid #ccc}th{border-top:2px solid #34495e}h2{break-before:page}.rule{color:#777;background:#f4f4f4}</style></head><body>',
              f'<h1>{html.escape(dataset)} — MAED-v2 controlled experiments</h1>',
              '<p>Seed 2024; no across-seed standard deviation. C0: Semantic-CompGCN+ConvE; C3: C0+Q; C4: C0+Q+E. A and B use the same checkpoint; B independently adds only the current student known state. A is the fixed training graph control.</p>',
              '<p>Best checkpoint: minimum supervised BCE. NOV=KG4Ex-NOV-Set; Ep threshold=0.5. Per-N budgets versus E0: Ada absolute decline ≤0.01; NOV absolute decline ≤0.03. R is a rule reference, excluded from best/second-best highlighting. Ranking scores are logits, not calibrated probabilities.</p>',
              '<p>Bold/underline: best/second distinct value at displayed four decimals where a direction is specified. CFP, overlap and changed-set ratios are descriptive diagnostics. This is an observed-history endpoint snapshot protocol; B-A changes the state-input path as well as graph degrees.</p>']
        for metric in metrics:
            lower=metric in ('DED','ItemGini','KCGini')
            directional=lower or metric in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','ItemCoverage','KCCoverage','ListJaccardDistance')
            page.append(f'<h2>{html.escape(metric)}</h2><table><tr><th>N</th>'+''.join(f'<th>{html.escape(m)}</th>' for m in main)+'</tr>')
            for n in TOP_NS:
                values={m:results[m]['top_n'][str(n)].get(metric) for m in main}
                ranks=sorted({round(v,4) for m,v in values.items() if m!='R' and v is not None},reverse=not lower)
                cells=[]
                for m,v in values.items():
                    text='NA' if v is None else f'{v:.4f}'
                    if directional and m!='R' and v is not None:
                        if round(v,4)==ranks[0]:text=f'<b>{text}</b>'
                        elif len(ranks)>1 and round(v,4)==ranks[1]:text=f'<u>{text}</u>'
                    cells.append(f'<td class="{"rule" if m=="R" else ""}">{text}</td>')
                    if v is not None:all_rows.append({'dataset':dataset,'method':m,'checkpoint':'best_or_frozen','N':n,'metric':metric,'value':v})
                page.append(f'<tr><td>{n}</td>'+''.join(cells)+'</tr>')
            page.append('</table>')
        page.append('<h2>CFP diagnostic</h2><table><tr><th>Method</th><th>CFP</th><th>Eligible students</th></tr>')
        for m in main:
            c=results[m]['CFP'];page.append(f'<tr><td>{html.escape(m)}</td><td>{c["value"]}</td><td>{c["users"]}</td></tr>')
        page.append('</table></body></html>')
        (output/f'{dataset}_one_metric_per_table.html').write_text('\n'.join(page),encoding='utf-8')
        last=[]
        for method in ('C0','C3','C4'):
            source=root/dataset/method/'seed2024'
            path=source/'initialization_audit.json'
            if path.exists():
                row=read_json(path);init.append({'dataset':dataset,'method':method,'model_state_sha256':row['model_state_sha256']})
            for path in sorted((source/'epochs').glob('*.json')):
                row=read_json(path);sources[str(path)]=sha256(path)
                audit=row['training_exposure_audit'];prefix={'dataset':dataset,'method':method,'epoch':row['epoch']}
                components.extend({**prefix,'component':name,**values,**{k:v for k,v in audit['Q_cancellation'].items() if k!='interpretation'}} for name,values in audit['score_components'].items())
                boundaries.extend({**prefix,**r} for r in audit['boundary_per_N'])
                gradients.extend({**prefix,**{k:','.join(v) if k=='roles' else v for k,v in r.items()}} for r in row['gradient_balance'])
            for p in ('A','B'):
                path=source/f'last/inference_{p}/eval/metrics.json'
                if path.exists():
                    result=checked_result(path)
                    last.extend({'method':method,'protocol':p,'N':n,**values} for n,values in result['top_n'].items())
            marker=source/'stage_complete.json'
            statuses.append({'dataset':dataset,'method':method,'complete':marker.exists(),
                             'epochs':len(list((source/'epochs').glob('*.json')))})
        write_csv(folder/'last_checkpoint_diagnostics.csv',last)
    write_csv(output/'all_metrics.csv',all_rows)
    write_csv(output/'training_score_components.csv',components)
    write_csv(output/'training_boundary_alignment.csv',boundaries)
    write_csv(output/'training_gradient_balance.csv',gradients)
    write_csv(output/'training_initializations.csv',init)
    for dataset in DATASETS:
        values={r['model_state_sha256'] for r in init if r['dataset']==dataset}
        if len(values)>1:raise ValueError(f'C0/C3/C4 initialization mismatch: {dataset}')
    write_json(output/'report_manifest.json',{'sources':sources,'rows':len(all_rows),'seed':2024,
               'primary_protocol':'B; C0-A always included','statuses':statuses,
               'all_nine_stages_complete':len(statuses)==9 and all(s['complete'] for s in statuses)})
    return output
