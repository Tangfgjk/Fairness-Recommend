"""No-training seed2024 mechanism audit and paper tables with a checked rule reference."""
import argparse
import html
import json
from pathlib import Path
import sys
import math

import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from debias.io import DATASETS,TOP_NS,read_json,write_json,sha256,digest
from debias.prepare import prepared
from debias.context import load_scores
from debias.metrics import evaluate,novelty_set
from maed.io import write_csv,lock_json,source_hash

VERSION='maed_seed2024_observational_audit_v2_exact_NOV_boundary_20261024'


def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None


def score_change(base,new):
    """Remove irrelevant row offsets, then split mean item shift from user residual."""
    base=np.asarray(base,float);new=np.asarray(new,float)
    if base.shape!=new.shape or base.ndim!=2 or not np.isfinite(base).all() or not np.isfinite(new).all():
        raise ValueError('Expected comparable finite user/catalog score matrices')
    delta=new-base;delta-=delta.mean(1,keepdims=True)
    shared=delta.mean(0,keepdims=True)
    residual=delta-shared
    total=float(np.square(delta).sum())
    common=float(len(delta)*np.square(shared).sum())
    individual=float(np.square(residual).sum())
    assert abs(total-common-individual)<1e-10*max(1.,total)
    return {'shared_item_energy_fraction':common/total if total>1e-30 else None,
            'individual_residual_energy_fraction':individual/total if total>1e-30 else None,
            'total_centered_energy':total},shared,residual


def novelty_change(old,new,history):
    """Exact symmetric accounting, not causal attribution; history is held fixed."""
    old=set(old);new=set(new);history=set(history)
    i0=len(old&history);i1=len(new&history)
    u0=len(old|history);u1=len(new|history)
    if min(u0,u1)==0:raise ValueError('Undefined novelty union')
    known_unit=-.5*(1/u0+1/u1)
    unseen_unit=.5*(i0+i1)/(u0*u1)
    overlap=(i1-i0)*known_unit
    denominator=(u1-u0)*unseen_unit
    delta=novelty_set(new,history)-novelty_set(old,history)
    numerator=i0*u1-i1*u0;denominator_exact=u0*u1
    assert abs(delta-overlap-denominator)<1e-12
    added=new-old;removed=old-new
    events=[{'kc':int(k),'operation':operation,'historically_correct_KC':k in history,
             'NOV_contribution':(1 if operation=='added' else -1)*(known_unit if k in history else unseen_unit)}
            for operation,ks in (('added',added),('removed',removed)) for k in sorted(ks)]
    assert abs(sum(e['NOV_contribution'] for e in events)-delta)<1e-12
    return {'old_NOV':novelty_set(old,history),'new_NOV':novelty_set(new,history),'delta_NOV':delta,
            'NOV_change_numerator':numerator,'NOV_change_denominator':denominator_exact,
            'loss_gt_0p03_exact':100*numerator < -3*denominator_exact,
            'loss_eq_0p03_exact':100*numerator == -3*denominator_exact,
            'known_overlap_contribution':overlap,'unseen_union_contribution':denominator,
            'old_known_KCs':i0,'new_known_KCs':i1,'old_KCs':len(old),'new_KCs':len(new),
            'history_KCs':len(history),'added_KCs':sorted(int(k) for k in added),'removed_KCs':sorted(int(k) for k in removed),
            'added_known_KCs':len(added&history),'removed_known_KCs':len(removed&history),
            'added_unseen_KCs':len(added-history),'removed_unseen_KCs':len(removed-history)},events


def pair_distances(rank,n):
    sets=[set(row[:n]) for row in rank]
    return np.array([1-len(sets[i]&sets[j])/len(sets[i]|sets[j]) for i in range(len(sets)) for j in range(i)])


def checked_method(folder,ctx,arrays,manifest,sources,baseline):
    score=folder/'uid_ex_scores.json';path=folder/'eval/metrics.json';marker=folder/'eval/evaluation_complete.json'
    meta=read_json(marker);result=read_json(path)
    for p,want in ((score,meta['contract']['scores_sha256']),(path,meta['metrics_sha256'])):
        got=sha256(p)
        if got!=want:raise ValueError(f'Artifact mismatch: {p}')
        sources[str(p)]=got
    sources[str(marker)]=sha256(marker)
    contract=meta['contract']
    if contract['context_id']!=manifest['context_id'] or contract['pair_hash']!=manifest['pairs_sha256']:
        raise ValueError('Different cohort/matched-pair protocol')
    if contract['NOV']!='KG4Ex-NOV-Set' or contract['Ep_threshold']!=.5 or contract['top_ns']!=list(TOP_NS):
        raise ValueError('Different metric protocol')
    if result['context_hash']!=ctx.fingerprint:raise ValueError('Different data inputs')
    values=load_scores(score,ctx)
    fresh=evaluate(ctx,values,arrays['evaluation'],baseline=baseline)
    assert fresh['ranked_top50']==result['ranked_top50']
    for n in TOP_NS:
        for metric,value in fresh['top_n'][str(n)].items():
            # E0 BaselineOverlap is against archived original, other methods against E0.
            if metric in ('BaselineOverlap','ChangedSet'):continue
            if abs(value-result['top_n'][str(n)][metric])>1e-10:raise ValueError('Metric regression')
    assert abs(fresh['CFP']['value']-result['CFP']['value'])<1e-12
    ep=folder/'eval/ep/ep_per_user.json';sources[str(ep)]=sha256(ep)
    people=read_json(ep)
    for n in TOP_NS:
        rows=[r for r in people if r['top_n']==n]
        assert len(rows)==len(ctx.test) and {int(r['uid'].removeprefix('uid')) for r in rows}==set(ctx.test)
        assert all(r['threshold']==.5 for r in rows)
        assert abs(np.mean([r['ep'] for r in rows])-result['top_n'][str(n)]['Ep_t0.5'])<1e-12
    return result,values


def training_audit(root,dataset,sources):
    boundary=[];gradients=[];score=[];summary=[]
    for method in ('M3','M4'):
        folder=root/dataset/method/'seed2024'
        initial=folder/'initialization_audit.json';init=read_json(initial)
        sources[str(initial)]=sha256(initial)
        entries=[(0,{'training_exposure_audit':init['training_audit']})]
        for epoch in range(1,26):
            p=folder/'epochs'/f'{epoch:03d}.json';row=read_json(p)
            sources[str(p)]=sha256(p);entries.append((epoch,row))
        for epoch,entry in entries:
            audit=entry['training_exposure_audit'];assert not audit['test_data_used']
            for i,b in enumerate(audit['boundary_per_N']):
                boundary.append({'dataset':dataset,'method':method,'epoch':epoch,**b,
                    'soft_TV':audit['soft_TV'][i],'hard_DED':audit['hard_TV'][i],
                    'soft_hard_DED_gap':audit['soft_TV'][i]-audit['hard_TV'][i]})
            for name,values in audit['score_components'].items():
                score.append({'dataset':dataset,'method':method,'epoch':epoch,'component':name,**values,
                    **{k:v for k,v in audit['Q_cancellation'].items() if k!='interpretation'}})
            for g in entry.get('gradient_balance',[]):
                gradients.append({'dataset':dataset,'method':method,'epoch':epoch,
                                  **{k:','.join(v) if k=='roles' else v for k,v in g.items()}})
        for n in TOP_NS:
            rows=[r for r in boundary if r['method']==method and r['N']==n and r['epoch']>0]
            ds=np.diff([r['soft_TV'] for r in rows]);dh=np.diff([r['hard_DED'] for r in rows])
            summary.append({'dataset':dataset,'method':method,'N':n,
                'first_soft_mass':rows[0]['actual_TopN_soft_mass_fraction'],
                'last_soft_mass':rows[-1]['actual_TopN_soft_mass_fraction'],
                'last_gap_over_temperature':rows[-1]['median_gap_over_temperature'],
                'warning_epochs':sum(r['mechanism_warning'] for r in rows),
                'soft_decrease_hard_increase_transitions':int(np.sum((ds<-1e-8)&(dh>1e-8))),
                'transitions':len(ds),'last_soft_TV':rows[-1]['soft_TV'],'last_hard_DED':rows[-1]['hard_DED']})
    return boundary,gradients,score,summary


def personalization(ctx,scores,arrays):
    base=scores['M3'];new=scores['M4']
    raw,shift,_=score_change(base,new)
    standard=lambda x:(x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12)
    normalized,_,_=score_change(standard(base),standard(new))
    counter=evaluate(ctx,base+shift,arrays['evaluation'],baseline=base)
    rank={k:np.argsort(-v,axis=1,kind='stable') for k,v in scores.items()}
    rank['shared_item_only']=np.argsort(-(base+shift),axis=1,kind='stable')
    m=ctx.mastery[ctx.test]
    mastery=np.array([np.sqrt(np.square(m[i]-m[j]).mean()) for i in range(len(m)) for j in range(i)])
    rows=[]
    for n in TOP_NS:
        rule=pair_distances(rank['R'],n)
        for method in ('E0','Q-fixed','M3','M4','shared_item_only'):
            distance=pair_distances(rank[method],n)
            rows.append({'dataset':ctx.name,'method':method,'N':n,'test_students':len(ctx.test),
                'student_pairs':len(distance),'mastery_distance_list_distance_spearman':correlation(mastery,distance),
                'rule_list_distance_model_list_distance_spearman':correlation(rule,distance),
                'mean_list_Jaccard_distance':float(distance.mean()),
                'scope':'descriptive dependent student pairs; no independent-pair p-value'})
    return {'dataset':ctx.name,'M3_to_M4':raw,'row_standardized_sensitivity':normalized,
            'scope':'test-cohort observational score accounting, not parameter or causal disentanglement'},counter,rows


def novelty_audit(ctx,results,scores):
    rank={k:np.argsort(-v,axis=1,kind='stable') for k,v in scores.items()}
    students=[];kc_totals={};prefix=[]
    for reference in ('E0','M3'):
        for n in TOP_NS:
            for row,uid in enumerate(ctx.test):
                old=rank[reference][row,:n];new=rank['M4'][row,:n]
                old_kc=set(np.flatnonzero(ctx.q[old].sum(0)));new_kc=set(np.flatnonzero(ctx.q[new].sum(0)))
                change,events=novelty_change(old_kc,new_kc,ctx.correct[int(uid)])
                a=next(x for x in results['M4']['per_user'] if x['uid']==uid and x['N']==n)
                b=next(x for x in results[reference]['per_user'] if x['uid']==uid and x['N']==n)
                assert abs(change['delta_NOV']-(a['KG4Ex-NOV-Set']-b['KG4Ex-NOV-Set']))<1e-12
                students.append({'dataset':ctx.name,'reference':reference,'uid':int(uid),'N':n,
                    **{k:json.dumps(v) if isinstance(v,list) else v for k,v in change.items()},
                    'Ada_delta':a['Ada']-b['Ada'],'old_exercises':json.dumps(old.tolist()),'new_exercises':json.dumps(new.tolist())})
                if ctx.name=='algebra2005':
                    for e in events:
                        key=(reference,n,e['kc'],e['operation'],e['historically_correct_KC'])
                        total=kc_totals.setdefault(key,{'students':0,'NOV_contribution_sum':0.})
                        total['students']+=1;total['NOV_contribution_sum']+=e['NOV_contribution']
    for n in TOP_NS:
        for row,uid in enumerate(ctx.test):
            top=rank['M4'][row,:n];earlier=rank['M4'][row,:max(0,n-5)]
            old=set(np.flatnonzero(ctx.q[earlier].sum(0)));new=set(np.flatnonzero(ctx.q[top].sum(0)))
            first=new-old;known=first&ctx.correct[int(uid)]
            prefix.append({'dataset':ctx.name,'uid':int(uid),'N':n,'new_prefix_KCs':len(first),
                'new_prefix_known_KCs':len(known),'new_prefix_unseen_KCs':len(first-known),
                'new_KC_ids':json.dumps(sorted(int(k) for k in first)),
                'new_exercises':json.dumps(top[max(0,n-5):].tolist())})
    summary=[]
    for reference in ('E0','M3'):
        for n in TOP_NS:
            rows=[r for r in students if r['reference']==reference and r['N']==n]
            delta=np.array([r['delta_NOV'] for r in rows]);loss=np.maximum(-delta,0);worst=max(1,math.ceil(len(rows)*.1))
            order=np.argsort(delta,kind='stable')[:worst]
            summary.append({'dataset':ctx.name,'reference':reference,'N':n,'students':len(rows),
                'mean_delta_NOV':float(delta.mean()),
                'loss_gt_0p03_students':sum(r['loss_gt_0p03_exact'] for r in rows),
                'loss_gt_0p03_rate':sum(r['loss_gt_0p03_exact'] for r in rows)/len(rows),
                'loss_eq_0p03_students':sum(r['loss_eq_0p03_exact'] for r in rows),
                'legacy_float_loss_gt_0p03_students':int(np.sum(delta<-.03)),
                'worst_10pct_mean_loss':float(np.sort(loss)[-worst:].mean()),'max_loss':float(loss.max()),
                'worst_10pct_share_of_total_loss':float(np.sort(loss)[-worst:].sum()/loss.sum()) if loss.sum()>0 else None,
                'worst_student_ids':json.dumps([rows[i]['uid'] for i in order]),
                'known_overlap_contribution':float(np.mean([r['known_overlap_contribution'] for r in rows])),
                'unseen_union_contribution':float(np.mean([r['unseen_union_contribution'] for r in rows])),
                'mean_added_known_KCs':float(np.mean([r['added_known_KCs'] for r in rows])),
                'mean_added_unseen_KCs':float(np.mean([r['added_unseen_KCs'] for r in rows]))})
    events=[{'dataset':ctx.name,'reference':key[0],'N':key[1],'kc':key[2],'operation':key[3],
             'historically_correct_KC':key[4],**values,
             'population_mean_NOV_contribution':values['NOV_contribution_sum']/len(ctx.test)} for key,values in kc_totals.items()]
    return students,events,prefix,summary


def paper_tables(dataset,results,output):
    main=[m for m in ('E0','Q-fixed','M3','M4','E1-v1','E2-prob') if m in results]
    columns=main+['R'];pages=[
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><style>body{font-family:Arial,sans-serif;margin:32px;color:#17283b}table{border-collapse:collapse;width:100%;margin:18px 0 36px}th,td{text-align:center;padding:8px;border-bottom:1px solid #ccd3db}th{border-top:2px solid #283e55}.reference{background:#f0f3f6}h2{break-before:page}p{line-height:1.5}small{color:#49576b}</style></head><body>',
        f'<h1>{html.escape(dataset)} — MAED seed 2024</h1>',
        '<p>M3: ConvE+Q; M4: ConvE+Q+exposure. One metric per table; no across-seed standard deviation. Best learned result is bold; second distinct result is underlined.</p>',
        '<p>R is direct educational-rule ranking (no training), shown as a separate reference and excluded from bold/underline ranking. DED=0 and RuleOverlap=1 for R by definition, not independent evidence of superior fairness.</p>',
        '<p>NOV = KG4Ex-NOV-Set; Ep threshold = 0.5 (KT answer probability, not recommendation scores). Primary learned checkpoints use minimum supervised BCE. E1/E2 are archived comparisons.</p>',
        '<p>The original exploratory budgets (Ada decline 0.01, NOV decline 0.03 versus E0) remain in the historical audit. Algebra2005 NOV losses are retained as an accepted tradeoff for this research stage, not silently relabeled as meeting that original threshold.</p>',
        '<p>CFP uses eligible matched test students only. CFP score gaps depend on score scale; probability and logit gaps must not be compared directly. Endpoint-history snapshot and inherited preprocessing limitations remain.</p>']
    all_rows=[]
    metrics=sorted({k for r in results.values() for row in r['top_n'].values() for k in row})
    for metric in metrics:
        pages.append(f'<h2>{html.escape(metric)}</h2><table><thead><tr><th>N</th>'+''.join(f'<th class="reference">R (rule reference)</th>' if m=='R' else f'<th>{html.escape(m)}</th>' for m in columns)+'</tr></thead><tbody>')
        for n in TOP_NS:
            values={m:results[m]['top_n'][str(n)].get(metric) for m in columns}
            directional=metric in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','DED','ItemGini','KCGini','ItemCoverage','KCCoverage')
            ranks=sorted({values[m] for m in main if values[m] is not None},reverse=metric not in ('DED','ItemGini','KCGini'))
            cells=[]
            for m,v in values.items():
                text='NA' if v is None else f'{v:.4f}'
                if m!='R' and directional and v is not None:
                    if v==ranks[0]:text=f'<b>{text}</b>'
                    elif len(ranks)>1 and v==ranks[1]:text=f'<u>{text}</u>'
                cells.append(('<td class="reference">' if m=='R' else '<td>')+text+'</td>')
                if v is not None:all_rows.append({'dataset':dataset,'method':m,'N':n,'metric':metric,'value':v,'reference_only':m=='R'})
            pages.append(f'<tr><td>{n}</td>'+''.join(cells)+'</tr>')
        pages.append('</tbody></table>')
    pages.append('<h2>CFP — descriptive matched-pair diagnostic</h2><table><tr><th>Method</th><th>CFP</th><th>Eligible students</th></tr>')
    for m in columns:
        c=results[m]['CFP'];pages.append(f'<tr><td>{m}</td><td>{c["value"]:.4f}</td><td>{c["users"]}</td></tr>')
    pages.append('</table></body></html>')
    (output/f'{dataset}_one_metric_per_table_with_R.html').write_text('\n'.join(pages),encoding='utf-8')
    return all_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/maed_controlled_monitored_seed2024_20261023')
    parser.add_argument('--baseline-root',type=Path,default=ROOT/'runs/merit_frequency_20261022')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/论文格式文档统计/maed_seed2024_mechanism_R_verified_20261024')
    args=parser.parse_args();run=args.run_root.resolve();baseline=args.baseline_root.resolve();out=args.output.resolve()
    if out.is_relative_to(run) or out.is_relative_to(baseline):raise ValueError('Output must not overlap source results')
    if read_json(run/'status.json')['state']!='complete':raise ValueError('Audit requires completed run')
    protocol=read_json(run/'protocol.json')
    if protocol['source_hash']!=source_hash():raise ValueError('Training implementation changed since the run')
    out.mkdir(parents=True,exist_ok=True)
    lock_json(out/'audit_protocol.json',{'version':VERSION,'script_sha256':sha256(Path(__file__)),
        'run':str(run),'baseline':str(baseline),'training_protocol':digest(protocol),'seed':2024,
        'read_only_sources':True,'new_training':False,'test_hyperparameter_selection':False})
    sources={str(run/'protocol.json'):sha256(run/'protocol.json')};summary={};tables=[]
    boundary=[];gradients=[];components=[];transitions=[];students=[];kcs=[];prefixes=[];nov=[];personal=[];shared=[];r_delta=[]
    for dataset in DATASETS:
        print('Audit',dataset,flush=True)
        ctx,arrays,manifest=prepared(dataset)
        sources.update({str(ROOT/p):h for p,h in ctx.files.items()})
        results={};scores={}
        methods={'E0':baseline/dataset/'E0/seed2024','Q-fixed':run/dataset/'Q-fixed/seed2024',
                 'M3':run/dataset/'M3/seed2024/best','M4':run/dataset/'M4/seed2024/best',
                 'E1-v1':baseline/dataset/'E1/seed2024','E2-prob':baseline/dataset/'E2/seed2024','R':baseline/dataset/'R/seed2024'}
        for method,path in methods.items():
            results[method],scores[method]=checked_method(path,ctx,arrays,manifest,sources,scores.get('E0'))
        if not np.allclose(scores['R'],-ctx.distance[ctx.test],atol=1e-12,rtol=0):raise ValueError('R is not direct formal rule')
        assert all(results['R']['top_n'][str(n)]['DED']==0 and results['R']['top_n'][str(n)]['RuleOverlap']==1 for n in TOP_NS)
        b,g,c,t=training_audit(run,dataset,sources);boundary+=b;gradients+=g;components+=c;transitions+=t
        decomposition,counter,p=personalization(ctx,scores,arrays);personal+=p
        for n in TOP_NS:
            for metric,value in counter['top_n'][str(n)].items():
                shared.append({'dataset':dataset,'N':n,'metric':metric,'M3':results['M3']['top_n'][str(n)][metric],
                    'shared_item_only':value,'M4':results['M4']['top_n'][str(n)][metric]})
            for metric,value in results['M4']['top_n'][str(n)].items():
                r=results['R']['top_n'][str(n)][metric]
                r_delta.append({'dataset':dataset,'N':n,'metric':metric,'M4':value,'R':r,'delta_M4_minus_R':value-r})
        a,e,prefix,nv=novelty_audit(ctx,results,scores);students+=a;kcs+=e;prefixes+=prefix;nov+=nv
        tables+=paper_tables(dataset,results,out)
        write_json(out/dataset/'shared_item_only_metrics.json',counter)
        comparisons={metric:{'better_Ns':sum((results['M4']['top_n'][str(n)][metric]-results['R']['top_n'][str(n)][metric])*(1 if metric not in ('DED','ItemGini','KCGini') else -1)>1e-12 for n in TOP_NS),
            'equal_Ns':sum(abs(results['M4']['top_n'][str(n)][metric]-results['R']['top_n'][str(n)][metric])<=1e-12 for n in TOP_NS)}
            for metric in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','DED','ItemGini','KCGini','ItemCoverage','KCCoverage')}
        summary[dataset]={'test_students':len(ctx.test),'R_protocol_checked':True,'score_decomposition':decomposition,
            'M4_vs_R':comparisons,'Top10':{m:r['top_n']['10'] for m,r in results.items()},
            'CFP':{m:{k:r['CFP'][k] for k in ('value','users','pairs')} for m,r in results.items()},
            'novaudit':[r for r in nv if r['reference']=='E0'],
            'last_training_boundary':[r for r in t if r['method']=='M4'],
            'personalization':[r for r in p if r['N']==10]}
    for name,rows in (('training_boundary_audit.csv',boundary),('training_gradient_audit.csv',gradients),
        ('training_Q_components.csv',components),('training_soft_hard_transitions.csv',transitions),
        ('student_NOV_KC_changes.csv',students),('algebra2005_KC_contribution.csv',kcs),
        ('M4_prefix_new_KCs.csv',prefixes),('NOV_loss_summary.csv',nov),
        ('personalization_pair_associations.csv',personal),('shared_item_counterfactual.csv',shared),
        ('M4_vs_R_all_N.csv',r_delta),('all_metrics_with_R.csv',tables)):
        write_csv(out/name,rows)
    # Recheck all recorded originals after writing outputs, including graph/front inputs.
    for path,expected in sources.items():
        if sha256(path)!=expected:raise ValueError(f'Source changed during audit: {path}')
    write_json(out/'audit_summary.json',summary)
    write_json(out/'audit_manifest.json',{'version':VERSION,'seed':2024,'sources':sources,
        'script_sha256':sha256(Path(__file__)),'source_count':len(sources),'originals_unchanged':True,
        'new_training':False,'R_highlighted_as_best':False,'production_source_hash':source_hash(),
        'scope':'observational diagnostics; dependent student pairs; no causal or cross-seed claims'})
    print('Completed without training:',out,flush=True)


if __name__=='__main__':main()
