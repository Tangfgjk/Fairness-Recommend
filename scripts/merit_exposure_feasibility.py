"""Offline quality-constrained exposure feasibility, not a learned recommender.

Test-cohort reference exposure and evaluation histories are used by the solver:
this is exploratory oracle-assisted allocation, never a deployable/test-blind model.
All candidate sizes and solver settings are fixed before reading new outcomes.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT,DATASETS,TOP_NS,read_json,write_json,sha256
from debias.prepare import prepared
from debias.context import load_scores
from debias.metrics import evaluate
from debias.evaluation import evaluate_file,write_score_matrix

RUN=ROOT/'runs/merit_frequency_20261022'
OUT=ROOT/'runs/论文格式文档统计/merit_frequency_20261022/曝光可行性测试_20261023'
PROTOCOL={'candidate_rule_top_k':[100,200],'quality_budget':.01,'sweeps':5,'seed':20261023,
          'candidate_union':'E0 Top50 UNION per-student education-rule TopK',
          'quality_definition':'Full test cohort mean Ada and KG4Ex-NOV-Set, every N=5:5:50',
          'proxy_guard':'Replacement education distance must not increase for this student',
          'FQ':'Greedy distance reduction; no exposure objective',
          'FQE':'Greedy reduction in mean actual DED; no individual N DED increase',
          'scope':'Exploratory test-cohort oracle allocation, not trained model; no optimality/infeasibility proof'}


def bits(row):
    return sum(1<<int(k) for k in np.flatnonzero(row))


def novelty(bitset,history):
    total=(bitset|history).bit_count()
    if not total:raise ValueError('Undefined novelty')
    return 1-(bitset&history).bit_count()/total


def ranking_scores(full_rank,top):
    """Synthetic ordinal scores serialize one complete ranking, not probabilities."""
    out=np.empty(full_rank.shape,dtype=float)
    for i,(row,new) in enumerate(zip(full_rank,top)):
        if len(set(new))!=len(new):raise ValueError('Duplicate recommendation')
        rest=row[~np.isin(row,new)]
        combined=np.concatenate((new,rest))
        if len(combined)!=len(row):raise ValueError('Incomplete ranking')
        out[i,combined]=-np.arange(len(row),dtype=float)
    return out


def quality_arrays(ctx):
    out=np.empty((len(ctx.test),len(ctx.q)))
    for i,u in enumerate(ctx.test):
        m=ctx.ada_mastery[int(u)]
        out[i]=[1-abs(.8-np.prod([m[int(k)] for k in np.flatnonzero(q)])) for q in ctx.q]
    return out


def prefix_quality(rank,ada,qbits,hist,ns):
    acc=np.cumsum(ada[rank])
    concept=0;nov=[]
    for i,e in enumerate(rank,1):
        concept |= qbits[int(e)]
        if i in ns:nov.append(novelty(concept,hist))
    return acc[np.array(ns)-1]/np.array(ns),np.array(nov)


def allocate(ctx,scores,rule_k,method,sweeps=5,budget=.01,ns=TOP_NS):
    if method not in ('FQ','FQE'):raise ValueError(method)
    ns=tuple(ns);max_n=max(ns);nusers,nitems=scores.shape
    full=np.argsort(-scores,axis=1,kind='stable');top=full[:,:max_n].copy()
    rules=np.array([ctx.rule(int(u),max_n) for u in ctx.test])
    pools=[np.union1d(row,ctx.rule(int(u),min(rule_k,nitems))).astype(int) for u,row in zip(ctx.test,top)]
    ada=quality_arrays(ctx);qb=[bits(row) for row in ctx.q]
    histories=[sum(1<<int(k) for k in ctx.correct[int(u)]) for u in ctx.test]
    user_ada,user_nov=map(np.array,zip(*(prefix_quality(row,ada[i],qb,histories[i],ns) for i,row in enumerate(top))))
    initial_a=user_ada.mean(0);initial_n=user_nov.mean(0)
    total_a=user_ada.sum(0);total_n=user_nov.sum(0)
    floor_a=(initial_a-budget)*nusers;floor_n=(initial_n-budget)*nusers
    counts=np.array([np.bincount(top[:,:n].ravel(),minlength=nitems) for n in ns])
    ref=np.array([np.bincount(rules[:,:n].ravel(),minlength=nitems) for n in ns])
    rng=np.random.default_rng(20261023);logs=[];accepted=0
    for sweep in range(sweeps):
        changes=0
        for i in rng.permutation(nusers):
            uid=int(ctx.test[i]);row=top[i]
            prefix_counts=np.array([ctx.q[row[:n]].sum(0,dtype=int) for n in ns])
            # Five sweeps cycle across every position; each student gets equal visits.
            for pos in range(sweep%5,max_n,5):
                old=int(row[pos]);cand=pools[i][~np.isin(pools[i],row)]
                cand=cand[ctx.distance[uid,cand]<=ctx.distance[uid,old]+1e-12]
                if not len(cand):continue
                affected=np.flatnonzero(np.array(ns)>pos);lengths=np.array(ns)[affected]
                da=(ada[i,cand,None]-ada[i,old])/lengths[None,:]
                dn=np.empty((len(cand),len(affected)))
                proposed_nov=np.empty_like(dn)
                for j,k in enumerate(affected):
                    before=bits(prefix_counts[k]-ctx.q[old])
                    proposed_nov[:,j]=[novelty(before|qb[int(e)],histories[i]) for e in cand]
                    dn[:,j]=proposed_nov[:,j]-user_nov[i,k]
                valid=np.all(total_a[affected]+da>=floor_a[affected]-1e-10,axis=1)&np.all(total_n[affected]+dn>=floor_n[affected]-1e-10,axis=1)
                delta_ded=np.empty_like(da)
                for j,k in enumerate(affected):
                    residual=counts[k]-ref[k]
                    delta_ded[:,j]=(abs(residual[old]-1)-abs(residual[old])+abs(residual[cand]+1)-abs(residual[cand]))/(2*nusers*ns[k])
                if method=='FQE':
                    gain=-delta_ded.sum(1)/len(ns)
                    valid &= np.all(delta_ded<=1e-12,axis=1)&(gain>1e-12)
                else:
                    gain=ctx.distance[uid,old]-ctx.distance[uid,cand]
                    valid &= gain>1e-12
                if not valid.any():continue
                ix=np.flatnonzero(valid)
                # Deterministic ties: objective, then original model score, then ID.
                chosen=ix[np.lexsort((cand[ix],-scores[i,cand[ix]],-gain[ix]))[0]]
                new=int(cand[chosen]);row[pos]=new
                total_a[affected]+=da[chosen];total_n[affected]+=dn[chosen]
                user_ada[i,affected]+=da[chosen];user_nov[i,affected]=proposed_nov[chosen]
                for k in affected:
                    counts[k,old]-=1;counts[k,new]+=1
                    prefix_counts[k]+=ctx.q[new]-ctx.q[old]
                accepted+=1;changes+=1
        logs.append({'sweep':sweep+1,'accepted':changes,'Ada':(total_a/nusers).tolist(),
                     'NOV':(total_n/nusers).tolist(),'DED':[float(np.abs(counts[k]-ref[k]).sum()/(2*nusers*n)) for k,n in enumerate(ns)]})
        print(f'  {method} K={rule_k} sweep={sweep+1}: {changes} accepted',flush=True)
        if not changes:break
    # Recompute independently; never rely solely on incremental budget bookkeeping.
    final_a,final_n=map(np.array,zip(*(prefix_quality(row,ada[i],qb,histories[i],ns) for i,row in enumerate(top))))
    assert np.allclose(final_a,user_ada) and np.allclose(final_n,user_nov)
    if np.any(final_a.mean(0)<initial_a-budget-1e-10) or np.any(final_n.mean(0)<initial_n-budget-1e-10):
        raise AssertionError('Quality constraint violated')
    return ranking_scores(full,top),{'accepted_swaps':accepted,'sweeps':logs,'rule_K':rule_k,'method':method,
                'per_user_quality_changes':[{'uid':int(u),'delta_Ada':(final_a[i]-prefix_quality(full[i,:max_n],ada[i],qb,histories[i],ns)[0]).tolist(),
                  'delta_NOV':(final_n[i]-prefix_quality(full[i,:max_n],ada[i],qb,histories[i],ns)[1]).tolist()} for i,u in enumerate(ctx.test)]}


def support(ctx,base_scores,k,folder):
    rng=np.random.default_rng(20261023)
    output={'rule_K':k,'candidate_union':'Rule TopK plus E0 Top50 for test; rule TopK plus original rec50 for train',
            'scope':'Rule K fixed before outcomes; no test tuning of K', 'groups':{}}
    for label,users in [('train',ctx.train),('test',ctx.test)]:
        q=ctx.q;rule_sets=[];union_sets=[];records=[]
        base=np.argsort(-base_scores,axis=1,kind='stable')[:,:50] if label=='test' else None
        for i,u in enumerate(users):
            rule=set(ctx.rule(int(u),k).tolist());rule_sets.append(rule)
            initial=set(base[i]) if base is not None else set(ctx.positives[int(u)])
            union=rule|initial;union_sets.append(union)
            records.append({'uid':int(u),'rule_candidates':len(rule),'union_candidates':len(union),
                            'rule_alternatives_outside_baseline50':len(rule-initial),
                            'candidate_KCs':int((q[list(union)].sum(0)>0).sum()),
                            'distance_Kth':float(ctx.distance[int(u),ctx.rule(int(u),k)[-1]])})
        ids=np.array([(i,j) for i in range(len(users)) for j in range(i)],dtype=int)
        if len(ids)>20000:ids=ids[rng.choice(len(ids),20000,replace=False)]
        jc=[len(rule_sets[i]&rule_sets[j])/len(rule_sets[i]|rule_sets[j]) for i,j in ids]
        all_ex=set().union(*union_sets)
        output['groups'][label]={'students':len(users),'records':records,
            'union_item_coverage':len(all_ex)/len(q),'union_KC_count':int((q[list(all_ex)].sum(0)>0).sum()),
            'mean_rule_candidate_Jaccard':float(np.mean(jc)),
            'sampled_student_pairs':len(jc),'unique_rule_candidate_sets':len({frozenset(s) for s in rule_sets}),
            'mean_alternatives':float(np.mean([r['rule_alternatives_outside_baseline50'] for r in records])),
            'min_alternatives':min(r['rule_alternatives_outside_baseline50'] for r in records)}
    write_json(folder/f'candidate_support_K{k}.json',output)


def report():
    lines=['# 教育质量约束下的曝光可行性测试（2026-10-23）','',
           '这是利用测试人群规则曝光与评价历史进行的探索性群体分配，不是训练模型，不是可部署的无测试信息方法，也不是最优解或不可能性的证明。',
           'K=100/200、预算0.01、五轮贪心扫描及非增距离保护均在本轮结果产生前固定，完整报告两档，不据此选正式训练参数。所有N来自同一Top50前缀。',
           'FQ仅优化教育距离；FQE在相同候选池和质量约束下优化真实DED，且每个N的DED不得增加。预算约束针对全体测试学生平均，未强制每位学生均满足0.01；学生级变化另存。',
           'E0原推荐始终在候选池内，保证起点可行；每一步要求替换题教育距离不增加。启发式只做替换，不保证找到所有可能排序。','']
    rows=[]
    for d in DATASETS:
        base_path=RUN/d/'E0/seed2024/eval/metrics.json'
        if not base_path.exists():continue
        base=read_json(base_path)
        lines += [f'## {d}','']
        for k in [100,200]:
            p=OUT/d/f'candidate_support_K{k}.json'
            if not p.exists():continue
            s=read_json(p)
            for pop,g in s['groups'].items():
                lines.append(f"K={k}，{pop}：平均新增可接受替代题 {g['mean_alternatives']:.1f}，最少 {g['min_alternatives']}；规则候选平均 Jaccard={g['mean_rule_candidate_Jaccard']:.4f}；候选并集覆盖题库 {g['union_item_coverage']:.2%}。")
            lines += ['',f'### K={k}：逐N变化（相对E0）','', '|N|方法|ΔAda|ΔNOV|预算通过|DED|ΔDED|覆盖题数|不同题集比例|','|---|---|---:|---:|---|---:|---:|---:|---:|']
            for n in TOP_NS:
                for m in ['FQ','FQE']:
                    p=OUT/d/f'K{k}_{m}/eval/metrics.json'
                    if not p.exists():continue
                    x=read_json(p);a=x['top_n'][str(n)];b=base['top_n'][str(n)]
                    da=a['Ada']-b['Ada'];dn=a['KG4Ex-NOV-Set']-b['KG4Ex-NOV-Set'];ok=da>=-.01-1e-10 and dn>=-.01-1e-10
                    lines.append(f"|{n}|{m}|{da:+.5f}|{dn:+.5f}|{'是' if ok else '否'}|{a['DED']:.4f}|{a['DED']-b['DED']:+.4f}|{round(a['ItemCoverage']*x['candidate_exercises'])}|{a['UniqueSetRatio']:.4f}|")
                    for metric,value in a.items():rows.append([d,k,m,n,metric,value,value-b[metric]])
            lines+=['']
        lines+=['R是同一教育规则参照，其DED=0由定义决定，不能凭此证明真实公平；原R指标在 references/R_metrics.json。','']
    (OUT/'可行性测试报告_20261023.md').write_text('\n'.join(lines),encoding='utf-8')
    with (OUT/'全部指标与变化.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','rule_K','method','N','metric','value','delta_vs_E0']);w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',default=','.join(DATASETS));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.json'
    if protocol.exists() and read_json(protocol)!=PROTOCOL:raise ValueError('Changed feasibility protocol')
    write_json(protocol,PROTOCOL)
    for d in args.datasets.split(','):
        if d not in DATASETS:parser.error(d)
        print(f'{d}: support and allocation feasibility',flush=True)
        ctx,arrays,manifest=prepared(d);folder=OUT/d;folder.mkdir(parents=True,exist_ok=True)
        source=RUN/d/'E0/seed2024/uid_ex_scores.json'
        base=load_scores(source,ctx)
        base_metrics=read_json(source.parent/'eval/metrics.json')
        if sha256(source)!=base_metrics['provenance']['scores_sha256']:raise ValueError('Baseline scores changed')
        write_json(folder/'references/E0_metrics.json',base_metrics)
        write_json(folder/'references/R_metrics.json',read_json(RUN/d/'R/seed2024/eval/metrics.json'))
        for k in PROTOCOL['candidate_rule_top_k']:
            support(ctx,base,k,folder)
            for method in ['FQ','FQE']:
                target=folder/f'K{k}_{method}';target.mkdir(parents=True,exist_ok=True)
                scorefile=target/'uid_ex_scores.json'
                if not (target/'allocation_audit.json').exists():
                    scores,audit=allocate(ctx,base,k,method)
                    write_score_matrix(scorefile,ctx,scores)
                    write_json(target/'allocation_audit.json',audit)
                result=evaluate_file(ctx,arrays,manifest,scorefile,target/'eval',baseline=source)
                for n,a in result['top_n'].items():
                    b=base_metrics['top_n'][n]
                    if a['Ada']<b['Ada']-.01-1e-10 or a['KG4Ex-NOV-Set']<b['KG4Ex-NOV-Set']-.01-1e-10:
                        raise AssertionError(f'Independent evaluator rejected quality budget: {d} {k} {method} N={n}')
                    if method=='FQE' and a['DED']>b['DED']+1e-12:raise AssertionError('DED increased')
                print(f'{d} K={k} {method}: evaluated and quality checked',flush=True)
        write_json(folder/'complete.json',{'script_sha256':sha256(Path(__file__)),'dataset_hash':ctx.fingerprint,
                   'baseline_scores_sha256':sha256(source),'status':'completed','training_updates':0})
        report()
    print(OUT)


if __name__=='__main__':main()
