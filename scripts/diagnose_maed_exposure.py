"""M2 follow-up: training-only mechanism diagnosis and exploratory boundary calibration."""
import argparse
from pathlib import Path
import sys
import gc
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT,DATASETS,TOP_NS,read_json,write_json,sha256,digest
from maed.io import source_hash,input_hashes,lock_json,write_csv
from maed.state import load_state
from maed.scoring import EducationScale,fuse_logits
from maed.boundary_exposure import calibrate_temperatures
from maed.exposure import rule_targets
from maed.m2_analysis import score_diagnostics,trajectory,model_diagnostics


def protocol():
    return {'version':'maed_m2_mechanism_boundary_v1_exploratory','beta':.5,'distance_clip':3.,
            'top_ns':list(TOP_NS),'training_cohort_seeds':[20261023,20261024,20261025],
            'matrix_users':48,'model_users':16,'model_user_sampling':'uniform subset of matrix users, seed+17',
            'trajectory_steps':16,'fixed_RMS_step_fraction':.02,
            'arms':['IQR_fixed','boundary_fixed','boundary_training_guard'],
            'boundary_temperature':'training-wide per-N median positive N/N+1 gap; multiplier 1; floor=max(IQR*1e-4,1e-6)',
            'training_guard':'soft mean strictly decreases; hard mean and hard@10 nonincrease; up to 10 halvings',
            'model_relative_steps':[1e-6,1e-5,1e-4,1e-3], 'primary_model_relative_step':1e-4,
            'mechanism_support':'joint soft mean/hard mean/hard@10 decrease at terminal step, Top10 changes; all cohorts reported',
            'thresholds':{'Ada':.01,'KG4Ex-NOV-Set':.03},
            'test_students_in_objective':False,'optimizer_updates':0,'formal_training':False,
            'scope':'exploratory follow-up after seeing v0; no retrospective reclassification or test-based parameter selection'}


def verified_cache(state,source,baseline):
    root=source/state.dataset
    contract=read_json(root/'source_contract.json')
    if contract['state']['state_hash']!=state.provenance['state_hash']:
        raise ValueError('Educational state differs from historical cache')
    if contract['all_input_files']!=input_hashes(state):
        raise ValueError('Historical input files changed')
    checkpoint=baseline/state.dataset/'E0/seed2024/best.pt'
    if sha256(checkpoint)!=contract['E0']['best.pt']:
        raise ValueError('Historical E0 checkpoint changed')
    cache=root/'M2'
    ck=read_json(cache/'train_logit_contract.json')
    if ck['uids']!=state.train.tolist() or ck['checkpoint']!=sha256(checkpoint):
        raise ValueError('Cached logit UID/checkpoint mismatch')
    path=cache/'train_logits.npy'
    marker=read_json(cache/'train_logits_complete.json')
    if marker['sha256']!=sha256(path):
        raise ValueError('Cached logit matrix changed')
    z=np.load(path).astype(float)
    if z.shape!=(len(state.train),len(state.q)) or not np.isfinite(z).all():
        raise ValueError('Bad cached matrix')
    sources={str(path):sha256(path),str(checkpoint):sha256(checkpoint),
             str(cache/'train_logit_contract.json'):sha256(cache/'train_logit_contract.json'),
             str(root/'source_contract.json'):sha256(root/'source_contract.json')}
    return z,sources


def run_dataset(state,train_z,baseline,output,settings,device):
    scale=EducationScale.fit(state,settings['distance_clip'])
    fused=fuse_logits(train_z,state.distance[state.train],scale,settings['beta'])
    calibration=calibrate_temperatures(fused,TOP_NS)
    write_json(output/'temperature_calibration.json',calibration)
    temps={'IQR': [calibration['global_IQR']]*len(TOP_NS),
           'boundary':[r['temperature'] for r in calibration['per_N']]}
    boundaries=[];items=[];trace_rows=[];summary_rows=[];model_rows=[];agreements=[];details=[]
    for seed in settings['training_cohort_seeds']:
        selected=np.sort(np.random.default_rng(seed).choice(len(state.train),settings['matrix_users'],replace=False))
        users=state.train[selected]
        target=rule_targets(state.training_distance(users),TOP_NS)
        initial=torch.tensor(fused[selected],device=device,dtype=torch.float64)
        item_diagnostics={}
        for name,temperature in temps.items():
            diagnostic,item_rows=score_diagnostics(initial,target,temperature,global_scale=calibration['global_IQR'])
            item_diagnostics[name]=diagnostic
            boundaries.extend({'dataset':state.dataset,'cohort_seed':seed,'proxy':name,**r} for r in diagnostic['boundary_and_gradient'])
            items.extend({'cohort_seed':seed,'proxy':name,**r} for r in item_rows)
        trajectories={}
        for arm in settings['arms']:
            proxy='IQR' if arm=='IQR_fixed' else 'boundary'
            value=trajectory(initial,target,temps[proxy],calibration['global_IQR'],
                             settings['trajectory_steps'],guarded=arm.endswith('training_guard'))
            trajectories[arm]=value
            before=value['trace'][0];end=value['trace'][-1]
            summary_rows.append({'dataset':state.dataset,'cohort_seed':seed,'arm':arm,
                'soft_mean_before':before['soft_mean_TV'],'soft_mean_after':end['soft_mean_TV'],
                'hard_mean_before':float(np.mean(before['hard_TV'])),'hard_mean_after':float(np.mean(end['hard_TV'])),
                'hard10_before':before['hard_TV'][1],'hard10_after':end['hard_TV'][1],
                'steps_completed':end['step'],'terminal_joint_improvement':value['terminal_joint_improvement']})
            for step in value['trace']:
                for j,n in enumerate(TOP_NS):
                    trace_rows.append({'cohort_seed':seed,'arm':arm,'step':step['step'],'N':n,
                        'soft_TV':step['soft_TV'][j],'hard_TV':step['hard_TV'][j],
                        'changed_users':step['changed_users'][j],'entered_items':step['entered_items'][j],
                        'backtracks':step['backtracks']})
            agreements.extend({'cohort_seed':seed,'arm':arm,**r} for r in value['direction_agreement'])
            print(f'{state.dataset} cohort={seed} {arm}: hard@10 {before["hard_TV"][1]:.4f}->{end["hard_TV"][1]:.4f}; joint={value["terminal_joint_improvement"]}',flush=True)
        probe_users=np.sort(np.random.default_rng(seed+17).choice(users,settings['model_users'],replace=False))
        model_details={}
        for proxy,temperature in temps.items():
            probe=model_diagnostics(state,baseline/state.dataset/'E0/seed2024',probe_users,scale,
                        settings['beta'],temperature,device,settings['model_relative_steps'])
            model_details[proxy]=probe
            for row in probe['probes']:
                model_rows.append({'dataset':state.dataset,'cohort_seed':seed,'proxy':proxy,
                    'relative_parameter_step':row['relative_parameter_step'],
                    'soft_mean_before':probe['initial']['soft_mean_TV'],'soft_mean_after':row['soft_mean_TV'],
                    'hard_mean_before':float(np.mean(probe['initial']['hard_TV'])),
                    'hard_mean_after':float(np.mean(row['hard_TV'])),
                    'hard10_before':probe['initial']['hard_TV'][1],'hard10_after':row['hard_TV'][1],
                    'shared_item_score_change_fraction':row['score_change_shared_item_fraction'],
                    'terminal_joint_improvement':row['terminal_joint_improvement']})
        details.append({'cohort_seed':seed,'users':users.tolist(),'score_diagnostics':item_diagnostics,
                        'trajectories':trajectories,'model_diagnostics':model_details})
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    write_csv(output/'boundary_saturation_and_gradient.csv',boundaries)
    write_csv(output/'item_gradient_directions.csv',items)
    write_csv(output/'trajectory_summary.csv',summary_rows)
    write_csv(output/'trajectory_all_N.csv',trace_rows)
    write_csv(output/'soft_hard_direction_agreement.csv',agreements)
    write_csv(output/'model_direction_summary.csv',model_rows)
    write_json(output/'mechanism_diagnostics.json',{'dataset':state.dataset,'cohorts':details,
                'training_users_only':True,'official_training_updates':0,'v0_verdict_unchanged':True})
    return summary_rows,model_rows,boundaries


def report(output,summaries,models,boundaries,settings):
    lines=['# M2 曝光代理机制诊断与边界校准（2026-10-23）','',
           '本次为看过v0结果后的独立探索性跟进。所有标定、目标、抽样和方向探针仅使用训练学生；不改写旧M2结论。',
           'IQR_fixed为原主温度和固定步长；boundary_fixed只改变温度；boundary_training_guard在边界温度上另加训练侧回溯约束。',
           '训练侧回溯使用真实硬曝光，不代表可微损失自身给出了保证；三个诊断臂均不是已训练推荐模型。','',
           '## 1. Top-10软权重与实际列表的对应','',
           '|数据集|代理|三个训练批次Top10平均承载软权重比例|共享题目梯度占比均值|','|---|---|---:|---:|']
    for ds in DATASETS:
        for proxy in ('IQR','boundary'):
            rs=[r for r in boundaries if r['dataset']==ds and r['proxy']==proxy and r['N']==10]
            lines.append(f'|{ds}|{proxy}|{np.mean([r["actual_TopN_soft_mass_fraction"] for r in rs]):.4f}|{np.mean([r["shared_item_gradient_fraction"] for r in rs]):.4f}|')
    lines+=['','## 2. 自由评分矩阵终点联合改善（每组3个训练批次）','',
            '|数据集|原IQR固定步长|边界温度固定步长|边界温度+训练侧回溯|','|---|---:|---:|---:|']
    for ds in DATASETS:
        counts=[sum(r['terminal_joint_improvement'] for r in summaries if r['dataset']==ds and r['arm']==a) for a in settings['arms']]
        lines.append('|'+ds+'|'+'|'.join(f'{n}/3' for n in counts)+'|')
    lines+=['','## 3. 真实模型参数方向（主相对扰动1e-4，每组3个训练批次）','',
            '|数据集|IQR温度联合改善|边界温度联合改善|','|---|---:|---:|']
    for ds in DATASETS:
        counts=[sum(r['terminal_joint_improvement'] for r in models if r['dataset']==ds and r['proxy']==p and
                    r['relative_parameter_step']==settings['primary_model_relative_step']) for p in ('IQR','boundary')]
        lines.append(f'|{ds}|{counts[0]}/3|{counts[1]}/3|')
    lines+=['','## 4. 限制与后续','',
            '- 上述批次可能重叠，不是三个独立训练种子；不报告跨训练种子稳定性结论。',
            '- 完整跨N评分梯度/模型参数梯度余弦矩阵保存在各数据集mechanism_diagnostics.json。',
            '- 即使边界校准改善，仍须完成质量受限的Q/QE受控训练对照，才能判断最终推荐效果。',
            '- 本次没有使用测试人群来选择温度、回溯步长或模型；没有更新保存的模型参数。',
            '- 新质量门槛为Ada下降≤0.01、NOV绝对下降≤0.03；本项诊断不重新宣称旧M2通过。','']
    (output/'M2机制诊断与边界校准结论_20261023.md').write_text('\n'.join(lines),encoding='utf-8')
    write_csv(output/'all_trajectory_summary.csv',summaries)
    write_csv(output/'all_model_direction_summary.csv',models)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/论文格式文档统计/maed_m0_m2_20261023')
    parser.add_argument('--baseline-run',type=Path,default=ROOT/'runs/merit_frequency_20261022')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/论文格式文档统计/maed_m2_mechanism_20261023')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();settings=protocol()
    if args.plan:
        import json
        print(json.dumps(settings,indent=2,ensure_ascii=False));return
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError('Use a separate output directory; preserve v0 results')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    execution={'settings':settings,'source':str(args.source.resolve()),'baseline':str(args.baseline_run.resolve()),
               'code_hash':digest({'maed':source_hash(),'entry':sha256(Path(__file__))}),'device':args.device}
    lock_json(args.output/'protocol.json',execution)
    summaries=[];models=[];boundaries=[]
    for ds in DATASETS:
        state=load_state(ds)
        z,sources=verified_cache(state,args.source,args.baseline_run)
        folder=args.output/ds
        lock_json(folder/'source_contract.json',{'sources':sources,'state':state.provenance})
        marker=folder/'complete.json'
        if marker.exists():
            for rel,h in read_json(marker)['outputs'].items():
                if sha256(folder/rel)!=h:raise ValueError('Completed diagnostics changed')
            import csv
            # Rebuild root report using typed rows serialized with the completion marker.
            saved=read_json(marker)
            a,b,c=saved['summary_rows'],saved['model_rows'],saved['boundary_rows']
        else:
            a,b,c=run_dataset(state,z,args.baseline_run,folder,settings,args.device)
            if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Original artifacts changed')
            write_json(marker,{'summary_rows':a,'model_rows':b,'boundary_rows':c,'optimizer_updates':0,
                'outputs':{p.name:sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='complete.json'}})
        summaries.extend(a);models.extend(b);boundaries.extend(c)
    report(args.output,summaries,models,boundaries,settings)
    print(args.output,flush=True)


if __name__=='__main__':main()
