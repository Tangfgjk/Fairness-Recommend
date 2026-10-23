"""One-command controlled Q/QE pilot; default plan, --stage run explicitly starts training."""
import argparse
from pathlib import Path
import sys
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT,DATASETS,read_json,write_json,sha256,digest
from maed.io import source_hash,lock_json


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('plan','prepare','run','report'),default='plan')
    p.add_argument('--datasets',default=','.join(DATASETS))
    p.add_argument('--methods',default='M3,M4')
    p.add_argument('--seed',type=int,choices=(2024,),default=2024)
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--exposure-weight',type=float,default=.01)
    p.add_argument('--users-per-step',type=int,default=16)
    p.add_argument('--cuda',choices=('true','false'),default='true')
    p.add_argument('--run-id',default='maed_controlled_boundary_seed2024_20261023')
    p.add_argument('--baseline-run',type=Path,default=ROOT/'runs/merit_frequency_20261022')
    p.add_argument('--reference-diagnostics',type=Path,default=ROOT/'runs/论文格式文档统计/maed_m0_m2_20261023')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker-method',choices=('Q-fixed','M3','M4'),help=argparse.SUPPRESS)
    return p.parse_args()


def settings(args):
    import math
    if any(not math.isfinite(v) or v<0 for v in (args.beta,args.exposure_weight)) or args.users_per_step<1:
        raise ValueError('Invalid beta/weight/users-per-step')
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('run-id must be a single alphanumeric/_/- directory')
    return {'version':'maed_controlled_boundary_v1_20261023','source_hash':source_hash(),
            'seed':args.seed,'beta':args.beta,'M4_exposure_weight':args.exposure_weight,
            'users_per_step':args.users_per_step,'cuda':args.cuda,'distance_clip':3.,
            'baseline_run':str(args.baseline_run.resolve()),'reference_diagnostics':str(args.reference_diagnostics.resolve()),
            'quality_budgets':{'Ada':.01,'KG4Ex-NOV-Set':.03},'Ep_threshold':.5,
            'temperature_policy':'frozen training-wide per-N boundary calibration of same-round E0+Q',
            'checkpoint_policy':'minimum supervised BCE; last epoch reported separately',
            'optimizer_policy':'same original Adam/BCE/negative sampling/epochs/lr/batch size; no clipping or extra optimizer steps',
            'protocol_scope':'exploratory observed-history endpoint snapshot, inherited shared-preprocessing limitations',
            'test_hyperparameter_selection':False,
            'coefficient_status':f'lambdaE={args.exposure_weight} exploratory value fixed before this pilot; not validated optimal'}


def worker(dataset,method,args,root,protocol):
    import numpy as np
    from maed.state import load_state
    from maed.training import prepare_assets,train,export_scores,scale_from_assets
    from maed.diagnostics import checked_test_logits
    from maed.scoring import fuse_logits
    from debias.prepare import prepared
    from debias.evaluation import evaluate_file,write_score_matrix
    state=load_state(dataset)
    assets=prepare_assets(state,args.reference_diagnostics,args.baseline_run,root/'assets'/dataset,args.beta)
    folder=root/dataset/method/f'seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'stage_complete.json'
    if complete.exists():
        marker=read_json(complete)
        if marker['protocol_hash']!=digest(protocol):raise ValueError('Completed stage protocol differs')
        for rel,h in marker['outputs'].items():
            if sha256(folder/rel)!=h:raise ValueError('Completed stage artifact changed')
        print(f'Skip completed {dataset}/{method}',flush=True);return
    baseline=args.baseline_run/dataset/'E0'/f'seed{args.seed}'
    # Read evaluation-only context separately; it never enters train() or ExposureObjective.
    if method=='Q-fixed':
        z,p=checked_test_logits(state,baseline)
        values=p if args.beta==0 else fuse_logits(z,state.distance[state.test],scale_from_assets(assets),args.beta)
        ctx,arrays,manifest=prepared(dataset)
        score=folder/'uid_ex_scores.json'
        write_score_matrix(score,ctx,values)
        evaluate_file(ctx,arrays,manifest,score,folder/'eval',baseline=baseline/'uid_ex_scores.json')
    else:
        train(state,assets,folder,method,args.seed,protocol['source_hash'],args.cuda,args.exposure_weight,args.users_per_step)
        ctx,arrays,manifest=prepared(dataset)
        for label in ('best','last'):
            score=folder/label/'uid_ex_scores.json';meta=folder/label/'score_provenance.json'
            checkpoint=folder/f'{label}.pt'
            if meta.exists():
                saved=read_json(meta)
                if saved['checkpoint_sha256']!=sha256(checkpoint) or saved['score_sha256']!=sha256(score) or saved['logits_sha256']!=sha256(score.with_suffix('.logits.npy')):
                    raise ValueError('Saved export/checkpoint provenance changed')
            else:
                write_json(meta,export_scores(state,checkpoint,score,args.cuda))
            evaluate_file(ctx,arrays,manifest,score,folder/label/'eval',baseline=baseline/'uid_ex_scores.json')
    outputs={p.relative_to(folder).as_posix():sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('stage.log','stage_complete.json')}
    write_json(complete,{'protocol_hash':digest(protocol),'outputs':outputs,'seed':args.seed,'method':method})


def main():
    args=parse_args()
    datasets=args.datasets.split(',');methods=args.methods.split(',')
    if not set(datasets)<=set(DATASETS) or not set(methods)<={'M3','M4'} or any(len(v)!=len(set(v)) for v in (datasets,methods)):
        raise ValueError('Unsupported or duplicate datasets/methods')
    protocol=settings(args)
    jobs=[(d,m) for d in datasets for m in ('Q-fixed',*methods)]
    root=ROOT/'runs'/args.run_id
    if args.stage=='plan':
        for d,m in jobs:print(f'{d}/seed{args.seed}/{m}: '+('frozen score fusion + evaluation' if m=='Q-fixed' else '25 original epochs + best/last evaluation'))
        print(f'Training jobs: {len(datasets)*len(methods)}; Q-fixed stages: {len(datasets)}')
        print(f'M4 lambdaE={args.exposure_weight}; beta={args.beta}; results={root}')
        return
    lock_json(root/'protocol.json',protocol)
    if args.stage=='report':
        from maed.training_report import report
        print(report(root,args.baseline_run));return
    if args.worker:
        if len(datasets)!=1 or args.worker_method is None:raise ValueError('Worker needs exactly one dataset/method')
        worker(datasets[0],args.worker_method,args,root,protocol);return
    # Fail before the first long training job if any selected dataset/reference is missing.
    from maed.state import load_state
    from maed.training import prepare_assets
    from debias.prepare import prepared
    for dataset in datasets:
        state=load_state(dataset)
        prepare_assets(state,args.reference_diagnostics,args.baseline_run,root/'assets'/dataset,args.beta)
        prepared(dataset)
        print(f'Preflight passed: {dataset}',flush=True)
    if args.stage=='prepare':
        print(f'All selected inputs and calibration assets verified; no training. {root}',flush=True)
        return
    from scripts.run_merit_debias_suite import run_stage
    for i,(dataset,method) in enumerate(jobs,1):
        print(f'[{i}/{len(jobs)}] {dataset}/{method}',flush=True)
        write_json(root/'status.json',{'state':'running','stage':i,'total_stages':len(jobs),'dataset':dataset,'method':method})
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','run','--worker','--worker-method',method,
                 '--datasets',dataset,'--methods',args.methods,'--seed',str(args.seed),'--beta',str(args.beta),
                 '--exposure-weight',str(args.exposure_weight),'--users-per-step',str(args.users_per_step),
                 '--cuda',args.cuda,'--run-id',args.run_id,'--baseline-run',str(args.baseline_run),
                 '--reference-diagnostics',str(args.reference_diagnostics)]
        try:run_stage(command,root/dataset/method/f'seed{args.seed}'/'stage.log')
        except BaseException:
            write_json(root/'status.json',{'state':'interrupted','stage':i,'total_stages':len(jobs),'dataset':dataset,'method':method})
            raise
    from maed.training_report import report
    location=report(root,args.baseline_run)
    write_json(root/'status.json',{'state':'complete','stages':len(jobs),'training_jobs':len(datasets)*len(methods),'report':str(location)})
    print(f'Completed; tables: {location}',flush=True)


if __name__=='__main__':main()
