"""Controlled nine-run MAED-v2 suite. Default plan; original training settings fixed."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from debias.io import DATASETS,read_json,write_json,sha256,digest
from maed.io import lock_json


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('plan','prepare','preflight','run','report'),default='plan')
    p.add_argument('--datasets',default=','.join(DATASETS))
    p.add_argument('--methods',default='C0,C3,C4')
    p.add_argument('--run-id',default='maed_v2_semantic_compgcn_seed2024_20261024_controlled')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--p0',type=Path,default=ROOT/'artifacts/maed_v2/P0_full_acceptance_20261024')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker-method',choices=('C0','C3','C4'),help=argparse.SUPPRESS)
    return p.parse_args()


def configuration(args):
    from maed_v2.training import source_hash,VERSION
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('Invalid run-id')
    return {'version':VERSION,'source_hash':source_hash(),'seed':2024,'epochs':25,'batch_size':1024,
            'negative_ratio':5,'learning_rate':.001,'beta':.5,'exposure_weight':.01,'users_per_step':16,
            'device':args.device,'p0':str(args.p0.resolve()),'propagation_layers':1,'operation':'mult',
            'graph_dropout':0.,'residual':.5,'recompute':True,
            'formal_inference':'B full reference, one test student at a time',
            'same_checkpoint_control':'A required in primary results',
            'quality_budgets':{'Ada':.01,'KG4Ex-NOV-Set':.03},'Ep_threshold':.5,
            'training_selection':'minimum supervised BCE, last diagnostics separate',
            'temperatures':'trained best C0 + Q on all training students, frozen before C3/C4',
            'history_policy':'inherited observed-history endpoint snapshot, not strict future prediction'}


def base_assets(dataset,args,root):
    import torch
    from maed.state import load_state
    from maed.scoring import EducationScale
    from maed_v2.graph_protocol import GraphProtocol
    from debias.io import model_modules
    state=load_state(dataset)
    p0=read_json(args.p0/dataset/'P0_result.json')
    chosen=p0['training_profile']['chosen_chunk']
    candidate=next(r for r in p0['training_profile']['rows'] if r['chunk_size']==chosen)
    if p0['status']!='passed' or candidate['status']!='passed':raise ValueError('P0 chunk not accepted')
    cfg_path=ROOT/'runs/merit_frequency_20261022'/dataset/'E0/seed2024/config.json'
    cfg=read_json(cfg_path)
    if any(cfg[k]!=v for k,v in {'epochs':25,'bs':1024,'negative_ratio':5,'learning_rate':.001,'embedding_dim':200,'ablation':'feature_only'}.items()):
        raise ValueError('Original fixed training settings differ')
    if cfg.get('include_test_triples',False):raise ValueError('No test supervision allowed')
    bundle=model_modules().load_semantic_feature_bundle(state.graph,None,device=torch.device('cpu'))
    protocol=GraphProtocol(state,bundle.entity2id,bundle.relation2id)
    if protocol.base.graph_id!=p0['graph_base_id']:raise ValueError('P0/input graph changed; repeat P0')
    scale=EducationScale.fit(state).to_dict();scale['train_users']=list(scale['train_users'])
    assets={'reference_config':cfg,'reference_config_sha256':sha256(cfg_path),
            'state_hash':state.provenance['state_hash'],'base_graph_id':protocol.base.graph_id,
            'graph_audit':protocol.manifest,'scale':scale,'beta':.5,'top_ns':list(range(5,51,5)),
            'chunk_size':chosen,'p0_result_sha256':sha256(args.p0/dataset/'P0_result.json')}
    lock_json(root/'assets'/dataset/'C0_assets.json',assets)
    return state,assets


def worker(dataset,method,args,root,protocol):
    import numpy as np
    from maed_v2.training import train,calibrate_assets,export_scores
    from debias.prepare import prepared
    from debias.evaluation import evaluate_file
    state,assets=base_assets(dataset,args,root)
    folder=root/dataset/method/'seed2024';folder.mkdir(parents=True,exist_ok=True)
    marker_path=folder/'stage_complete.json'
    if marker_path.exists():
        marker=read_json(marker_path)
        if marker['protocol_hash']!=digest(protocol):raise ValueError('Completed protocol changed')
        for rel,h in marker['outputs'].items():
            if sha256(folder/rel)!=h:raise ValueError('Completed stage output changed')
        print(f'Skip completed {dataset}/{method}',flush=True);return
    if method!='C0':
        c0=root/dataset/'C0/seed2024'
        completion=read_json(c0/'training_complete.json')
        if completion['checkpoints']['best.pt']!=sha256(c0/'best.pt'):raise ValueError('C0 reference checkpoint changed')
        assets=calibrate_assets(state,c0/'best.pt',root/'assets'/dataset/'C0_reference',args.device)
    train(state,assets,folder,method,2024,protocol['source_hash'],args.device)
    # Evaluation context is deliberately loaded only AFTER all training updates.
    ctx,arrays,manifest=prepared(dataset)
    baseline=ROOT/'runs/merit_frequency_20261022'/dataset/'E0/seed2024/uid_ex_scores.json'
    for checkpoint_label in ('best','last'):
        for inference in ('B','A'):
            target=folder/checkpoint_label/f'inference_{inference}'
            score=target/'uid_ex_scores.json';meta=target/'score_provenance.json'
            checkpoint=folder/f'{checkpoint_label}.pt'
            # Identical best/last states may reuse exports, preserving checkpoint identity.
            if not meta.exists():
                write_json(meta,export_scores(state,checkpoint,score,args.device,inference))
            saved=read_json(meta)
            if (saved['checkpoint_sha256']!=sha256(checkpoint) or saved['score_sha256']!=sha256(score)
                or saved['logits_sha256']!=sha256(score.with_suffix('.logits.npy'))):
                raise ValueError('Export provenance changed')
            evaluate_file(ctx,arrays,manifest,score,target/'eval',baseline=baseline)
    if method=='C0':
        from maed_v2.audit import audit_checkpoint
        audit_checkpoint(state,folder/'best.pt',folder/'diagnostics/best_C0',ctx,arrays,args.device)
    outputs={p.relative_to(folder).as_posix():sha256(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name not in ('stage.log','stage_complete.json')}
    write_json(marker_path,{'protocol_hash':digest(protocol),'method':method,'seed':2024,'outputs':outputs})


def main():
    args=arguments();datasets=args.datasets.split(',');methods=args.methods.split(',')
    if (not datasets or not methods or len(set(datasets))!=len(datasets) or len(set(methods))!=len(methods)
        or not set(datasets)<=set(DATASETS) or not set(methods)<={'C0','C3','C4'}):raise ValueError('Invalid jobs')
    import torch
    torch.set_num_threads(2)
    protocol=configuration(args);root=ROOT/'runs'/args.run_id
    if args.stage=='plan':
        for d in datasets:
            for m in methods:print(f'{d}/{m}/seed2024: 25 epochs; best/last A+B evaluation')
        print(f'{len(datasets)*len(methods)} training jobs; outputs={root}');return
    lock_json(root/'protocol.json',protocol)
    if args.stage=='report':
        from maed_v2.report import report
        print(report(root));return
    if args.worker:
        if len(datasets)!=1 or args.worker_method is None:raise ValueError('Worker requires one dataset/method')
        worker(datasets[0],args.worker_method,args,root,protocol);return
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    for dataset in datasets:
        base_assets(dataset,args,root)
        print(f'Input/P0 checks passed: {dataset}',flush=True)
    if args.stage=='prepare':return
    if args.stage=='preflight':
        from maed_v2.preflight import run_preflight
        print(run_preflight(root,args,protocol));return
    gate=read_json(root/'engineering_gate.json')
    if gate['source_hash']!=protocol['source_hash'] or gate['status']!='passed':raise ValueError('Current-source engineering gate must pass before training')
    from scripts.run_merit_debias_suite import run_stage
    jobs=[(d,m) for d in datasets for m in ('C0','C3','C4') if m in methods]
    for i,(dataset,method) in enumerate(jobs,1):
        write_json(root/'status.json',{'state':'running','stage':i,'total_stages':len(jobs),'dataset':dataset,'method':method})
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','run','--worker',
                 '--worker-method',method,'--datasets',dataset,'--methods',args.methods,'--run-id',args.run_id,
                 '--device',args.device,'--p0',str(args.p0)]
        print(f'[{i}/{len(jobs)}] {dataset}/{method}',flush=True)
        try:run_stage(command,root/dataset/method/'seed2024/stage.log')
        except BaseException:
            write_json(root/'status.json',{'state':'interrupted','stage':i,'total_stages':len(jobs),'dataset':dataset,'method':method})
            raise
    from maed_v2.report import report
    location=report(root)
    write_json(root/'status.json',{'state':'complete','training_jobs':len(jobs),'report':str(location)})
    print(f'All controlled stages complete: {location}',flush=True)


if __name__=='__main__':main()
