"""Controlled M3/M4 training; original BCE/optimizer/sampler, explicit auxiliary path."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from debias.io import read_json,write_json,digest,sha256,model_modules,TOP_NS
from debias.training import build_model,rng_state,restore_rng,atomic_checkpoint
from debias.scoring import fixed_statistics
from .io import lock_json
from .scoring import EducationScale,score_rec,fuse_logits
from .exposure import rule_targets
from .boundary_exposure import calibrated_exposure_loss,calibrate_temperatures
from .training_monitor import TrainingMonitor,MonitoredBatches,gradient_balance,initialization_hash,MONITOR_POLICY

TRAINING_VERSION='maed_controlled_training_boundary_monitored_v2_20261023'


def prepare_assets(state,reference_root,baseline_root,output,beta=.5,clip=3.):
    from scripts.diagnose_maed_exposure import verified_cache
    z,sources=verified_cache(state,reference_root,baseline_root)
    scale=EducationScale.fit(state,clip)
    scaled=scale.to_dict();scaled['train_users']=list(scale.train_users)
    fused=fuse_logits(z,state.distance[state.train],scale,beta)
    calibration=calibrate_temperatures(fused,TOP_NS)
    source=baseline_root/state.dataset/'E0/seed2024'
    config=read_json(source/'config.json')
    if config['ablation']!='feature_only' or config.get('include_test_triples',False):
        raise ValueError('Only the original feature_only training-graph protocol is supported')
    sources[str(source/'config.json')]=sha256(source/'config.json')
    data={'training_version':TRAINING_VERSION,'dataset':state.dataset,'state_hash':state.provenance['state_hash'],
          'scale':scaled,'beta':beta,'temperatures':[r['temperature'] for r in calibration['per_N']],
          'top_ns':list(TOP_NS),'temperature_calibration':calibration,'reference_config':config,
          'source_files':sources,'target_policy':'per-N rule exposure of same uniformly sampled training cohort',
          'protocol_scope':'inherited observed-history endpoint snapshot; Eedi shared forgetting calibration is transductive',
          'calibration_frozen_during_training':True}
    lock_json(Path(output)/'assets.json',data)
    return data


def scale_from_assets(assets):
    data=dict(assets['scale']);data['train_users']=tuple(data['train_users'])
    return EducationScale(**data)


class QScorer:
    """One rec-only fusion definition, shared by supervised and auxiliary scoring."""
    def __init__(self,state,bundle,scale,beta,device):
        if not np.isfinite(beta) or beta<0:raise ValueError('Invalid beta')
        self.beta=beta;self.scale=scale;self.rec=bundle.relation2id['rec'];self.bundle=bundle
        self.distance=torch.as_tensor(state.distance,dtype=torch.float32,device=device) if beta else None
        size=max(bundle.entity2id.values())+1
        self.user_index=torch.full((size,),-1,dtype=torch.long,device=device)
        self.item_index=torch.full((size,),-1,dtype=torch.long,device=device)
        for uid in range(len(state.distance)):
            key=f'uid{uid}'
            if key in bundle.entity2id:self.user_index[bundle.entity2id[key]]=uid
        for e in range(len(state.q)):
            self.item_index[bundle.entity2id[f'ex{e}']]=e

    def fuse_triples(self,logits,h,r,t):
        if self.beta==0:return logits
        mask=r==self.rec
        if not mask.any():return logits
        u=self.user_index[h[mask]];e=self.item_index[t[mask]]
        if (u<0).any() or (e<0).any():raise ValueError('rec must be uid -> exercise')
        offset=torch.zeros_like(logits)
        offset[mask]=self.beta*self.scale.transform(self.distance[u,e].to(logits.dtype))
        return logits+offset

    def probabilities(self,model,h,r,t):
        if self.beta==0:
            return model.score_triples(h,r,t)
        return torch.sigmoid(self.fuse_triples(model.score_logits(h,r,t),h,r,t))

    def rec_logits(self,model,uids,nitems):
        z=score_rec(model,self.bundle,uids,nitems)
        if self.beta==0:return z
        d=self.distance[torch.as_tensor(uids,dtype=torch.long,device=z.device)].to(z.dtype)
        return fuse_logits(z,d,self.scale,self.beta)


class ExposureObjective:
    def __init__(self,state,scorer,temperatures,ns,seed,users_per_step=16):
        if users_per_step<1:raise ValueError('users_per_step must be positive')
        self.state=state;self.scorer=scorer;self.temperatures=tuple(temperatures);self.ns=tuple(ns)
        self.rng=np.random.default_rng(seed+790003)
        self.users_per_step=users_per_step
        # Check all sources are legitimate training states before first optimizer update.
        state.training_distance(state.train)

    def state_dict(self):return self.rng.bit_generator.state
    def load_state_dict(self,value):self.rng.bit_generator.state=value

    def __call__(self,model):
        selected=self.rng.choice(self.state.train,min(len(self.state.train),self.users_per_step),replace=False)
        target=rule_targets(self.state.training_distance(selected),self.ns)
        with fixed_statistics(model),torch.backends.cudnn.flags(enabled=False):
            z=self.scorer.rec_logits(model,selected,len(self.state.q))
            loss,per,_=calibrated_exposure_loss(z,target,self.ns,self.temperatures)
        return loss,{'users':len(selected),'per_N_soft_TV':per.detach().cpu().tolist()}


def joint_loss(model,bce,objective,weight,has_rec):
    if not np.isfinite(weight) or weight<0:raise ValueError('Invalid exposure weight')
    if weight==0 or not has_rec:return bce,None,{}
    exposure,stats=objective(model)
    return bce+weight*exposure,exposure,stats


def train(state,assets,output,method,seed,execution_hash,cuda='true',exposure_weight=.01,users_per_step=16):
    if method not in ('M3','M4'):raise ValueError('Expected M3 or M4')
    if not np.isfinite(exposure_weight) or exposure_weight<0:raise ValueError('Invalid exposure weight')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original=model_modules()
    cfg=dict(assets['reference_config'])
    cfg.update(data_path=str(state.graph),save_path=str(output),feature_dir=None,seed=seed,cuda=cuda)
    effective=exposure_weight if method=='M4' else 0.
    contract={'training_version':TRAINING_VERSION,'method':method,'config':cfg,'assets_hash':digest(assets),
              'source_hash':execution_hash,'beta':assets['beta'],'effective_exposure_weight':effective,
              'users_per_step':users_per_step,'aux_mode':'frozen_BN_dropout_with_autograd_native_GRU',
              'monitor_policy':MONITOR_POLICY,
              'checkpoint_selection':'minimum mean original-style supervised BCE, not test metrics'}
    lock_json(output/'run_config.json',contract)
    ch=digest(contract)
    complete=output/'training_complete.json'
    if complete.exists():
        marker=read_json(complete)
        if marker['contract_hash']!=ch or any(sha256(output/p)!=h for p,h in marker['checkpoints'].items()):
            raise ValueError('Completed training contract/checkpoint changed')
        return marker
    device=original.resolve_device(cuda)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    # Same initialization and DataLoader order as the original E0 implementation.
    original.set_seed(seed,deterministic=cfg['deterministic'])
    bundle=original.load_semantic_feature_bundle(state.graph,None,device=device)
    triples=original.read_triples(state.graph/'triples.txt',bundle.entity2id,bundle.relation2id)
    dataset=original.TripleDataset(triples,bundle.id2relation,bundle.exercise_entity_ids.cpu().tolist(),
                                  original.build_positive_tails_by_hr(triples),cfg['negative_ratio'],seed)
    loader=DataLoader(dataset,batch_size=cfg['bs'],shuffle=True)
    monitored_batches=MonitoredBatches(loader.batch_sampler,dataset,bundle.relation2id['rec']) if effective else None
    if monitored_batches is not None:loader=DataLoader(dataset,batch_sampler=monitored_batches)
    model=build_model(original,bundle,cfg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scorer=QScorer(state,bundle,scale_from_assets(assets),assets['beta'],device)
    # No auxiliary sampler construction/forward/RNG draws when its weight is zero.
    objective=ExposureObjective(state,scorer,assets['temperatures'],assets['top_ns'],seed,users_per_step) if effective else None
    monitor=TrainingMonitor(state,scorer,assets,seed,triples)
    initial={'model_state_sha256':initialization_hash(model),'seed':seed,'contract_hash':ch,
             'training_audit':monitor.evaluate(model)}
    lock_json(output/'initialization_audit.json',initial)
    start=0;best=float('inf');best_epoch=0;elapsed=0.
    if (output/'last.pt').exists():
        ck=torch.load(output/'last.pt',map_location=device,weights_only=False)
        if ck['contract_hash']!=ch:raise ValueError('Checkpoint/contract mismatch')
        model.load_state_dict(ck['model_state_dict'],strict=True);optimizer.load_state_dict(ck['optimizer_state_dict'])
        start=ck['epoch'];best=ck['best_loss'];best_epoch=ck['best_epoch'];elapsed=ck['training_seconds']
        if objective is not None:objective.load_state_dict(ck['exposure_rng'])
        restore_rng(ck['rng'])
    for epoch in range(start,cfg['epochs']):
        began=time.perf_counter();model.train()
        totals={'bce':0.,'exposure':0.,'total':0.};batches=0;aux_steps=0
        balances=[]
        for batch_index,(h,r,t,labels) in enumerate(tqdm(loader,desc=f'{state.dataset}/{method}/seed{seed} epoch {epoch+1}/{cfg["epochs"]}')):
            h,r,t,labels=(x.to(device) for x in (h,r,t,labels))
            optimizer.zero_grad()
            probabilities=scorer.probabilities(model,h,r,t)
            bce=F.binary_cross_entropy(probabilities.view(-1),labels.view(-1))
            total,aux,stats=joint_loss(model,bce,objective,effective,bool((r==scorer.rec).any()) if effective else False)
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite MAED objective')
            if aux is not None and batch_index in monitored_batches.targets:
                balances.append({'batch':batch_index+1,'effective_exposure_batch':aux_steps+1,
                                 'roles':monitored_batches.targets[batch_index],
                                 **gradient_balance(model,bce,aux,effective)})
            total.backward();optimizer.step()
            totals['bce']+=float(bce.detach());totals['total']+=float(total.detach());batches+=1
            if aux is not None:totals['exposure']+=float(aux.detach());aux_steps+=1
        bce_mean=totals['bce']/batches
        is_best=bce_mean<best
        if is_best:best=bce_mean;best_epoch=epoch+1
        audit=monitor.evaluate(model)
        elapsed+=time.perf_counter()-began
        log={'epoch':epoch+1,'batches':batches,'bce':bce_mean,'exposure':totals['exposure']/max(1,aux_steps),
             'total':totals['total']/batches,'auxiliary_steps':aux_steps,'gradient_balance':balances,
             'training_exposure_audit':audit,'training_seconds':elapsed,'best_epoch':best_epoch}
        write_json(output/'epochs'/f'{epoch+1:03d}.json',log);print(json.dumps(log),flush=True)
        payload={'training_version':TRAINING_VERSION,'config':cfg,'assets':assets,'model_state_dict':model.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch+1,'loss':bce_mean,
                 'best_loss':best,'best_epoch':best_epoch,'training_seconds':elapsed,'contract_hash':ch,
                 'rng':rng_state(),'exposure_rng':objective.state_dict() if objective is not None else None}
        if is_best:atomic_checkpoint(output/'best.pt',payload)
        atomic_checkpoint(output/'last.pt',payload)
    write_json(output/'config.json',cfg)
    marker={'contract_hash':ch,'best_epoch':best_epoch,'epochs':cfg['epochs'],'training_seconds':elapsed,
            'checkpoints':{p:sha256(output/p) for p in ('best.pt','last.pt')}}
    write_json(complete,marker)
    return marker


def export_scores(state,checkpoint,output,cuda='true'):
    checkpoint=Path(checkpoint);output=Path(output)
    original=model_modules();device=original.resolve_device(cuda)
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck.get('training_version')!=TRAINING_VERSION:raise ValueError('Not a controlled MAED checkpoint')
    bundle=original.load_semantic_feature_bundle(state.graph,None,device=device)
    model=build_model(original,bundle,ck['config']).to(device)
    model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
    scorer=QScorer(state,bundle,scale_from_assets(ck['assets']),ck['assets']['beta'],device)
    rows=[];logits=[]
    with torch.no_grad():
        for users in np.array_split(state.test,max(1,int(np.ceil(len(state.test)/64)))):
            z=scorer.rec_logits(model,users,len(state.q))
            # Positive-beta M1/M3/M4 rank actual fused logits, avoiding sigmoid ties.
            scores=z if scorer.beta else torch.sigmoid(z)
            rows.extend({'uid':f'uid{u}','scores':values.tolist()} for u,values in zip(users,scores.cpu().numpy()))
            logits.extend(z.cpu().numpy())
    write_json(output,rows);np.save(output.with_suffix('.logits.npy'),np.asarray(logits))
    return {'checkpoint_sha256':sha256(checkpoint),'score_sha256':sha256(output),
            'logits_sha256':sha256(output.with_suffix('.logits.npy')),'contract_hash':ck['contract_hash'],
            'beta':scorer.beta,'score_kind':'Q_fused_logit' if scorer.beta else 'original_float32_probability'}
