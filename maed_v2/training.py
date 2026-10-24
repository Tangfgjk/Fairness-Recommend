"""C0/C3/C4 training with explicit branch snapshots and complete-epoch replay.

The frozen MAED-v1 modules are reused read-only for Q, D, E and diagnostics.
Test-state graph B is never passed to the training/auxiliary forward.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from debias.io import ROOT, model_modules, read_json, write_json, sha256, digest, TOP_NS
from debias.training import build_model, rng_state, restore_rng, atomic_checkpoint
from debias.scoring import fixed_statistics
from maed.io import lock_json
from maed.scoring import EducationScale, fuse_logits
from maed.training import QScorer, scale_from_assets
from maed.exposure import rule_targets
from maed.boundary_exposure import calibrated_exposure_loss, calibrate_temperatures
from maed.training_monitor import initialization_hash, score_audit, MonitoredBatches, gradient_balance
from .model import SemanticCompGCN
from .graph_protocol import GraphProtocol
from .diagnostics import gradient_groups, encoding_diagnostics

VERSION = 'maed_v2_controlled_training_20261024'
ACCEPTED_CHUNKS = {'Eedi':4096, 'algebra2005':65536, 'XES3G5M-sub-small':131072}


def check_chunk(dataset, chunk):
    if dataset in ACCEPTED_CHUNKS and chunk != ACCEPTED_CHUNKS[dataset]:
        raise ValueError(f'Unaccepted formal chunk for {dataset}: {chunk}; expected {ACCEPTED_CHUNKS[dataset]}')


def source_hash():
    from maed.io import source_hash as inherited_hash
    files = sorted((ROOT/'maed_v2').glob('*.py'))
    files += [ROOT/'scripts'/n for n in ('run_maed_v2_training.py',)]
    return digest({'inherited':inherited_hash(),
                   'v2':{str(p.relative_to(ROOT)):sha256(p) for p in files}})


def build_runtime(state, cfg, chunk, seed, device):
    check_chunk(state.dataset, chunk)
    original = model_modules()
    original.set_seed(seed, deterministic=cfg['deterministic'])
    bundle = original.load_semantic_feature_bundle(state.graph, None, device=device)
    protocol = GraphProtocol(state, bundle.entity2id, bundle.relation2id)
    base = build_model(original, bundle, cfg).to(device)
    model = SemanticCompGCN(base, protocol.inverse_original_ids, test_entities=protocol.test_entities,
                           chunk_size=chunk, recompute=True, graph_seed=seed).to(device)
    return model, bundle, protocol


class GraphQScorer(QScorer):
    """The same rec-only Q, evaluated on explicit graph-encoded logits."""
    def __init__(self, state, bundle, scale, beta, device, graph):
        super().__init__(state, bundle, scale, beta, device)
        self.graph = graph
        self.items = torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)

    def probabilities(self, model, h, r, t, encoded=None):
        encoded = model.encode_graph(self.graph) if encoded is None else encoded
        return torch.sigmoid(self.fuse_triples(model.score_logits(h,r,t,encoded),h,r,t))

    def rec_logits(self, model, uids, nitems, encoded=None):
        if nitems != len(self.items): raise ValueError('Catalog changed')
        encoded = model.encode_graph(self.graph) if encoded is None else encoded
        heads = torch.tensor([self.bundle.entity2id[f'uid{int(u)}'] for u in uids],device=self.items.device)
        z = model.score_rec_all(heads,self.items,encoded,self.rec)
        if not self.beta:return z
        d = self.distance[torch.as_tensor(uids,device=z.device,dtype=torch.long)].to(z.dtype)
        return fuse_logits(z,d,self.scale,self.beta)


class GraphExposure:
    def __init__(self, state, scorer, temperatures, ns, seed, users_per_step=16):
        state.training_distance(state.train)
        self.state=state;self.scorer=scorer;self.temperatures=tuple(temperatures);self.ns=tuple(ns)
        self.rng=np.random.default_rng(seed+790003);self.users_per_step=users_per_step

    def state_dict(self):return self.rng.bit_generator.state
    def load_state_dict(self,value):self.rng.bit_generator.state=value

    def __call__(self,model):
        users=self.rng.choice(self.state.train,min(len(self.state.train),self.users_per_step),replace=False)
        target=rule_targets(self.state.training_distance(users),self.ns)
        with fixed_statistics(model),torch.backends.cudnn.flags(enabled=False):
            # Exactly one fresh graph encoding in this branch, shared by all N.
            z=self.scorer.rec_logits(model,users,len(self.state.q))
            loss,per,_=calibrated_exposure_loss(z,target,self.ns,self.temperatures)
        return loss,{'users':users.tolist(),'per_N_soft_TV':per.detach().cpu().tolist()}


class GraphMonitor:
    """Fixed training cohort only; observations never choose model/temperature."""
    def __init__(self,state,scorer,assets,seed,triples):
        self.state=state;self.scorer=scorer;self.ns=tuple(assets['top_ns'])
        self.temperatures=assets.get('temperatures')
        self.users=np.sort(np.random.default_rng(seed+820001).choice(state.train,min(48,len(state.train)),replace=False))
        self.target=rule_targets(state.training_distance(self.users),self.ns)
        self.frequency=np.zeros(len(state.q),dtype=np.int64)
        for h,r,t in triples:
            if r==scorer.rec:self.frequency[int(scorer.item_index[t])]+=1
        self.initial_core=None;self.initial_ratios=None

    def state_dict(self):
        return {'initial_core':self.initial_core,'initial_ratios':self.initial_ratios,
                'temperatures':self.temperatures}

    def load_state_dict(self,value):
        self.initial_core=value['initial_core'];self.initial_ratios=value['initial_ratios']
        self.temperatures=value['temperatures']

    def evaluate(self,model):
        with fixed_statistics(model),torch.no_grad():
            encoded=model.encode_graph(self.scorer.graph)
            heads=torch.tensor([self.scorer.bundle.entity2id[f'uid{u}'] for u in self.users],device=self.scorer.items.device)
            raw=model.score_rec_all(heads,self.scorer.items,encoded,self.scorer.rec)
            reward=self.scorer.beta*self.scorer.scale.transform(self.scorer.distance[self.users].to(raw.dtype)) if self.scorer.beta else torch.zeros_like(raw)
            core=raw.cpu().numpy();reward=reward.cpu().numpy()
        if self.temperatures is None:
            # C0 has no exposure objective. These scales ONLY define a fixed
            # training-monitor diagnostic; C3/C4 official scales come from best C0.
            self.temperatures=[r['temperature'] for r in calibrate_temperatures(core+reward,self.ns)['per_N']]
        result=score_audit(core,reward,self.frequency,self.target,self.ns,self.temperatures,
                           self.initial_core,self.initial_ratios)
        if self.initial_core is None:
            self.initial_core=core.copy()
            self.initial_ratios=[r['median_gap_over_temperature'] for r in result['boundary_per_N']]
        return {'training_users':self.users.tolist(),'test_data_used':False,
                'temperature_scope':'C0 initial-cohort monitoring only' if not self.scorer.beta else 'frozen best-C0 training-wide Q calibration',
                **result}


def train(state, assets, output, method, seed, execution_hash, device='cuda', *,
          exposure_weight=.01, users_per_step=16, batch_hook=None, engineering_max_batches=None):
    if method not in ('C0','C3','C4') or not np.isfinite(exposure_weight) or exposure_weight<0:
        raise ValueError('Invalid controlled method/weight')
    if execution_hash != source_hash():raise ValueError('Training source hash differs from current implementation')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);device=torch.device(device)
    cfg=dict(assets['reference_config']);chunk=assets['chunk_size']
    cfg.update(data_path=str(state.graph),save_path=str(output),seed=seed,cuda='true' if device.type=='cuda' else 'false')
    beta=0. if method=='C0' else assets['beta']
    effective=exposure_weight if method=='C4' else 0.
    if method!='C0' and not assets.get('C0_checkpoint_sha256'):raise ValueError('C3/C4 require frozen trained C0 reference')
    if effective and not assets.get('temperatures'):raise ValueError('C4 temperatures missing')
    contract={'version':VERSION,'method':method,'seed':seed,'config':cfg,'assets_hash':digest(assets),
              'source_hash':execution_hash,'beta':beta,'exposure_weight':effective,'users_per_step':users_per_step,
              'chunk_size':chunk,'recompute':True,'device':str(device),'inference':'B_full_reference; A same-checkpoint control',
              'checkpoint_policy':'minimum mean supervised BCE; last separate',
              'engineering_only_max_batches':engineering_max_batches,
              'replay_policy':'last complete epoch including epoch zero; full RNG and Adam restoration'}
    lock_json(output/'run_config.json',contract);ch=digest(contract)
    complete=output/'training_complete.json'
    if complete.exists():
        marker=read_json(complete)
        if marker['contract_hash']!=ch or any(sha256(output/p)!=h for p,h in marker['checkpoints'].items()):
            raise ValueError('Completed training differs')
        return marker
    model,bundle,protocol=build_runtime(state,cfg,chunk,seed,device)
    graph=protocol.base.to(device)
    if protocol.base.graph_id!=assets['base_graph_id']:raise ValueError('Training graph identity changed')
    original=model_modules()
    triples=original.read_triples(state.graph/'triples.txt',bundle.entity2id,bundle.relation2id)
    dataset=original.TripleDataset(triples,bundle.id2relation,bundle.exercise_entity_ids.cpu().tolist(),
                                  original.build_positive_tails_by_hr(triples),cfg['negative_ratio'],seed)
    loader=DataLoader(dataset,batch_size=cfg['bs'],shuffle=True)
    batches=MonitoredBatches(loader.batch_sampler,dataset,bundle.relation2id['rec']) if effective else None
    if batches is not None:loader=DataLoader(dataset,batch_sampler=batches)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scorer=GraphQScorer(state,bundle,scale_from_assets(assets),beta,device,graph)
    objective=GraphExposure(state,scorer,assets['temperatures'],assets['top_ns'],seed,users_per_step) if effective else None
    monitor=GraphMonitor(state,scorer,assets,seed,triples)
    initialization={'model_state_sha256':initialization_hash(model),'seed':seed,'contract_hash':ch,
                    'training_audit':monitor.evaluate(model)}
    audit_path=output/'initialization_audit.json'
    if audit_path.exists():
        previous=read_json(audit_path)
        # CUDA aggregation is not bit-deterministic. Contract and actual initial
        # state must match; recomputed floating diagnostic summaries need not.
        if any(previous[k]!=initialization[k] for k in ('model_state_sha256','seed','contract_hash')):
            raise ValueError('Initialization/contract changed')
    else:
        lock_json(audit_path,initialization)
    start=0;best=float('inf');best_epoch=0;elapsed=0.

    def payload(epoch,loss):
        return {'training_version':VERSION,'config':cfg,'assets':assets,'method':method,
                'source_hash':execution_hash,'contract':copy.deepcopy(contract),
                'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'epoch':epoch,'loss':loss,'best_loss':best,'best_epoch':best_epoch,'training_seconds':elapsed,
                'contract_hash':ch,'rng':rng_state(),'exposure_rng':objective.state_dict() if objective else None}

    if (output/'last.pt').exists():
        ck=read_checkpoint(state,output/'last.pt',expected_contract_hash=ch)
        model.load_state_dict(ck['model_state_dict'],strict=True);optimizer.load_state_dict(ck['optimizer_state_dict'])
        start=ck['epoch'];best=ck['best_loss'];best_epoch=ck['best_epoch'];elapsed=ck['training_seconds']
        if objective:objective.load_state_dict(ck['exposure_rng'])
        monitor.load_state_dict(ck['monitor_state'])
        restore_rng(ck['rng'])
    else:
        # A crash in epoch one can now replay from its exact original RNG state.
        initial_payload=payload(0,None);initial_payload['monitor_state']=monitor.state_dict()
        atomic_checkpoint(output/'last.pt',initial_payload)
    for epoch in range(start,cfg['epochs']):
        model.train();began=time.perf_counter();totals={'bce':0.,'exposure':0.,'total':0.};count=0;aux_steps=0;balances=[]
        for i,(h,r,t,labels) in enumerate(tqdm(loader,desc=f'{state.dataset}/{method}/seed{seed} epoch {epoch+1}/{cfg["epochs"]}')):
            h,r,t,labels=(x.to(device) for x in (h,r,t,labels));optimizer.zero_grad(set_to_none=True)
            encoded=model.encode_graph(graph)
            bce=F.binary_cross_entropy(scorer.probabilities(model,h,r,t,encoded),labels)
            aux=None;total=bce
            if effective and bool((r==scorer.rec).any()):
                aux,_=objective(model);total=bce+effective*aux
                if i in batches.targets:
                    balances.append({'batch':i+1,'roles':batches.targets[i],**gradient_balance(model,bce,aux,effective)})
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite supervised/exposure objective')
            total.backward()
            if i==0:
                groups=gradient_groups(model)
                if any(not row['finite'] for row in groups.values()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            totals['bce']+=float(bce.detach());totals['total']+=float(total.detach());count+=1
            if aux is not None:totals['exposure']+=float(aux.detach());aux_steps+=1
            del encoded,total,bce,aux
            if batch_hook is not None:batch_hook(epoch+1,i+1)
            if engineering_max_batches is not None and count>=engineering_max_batches:break
        mean=totals['bce']/count;is_best=mean<best
        if is_best:best=mean;best_epoch=epoch+1
        audit=monitor.evaluate(model);elapsed+=time.perf_counter()-began
        log={'epoch':epoch+1,'batches':count,'bce':mean,'exposure':totals['exposure']/max(1,aux_steps),
             'total':totals['total']/count,'auxiliary_steps':aux_steps,'gradient_balance':balances,
             'first_batch_gradient_groups':groups,'training_exposure_audit':audit,
             'training_seconds':elapsed,'epoch_seconds':time.perf_counter()-began,'best_epoch':best_epoch,
             'graph_encodings':'one supervised per batch, plus one per positive exposure branch'}
        write_json(output/'epochs'/f'{epoch+1:03d}.json',log)
        print(json.dumps({k:v for k,v in log.items() if k not in ('training_exposure_audit','gradient_balance','first_batch_gradient_groups')}),flush=True)
        ck=payload(epoch+1,mean);ck['monitor_state']=monitor.state_dict()
        if is_best:atomic_checkpoint(output/'best.pt',ck)
        atomic_checkpoint(output/'last.pt',ck)
    write_json(output/'config.json',cfg)
    marker={'contract_hash':ch,'epochs':cfg['epochs'],'best_epoch':best_epoch,'training_seconds':elapsed,
            'checkpoints':{p:sha256(output/p) for p in ('best.pt','last.pt')}}
    write_json(complete,marker)
    return marker


def read_checkpoint(state,path,*,expected_contract_hash=None):
    """Validate standalone provenance before building a model or using cached results.

    Checkpoints are trusted local PyTorch files, not untrusted serialized input.
    Old checkpoints missing this receipt fail closed; no silent legacy bypass.
    """
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('training_version')!=VERSION:raise ValueError('Not a MAED-v2 controlled checkpoint')
    contract=ck.get('contract')
    if not isinstance(contract,dict):raise ValueError('Checkpoint contract receipt missing; legacy checkpoint cannot be reused')
    current=source_hash()
    if ck.get('source_hash')!=current or contract.get('source_hash')!=current:
        raise ValueError('Checkpoint source hash differs from current implementation')
    if digest(contract)!=ck.get('contract_hash') or (expected_contract_hash is not None and ck['contract_hash']!=expected_contract_hash):
        raise ValueError('Checkpoint contract hash mismatch')
    if (contract.get('version')!=VERSION or ck.get('method')!=contract.get('method')
        or ck.get('method') not in ('C0','C3','C4') or ck.get('config')!=contract.get('config')
        or digest(ck.get('assets'))!=contract.get('assets_hash')):
        raise ValueError('Checkpoint payload differs from its contract')
    if ck['assets'].get('state_hash')!=state.provenance['state_hash']:
        raise ValueError('Checkpoint education state changed')
    if (contract.get('chunk_size')!=ck['assets']['chunk_size']
        or contract.get('seed')!=ck['config']['seed']):
        raise ValueError('Checkpoint execution settings differ from contract')
    check_chunk(state.dataset,ck['assets']['chunk_size'])
    return ck


def load_checkpoint(state,path,device,*,expected_contract_hash=None):
    ck=read_checkpoint(state,path,expected_contract_hash=expected_contract_hash)
    model,bundle,protocol=build_runtime(state,ck['config'],ck['assets']['chunk_size'],ck['contract']['seed'],torch.device(device))
    model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
    if protocol.base.graph_id!=ck['assets']['base_graph_id']:raise ValueError('Checkpoint graph changed')
    return model,bundle,protocol,ck


def calibrate_assets(state,c0_checkpoint,output,device):
    checked=read_checkpoint(state,c0_checkpoint)
    if checked['method']!='C0':raise ValueError('Temperature reference must be trained C0')
    path=Path(output)/'assets.json'
    if path.exists():
        saved=read_json(path)
        if (saved['C0_checkpoint_sha256']!=sha256(c0_checkpoint) or saved['state_hash']!=state.provenance['state_hash']
            or saved['calibration_source_hash']!=source_hash() or saved['calibration_train_users']!=state.train.tolist()
            or saved.get('C0_contract_hash')!=checked['contract_hash']):
            raise ValueError('Frozen C0 calibration provenance changed')
        # Never re-run GPU calibration on resume: atomic reductions may move a
        # boundary by a few ULPs, and temperatures must remain exactly frozen.
        return saved
    model,bundle,protocol,ck=load_checkpoint(state,c0_checkpoint,device)
    if ck['method']!='C0':raise ValueError('Temperature reference must be trained C0')
    scale=scale_from_assets(ck['assets'])
    scorer=GraphQScorer(state,bundle,scale,.5,torch.device(device),protocol.base.to(device))
    with torch.no_grad():
        encoded=model.encode_graph(scorer.graph)
        values=torch.cat([scorer.rec_logits(model,u,len(state.q),encoded).cpu() for u in np.array_split(state.train,max(1,int(np.ceil(len(state.train)/32))))]).numpy()
    calibration=calibrate_temperatures(values,TOP_NS)
    assets={**ck['assets'],'beta':.5,'C0_checkpoint_sha256':sha256(c0_checkpoint),
            'C0_contract_hash':ck['contract_hash'],
            'temperature_calibration':calibration,'temperatures':[r['temperature'] for r in calibration['per_N']],
            'calibration_source_hash':source_hash(),
            'temperature_policy':'all training students; fixed trained C0+Q, original per-N boundary rule',
            'calibration_train_users':state.train.tolist()}
    lock_json(Path(output)/'assets.json',assets)
    return assets


def export_scores(state,checkpoint,output,device='cuda',protocol_name='B',intervention='full'):
    if protocol_name not in ('A','B') or intervention not in ('full','no_neighbors','semantic'):
        raise ValueError('Unsupported inference/diagnostic protocol')
    checkpoint=Path(checkpoint);output=Path(output)
    model,bundle,protocol,ck=load_checkpoint(state,checkpoint,device);device=torch.device(device)
    output.parent.mkdir(parents=True,exist_ok=True)
    beta=0. if ck['method']=='C0' else ck['assets']['beta']
    items=torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)
    rows=[];logits=[];queries=[];began=time.perf_counter()
    with torch.no_grad():
        initial=model.semantic_initials()
        a=None
        if protocol_name=='A':a=model.encode_graph(protocol.base.to(device),initial=initial)
        for value in state.test:
            uid=int(value);u=torch.tensor([bundle.entity2id[f'uid{uid}']],device=device)
            graph=protocol.query(uid).to(device) if protocol_name=='B' else protocol.base.to(device)
            if intervention=='semantic':encoded=model.encode_graph(graph,'bypass')
            elif a is not None and intervention=='full':encoded=a
            else:encoded=model.encode_graph(graph,initial=initial,neighbors=intervention!='no_neighbors')
            z=model.score_rec_all(u,items,encoded,bundle.relation2id['rec'])
            if beta:z=fuse_logits(z,torch.as_tensor(state.distance[uid:uid+1],dtype=z.dtype,device=device),scale_from_assets(ck['assets']),beta)
            values=z.cpu().numpy()[0];logits.append(values);rows.append({'uid':f'uid{uid}','scores':values.tolist()})
            queries.append({'uid':uid,'graph_id':graph.graph_id,'query_state':graph.audit if protocol_name=='B' else None,
                            'degree_hash':digest(graph.degree.cpu().tolist())})
    write_json(output,rows);np.save(output.with_suffix('.logits.npy'),np.asarray(logits))
    return {'checkpoint_sha256':sha256(checkpoint),'score_sha256':sha256(output),
            'source_hash':ck['source_hash'],'encoding_mode':'full' if intervention!='semantic' else 'bypass',
            'logits_sha256':sha256(output.with_suffix('.logits.npy')),'contract_hash':ck['contract_hash'],
            'score_kind':'ranking_logit_not_probability','beta':beta,'method':ck['method'],
            'inference_protocol':protocol_name,'intervention':intervention,'elapsed_seconds':time.perf_counter()-began,
            'graph_base_id':protocol.base.graph_id,'state_hash':state.provenance['state_hash'],
            'checkpoint_selection':'supervised BCE best or last diagnostic; never selected on test metrics',
            'queries':queries}
