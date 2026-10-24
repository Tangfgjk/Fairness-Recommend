"""Positive-C4 real-device and real-Eedi mid-epoch replay gates; not experiments."""
import copy
import gc
from pathlib import Path
import time

import numpy as np
import torch

from debias.io import ROOT,DATASETS,read_json,write_json,TOP_NS
from maed.scoring import EducationScale
from maed.boundary_exposure import calibrate_temperatures
from maed.training_monitor import initialization_hash
from .training import build_runtime,GraphQScorer,GraphExposure,train,source_hash
from .p0 import first_batch,memory,synchronize
from .diagnostics import gradient_groups


def same_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b


def engineering_assets(state,assets,device):
    """The untrained calibration below is ONLY for an engineering stress/replay test."""
    model,bundle,protocol=build_runtime(state,assets['reference_config'],assets['chunk_size'],2024,device)
    scale=EducationScale.fit(state)
    scorer=GraphQScorer(state,bundle,scale,.5,device,protocol.base.to(device))
    model.eval()
    with torch.no_grad():z=scorer.rec_logits(model,state.train[:min(48,len(state.train))],len(state.q)).cpu().numpy()
    data={**assets,'C0_checkpoint_sha256':'engineering_untrained_reference_not_for_formal_use',
          'temperatures':[x['temperature'] for x in calibrate_temperatures(z,TOP_NS)['per_N']]}
    return data,model,bundle,protocol,scorer


def device_step(state,assets,device,output):
    from debias.io import model_modules
    data,model,bundle,protocol,scorer=engineering_assets(state,assets,device)
    batch=first_batch(model_modules(),bundle,state,assets['reference_config'],device)
    objective=GraphExposure(state,scorer,data['temperatures'],TOP_NS,2024,16)
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    began=time.perf_counter()
    encoded=model.encode_graph(scorer.graph)
    bce=torch.nn.functional.binary_cross_entropy(scorer.probabilities(model,*batch[:3],encoded),batch[3])
    modes=[m.training for m in model.modules()]
    aux,_=objective(model)
    if modes!=[m.training for m in model.modules()]:raise AssertionError('Auxiliary mode restoration failed')
    # Check exposure itself, not merely gradients of BCE+exposure.
    parameters=[(n,p) for n,p in model.named_parameters() if n.startswith('layer.') or
                n.startswith('base.uid_feature_only_fusion') or n.startswith('base.exercise_feature_only_fusion')]
    gradients=torch.autograd.grad(aux,[p for _,p in parameters],retain_graph=True,allow_unused=True)
    exposure_groups={n:None if g is None else float(g.detach().double().norm()) for (n,_),g in zip(parameters,gradients)}
    if any(g is None or not torch.isfinite(g).all() for g in gradients):raise AssertionError('Exposure gradient absent/nonfinite')
    if not any(v>0 for n,v in exposure_groups.items() if n.startswith('layer.')):raise AssertionError('Exposure graph gradient is zero')
    total=bce+.01*aux;total.backward();optimizer.step();synchronize(device)
    result={'dataset':state.dataset,'status':'passed','temporary_update_only':True,'BCE':float(bce.detach()),
            'exposure':float(aux.detach()),'exposure_gradient_norms':exposure_groups,
            'total_gradient_groups':gradient_groups(model),'complete_step_seconds':time.perf_counter()-began,
            'batch_size':len(batch[0]),'negative_ratio':5,'auxiliary_students':16,'top_ns':list(TOP_NS),
            'graph_encodings_per_joint_step':2,**memory(device)}
    write_json(output/'positive_C4_step.json',result)
    return result


def _real_replay(state,assets,output):
    """Full real Eedi graph/features, bs1024/neg5; two engineering epochs of 3 batches.

    Single-thread CPU is intentional for exact replay; repeated-index gather
    backward is not bit-reproducible on multithread CPU, nor on inherited CUDA.
    The real CUDA positive-C4 path is independently checked in device_step.
    """
    data,model,bundle,protocol,scorer=engineering_assets(state,assets,torch.device('cpu'))
    del model,bundle,protocol,scorer;gc.collect()
    data=copy.deepcopy(data);data['reference_config']['epochs']=2
    options={'exposure_weight':.01,'users_per_step':16,'engineering_max_batches':3}
    start=time.perf_counter()
    execution_hash=source_hash()
    train(state,data,output/'uninterrupted','C4',2024,execution_hash,'cpu',**options)
    def interrupt(epoch,batch):
        if epoch==2 and batch==2:raise RuntimeError('intentional real-data mid-epoch interruption')
    try:
        train(state,data,output/'interrupted','C4',2024,execution_hash,'cpu',batch_hook=interrupt,**options)
    except RuntimeError as error:
        if 'intentional real-data' not in str(error):raise
    else:raise AssertionError('Interruption did not occur')
    previous=torch.load(output/'interrupted/last.pt',map_location='cpu',weights_only=False)
    if previous['epoch']!=1:raise AssertionError('Incomplete epoch was checkpointed')
    train(state,data,output/'interrupted','C4',2024,execution_hash,'cpu',**options)
    a=torch.load(output/'uninterrupted/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(output/'interrupted/last.pt',map_location='cpu',weights_only=False)
    checks={k:same_tree(a[k],b[k]) for k in ('model_state_dict','optimizer_state_dict','rng','exposure_rng','monitor_state','loss','best_loss','best_epoch')}
    if not all(checks.values()):raise AssertionError(f'Real replay differed: {checks}')
    result={'status':'passed','dataset':state.dataset,'device':'cpu','CPU_threads':torch.get_num_threads(),'original_batch_size':1024,
            'negative_ratio':5,'engineering_epochs':2,'batches_per_engineering_epoch':3,
            'interruption':'after optimizer update at epoch2/batch2; replay previous complete epoch',
            'checks':checks,'elapsed_seconds':time.perf_counter()-start,'formal_training':False}
    write_json(output/'replay_result.json',result)
    return result


def real_replay(state,assets,output):
    previous=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _real_replay(state,assets,output)
    finally:
        torch.set_num_threads(previous)


def run_preflight(root,args,protocol):
    from scripts.run_maed_v2_training import base_assets
    out=Path(root)/'engineering';out.mkdir(parents=True,exist_ok=True)
    steps=[]
    for dataset in args.datasets.split(','):
        state,assets=base_assets(dataset,args,root)
        print(f'Engineering real positive C4 step: {dataset}',flush=True)
        steps.append(device_step(state,assets,torch.device(args.device),out/dataset))
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    state,assets=base_assets('Eedi',args,root)
    print('Engineering real Eedi CPU positive-C4 mid-epoch replay (not formal training)',flush=True)
    replay=real_replay(state,assets,out/'real_Eedi_replay')
    result={'status':'passed','source_hash':protocol['source_hash'],'device_steps':steps,'real_replay':replay,
            'formal_training_started':False}
    write_json(Path(root)/'engineering_gate.json',result)
    return result
