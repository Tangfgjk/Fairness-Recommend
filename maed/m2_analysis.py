"""Mechanism diagnostics, training-only cohorts, no persistent parameter updates."""
import numpy as np
import torch

from debias.io import TOP_NS
from debias.scoring import fixed_statistics
from .model import load_frozen
from .scoring import score_rec, fuse_logits
from .boundary_exposure import calibrated_exposure_loss
from .exposure import hard_exposure, rule_targets


def cosine_matrix(gram):
    gram=np.asarray(gram,float)
    norms=np.sqrt(np.maximum(np.diag(gram),0))
    denom=norms[:,None]*norms[None,:]
    return [[float(gram[i,j]/denom[i,j]) if denom[i,j]>1e-30 else None
             for j in range(len(norms))] for i in range(len(norms))]


def shared_item_fraction(matrix):
    x=np.asarray(matrix,float)
    x=x-x.mean(1,keepdims=True)
    total=float(np.square(x).sum())
    return float(len(x)*np.square(x.mean(0)).sum()/total) if total>1e-30 else None


def snapshot(logits,target,ns,temperatures):
    z=logits.detach() if torch.is_tensor(logits) else torch.as_tensor(logits,dtype=torch.float64)
    with torch.no_grad():
        soft,per,w=calibrated_exposure_loss(z,target,ns,temperatures)
    hard=hard_exposure(z.cpu().numpy(),ns)
    return {'soft_mean_TV':float(soft),'soft_TV':per.cpu().tolist(),
            'hard_TV':(.5*np.abs(hard-target).sum(1)).tolist()}


def change_counts(before,after,ns):
    a=np.argsort(-before,axis=1,kind='stable')
    b=np.argsort(-after,axis=1,kind='stable')
    changed=[]; entered=[]
    for n in ns:
        counts=[len(set(y[:n])-set(x[:n])) for x,y in zip(a,b)]
        changed.append(sum(c>0 for c in counts));entered.append(sum(counts))
    return changed,entered


def score_diagnostics(initial,target,temperatures,ns=TOP_NS,step_scale=0.02,global_scale=1.):
    z=initial.detach().clone().requires_grad_(True)
    _,per,w=calibrated_exposure_loss(z,target,ns,temperatures)
    gradients=[]
    for i in range(len(ns)):
        gradients.append(torch.autograd.grad(per[i],z,retain_graph=i<len(ns)-1)[0].detach())
    g=torch.stack(gradients)
    flat=g.reshape(len(ns),-1)
    gram=(flat.double()@flat.double().T).cpu().numpy()
    aggregate=g.mean(0)
    rms=float(aggregate.square().mean().sqrt())
    step=step_scale*global_scale/max(rms,1e-30)
    raw=z.detach().cpu().numpy();grad=aggregate.cpu().numpy()
    weights=w.detach().cpu().numpy()
    per_grad=g.cpu().numpy()
    rank=np.argsort(-raw,axis=1,kind='stable')
    hard=hard_exposure(raw,ns)
    next_z=raw-step*grad
    changed,entered=change_counts(raw,next_z,ns)
    bounds=[];items=[]
    for i,n in enumerate(ns):
        inside=rank[:,n-1];outside=rank[:,n];ix=np.arange(len(raw))
        gaps=raw[ix,inside]-raw[ix,outside]
        slope=grad[ix,inside]-grad[ix,outside]
        possible=slope>1e-30
        critical=gaps[possible]/slope[possible]
        local_g=per_grad[i]
        norms=np.linalg.norm(local_g,axis=1)
        valid=norms>1e-30
        unit=local_g[valid]/norms[valid,None]
        pair_cos=unit@unit.T
        off=pair_cos[np.triu_indices(len(unit),1)]
        residual=hard[i]-target[i]
        mean_g=local_g.mean(0)
        excess=np.maximum(residual,0)
        bounds.append({'N':n,'temperature':temperatures[i],
                       'boundary_gap_median':float(np.median(gaps)),
                       'temperature_over_median_gap':float(temperatures[i]/max(np.median(gaps),1e-30)),
                       'actual_TopN_soft_mass_fraction':float(np.take_along_axis(weights[i],rank[:,:n],axis=1).sum(1).mean()/n),
                       'weights_le_0p01_fraction':float(np.mean(weights[i]<=.01)),
                       'weights_ge_0p99_fraction':float(np.mean(weights[i]>=.99)),
                       'active_weights_fraction':float(np.mean((weights[i]>.01)&(weights[i]<.99))),
                       'shared_item_gradient_fraction':shared_item_fraction(local_g),
                       'student_gradient_cosine_mean':float(off.mean()) if len(off) else None,
                       'overexposure_mass_with_downward_gradient_fraction':float(excess[mean_g>0].sum()/excess.sum()) if excess.sum()>0 else None,
                       'aggregate_step_size':step,'predicted_adjacent_boundary_crossings':int(np.sum(possible&(step*slope>gaps))),
                       'finite_adjacent_crossing_threshold_median':float(np.median(critical)) if len(critical) else None,
                       'actual_changed_users_after_first_step':changed[i],
                       'actual_entered_items_after_first_step':entered[i]})
        for e in range(raw.shape[1]):
            items.append({'N':n,'exercise':e,'hard_exposure':float(hard[i,e]),'rule_target':float(target[i,e]),
                          'excess':float(residual[e]),'mean_gradient':float(mean_g[e]),
                          'gradient_descent_score_direction':float(-mean_g[e])})
    return {'gradient_cosine':cosine_matrix(gram),'gradient_norms':np.sqrt(np.maximum(np.diag(gram),0)).tolist(),
            'aggregate_shared_item_gradient_fraction':shared_item_fraction(grad),'boundary_and_gradient':bounds},items


def trajectory(initial,target,temperatures,global_scale,steps=16,guarded=False,ns=TOP_NS):
    z=initial.detach().clone()
    first=z.cpu().numpy()
    current=snapshot(z,target,ns,temperatures)
    trace=[{'step':0,**current,'changed_users':[0]*len(ns),'entered_items':[0]*len(ns),'backtracks':0}]
    for i in range(steps):
        z=z.detach().requires_grad_(True)
        loss,_,_=calibrated_exposure_loss(z,target,ns,temperatures)
        (g,)=torch.autograd.grad(loss,z)
        rms=float(g.detach().square().mean().sqrt())
        if not np.isfinite(rms) or rms<1e-20:
            break
        lr=.02*global_scale/rms
        accepted=False
        for backtrack in range(11 if guarded else 1):
            candidate=(z-lr*(.5**backtrack)*g).detach()
            info=snapshot(candidate,target,ns,temperatures)
            # This guard uses training cohort outcomes only. It is diagnostic
            # optimizer control, NOT a new guarantee from the smooth loss.
            if not guarded or (info['soft_mean_TV']<current['soft_mean_TV']-1e-9 and
                 np.mean(info['hard_TV'])<=np.mean(current['hard_TV'])+1e-12 and
                 info['hard_TV'][1]<=current['hard_TV'][1]+1e-12):
                accepted=True
                break
        if not accepted:
            break
        z=candidate;current=info
        changed,entered=change_counts(first,z.cpu().numpy(),ns)
        trace.append({'step':i+1,**info,'changed_users':changed,'entered_items':entered,'backtracks':backtrack})
    transitions=[]
    for n_idx,n in enumerate(ns):
        ds=np.diff([r['soft_TV'][n_idx] for r in trace])
        dh=np.diff([r['hard_TV'][n_idx] for r in trace])
        moving=(np.abs(ds)>1e-8)&(np.abs(dh)>1e-8)
        transitions.append({'N':n,'transitions':len(ds),'soft_decrease_count':int(np.sum(ds < -1e-8)),
                            'soft_decrease_hard_increase_count':int(np.sum((ds < -1e-8)&(dh > 1e-8))),
                            'soft_decrease_hard_flat_count':int(np.sum((ds < -1e-8)&(np.abs(dh)<=1e-8))),
                            'nonflat_direction_comparisons':int(moving.sum()),
                            'nonflat_direction_agreement':float(np.mean(np.sign(ds[moving])==np.sign(dh[moving]))) if moving.any() else None})
    start,end=trace[0],trace[-1]
    passed=(end['soft_mean_TV']<start['soft_mean_TV']-1e-6 and
            np.mean(end['hard_TV'])<np.mean(start['hard_TV'])-1e-6 and
            end['hard_TV'][1]<start['hard_TV'][1]-1e-6 and end['changed_users'][1]>0)
    return {'trace':trace,'direction_agreement':transitions,'terminal_joint_improvement':bool(passed),
            'guarded_trajectory':guarded,'limitation':'training-matrix feasibility only; no learned model claim'}


def model_diagnostics(state,source,uids,scale,beta,temperatures,device,relative_steps,ns=TOP_NS):
    model,bundle=load_frozen(state,source,device)
    params=[p for p in model.parameters() if p.requires_grad]
    saved=[p.detach().clone() for p in params]
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    distance=torch.tensor(state.distance[uids],device=device,dtype=params[0].dtype)
    target=rule_targets(state.training_distance(uids),ns)
    try:
        with fixed_statistics(model),torch.backends.cudnn.flags(enabled=False):
            z=fuse_logits(score_rec(model,bundle,uids,len(state.q)),distance,scale,beta)
            initial=z.detach().cpu().numpy()
            _,per,_=calibrated_exposure_loss(z,target,ns,temperatures)
            grads=[]
            for i in range(len(ns)):
                raw=torch.autograd.grad(per[i],params,allow_unused=True,retain_graph=i<len(ns)-1)
                grads.append([g.detach().cpu() if g is not None else None for g in raw])
            gram=torch.zeros((len(ns),len(ns)),dtype=torch.float64)
            mean_gradient=[]
            for j,p in enumerate(params):
                col=[row[j] for row in grads]
                if all(g is None for g in col):
                    mean_gradient.append(None)
                    continue
                stack=torch.stack([g.reshape(-1).double() if g is not None else torch.zeros(p.numel(),dtype=torch.float64) for g in col])
                gram+=stack@stack.T
                mean_gradient.append(stack.mean(0).reshape(p.shape).to(device=device,dtype=p.dtype))
            norm=float(torch.sqrt(sum(g.double().square().sum() for g in mean_gradient if g is not None)))
            pnorm=float(torch.sqrt(sum(p.detach().double().square().sum() for p in params)))
            active_norm=float(torch.sqrt(sum(p.detach().double().square().sum() for p,g in zip(params,mean_gradient) if g is not None)))
            before=snapshot(torch.tensor(initial,device=device),target,ns,temperatures)
            probes=[]
            if norm>1e-30 and np.isfinite(norm):
                for relative in relative_steps:
                    with torch.no_grad():
                        for p,old,g in zip(params,saved,mean_gradient):
                            p.copy_(old if g is None else old-relative*pnorm/norm*g)
                        after=fuse_logits(score_rec(model,bundle,uids,len(state.q)),distance,scale,beta).detach()
                    changed,entered=change_counts(initial,after.cpu().numpy(),ns)
                    info=snapshot(after,target,ns,temperatures)
                    passed=(info['soft_mean_TV']<before['soft_mean_TV']-1e-6 and
                            info['hard_TV'][1]<before['hard_TV'][1]-1e-6 and
                            np.mean(info['hard_TV'])<np.mean(before['hard_TV'])-1e-6 and changed[1]>0)
                    probes.append({'relative_parameter_step':relative,**info,'changed_users':changed,'entered_items':entered,
                        'score_change_shared_item_fraction':shared_item_fraction(after.cpu().numpy()-initial),
                        'terminal_joint_improvement':bool(passed)})
            return {'users':uids.tolist(),'initial':before,'gradient_norm':norm,
                    'full_parameter_norm':pnorm,'active_parameter_norm':active_norm,
                    'gradient_cosine':cosine_matrix(gram.numpy()),'probes':probes,
                    'scope':'temporary independent checkpoint copy, restored; no optimizer or saved update'}
    finally:
        with torch.no_grad():
            for p,old in zip(params,saved):
                p.copy_(old)
        if any(not torch.equal(v,buffers[k]) for k,v in model.named_buffers()):
            raise ValueError('Frozen probe modified model buffers')
