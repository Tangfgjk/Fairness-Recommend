"""Exploratory per-N boundary-calibrated exposure; retains the v0 sigmoid family."""
import numpy as np
import torch


class _MultiTopN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, lengths, temperatures):
        z = logits.double()
        z = (z-z.mean(1,keepdim=True)).unsqueeze(0)
        t = temperatures.double().reshape(-1,1,1)
        lengths = lengths.double().reshape(-1,1,1)
        lo = z.min(2,keepdim=True).values-50*t
        hi = z.max(2,keepdim=True).values+50*t
        for _ in range(64):
            mid=(lo+hi)/2
            mass=torch.sigmoid((z-mid)/t).sum(2,keepdim=True)
            lo=torch.where(mass>lengths,mid,lo)
            hi=torch.where(mass>lengths,hi,mid)
        w=torch.sigmoid((z-(lo+hi)/2)/t)
        ctx.save_for_backward(w*(1-w)/t)
        ctx.dtype=logits.dtype
        return w.to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (a,)=ctx.saved_tensors
        g=grad_output.double()
        den=a.sum(2,keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        gradient=(a*(g-(a*g).sum(2,keepdim=True)/den)).sum(0)
        return gradient.to(ctx.dtype),None,None


def multi_top_n(logits, ns, temperatures):
    if logits.ndim!=2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('Expected finite 2-D logits')
    if not ns or tuple(sorted(set(ns)))!=tuple(ns) or any(not isinstance(n,int) or not 0<n<logits.shape[1] for n in ns):
        raise ValueError('Ns must be unique increasing integer catalog cutoffs')
    t=torch.as_tensor(temperatures,dtype=torch.float64,device=logits.device)
    if t.shape!=(len(ns),) or not torch.isfinite(t).all() or (t<=0).any():
        raise ValueError('Expected one positive finite temperature per N')
    lengths=torch.tensor(ns,dtype=torch.float64,device=logits.device)
    return _MultiTopN.apply(logits,lengths,t)


def calibrated_exposure_loss(logits, targets, ns, temperatures):
    target=torch.as_tensor(targets,dtype=logits.dtype,device=logits.device)
    if target.shape!=(len(ns),logits.shape[1]) or not torch.isfinite(target).all() or (target<0).any():
        raise ValueError('Invalid per-N training exposure target')
    if not torch.allclose(target.sum(1),torch.ones(len(ns),device=target.device,dtype=target.dtype),atol=1e-6):
        raise ValueError('Target mass must be one per N')
    w=multi_top_n(logits,ns,temperatures)
    counts=torch.tensor(ns,dtype=logits.dtype,device=logits.device).reshape(-1,1)
    per_n=.5*(w.mean(1)/counts-target).abs().sum(1)
    return per_n.mean(),per_n,w


def calibrate_temperatures(training_logits, ns, floor_ratio=1e-4):
    """Training cohort ONLY; freeze returned scales for every diagnostic batch."""
    z=np.asarray(training_logits,dtype=float)
    if z.ndim!=2 or len(z)==0 or not np.isfinite(z).all() or not 0<floor_ratio<1:
        raise ValueError('Invalid training-only temperature calibration')
    if not ns or any(not 0<n<z.shape[1] for n in ns):
        raise ValueError('Invalid rank boundary')
    centered=z-z.mean(1,keepdims=True)
    iqr=max(float(np.quantile(centered,.75)-np.quantile(centered,.25)),1e-3)
    floor=max(iqr*floor_ratio,1e-6)
    ordered=np.sort(z,axis=1)[:,::-1]
    rows=[]
    for n in ns:
        gaps=ordered[:,n-1]-ordered[:,n]
        positive=gaps[gaps>0]
        raw=float(np.median(positive)) if len(positive) else 0.
        rows.append({'N':n,'median_positive_boundary_gap':raw,'zero_gap_fraction':float(np.mean(gaps==0)),
                     'temperature':max(raw,floor)})
    return {'calibration':'per-N median positive N/N+1 logit gap across all training users',
            'fixed_multiplier':1.,'training_users':len(z),'global_IQR':iqr,'temperature_floor':floor,
            'per_N':rows}
