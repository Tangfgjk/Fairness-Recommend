"""Training-only mechanism monitoring; never selects temperatures or checkpoints."""
import hashlib
import numpy as np
import torch
from scipy.stats import spearmanr

from debias.scoring import fixed_statistics
from .scoring import score_rec
from .exposure import hard_exposure, rule_targets
from .boundary_exposure import calibrated_exposure_loss


MONITOR_POLICY = {
    'users': 'fixed uniform training subset, at most 48; same seed across methods/epochs',
    'gradient_batches': 'first, middle (lower median), last actual rec-containing batch each epoch',
    'temperature_warning': 'gap/temperature changes >4x from initialization OR TopN soft mass <0.80',
    'warning_scope': 'exploratory diagnostic flags only; no training/selection/temperature changes',
    'Q_scope': 'same-parameter core vs fused ablation; cancellation association is not causal evidence',
}


def _correlation(a, b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None


def _slope(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    x=x-x.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True)
    den=float(np.square(x).sum())
    return float((x*y).sum()/den) if den>1e-30 else None


def score_audit(core,reward,frequency,target,ns,temperatures,initial_core=None,initial_ratios=None):
    """Summaries use common users/catalog; no test labels and no parameter updates."""
    # Add in the scorer's native dtype before using float64 for statistics.
    core=np.asarray(core);reward=np.asarray(reward);frequency=np.asarray(frequency)
    fused=np.asarray(core+reward,float)
    core=core.astype(float);reward=reward.astype(float)
    if core.shape!=reward.shape or frequency.shape!=(core.shape[1],):raise ValueError('Audit shapes differ')
    z=torch.as_tensor(fused,dtype=torch.float64)
    with torch.no_grad():soft,per,weights=calibrated_exposure_loss(z,target,ns,temperatures)
    weights=weights.numpy()
    rank=np.argsort(-fused,axis=1,kind='stable');raw_rank=np.argsort(-core,axis=1,kind='stable')
    ordered=np.take_along_axis(fused,rank,axis=1)
    hard=hard_exposure(fused,ns);core_hard=hard_exposure(core,ns)
    rows=[]
    for i,n in enumerate(ns):
        gaps=ordered[:,n-1]-ordered[:,n]
        ratio=float(np.median(gaps)/temperatures[i])
        start=initial_ratios[i] if initial_ratios is not None else ratio
        relative=ratio/start if start>1e-30 else None
        mass=float(np.take_along_axis(weights[i],rank[:,:n],axis=1).sum(1).mean()/n)
        overlap=np.array([len(set(a[:n])&set(b[:n]))/n for a,b in zip(rank,raw_rank)])
        drift=(relative is not None and (relative<.25 or relative>4)) or (start<=1e-30 and ratio>1e-30)
        rows.append({'N':n,'temperature':float(temperatures[i]),'median_boundary_gap':float(np.median(gaps)),
                     'median_gap_over_temperature':ratio,'ratio_relative_to_initialization':relative,
                     'zero_boundary_gap_fraction':float(np.mean(gaps==0)),
                     'actual_TopN_soft_mass_fraction':mass,'core_fused_set_overlap':float(overlap.mean()),
                     'Q_changed_set_users':int(np.sum(overlap<1-1e-12)),
                     'Q_changed_order_users':int(np.sum(np.any(rank[:,:n]!=raw_rank[:,:n],axis=1))),
                     'core_hard_DED':float(.5*np.abs(core_hard[i]-target[i]).sum()),
                     'fused_hard_DED':float(.5*np.abs(hard[i]-target[i]).sum()),
                     'temperature_drift_flag':bool(drift),'soft_hard_mass_flag':mass<.80,
                     'mechanism_warning':bool(drift or mass<.80)})
    correlations={}
    for name,values in (('core',core),('Q_reward',reward),('fused',fused)):
        mean=values.mean(0)
        correlations[name]={'item_mean_vs_train_rec_frequency_spearman':_correlation(mean,frequency),
                            'item_mean_vs_positive_rec_frequency_spearman':_correlation(mean[frequency>0],frequency[frequency>0]),
                            'mean':float(values.mean()),'std':float(values.std()),
                            'within_student_std':float(values.std(1).mean())}
    delta=core-initial_core if initial_core is not None else np.zeros_like(core)
    centered_reward=reward-reward.mean(1,keepdims=True)
    centered_delta=delta-delta.mean(1,keepdims=True)
    return {'soft_mean_TV':float(soft),'soft_TV':per.tolist(),
            'hard_TV':(.5*np.abs(hard-target).sum(1)).tolist(),
            'boundary_per_N':rows,'score_components':correlations,
            'Q_cancellation':{'core_on_reward_within_student_slope':_slope(reward,core),
                              'core_change_on_reward_within_student_slope':_slope(reward,delta),
                              'core_change_reward_centered_spearman':_correlation(centered_delta,centered_reward),
                              'interpretation':'change slope near -1 is cancellation-compatible, not proof; inspect Q ranking increments'},
            'mechanism_warning_Ns':[r['N'] for r in rows if r['mechanism_warning']]}


class TrainingMonitor:
    def __init__(self,state,scorer,assets,seed,triples):
        self.state=state;self.scorer=scorer;self.ns=tuple(assets['top_ns']);self.temperatures=assets['temperatures']
        self.users=np.sort(np.random.default_rng(seed+820001).choice(state.train,min(48,len(state.train)),replace=False))
        distances=state.training_distance(self.users)
        self.target=rule_targets(distances,self.ns)
        self.frequency=np.zeros(len(state.q),dtype=np.int64)
        seen=set()
        for h,r,t in triples:
            if r!=scorer.rec:continue
            u=int(scorer.user_index[h]);e=int(scorer.item_index[t])
            if u not in state.train or e<0 or (u,e) in seen:raise ValueError('Invalid training rec frequency source')
            seen.add((u,e));self.frequency[e]+=1
        self.initial_core=None;self.initial_ratios=None

    def evaluate(self,model):
        with fixed_statistics(model),torch.no_grad():
            raw=score_rec(model,self.scorer.bundle,self.users,len(self.state.q))
            reward=self.scorer.beta*self.scorer.scale.transform(self.scorer.distance[self.users].to(raw.dtype)) if self.scorer.beta else torch.zeros_like(raw)
            core=raw.cpu().numpy();reward=reward.cpu().numpy()
        result=score_audit(core,reward,self.frequency,self.target,self.ns,self.temperatures,
                           self.initial_core,self.initial_ratios)
        core_rank=np.argsort(-core,axis=1,kind='stable')
        fused_rank=np.argsort(-(core+reward),axis=1,kind='stable')
        distances=self.state.training_distance(self.users)
        for row,n in zip(result['boundary_per_N'],self.ns):
            row['core_mean_educational_distance']=float(np.take_along_axis(distances,core_rank[:,:n],axis=1).mean())
            row['fused_mean_educational_distance']=float(np.take_along_axis(distances,fused_rank[:,:n],axis=1).mean())
        if self.initial_core is None:
            self.initial_core=core.copy()
            self.initial_ratios=[r['median_gap_over_temperature'] for r in result['boundary_per_N']]
        return {'training_users':self.users.tolist(),'test_data_used':False,
                'train_rec_frequency':self.frequency.tolist(),'policy':MONITOR_POLICY,**result}


class MonitoredBatches:
    """Same sampler/order; find actual first/middle/last eligible batches without a second draw."""
    def __init__(self,batches,dataset,rec_id):
        self.batches=batches;self.targets={}
        self.rec=np.array([dataset.triples[i][1]==rec_id for i,_ in dataset.sample_index])

    def __len__(self):return len(self.batches)

    def __iter__(self):
        batches=list(self.batches)
        eligible=[i for i,indices in enumerate(batches) if self.rec[indices].any()]
        self.targets={}
        if eligible:
            for role,i in zip(('first','middle','last'),(eligible[0],eligible[(len(eligible)-1)//2],eligible[-1])):
                self.targets.setdefault(i,[]).append(role)
        yield from batches


def gradient_balance(model,bce,aux,weight):
    params=[p for p in model.parameters() if p.requires_grad]
    bg=torch.autograd.grad(bce,params,retain_graph=True,allow_unused=True)
    eg=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
    bn2=sum(float(g.detach().double().square().sum()) for g in bg if g is not None)
    en2=sum(float(g.detach().double().square().sum()) for g in eg if g is not None)
    dot=sum(float((b.detach().double()*e.detach().double()).sum()) for b,e in zip(bg,eg) if b is not None and e is not None)
    bn=bn2**.5;en=en2**.5
    cosine=float(np.clip(dot/(bn*en),-1,1)) if bn*en>1e-30 else None
    return {'BCE_gradient_norm':bn,'weighted_exposure_gradient_norm':weight*en,
            'weighted_exposure_over_BCE':weight*en/bn if bn>1e-30 else None,
            'BCE_exposure_gradient_cosine':cosine,
            'BCE_exposure_gradient_angle_degrees':float(np.degrees(np.arccos(cosine))) if cosine is not None else None}


def initialization_hash(model):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode());value.update(str(array.dtype).encode());value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()
