"""Read-only checkpoint probes and E2-logit evaluation; never creates an optimizer."""
import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT,DATASETS,read_json,write_json,sha256,model_modules
from debias.prepare import prepared,baseline_folder,ARTIFACTS
from debias.context import load_scores
from debias.pair_matching import MatchConfig,audit_pairs,exposure_audit
from debias.training import build_model,rng_state,restore_rng
from debias.objectives import FairObjective,matched_hinge
from debias.scoring import pair_logits
from debias.evaluation import write_score_matrix,evaluate_file,frequency_correct

RUN=ROOT/'runs/merit_frequency_20261022'
OUT=ROOT/'runs/论文格式文档统计/merit_frequency_20261022/机制诊断_20261023'


def correlation(a,b):
    def rank(v):
        _,inv,count=np.unique(v,return_inverse=True,return_counts=True)
        return (np.cumsum(count)-(count-1)/2)[inv]
    x,y=rank(np.asarray(a)),rank(np.asarray(b))
    return float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>0 else None


def effects(matrix):
    z=np.asarray(matrix,dtype=float)
    grand=z.mean(); item=z.mean(0)-grand; user=z.mean(1)-grand
    residual=z-grand-user[:,None]-item[None,:]
    iss=float(len(z)*np.square(item).sum()); rss=float(np.square(residual).sum())
    return {'item_SS':iss,'interaction_SS':rss,'item_fraction_after_removing_user_offsets':iss/(iss+rss) if iss+rss else None,
            'row_centered_std':float((z-z.mean(1,keepdims=True)).std()),
            'scope':'Full catalog descriptive variance; neither causal contribution nor Top-N variance'}


def verified_scores(ctx,method):
    folder=RUN/ctx.name/method/'seed2024'
    result=read_json(folder/'eval/metrics.json')
    marker=read_json(folder/'eval/evaluation_complete.json')
    if sha256(folder/'eval/metrics.json')!=marker['metrics_sha256']:
        raise ValueError('Changed original evaluation')
    score=load_scores(folder/'uid_ex_scores.json',ctx)
    if sha256(folder/'uid_ex_scores.json')!=result['provenance']['scores_sha256']:
        raise ValueError('Changed original scores')
    return folder,result,score


def decompose(ctx,arrays,folder):
    original=model_modules()
    sources={}; zs={}; bias={}; details={}
    for method in ['E0','E1']:
        path,result,score=verified_scores(ctx,method)
        z=np.load(path/'uid_ex_scores.logits.npy').astype(float)
        if z.shape!=score.shape or not np.isfinite(z).all() or not np.allclose(1/(1+np.exp(-z)),score,atol=2e-7,rtol=2e-6):
            raise ValueError('Logit rows do not match official UID-ordered probabilities')
        ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
        from feature_loader import read_dict
        ent=read_dict(ctx.graph/'entities.dict')
        bias[method]=ck['model_state_dict']['b'][[ent[f'ex{e}'] for e in range(len(ctx.q))]].numpy().astype(float)
        zs[method]=z
        details[method]={'decomposition':effects(z),'mean_logit_rec_frequency_spearman':correlation(z.mean(0),ctx.frequency),
                         'mean_logit_mean_education_distance_spearman':correlation(z.mean(0),ctx.distance[ctx.test].mean(0)),
                         'checkpoint_epoch':ck['epoch'],'high_exposure_pair_coverage':exposure_audit(ctx,score,arrays['E1'])}
        sources[method]={p.name:sha256(p) for p in [path/'best.pt',path/'uid_ex_scores.json',path/'uid_ex_scores.logits.npy']}
        del ck
    delta=zs['E1']-zs['E0']; db=bias['E1']-bias['E0']
    details['score_change']={'decomposition':effects(delta),'after_subtracting_explicit_bias_change':effects(delta-db[None,:]),
                             'mean_shift_rec_frequency_spearman':correlation(delta.mean(0),ctx.frequency),
                             'bias_change_mean_shift_spearman':correlation(db,delta.mean(0)),
                             'explicit_bias_change_RMS':float(np.sqrt(np.mean(db**2))),
                             'total_logit_change_RMS':float(np.sqrt(np.mean(delta**2)))}
    archive=read_json(ARTIFACTS/'debias_context'/ctx.name/'baseline_seed2024.json')
    base=read_json(RUN/ctx.name/'E0/seed2024/eval/metrics.json')
    keys=['epochs','bs','learning_rate','negative_ratio','embedding_dim','embedding_shape1','hidden_size','input_drop','hidden_drop','feat_drop','ablation','deterministic','include_test_triples','use_bias','seed']
    a=read_json(baseline_folder(ctx.name,2024)/'config.json');b=read_json(RUN/ctx.name/'E0/seed2024/config.json')
    details['baseline_audit']={'config_differences':{k:[a.get(k),b.get(k)] for k in keys if a.get(k)!=b.get(k)},
                               'archive_best_epoch':int(torch.load(baseline_folder(ctx.name,2024)/'best.pt',map_location='cpu',weights_only=False)['epoch']),
                               'retrained_best_epoch':details['E0']['checkpoint_epoch'],
                               'N5_archive_Ada':archive['top_n']['5']['Ada'],'N5_retrained_Ada':base['top_n']['5']['Ada'],
                               'N5_archive_overlap':base['top_n']['5']['BaselineOverlap'],
                               'limits':'Matching configs do not establish identical hardware, numerical execution or full training trajectory.'}
    with (folder/'item_score_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['exercise','train_rec_frequency','mean_logit_E0','mean_logit_E1','mean_shift','explicit_bias_change'])
        w.writerows([e,int(ctx.frequency[e]),float(zs['E0'][:,e].mean()),float(zs['E1'][:,e].mean()),float(delta[:,e].mean()),float(db[e])] for e in range(len(ctx.q)))
    details['sources']=sources
    write_json(folder/'score_decomposition.json',details)
    return zs


def pair_audit(ctx,arrays,manifest,folder):
    cfg=MatchConfig(**manifest['contract']['matching'])
    e1={tuple(p) for p in arrays['E1']}
    per=[]
    for u in np.unique(arrays['E3'][:,0]):
        p=arrays['E3'][arrays['E3'][:,0]==u]
        n=sum(tuple(v) in e1 for v in p)
        per.append({'uid':int(u),'E3_pairs':len(p),'also_matched':n,'matched_fraction':n/len(p)})
    result={'E3_matched_pair_fraction':sum(p['also_matched'] for p in per)/len(arrays['E3']),
            'E3_student_macro_matched_fraction':float(np.mean([p['matched_fraction'] for p in per])),
            'per_user':per,'audit':{k:audit_pairs(ctx,arrays[k],cfg) for k in ['E1','E3']}}
    write_json(folder/'E3_overlap_audit.json',result)


def group(name):
    if name=='b':return 'item_bias'
    if name.startswith('uid_feature_only_fusion'):return 'student_encoder'
    if name.startswith('exercise_feature_only_fusion'):return 'exercise_encoder'
    if name.startswith('shared_text_encoder'):return 'shared_text_encoder'
    if name.split('.')[0] in ['conv1','fc','bn0','bn1','bn2']:return 'ConvE'
    return 'other_shared_parameters'


def compare_gradients(named,bce,fair,weight):
    groups=sorted({group(n) for n,_ in named})+['ALL']
    out={}
    for g in groups:
        s0=s1=dot=0.; nb=nf=0
        for (n,_),a,b in zip(named,bce,fair):
            if g!='ALL' and group(n)!=g:continue
            if a is not None:s0+=float(a.double().square().sum());nb+=1
            if b is not None:s1+=float(b.double().square().sum())*weight**2;nf+=1
            if a is not None and b is not None:dot+=float((a.double()*b.double()).sum())*weight
        out[g]={'BCE_norm':s0**.5,'weighted_fair_norm':s1**.5,
                'weighted_fair_over_BCE':(s1/s0)**.5 if s0>1e-24 else None,
                'cosine':dot/(s0*s1)**.5 if s0*s1>1e-24 else None,'BCE_used_tensors':nb,'fair_used_tensors':nf}
    return out


def gradient_probe(ctx,arrays,folder,batches=2):
    original=model_modules();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle=original.load_semantic_feature_bundle(ctx.graph,None,device=device)
    triples=original.read_triples(ctx.graph/'triples.txt',bundle.entity2id,bundle.relation2id)
    data=original.TripleDataset(triples,bundle.id2relation,bundle.exercise_entity_ids.cpu().tolist(),original.build_positive_tails_by_hr(triples),5,2024)
    ix=np.random.default_rng(20261023).choice(len(data),size=(batches,1024),replace=False)
    global_rng=rng_state();results=[]
    try:
        for method in ['E0','E1']:
            path=RUN/ctx.name/method/'seed2024'
            # Last checkpoints align the probes at epoch25; exported ranking uses best.
            ck=torch.load(path/'last.pt',map_location=device,weights_only=False)
            model=build_model(original,bundle,ck['config']).to(device)
            for batch,indices in enumerate(ix):
                original.set_seed(20261023+batch)
                model.load_state_dict(ck['model_state_dict'],strict=True);model.train()
                columns=list(zip(*(data[int(i)] for i in indices)))
                h,r,t,labels=[torch.stack(col).to(device) for col in columns]
                named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
                pred=model.score_triples(h,r,t)
                loss=F.binary_cross_entropy(pred.view(-1),labels.view(-1))
                bg=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
                bg=[v.detach().cpu() if v is not None else None for v in bg]
                objective=FairObjective(arrays['E1'],bundle,2024+batch,margin=.1)
                fair,stats=objective(model,device)
                fg=torch.autograd.grad(fair,[p for _,p in named],allow_unused=True)
                fg=[v.detach().cpu() if v is not None else None for v in fg]
                results.append({'method':method,'epoch':ck['epoch'],'batch':batch,'BCE_loss':float(loss.detach()),
                                'fair_loss':float(fair.detach()),'stats':stats,'groups':compare_gradients(named,bg,fg,.1)})
                del bg,fg
            # Inspect endpoints independently, without a parameter update.
            p=arrays['E1'][:128]
            make=lambda v:torch.tensor(v,dtype=torch.long,device=device)
            h=make([bundle.entity2id[f'uid{u}'] for u in p[:,0]])
            hi=make([bundle.entity2id[f'ex{e}'] for e in p[:,1]])
            lo=make([bundle.entity2id[f'ex{e}'] for e in p[:,2]])
            hz,lz=pair_logits(model,h,torch.full_like(h,bundle.relation2id['rec']),hi,lo)
            loss=matched_hinge(hz,lz,make(p[:,0]),.1)
            gh,gl=torch.autograd.grad(loss,(hz,lz))
            results.append({'method':method,'endpoint_probe':{'pairs':len(p),'active_pairs':int(((hz-lz)>.1).sum()),
                            'high_gradient_norm':float(gh.norm()),'low_gradient_norm':float(gl.norm()),
                            'opposite_gradients':bool(torch.allclose(gh,-gl))}})
            del model,ck;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    finally:restore_rng(global_rng)
    write_json(folder/'gradient_probes.json',{'device':str(device),'batch_size':1024,'batches':batches,'weight':.1,'margin':.1,
             'scope':'Checkpoint epoch25 fixed-batch probes, not historical training gradients; no optimizer/updates.',
             'batch_index_sha256':__import__('hashlib').sha256(ix.tobytes()).hexdigest(),'results':results})


def logit_control(ctx,arrays,manifest,z,folder):
    _,base,prob=verified_scores(ctx,'E0')
    corrected=frequency_correct(z,ctx.frequency,.1)
    output=folder/'E2_logit_alpha0p1'
    output.mkdir(parents=True,exist_ok=True)
    path=output/'uid_ex_scores.json'
    write_score_matrix(path,ctx,corrected)
    write_json(output/'correction_config.json',{'scale':'raw saved logit','strength':.1,'feature':'minmax(log1p(train_rec_unique_users))',
               'caveat':'0.1 logit and 0.1 probability are unequal interventions; no test-set strength search',
               'alpha0_Top10_overlap_with_probability':float(np.mean([len(set(a)&set(b))/10 for a,b in zip(np.argsort(-z,axis=1,kind='stable')[:,:10],np.argsort(-prob,axis=1,kind='stable')[:,:10])]))})
    evaluate_file(ctx,arrays,manifest,path,output/'eval',baseline=RUN/ctx.name/'E0/seed2024/uid_ex_scores.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',default=','.join(DATASETS))
    parser.add_argument('--gradient-batches',type=int,default=2)
    args=parser.parse_args()
    if args.gradient_batches < 1:
        parser.error('--gradient-batches must be positive')
    for d in args.datasets.split(','):
        if d not in DATASETS:parser.error(d)
        folder=OUT/d;folder.mkdir(parents=True,exist_ok=True)
        print(f'{d}: preparing and decomposing',flush=True)
        ctx,arrays,manifest=prepared(d)
        zs=decompose(ctx,arrays,folder);pair_audit(ctx,arrays,manifest,folder)
        print(f'{d}: fixed-batch gradient probes; no updates',flush=True)
        gradient_probe(ctx,arrays,folder,args.gradient_batches)
        print(f'{d}: E2-logit evaluation',flush=True)
        logit_control(ctx,arrays,manifest,zs['E0'],folder)
        write_json(folder/'complete.json',{'script_sha256':sha256(Path(__file__)),'dataset_hash':ctx.fingerprint,'seed':2024})
        print(f'{d}: complete',flush=True)
    print(OUT)


if __name__=='__main__':main()
