"""Fixed-checkpoint neighborhood interventions; read-only, no repeated Ep simulation."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

from debias.io import write_json,sha256,digest
from debias.metrics import evaluate
from .training import load_checkpoint
from .diagnostics import encoding_diagnostics,query_shift,distribution


def representations(value):
    x=np.asarray(value,dtype=float)
    singular=np.linalg.svd(x,compute_uv=False);p=singular/singular.sum() if singular.sum()>0 else np.zeros_like(singular)
    effective=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))) if p.sum()>0 else 0.
    normalized=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    cosine=normalized@normalized.T;pair=cosine[np.triu_indices(len(x),1)]
    return {'effective_rank_entropy_singular_values':effective,'students':len(x),
            'pair_cosine_mean':float(pair.mean()) if len(pair) else None,
            'pair_cosine_p05':float(np.quantile(pair,.05)) if len(pair) else None,
            'pair_cosine_p95':float(np.quantile(pair,.95)) if len(pair) else None}


def audit_checkpoint(state,checkpoint,output,ctx,arrays,device='cuda'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    complete=output/'audit_complete.json'
    if complete.exists():
        receipt=__import__('json').loads(complete.read_text(encoding='utf-8'))
        if receipt['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Diagnostic checkpoint changed')
        for name,h in receipt['outputs'].items():
            if sha256(output/name)!=h:raise ValueError('Diagnostic output changed')
        return
    model,bundle,protocol,ck=load_checkpoint(state,checkpoint,device)
    if ck['method']!='C0':raise ValueError('This first controlled intervention is defined for C0 only')
    device=torch.device(device)
    items=torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)
    groups={'B_full':[],'B_no_neighbors':[],'semantic_current_C0':[]};queries=[];test_h0=[];test_h1=[]
    with torch.no_grad():
        initial=model.semantic_initials();base=model.encode_graph(protocol.base.to(device),initial=initial,diagnostics=True)
        write_json(output/'base_encoding.json',encoding_diagnostics(base,model.base.entity_type_ids))
        for user in state.test:
            uid=int(user);head=torch.tensor([bundle.entity2id[f'uid{uid}']],device=device)
            graph=protocol.query(uid).to(device)
            full=model.encode_graph(graph,initial=initial,diagnostics=True)
            no=model.encode_graph(graph,initial=initial,neighbors=False)
            sem=model.encode_graph(graph,'bypass')
            for name,encoded in (('B_full',full),('B_no_neighbors',no),('semantic_current_C0',sem)):
                groups[name].append(model.score_rec_all(head,items,encoded,bundle.relation2id['rec']).cpu().numpy()[0])
            d=full.diagnostics;e=int(head[0]);ml=d['mlkc_forward'][e]+d['mlkc_inverse'][e];ex=d['exfr_forward'][e]+d['exfr_inverse'][e]
            row={'uid':uid,'graph_id':graph.graph_id,'degree':int(graph.degree[e]),'state_edges':graph.audit,
                 'candidate_shift':query_shift(base,full,items),'h0_norm':float(initial.entities[e].norm()),
                 'h1_norm':float(full.entities[e].norm()),'delta':float((full.entities[e]-initial.entities[e]).norm()),
                 'self_norm':float(d['self'][e].norm()),'forward_norm':float(d['forward'][e].norm()),
                 'inverse_norm':float(d['inverse'][e].norm()),'mlkc_norm':float(ml.norm()),'exfr_norm':float(ex.norm()),
                 'mlkc_norm_share_not_causal':float(ml.norm()/(ml.norm()+ex.norm()+1e-8)),
                 'family_cancellation_ratio':float((ml+ex).norm()/(ml.norm()+ex.norm()+1e-8)),
                 'tanh_saturated_fraction':float((d['candidate'][e].abs()>=.99).float().mean())}
            queries.append(row);test_h0.append(initial.entities[e].cpu().numpy());test_h1.append(full.entities[e].cpu().numpy())
        train_ids=torch.tensor([bundle.entity2id[f'uid{int(u)}'] for u in state.train],device=device)
        representation_audit={'train_h0':representations(initial.entities[train_ids].cpu().numpy()),
                              'train_H1_base':representations(base.entities[train_ids].cpu().numpy()),
                              'test_h0':representations(test_h0),'test_H1_own_B':representations(test_h1),
                              'scope':'separate B graph per test row; different cohort distributions are descriptive'}
    for name,rows in groups.items():
        scores=np.asarray(rows)
        np.save(output/f'{name}.logits.npy',scores)
        result=evaluate(ctx,scores,arrays['evaluation'],baseline=np.asarray(groups['B_full']))
        write_json(output/f'{name}.metrics.json',result)
    frequency=np.zeros(len(state.q),dtype=np.int64)
    with (state.graph/'triples.txt').open(encoding='utf-8') as stream:
        for line in stream:
            h,r,t=line.split()
            if r=='rec':frequency[int(t[2:])]+=1
    item_degree=protocol.base.degree[items.cpu()].cpu().numpy()
    item_mean=np.asarray(groups['B_full']).mean(0)
    def correlation(a,b):
        if np.ptp(a)==0 or np.ptp(b)==0:return None
        value=float(spearmanr(a,b).statistic)
        return value if np.isfinite(value) else None
    write_json(output/'mechanism.json',{'representation':representation_audit,'queries':queries,
        'degree_vs_mean_logit':correlation(item_degree,item_mean),
        'train_rec_frequency_vs_mean_logit':correlation(frequency,item_mean),
        'degree_vs_rec_frequency':correlation(item_degree,frequency),
        'interpretation':'state graph degree, pseudo-label rec frequency and observed-history popularity are distinct; no causal attribution',
        'interventions':'B no-neighbors preserves query graph degree/mask/self/relations; semantic uses current C0 weights, not frozen E0',
        'Ep_repeated':False,'parameters_updated':False})
    outputs={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p!=complete}
    write_json(complete,{'checkpoint_sha256':sha256(checkpoint),'outputs':outputs})
