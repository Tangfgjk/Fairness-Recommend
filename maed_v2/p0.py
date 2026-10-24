"""Bounded engineering acceptance only. No epochs, checkpoints, or test tuning."""
import copy
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from debias.io import ROOT, model_modules, read_json, write_json, sha256, digest, TOP_NS
from debias.training import build_model, rng_state, restore_rng
from maed.state import load_state
from maed.training_monitor import initialization_hash
from .graph_protocol import GraphProtocol
from .model import SemanticCompGCN, stable_top_n
from .diagnostics import gradient_groups, encoding_diagnostics, query_shift, distribution


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device); start=time.perf_counter(); result=function(); synchronize(device)
    return result,time.perf_counter()-start


def memory(device):
    if device.type != 'cuda': return {"device":"cpu","peak_allocated_bytes":None,"peak_reserved_bytes":None}
    free,total=torch.cuda.mem_get_info(device)
    return {"device":str(device),"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),"current_free_bytes":free,
            "total_bytes":total,"scope":"PyTorch allocator peaks; excludes other apps/driver allocations"}


def compare(a,b,atol=1e-5,rtol=1e-4):
    if a.shape!=b.shape: raise AssertionError('Comparison shapes differ')
    delta=(a-b).abs();limit=atol+rtol*b.abs()
    return {"passed":bool((delta<=limit).all()),"max_abs_error":float(delta.max()) if delta.numel() else 0.,
            "outside_tolerance":int((delta>limit).sum()),"atol":atol,"rtol":rtol}


def first_batch(original,bundle,state,cfg,device):
    triples=original.read_triples(state.graph/'triples.txt',bundle.entity2id,bundle.relation2id)
    ds=original.TripleDataset(triples,bundle.id2relation,bundle.exercise_entity_ids.cpu().tolist(),
                             original.build_positive_tails_by_hr(triples),cfg['negative_ratio'],2024)
    loader=DataLoader(ds,batch_size=cfg['bs'],shuffle=True)
    batch=tuple(x.to(device) for x in next(iter(loader)))
    if not (batch[1]==bundle.relation2id['rec']).any():
        raise ValueError('Fixed engineering batch must contain rec training examples')
    return batch


def profile_training(model,graph,batch,initial_state,initial_rng,cfg,chunks,warmup,measured,output):
    """Identical first batch, initialization, Adam, RNG for each chunk candidate."""
    device=model.base.b.device; rows=[]; reference=None
    for chunk in chunks:
        model.load_state_dict(initial_state);model.layer.chunk_size=chunk;model.train()
        restore_rng(initial_rng);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        durations=[];first=None
        print(f'P0 profile chunk={chunk}: {warmup} warmup + {measured} measured updates (temporary model)',flush=True)
        try:
            for step in range(warmup+measured):
                if step==warmup and device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                synchronize(device);began=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                encoded=model.encode_graph(graph)
                logits=model.score_logits(*batch[:3],encoded)
                loss=torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits),batch[3])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite BCE')
                loss.backward()
                if step==0:
                    first={'logits':logits.detach().cpu(),'loss':loss.item(),
                           'gradients':{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None},
                           'gradient_groups':gradient_groups(model)}
                optimizer.step();synchronize(device)
                duration=time.perf_counter()-began
                if step>=warmup:durations.append(duration)
                del encoded,logits,loss
            if reference is None:
                reference=first;equivalence={'passed':True,'reference_chunk':chunk,'scope':'reference itself'}
            else:
                grad={n:compare(first['gradients'][n],reference['gradients'][n]) for n in reference['gradients']}
                if set(first['gradients'])!=set(reference['gradients']): raise AssertionError('Gradient presence changed')
                equivalent_logits=compare(first['logits'],reference['logits'])
                equivalence={'passed':equivalent_logits['passed'] and all(r['passed'] for r in grad.values()),
                             'logits':equivalent_logits,'gradients':grad,'reference_chunk':chunks[0]}
            row={'chunk_size':chunk,'status':'passed' if equivalence['passed'] else 'equivalence_failed',
                 'mean_step_seconds':float(np.mean(durations)), 'step_seconds':durations,
                 'equivalence':equivalence,'first_BCE':first['loss'],
                 'first_gradient_groups':first['gradient_groups'],**memory(device)}
        except torch.cuda.OutOfMemoryError as error:
            # Clear failed autograd references before trying the next engineering chunk.
            optimizer.zero_grad(set_to_none=True)
            for name in ('encoded','logits','loss'):
                if name in locals():
                    if name=='encoded': encoded=None
                    elif name=='logits':logits=None
                    else:loss=None
            row={'chunk_size':chunk,'status':'out_of_memory','error':str(error),**memory(device)}
        rows.append(row);write_json(output/'chunk_profile.json',rows)
        del optimizer,first;gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    passed=[r for r in rows if r['status']=='passed']
    if not passed: raise RuntimeError('No chunk passed GPU memory/equivalence checks')
    chosen=min(passed,key=lambda r:r['mean_step_seconds'])['chunk_size']
    model.load_state_dict(initial_state);model.zero_grad(set_to_none=True)
    model.layer.chunk_size=chosen;model.eval()
    return {'chosen_chunk':chosen,'policy':'fastest measured complete step among equivalent fitting candidates; training batch only',
            'warmup_steps':warmup,'measured_steps':measured,'rows':rows}


def original_regression(model,protocol,bundle,state,output):
    """Use REAL E0 checkpoint, including text encoder, decoder BN and tail bias."""
    source=ROOT/'runs/merit_frequency_20261022'/state.dataset/'E0/seed2024/best.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    before=copy.deepcopy(model.state_dict())
    model.base.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
    device=model.base.b.device;g=protocol.base.to(device)
    users=torch.tensor([bundle.entity2id[f'uid{u}'] for u in state.test[:2]],device=device)
    items=torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)
    r=torch.zeros_like(users)+bundle.relation2id['rec']
    with torch.no_grad():
        snapshot=model.encode_graph(g,'bypass')
        a=model.score_logits(users,r,items[:len(users)],snapshot)
        b=model.base.score_logits(users,r,items[:len(users)])
        z=model.score_rec_all(users,items,snapshot,bundle.relation2id['rec'])
        x=model.base.conve_transform(model.base.entity_embedding(users),model.base.relation_embedding(r))
        old_z=x@model.base.entity_embedding(items).T+model.base.b[items].unsqueeze(0)
        report={'source':str(source.relative_to(ROOT)),'checkpoint_sha256':sha256(source),
                'triple_logit':compare(a,b,0,0),'catalog_logit':compare(z,old_z,0,0),
                'catalog_sigmoid':compare(torch.sigmoid(z),model.base.score_tails(users,r,items),0,0)}
    model.load_state_dict(before)
    if not all(report[k]['passed'] for k in ('triple_logit','catalog_logit','catalog_sigmoid')):
        raise AssertionError('Original zero-layer decoder regression failed')
    write_json(output/'original_regression.json',report)
    return report


def inference_probe(model,protocol,bundle,state,output,query_limit):
    """Full B is authoritative. Local reference may only pass after rank agreement."""
    model.eval();device=model.base.b.device
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    items=torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)
    queries=state.test if query_limit==0 else state.test[:query_limit]
    records=[];first_graph=None;first_logits=None
    parameter_hash=initialization_hash(model)
    synchronize(device);cohort_start=time.perf_counter()
    with torch.no_grad():
        initial,semantic_time=timed(device,model.semantic_initials)
        base_graph,base_transfer=timed(device,lambda:protocol.base.to(device))
        base,base_time=timed(device,lambda:model.encode_graph(base_graph,initial=initial))
        for u in queries:
            uid=int(u);users=torch.tensor([bundle.entity2id[f'uid{uid}']],device=device)
            graph,build_time=timed(device,lambda:protocol.query(uid).to(device))
            full,encode_time=timed(device,lambda:model.encode_graph(graph,initial=initial))
            logits,score_time=timed(device,lambda:model.score_rec_all(users,items,full,bundle.relation2id['rec']))
            ranking,rank_time=timed(device,lambda:stable_top_n(logits,max(TOP_NS)))
            # Local reference does not use base H1 as sources. It recomputes all
            # receiver aggregates for {query + all exercises} with FULL new degrees.
            local,local_time=timed(device,lambda:model.encode_graph(graph,'local',initial=initial,
                                  output_ids=torch.cat((users,items))))
            local_z=model.score_rec_all(users,items,local,bundle.relation2id['rec'])
            emb=compare(local.entities[torch.cat((users,items))],full.entities[torch.cat((users,items))])
            logit_eq=compare(local_z,logits)
            top={str(n):bool(torch.equal(stable_top_n(local_z,n),ranking[:,:n])) for n in TOP_NS}
            # Cross-query scoring MUST fail before any decoder call.
            other=int(next(v for v in state.test if int(v)!=uid))
            try:
                model.score_rec_all(torch.tensor([bundle.entity2id[f'uid{other}']],device=device),items,full,bundle.relation2id['rec'])
            except ValueError:
                cross_query_rejected=True
            else: raise AssertionError('Cross-query B scoring was allowed')
            # Triplet and catalog must decode the same graph and the same relation.
            pair=model.score_logits(users.expand(len(items)),torch.full_like(items,bundle.relation2id['rec']),items,full)
            pair_eq=compare(pair,logits[0])
            if not pair_eq['passed']: raise AssertionError('Triple/catalog score mismatch')
            a=model.score_rec_all(users,items,base,bundle.relation2id['rec'])
            record={'uid':uid,'graph_id':graph.graph_id,'base_id':graph.base_id,
                    'query_audit':graph.audit,'degree_hash':digest(graph.degree.cpu().tolist()),
                    'parameters_sha256':parameter_hash,'no_gradient_inference':True,
                    'B_graph_build_transfer_seconds':build_time,'B_full_encode_seconds':encode_time,
                    'B_catalog_score_seconds':score_time,'B_stable_rank_seconds':rank_time,
                    'local_reference_encode_seconds':local_time,
                    'local_equivalence':{'entities':emb,'logits':logit_eq,'TopN':top,
                                         'accepted':emb['passed'] and logit_eq['passed'] and all(top.values())},
                    'formal_output':'full_reference_even_if_local_passes',
                    'cross_query_rejected':cross_query_rejected,'triple_catalog':pair_eq,
                    'A_B_score_change':distribution(logits-a),'candidate_shift':query_shift(base,full,items),
                    'B_Top50':ranking.cpu().tolist()[0],
                    'A_Top50':stable_top_n(a,max(TOP_NS)).cpu().tolist()[0]}
            records.append(record)
            if first_graph is None:
                first_graph=graph;first_logits=logits.detach().clone()
                diagnostic=model.encode_graph(graph,initial=initial,diagnostics=True)
                write_json(output/'query_encoding_diagnostics.json',encoding_diagnostics(diagnostic,model.base.entity_type_ids))
                # Save values for an independent check, not as formal experiment scores.
                np.savez_compressed(output/'untrained_query_example.npz',uid=uid,logits=logits.cpu().numpy(),
                    A_logits=a.cpu().numpy(),degree=graph.degree.cpu().numpy())
                del diagnostic
            if len(records)<=2 or len(records)%50==0 or len(records)==len(queries):
                print(f'P0 B query {len(records)}/{len(queries)} (uid={uid}): full {encode_time:.3f}s; local accepted={record["local_equivalence"]["accepted"]}',flush=True)
            del local,full,logits,local_z
        # Re-run first query AFTER all other queries to detect accumulated state.
        repeat=model.encode_graph(protocol.query(int(queries[0])).to(device),initial=initial)
        repeat_z=model.score_rec_all(torch.tensor([first_graph.query_entity],device=device),items,repeat,bundle.relation2id['rec'])
        # CUDA index_add uses atomic reductions. Independent identical graphs may
        # differ in roundoff; use the specified GPU tolerance, never bit equality.
        tolerance=(0.,0.) if device.type=='cpu' else (1e-5,1e-4)
        isolation=compare(repeat_z,first_logits,*tolerance)
        isolation['TopN_identical']={str(n):bool(torch.equal(stable_top_n(repeat_z,n),stable_top_n(first_logits,n))) for n in TOP_NS}
        isolation['same_graph_id']=repeat.graph.graph_id==first_graph.graph_id
        isolation['interpretation']='Identical graph/query, no accumulated edges; GPU atomic reduction roundoff reported separately'
        if not isolation['passed']:raise AssertionError('Query state accumulated between calls')
        base_diag=model.encode_graph(base_graph,initial=initial,diagnostics=True)
        write_json(output/'base_encoding_diagnostics.json',encoding_diagnostics(base_diag,model.base.entity_type_ids))
    synchronize(device)
    report={'status':'passed','model_state':'fresh initialization, engineering only; not C0-trained',
            'semantic_prepare_seconds':semantic_time,'base_transfer_seconds':base_transfer,
            'base_encode_seconds':base_time,'queries_measured':len(queries),'total_test_students':len(state.test),
            'full_cohort':len(queries)==len(state.test),
            'instrumented_validation_wall_seconds':time.perf_counter()-cohort_start,
            'timing_scope':'synchronized phase measurements + assertions/diagnostics; NOT production throughput',
            'cache_policy':'one fixed-parameter semantic H0/R0; one base H1 for A; B re-encodes each independent graph',
            'query_isolation_repeat':isolation,'query_records':records,**memory(device)}
    write_json(output/'inference_probe.json',report)
    return report


def cohort_cost(model,protocol,bundle,state,output):
    """Uninstrumented full-cohort wall clock; own clean shared semantic preparation.

    Validation above has already compared the reference/local outputs. Here ONLY
    full reference inference runs. CUDA phases are not synchronized individually.
    Query graph source checks were paid by the caller and are explicitly excluded.
    """
    model.eval();device=model.base.b.device
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    items=torch.tensor([bundle.entity2id[f'ex{e}'] for e in range(len(state.q))],device=device)
    params=initialization_hash(model)
    outputs={};result={}
    with torch.no_grad():
        for label in ('A','B'):
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            synchronize(device);began=time.perf_counter()
            initial=model.semantic_initials()
            if label=='A':
                encoding=model.encode_graph(protocol.base.to(device),initial=initial)
            rows=[]
            for value in state.test:
                u=int(value);users=torch.tensor([bundle.entity2id[f'uid{u}']],device=device)
                if label=='B':encoding=model.encode_graph(protocol.query(u).to(device),initial=initial)
                z=model.score_rec_all(users,items,encoding,bundle.relation2id['rec'])
                rank=stable_top_n(z,max(TOP_NS))
                # CPU export is included in the actual deployment-style wall time.
                rows.append((z.cpu().numpy()[0],rank.cpu().numpy()[0]))
            z=np.stack([r[0] for r in rows]);rank=np.stack([r[1] for r in rows])
            np.savez_compressed(output/f'P0_untrained_full_cohort_{label}.npz',uids=state.test,logits=z,top50=rank)
            synchronize(device);elapsed=time.perf_counter()-began
            outputs[label]=rank
            result[label]={"wall_seconds":elapsed,"students":len(state.test),
                           "seconds_per_student":elapsed/len(state.test),
                           "scoring_schedule":"one student per decoder call in both A/B",
                           "includes":"fresh shared semantic preparation, graph encoding, CPU query graph assembly/degree, transfer, all-catalog ConvE, stable Top50, CPU/NPZ export",
                           **memory(device)}
        result['A_B_Top50_identical_students']=int((outputs['A']==outputs['B']).all(axis=1).sum())
    result.update({"parameter_hash":params,"full_cohort_measured":True,
                   "model_state":"untrained P0, not recommendation-quality evidence",
                   "excluded":"process startup, source validation, checkpoint loading; resident warm device",
                   "production_output":"full graph B reference, no local approximation"})
    write_json(output/'full_cohort_cost.json',result)
    return result


def run_dataset(dataset,output,device,chunks,warmup,measured,query_limit):
    output=Path(output)/dataset;output.mkdir(parents=True,exist_ok=False)
    original=model_modules()
    started=time.perf_counter();state=load_state(dataset)
    cfg_path=ROOT/'runs/merit_frequency_20261022'/dataset/'E0/seed2024/config.json'
    cfg=read_json(cfg_path)
    if cfg['ablation']!='feature_only' or cfg.get('include_test_triples') or cfg['bs']!=1024 or cfg['negative_ratio']!=5:
        raise ValueError('Unexpected frozen base configuration')
    original.set_seed(2024,deterministic=cfg['deterministic'])
    bundle=original.load_semantic_feature_bundle(state.graph,None,device=device)
    protocol=GraphProtocol(state,bundle.entity2id,bundle.relation2id)
    write_json(output/'graph_audit.json',protocol.manifest)
    base=build_model(original,bundle,cfg).to(device)
    model=SemanticCompGCN(base,protocol.inverse_original_ids,test_entities=protocol.test_entities,
                         chunk_size=chunks[0]).to(device)
    initialization=initialization_hash(model)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    batch=first_batch(original,bundle,state,cfg,device)
    initial_rng=rng_state()
    regression=original_regression(model,protocol,bundle,state,output)
    profile=profile_training(model,protocol.base.to(device),batch,initial,initial_rng,cfg,chunks,warmup,measured,output)
    inference=inference_probe(model,protocol,bundle,state,output,query_limit)
    # Only claim measured cohort throughput when the complete validation cohort
    # has been requested; sampled-query runs never extrapolate to full cost.
    costs=cohort_cost(model,protocol,bundle,state,output) if query_limit==0 else None
    report={'dataset':dataset,'status':'passed','formal_training_started':False,
            'configuration':cfg,'configuration_source_sha256':sha256(cfg_path),
            'initialization_sha256':initialization,'registered_parameters':sum(p.numel() for p in model.parameters()),
            'graph_base_id':protocol.base.graph_id,'training_profile':profile,
            'original_regression':regression,'inference':inference,
            'full_cohort_cost':costs,
            'elapsed_seconds':time.perf_counter()-started,
            'scope':'P0 C0 engineering. C3/C4 training adapters, exposure-gradient/resume gates remain separate.'}
    write_json(output/'P0_result.json',report)
    return report
