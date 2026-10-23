import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

from debias.io import model_modules,read_json,write_json,ROOT
from debias.training import rng_state,restore_rng,build_model
from maed.state import EducationState,load_state
from maed.scoring import EducationScale
from maed.training import QScorer,ExposureObjective,joint_loss,train,export_scores

torch.set_num_threads(2)


def tiny_model():
    original=model_modules()
    return original.SemanticConvE(nentity=8,nrelation=2,text_dim=4,numeric_dim=3,
        relation_type_ids=torch.zeros(2,dtype=torch.long),relation_strengths=torch.tensor([[1.],[.5]]),
        entity_type_ids=torch.tensor([0,0,0,2,2,2,2,1]),cluster_ids=torch.zeros(8,dtype=torch.long),
        text_features=torch.randn(8,4),numeric_features=torch.rand(8,3),semantic_quality=torch.ones(8),
        numeric_feature_slices={'learner_irt':(0,1),'exercise_irt':(1,3)},
        embedding_dim=20,embedding_shape1=4,hidden_size=576,ablation_mode='feature_only')


def fixture(folder):
    state=EducationState('toy',folder,None,None,np.array([[1,0],[0,1],[1,1],[1,0]]),None,None,
        np.array([[.1,.2,.5,.8],[.8,.5,.2,.1],[.5,.2,.1,.6]]),np.array([0,1]),np.array([2]),{'state_hash':'toy'})
    entities={'uid0':0,'uid1':1,'uid2':2,'ex0':3,'ex1':4,'ex2':5,'ex3':6,'kc0':7}
    bundle=SimpleNamespace(entity2id=entities,relation2id={'rec':0,'mlkc0.50':1},id2relation={0:'rec',1:'mlkc0.50'},
                           exercise_entity_ids=torch.tensor([3,4,5,6]))
    scale=EducationScale.fit(state).to_dict();scale['train_users']=list(scale['train_users'])
    config={'ablation':'feature_only','deterministic':False,'negative_ratio':1,'bs':2,'learning_rate':.001,
            'epochs':2,'model_version':'synthetic_engineering_only'}
    assets={'reference_config':config,'beta':.5,'scale':scale,'temperatures':[.1,.2],'top_ns':[1,2]}
    triples=[(0,0,3),(1,0,4),(7,1,0),(7,1,1)]
    return state,bundle,assets,triples


class MAEDTrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1103)
        self.s,self.bundle,self.assets,self.triples=fixture(ROOT/'tmp')
        self.scale=EducationScale.fit(self.s)

    def test_only_rec_logits_change_and_probability_path_is_one_forward(self):
        m=tiny_model().eval()
        scorer=QScorer(self.s,self.bundle,self.scale,.5,'cpu')
        h=torch.tensor([0,7]);r=torch.tensor([0,1]);t=torch.tensor([3,0])
        raw=m.score_logits(h,r,t)
        fused=scorer.fuse_triples(raw,h,r,t)
        self.assertNotEqual(raw[0].item(),fused[0].item())
        self.assertEqual(raw[1].item(),fused[1].item())
        self.assertTrue(torch.equal(scorer.probabilities(m,h,r,t),torch.sigmoid(fused)))

    def test_train_pair_and_inference_matrix_use_same_Q(self):
        m=tiny_model().eval();scorer=QScorer(self.s,self.bundle,self.scale,.5,'cpu')
        h=torch.tensor([0,1]);r=torch.zeros(2,dtype=torch.long);t=torch.tensor([3,5])
        pair=scorer.fuse_triples(m.score_logits(h,r,t),h,r,t)
        matrix=scorer.rec_logits(m,[0,1],4)
        self.assertTrue(torch.allclose(pair,matrix[torch.arange(2),torch.tensor([0,2])],atol=2e-6))

    def test_zero_weight_skips_sampler_and_matches_original_update_rng(self):
        model=tiny_model().train();other=copy.deepcopy(model)
        h=torch.tensor([0,1,7,7]);r=torch.tensor([0,0,1,1]);t=torch.tensor([3,4,0,1]);labels=torch.ones(4)
        saved=rng_state()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss=F.binary_cross_entropy(model.score_triples(h,r,t),labels)
        loss.backward();optimizer.step();after=torch.get_rng_state().clone()
        restore_rng(saved)
        scorer=QScorer(self.s,self.bundle,self.scale,0,'cpu')
        opt=torch.optim.Adam(other.parameters(),lr=.001)
        bce=F.binary_cross_entropy(scorer.probabilities(other,h,r,t),labels)
        def forbidden(*_):raise AssertionError('Zero-weight auxiliary executed')
        total,aux,_=joint_loss(other,bce,forbidden,0,True)
        self.assertIs(total,bce);self.assertIsNone(aux)
        total.backward();opt.step()
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]),k)

    def test_exposure_gradients_preserve_BN_rng_and_reject_test_students(self):
        m=tiny_model().train();scorer=QScorer(self.s,self.bundle,self.scale,.5,'cpu')
        objective=ExposureObjective(self.s,scorer,[.1,.2],(1,2),2024,2)
        buffers={k:v.clone() for k,v in m.named_buffers()};rng=torch.get_rng_state().clone()
        loss,_=objective(m);loss.backward()
        self.assertGreater(m.conv1.weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(m.training)
        for k,v in m.named_buffers():self.assertTrue(torch.equal(v,buffers[k]),k)
        # A corrupted sampler is rejected by the state access guard when invoked.
        objective.rng=SimpleNamespace(choice=lambda *args,**kwargs:np.array([2]))
        with self.assertRaisesRegex(ValueError,'training cohorts'):objective(m)

    def test_auxiliary_mode_recovers_even_after_forward_exception(self):
        m=tiny_model().train();scorer=QScorer(self.s,self.bundle,self.scale,.5,'cpu')
        objective=ExposureObjective(self.s,scorer,[.1,.2],(1,2),2024,2)
        m.bn0.eval();before=[x.training for x in m.modules()]
        with patch.object(scorer,'rec_logits',side_effect=RuntimeError('probe error')):
            with self.assertRaisesRegex(RuntimeError,'probe error'):objective(m)
        self.assertEqual(before,[x.training for x in m.modules()])

    def test_full_training_resume_and_M4_zero_equivalence(self):
        scratch=ROOT/'tmp';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            folder=Path(name);state,bundle,assets,triples=fixture(folder)
            original=model_modules()
            import maed.training as training
            with patch.object(original,'load_semantic_feature_bundle',return_value=bundle), \
                 patch.object(original,'read_triples',return_value=triples), \
                 patch.object(training,'build_model',side_effect=lambda *a:tiny_model()):
                train(state,assets,folder/'full','M4',2024,'test',cuda='false',users_per_step=2)
                save=training.atomic_checkpoint
                def stop(path,payload):
                    save(path,payload)
                    if path.name=='last.pt' and payload['epoch']==1:raise RuntimeError('synthetic interruption')
                with patch.object(training,'atomic_checkpoint',side_effect=stop):
                    with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                        train(state,assets,folder/'resume','M4',2024,'test',cuda='false',users_per_step=2)
                train(state,assets,folder/'resume','M4',2024,'test',cuda='false',users_per_step=2)
                a=torch.load(folder/'full/last.pt',weights_only=False);b=torch.load(folder/'resume/last.pt',weights_only=False)
                for k,v in a['model_state_dict'].items():self.assertTrue(torch.equal(v,b['model_state_dict'][k]),k)
                self.assertEqual(a['loss'],b['loss']);self.assertEqual(a['exposure_rng'],b['exposure_rng'])
                train(state,assets,folder/'resume','M4',2024,'test',cuda='false',users_per_step=2)
                for method in ('M3','M4'):
                    train(state,assets,folder/method,method,2024,'test',cuda='false',exposure_weight=0,users_per_step=2)
                a=torch.load(folder/'M3/last.pt',weights_only=False);b=torch.load(folder/'M4/last.pt',weights_only=False)
                for k,v in a['model_state_dict'].items():self.assertTrue(torch.equal(v,b['model_state_dict'][k]),k)
                self.assertEqual(a['loss'],b['loss']);self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
                # The real export flow must serialize the checkpoint's Q, not core-only scores.
                export_scores(state,folder/'full/best.pt',folder/'export/uid_ex_scores.json',cuda='false')
                rows=read_json(folder/'export/uid_ex_scores.json')
                self.assertEqual([r['uid'] for r in rows],['uid2'])
                self.assertEqual(len(rows[0]['scores']),4)
                with self.assertRaisesRegex(ValueError,'Contract differs'):
                    train(state,assets,folder/'full','M4',2024,'test',cuda='false',exposure_weight=.02,users_per_step=2)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA engineering check')
    def test_actual_1024_sample_BCE_plus_full_catalog_exposure_backward(self):
        # One backward only, no optimizer; verifies memory/GRU/BN on the actual interface.
        from debias.training import build_model as original_build
        state=load_state('Eedi');source=ROOT/'runs/merit_frequency_20261022/Eedi/E0/seed2024'
        original=model_modules();device=torch.device('cuda')
        bundle=original.load_semantic_feature_bundle(state.graph,None,device=device)
        cfg=read_json(source/'config.json')
        model=original_build(original,bundle,cfg).to(device)
        ck=torch.load(source/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model_state_dict'],strict=True);model.train()
        triples=original.read_triples(state.graph/'triples.txt',bundle.entity2id,bundle.relation2id)
        data=original.TripleDataset(triples,bundle.id2relation,bundle.exercise_entity_ids.cpu().tolist(),
                                   original.build_positive_tails_by_hr(triples),cfg['negative_ratio'],2024)
        indices=np.random.default_rng(20261023).choice(len(data),1024,replace=False)
        columns=list(zip(*(data[int(i)] for i in indices)))
        h,r,t,labels=[torch.stack(col).to(device) for col in columns]
        calibration=read_json(ROOT/'runs/论文格式文档统计/maed_m2_mechanism_20261023/Eedi/temperature_calibration.json')
        scorer=QScorer(state,bundle,EducationScale.fit(state),.5,device)
        bce=F.binary_cross_entropy(scorer.probabilities(model,h,r,t),labels)
        before={k:v.clone() for k,v in model.named_buffers() if 'running_' in k or 'num_batches_tracked' in k}
        objective=ExposureObjective(state,scorer,[row['temperature'] for row in calibration['per_N']],tuple(range(5,51,5)),2024,16)
        total,aux,_=joint_loss(model,bce,objective,.01,True)
        self.assertIsNotNone(aux)
        # Exercise the actual first-batch balance logging path before joint backward.
        parameters=[p for p in model.parameters() if p.requires_grad]
        for term in (bce,aux):
            gradients=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))
            del gradients
        total.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.shared_text_encoder.parameters()))
        for k,v in model.named_buffers():
            if k in before:self.assertTrue(torch.equal(before[k],v),k)
        self.assertTrue(model.training)

    def test_report_uses_M3_as_direct_QE_reference(self):
        from maed.training_report import report
        from debias.io import TOP_NS,sha256
        scratch=ROOT/'tmp';scratch.mkdir(exist_ok=True)
        def result(ada,nov,ded):
            return {'top_n':{str(n):{'Ada':ada,'KG4Ex-NOV-Set':nov,'Ep_t0.5':.01,'DED':ded} for n in TOP_NS},
                    'per_user':[{'uid':u,'N':n,'Ada':ada,'KG4Ex-NOV-Set':nov} for n in TOP_NS for u in (0,1,2,3)],
                    'CFP':{'value':.5,'users':4}}
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            folder=Path(name);root=folder/'run';base=folder/'base'
            entries=[(base/'Eedi/E0/seed2024/eval/metrics.json',result(.8,.95,.5)),
                     (root/'Eedi/Q-fixed/seed2024/eval/metrics.json',result(.8,.94,.4)),
                     (root/'Eedi/M3/seed2024/best/eval/metrics.json',result(.81,.94,.4)),
                     (root/'Eedi/M4/seed2024/best/eval/metrics.json',result(.80,.94,.3))]
            for p,value in entries:
                write_json(p,value);write_json(p.parent/'evaluation_complete.json',{'metrics_sha256':sha256(p)})
            output=report(root,base)
            import csv
            with (output/'Eedi/all_metrics_M4_vs_M3.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            difference=next(float(r['delta_M4_minus_M3']) for r in rows if r['metric']=='Ada' and r['N']=='10')
            self.assertAlmostEqual(difference,-.01)
            self.assertTrue((output/'Eedi/student_deltas_M4_vs_M3.csv').exists())


if __name__=='__main__':unittest.main()
