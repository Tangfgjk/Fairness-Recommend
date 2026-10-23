import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from debias.io import model_modules
from debias.metrics import conditional_preference, novelty_set, total_variation, gini
from debias.objectives import FairObjective, combine_loss, matched_hinge
from debias.pair_matching import MatchConfig, build_pairs
from debias.scoring import fixed_statistics, pair_logits
from debias.training import rng_state, restore_rng

torch.set_num_threads(2)


def small_model():
    original = model_modules()
    torch.manual_seed(123)
    return original.SemanticConvE(
        nentity=7, nrelation=1, text_dim=4, numeric_dim=3,
        relation_type_ids=torch.zeros(1, dtype=torch.long), relation_strengths=torch.ones(1,1),
        entity_type_ids=torch.tensor([0,0,2,2,2,2,1]), cluster_ids=torch.zeros(7,dtype=torch.long),
        text_features=torch.randn(7,4), numeric_features=torch.rand(7,3), semantic_quality=torch.ones(7),
        numeric_feature_slices={"learner_irt":(0,1), "exercise_irt":(1,3)},
        embedding_dim=20, embedding_shape1=4, hidden_size=576)


def reference_probability(model, h, r, t):
    # Literal original score_triples expression, independent of the new logit method.
    he, re, te = model.entity_embedding(h), model.relation_embedding(r), model.entity_embedding(t)
    x = model.conve_transform(he, re)
    return torch.sigmoid(torch.sum(x * te, dim=1) + model.b[t])


class ModelTests(unittest.TestCase):
    def test_zero_weight_exact_original_update_and_rng(self):
        model = small_model().train()
        other = copy.deepcopy(model)
        h,r,t = torch.tensor([0,1,0,1]), torch.zeros(4,dtype=torch.long), torch.tensor([2,3,4,5])
        labels = torch.tensor([1.,1.,0.,0.])
        rng = rng_state()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        opt.zero_grad()
        loss = F.binary_cross_entropy(reference_probability(model,h,r,t),labels)
        loss.backward(); opt.step()
        after = torch.get_rng_state().clone()
        restore_rng(rng)
        opt2 = torch.optim.Adam(other.parameters(), lr=.001)
        opt2.zero_grad()
        bce = F.binary_cross_entropy(other.score_triples(h,r,t),labels)
        def forbidden(*args):
            raise AssertionError("Zero weight executed fairness forward")
        result, auxiliary, _ = combine_loss(other,bce,forbidden,0,True,torch.device('cpu'))
        self.assertIs(result,bce)
        self.assertIsNone(auxiliary)
        result.backward(); opt2.step()
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(loss.item(),result.item())
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,other.state_dict()[key]), key)
        for (name,p),(name2,p2) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad,p2.grad),name)

    def test_logits_and_probability_eval(self):
        m = small_model().eval()
        h,r,t = torch.tensor([0,1]),torch.zeros(2,dtype=torch.long),torch.tensor([2,3])
        self.assertTrue(torch.equal(torch.sigmoid(m.score_logits(h,r,t)),reference_probability(m,h,r,t)))

    def test_auxiliary_gradients_and_bn_preserved(self):
        m = small_model().train()
        with torch.no_grad():
            m.b[2] = 30
        before = {n:v.clone() for n,v in m.named_buffers()}
        rng = torch.get_rng_state().clone()
        h,r,hi,lo = torch.tensor([0,1]),torch.zeros(2,dtype=torch.long),torch.tensor([2,2]),torch.tensor([3,4])
        high,low = pair_logits(m,h,r,hi,lo)
        loss = matched_hinge(high,low,h,.1)
        loss.backward()
        self.assertGreater(loss.item(),0)
        for module in (m.conv1,m.uid_feature_only_fusion,m.exercise_feature_only_fusion):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertGreater(m.b.grad.abs().sum().item(),0)
        for n,v in m.named_buffers():
            self.assertTrue(torch.equal(before[n],v),n)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(m.training)

    def test_exception_restores_mixed_modes(self):
        m=small_model().train(); m.bn1.eval()
        modes=[x.training for x in m.modules()]
        with self.assertRaises(RuntimeError):
            with fixed_statistics(m):
                self.assertFalse(any(x.training for x in m.modules()))
                raise RuntimeError('injected failure')
        self.assertEqual(modes,[x.training for x in m.modules()])

    def test_hinge_user_macro_and_direction(self):
        high=torch.tensor([2.,2.,4.],requires_grad=True)
        low=torch.zeros(3); users=torch.tensor([1,1,2])
        self.assertEqual(matched_hinge(high,low,users,0).item(),10.)
        self.assertEqual(matched_hinge(high,low,users,0,-torch.ones(3)).item(),0.)

    def test_sampler_resume(self):
        model=small_model()
        bundle=SimpleNamespace(entity2id={'uid0':0,'uid1':1,'ex0':2,'ex1':3,'ex2':4,'ex3':5},relation2id={'rec':0})
        pairs=np.array([[0,0,1],[0,2,3],[1,0,3],[1,2,1]])
        obj=FairObjective(pairs,bundle,2024,users_per_step=1,pairs_per_user=1)
        saved=copy.deepcopy(obj.state_dict())
        loss,stats=obj(model,torch.device('cpu'))
        obj.load_state_dict(saved)
        repeated,again=obj(model,torch.device('cpu'))
        self.assertEqual(stats,again); self.assertEqual(loss.item(),repeated.item())

    def test_joint_bce_fair_backward_without_extra_bn_updates(self):
        model=small_model().train()
        h,r,t=torch.tensor([0,1,0,1]),torch.zeros(4,dtype=torch.long),torch.tensor([2,3,4,5])
        bce=F.binary_cross_entropy(model.score_triples(h,r,t),torch.tensor([1.,1.,0.,0.]))
        buffers={n:v.clone() for n,v in model.named_buffers()}
        def auxiliary(m,device):
            a,b=pair_logits(m,h[:2],r[:2],torch.tensor([2,2]),torch.tensor([3,4]))
            return matched_hinge(a,b,h[:2],0),{}
        total,_,_=combine_loss(model,bce,auxiliary,.1,True,torch.device('cpu'))
        total.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        for n,v in model.named_buffers():self.assertTrue(torch.equal(v,buffers[n]),n)


class MetricTests(unittest.TestCase):
    def test_set_novelty_not_per_question_mean(self):
        self.assertAlmostEqual(novelty_set({1,2},{1}),.5)
        self.assertEqual(novelty_set({1},set()),1.)
        with self.assertRaises(ValueError): novelty_set(set(),set())

    def test_cfp_macro_ties_empty(self):
        scores=np.array([[1.,1.,0.],[0.,1.,0.]])
        pairs=np.array([[7,0,1],[7,0,2],[8,0,1]])
        self.assertEqual(conditional_preference(scores,[7,8],pairs)['value'],.375)
        self.assertIsNone(conditional_preference(scores,[7,8],np.empty((0,3),int))['value'])

    def test_exposure(self):
        self.assertEqual(total_variation([1,2],[2,4]),0.)
        self.assertEqual(total_variation([1,0],[0,1]),1.)
        self.assertEqual(gini([1,1,1]),0.)
        self.assertAlmostEqual(gini([0,0,3]),2/3)

    def test_e3_user_frequency_strata_and_determinism(self):
        q=np.eye(6,dtype=int)
        class C:
            train=np.array([0,1]); test=np.array([2]); frequency=np.array([100,50,25,12,6,2])
            positives={0:list(range(6)),1:list(range(6))}
            distance=np.array([[.1,.1,.1,.3,.3,.3]]*3)
            forget=np.ones((3,6))*.8
            difficulty=np.arange(6)/6; discrimination=np.arange(6)/8
            def product(self,u,es):return np.ones(len(es))*.8
            def rule(self,u):return np.arange(6)
        c=C();c.q=q
        cfg=MatchConfig()
        arrays,_=build_pairs(c,cfg); again,_=build_pairs(c,cfg)
        self.assertGreater(len(arrays['E1']),0)
        for k in arrays:self.assertTrue(np.array_equal(arrays[k],again[k]))
        for u in c.train:
            hists=[]
            for k in ('E1','E3'):
                p=arrays[k][arrays[k][:,0]==u]
                bins=np.floor((np.log1p(c.frequency[p[:,1]])-np.log1p(c.frequency[p[:,2]]))/cfg.frequency_bin_width).astype(int)
                hists.append(dict(zip(*np.unique(bins,return_counts=True))))
            self.assertEqual(*hists)
        self.assertTrue(set(arrays['evaluation'][:,0]) <= set(c.test))


class RealDataIntegration(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA regression requires a GPU')
    def test_cuda_joint_shared_gru_backward(self):
        from debias.prepare import prepared,baseline_folder
        from debias.training import build_model
        from debias.io import read_json
        ctx,arrays,_=prepared('Eedi')
        original=model_modules()
        device=torch.device('cuda')
        bundle=original.load_semantic_feature_bundle(ctx.graph,None,device=device)
        model=build_model(original,bundle,read_json(baseline_folder('Eedi',2024)/'config.json')).to(device).train()
        p=arrays['E1'][:4]
        h=torch.tensor([bundle.entity2id[f'uid{u}'] for u in p[:,0]],device=device)
        r=torch.full_like(h,bundle.relation2id['rec'])
        t=torch.tensor([bundle.entity2id[f'ex{e}'] for e in p[:,1]],device=device)
        bce=F.binary_cross_entropy(model.score_triples(h,r,t),torch.ones(4,device=device))
        objective=FairObjective(arrays['E1'],bundle,2024,users_per_step=2,pairs_per_user=2)
        before={n:v.clone() for n,v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
        total,_,_=combine_loss(model,bce,objective,.1,True,device)
        total.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertTrue(any(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
                            for x in model.shared_text_encoder.parameters()))
        self.assertTrue(model.training)
        for n,v in model.named_buffers():
            if n in before:self.assertTrue(torch.equal(v,before[n]),n)

    def test_prepared_data_and_real_model_gradient(self):
        from debias.prepare import prepared,baseline_folder
        from debias.training import build_model
        from debias.io import read_json
        ctx,arrays,manifest=prepared('Eedi')
        self.assertEqual((len(ctx.train),len(ctx.test),len(ctx.q)),(701,234,948))
        self.assertEqual(len(arrays['E1']),36564)
        self.assertFalse(set(arrays['E1'][:,0]) & set(ctx.test))
        for u in np.unique(arrays['E1'][:,0]):
            for key in ('E1','E3'):
                p=arrays[key][arrays[key][:,0]==u]
                self.assertTrue(set(p[:,1])|set(p[:,2]) <= set(ctx.positives[int(u)]))
        orig=model_modules()
        bundle=orig.load_semantic_feature_bundle(ctx.graph,None,device=torch.device('cpu'))
        base=baseline_folder('Eedi',2024)
        config=read_json(base/'config.json')
        model=build_model(orig,bundle,config)
        ck=torch.load(base/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model_state_dict'],strict=True)
        model.train()
        buffers={n:b.clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
        objective=FairObjective(arrays['E1'],bundle,2024,margin=.1,users_per_step=4,pairs_per_user=2)
        loss,stats=objective(model,torch.device('cpu'))
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        self.assertGreater(model.conv1.weight.grad.abs().sum().item(),0)
        for n,b in model.named_buffers():
            if n in buffers:self.assertTrue(torch.equal(b,buffers[n]),n)
        self.assertTrue(model.training)

    def test_ep_interface_with_frozen_context(self):
        from debias.io import ep_module
        from debias.prepare import prepared,baseline_folder
        from debias.context import load_scores
        ctx,_,_=prepared('Eedi')
        ep=ep_module()
        frozen=ep.load_context(ctx.name,ctx.data_root,ctx.data_root/'_ep_readouts',torch.device('cpu'))
        scores=load_scores(baseline_folder('Eedi',2024)/'eval/uid_ex_scores.json',ctx)
        ranked=np.argsort(-scores[0],kind='stable')[:5].tolist()
        gains,_,_=ep.simulate(frozen,int(ctx.test[0]),ranked,[5],.5,False)
        self.assertTrue(np.isfinite(gains[5]))


class TrainingControlFlow(unittest.TestCase):
    def test_epoch_checkpoint_resume_matches_uninterrupted_toy_run(self):
        # Exercise the real runner with synthetic triples, not a dataset experiment.
        from debias import training
        from debias.io import write_json
        original=model_modules()
        entities={'uid0':0,'uid1':1,'ex0':2,'ex1':3,'ex2':4,'ex3':5,'kc0':6}
        bundle=SimpleNamespace(entity2id=entities,relation2id={'rec':0},id2relation={0:'rec'},exercise_entity_ids=torch.tensor([2,3,4,5]))
        triples=[(0,0,2),(1,0,3)]
        config={'ablation':'feature_only','deterministic':False,'negative_ratio':1,'bs':2,'learning_rate':.001,'epochs':2,'model_version':'test_only'}
        arrays={'E1':np.array([[0,0,1],[1,1,2]])}
        manifest={'context_id':'toy','pairs_sha256':'toy'}
        tmp_root=Path(__file__).resolve().parents[1]/'tmp'
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(tmp_root.resolve()))
            tmp=Path(tmp);write_json(tmp/'config.json',config)
            ctx=SimpleNamespace(name='toy',graph=tmp)
            uninterrupted,interrupted=tmp/'full',tmp/'resume'
            with patch.object(training,'baseline_folder',return_value=tmp), patch.object(training,'source_digest',return_value='test'), \
                 patch.object(original,'load_semantic_feature_bundle',return_value=bundle), \
                 patch.object(original,'read_triples',return_value=triples), \
                 patch.object(training,'build_model',side_effect=lambda *a:small_model()):
                training.train(ctx,arrays,manifest,uninterrupted,'E1',2024,cuda='false')
                save=training.atomic_checkpoint
                def stop_after_epoch(path,payload):
                    save(path,payload)
                    if path.name=='last.pt' and payload['epoch']==1:
                        raise RuntimeError('simulated interruption')
                with patch.object(training,'atomic_checkpoint',side_effect=stop_after_epoch):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        training.train(ctx,arrays,manifest,interrupted,'E1',2024,cuda='false')
                training.train(ctx,arrays,manifest,interrupted,'E1',2024,cuda='false')
                a=torch.load(uninterrupted/'last.pt',weights_only=False)
                b=torch.load(interrupted/'last.pt',weights_only=False)
                for key,value in a['model_state_dict'].items():
                    self.assertTrue(torch.equal(value,b['model_state_dict'][key]),key)
                self.assertEqual(a['loss'],b['loss'])
                self.assertEqual(a['fair_rng'],b['fair_rng'])
                training.train(ctx,arrays,manifest,interrupted,'E1',2024,cuda='false')  # completion skip
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    training.train(ctx,arrays,manifest,interrupted,'E1',2024,margin=.2,cuda='false')


if __name__=='__main__':unittest.main()
