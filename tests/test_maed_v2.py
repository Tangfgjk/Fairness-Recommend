"""P0 protocol/gradient/decoder checks; no experimental training is performed."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from debias.io import ROOT, model_modules
from maed_v2.graph_protocol import GraphProtocol, make_graph
from maed_v2.compgcn_layer import SemanticCompGCNLayer
from maed_v2.model import SemanticCompGCN, stable_top_n

torch.set_num_threads(2)


def fixture(folder, nitems=52):
    entities = {n: i for i, n in enumerate([f'uid{u}' for u in range(4)] +
                [f'ex{e}' for e in range(nitems)] + [f'kc{k}' for k in range(3)])}
    relations = {n: i for i, n in enumerate(['rec'] + [f'{p}{i/100:.2f}' for p in
                 ('mlkc', 'exfr') for i in range(101)])}
    mastery = np.array([[.2,.5,.8],[.3,.6,.9],[.4,.7,0],[.6,.8,.1]])
    forget = np.full((4, nitems), .4); forget[:2, -2:] = 0; forget[2, 0] = 0
    state = SimpleNamespace(graph=folder, train=np.array([0,1]), test=np.array([2,3]),
        mastery=mastery, forget=forget, provenance={'state_hash':'toy', 'files':{}})
    for name, mapping in (('entities.dict', entities), ('relations.dict', relations)):
        (folder/name).write_text(''.join(f'{i}\t{n}\n' for n,i in mapping.items()), encoding='utf-8')
    for split, users in (('triples.txt', [0,1]), ('test_triples.txt', [2,3])):
        rows = []
        for u in users:
            rows += [f'kc{k}\tmlkc{v:.2f}\tuid{u}\n' for k,v in enumerate(mastery[u])]
            rows += [f'ex{e}\texfr{v:.2f}\tuid{u}\n' for e,v in enumerate(forget[u])
                     if split != 'triples.txt' or f'{v:.2f}' != '0.00']
            if split == 'triples.txt': rows.append(f'uid{u}\trec\tex{u}\n')
        (folder/split).write_text(''.join(rows), encoding='utf-8')
    return state, entities, relations


def small_base(entities, relations):
    original = model_modules(); n = len(entities)
    types = torch.tensor([0 if n.startswith('uid') else 2 if n.startswith('ex') else 1 for n in entities])
    return original.SemanticConvE(nentity=n, nrelation=len(relations), text_dim=4, numeric_dim=3,
        relation_type_ids=torch.tensor([0]+[1]*101+[2]*101),
        relation_strengths=torch.tensor([[1.]]+[[i/100] for i in range(101)]*2),
        entity_type_ids=types, cluster_ids=torch.zeros(n,dtype=torch.long),
        text_features=torch.randn(n,4), numeric_features=torch.rand(n,3), semantic_quality=torch.ones(n),
        numeric_feature_slices={'learner_irt':(0,1),'exercise_irt':(1,3)},
        embedding_dim=20, embedding_shape1=4, hidden_size=576, ablation_mode='feature_only')


class GraphChecks(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'tmp'); self.folder = Path(self.tmp.name)
        self.s, self.entities, self.relations = fixture(self.folder)
        self.p = GraphProtocol(self.s, self.entities, self.relations)

    def tearDown(self): self.tmp.cleanup()

    def test_actual_degrees_filtering_mapping_and_query_isolation(self):
        p = self.p; base = p.base; a = p.query(2); b = p.query(3)
        self.assertEqual(len(p.inverse_ids), 202)
        self.assertEqual(set(p.inverse_ids.values()), set(range(203,405)))
        self.assertNotIn(0, base.relation.tolist())
        for u in (2,3): self.assertEqual(base.degree[self.entities[f'uid{u}']], 0)
        self.assertEqual(a.degree[self.entities['uid3']], 0)
        self.assertEqual(b.degree[self.entities['uid2']], 0)
        self.assertEqual(base.degree[self.entities['ex51']], 0)
        self.assertEqual(a.degree[self.entities['ex51']], 1)
        self.assertEqual(a.audit['zero_exfr_filtered'], 1)
        self.assertEqual(a.degree.sum(), len(a.source))
        self.assertTrue(torch.equal(a.source, p.query(2).source))
        self.assertTrue(torch.equal(a.degree, p.query(2).degree))
        self.assertEqual(a.graph_id, p.query(2).graph_id)
        self.assertTrue((a.relation != self.relations['exfr0.00']).all())
        self.assertIn(self.relations['mlkc0.00'], a.relation.tolist())
        with self.assertRaises(ValueError): p.query(0)

    def test_snapshot_and_disallowed_inputs_fail_closed(self):
        file = self.folder/'test_triples.txt'; original = file.read_text()
        for suffix in ('uid2\trec\tex1\n', 'uid0\trec\tex0\n',
                       'ex0\texfr0.40\tuid0\n', 'kc0\tunknown\tuid2\n'):
            file.write_text(original+suffix)
            with self.assertRaises(ValueError): GraphProtocol(self.s,self.entities,self.relations)
        file.write_text(original.replace('mlkc0.40', 'mlkc0.80', 1))
        with self.assertRaisesRegex(ValueError, 'snapshot'): GraphProtocol(self.s,self.entities,self.relations)
        file.write_text(original+original.splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): GraphProtocol(self.s,self.entities,self.relations)

    def test_other_query_state_cannot_change_current_graph(self):
        before = self.p.query(2)
        self.p._queries[3] = []  # Deliberate unrelated-query corruption.
        after = self.p.query(2)
        self.assertEqual(before.graph_id,after.graph_id)
        self.assertTrue(torch.equal(before.source,after.source))

    def test_cross_query_delta_injection_is_rejected(self):
        self.p._queries[2].append(self.p._queries[3][0])
        with self.assertRaisesRegex(ValueError,"Other student's"):
            self.p.query(2)


class LayerChecks(unittest.TestCase):
    def test_hand_normalization_empty_and_isolated(self):
        g = make_graph([(0,1,2,0),(1,2,2,1)], 4, {1:3,2:4})
        layer = SemanticCompGCNLayer(2,1).double()
        h = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],dtype=torch.double)
        r = torch.ones((3,2),dtype=torch.double)
        r0 = layer.relation_initials(r,(1,2))
        aggregate = layer.aggregate(h,r0,g,0)
        torch.testing.assert_close(aggregate[2],(h[0]+h[1])/np.sqrt(2),atol=1e-12,rtol=1e-12)
        self.assertEqual(g.degree.tolist(),[1,1,2,0])
        result=layer(h,r,g,(1,2))
        self.assertTrue(torch.equal(result.entities[3],h[3]))
        empty=make_graph([],4,{1:3,2:4})
        self.assertTrue(torch.equal(layer(h,r,empty,(1,2)).entities,h))

    def test_chunk_recomputation_values_and_gradients_match_reference(self):
        torch.manual_seed(2210)
        rows=[(s,1+(s%2),t,s%2) for s in range(30) for t in range(30,40)]
        g=make_graph(rows,42,{1:3,2:4})
        h=torch.randn(42,7,dtype=torch.double,requires_grad=True)
        r=torch.randn(3,7,dtype=torch.double,requires_grad=True)
        reference=SemanticCompGCNLayer(7,10000,False).double()
        small=copy.deepcopy(reference); small.chunk_size=23;small.recompute=True
        coefficient=torch.randn(42,7,dtype=torch.double)
        outs=[]; gradients=[]
        for m in (reference,small):
            result=m(h,r,g,(1,2));outs.append(result)
            loss=(result.entities*coefficient).sum()+result.relations.square().sum()
            gradients.append(torch.autograd.grad(loss,(h,r,*m.parameters())))
        torch.testing.assert_close(outs[0].entities,outs[1].entities,atol=1e-9,rtol=1e-7)
        for a,b in zip(*gradients):
            torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-7)
            self.assertTrue(torch.isfinite(b).all());self.assertGreater(b.abs().sum(),0)

    def test_inverse_before_relation_update_noncommutative(self):
        layer=SemanticCompGCNLayer(2).double()
        with torch.no_grad():
            layer.inverse_transform.copy_(torch.tensor([[1.,2.],[0.,1.]]))
            layer.w_relation.copy_(torch.tensor([[2.,0.],[1.,3.]]))
        h=torch.randn(3,2,dtype=torch.double);r=torch.tensor([[1.,2.],[2.,3.]],dtype=torch.double)
        g=make_graph([(0,1,1,0)],3,{1:2})
        out=layer(h,r,g,(1,))
        inv=r[1:]@layer.inverse_transform
        expected=.5*inv+.5*inv@layer.w_relation
        torch.testing.assert_close(out.relations[2:],expected)
        wrong=(.5*r[1:]+.5*r[1:]@layer.w_relation)@layer.inverse_transform
        self.assertFalse(torch.allclose(expected,wrong))

    def test_old_edge_weight_must_change_when_query_degree_changes(self):
        # Base: exercise0 -> train2. Query: same exercise0 -> test3.
        # Existing 0->2 coefficient changes from 1 to 1/sqrt(2).
        layer=SemanticCompGCNLayer(2,1,False).double()
        h=torch.tensor([[2.,4.],[1.,1.],[3.,5.],[7.,8.]],dtype=torch.double)
        r=torch.ones((2,2),dtype=torch.double)
        base=make_graph([(0,1,2,1)],4,{1:2})
        query=make_graph([(0,1,2,1),(0,1,3,1)],4,{1:2})
        old=layer.aggregate(h,layer.relation_initials(r,(1,)),base,0)
        new=layer.aggregate(h,layer.relation_initials(r,(1,)),query,0)
        torch.testing.assert_close(new[2],old[2]/np.sqrt(2),atol=1e-12,rtol=1e-12)
        # Reverse message of the existing training student is also reweighted.
        inv=layer.aggregate(h,layer.relation_initials(r,(1,)),query,1)
        expected=(h[2]+h[3])*(r[1]@layer.inverse_transform)/np.sqrt(2)
        torch.testing.assert_close(inv[0],expected,atol=1e-12,rtol=1e-12)


class ModelChecks(GraphChecks):
    def setUp(self):
        super().setUp();torch.manual_seed(2024)
        self.base=small_base(self.entities,self.relations)
        self.model=SemanticCompGCN(self.base,self.p.inverse_original_ids,
            test_entities=self.p.test_entities,chunk_size=23)
        self.items=torch.tensor([self.entities[f'ex{e}'] for e in range(52)])
        self.u=torch.tensor([self.entities['uid2']])

    def test_zero_layer_complete_decoder_regression_and_rng(self):
        m=self.model.eval();h=torch.tensor([0,1,2,56]);r=torch.tensor([0,0,0,51]);t=torch.tensor([4,5,6,0])
        encoded=m.encode_graph(self.p.base,'bypass')
        saved=torch.get_rng_state().clone()
        self.assertTrue(torch.equal(m.score_logits(h,r,t,encoded),self.base.score_logits(h,r,t)))
        z=m.score_rec_all(self.u,self.items,encoded,0)
        expected=self.base.score_tails(self.u,torch.zeros_like(self.u),self.items)
        torch.testing.assert_close(torch.sigmoid(z),expected,atol=0,rtol=0)
        self.assertTrue(torch.equal(saved,torch.get_rng_state()))

    def test_full_local_all_candidates_and_query_isolation(self):
        m=self.model.eval()
        with torch.no_grad():
            initial=m.semantic_initials();base=m.encode_graph(self.p.base,initial=initial)
            graph=self.p.query(2);full=m.encode_graph(graph,initial=initial)
            local=m.encode_graph(graph,'local',initial=initial,output_ids=torch.cat((self.u,self.items)))
            valid=torch.cat((self.u,self.items))
            torch.testing.assert_close(full.entities[valid],local.entities[valid],atol=1e-6,rtol=1e-5)
            a=m.score_rec_all(self.u,self.items,full,0);b=m.score_rec_all(self.u,self.items,local,0)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            for n in range(5,51,5): self.assertTrue(torch.equal(stable_top_n(a,n),stable_top_n(b,n)))
            pairs=m.score_logits(self.u.expand(52),torch.zeros(52,dtype=torch.long),self.items,full)
            torch.testing.assert_close(a[0],pairs,atol=2e-6,rtol=1e-5)
            self.assertTrue(torch.equal(base.entities[self.u],initial.entities[self.u]))
            self.assertFalse(torch.equal(base.entities[self.items[-1]],full.entities[self.items[-1]]))
            m.encode_graph(self.p.query(3),initial=initial)
            again=m.encode_graph(self.p.query(2),initial=initial)
            self.assertTrue(torch.equal(full.entities,again.entities))
            with self.assertRaisesRegex(ValueError,'Cross-query'):
                m.score_rec_all(torch.tensor([3]),self.items,full,0)
            with self.assertRaisesRegex(ValueError,'not encoded'):
                m.score_logits(torch.tensor([56]),torch.tensor([1]),torch.tensor([0]),local)

    def test_stale_graph_parameter_mode_and_foreign_snapshots_rejected(self):
        m=self.model.eval();g=self.p.query(2)
        with torch.no_grad(): e=m.encode_graph(g)
        m.train()
        with self.assertRaisesRegex(ValueError,'Stale'):m.score_rec_all(self.u,self.items,e,0)
        m.eval()
        with torch.no_grad():m.layer.w_loop.add_(.1)
        with self.assertRaisesRegex(ValueError,'Stale'):m.score_rec_all(self.u,self.items,e,0)
        with torch.no_grad():e=m.encode_graph(g);g.degree.add_(1)
        with self.assertRaisesRegex(ValueError,'Graph tensors'):m.score_rec_all(self.u,self.items,e,0)
        other=copy.deepcopy(m)
        with self.assertRaisesRegex(ValueError,'same-model'):other.score_rec_all(self.u,self.items,e,0)

    def test_training_valid_rows_gradient_and_query_rejection(self):
        m=self.model.train()
        with self.assertRaisesRegex(ValueError,'cannot enter training'):m.encode_graph(self.p.query(2))
        encoded=m.encode_graph(self.p.base)
        self.assertFalse(encoded.valid[2]);self.assertFalse(encoded.valid[3])
        with self.assertRaisesRegex(ValueError,'not encoded'):m.score_rec_all(self.u,self.items,encoded,0)
        logits=m.score_logits(torch.tensor([0,1,56,4]),torch.tensor([0,0,51,142]),
                              torch.tensor([4,5,0,1]),encoded)
        torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits),torch.tensor([1.,0.,1.,0.])).backward()
        for name,p in m.layer.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum(),0,name)
        self.assertGreater(m.base.uid_feature_only_fusion.output[0].weight.grad.abs().sum(),0)

    def test_neighbor_control_and_graph_rng_initialization(self):
        m=self.model.eval()
        with torch.no_grad():
            initial=m.semantic_initials();g=self.p.query(2)
            full=m.encode_graph(g,initial=initial)
            no=m.encode_graph(g,initial=initial,neighbors=False)
            self.assertFalse(torch.equal(full.entities[self.u],no.entities[self.u]))
            self.assertFalse(torch.equal(m.score_rec_all(self.u,self.items,full,0),m.score_rec_all(self.u,self.items,no,0)))
        before=torch.get_rng_state().clone()
        if torch.cuda.is_available():torch.rand(7,device='cuda')
        cuda_before=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        SemanticCompGCN(copy.deepcopy(self.base),self.p.inverse_original_ids)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for old,new in zip(cuda_before,torch.cuda.get_rng_state_all() if cuda_before else []):
            self.assertTrue(torch.equal(old,new),'Graph initialization changed CUDA RNG')

    def test_eval_auxiliary_gradients_and_exception_restore_modes(self):
        from debias.scoring import fixed_statistics
        m=self.model.train();m.base.bn0.eval()
        flags=[x.training for x in m.modules()]
        buffers={k:v.clone() for k,v in m.named_buffers()};rng=torch.get_rng_state().clone()
        with fixed_statistics(m):
            encoded=m.encode_graph(self.p.base)
            users=torch.tensor([0,1])
            z=m.score_rec_all(users,self.items,encoded,0)
            from maed.boundary_exposure import calibrated_exposure_loss
            target=np.full((2,52),1/52)
            loss,_,_=calibrated_exposure_loss(z,target,(5,10),(.1,.2))
        loss.backward()
        for name,p in m.layer.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum(),0,name)
        self.assertEqual(flags,[x.training for x in m.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in m.named_buffers():self.assertTrue(torch.equal(v,buffers[k]),k)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with fixed_statistics(m): raise RuntimeError('intentional')
        self.assertEqual(flags,[x.training for x in m.modules()])

    def test_invalid_negative_entity_id_and_stale_semantics(self):
        m=self.model.eval()
        with torch.no_grad():
            initial=m.semantic_initials();encoded=m.encode_graph(self.p.base,initial=initial)
            with self.assertRaisesRegex(ValueError,'Invalid original'):
                m.score_rec_all(torch.tensor([-1]),self.items,encoded,0)
            with self.assertRaisesRegex(ValueError,'Invalid/duplicate'):
                m.encode_graph(self.p.base,'local',initial=initial,output_ids=torch.tensor([-1]))
            m.layer.w_loop.add_(.01)
            with self.assertRaisesRegex(ValueError,'Stale'):
                m.encode_graph(self.p.base,initial=initial)


if __name__=='__main__': unittest.main()
