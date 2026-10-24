import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from debias.io import ROOT,read_json,write_json,model_modules
from maed.state import EducationState,education_distances
from maed.scoring import EducationScale
from maed_v2.training import train,GraphQScorer,GraphExposure,build_runtime,export_scores,calibrate_assets,load_checkpoint
from maed_v2.preflight import same_tree
from test_maed_v2 import fixture,small_base


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);(ROOT/'tmp').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.folder=Path(self.tmp.name)
        s,entities,relations=fixture(self.folder)
        q=np.array([[int(k==e%3) for k in range(3)] for e in range(52)])
        state=EducationState('toy',s.graph,None,None,q,s.mastery,s.forget,
            education_distances(s.mastery,s.forget,q),s.train,s.test,s.provenance)
        self.state=state;self.entities=entities;self.relations=relations
        self.bundle=SimpleNamespace(entity2id=entities,relation2id=relations,id2relation={i:n for n,i in relations.items()},
                                    exercise_entity_ids=torch.tensor([entities[f'ex{e}'] for e in range(52)]))
        self.original=model_modules()
        self.patches=[patch.object(self.original,'load_semantic_feature_bundle',return_value=self.bundle),
                      patch('maed_v2.training.source_hash',return_value='test'),
                      patch('maed_v2.training.build_model',side_effect=lambda *a:small_base(entities,relations))]
        for p in self.patches:p.start()
        cfg={'ablation':'feature_only','deterministic':False,'epochs':2,'bs':16,'negative_ratio':1,
             'learning_rate':.001,'seed':2024}
        model,bundle,p=build_runtime(state,cfg,23,2024,torch.device('cpu'))
        scaled=EducationScale.fit(state).to_dict();scaled['train_users']=list(scaled['train_users'])
        self.assets={'reference_config':cfg,'scale':scaled,'beta':.5,'chunk_size':23,'top_ns':list(range(5,51,5)),
                     'state_hash':state.provenance['state_hash'],
                     'base_graph_id':p.base.graph_id,'C0_checkpoint_sha256':'engineering-test',
                     'temperatures':[.1]*10}

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_positive_C4_incomplete_epoch_replay_is_exact(self):
        options={'engineering_max_batches':3,'users_per_step':2}
        train(self.state,self.assets,self.folder/'full','C4',2024,'test','cpu',**options)
        def fail(epoch,batch):
            if epoch==2 and batch==2:raise RuntimeError('mid-epoch injected failure')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            train(self.state,self.assets,self.folder/'resume','C4',2024,'test','cpu',batch_hook=fail,**options)
        prior=torch.load(self.folder/'resume/last.pt',weights_only=False)
        self.assertEqual(prior['epoch'],1)
        train(self.state,self.assets,self.folder/'resume','C4',2024,'test','cpu',**options)
        a=torch.load(self.folder/'full/last.pt',weights_only=False);b=torch.load(self.folder/'resume/last.pt',weights_only=False)
        for k in ('model_state_dict','optimizer_state_dict','rng','exposure_rng','monitor_state','loss','best_loss','best_epoch'):
            self.assertTrue(same_tree(a[k],b[k]),k)
        train(self.state,self.assets,self.folder/'resume','C4',2024,'test','cpu',**options)

    def test_resume_tolerates_recomputed_monitor_roundoff_but_preserves_reference(self):
        from maed_v2.training import GraphMonitor
        def stop(epoch,batch):
            if epoch==1 and batch==1:raise RuntimeError('probe failure')
        with self.assertRaises(RuntimeError):
            train(self.state,self.assets,self.folder/'noise','C4',2024,'test','cpu',batch_hook=stop,engineering_max_batches=2)
        saved=read_json(self.folder/'noise/initialization_audit.json')
        actual=GraphMonitor.evaluate
        def noisy(monitor,model):
            result=actual(monitor,model);result['soft_mean_TV']+=1e-12
            return result
        with patch.object(GraphMonitor,'evaluate',noisy):
            train(self.state,self.assets,self.folder/'noise','C4',2024,'test','cpu',engineering_max_batches=2)
        self.assertEqual(saved,read_json(self.folder/'noise/initialization_audit.json'))

    def test_zero_exposure_C4_exactly_matches_C3_and_skips_sampler(self):
        with patch('maed_v2.training.GraphExposure',side_effect=AssertionError('zero branch used')):
            for method in ('C3','C4'):
                train(self.state,self.assets,self.folder/method,method,2024,'test','cpu',
                      exposure_weight=0,engineering_max_batches=2)
        a=torch.load(self.folder/'C3/last.pt',weights_only=False);b=torch.load(self.folder/'C4/last.pt',weights_only=False)
        for k in ('model_state_dict','optimizer_state_dict','rng','exposure_rng','loss'):
            self.assertTrue(same_tree(a[k],b[k]),k)
        self.assertEqual(read_json(self.folder/'C3/initialization_audit.json')['model_state_sha256'],
                         read_json(self.folder/'C4/initialization_audit.json')['model_state_sha256'])

    def test_Q_triple_and_catalog_scores_and_auxiliary_modes(self):
        model,bundle,p=build_runtime(self.state,self.assets['reference_config'],23,2024,torch.device('cpu'))
        scorer=GraphQScorer(self.state,bundle,EducationScale.fit(self.state),.5,torch.device('cpu'),p.base)
        model.eval()
        with torch.no_grad():
            e=model.encode_graph(p.base);h=torch.tensor([0,1]);r=torch.zeros(2,dtype=torch.long);t=torch.tensor([4,5])
            prob=scorer.probabilities(model,h,r,t,e)
            z=scorer.rec_logits(model,[0,1],52,e)
            torch.testing.assert_close(prob,torch.sigmoid(z[[0,1],[0,1]]),atol=2e-6,rtol=1e-5)
        model.train();flags=[m.training for m in model.modules()]
        obj=GraphExposure(self.state,scorer,[.1]*10,tuple(range(5,51,5)),2024,2)
        loss,_=obj(model);loss.backward()
        self.assertEqual(flags,[m.training for m in model.modules()])
        for n,p in model.layer.named_parameters():self.assertTrue(torch.isfinite(p.grad).all(),n)

    def test_different_graph_source_cannot_resume_and_epoch_zero_exists(self):
        def fail(epoch,batch):raise RuntimeError('first epoch failure')
        with self.assertRaisesRegex(RuntimeError,'first epoch'):
            train(self.state,self.assets,self.folder/'run','C4',2024,'test','cpu',batch_hook=fail,engineering_max_batches=2)
        self.assertEqual(torch.load(self.folder/'run/last.pt',weights_only=False)['epoch'],0)
        with self.assertRaisesRegex(ValueError,'source hash differs'):
            train(self.state,self.assets,self.folder/'run','C4',2024,'changed source','cpu',engineering_max_batches=2)

    def test_checkpoint_export_and_exactly_frozen_C0_temperature(self):
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=2)
        checkpoint=self.folder/'C0/best.pt'
        assets=calibrate_assets(self.state,checkpoint,self.folder/'calibration','cpu')
        self.assertEqual(assets['calibration_train_users'],[0,1])
        with patch('maed_v2.training.load_checkpoint',side_effect=AssertionError('recalibration occurred')):
            again=calibrate_assets(self.state,checkpoint,self.folder/'calibration','cpu')
        self.assertEqual(assets,again)
        for inference in ('A','B'):
            destination=self.folder/f'export_{inference}/uid_ex_scores.json'
            meta=export_scores(self.state,checkpoint,destination,'cpu',inference)
            scores=read_json(destination)
            self.assertEqual([r['uid'] for r in scores],['uid2','uid3'])
            self.assertEqual(len(scores[0]['scores']),52)
            self.assertEqual(meta['inference_protocol'],inference)
            self.assertEqual(meta['score_kind'],'ranking_logit_not_probability')
            self.assertTrue(np.array_equal(np.asarray([r['scores'] for r in scores]),np.load(destination.with_suffix('.logits.npy'))))

    def test_standalone_export_rejects_checkpoint_from_another_source(self):
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=1)
        checkpoint=self.folder/'C0/best.pt'
        output=self.folder/'rejected/uid_ex_scores.json'
        with patch('maed_v2.training.source_hash',return_value='different-source'):
            with self.assertRaisesRegex(ValueError,'source'):
                export_scores(self.state,checkpoint,output,'cpu')
        self.assertFalse(output.exists())

    def test_checkpoint_contract_is_self_contained_and_checked_before_build(self):
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=1)
        checkpoint=self.folder/'C0/best.pt'
        ck=torch.load(checkpoint,weights_only=False)
        moved=self.folder/'standalone.pt';torch.save(ck,moved)
        # A standalone checkpoint does not need an adjacent run_config.json.
        load_checkpoint(self.state,moved,'cpu')
        with self.assertRaisesRegex(ValueError,'contract hash mismatch'):
            load_checkpoint(self.state,moved,'cpu',expected_contract_hash='another-run')
        legacy={k:v for k,v in ck.items() if k not in ('contract','source_hash')};torch.save(legacy,moved)
        with self.assertRaisesRegex(ValueError,'receipt missing'):load_checkpoint(self.state,moved,'cpu')
        for field,value in (('source_hash','other'),('contract_hash','other'),
                            ('config',{**ck['config'],'learning_rate':.5}),
                            ('assets',{**ck['assets'],'chunk_size':99})):
            with self.subTest(field=field):
                invalid={**ck,field:value};torch.save(invalid,moved)
                with patch('maed_v2.training.build_runtime',side_effect=AssertionError('built before validation')):
                    with self.assertRaises(ValueError):load_checkpoint(self.state,moved,'cpu')

    def test_cached_calibration_and_audit_cannot_bypass_source_validation(self):
        from maed_v2.audit import audit_checkpoint
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=1)
        checkpoint=self.folder/'C0/best.pt'
        calibrate_assets(self.state,checkpoint,self.folder/'calibration','cpu')
        # Matching cached-file metadata alone must not authorize a stale checkpoint.
        saved=read_json(self.folder/'calibration/assets.json');saved['calibration_source_hash']='new-source'
        write_json(self.folder/'calibration/assets.json',saved)
        from debias.io import sha256
        write_json(self.folder/'audit/audit_complete.json',{'checkpoint_sha256':sha256(checkpoint),
                   'source_hash':'new-source','contract_hash':'irrelevant','outputs':{}})
        with patch('maed_v2.training.source_hash',return_value='new-source'):
            with self.assertRaisesRegex(ValueError,'source hash differs'):
                calibrate_assets(self.state,checkpoint,self.folder/'calibration','cpu')
            with self.assertRaisesRegex(ValueError,'source hash differs'):
                audit_checkpoint(self.state,checkpoint,self.folder/'audit',None,None,'cpu')

    def test_formal_chunks_and_full_B_are_enforced(self):
        from maed_v2.training import check_chunk,ACCEPTED_CHUNKS
        for name,chunk in ACCEPTED_CHUNKS.items():
            check_chunk(name,chunk)
            with self.assertRaisesRegex(ValueError,'Unaccepted formal chunk'):check_chunk(name,chunk+1)
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=1)
        from maed_v2.model import SemanticCompGCN
        encode=SemanticCompGCN.encode_graph;modes=[]
        def full_only(model,graph,mode='full',**kw):
            modes.append(mode)
            if mode!='full':raise AssertionError('Non-reference formal inference')
            return encode(model,graph,mode,**kw)
        with patch.object(SemanticCompGCN,'encode_graph',full_only):
            meta=export_scores(self.state,self.folder/'C0/best.pt',self.folder/'B.json','cpu','B')
        self.assertEqual(modes,['full']*len(self.state.test))
        self.assertEqual(meta['encoding_mode'],'full')

    def test_trained_checkpoint_mechanism_audit_does_not_update_parameters(self):
        from maed_v2.audit import audit_checkpoint
        from debias.io import sha256
        train(self.state,self.assets,self.folder/'C0','C0',2024,'test','cpu',engineering_max_batches=2)
        checkpoint=self.folder/'C0/best.pt';before=sha256(checkpoint)
        context=SimpleNamespace(name='toy',test=self.state.test,q=self.state.q,ada_mastery=self.state.mastery,
            correct={2:{0},3:{1}},fingerprint='toy',rule=lambda u,n:np.argsort(self.state.distance[u],kind='stable')[:n])
        audit_checkpoint(self.state,checkpoint,self.folder/'audit',context,
                         {'evaluation':np.array([[2,0,1],[3,2,3]],dtype=np.int64)},'cpu')
        self.assertEqual(before,sha256(checkpoint))
        self.assertTrue((self.folder/'audit/audit_complete.json').exists())
        self.assertFalse(read_json(self.folder/'audit/mechanism.json')['parameters_updated'])

    def test_partial_paper_report_is_not_marked_complete(self):
        from maed_v2 import report as reporting
        from debias.io import sha256
        result={'CFP':{'value':.5,'users':2},
                'top_n':{str(n):{'Ada':.7,'KG4Ex-NOV-Set':.8,'Ep_t0.5':.1,'DED':.3} for n in range(5,51,5)},
                'per_user':[{'uid':u,'N':n,'Ada':.7,'KG4Ex-NOV-Set':.8} for n in range(5,51,5) for u in (2,3)]}
        root=self.folder/'runs/pilot'
        paths=[self.folder/'runs/merit_frequency_20261022/toy/E0/seed2024/eval/metrics.json']
        paths += [root/f'toy/C0/seed2024/best/inference_{p}/eval/metrics.json' for p in ('A','B')]
        for path in paths:
            write_json(path,result);write_json(path.parent/'evaluation_complete.json',{'metrics_sha256':sha256(path)})
        with patch.object(reporting,'ROOT',self.folder),patch.object(reporting,'DATASETS',('toy',)):
            out=reporting.report(root)
        self.assertFalse(read_json(out/'report_manifest.json')['all_nine_stages_complete'])
        text=(out/'toy_one_metric_per_table.html').read_text(encoding='utf-8')
        self.assertIn('C0-A',text);self.assertIn('C0-B',text)


if __name__=='__main__':unittest.main()
