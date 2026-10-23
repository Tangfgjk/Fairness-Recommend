import unittest
import numpy as np
import torch

from maed.exposure import smooth_top_n,rule_targets,exposure_loss
from maed.boundary_exposure import multi_top_n,calibrate_temperatures,calibrated_exposure_loss
from maed.m2_analysis import cosine_matrix,shared_item_fraction,change_counts,trajectory


class BoundaryExposureChecks(unittest.TestCase):
    def test_vectorized_proxy_matches_v0_values_and_gradients(self):
        z=torch.tensor([[4.,2.,1.,0.,-.5],[2.,1.,.5,.2,-.1]],dtype=torch.double,requires_grad=True)
        ns=(1,3);temperature=.7
        w=multi_top_n(z,ns,[temperature]*2)
        old=torch.stack([smooth_top_n(z,n,temperature) for n in ns])
        self.assertTrue(torch.allclose(w,old,atol=1e-12))
        coefficients=torch.arange(w.numel(),dtype=torch.double).reshape(w.shape)
        g=torch.autograd.grad((w*coefficients).sum(),z,retain_graph=True)[0]
        old_g=torch.autograd.grad((old*coefficients).sum(),z)[0]
        self.assertTrue(torch.allclose(g,old_g,atol=1e-11))

    def test_vectorized_boundary_gradient_matches_finite_difference(self):
        z=torch.tensor([[1.,.7,.1,-.2]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:multi_top_n(x,(1,2),[.1,.2]),(z,),atol=1e-5))

    def test_boundary_calibration_tracks_rank_gap_without_test_data(self):
        z=np.array([[5.,4.9,0.,-1.],[7.,6.8,1.,0.]])
        c=calibrate_temperatures(z,(1,2))
        self.assertAlmostEqual(c['per_N'][0]['temperature'],.15)
        self.assertAlmostEqual(c['per_N'][1]['temperature'],5.35)
        self.assertEqual(c['training_users'],2)

    def test_boundary_temperature_aligns_soft_mass_with_real_topn(self):
        # Reproduces the wide-temperature symptom: catalog mass spreads far from Top1.
        z=torch.linspace(0,1,100,dtype=torch.double).reshape(1,-1)
        wide=multi_top_n(z,(1,),[1.])
        c=calibrate_temperatures(z.numpy(),(1,))
        narrow=multi_top_n(z,(1,),[c['per_N'][0]['temperature']])
        self.assertLess(float(wide[0,0,-1]),.1)
        self.assertGreater(float(narrow[0,0,-1]),.5)
        self.assertTrue(torch.allclose(narrow.sum(-1),torch.ones((1,1),dtype=torch.double),atol=1e-10))

    def test_cosines_and_shared_effect_match_known_patterns(self):
        self.assertEqual(cosine_matrix([[1.,-1.],[-1.,1.]])[0][1],-1.)
        self.assertIsNone(cosine_matrix([[0.,0.],[0.,1.]])[0][1])
        self.assertAlmostEqual(shared_item_fraction([[1.,-1.],[1.,-1.]]),1.)
        self.assertAlmostEqual(shared_item_fraction([[1.,-1.],[-1.,1.]]),0.)

    def test_change_counts_count_exercises_not_only_students(self):
        a=np.array([[4.,3.,2.,1.]])
        b=np.array([[1.,2.,3.,4.]])
        changed,entered=change_counts(a,b,(1,2))
        self.assertEqual(changed,[1,1]);self.assertEqual(entered,[1,2])

    def test_training_guard_never_increases_guarded_hard_metrics(self):
        ns=(1,2)
        d=np.array([[0.,1.,2.,3.],[3.,2.,1.,0.]])
        z=torch.tensor([[3.,2.,1.,0.],[.6,.5,.4,.3]],dtype=torch.double)
        result=trajectory(z,rule_targets(d,ns),[.5,.5],1.,steps=4,guarded=True,ns=ns)
        t=result['trace']
        for a,b in zip(t,t[1:]):
            self.assertLess(b['soft_mean_TV'],a['soft_mean_TV'])
            self.assertLessEqual(np.mean(b['hard_TV']),np.mean(a['hard_TV'])+1e-12)
            self.assertLessEqual(b['hard_TV'][1],a['hard_TV'][1]+1e-12)


if __name__=='__main__':unittest.main()
