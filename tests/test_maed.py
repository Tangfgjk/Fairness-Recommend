import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch

from maed.state import EducationState, education_distances
from maed.scoring import EducationScale, fuse_logits
from maed.exposure import smooth_top_n, rule_targets, exposure_loss, hard_exposure
from maed.evaluation import loss_statistics, quality_budgets
from maed.io import lock_json


def toy_state():
    distance = np.array([[.1,.2,.3,.4], [.3,.2,.1,.4], [900.,800.,700.,600.]])
    return EducationState('toy',None,None,None,None,None,None,distance,np.array([0,1]),np.array([2]),{})


class MAEDChecks(unittest.TestCase):
    def test_scale_is_fit_on_training_users_only(self):
        state = toy_state()
        scale = EducationScale.fit(state)
        expected = np.median(state.distance[state.train])
        self.assertEqual(scale.center,expected)
        state.distance[state.test] *= 100
        self.assertEqual(EducationScale.fit(state),scale)

    def test_no_test_students_or_duplicate_weighting_in_target(self):
        state = toy_state()
        for users in ([0,2],[0,0],[]):
            with self.assertRaises(ValueError):
                state.training_distance(users)

    def test_beta_zero_is_exact_bypass_and_positive_beta_preserves_gradient(self):
        z = torch.tensor([[.1,.2,.3,.4]],dtype=torch.double,requires_grad=True)
        d = torch.tensor([[.4,.3,.2,.1]],dtype=torch.double)
        scale = EducationScale.fit(toy_state())
        self.assertIs(fuse_logits(z,d,scale,0),z)
        out = fuse_logits(z,d,scale,.5)
        out.sum().backward()
        self.assertTrue(torch.equal(z.grad,torch.ones_like(z)))
        self.assertGreater(float((out[0,3]-z[0,3]).detach()),float((out[0,0]-z[0,0]).detach()))

    def test_product_distance_is_original_formula(self):
        q=np.array([[1,1],[0,1]])
        m=np.array([[.8,.5]])
        f=np.array([[.8,.3]])
        result=education_distances(m,f,q)
        np.testing.assert_allclose(result,[[.4,np.sqrt(.3**2+.5**2)]])

    def test_soft_topn_implicit_gradient_matches_finite_difference(self):
        z=torch.tensor([[.2,1.,-.4,.7],[.8,.5,.2,-.3]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:smooth_top_n(x,2,.7), (z,), atol=1e-5,rtol=1e-4))

    def test_mass_and_shift_invariance_with_saturated_and_tied_scores(self):
        z=torch.tensor([[1.,1.,1.,1.],[1000.,-1000.,2.,-2.]],dtype=torch.double,requires_grad=True)
        w=smooth_top_n(z,2,.5)
        self.assertTrue(torch.allclose(w.sum(1),torch.tensor([2.,2.],dtype=torch.double),atol=1e-10))
        self.assertTrue(torch.allclose(w,smooth_top_n(z+100,2,.5),atol=1e-10))
        (w*torch.arange(4)).sum().backward()
        self.assertTrue(torch.allclose(z.grad.sum(1),torch.zeros(2,dtype=torch.double),atol=1e-12))

    def test_small_temperature_approaches_hard_topn(self):
        z=torch.tensor([[4.,3.,1.,0.]],dtype=torch.double)
        self.assertTrue(torch.allclose(smooth_top_n(z,2,.01),torch.tensor([[1.,1.,0.,0.]],dtype=torch.double),atol=1e-8))

    def test_per_n_rule_targets_have_equal_student_mass(self):
        d=np.array([[0.,1.,2.,3.],[3.,2.,1.,0.]])
        target=rule_targets(d,(1,2))
        np.testing.assert_allclose(target,[[.5,0,0,.5],[.25,.25,.25,.25]])
        np.testing.assert_allclose(hard_exposure(-d,(1,2)),target)

    def test_exposure_gradient_reduces_actual_hard_mismatch_in_toy(self):
        d=np.array([[.1,.3,.4,.5],[.5,.4,.3,.1]])
        target=rule_targets(d,(1,2))
        z=torch.tensor([[3.,2.,1.,0.],[.6,.5,.4,.3]],dtype=torch.double,requires_grad=True)
        before=.5*np.abs(hard_exposure(z.detach().numpy(),(1,2))-target).sum()
        for _ in range(30):
            loss,_=exposure_loss(z,target,(1,2),1.)
            (g,)=torch.autograd.grad(loss,z)
            z=(z.detach()-2*g).requires_grad_(True)
        after=.5*np.abs(hard_exposure(z.detach().numpy(),(1,2))-target).sum()
        self.assertLess(after,before)

    def test_global_exposure_loss_cannot_break_identical_user_scores_by_itself(self):
        z=torch.tensor([[3.,2.,1.,0.],[3.,2.,1.,0.]],dtype=torch.double,requires_grad=True)
        target=rule_targets(np.array([[.1,.3,.4,.5],[.5,.4,.3,.1]]),(1,2))
        loss,_=exposure_loss(z,target,(1,2),1.)
        (gradient,)=torch.autograd.grad(loss,z)
        self.assertTrue(torch.equal(gradient[0],gradient[1]))

    def test_quality_risk_includes_zeros_and_uses_strict_harm_threshold(self):
        result=loss_statistics(np.array([-.01,-.2]+[.5]*18))
        self.assertEqual(result['harm_count'],1)
        self.assertEqual(result['tail_students'],2)
        self.assertAlmostEqual(result['Worst10PercentLoss'],.105)
        self.assertTrue(result['population_quality_pass'])

    def test_soft_topn_rejects_bad_dimensions_and_temperatures(self):
        for n,t in [(0,1),(4,1),(2,0),(2,float('nan'))]:
            with self.assertRaises(ValueError):
                smooth_top_n(torch.zeros((2,4)),n,t)

    def test_resume_contract_rejects_changed_inputs_without_overwriting(self):
        scratch=Path(__file__).resolve().parents[1]/'tmp'
        scratch.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)/'contract.json'
            lock_json(p,{'input_hash':'one'})
            saved=p.read_bytes()
            lock_json(p,{'input_hash':'one'})
            with self.assertRaises(ValueError):
                lock_json(p,{'input_hash':'two'})
            self.assertEqual(p.read_bytes(),saved)

    def test_new_quality_budgets_are_absolute_and_metric_specific(self):
        budgets=quality_budgets()
        self.assertEqual(budgets,{'Ada':.01,'KG4Ex-NOV-Set':.03})
        # NOV 0.95 -> 0.92 is permitted: 0.03 absolute, not 3% of 0.95.
        result=loss_statistics(np.array([.92-.95]),budget=budgets['KG4Ex-NOV-Set'])
        self.assertTrue(result['population_quality_pass'])
        self.assertFalse(loss_statistics(np.array([-.0301]),budget=.03)['population_quality_pass'])
        self.assertFalse(loss_statistics(np.array([-.011]),budget=budgets['Ada'])['population_quality_pass'])

    def test_population_budget_does_not_relax_student_harm_reporting(self):
        result=loss_statistics(np.array([-.02,0.]),budget=.03)
        self.assertTrue(result['population_quality_pass'])
        self.assertEqual(result['harm_count'],1)
        self.assertEqual(result['individual_budget_exceed_count'],0)

    def test_incomplete_or_invalid_metric_budgets_are_rejected(self):
        for budgets in ({'Ada':.01},{'Ada':.01,'KG4Ex-NOV-Set':float('nan')},{'Ada':-.01,'KG4Ex-NOV-Set':.03}):
            with self.assertRaises(ValueError):
                quality_budgets(budgets)


if __name__=='__main__':
    unittest.main()
