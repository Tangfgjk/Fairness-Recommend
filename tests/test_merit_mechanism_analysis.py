import unittest
import numpy as np
import torch
from scripts.diagnose_merit_mechanisms import effects,compare_gradients,correlation


class AnalysisChecks(unittest.TestCase):
    def test_pure_common_item_component(self):
        z=np.array([[1.,2.,3.],[11.,12.,13.]])
        self.assertAlmostEqual(effects(z)['item_fraction_after_removing_user_offsets'],1.)
        self.assertAlmostEqual(effects(z)['interaction_SS'],0.)

    def test_pure_interaction(self):
        z=np.array([[1.,-1.],[-1.,1.]])
        self.assertAlmostEqual(effects(z)['item_fraction_after_removing_user_offsets'],0.)

    def test_gradient_weight_and_cosine(self):
        b=[torch.tensor([3.,4.])];f=[torch.tensor([-6.,-8.])]
        result=compare_gradients([('b',None)],b,f,.1)['ALL']
        self.assertAlmostEqual(result['weighted_fair_over_BCE'],.2)
        self.assertAlmostEqual(result['cosine'],-1.)

    def test_tied_spearman(self):
        self.assertAlmostEqual(correlation([1,1,2],[3,3,0]),-1.)
        self.assertIsNone(correlation([1,1],[1,2]))


if __name__=='__main__':unittest.main()
