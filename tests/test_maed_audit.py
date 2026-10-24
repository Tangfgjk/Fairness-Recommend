import tempfile
import unittest
from pathlib import Path
import numpy as np

from scripts.audit_maed_seed2024 import score_change,novelty_change,paper_tables
from debias.io import TOP_NS,ROOT


class AuditChecks(unittest.TestCase):
    def test_score_decomposition_removes_offsets_and_recovers_shared_shift(self):
        base=np.array([[.1,.2,.3],[.4,.1,.2]])
        item=np.array([.3,-.2,-.1])
        new=base+item+np.array([[100.],[-8.]])
        stats,shared,residual=score_change(base,new)
        self.assertAlmostEqual(stats['shared_item_energy_fraction'],1)
        np.testing.assert_allclose(shared[0],item,atol=1e-12)
        np.testing.assert_allclose(residual,0,atol=1e-12)
        stats,_,_=score_change(base,base+np.array([[1.],[2.]]))
        # Near-zero floating residual is treated as an undefined energy ratio.
        self.assertIsNone(stats['shared_item_energy_fraction'])

    def test_individual_changes_do_not_look_like_a_common_item_shift(self):
        base=np.zeros((2,3));change=np.array([[1.,-1.,0.],[-1.,1.,0.]])
        stats,shared,residual=score_change(base,change)
        self.assertAlmostEqual(stats['individual_residual_energy_fraction'],1)
        np.testing.assert_array_equal(shared,0)
        np.testing.assert_array_equal(residual,change)

    def test_NOV_accounting_matches_actual_delta_for_random_sets(self):
        rng=np.random.default_rng(20261024)
        for _ in range(150):
            old=set(rng.choice(12,5,replace=False));new=set(rng.choice(12,7,replace=False))
            h=set(rng.choice(12,int(rng.integers(0,12)),replace=False))
            result,events=novelty_change(old,new,h)
            self.assertAlmostEqual(result['delta_NOV'],sum(r['NOV_contribution'] for r in events))
            self.assertAlmostEqual(result['delta_NOV'],result['known_overlap_contribution']+result['unseen_union_contribution'])

    def test_added_known_KC_reduces_NOV_and_unseen_KC_increases_NOV(self):
        result,events=novelty_change({0,2},{0,1,2,3},{0,1})
        known=next(e for e in events if e['kc']==1)
        unseen=next(e for e in events if e['kc']==3)
        self.assertLess(known['NOV_contribution'],0)
        self.assertGreater(unseen['NOV_contribution'],0)
        self.assertEqual(result['added_known_KCs'],1)
        self.assertEqual(result['added_unseen_KCs'],1)

    def test_exact_0p03_NOV_loss_is_not_counted_as_exceeding_budget(self):
        result,_=novelty_change(set(range(10)),set(range(13)),set(range(100)))
        self.assertLess(result['delta_NOV'],-.03)  # Reproduces the legacy float boundary issue.
        self.assertTrue(result['loss_eq_0p03_exact'])
        self.assertFalse(result['loss_gt_0p03_exact'])

    def test_rule_reference_is_not_bold_or_underlined_even_if_numerically_best(self):
        def result(ada,ded):
            return {'top_n':{str(n):{'Ada':ada,'DED':ded} for n in TOP_NS},'CFP':{'value':.7,'users':8}}
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            output=Path(name)
            rows=paper_tables('toy',{'E0':result(.6,.8),'M3':result(.7,.6),'M4':result(.8,.3),'R':result(.9,0)},output)
            text=(output/'toy_one_metric_per_table_with_R.html').read_text(encoding='utf-8')
            self.assertIn('<td><b>0.8000</b></td>',text)
            self.assertIn('<td><u>0.7000</u></td>',text)
            self.assertIn('<td class="reference">0.9000</td>',text)
            self.assertNotIn('<b>0.9000</b>',text)
            self.assertNotIn('<b>0.0000</b>',text)
            self.assertEqual(len(rows),len(TOP_NS)*4*2)


if __name__=='__main__':unittest.main()
