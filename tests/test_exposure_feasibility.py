import unittest
from types import SimpleNamespace
import numpy as np
from scripts.merit_exposure_feasibility import allocate,ranking_scores,prefix_quality,bits,quality_arrays


def toy():
    q=np.array([[0,1],[0,1],[0,1],[0,1],[1,0],[0,1]],dtype=int)
    distances=np.array([[.1,.2,.3,.4,.5,.6],[.9,.8,.3,.4,.1,.2]])
    class Context:
        test=np.array([0,1])
        correct={0:set(),1:set()}
        ada_mastery={0:{0:.8,1:.8},1:{0:.8,1:.8}}
        def rule(self,u,n):return np.argsort(self.distance[u],kind='stable')[:n]
    c=Context();c.q=q;c.distance=distances
    return c,np.tile(np.arange(6,0,-1,dtype=float),(2,1))


class FeasibilityChecks(unittest.TestCase):
    def test_ordinal_serialization_complete_and_prefix_consistent(self):
        full=np.array([[0,1,2,3,4,5]])
        top=np.array([[4,1,0,3]])
        scores=ranking_scores(full,top)
        ranked=np.argsort(-scores,axis=1,kind='stable')
        self.assertEqual(ranked[0,:4].tolist(),top[0].tolist())
        self.assertEqual(len(set(ranked[0])),6)

    def test_fair_allocation_reduces_exact_exposure_with_zero_quality_budget(self):
        c,s=toy()
        new,a=allocate(c,s,4,'FQE',sweeps=5,budget=0,ns=(2,4))
        self.assertGreater(a['accepted_swaps'],0)
        before=np.argsort(-s,axis=1)[:,:4];after=np.argsort(-new,axis=1)[:,:4]
        rules=np.array([c.rule(u,4) for u in c.test])
        ada=quality_arrays(c);qb=[bits(q) for q in c.q]
        for i in range(2):
            ap,np_=prefix_quality(before[i],ada[i],qb,0,(2,4))
            aq,nq=prefix_quality(after[i],ada[i],qb,0,(2,4))
            self.assertTrue(np.all(aq>=ap-1e-12))
            self.assertTrue(np.all(nq>=np_-1e-12))
        for n in (2,4):
            ref=np.bincount(rules[:,:n].ravel(),minlength=6)
            initial=np.abs(np.bincount(before[:,:n].ravel(),minlength=6)-ref).sum()
            final=np.abs(np.bincount(after[:,:n].ravel(),minlength=6)-ref).sum()
            self.assertLess(final,initial)

    def test_actual_novelty_budget_is_enforced(self):
        c,s=toy();c.correct={0:{0},1:{0}}
        new,a=allocate(c,s,4,'FQE',sweeps=5,budget=.01,ns=(2,4))
        ada=quality_arrays(c);qb=[bits(q) for q in c.q]
        before=np.argsort(-s,axis=1)[:,:4];after=np.argsort(-new,axis=1)[:,:4]
        old=[];final=[]
        for i in range(2):
            old.append(prefix_quality(before[i],ada[i],qb,1,(2,4))[1])
            final.append(prefix_quality(after[i],ada[i],qb,1,(2,4))[1])
        self.assertTrue(np.all(np.mean(final,axis=0)>=np.mean(old,axis=0)-.01-1e-12))


if __name__=='__main__':unittest.main()
