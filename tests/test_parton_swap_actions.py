from itertools import combinations

import numpy as np

from su2_nqs.benchmarks.parton_swap_actions import ClosedShellSwapEvaluator
from su2_nqs.operators.projected_permutations import disjoint_swap_masks,evaluate_swap_bank


def test_rank_three_updates_match_direct_determinants_and_signs():
    rng=np.random.default_rng(2026092761)
    v=np.asarray([np.linalg.qr(rng.normal(size=(8,4)))[0] for _ in range(3)])
    weights=np.asarray([1.,-1.,-1.])/3
    states=[]
    for up in combinations(range(8),4):
        row=np.zeros(8,dtype=np.int8);row[list(up)]=1;states.append(row)
    states=np.asarray(states)
    def direct(batch):
        return np.asarray([(-1.)**np.flatnonzero(s).sum()*sum(w*np.linalg.det(x[np.flatnonzero(s)])*np.linalg.det(x[np.flatnonzero(1-s)]) for w,x in zip(weights,v)) for s in batch])
    permutations=[]
    for count in range(4):
        for _ in range(5):
            sites=rng.permutation(8);p=np.arange(8)
            for i,j in sites[:2*count].reshape(-1,2):p[i],p[j]=p[j],p[i]
            permutations.append(p)
    p=np.asarray(permutations);c=rng.normal(size=(len(p),2))
    expected,_=evaluate_swap_bank(states,direct,disjoint_swap_masks(p),c)
    evaluator=ClosedShellSwapEvaluator(v,weights,p,c,direct,chunk_size=32)
    actual,cost=evaluator(states)
    np.testing.assert_allclose(actual,expected,atol=4e-13,rtol=2e-11)
    assert cost["direct_fallback_count"]==0
    # Independent S^2 check on the complete N=8, Sz=0 support.
    from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
    spin_squared,integer_basis=spin_squared_matrix(8,n_up=4)
    positions={int(row@(1<<np.arange(7,-1,-1))):i for i,row in enumerate(states)}
    ordered=actual[[positions[int(x)] for x in integer_basis]]
    assert np.linalg.norm(spin_squared@ordered)/np.linalg.norm(ordered)<3e-12
    # Force safe direct evaluation, never regularize the determinant inverse.
    fallback=ClosedShellSwapEvaluator(v,weights,p,c,direct,chunk_size=32,condition_limit=1.01)
    observed,cost=fallback(states[:3])
    np.testing.assert_allclose(observed,expected[:3],atol=2e-14)
    assert cost["direct_fallback_count"]==3
