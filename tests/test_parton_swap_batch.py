from itertools import combinations

import numpy as np

from su2_nqs.benchmarks.parton_swap_actions import ClosedShellSwapEvaluator
from su2_nqs.benchmarks.parton_swap_batch import ScannedClosedShellSwapEvaluator
from su2_nqs.operators.projected_permutations import disjoint_swap_masks, evaluate_swap_bank


def test_scanned_batch_matches_direct_and_sealed_evaluator_with_partial_batches():
    rng = np.random.default_rng(2026092780)
    orbitals = np.asarray([np.linalg.qr(rng.normal(size=(8, 4)))[0] for _ in range(3)])
    weights = np.array([1., -1., -1.])/3
    states = np.zeros((70, 8), dtype=np.int8)
    for row, up in zip(states, combinations(range(8), 4)):
        row[list(up)] = 1
    def direct(batch):
        return np.asarray([(-1.)**np.flatnonzero(s).sum()*sum(w*np.linalg.det(v[np.flatnonzero(s)])*np.linalg.det(v[np.flatnonzero(1-s)]) for w,v in zip(weights,orbitals)) for s in batch])
    permutations = []
    for i in range(23):
        p = np.arange(8); sites = rng.permutation(8)
        for a,b in sites[:2*(i%4)].reshape(-1,2):
            p[a],p[b] = p[b],p[a]
        permutations.append(p)
    p = np.asarray(permutations); c = rng.normal(size=(len(p), 2))
    expected,_ = evaluate_swap_bank(states,direct,disjoint_swap_masks(p),c)
    reference = ClosedShellSwapEvaluator(orbitals,weights,p,c,direct,chunk_size=8)
    for size in (1,4,7):
        evaluator = ScannedClosedShellSwapEvaluator(orbitals,weights,p,c,direct,chunk_size=8,sample_batch_size=size)
        observed,cost = evaluator(states)
        np.testing.assert_allclose(observed,expected,atol=5e-13,rtol=2e-11)
        np.testing.assert_allclose(observed,reference(states)[0],atol=5e-13,rtol=2e-11)
        assert cost['direct_fallback_count']==0
    fallback = ScannedClosedShellSwapEvaluator(orbitals,weights,p,c,direct,chunk_size=8,condition_limit=1.01)
    observed,cost = fallback(states[:3])
    np.testing.assert_allclose(observed,expected[:3],atol=2e-14)
    assert cost['direct_fallback_count']==3
