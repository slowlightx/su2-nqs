import numpy as np
import pytest

from su2_nqs.operators.projected_permutations import reynolds_permutation_bank, disjoint_swap_masks, evaluate_swap_bank


def test_reynolds_amplitudes_and_finite_step_norm_match_dense_reference():
    # Z4 translations, SU(2)-scalar S0.S1=P01/2-I/4 on four spins.
    group=np.asarray([np.roll(np.arange(4),k) for k in range(4)])
    p=np.asarray([[1,0,2,3],[0,1,2,3]])
    c=np.asarray([[.5],[-.25]])
    bank,weights,_=reynolds_permutation_bank(p,c,group)
    states=((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1).astype(np.int8)
    powers=1<<np.arange(3,-1,-1)
    projector=sum(np.eye(16)[states[:,g]@powers] for g in group)/len(group)
    o=.5*np.eye(16)[states[:,p[0]]@powers]-.25*np.eye(16)
    rng=np.random.default_rng(928)
    psi=projector@(rng.normal(size=16)+1j*rng.normal(size=16));psi/=np.linalg.norm(psi)
    lookup=lambda x:psi[x@powers]
    actual,_=evaluate_swap_bank(states,lookup,disjoint_swap_masks(bank),weights,sample_batch_size=3)
    expected=projector@o@psi
    np.testing.assert_allclose(actual[:,0],expected,atol=2e-14)
    for epsilon in (-.5,0,.5):
        dressed=psi+epsilon*actual[:,0]
        np.testing.assert_allclose(projector@dressed,dressed,atol=2e-14)
        norm=np.vdot(dressed,dressed).real
        exact=1+2*epsilon*np.vdot(psi,expected).real+epsilon**2*np.vdot(expected,expected).real
        assert abs(norm-exact)<2e-14
        np.testing.assert_allclose(np.sum(abs(dressed/np.sqrt(norm))**2),1,atol=2e-14)
    with pytest.raises(ValueError,match="closed"):
        reynolds_permutation_bank(p,c,group[:2])


def test_packed_swap_evaluation_matches_unpacked_local_environment():
    from su2_nqs.operators.environment_permutations import environment_permutation_bank,evaluate_environment_actions
    cells=((0,1,4,5),(2,3,6,7),(8,9,12,13),(10,11,14,15))
    p,c=environment_permutation_bank(16,cells)
    rng=np.random.default_rng(2026092760)
    lookup=rng.normal(size=65536)
    states=rng.integers(0,2,size=(11,16),dtype=np.int8)
    amplitude=lambda x:lookup[x@(1<<np.arange(15,-1,-1))]
    expected,_=evaluate_environment_actions(states,amplitude,p,c)
    actual,_=evaluate_swap_bank(states,amplitude,disjoint_swap_masks(p),c,sample_batch_size=2)
    np.testing.assert_allclose(actual,expected,atol=3e-14)
