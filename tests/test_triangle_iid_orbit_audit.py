"""Independent finite-Hilbert moments, retaining exact target-zero outcomes."""
import numpy as np
import pytest
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import (
    common_active_profile, paired_iid_moments,
)


def test_complete_H_squared_at_nodes_and_paired_ratios_match_physical_vectors():
    psi=np.array([[1.,0.],[0.,1.],[1j,1.],[0.,1j]],complex)
    psi/=np.linalg.norm(psi,axis=0)
    h=np.array([[.2,1.,0.,.3],[1.,-.1,.4,0.],[0.,.4,.6,-.2j],[.3,0.,.2j,-.3]])
    hp=h@psi
    # Exhaustive uniform-Q quadrature repeated identically in each group.
    rows=np.tile(np.arange(4),8);groups=np.repeat(np.arange(8),4)
    result=paired_iid_moments(psi[rows],hp[rows],np.full(32,.25),groups,n_sites=2,seed=73)
    expected=[]
    for k in range(2):
        energy=np.vdot(psi[:,k],hp[:,k]);variance=np.linalg.norm(hp[:,k]-energy*psi[:,k])**2
        expected.append([energy.real/2,energy.imag/2,variance/2,1.])
    np.testing.assert_allclose(result['endpoint_values'],expected,atol=8e-15)
    assert all(x>0 for x in result['target_node_H2_contributions'])
    np.testing.assert_allclose(result['candidate_minus_control'],np.asarray(expected)[1]-expected[0],atol=8e-15)
    np.testing.assert_allclose(result['paired_independent_block_jackknife_SE'],0.,atol=8e-15)


def test_nonfinite_or_zero_proposal_is_rejected_not_dropped():
    p=np.ones((16,2));q=np.ones(16);groups=np.repeat(np.arange(4),4)
    for invalid in (0.,np.nan):
        changed=q.copy();changed[0]=invalid
        with pytest.raises(ValueError):paired_iid_moments(p,p,changed,groups,n_sites=2,seed=1)


def test_union_profile_preserves_both_endpoints_without_mutating_them():
    old=[{0:8,2:8},{0:1}];new=[{0:10,2:6,4:4},{0:1}]
    assert common_active_profile([old,new])==[{0:10,2:8,4:4},{0:1}]
    assert old==[{0:8,2:8},{0:1}]
    assert new==[{0:10,2:6,4:4},{0:1}]
    for invalid in ([],[[]],[old,new[:1]]):
        with pytest.raises(ValueError):common_active_profile(invalid)
