"""Cached audit diagnostics: no training, new queries, or data selection."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_saved_state_physics import (
    _cached_early_selector, _cached_moment_sequence, _cached_residual_tails, _regional,
    _independent_difference,
)


def test_cached_direct_moments_keep_nodes_and_explicit_chain_ids():
    psi=np.array([1.,0.,-2.,3.,1.,.2,-1.,2.])
    action=np.array([.3,4.,1.,-2.,.7,.5,.8,-.3])
    q=np.linspace(.2,1.8,len(psi));groups=np.array([10,20,30,40]*2)
    result=_cached_moment_sequence({'previous_endpoint':(psi,action),
        'endpoint':(2*psi,2*action)},q,groups,n_sites=4)
    energy=(psi@ (action/q))/np.sum(psi**2/q)
    residual=np.sum((action-energy*psi)**2/q)/np.sum(psi**2/q)
    assert result['states']['endpoint']['energy_per_site']['value']==pytest.approx(energy/4,abs=2e-15)
    assert result['states']['endpoint']['variance_per_site']['value']==pytest.approx(residual/4,abs=2e-15)
    for name in ('energy_per_site','variance_per_site'):
        assert result['paired']['endpoint-minus-previous_endpoint'][name]['value']==pytest.approx(0.,abs=2e-15)
        assert result['paired']['endpoint-minus-previous_endpoint'][name]['chain_SE']<2e-15


def test_cached_residual_tail_concentration_is_node_safe_and_scale_invariant():
    psi=np.array([1.,0.,-2.,3.,1.,.2,-1.,2.])
    action=np.array([.3,4.,1.,-2.,.7,.5,.8,-.3])
    q=np.linspace(.2,1.8,len(psi));groups=np.array([10,20,30,40]*2)
    got=_cached_residual_tails(psi,action,q,groups,n_sites=4)
    energy=(psi@(action/q))/np.sum(psi**2/q)
    contributions=(action-energy*psi)**2/q
    assert got['largest_record_residual_share']==pytest.approx(contributions.max()/contributions.sum(),abs=2e-15)
    assert got['residual_share_on_exact_nodes']>0
    other=_cached_residual_tails(3j*psi,3j*action,7*q,groups,n_sites=4)
    for name in ('variance_per_site','energy_per_site','top_1_percent_residual_share',
            'largest_record_residual_share','residual_share_on_exact_nodes'):
        assert other[name]==pytest.approx(got[name],abs=2e-14)
    exact=_cached_residual_tails(psi,2*psi,q,groups,n_sites=4)
    assert exact['variance_per_site']==0 and exact['largest_record_residual_share']==0


def test_cached_halves_are_temporal_within_each_independent_chain():
    groups=np.tile([10,20,30,40],8)
    selector=_cached_early_selector(groups)
    np.testing.assert_array_equal(selector,np.arange(32)<16)
    # A later independent stream must not be called a later part of stream one.
    groups=np.concatenate([groups,groups+100])
    selector=_cached_early_selector(groups)
    np.testing.assert_array_equal(selector,np.tile(np.arange(32)<16,2))
    with pytest.raises(ValueError):
        _cached_early_selector(np.array([0,1,0,1,0,1]))


def test_compact4_has_legal_joint_weights_but_no_six_spin_observables():
    result=_regional(np.eye(16)/16,[0,1],compact=False)
    assert result['spin_squared']==pytest.approx(3.,abs=2e-14)
    assert sum(result[k] for k in result if k.startswith('p_q'))==pytest.approx(1.,abs=2e-14)
    assert 'high_q3' not in result and 'three_bridge_connected' not in result


def test_different_audits_add_variances_instead_of_artificial_pairing():
    before={'p':dict(value=.2,chain_SE=.03,jackknife_values=[.18,.22])}
    after={'p':dict(value=.1,chain_SE=.04,jackknife_values=[.08,.12])}
    change=_independent_difference(before,after)['p']
    assert change['value']==pytest.approx(-.1)
    assert change['independent_chain_SE']==pytest.approx(.05)
    assert change['absolute_signal_over_SE']==pytest.approx(2.)
    assert change['paired'] is False


def test_six_spin_high_part_matches_independent_low_order_projector_split():
    from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
    rng=np.random.default_rng(371)
    a=rng.normal(size=(64,7))+1j*rng.normal(size=(64,7))
    rho=a@a.conj().T; rho/=np.trace(rho)
    regional=_regional(rho,[0,1,2],compact=False)
    for q in (1,3):
        reference=joint_spin_operator_split(6,[0,1,2],(q,q,0)).measure(rho)
        assert regional[f'high_q{q}']==pytest.approx(reference['high'],abs=2e-14)
