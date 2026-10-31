"""Work-ledger guards for bounded native hierarchy audits."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import (
    audit_cost_forecast, checked_active_widths, compare_saved_A1_amplitudes, finite_relative_error,
    paired_endpoint_moments,
)


def test_forecast_counts_full_H_centers_and_proposals_without_fake_sweeps():
    result = audit_cost_forecast(chains=8, records=4, burn=256, between=32,
                                 amplitude_batch_seconds=2., H_batch_seconds=30.)
    assert result["full_H_centers"] == result["sample_records"] == 32
    assert result["proposal_batches"] == 384
    assert result["estimated_sampling_seconds"] == 770.
    assert result["estimated_H_seconds"] == 120.
    assert result["conservative_seconds"] == 1335.
    assert result["forecast_is_not_equilibrium_certificate"]


@pytest.mark.parametrize("changes", [{"chains": True}, {"records": 0}, {"burn": 1.5},
    {"between": -1}, {"amplitude_batch_seconds": np.nan}, {"H_batch_seconds": 0.}])
def test_bad_counts_and_nonfinite_measured_timings_fail_before_running(changes):
    spec = dict(chains=8, records=4, burn=256, between=32,
                amplitude_batch_seconds=1., H_batch_seconds=2.)
    with pytest.raises(ValueError):
        audit_cost_forecast(**(spec | changes))


def test_active_width_metadata_uses_hierarchy_indices_not_insertion_order():
    assert checked_active_widths({"1": {"0": 1}, "0": {"0": 4, "2": 4}}, nodes=2) == ({0: 4, 2: 4}, {0: 1})
    assert checked_active_widths([{0: 4}, {0: 1}], nodes=2) == ({0: 4}, {0: 1})
    for bad in ({2: {0: 1}, 0: {0: 4}}, [4, 1], [{0: 1.5}, {0: 1}],
                [{.5: 4}, {0: 1}], [{True: 4}, {0: 1}], [{"0.5": 4}, {0: 1}]):
        with pytest.raises(ValueError):
            checked_active_widths(bad, nodes=2)


def test_replay_comparison_does_not_pass_NaN_or_silent_shape_broadcast():
    assert finite_relative_error(np.array([1+2j]), np.array([1+2j])) == 0
    for observed, expected in [([np.nan], [1.]), ([1.], [np.inf]),
                                ([[1.]], [1.]), ([], [])]:
        with pytest.raises(FloatingPointError):
            finite_relative_error(np.asarray(observed), np.asarray(expected))


@pytest.mark.parametrize('field', ['overlap_samples_sha256', 'overlap_candidate_sha256'])
def test_overlap_rejects_changed_registered_inputs_before_evaluation(tmp_path, field):
    samples = tmp_path/'samples.npz'; samples.write_bytes(b'test input')
    candidate = tmp_path/'candidate.pkl'; candidate.write_bytes(b'candidate input')
    spec = dict(overlap_samples_path=str(samples), overlap_candidate_checkpoint=str(candidate))
    spec[field] = '0'*64
    with pytest.raises(ValueError, match='fixed overlap input SHA mismatch'):
        compare_saved_A1_amplitudes(spec, None, None, None, None, tmp_path, {}, None)


def test_paired_endpoint_moments_match_complete_physical_support_including_target_nodes():
    rng = np.random.default_rng(2610237100)
    # Independent finite four-spin operator, with offdiagonal hopping into
    # exact nodes. Every physical bit configuration is evaluated once.
    h = np.diag(np.arange(16)/16).astype(complex)
    for a in range(16):
        b = a ^ 3
        h[a,b] = .3
    p = rng.normal(size=(16,2))+1j*rng.normal(size=(16,2))
    p[:3,0] = 0.; p[7:9,1] = 0.
    p /= np.linalg.norm(p,axis=0)
    hp = h@p
    q = np.full(16,1/16);groups = np.arange(16)%4
    report = paired_endpoint_moments(p,hp,q,groups,n_sites=4)
    energy = np.einsum('ij,ij->j',p.conj(),hp)
    variance = np.sum(abs(hp-energy*p)**2,axis=0)
    expected = np.stack((energy.real/4,energy.imag/4,variance/4),axis=1)
    np.testing.assert_allclose(report['values'],expected,atol=2e-16)
    assert report['exact_node_records'] == [3,2]
    assert report['guide_normalization_known'] is False
    assert report['finite_burn_equilibrium_certified'] is False
    # Unknown common normalization cancels; endpoint global phases cancel.
    again = paired_endpoint_moments(p*np.exp(1j*np.array([.2,.7])),
        hp*np.exp(1j*np.array([.2,.7])),17*q,groups,n_sites=4)
    np.testing.assert_allclose(report['values'],again['values'],atol=2e-16)
    np.testing.assert_allclose(report['paired_whole_chain_jackknife_SE'],
                               again['paired_whole_chain_jackknife_SE'],atol=2e-16)


def test_common_records_give_zero_paired_error_for_identical_endpoints():
    p = np.column_stack((np.arange(1,17),np.arange(1,17))).astype(complex)
    hp = p*np.arange(16)[:,None]/20
    report=paired_endpoint_moments(p,hp,np.ones(16),np.arange(16)%4,n_sites=4)
    np.testing.assert_array_equal(report['candidate_minus_control'],np.zeros(3))
    np.testing.assert_array_equal(report['paired_whole_chain_jackknife_SE'],np.zeros(3))


@pytest.mark.parametrize('changes',[
    {'density':np.zeros(16)}, {'density':np.full(16,np.nan)},
    {'groups':np.zeros(16)}, {'psi':np.ones((16,3))},
])
def test_paired_endpoint_guards_fail_before_any_simulation(changes):
    args=dict(psi=np.ones((16,2)),hpsi=np.ones((16,2)),
              density=np.ones(16),groups=np.arange(16)%4,n_sites=4)
    with pytest.raises(ValueError):paired_endpoint_moments(**(args|changes))
