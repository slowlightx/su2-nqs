"""Independent exhaustive and node-safe fixed-proposal overlap checks."""
import numpy as np
import pytest
from su2_nqs.vmc.dual_sampled_sr import importance_overlap_diagnostics


def test_exhaustive_complex_overlap_with_nonuniform_proposal_and_nodes():
    a = np.array([0, 1+2j, -3j, .4])
    b = np.array([2j, 0, .7+1j, -.2j])
    q = np.array([.03, .47, .11, .39])
    measured = importance_overlap_diagnostics(a, b, q, observation_weights=q)
    expected = abs(np.vdot(a, b))**2/(np.vdot(a, a).real*np.vdot(b, b).real)
    assert abs(measured['fidelity_squared']-expected) < 3e-15
    assert abs(measured['second_over_first_norm_ratio']-np.vdot(b,b).real/np.vdot(a,a).real) < 3e-15
    # Q is not the first state's Born law; blindly averaging b/a is not legal.
    assert not measured['complete_support_certified']


def test_large_gauge_scales_unknown_Q_partition_and_global_phase():
    a = np.array([0, 2+1j, -3j, .5])
    q = np.array([1e-240, 2e-240, 4e-240, 3e-240])
    report = importance_overlap_diagnostics(1e-90*a, 2e-90j*a, q)
    assert abs(report['fidelity_squared']-1) < 3e-15
    assert abs(report['second_over_first_norm_ratio']-4) < 2e-12
    common = importance_overlap_diagnostics(a, 2j*a, q*1e200)
    assert abs(common['fidelity_squared']-report['fidelity_squared']) < 3e-15


@pytest.mark.parametrize('a,b,q', [([1], [1], [0]), ([1], [1], [-1]),
    ([np.nan], [1], [1]), ([1], [1], [np.inf]), ([1], [1], [1+0j]),
    ([0], [1], [1]), ([[1]], [[1]], [[1]])])
def test_bad_or_zero_norm_inputs_reject(a, b, q):
    with pytest.raises((ValueError, FloatingPointError)):
        importance_overlap_diagnostics(np.asarray(a), np.asarray(b), np.asarray(q))
