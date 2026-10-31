import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.replay_validation import compare_replay


def test_discrete_chains_rng_and_acceptance_cannot_use_float_tolerance():
    parent = dict(states=np.array([0, 1], np.int32), rng=np.array([2, 3], np.uint32), accepted=True)
    assert compare_replay(parent, parent)['passed']
    for name in ('states', 'rng', 'accepted'):
        changed = dict(parent)
        changed[name] = not parent[name] if name == 'accepted' else parent[name]+1
        assert not compare_replay(parent, changed)['passed']


def test_float_roundoff_is_reported_not_claimed_bitwise_exact():
    a = dict(amplitudes=np.array([1., 2.])*1e-150, energy=np.array([-10., -9.]))
    b = dict(amplitudes=a['amplitudes']*(1+1e-13), energy=a['energy']+1e-13)
    r = compare_replay(a, b)
    assert r['passed'] and not r['bitwise_exact']
    assert all(z['relative_vector_error'] < 2e-11 for z in r['leaves'])
    b['amplitudes'] = a['amplitudes']*1.001
    assert not compare_replay(a, b)['passed']


def test_shape_dtype_structure_nonfinite_and_zero_changes_fail_closed():
    a = np.zeros(2, np.float64)
    for b in (np.zeros(3), np.zeros(2, np.float32), np.array([np.nan, 0.]),
              np.array([np.inf, 0.]), np.array([1e-200, 0.])):
        assert not compare_replay(a, b)['passed']
    assert not compare_replay(dict(x=a), dict(y=a))['passed']
    for tolerance in (0, 3e-11, np.nan):
        with pytest.raises(ValueError):
            compare_replay(a, a, tolerance=tolerance)


def test_only_declared_scalar_diagnostics_receive_natural_unit_tolerance():
    a = (np.array([1e-80]), dict(residual=np.array(1e-14), accepted=True))
    b = (a[0], dict(residual=np.array(2e-14), accepted=True))
    assert not compare_replay(a, b)['passed']
    r = compare_replay(a, b, diagnostic_prefixes=('[1]',))
    assert r['passed'] and not r['bitwise_exact']
    assert not compare_replay(a, (a[0]*2, b[1]), diagnostic_prefixes=('[1]',))['passed']
    assert not compare_replay(a, (a[0], dict(residual=np.array(1e-8), accepted=True)),
                              diagnostic_prefixes=('[1]',))['passed']
