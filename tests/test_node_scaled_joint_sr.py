"""Independent dense oracles for full-native, node-scaled cached SR."""
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import pytest

from su2_nqs.vmc.cached_amplitude_operators import CachedAmplitudeJacobian, solve_cached_projected_sr
from su2_nqs.vmc.node_scaled_joint_sr import solve_cached_node_scaled_joint_sr


def problem(seed=61, rows=13, parameters=19):
    rng = np.random.default_rng(seed)
    jac = rng.normal(size=(rows, parameters))
    psi = rng.normal(size=rows)
    psi[2] = 0.  # All native derivatives at this target node must remain.
    hp = rng.normal(size=rows)
    q = np.exp(rng.normal(size=rows))
    return jac, psi, hp, q


def cache_for(jac, psi, q):
    return CachedAmplitudeJacobian(np.asarray(jac), np.asarray(psi), np.asarray(q), dict(complete=True))


def solve(jac, psi, hp, q, blocks, **kwargs):
    return solve_cached_node_scaled_joint_sr(cache_for(jac, psi, q), psi, hp, q,
        blocks=blocks, maximum_host_bytes=10**8, maximum_device_bytes=10**8,
        parameter_chunk_size=kwargs.pop('parameter_chunk_size', 4), **kwargs)


def dense(jac, psi, hp, q, blocks, relative=.5, floor_relative=1e-12):
    scale = 1/np.sqrt(q*np.sum(psi**2/q))
    unit = psi*scale
    a = (np.eye(len(psi))-np.outer(unit, unit))@(jac*scale[:, None])
    energy = np.sum(psi*hp/q)/np.sum(psi**2/q)
    force = 2*(hp-energy*psi)*scale
    eigen_scales = [np.trace(a[:, first:final]@a[:, first:final].T)/(len(psi)-1)
                    for _, first, final in blocks]
    floor = floor_relative*max(eigen_scales)
    ridges = [max(relative*s, floor) for s in eigen_scales]
    lam = np.concatenate([np.full(final-first, ridge)
                          for (_, first, final), ridge in zip(blocks, ridges)])
    gradient = a.T@force
    direction = np.linalg.solve(a.T@a+np.diag(lam), gradient)
    return direction, gradient, a, force, lam


@pytest.mark.parametrize('chunk', [1, 4, 64])
def test_exact_dense_full_fisher_plus_node_ridge_all_columns(chunk):
    jac, p, hp, q = problem()
    blocks = [('low/0', 0, 3), ('top/0', 3, 11), ('root', 11, 19)]
    expected, gradient, a, _, lam = dense(jac, p, hp, q, blocks)
    direction, actual_gradient, report = solve(jac, p, hp, q, blocks, parameter_chunk_size=chunk)
    np.testing.assert_allclose(direction, expected, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(actual_gradient, gradient, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(report['chosen_direction_tangent'], a@expected, atol=3e-13)
    assert report['relative_linear_residual'] < 1e-12
    assert report['dual_relative_linear_residual'] < 1e-12
    assert report['full_global_Fisher_solved'] and report['all_native_columns_retained']
    assert report['all_records_retained'] and report['no_AD_recomputed_in_solve']
    assert report['exact_node_records'] == 1 and report['exact_Jacobian_passes'] == 3
    # A block-diagonal Fisher update is a genuinely different algorithm.
    block_d = np.zeros(19)
    for _, first, final in blocks:
        ab = a[:, first:final]
        block_d[first:final] = np.linalg.solve(ab.T@ab+np.diag(lam[first:final]), gradient[first:final])
    assert np.linalg.norm(a@(block_d-expected)) > .01


def test_single_node_uniform_ridge_reduces_to_existing_exact_global_sr():
    jac, p, hp, q = problem()
    direction, gradient, report = solve(jac, p, hp, q, [('all', 0, 19)])
    old_direction, old_gradient, old_report = solve_cached_projected_sr(cache_for(jac, p, q), p, hp, q,
        damping=report['blocks'][0]['ridge'], maximum_device_bytes=10**8, maximum_host_bytes=10**8,
        parameter_chunk_size=5)
    np.testing.assert_allclose(direction, old_direction, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(gradient, old_gradient, atol=2e-13)
    np.testing.assert_allclose(report['raw_fisher_norm'], old_report['raw_fisher_norm'], atol=2e-13)


def test_node_scalar_reparameterization_and_parent_normalization_gauge():
    jac, p, hp, q = problem()
    blocks = [('a', 0, 3), ('b', 3, 11), ('c', 11, 19)]
    d, _, original = solve(jac, p, hp, q, blocks)
    coordinate_scale = np.repeat([.2, 3., 7.], [3, 8, 8])
    changed, _, rescaled = solve(jac*coordinate_scale, p, hp, q, blocks)
    np.testing.assert_allclose(changed*coordinate_scale, d, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(rescaled['chosen_direction_tangent'], original['chosen_direction_tangent'], atol=1e-12)
    gauge = p[:, None]*np.linspace(-3., 2., 19)[None, :]
    gd, _, gauged = solve(jac+gauge, p, hp, q, blocks)
    np.testing.assert_allclose(gd, d, atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(gauged['chosen_direction_tangent'], original['chosen_direction_tangent'], atol=1e-12)
    scaled_d, _, scaled = solve(jac*11., p*11., hp*11., q*.017, blocks)
    np.testing.assert_allclose(scaled_d, d, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(scaled['chosen_direction_tangent'], original['chosen_direction_tangent'], atol=1e-12)


def test_zero_node_is_retained_with_disclosed_floor_and_zero_direction():
    jac, p, hp, q = problem()
    jac[:, :3] = 0
    blocks = [('zero', 0, 3), ('active', 3, 19)]
    direction, _, report = solve(jac, p, hp, q, blocks)
    np.testing.assert_array_equal(np.asarray(direction)[:3], np.zeros(3))
    assert report['blocks'][0]['numerical_floor_active']
    assert report['blocks'][0]['ridge'] == report['numerical_ridge_floor'] > 0
    with pytest.raises(ArithmeticError, match='all native blocks'):
        solve(jac*0, p, hp, q, blocks)


@pytest.mark.parametrize('kind', ['gap', 'duplicate_label', 'missing', 'zero_Q', 'negative_Q', 'nonfinite_J'])
def test_invalid_partition_or_support_rejected_before_solving(kind):
    jac, p, hp, q = problem()
    blocks = [('a', 0, 3), ('b', 3, 19)]
    if kind == 'gap': blocks = [('a', 0, 3), ('b', 4, 19)]
    if kind == 'duplicate_label': blocks = [('a', 0, 3), ('a', 3, 19)]
    if kind == 'missing': blocks = [('a', 0, 18)]
    if kind == 'zero_Q': q[2] = 0.
    if kind == 'negative_Q': q[0] = -1.
    if kind == 'nonfinite_J': jac[3, 10] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):
        solve(jac, p, hp, q, blocks)


def test_budget_rejected_without_materializing_parameter_square():
    jac, p, hp, q = problem()
    with pytest.raises(MemoryError, match='host'):
        solve_cached_node_scaled_joint_sr(cache_for(jac, p, q), p, hp, q,
            blocks=[('all', 0, 19)], maximum_host_bytes=1, maximum_device_bytes=10**8)


def test_finite_but_inaccurate_dual_is_rejected(monkeypatch):
    jac, p, hp, q = problem()
    monkeypatch.setattr(jax.scipy.linalg, 'solve', lambda matrix, force, **kw: jax.numpy.ones_like(force))
    with pytest.raises(ArithmeticError, match='relative normal residual tolerance'):
        solve(jac, p, hp, q, [('all', 0, 19)])


def test_heldout_physical_prediction_uses_frozen_fit_direction_and_own_norm():
    # A complete diagonal Hermitian model gives an independent finite quotient
    # oracle on separate records, not a claim that FIT solves generalize.
    rng = np.random.default_rng(1103)
    full_j = rng.normal(size=(34, 9))
    theta = rng.normal(size=9)
    full_p = full_j@theta
    diagonal_h = rng.normal(size=34)
    full_q = np.exp(rng.normal(size=34))
    fit, heldout = slice(0, 17), slice(17, None)
    blocks = [('lower', 0, 4), ('upper', 4, 9)]
    d, _, _ = solve(full_j[fit], full_p[fit], (diagonal_h*full_p)[fit], full_q[fit], blocks)
    expected, _, _, _, _ = dense(full_j[fit], full_p[fit], (diagonal_h*full_p)[fit], full_q[fit], blocks)
    np.testing.assert_allclose(d, expected, atol=2e-12, rtol=2e-12)
    p = full_p[heldout]; tangent = -full_j[heldout]@np.asarray(d)
    q = full_q[heldout]; h = diagonal_h[heldout]
    energy = np.sum(p*p*h/q)/np.sum(p*p/q)
    slope = 2*np.sum(tangent*p*(h-energy)/q)/np.sum(p*p/q)
    epsilon = 1e-5
    quotient = lambda t: np.sum((p+t*tangent)**2*h/q)/np.sum((p+t*tangent)**2/q)
    finite_difference = (quotient(epsilon)-quotient(-epsilon))/(2*epsilon)
    np.testing.assert_allclose(slope, finite_difference, atol=2e-10, rtol=2e-9)
