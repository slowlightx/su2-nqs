"""Independent exact checks for collective physical-tangent query fitting."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.collective_queries import (
    collective_subspace, fit_collective_update, audit_collective_update,
)


def test_restricted_jvp_matches_complete_gradient_times_q():
    rng = np.random.default_rng(74201)
    x = jnp.asarray(rng.normal(size=(40, 12)))
    q = collective_subspace(rng.normal(size=(12, 4)), 4)
    theta = jnp.asarray(rng.normal(size=12))
    def log_amplitude(parameters):
        return jnp.tanh(x @ parameters)+1j*jnp.sin(x @ parameters/3)
    full = np.asarray(jax.jacfwd(log_amplitude)(theta))
    restricted = np.asarray(jax.jacfwd(lambda u: log_amplitude(theta+jnp.asarray(q) @ u))(jnp.zeros(4)))
    np.testing.assert_allclose(restricted, full @ q, atol=3e-15)


def test_collective_solver_is_invariant_under_subspace_basis_rotations():
    rng = np.random.default_rng(74202)
    scores = rng.normal(size=(128, 4))+1j*rng.normal(size=(128, 4))
    energies = scores @ np.array([.2, -.3, .1, .7])+rng.normal(size=128)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    first = fit_collective_update(scores, energies)
    second = fit_collective_update(scores @ rotation, energies)
    np.testing.assert_allclose(rotation @ second.coefficients, first.coefficients, atol=2e-18)
    np.testing.assert_allclose(first.coefficients @ first.gram @ first.coefficients, 1e-6, atol=2e-20)
    assert 2*first.force @ first.coefficients < 0


def test_only_queried_subspace_is_used_and_constant_gauges_cancel():
    rng = np.random.default_rng(74203)
    scores = rng.normal(size=(96, 10))+1j*rng.normal(size=(96, 10))
    q = collective_subspace(rng.normal(size=(10, 2)), 2)
    other = rng.normal(size=scores.shape) @ (np.eye(10)-q @ q.T)
    energies = rng.normal(size=96)
    first = fit_collective_update(scores @ q, energies)
    second = fit_collective_update((scores+100*other) @ q+(2+3j), energies-12)
    np.testing.assert_allclose(second.coefficients, first.coefficients, atol=4e-17)
    # A strictly constant gauge query has no physical norm or force.
    null = fit_collective_update(np.ones((96, 2))*(2+3j), energies)
    np.testing.assert_array_equal(null.coefficients, 0.)


def test_collective_basis_keeps_all_legal_tangents_and_jackknife_is_chainwise():
    rng = np.random.default_rng(74204)
    q = collective_subspace(rng.normal(size=(24, 4)), 4)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=2e-15)
    assert np.all(np.sum(abs(q), axis=1) > 1e-8)  # Not a coordinate top-k selector.
    scores = rng.normal(size=(64, 4))
    energies = scores @ np.arange(4)+rng.normal(size=64)
    fitted = fit_collective_update(scores[:32], energies[:32])
    metrics = audit_collective_update(fitted.coefficients, scores[32:], energies[32:], np.repeat(np.arange(8), 4))
    assert metrics['chains'] == 8
    assert len(metrics['energy_slope_per_fs_total_chain_jackknife']) == 8
    assert metrics['energy_slope_total'] < 0


def test_no_zero_column_query_and_projected_singlet_linear_combinations_remain_exact():
    from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
    from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
    from su2_nqs.su2.fusion_path import enumerate_paths
    with pytest.raises(ValueError, match='query column'):
        fit_collective_update(np.empty((20, 0)), np.ones(20))
    n = 6
    basis = coupled_basis_matrix(enumerate_paths(n, 0), 0, 0)
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    translated = np.roll(bits, 1, axis=1) @ (1 << np.arange(n-1, -1, -1))
    permutation = np.eye(2**n)[translated]
    projector = sum(np.linalg.matrix_power(permutation, k) for k in range(n))/n
    rng = np.random.default_rng(74205)
    legal_tangents = projector @ basis @ rng.normal(size=(basis.shape[1], 7))
    q = collective_subspace(rng.normal(size=(7, 4)), 4)
    collective = legal_tangents @ q
    assert np.linalg.norm(collective) > .1
    s2, _ = spin_squared_matrix(n)
    np.testing.assert_allclose(s2 @ collective, 0., atol=3e-14)
    np.testing.assert_allclose(permutation @ collective, collective, atol=2e-15)
