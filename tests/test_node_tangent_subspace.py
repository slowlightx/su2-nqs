"""Independent dense references for optimizer-only coupled native directions."""
import numpy as np
import pytest

from su2_nqs.vmc.node_tangent_subspace import (
    solve_physical_node_subspace, stream_native_node_tangents,
)


def test_full_coupled_metric_matches_independent_dense_scaled_preconditioner():
    rng = np.random.default_rng(115)
    p = rng.normal(size=53); p[0] = 0
    hp = rng.normal(size=53); q = rng.uniform(.1, 1., 53)
    J = rng.normal(size=(53, 17)); d = rng.normal(size=17)
    blocks = [('first', 0, 3), ('second', 3, 11), ('root', 11, 17)]
    images = stream_native_node_tangents(J, d, blocks=blocks, parameter_chunk=4)
    independent = np.stack([J[:, a:b] @ d[a:b] for _, a, b in blocks], axis=1)
    np.testing.assert_allclose(images, independent, rtol=1e-14, atol=1e-14)
    c, t, info = solve_physical_node_subspace(p, hp, q, images, n_sites=6)
    u = p / np.sqrt(q); u /= np.linalg.norm(u)
    z = np.sum(p*p/q)
    A = images / np.sqrt(q*z)[:, None]
    A = (np.eye(len(p)) - np.outer(u, u)) @ A
    scale = np.linalg.norm(A, axis=0)
    B = A / scale
    E = np.sum(p*hp/q) / z
    force = 2 * (hp-E*p) / np.sqrt(q*z)
    G = B.T @ B; ridge = .5*np.trace(G)/(len(p)-1)
    expected = -np.linalg.solve(G+ridge*np.eye(3), B.T @ force) / scale
    np.testing.assert_allclose(c, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(t, A @ expected, rtol=2e-13, atol=2e-14)
    assert info['exact_coupled_normal_relative_residual'] < 1e-13
    assert abs(info['tangent_parent_overlap']) < 1e-14
    assert info['predicted_Hermitian_energy_slope_per_site'] < 0
    # Independent node solves omit sizeable cross-node terms in this example.
    assert np.max(abs(G-np.diag(np.diag(G)))) > .05


def test_column_scale_and_normalization_gauge_do_not_change_physical_step():
    rng = np.random.default_rng(721)
    p, h = rng.normal(size=(2, 44)); q = rng.uniform(.2, 2., 44)
    raw = rng.normal(size=(44, 4))
    factors = np.array([1e-5, -3., 1e4, .7])
    c, t, _ = solve_physical_node_subspace(p, h, q, raw)
    shifted = (raw+p[:, None]*np.array([.7, -5., 2., .1]))*factors
    changed, tangent, _ = solve_physical_node_subspace(p, h, q, shifted)
    np.testing.assert_allclose(changed*factors, c, rtol=3e-13, atol=1e-13)
    np.testing.assert_allclose(tangent, t, rtol=3e-13, atol=1e-13)


def test_duplicate_node_directions_are_coupled_not_repeatedly_added():
    rng = np.random.default_rng(92)
    p, h, d = rng.normal(size=(3, 64)); q = np.ones(64)
    _, first, _ = solve_physical_node_subspace(p, h, q, d[:, None])
    _, repeated, info = solve_physical_node_subspace(p, h, q, np.column_stack([d, 2*d, -d, 17*d]))
    np.testing.assert_allclose(repeated, first, rtol=2e-13, atol=2e-14)
    assert info['effective_rank'] == pytest.approx(1.)


def test_gauge_only_and_zero_columns_are_not_physical_capacity():
    rng = np.random.default_rng(8); p, h, d = rng.normal(size=(3, 40))
    c, tangent, info = solve_physical_node_subspace(p, h, np.ones(40), np.column_stack([d, p, 0*p]))
    assert info['active_columns'] == [0]
    assert c[1] == c[2] == 0
    assert np.linalg.norm(tangent) > 0
    with pytest.raises(ArithmeticError):
        solve_physical_node_subspace(p, h, np.ones(40), np.column_stack([p, p]))


def test_invalid_support_or_incomplete_native_partition_fails():
    with pytest.raises(ValueError):
        solve_physical_node_subspace(np.ones(8), np.ones(8), np.zeros(8), np.ones((8, 2)))
    with pytest.raises(ValueError):
        stream_native_node_tangents(np.ones((8, 5)), np.ones(5), blocks=[('bad', 0, 4)])
