"""Independent weighted Hilbert-space oracles for all-column node SR."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cached_amplitude_operators import (
    build_cached_amplitude_jacobian, solve_cached_node_block_sr,
)


def fixture(columns=None):
    rng = np.random.default_rng(31112026)
    psi = rng.normal(size=14); psi[2] = 0.
    q = np.exp(rng.normal(size=14))
    hp = rng.normal(size=14)
    jac = rng.normal(size=(14, 11)) if columns is None else columns
    rows = np.arange(14, dtype=np.int32)[:, None]
    theta = np.zeros(jac.shape[1])
    batch = lambda t, r: jnp.asarray(psi)[r[:, 0]]+jnp.asarray(jac)[r[:, 0]]@t
    cache = build_cached_amplitude_jacobian(batch, theta, rows, psi, q,
        maximum_host_bytes=10**7, row_batch_size=2, source_identity='independent-node-SR')
    return cache, psi, q, hp, jac


def oracle(psi, q, hp, jac, blocks, damping=.5):
    scale = 1/np.sqrt(q*np.sum(psi**2/q)); unit = psi*scale
    a = (np.eye(len(psi))-np.outer(unit, unit))@(jac*scale[:, None])
    e = np.sum(psi*hp/q)/np.sum(psi**2/q)
    force = 2*(hp-e*psi)*scale
    gradient = a.T@force
    trace_scales = [np.trace(a[:, b:c].T@a[:, b:c])/(len(psi)-1) for _, b, c in blocks]
    floor = max(trace_scales)*1e-12
    direction = np.zeros(jac.shape[1]); tangents = []
    for (_, first, final), scale in zip(blocks, trace_scales):
        f = a[:, first:final].T@a[:, first:final]
        ridge = max(damping*scale, floor)
        direction[first:final] = np.linalg.solve(f+ridge*np.eye(final-first), gradient[first:final])
        tangents.append(a[:, first:final]@direction[first:final])
    return direction, gradient, np.asarray(tangents), a


def test_full_node_direction_matches_independent_dense_block_metric_and_global_tangent():
    cache, psi, q, hp, jac = fixture()
    blocks = [('local0', 0, 3), ('upper0', 3, 7), ('root', 7, 11)]
    expected, gradient, tangents, a = oracle(psi, q, hp, jac, blocks)
    d, g, info = solve_cached_node_block_sr(cache, psi, hp, q, blocks=blocks,
        parameter_chunk_size=2, maximum_device_bytes=10**7, maximum_host_bytes=10**7)
    np.testing.assert_allclose(d, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(g, gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(info['chosen_direction_tangent'], a@expected, atol=2e-12)
    np.testing.assert_allclose(info['node_tangent_Gram'], tangents@tangents.T, atol=2e-12)
    assert info['relative_linear_residual'] < 1e-12
    assert info['global_cross_node_normal_residual'] > 1e-3
    assert not info['full_global_Fisher_solved']
    assert info['all_records_retained'] and info['all_native_columns_retained']


def test_node_coordinate_rescaling_preserves_combined_physical_update():
    cache, psi, q, hp, jac = fixture()
    blocks = [('a', 0, 4), ('b', 4, 11)]
    _, _, info = solve_cached_node_block_sr(cache, psi, hp, q, blocks=blocks,
        parameter_chunk_size=3, maximum_device_bytes=10**7, maximum_host_bytes=10**7)
    scales = np.r_[np.full(4, .02), np.full(7, 15.)]
    other, pp, qq, hh, _ = fixture(jac*scales)
    np.testing.assert_array_equal(pp, psi); np.testing.assert_array_equal(qq, q)
    _, _, scaled = solve_cached_node_block_sr(other, pp, hh, qq, blocks=blocks,
        parameter_chunk_size=3, maximum_device_bytes=10**7, maximum_host_bytes=10**7)
    np.testing.assert_allclose(info['chosen_direction_tangent'], scaled['chosen_direction_tangent'],
                               rtol=3e-12, atol=3e-12)


def test_zero_node_is_retained_and_numerical_ridge_floor_is_disclosed():
    _, psi, _, _, jac = fixture()
    # First block is a redundant global-amplitude derivative, hence gauge.
    jac[:, :2] = psi[:, None]
    cache, psi, q, hp, jac = fixture(jac)
    d, _, info = solve_cached_node_block_sr(cache, psi, hp, q,
        blocks=[('gauge', 0, 2), ('physical', 2, 11)],
        parameter_chunk_size=3, maximum_device_bytes=10**7, maximum_host_bytes=10**7)
    assert np.isfinite(d).all() and info['numerical_ridge_floor'] > 0
    assert info['blocks'][0]['ridge'] == info['numerical_ridge_floor']
    assert info['blocks'][0]['parameters'] == 2


@pytest.mark.parametrize('blocks', [[], [('missing', 0, 10)], [('x', 1, 11)],
    [('x', 0, 5), ('x', 5, 11)], [('x', 0, 6), ('y', 5, 11)]])
def test_incomplete_or_ambiguous_column_partitions_are_rejected(blocks):
    cache, psi, q, hp, _ = fixture()
    with pytest.raises(ValueError):
        solve_cached_node_block_sr(cache, psi, hp, q, blocks=blocks,
            maximum_device_bytes=10**7, maximum_host_bytes=10**7)


def test_bad_data_and_resource_budgets_fail_before_large_work():
    cache, psi, q, hp, _ = fixture()
    kwargs = dict(blocks=[('all', 0, 11)], maximum_device_bytes=10**7, maximum_host_bytes=10**7)
    with pytest.raises(ValueError):
        solve_cached_node_block_sr(cache, psi, hp, q, relative_damping=0., **kwargs)
    with pytest.raises(ValueError):
        solve_cached_node_block_sr(cache, psi, hp, q*2, **kwargs)
    with pytest.raises(MemoryError):
        solve_cached_node_block_sr(cache, psi, hp, q, **dict(kwargs, maximum_host_bytes=1))
    with pytest.raises(MemoryError):
        solve_cached_node_block_sr(cache, psi, hp, q, **dict(kwargs, maximum_device_bytes=1))
