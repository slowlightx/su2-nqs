"""Dense complex Hilbert-space and real-coordinate SR reference oracles."""
from functools import partial

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import (
    make_complex_importance_amplitude_sr,
    solve_complex_projected_block_sr,
)
from su2_nqs.vmc.importance_amplitude_sr import solve_projected_block_sr as solve_real_sr


def dense(j, p, h, q, ridge):
    """Independent parameter-space solve, not the realified dual algorithm."""
    norm = np.sum(abs(p)**2 / q)
    u = p / np.sqrt(q * norm)
    d = j / np.sqrt(q * norm)[:, None]
    a = d - np.outer(u, u.conj() @ d)
    energy = np.real(np.sum(p.conj() * h / q) / norm)
    b = 2 * (h - energy * p) / np.sqrt(q * norm)
    fisher = np.real(a.conj().T @ a)
    gradient = np.real(a.conj().T @ b)
    direction = np.linalg.solve(fisher + ridge * np.eye(j.shape[1]), gradient)
    return direction, gradient, a, b, energy


@pytest.mark.parametrize('parameters,chunk', [(1, 8), (17, 1), (17, 6), (17, 17), (17, 64), (34, 8)])
def test_complex_blocks_match_independent_dense_real_parameter_solve(parameters, chunk):
    rng = np.random.default_rng(28102801 + parameters)
    n = 9
    p = rng.normal(size=n) + 1j * rng.normal(size=n)
    p[[0, 4]] = 0.
    h = rng.normal(size=n) + 1j * rng.normal(size=n)
    q = rng.uniform(.05, 2., n)
    j = rng.normal(size=(n, parameters)) + 1j * rng.normal(size=(n, parameters))
    j[:, 0] = p
    if parameters > 1:
        j[:, 1] = 1j * p  # independent real-coordinate global-phase gauge
    expected, gradient, a, b, energy = dense(j, p, h, q, .17)
    solve = jax.jit(partial(solve_complex_projected_block_sr, damping=.17, parameter_chunk_size=chunk))
    result, actual_g, info = solve(*map(jnp.asarray, (j, p, h, q)))
    np.testing.assert_allclose(result, expected, atol=8e-14, rtol=4e-12)
    np.testing.assert_allclose(actual_g, gradient, atol=5e-14, rtol=4e-13)
    assert result.dtype == actual_g.dtype == jnp.float64
    assert bool(info['numerically_valid']) and int(info['exact_node_records']) == 2
    assert float(info['relative_linear_residual']) < 3e-12
    assert float(info['sampled_energy_total']) == pytest.approx(energy, abs=3e-15)
    np.testing.assert_allclose(info['raw_fisher_norm'], np.linalg.norm(a @ expected), atol=5e-14, rtol=4e-12)
    assert int(info['maximum_normalized_jacobian_block_entries']) == n * min(chunk, parameters)
    assert int(info['dual_real_dimension']) == 2 * n
    if parameters > 2:
        product = a @ expected
        mass = abs(product)**2
        signed_force = np.real(product.conj() * b)
        force_mass = abs(signed_force)
        np.testing.assert_allclose(info['chosen_direction_tangent_square_ESS'], mass.sum()**2 / (mass @ mass), rtol=4e-12)
        np.testing.assert_allclose(info['chosen_direction_tangent_square_maximum_fraction'], mass.max() / mass.sum(), rtol=4e-12)
        np.testing.assert_allclose(info['chosen_direction_absolute_force_projection_ESS'], force_mass.sum()**2 / (force_mass @ force_mass), rtol=4e-12)
        np.testing.assert_allclose(info['chosen_direction_signed_force_projection_sum'], expected @ gradient, rtol=4e-12)


def test_complex_exact_node_force_matches_true_Hermitian_energy_derivative():
    p = np.array([1., 0.], complex)
    h = np.array([0., 2 + 3j])
    j = np.array([[0.], [1 + 2j]])
    d, g, info = solve_complex_projected_block_sr(*map(jnp.asarray, (j, p, h, np.ones(2))), damping=.5)
    np.testing.assert_allclose(g, [16.], atol=0, rtol=0)
    np.testing.assert_allclose(d, [16 / 5.5], atol=2e-15, rtol=0)
    assert int(info['exact_node_records']) == 1
    ham = jnp.array([[0., 2 - 3j], [2 + 3j, 0.]])
    def energy(theta):
        state = jnp.array([1., theta * (1 + 2j)])
        return jnp.real(jnp.vdot(state, ham @ state) / jnp.vdot(state, state))
    assert jax.grad(energy)(0.) == pytest.approx(16., abs=0.)
    assert float(info['raw_fisher_norm']) == pytest.approx(np.sqrt(5) * 16 / 5.5)


def test_real_and_imaginary_tangents_need_realified_dual_not_complex_N_gram():
    # Two real parameters span the real and imaginary directions at one
    # physical node.  Their real FS metric is I_2, not a single eigenvalue 2.
    jac = jnp.array([[0., 0.], [1., 1j]], dtype=jnp.complex128)
    psi = jnp.array([1., 0.], dtype=jnp.complex128)
    hpsi = jnp.array([0., 1. + 2j], dtype=jnp.complex128)
    direction, gradient, info = solve_complex_projected_block_sr(
        jac, psi, hpsi, jnp.ones(2), damping=.5, parameter_chunk_size=1)
    np.testing.assert_allclose(gradient, [2., 4.], atol=0, rtol=0)
    np.testing.assert_allclose(direction, np.array([2., 4.]) / 1.5, atol=1e-15, rtol=0)
    assert info['numerically_valid']


def test_complex_gauge_phase_scale_density_and_energy_shift_invariance():
    rng = np.random.default_rng(28102803)
    j = rng.normal(size=(9, 7)) + 1j * rng.normal(size=(9, 7))
    p = rng.normal(size=9) + 1j * rng.normal(size=9)
    h = rng.normal(size=9) + 1j * rng.normal(size=9)
    q = rng.uniform(.03, .7, 9)
    baseline = solve_complex_projected_block_sr(*map(jnp.asarray, (j, p, h, q)), damping=.5, parameter_chunk_size=3)
    for scale, qscale, offset in [(3j, .1, 17.), (1e-40 * np.exp(.71j), 1e-20, -3.)]:
        actual = solve_complex_projected_block_sr(*map(jnp.asarray,
            (scale * j, scale * p, scale * (h + offset * p), q * qscale)), damping=.5, parameter_chunk_size=3)
        np.testing.assert_allclose(actual[0], baseline[0], atol=5e-14, rtol=4e-12)
        np.testing.assert_allclose(actual[1], baseline[1], atol=5e-14, rtol=4e-12)
    # Parameter-dependent complex rescaling changes J by psi times any
    # complex row; both its normalization and phase pieces must disappear.
    gauge = rng.normal(size=7) + 1j * rng.normal(size=7)
    actual = solve_complex_projected_block_sr(*map(jnp.asarray, (j + p[:, None] * gauge, p, h, q)),
        damping=.5, parameter_chunk_size=3)
    np.testing.assert_allclose(actual[0], baseline[0], atol=5e-14, rtol=4e-12)
    np.testing.assert_allclose(actual[1], baseline[1], atol=5e-14, rtol=4e-12)


def test_real_limit_matches_unchanged_production_real_solver():
    rng = np.random.default_rng(28102804)
    arrays = [jnp.asarray(rng.normal(size=(11, 9))), jnp.asarray(rng.normal(size=11)),
        jnp.asarray(rng.normal(size=11)), jnp.asarray(rng.uniform(.1, 1., 11))]
    real = solve_real_sr(*arrays, damping=.3, parameter_chunk_size=4)
    complex_result = solve_complex_projected_block_sr(*arrays, damping=.3, parameter_chunk_size=4)
    np.testing.assert_allclose(complex_result[0], real[0], atol=8e-14, rtol=4e-12)
    np.testing.assert_allclose(complex_result[1], real[1], atol=8e-14, rtol=4e-12)


def fixture():
    params = dict(matrix=jnp.array([[.7, .2], [-.15, .9]]), vector=jnp.array([.13, -.17]),
        scale=jnp.array(.11), phase=jnp.array(.23))
    rows = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 3)
    def prepare(p):
        return dict(weights=p['matrix'] @ p['vector'] + 1j * (p['matrix'].T @ p['vector']),
            pair=jnp.sin(p['matrix'][0, 1]) + p['vector'][0] * p['vector'][1],
            gauge=p['scale'] + 1j * p['phase'])
    def prepared(bank, row):
        a, c = 2 * row[0] - 1, 2 * row[1] - 1
        return (1 - 2 * row[0] * row[1]) * jnp.exp(
            bank['weights'] @ jnp.array([a, c]) + bank['pair'] * a * c + bank['gauge'])
    def amplitude(p, row):
        return prepared(prepare(p), row)
    a, c = np.asarray(2 * rows[:, 0] - 1), np.asarray(2 * rows[:, 1] - 1)
    ac = a * c
    m, v = np.asarray(params['matrix']), np.asarray(params['vector'])
    # ravel_pytree's deterministic dictionary order: matrix, phase, scale, vector.
    scores = np.column_stack((
        a * v[0] + 1j * v[0] * a,
        a * v[1] + ac * np.cos(m[0, 1]) + 1j * v[0] * c,
        c * v[0] + 1j * v[1] * a,
        c * v[1] + 1j * v[1] * c,
        1j * np.ones(len(rows)), np.ones(len(rows)),
        a * m[0, 0] + c * m[1, 0] + ac * v[1] + 1j * (m[0, 0] * a + m[0, 1] * c),
        a * m[0, 1] + c * m[1, 1] + ac * v[0] + 1j * (m[1, 0] * a + m[1, 1] * c)))
    psi = np.asarray(jax.vmap(lambda row: amplitude(params, row))(rows))
    return params, rows, prepare, prepared, amplitude, psi, scores * psi[:, None]


@pytest.mark.parametrize('chunk', [1, 2, 5])
def test_complex_prepared_native_Jacobian_JVP_update_and_checkpoint(chunk, tmp_path):
    p, rows, prepare, prepared, amp, psi, jac = fixture()
    flat, unravel = ravel_pytree(p)
    rng = np.random.default_rng(28102805)
    h = rng.normal(size=len(rows)) + 1j * rng.normal(size=len(rows))
    q = rng.uniform(.05, 1., len(rows))
    direction, gradient, a, _, _ = dense(jac, psi, h, q, .5)
    jvp = jax.jvp(lambda values: jax.vmap(lambda row: amp(unravel(values), row))(rows),
        (flat,), (jnp.asarray(direction),))[1]
    np.testing.assert_allclose(jvp, jac @ direction, atol=5e-14, rtol=3e-13)
    moments = (jnp.array(41), unravel(jnp.linspace(.001, .003, len(flat))),
        unravel(jnp.linspace(.01, .03, len(flat))))
    expected_scale = min(.03, .003 / np.linalg.norm(a @ direction))
    expected = np.asarray(flat) - expected_scale * direction
    outputs = []
    for callbacks in ({}, dict(prepare=prepare, prepared_amplitude=prepared)):
        update = make_complex_importance_amplitude_sr(amp, p, score_chunk=chunk,
            parameter_chunk_size=3, **callbacks)
        proposed, state, info = update(p, moments, rows, jnp.asarray(h), jnp.asarray(q), .03)
        assert info['accepted'] and info['proposal_only_requires_fresh_finite_guard']
        np.testing.assert_allclose(ravel_pytree(proposed)[0], expected, atol=5e-15, rtol=0)
        assert float(info['sampled_tangent_FS']) <= .003 * (1 + 3e-15)
        assert int(state[0]) == 42
        for before, after in zip(jax.tree.leaves(moments[1:]), jax.tree.leaves(state[1:]), strict=True):
            np.testing.assert_array_equal(before, after)
        assert info['raw_energy_directional_derivative'] == pytest.approx(
            float(gradient @ (expected - flat)), abs=5e-15)
        outputs.append((proposed, state, info))
    for left, right in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True):
        np.testing.assert_allclose(left, right, atol=6e-14, rtol=4e-12)
    path = tmp_path / 'complex_sr.pkl'
    save_checkpoint(path, outputs[1][0], optimizer_state=outputs[1][1])
    loaded = load_checkpoint(path)
    update = make_complex_importance_amplitude_sr(amp, p, score_chunk=chunk,
        prepare=prepare, prepared_amplitude=prepared)
    restored = update(loaded['params'], loaded['optimizer_state'], rows, jnp.asarray(h), jnp.asarray(q), .001)
    direct = update(outputs[1][0], outputs[1][1], rows, jnp.asarray(h), jnp.asarray(q), .001)
    for left, right in zip(jax.tree.leaves(restored), jax.tree.leaves(direct), strict=True):
        np.testing.assert_array_equal(left, right)


def test_full_H_jvp_and_exact_complete_quadrature_energy_gradient():
    rng = np.random.default_rng(28102806)
    n, parameters = 7, 5
    p = rng.normal(size=n) + 1j * rng.normal(size=n)
    p[0] = 0.
    jac = rng.normal(size=(n, parameters)) + 1j * rng.normal(size=(n, parameters))
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    ham = raw + raw.conj().T
    state = lambda theta: jnp.asarray(p) + jnp.asarray(jac) @ theta
    energy = lambda theta: jnp.real(jnp.vdot(state(theta), jnp.asarray(ham) @ state(theta))
        / jnp.vdot(state(theta), state(theta)))
    theta = jnp.zeros(parameters)
    d, g, _ = solve_complex_projected_block_sr(*map(jnp.asarray,
        (jac, p, ham @ p, np.ones(n))), damping=.5, parameter_chunk_size=2)
    np.testing.assert_allclose(g, jax.grad(energy)(theta), atol=6e-14, rtol=4e-13)
    tangent = jax.jvp(state, (theta,), (d,))[1]
    htangent = jax.jvp(lambda x: jnp.asarray(ham) @ state(x), (theta,), (d,))[1]
    np.testing.assert_allclose(htangent, ham @ (jac @ d), atol=5e-14, rtol=4e-13)
    np.testing.assert_allclose(tangent, jac @ d, atol=2e-14, rtol=4e-13)


def test_dynamic_invalid_Q_never_repairs_or_advances_optimizer():
    p = jnp.array([.2])
    rows = jnp.arange(4)[:, None]
    amp = lambda theta, row: 1 + (1 + 2j) * theta[0] * (row[0] - 1.5)
    update = make_complex_importance_amplitude_sr(amp, p)
    moments = _manual_adam_init(p)
    for q in (jnp.array([1., 0., 1., 1.]), jnp.array([1., jnp.nan, 1., 1.])):
        proposed, state, info = update(p, moments, rows, jnp.ones(4, jnp.complex128), q, .03)
        assert not info['accepted'] and not info['numerically_valid']
        np.testing.assert_array_equal(proposed, p)
        assert int(state[0]) == 0


def test_zero_force_statistics_remain_explicitly_undefined():
    jac = jnp.array([[0., 1j], [1j, 2.], [2., -1j]])
    _, gradient, info = solve_complex_projected_block_sr(jac, jnp.ones(3), jnp.zeros(3),
        jnp.ones(3), damping=.5)
    np.testing.assert_array_equal(gradient, np.zeros(2))
    assert info['numerically_valid'] and info['raw_fisher_norm'] == 0
    assert not info['chosen_direction_tangent_concentration_defined']
    assert not info['chosen_direction_force_concentration_defined']
    assert info['chosen_direction_tangent_square_ESS'] == 0
    assert info['chosen_direction_absolute_force_projection_ESS'] == 0


@pytest.mark.parametrize('defect', ['shape', 'float32', 'complex64', 'Qcomplex', 'Qzero',
    'Qnan', 'zero_norm', 'ridge', 'chunk'])
def test_solver_contract_rejects_invalid_inputs(defect):
    j = jnp.ones((4, 3), jnp.complex128)
    p, h, q = jnp.ones(4, jnp.complex128), jnp.arange(4, dtype=jnp.float64), jnp.ones(4)
    options = dict(damping=.5, parameter_chunk_size=2)
    if defect == 'shape': j = j[0]
    if defect == 'float32': h = h.astype(jnp.float32)
    if defect == 'complex64': j = j.astype(jnp.complex64)
    if defect == 'Qcomplex': q = q.astype(jnp.complex128)
    if defect == 'Qzero': q = q.at[1].set(0)
    if defect == 'Qnan': q = q.at[1].set(jnp.nan)
    if defect == 'zero_norm': p = p * 0
    if defect == 'ridge': options['damping'] = False
    if defect == 'chunk': options['parameter_chunk_size'] = True
    with pytest.raises(ValueError):
        solve_complex_projected_block_sr(j, p, h, q, **options)


def test_update_rejects_complex_parameters_and_incomplete_preparation():
    amp = lambda p, row: p[0] * (1 + 1j)
    with pytest.raises(ValueError):
        make_complex_importance_amplitude_sr(amp, jnp.array([1j]))
    with pytest.raises(ValueError):
        make_complex_importance_amplitude_sr(amp, jnp.ones(1), prepare=lambda p: p)
    update = make_complex_importance_amplitude_sr(amp, jnp.ones(1))
    with pytest.raises(ValueError):
        update(jnp.ones(1, jnp.complex128), _manual_adam_init(jnp.ones(1)),
            jnp.arange(2)[:, None], jnp.ones(2, jnp.complex128), jnp.ones(2), .01)
    try:
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(RuntimeError):
            make_complex_importance_amplitude_sr(amp, np.ones(1))
    finally:
        jax.config.update('jax_enable_x64', True)


def test_workspace_is_block_bounded_and_not_parameter_space_quadratic():
    count, parameters, chunk = 16, 131071, 1024
    function = jax.jit(partial(solve_complex_projected_block_sr, damping=.5, parameter_chunk_size=chunk))
    compiled = function.lower(jax.ShapeDtypeStruct((count, parameters), jnp.complex128),
        jax.ShapeDtypeStruct((count,), jnp.complex128), jax.ShapeDtypeStruct((count,), jnp.complex128),
        jax.ShapeDtypeStruct((count,), jnp.float64)).compile()
    memory = compiled.memory_analysis()
    raw_bytes = count * parameters * 16
    assert memory.temp_size_in_bytes < raw_bytes * .5, memory
