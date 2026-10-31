"""Independent dense references for exact Slater row updates and their nodes."""

from itertools import combinations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.slater_minor_updates import (
    closed_shell_amplitudes, compile_slater_minor_plan, slater_minor_updates,
)


def _sector(n):
    bits = np.zeros((len(list(combinations(range(n), n // 2))), n), dtype=np.int8)
    for row, up in enumerate(combinations(range(n), n // 2)):
        bits[row, list(up)] = 1
    return bits


def _dense(v, states):
    return np.asarray([[np.linalg.det(v[np.flatnonzero(row == spin)])
                        for spin in (1, 0)] for row in states])


def _agp(v, states):
    pairing = v @ v.T  # Symmetric singlet pairing, not a Hermitian projector.
    values = []
    for row in states:
        up, down = np.flatnonzero(row), np.flatnonzero(1-row)
        inversions = sum(d < u for u in up for d in down)
        parity = inversions + len(up)*(len(up)-1)//2
        values.append((-1)**parity*np.linalg.det(pairing[np.ix_(up, down)]))
    return np.asarray(values)


def test_exhaustive_complex_n8_determinants_sorting_parity_and_singlet_pairing():
    rng = np.random.default_rng(2026101301)
    v = np.linalg.qr(rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)))[0]
    states = _sector(8)
    refs = states[[0, 23, 69]]
    indices = np.arange(len(states)) % len(refs)
    plan = compile_slater_minor_plan(refs, states, indices, max_rank=4)
    actual = jax.jit(lambda orbitals:slater_minor_updates(orbitals, plan))(v)
    assert not np.any(actual.dense_fallback)
    assert plan.numeric_plan_bytes > 0
    np.testing.assert_allclose(actual.determinants, _dense(v, states), atol=3e-15, rtol=3e-13)
    np.testing.assert_allclose(closed_shell_amplitudes(actual, plan), _agp(v, states), atol=3e-15, rtol=3e-13)
    assert len(np.unique(plan.sorting_signs)) == 2
    assert not plan.target_bits.flags.writeable


def test_all_ranks_to_six_and_arbitrary_query_reference_order():
    rng = np.random.default_rng(2026101302)
    v = np.linalg.qr(rng.normal(size=(16, 8))+1j*rng.normal(size=(16, 8)))[0]
    reference = np.array([1]*8+[0]*8, dtype=np.int8)
    states = []
    for rank in range(7):
        target = reference.copy()
        target[:rank], target[8:8+rank] = 0, 1
        states.append(target)
    states = np.asarray(states)
    plan = compile_slater_minor_plan(reference[None], states, max_rank=6)
    result = slater_minor_updates(v, plan)
    np.testing.assert_array_equal(result.ranks[:, 0], np.arange(7))
    np.testing.assert_allclose(result.determinants, _dense(v, states), atol=2e-15, rtol=2e-12)
    order = np.array([6, 0, 2, 2, 5, 1, 4, 3])
    second = compile_slater_minor_plan(np.stack([reference, states[3]]), states[order],
        np.arange(len(order)) % 2, max_rank=6)
    np.testing.assert_allclose(slater_minor_updates(v, second).determinants,
                               np.asarray(result.determinants)[order], atol=3e-15, rtol=3e-12)
    zero = compile_slater_minor_plan(reference[None], reference[None], max_rank=0)
    np.testing.assert_allclose(slater_minor_updates(v, zero).determinants,
                               _dense(v, reference[None]), atol=1e-15)


def test_complex_orbital_JVP_and_full_reverse_gradient_match_dense_finite_difference():
    rng = np.random.default_rng(2026101303)
    v = (rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)))/3
    direction = (rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)))/5
    states = _sector(8)[[0, 12, 20, 38, 53, 69]]
    plan = compile_slater_minor_plan(states[:1], states, max_rank=4)
    fn = lambda orbitals:closed_shell_amplitudes(slater_minor_updates(orbitals, plan), plan)
    _, tangent = jax.jit(lambda x, dx:jax.jvp(fn, (x,), (dx,)))(v, direction)
    eps = 2e-6
    finite = (_agp(v+eps*direction, states)-_agp(v-eps*direction, states))/(2*eps)
    np.testing.assert_allclose(tangent, finite, atol=1e-10, rtol=2e-8)
    weights = rng.normal(size=len(states))+1j*rng.normal(size=len(states))
    parameters = jnp.stack([v.real, v.imag])
    def loss(p): return jnp.real(jnp.vdot(jnp.asarray(weights), fn(p[0]+1j*p[1])))
    gradient = np.asarray(jax.jit(jax.grad(loss))(parameters))
    assert np.isfinite(gradient).all()
    contracted = np.sum(gradient*np.stack([direction.real, direction.imag]))
    np.testing.assert_allclose(contracted, np.vdot(weights, finite).real, atol=1e-10, rtol=2e-8)


def test_singular_reference_component_falls_back_without_losing_node_derivative():
    rng = np.random.default_rng(2026101304)
    v = np.vstack([np.diag([0., 1., 2., 3.]), rng.normal(size=(4, 4))]).astype(complex)
    reference = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int8)
    target = reference.copy(); target[:, [0, 4]] = target[:, [4, 0]]
    states = np.concatenate([reference, target])
    plan = compile_slater_minor_plan(reference, states, max_rank=1)
    result = jax.jit(lambda orbitals:slater_minor_updates(orbitals, plan))(v)
    np.testing.assert_array_equal(result.dense_fallback, [[True, False], [True, False]])
    np.testing.assert_allclose(result.determinants, _dense(v, states), atol=2e-14)
    assert result.determinants[0, 0] == 0
    assert abs(result.determinants[1, 0]) > .1
    direction = np.zeros_like(v); direction[:4] = np.eye(4)
    fun = lambda orbital:slater_minor_updates(orbital, plan).determinants
    _, tangent = jax.jvp(fun, (jnp.asarray(v),), (jnp.asarray(direction),))
    assert np.isfinite(tangent).all()
    np.testing.assert_allclose(tangent[0, 0], 6., atol=1e-13)
    finite = (_dense(v+1e-6*direction, states)-_dense(v-1e-6*direction, states))/(2e-6)
    np.testing.assert_allclose(tangent, finite, atol=1e-8, rtol=2e-8)
    reverse = jax.grad(lambda real_v:jnp.real(fun(real_v)[0, 0]))(jnp.asarray(v.real))
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(reverse[0, 0], 6., atol=1e-13)


def test_singular_target_minor_has_nonzero_JVP_and_reverse_gradient():
    rng = np.random.default_rng(2026101305)
    v = np.vstack([np.eye(4), rng.normal(size=(4, 4))]).astype(complex)
    v[4] = v[1]  # Replacing row 0 by site 4 makes only the target up Slater singular.
    reference = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int8)
    target = reference.copy(); target[:, [0, 4]] = target[:, [4, 0]]
    plan = compile_slater_minor_plan(reference, target, max_rank=4)
    direction = np.zeros_like(v); direction[4, 0] = 1
    fun = lambda orbital:slater_minor_updates(orbital, plan).determinants[0, 0]
    actual, tangent = jax.jvp(fun, (jnp.asarray(v),), (jnp.asarray(direction),))
    assert actual == 0
    np.testing.assert_allclose(tangent, -1., atol=1e-14)
    gradient = jax.grad(lambda real_v:jnp.real(fun(real_v)))(jnp.asarray(v.real))
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[4, 0], -1., atol=1e-14)


def test_ill_conditioned_reference_uses_exact_dense_and_coherent_nodes_do_not_trigger_it():
    rng = np.random.default_rng(2026101306)
    v = np.vstack([np.diag([1e-13, 1., 2., 3.]), rng.normal(size=(4, 4))])
    states = _sector(8)[[0, 1, 6, 12]]
    plan = compile_slater_minor_plan(states[:1], states, max_rank=4)
    result = slater_minor_updates(v, plan, condition_limit=1e9)
    assert np.all(np.asarray(result.dense_fallback)[:, 0])
    np.testing.assert_allclose(result.determinants, _dense(v, states), atol=2e-14)
    good = np.linalg.qr(rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)))[0]
    good_result = slater_minor_updates(good, plan)
    amps = closed_shell_amplitudes(good_result, plan)
    np.testing.assert_array_equal(amps-amps, np.zeros(len(states)))
    assert not np.any(good_result.dense_fallback)


def test_invalid_discrete_inputs_and_bounds_fail_closed():
    state = np.array([[1, 1, 0, 0]], dtype=np.int8)
    with pytest.raises(ValueError): compile_slater_minor_plan(state, 1-state, max_rank=1)
    with pytest.raises(ValueError): compile_slater_minor_plan(state, np.zeros_like(state), max_rank=1)
    with pytest.raises(ValueError): compile_slater_minor_plan(np.repeat(state, 2, axis=0), state, max_rank=1)
    with pytest.raises(ValueError): compile_slater_minor_plan(state, state, np.array([1]), max_rank=1)
    with pytest.raises(ValueError): compile_slater_minor_plan(state, state, max_rank=True)
    plan = compile_slater_minor_plan(state, state, max_rank=1)
    with pytest.raises(ValueError): slater_minor_updates(np.ones((4, 3)), plan)
    with pytest.raises(ValueError): slater_minor_updates(np.ones((4, 2)), plan, condition_limit=np.inf)
