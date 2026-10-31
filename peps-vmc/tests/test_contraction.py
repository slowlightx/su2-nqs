"""Independent exact contraction and exchange-cache checks."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from peps_vmc.contraction import (
    make_amplitude_batch,
    make_exact_plan,
    make_heisenberg_action,
    make_square_torus_plan,
    make_swap_query_plan,
    square_torus_edges,
)

jax.config.update("jax_enable_x64", True)


def test_cycle_matches_explicit_virtual_sum():
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    plan = make_exact_plan(4, edges, 2)
    rng = np.random.default_rng(173)
    tensors = rng.normal(size=(4, 2, 2, 2)) + 1j * rng.normal(size=(4, 2, 2, 2))
    spins = np.array([0, 1, 0, 1], dtype=np.int32)
    expected = 0j
    for virtual in itertools.product(range(2), repeat=4):
        term = 1 + 0j
        for i in range(4):
            term *= tensors[(i, spins[i], *(virtual[e] for e in plan.legs[i]))]
        expected += term
    actual = jax.jit(plan.amplitude)(jnp.asarray(tensors), jnp.asarray(spins))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(plan.numpy_amplitude(tensors, spins), expected, atol=2e-12)


def test_batch_and_derivative_are_consistent():
    plan = make_exact_plan(4, [(0, 1), (1, 2), (2, 3), (0, 3)], 2)
    tensors = np.random.default_rng(71).normal(size=(4, 2, 2, 2))
    spins = jnp.array([[0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)
    batch = jax.jit(make_amplitude_batch(plan, chunk_size=2))(jnp.asarray(tensors), spins)
    expected = [plan.numpy_amplitude(tensors, np.asarray(s)) for s in spins]
    np.testing.assert_allclose(batch, expected, atol=1e-12)
    direction = np.random.default_rng(19).normal(size=tensors.shape)
    fun = lambda x: plan.amplitude(x, spins[0]).real
    derivative = jax.jvp(fun, (jnp.asarray(tensors),), (jnp.asarray(direction),))[1]
    eps = 1e-5
    finite = (fun(tensors + eps * direction) - fun(tensors - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(derivative, finite, atol=1e-7, rtol=1e-8)


def test_swap_reuse_matches_independent_contractions_and_gradients():
    plan = make_exact_plan(9, square_torus_edges(3, 3), 2)
    swaps = [(0, 1), (0, 3), (0, 4), (1, 8), (3, 4)]
    reuse = make_swap_query_plan(plan, swaps)
    tensors = jnp.asarray(np.random.default_rng(35).normal(size=(9, 2, 2, 2, 2, 2)))
    spins = jnp.array([0, 1, 0, 1, 0, 0, 1, 1, 0], dtype=jnp.int32)
    rows = [spins]
    for a, b in swaps:
        rows.append(spins.at[a].set(spins[b]).at[b].set(spins[a]))
    expected = jax.jit(make_amplitude_batch(plan))(tensors, jnp.stack(rows))
    actual = jax.jit(reuse.amplitudes)(tensors, spins)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-12)
    derivative = jax.grad(lambda t: reuse.amplitudes(t, spins).sum())(tensors)
    independent = jax.grad(lambda t: make_amplitude_batch(plan)(t, jnp.stack(rows)).sum())(tensors)
    np.testing.assert_allclose(derivative, independent, atol=1e-9, rtol=2e-12)


def test_six_by_six_path_is_bounded_and_native_triangle_is_rejected():
    edges = square_torus_edges(6, 6)
    plan = make_exact_plan(36, edges, 3)
    assert plan.largest_intermediate <= 531441
    assert 0 < plan.estimated_flops < 1_100_000_000
    diagonal = [(y * 6 + x, ((y + 1) % 6) * 6 + (x + 1) % 6)
                for y in range(6) for x in range(6)]
    with pytest.raises(MemoryError, match="intermediate"):
        make_exact_plan(36, [*edges, *diagonal], 3)


def test_square_axis_order_is_left_right_up_down():
    plan = make_square_torus_plan(3, 4, 2)
    for y in range(3):
        for x in range(4):
            i = y * 4 + x
            expected = [y * 4 + (x - 1) % 4, y * 4 + (x + 1) % 4,
                        ((y - 1) % 3) * 4 + x, ((y + 1) % 3) * 4 + x]
            for label, neighbor in zip(plan.legs[i], expected):
                assert set(plan.edges[label]) == {i, neighbor}


def test_heisenberg_propagation_matches_dense_h_and_gradient():
    plan = make_exact_plan(4, [(0, 1), (1, 2), (2, 3), (0, 3)], 2)
    # Include a non-virtual diagonal, a negative coupling, and a repeated bond.
    bonds = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 0)]
    couplings = [1., 1., .5, -.2, .125, .3]
    action = make_heisenberg_action(plan, bonds, couplings)
    rng = np.random.default_rng(197)
    tensors = jnp.asarray(rng.normal(size=(4, 2, 2, 2))
                          + 1j * rng.normal(size=(4, 2, 2, 2)))
    rows = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.int32)
    psi = np.array([plan.numpy_amplitude(np.asarray(tensors), row) for row in rows])
    h = np.zeros((16, 16))
    for old, row in enumerate(rows):
        for (a, b), j in zip(bonds, couplings):
            h[old, old] += j * (2 * row[a] - 1) * (2 * row[b] - 1) / 4
            if row[a] != row[b]:
                changed = row.copy()
                changed[a], changed[b] = row[b], row[a]
                new = sum(int(x) << (3 - i) for i, x in enumerate(changed))
                h[new, old] += j / 2
    actual = jax.jit(jax.vmap(action.products, in_axes=(None, 0)))(tensors, jnp.asarray(rows))
    np.testing.assert_allclose(actual[:, 0], psi, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual[:, 1], h @ psi, atol=1e-11, rtol=2e-12)
    spins = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    direction = jnp.asarray(rng.normal(size=tensors.shape)
                            + 1j * rng.normal(size=tensors.shape))
    _, derivative = jax.jvp(lambda t: action.products(t, spins), (tensors,), (direction,))
    eps = 1e-5
    finite = (action.products(tensors + eps * direction, spins)
              - action.products(tensors - eps * direction, spins)) / (2 * eps)
    np.testing.assert_allclose(derivative, finite, atol=1e-8, rtol=1e-8)


def test_heisenberg_propagation_matches_nonlocal_swap_queries():
    plan = make_square_torus_plan(3, 3, 2)
    bonds = [(0, 1), (0, 3), (0, 4), (1, 8), (3, 4)]
    couplings = [1., 1., .125, -.3, .5]
    action = make_heisenberg_action(plan, bonds, couplings)
    swaps = make_swap_query_plan(plan, bonds)
    rng = np.random.default_rng(274)
    tensors = jnp.asarray(rng.normal(size=(9, 2, 2, 2, 2, 2)))
    row = jnp.array([0, 1, 0, 1, 0, 0, 1, 1, 0], dtype=jnp.int32)
    queried = jax.jit(swaps.amplitudes)(tensors, row)
    unequal = np.array([row[a] != row[b] for a, b in bonds])
    expected = np.dot(np.array(couplings) * unequal * .5, np.asarray(queried[1:]))
    expected += np.dot(np.array(couplings), .25 * (1 - 2 * unequal)) * queried[0]
    actual = jax.jit(action.products)(tensors, row)
    np.testing.assert_allclose(actual[0], queried[0], atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(actual[1], expected, atol=1e-10, rtol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError, match="duplicate"):
        make_exact_plan(2, [(0, 1), (1, 0)], 2)
    with pytest.raises(ValueError, match="isolated"):
        make_exact_plan(3, [(0, 1)], 2)
    with pytest.raises(ValueError, match="lengths"):
        square_torus_edges(2, 2)
