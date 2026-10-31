import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.dressing.gate_action import apply_local_gate
from su2_nqs.dressing.multiplicity_gate import plaquette_generators, plaquette_unitary
from su2_nqs.dressing.physical_gate_amplitude import (
    jax_plaquette_unitary, make_four_spin_gate_expansion,
    make_four_spin_gated_amplitude,
)


def _rows(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int32)


def _raw(params, bits):
    return params[bits @ (2**jnp.arange(bits.size-1, -1, -1))]


def test_jax_gate_identity_product_and_all_four_generator_jvps():
    zero = jnp.zeros(4)
    np.testing.assert_array_equal(jax_plaquette_unitary(zero), np.eye(16))
    angles = jnp.array([.31, -.27, .18, .47])
    actual = np.asarray(jax.jit(jax_plaquette_unitary)(angles))
    np.testing.assert_allclose(actual, plaquette_unitary(angles), atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual.T @ actual, np.eye(16), atol=2e-14, rtol=0)
    for index, expected in enumerate(plaquette_generators()):
        tangent = jax.jvp(jax_plaquette_unitary, (zero,), (jnp.eye(4)[index],))[1]
        np.testing.assert_allclose(tangent, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize('groups', [((5, 0, 3, 1),), ((5, 0, 3, 1), (2, 3, 0, 4))])
def test_complex_physical_amplitudes_gate_order_spin_and_jvp(groups):
    rng = np.random.default_rng(61027)
    state = rng.normal(size=64)+1j*rng.normal(size=64)
    state /= np.linalg.norm(state)
    angles = rng.normal(scale=.23, size=(len(groups), 4))
    direction = rng.normal(size=angles.shape)
    bits = jnp.asarray(_rows(6))
    amplitude = make_four_spin_gated_amplitude(_raw, 6, groups)
    all_amplitudes = jax.jit(lambda a: jax.vmap(lambda x: amplitude(jnp.asarray(state), a, x))(bits))
    np.testing.assert_array_equal(all_amplitudes(jnp.zeros_like(angles)), state)
    expected = state.copy()
    for sites, a in zip(groups, angles, strict=True):
        expected = apply_local_gate(expected, plaquette_unitary(a), sites)
    actual = np.asarray(all_amplitudes(angles))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert abs(np.vdot(actual, actual)-1) < 2e-14
    spin, _ = spin_squared_matrix(6)
    assert abs(np.vdot(actual, spin @ actual)-np.vdot(state, spin @ state)) < 3e-14
    tangent = np.asarray(jax.jvp(all_amplitudes, (jnp.asarray(angles),), (jnp.asarray(direction),))[1])
    delta = 2e-6
    finite = (np.asarray(all_amplitudes(angles+delta*direction))
              -np.asarray(all_amplitudes(angles-delta*direction)))/(2*delta)
    np.testing.assert_allclose(tangent, finite, atol=7e-11, rtol=0)
    assert abs(2*np.vdot(actual, tangent).real) < 3e-14
    # Identity JVP is the SUM of physical generator actions, not its transpose.
    expected_zero = np.zeros_like(state)
    for sites, d in zip(groups, direction, strict=True):
        generator = sum(a*g for a, g in zip(d, plaquette_generators(), strict=True))
        expected_zero += apply_local_gate(state, generator, sites)
    zero_tangent = jax.jvp(all_amplitudes, (jnp.zeros_like(angles),), (jnp.asarray(direction),))[1]
    np.testing.assert_allclose(zero_tangent, expected_zero, atol=2e-14, rtol=0)


def test_expansion_has_only_valid_equal_doubled_m_rows_and_fixed_support():
    expand = make_four_spin_gate_expansion(6, ((5, 0, 3, 1),))
    for row in _rows(6):
        rows, weights = expand(jnp.array([[.2, -.3, .1, .4]]), row)
        assert rows.shape == (6, 6) and weights.shape == (6,)
        assert np.all(np.asarray(rows).sum(1) == row.sum())
        np.testing.assert_array_equal(np.asarray(rows)[:, [2, 4]], np.broadcast_to(row[[2, 4]], (6, 2)))


def test_invalid_sites_angles_and_expansion_budgets_rejected():
    for sites in ((0, 1, 2, 2), (0., 1, 2, 3), (False, 1, 2, 3), (0, 1, 2, 6)):
        with pytest.raises(ValueError):
            make_four_spin_gate_expansion(6, (sites,))
    with pytest.raises(ValueError):
        make_four_spin_gate_expansion(6, ((0, 1, 2, 3),)*2, max_expansion_terms=6)
    for value in ([0, 0, 0], [0, 0, 0, np.nan], np.ones(4, complex)):
        with pytest.raises(ValueError):
            jax_plaquette_unitary(value)
    with pytest.raises(ValueError):
        make_four_spin_gate_expansion(6, ((0, 1, 2, 3),))(np.zeros((1, 4)), np.full(6, 2))
