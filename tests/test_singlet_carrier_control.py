"""Independent physical and finite-quadrature checks of the reusable control."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.singlet_carrier_control import (
    carrier_control_action, carrier_control_evaluation, carrier_control_loss,
    carrier_control_directional_force, parent_directional_force,
)

jax.config.update('jax_enable_x64', True)


def _problem(bonds=None):
    """Four physical spins, two analytic singlets and independent Pauli H."""
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(int)
    selected = np.flatnonzero(bits.sum(axis=1) == 2)
    singlet = np.array([[0., -1/np.sqrt(2)], [1/np.sqrt(2), 0.]])
    first = singlet[bits[:, 0], bits[:, 1]]*singlet[bits[:, 2], bits[:, 3]]
    second = singlet[bits[:, 0], bits[:, 2]]*singlet[bits[:, 1], bits[:, 3]]
    spin = (np.array([[0., .5], [.5, 0.]], complex),
            np.array([[0., .5j], [-.5j, 0.]], complex), np.diag([-.5, .5]))
    h = np.zeros((16, 16), complex)
    if bonds is None:
        bonds = ((0, 1, 1.), (1, 2, .7), (2, 3, .8), (0, 3, .9), (0, 2, .2))
    for i, j, coupling in bonds:
        for component in spin:
            term = np.ones((1, 1))
            for site in range(4):
                term = np.kron(term, component if site in (i, j) else np.eye(2))
            h += coupling*term
    return first[selected], second[selected], h[np.ix_(selected, selected)]


def _fixture():
    first, second, h = _problem()
    carrier = first+.13j*second
    return carrier, h@carrier, 3*np.diag(h).real, np.array([.2, 2., .5, 1.7, .4, 1.3]), second, h


def test_action_matches_independent_weighted_hermitian_rank_two_matrix():
    carrier, hcarrier, diagonal, q, direction, _ = _fixture()
    w = np.diag(1/q)
    norm = np.vdot(carrier, w@carrier).real
    r = hcarrier-diagonal*carrier
    anchor = np.vdot(carrier, w@r).real
    matrix = (np.diag(diagonal)+np.outer(r, carrier.conj())@w/norm
              + np.outer(carrier, r.conj())@w/norm
              - anchor*np.outer(carrier, carrier.conj())@w/norm**2)
    np.testing.assert_allclose(w@matrix, matrix.conj().T@w, atol=2e-14)
    candidate = carrier+(.12+.31j)*direction
    actual = carrier_control_action(candidate, carrier, hcarrier, q, diagonal)
    np.testing.assert_allclose(actual, matrix@candidate, atol=2e-14)
    energy = carrier_control_loss(candidate, carrier, hcarrier, q, diagonal)
    np.testing.assert_allclose(energy, (np.vdot(candidate, w@matrix@candidate)/np.vdot(candidate, w@candidate)).real, atol=2e-14)
    np.testing.assert_allclose(carrier_control_action(np.zeros(6), carrier, hcarrier, q, diagonal), 0., atol=0.)


def test_candidate_carrier_and_proposal_gauge_changes_do_not_change_the_energy():
    c, hc, d, q, direction, _ = _fixture()
    psi = c+(.17-.2j)*direction
    expected = carrier_control_evaluation(psi, c, hc, q, d)
    for scale in (1., .03, 9., -.3+.7j):
        candidate = carrier_control_evaluation(scale*psi, c, hc, q, d)
        np.testing.assert_allclose(candidate.energy, expected.energy, atol=2e-14)
        np.testing.assert_allclose(candidate.action, scale*expected.action, atol=3e-14)
        anchor = carrier_control_evaluation(psi, scale*c, scale*hc, q, d)
        np.testing.assert_allclose(anchor.energy, expected.energy, atol=2e-14)
        np.testing.assert_allclose(anchor.action, expected.action, atol=3e-14)
    changed_q = carrier_control_evaluation(psi, c, hc, 12.3*q, d)
    np.testing.assert_allclose(changed_q.energy, expected.energy, atol=2e-14)
    np.testing.assert_allclose(changed_q.action, expected.action, atol=3e-14)
    gauge = jax.grad(lambda x: carrier_control_loss(jnp.exp(x[0]+1j*x[1])*psi, c, hc, q, d))(jnp.zeros(2))
    np.testing.assert_allclose(gauge, 0., atol=3e-14)


def test_population_singlet_energy_and_gradient_match_full_dense_H():
    c, hc, d, _, direction, h = _fixture()
    q = jnp.ones(6)
    def psi(theta):
        return jnp.asarray(c)+(theta[0]+1j*theta[1])*jnp.asarray(direction)
    def exact(theta):
        candidate = psi(theta)
        return jnp.vdot(candidate, jnp.asarray(h)@candidate).real/jnp.vdot(candidate, candidate).real
    for theta in (jnp.zeros(2), jnp.array([.19, -.13])):
        current = lambda x: carrier_control_loss(psi(x), c, hc, q, d)
        value, derivative = jax.jit(jax.value_and_grad(current))(theta)
        target, gradient = jax.value_and_grad(exact)(theta)
        np.testing.assert_allclose(value, target, atol=2e-14)
        np.testing.assert_allclose(derivative, gradient, atol=3e-14)
    # The removed carrier forcing component is genuinely a spin-two tensor.
    _, _, all_pairs = _problem(tuple((i, j, 1.) for i, j in itertools.combinations(range(4), 2)))
    spin_squared = 3*np.eye(6)+2*all_pairs
    residual = hc-d*c
    np.testing.assert_allclose(spin_squared@residual, 6*residual, atol=4e-15)


def test_parent_force_centers_complex_energy_and_includes_carrier_nodes():
    c, direction, h = _problem()
    hc = h@c
    q = np.array([.2, 2., .5, 1.7, .4, 1.3])
    d = 3*np.diag(h).real
    tangent = np.column_stack((direction, 1j*direction, c, 1j*c))
    force = parent_directional_force(c, hc, tangent, q)
    measured = carrier_control_directional_force(c, tangent, c, hc, q, d)
    np.testing.assert_allclose(force, measured, atol=3e-14)
    np.testing.assert_allclose(force[2:], 0., atol=2e-14)
    nodes = c == 0
    assert np.linalg.norm(hc[nodes]) > .1
    nodal_tangent = np.where(nodes, hc, 0.)
    expected_node_force = 2*np.sum(abs(hc[nodes])**2/q[nodes])/np.sum(abs(c)**2/q)
    observed = parent_directional_force(c, hc, nodal_tangent, q)
    np.testing.assert_allclose(observed, expected_node_force, atol=2e-14)
    assert float(observed) > .01
    np.testing.assert_allclose(parent_directional_force(c, d*c, nodal_tangent, q), 0., atol=0.)
    # Complex carrier makes a finite-quadrature imaginary energy visible.
    complex_c = c+.13j*direction
    evaluation = carrier_control_evaluation(complex_c, complex_c, h@complex_c, q, d)
    assert abs(float(evaluation.carrier_energy_complex.imag)) > 1e-5
    np.testing.assert_allclose(parent_directional_force(complex_c, h@complex_c, 1j*complex_c, q), 0., atol=2e-14)


def test_arbitrary_state_directional_force_matches_jvp_and_chunked_centered_force():
    c, hc, d, q, direction, _ = _fixture()
    rng = np.random.default_rng(883)
    psi = c+(.17-.2j)*direction
    tangents = rng.normal(size=(6, 3))+1j*rng.normal(size=(6, 3))
    result = carrier_control_evaluation(psi, c, hc, q, d)
    actual = carrier_control_directional_force(psi, tangents, c, hc, q, d)
    expected = jax.grad(lambda theta: carrier_control_loss(
        jnp.asarray(psi)+jnp.asarray(tangents)@theta, c, hc, q, d))(jnp.zeros(3))
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    for size in (1, 4):
        accumulated = np.zeros(3)
        for start in range(0, 6, size):
            chunk = slice(start, start+size)
            accumulated += 2*np.sum(tangents[chunk].conj()*np.asarray(result.residual)[chunk, None]/q[chunk, None], axis=0).real/float(result.norm_sum)
        np.testing.assert_allclose(accumulated, actual, atol=3e-14)


def test_exact_eigen_carrier_zero_force_without_averaging_and_static_data_no_gradient():
    first, _, h = _problem(((0, 1, -1.), (2, 3, -1.)))
    c = np.array([1., -.5, -.5, -.5, -.5, 1.])/np.sqrt(3)
    hc = h@c
    q = np.array([.2, 2., .5, 1.7, .4, 1.3])
    d = 3*np.diag(h).real
    np.testing.assert_allclose(hc, -.5*c, atol=2e-15)
    forces = parent_directional_force(c, hc, np.column_stack((first, 1j*first)), q)
    np.testing.assert_allclose(forces, 0., atol=2e-14)
    assert abs(float(parent_directional_force(c, d*c, first, q))) > .1
    energy = carrier_control_loss(c, c, hc, q, d)
    np.testing.assert_allclose(energy, -.5, atol=2e-15)
    static_gradient = jax.grad(lambda density: carrier_control_loss(c, c, hc, density, d))(jnp.asarray(q))
    np.testing.assert_array_equal(static_gradient, np.zeros(6))


def test_invalid_inputs_fail_closed_without_regularizing_nodes():
    c, hc, d, q, _, _ = _fixture()
    for bad_q in (q*0, np.full(6, np.nan), -q):
        with pytest.raises(ValueError):
            carrier_control_loss(c, c, hc, bad_q, d)
    with pytest.raises(ValueError):
        carrier_control_loss(np.zeros(6), c, hc, q, d)
    with pytest.raises(ValueError):
        carrier_control_loss(c, np.zeros(6), hc, q, d)
    with pytest.raises(ValueError):
        parent_directional_force(c, hc, np.ones((5, 2)), q)
    with pytest.raises(TypeError):
        carrier_control_loss(c, c, hc, q, d.astype(complex))
    compiled = jax.jit(carrier_control_loss)
    assert np.isnan(float(compiled(c, c, hc, q*0, d)))
