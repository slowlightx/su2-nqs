"""Unsealed next-idea math oracle; no production estimator is changed here.

Tests compare naive carrier expansion and its finite-quadrature Hermitian,
homogeneous completion. Four-spin analytic singlets; no diagonalization.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def _problem(bonds=None):
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


def _inner(left, right, weights):
    return jnp.sum(jnp.asarray(weights)*jnp.conj(left)*right)


def _naive(psi, carrier, hcarrier, diagonal, weights):
    delta = psi-carrier
    numerator = (_inner(carrier, hcarrier, weights).real
                 + 2*_inner(delta, hcarrier, weights).real
                 + _inner(delta, diagonal*delta, weights).real)
    return numerator/_inner(psi, psi, weights).real


def _homogeneous(psi, carrier, hcarrier, diagonal, weights):
    """Rayleigh form of the empirical Hermitian rank-two completion."""
    norm = _inner(carrier, carrier, weights).real
    residual = hcarrier-diagonal*carrier
    anchor = _inner(carrier, residual, weights).real
    overlap = _inner(carrier, psi, weights)
    numerator = (_inner(psi, diagonal*psi, weights).real
                 + 2*(_inner(psi, residual, weights)*overlap/norm).real
                 - anchor*abs(overlap)**2/norm**2)
    return numerator/_inner(psi, psi, weights).real


def test_population_carrier_expansion_and_homogeneous_completion_are_exact_on_singlets():
    first, second, h = _problem()
    carrier = jnp.asarray(first+.13j*second)
    hcarrier = jnp.asarray(h)@carrier
    diagonal = jnp.asarray(3*np.diag(h).real)
    weights = jnp.ones(6)
    for theta in (jnp.array([0., 0.]), jnp.array([.21, -.13])):
        def state(t):
            return carrier+(t[0]+1j*t[1])*jnp.asarray(second)
        def exact(t):
            psi = state(t)
            return jnp.vdot(psi, jnp.asarray(h)@psi).real/jnp.vdot(psi, psi).real
        for estimator in (_naive, _homogeneous):
            objective = lambda t: estimator(state(t), carrier, hcarrier, diagonal, weights)
            value, gradient = jax.value_and_grad(objective)(theta)
            expected, expected_gradient = jax.value_and_grad(exact)(theta)
            np.testing.assert_allclose(value, expected, atol=2e-14, rtol=0.)
            np.testing.assert_allclose(gradient, expected_gradient, atol=3e-14, rtol=0.)


def test_finite_quadrature_naive_expansion_breaks_ray_gauge_but_completion_does_not():
    first, second, h = _problem()
    carrier = jnp.asarray(first+.13j*second)
    hcarrier = jnp.asarray(h)@carrier
    diagonal = jnp.asarray(3*np.diag(h).real)
    weights = jnp.array([.2, 2., .5, 1.7, .4, 1.3])
    k = _inner(carrier, hcarrier, weights)
    norm = _inner(carrier, carrier, weights).real
    naive_phase_force = jax.grad(lambda phase: _naive(
        jnp.exp(1j*phase)*carrier, carrier, hcarrier, diagonal, weights))(0.)
    np.testing.assert_allclose(naive_phase_force, 2*k.imag/norm, atol=2e-15)
    assert abs(float(naive_phase_force)) > 1e-4
    candidate = carrier+(.12-.08j)*jnp.asarray(second)
    reference = _homogeneous(candidate, carrier, hcarrier, diagonal, weights)
    for scale in (1., .03, 7., 1.3+.4j, -.7j):
        measured = _homogeneous(scale*candidate, carrier, hcarrier, diagonal, weights)
        np.testing.assert_allclose(measured, reference, atol=3e-14, rtol=0.)
    gradient = jax.grad(lambda gauge: _homogeneous(
        jnp.exp(gauge[0]+1j*gauge[1])*candidate,
        carrier, hcarrier, diagonal, weights))(jnp.zeros(2))
    np.testing.assert_allclose(gradient, 0., atol=2e-14)
    print({'naive_spurious_phase_force': float(naive_phase_force),
           'completed_gauge_gradient_max': float(jnp.max(abs(gradient)))})


def test_homogeneous_completion_equals_overlap_gauged_carrier_expansion():
    first, second, h = _problem()
    carrier = jnp.asarray(first+.13j*second)
    hcarrier = jnp.asarray(h)@carrier
    diagonal = jnp.asarray(3*np.diag(h).real)
    weights = jnp.array([.2, 2., .5, 1.7, .4, 1.3])
    psi = 2.3j*carrier+(.14+.1j)*jnp.asarray(second)
    overlap_scale = _inner(carrier, psi, weights)/_inner(carrier, carrier, weights)
    gauged = psi/overlap_scale
    np.testing.assert_allclose(_inner(carrier, gauged-carrier, weights), 0., atol=2e-15)
    np.testing.assert_allclose(_homogeneous(psi, carrier, hcarrier, diagonal, weights),
                               _naive(gauged, carrier, hcarrier, diagonal, weights), atol=3e-14)


def test_exact_eigen_carrier_has_zero_force_for_every_positive_quadrature():
    # Two triplet pairs coupled to S=0. Each ferromagnetic pair has energy
    # -1/4, so this analytic singlet has E=-1/2 with no eigensolver.
    first, _, h = _problem(((0, 1, -1.), (2, 3, -1.)))
    carrier = jnp.asarray(np.array([1., -.5, -.5, -.5, -.5, 1.])/np.sqrt(3))
    hcarrier = jnp.asarray(h)@carrier
    np.testing.assert_allclose(hcarrier, -.5*carrier, atol=2e-15)
    diagonal = jnp.asarray(3*np.diag(h).real)
    weights = jnp.array([.2, 2., .5, 1.7, .4, 1.3])
    def state(theta):
        return carrier+(theta[0]+1j*theta[1])*jnp.asarray(first)
    objective = lambda theta: _homogeneous(state(theta), carrier, hcarrier, diagonal, weights)
    value, gradient = jax.value_and_grad(objective)(jnp.zeros(2))
    np.testing.assert_allclose(value, -.5, atol=2e-15)
    np.testing.assert_allclose(gradient, 0., atol=2e-14)
    diag_loss = lambda theta: (_inner(state(theta), diagonal*state(theta), weights).real
                               /_inner(state(theta), state(theta), weights).real)
    old_force = np.asarray(jax.grad(diag_loss)(jnp.zeros(2)))
    assert np.linalg.norm(old_force) > .1
    # The removed forcing component is rank-two spin, not a singlet update.
    _, _, all_pair_h = _problem(tuple((i, j, 1.) for i, j in itertools.combinations(range(4), 2)))
    spin_squared = 3*np.eye(6)+2*all_pair_h
    removed = np.asarray(hcarrier-diagonal*carrier)
    np.testing.assert_allclose(spin_squared@removed, 6*removed, atol=4e-15)
    print({'completed_eigen_carrier_force_max': float(np.max(abs(gradient))),
           'diagonal_eigen_carrier_force_norm': float(np.linalg.norm(old_force))})
