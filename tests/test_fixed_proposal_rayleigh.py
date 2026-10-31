"""Independent finite-quadrature math tests; no physical simulations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.fixed_proposal_rayleigh import weighted_full_h_loss, weighted_full_h_moments

jax.config.update('jax_enable_x64', True)


def _fixture():
    rng = np.random.default_rng(15101)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    h = (raw + raw.conj().T) / 2
    base = np.array([0., .3+.2j, -.4j, .7, .1-.6j])
    tangent = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    indices = np.array([0, 1, 1, 3, 4, 0, 3])
    q = np.array([.2, .8, .8, .35, .9, .2, .35])
    return h, base, tangent, indices, q


def test_moments_match_explicit_nonuniform_quadrature_and_include_node_mass():
    h, psi, _, ids, q = _fixture()
    hp = h @ psi
    observed = weighted_full_h_moments(psi[ids], hp[ids], q)
    z = np.sum(abs(psi[ids])**2 / q)
    energy = np.sum(psi[ids].conj() * hp[ids] / q) / z
    second = np.sum(abs(hp[ids])**2 / q) / z
    assert abs(energy.imag) > .1  # A finite empirical H inner product is not Hermitian.
    np.testing.assert_allclose(observed.energy_complex, energy, atol=2e-15)
    np.testing.assert_allclose(observed.norm_sum, z, atol=2e-15)
    np.testing.assert_allclose(observed.second_moment, second, atol=2e-15)
    np.testing.assert_allclose(observed.variance, second - abs(energy)**2, atol=8e-15)
    nodes = psi[ids] == 0
    nodal_second = np.sum(abs(hp[ids][nodes])**2 / q[nodes]) / z
    assert nodal_second > 1e-3
    missing_nodes = np.sum(abs(hp[ids][~nodes])**2 / q[~nodes]) / z
    np.testing.assert_allclose(float(observed.second_moment) - missing_nodes, nodal_second, atol=3e-15)


def test_real_loss_gradient_differentiates_both_bra_and_ket_on_fixed_samples():
    h, base, tangent, ids, q = _fixture()
    theta = jnp.asarray([.13, -.21, .07])
    def objective(t):
        psi = jnp.asarray(base) + jnp.asarray(tangent) @ t
        return weighted_full_h_loss(psi[ids], (jnp.asarray(h) @ psi)[ids], q)
    value, derivative = jax.value_and_grad(objective)(theta)
    psi = base + tangent @ np.asarray(theta)
    hp, dh = h @ psi, h @ tangent
    z = np.sum(abs(psi[ids])**2 / q)
    numerator = np.sum(psi[ids].conj() * hp[ids] / q).real
    dn = np.sum((tangent[ids].conj() * hp[ids, None] + psi[ids, None].conj() * dh[ids]) / q[:, None], axis=0).real
    dz = 2 * np.sum(tangent[ids].conj() * psi[ids, None] / q[:, None], axis=0).real
    exact = dn / z - numerator * dz / z**2
    np.testing.assert_allclose(derivative, exact, atol=2e-14)
    epsilon = 1e-6
    finite = np.array([float((objective(theta + epsilon*jnp.eye(3)[k]) - objective(theta - epsilon*jnp.eye(3)[k])) / (2*epsilon)) for k in range(3)])
    np.testing.assert_allclose(derivative, finite, atol=3e-9, rtol=1e-9)
    shortcut = 2*np.sum(tangent[ids].conj() * (hp[ids, None] - float(value)*psi[ids, None]) / q[:, None], axis=0).real/z
    assert np.linalg.norm(shortcut - exact) > .1
    print('full-H fixed-Q finite-difference gradient max error', np.max(abs(finite-exact)))
    print('incorrect single-sided population force discrepancy', np.linalg.norm(shortcut-exact))


def test_exact_uniform_quadrature_recovers_hermitian_rayleigh_and_population_force():
    h, psi, tangent, _, _ = _fixture()
    q = np.ones(5)
    expected = np.vdot(psi, h@psi).real/np.vdot(psi, psi).real
    moment = weighted_full_h_moments(psi, h@psi, q)
    np.testing.assert_allclose(moment.energy_complex, expected, atol=8e-16)
    force = jax.grad(lambda t: weighted_full_h_loss(psi+jnp.asarray(tangent)@t,
                        jnp.asarray(h)@(psi+jnp.asarray(tangent)@t), q))(jnp.zeros(3))
    population = 2*(tangent.conj().T@(h@psi-expected*psi)).real/np.vdot(psi, psi).real
    np.testing.assert_allclose(force, population, atol=3e-15)


def test_phase_scale_Q_rescaling_units_and_stop_gradient():
    h, psi, _, ids, q = _fixture()
    p, hp = psi[ids], (h@psi)[ids]
    reference = weighted_full_h_moments(p, hp, q)
    for scale in (.02, 20., -.3+.7j):
        observed = weighted_full_h_moments(scale*p, scale*hp, q)
        np.testing.assert_allclose(observed.energy_complex, reference.energy_complex, atol=3e-15)
        np.testing.assert_allclose(observed.variance, reference.variance, atol=3e-14)
    changed_q = weighted_full_h_moments(p, hp, q*123.4)
    np.testing.assert_allclose(changed_q.energy_complex, reference.energy_complex, atol=3e-15)
    np.testing.assert_allclose(changed_q.variance, reference.variance, atol=3e-14)
    np.testing.assert_allclose(changed_q.norm_sum, reference.norm_sum/123.4, atol=2e-16)
    per_site = weighted_full_h_moments(p, hp/36, q)
    np.testing.assert_allclose(per_site.variance, reference.variance/36**2, atol=3e-17)
    np.testing.assert_array_equal(jax.grad(lambda x: weighted_full_h_loss(p, hp, x))(jnp.asarray(q)), np.zeros(len(q)))
    gauge = jax.grad(lambda t: weighted_full_h_loss(jnp.exp(t[0]+1j*t[1])*p,
                         jnp.exp(t[0]+1j*t[1])*hp, q))(jnp.zeros(2))
    np.testing.assert_allclose(gauge, 0., atol=2e-15)


def test_ratio_gradient_uses_global_norm_not_average_of_minibatch_ratios():
    h, base, tangent, ids, q = _fixture()
    def arrays(t):
        psi = jnp.asarray(base) + jnp.asarray(tangent) @ t
        return psi[ids], (jnp.asarray(h)@psi)[ids]
    def global_loss(t):
        psi, hp = arrays(t)
        return weighted_full_h_loss(psi, hp, q)
    def summed_loss(t):
        psi, hp = arrays(t)
        numerator, denominator = 0., 0.
        for sl in (slice(0, 3), slice(3, None)):
            numerator += jnp.sum(psi[sl].conj()*hp[sl]/q[sl]).real
            denominator += jnp.sum(abs(psi[sl])**2/q[sl])
        return numerator/denominator
    def wrong_loss(t):
        psi, hp = arrays(t)
        return (weighted_full_h_loss(psi[:3], hp[:3], q[:3]) + weighted_full_h_loss(psi[3:], hp[3:], q[3:]))/2
    theta = jnp.array([.13, -.21, .07])
    np.testing.assert_allclose(jax.value_and_grad(global_loss)(theta)[0], summed_loss(theta), atol=2e-15)
    np.testing.assert_allclose(jax.grad(global_loss)(theta), jax.grad(summed_loss)(theta), atol=3e-15)
    assert abs(float(global_loss(theta)-wrong_loss(theta))) > .01


def test_invalid_inputs_and_compiled_nonfinite_fail_closed():
    p, hp, q = np.array([0., 1.]), np.array([1., 2.]), np.ones(2)
    for bad in (np.zeros(2), -q, q*float('nan'), q.astype(complex)):
        with pytest.raises(ValueError):
            weighted_full_h_loss(p, hp, bad)
    for bad_p, bad_hp in ((np.zeros(2), hp), (p, np.ones(3)), (p, np.full(2, np.inf)), (np.ones((1, 2)), hp)):
        with pytest.raises(ValueError):
            weighted_full_h_moments(bad_p, bad_hp, q)
    assert np.isnan(float(jax.jit(weighted_full_h_loss)(p, hp, q*0)))
    assert np.isnan(float(jax.jit(weighted_full_h_loss)(p*0, hp, q)))
