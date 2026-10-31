"""Exact algebra/gradient checks for one globally calibrated residual chart."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.carrier_residual_chart import (
    ResidualChartCoefficients, make_residual_chart_anchor,
    fit_residual_chart_coefficients, residual_chart_amplitudes,
    additive_ray_parameters,
)

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(2026100811)
    complex_values = lambda n: rng.normal(size=n)+1j*rng.normal(size=n)
    c, d0, f = [complex_values(17) for _ in range(3)]
    c[[0, 4, 12]] = 0.
    return c, d0, f, np.exp(rng.normal(size=17))


def test_anchor_and_innovation_are_weighted_orthogonal_and_reference_norm_matches(arrays):
    c, d0, f, q = arrays
    anchor = make_residual_chart_anchor(c, d0, q)
    coeff = fit_residual_chart_coefficients(anchor, f)
    evaluation = residual_chart_amplitudes(anchor, coeff, c, d0, f, mix=.2+.3j, gain=.8-.1j)
    s0, innovation = map(np.asarray, (evaluation.reference_direction, evaluation.innovation))
    z = np.sum(abs(c)**2/q)
    errors = np.asarray([np.sum(c.conj()*s0/q), np.sum(c.conj()*innovation/q),
                         np.sum(s0.conj()*innovation/q), np.sum(abs(s0)**2/q)-z])
    assert np.max(abs(errors))/z < 3e-15
    np.testing.assert_allclose(f, coeff.a*c+coeff.b*s0+innovation, atol=8e-16)
    np.testing.assert_allclose(evaluation.amplitudes, c+(.2+.3j)*s0+(.8-.1j)*innovation, atol=8e-16)
    print({'weighted_chart_orthogonality_relative_max': float(np.max(abs(errors))/z)})


def test_one_global_fit_is_identical_across_evaluation_chunks_and_nodes(arrays):
    c, d0, f, q = arrays
    anchor = make_residual_chart_anchor(c[:11], d0[:11], q[:11])
    coeff = fit_residual_chart_coefficients(anchor, f[:11])
    evaluate = jax.jit(lambda cc, dd, ff: residual_chart_amplitudes(
        anchor, coeff, cc, dd, ff, mix=.13j, gain=.7+.1j))
    full = evaluate(c, d0, f)
    for block in (1, 4, 6):
        chunks = [evaluate(c[i:i+block], d0[i:i+block], f[i:i+block])
                  for i in range(0, len(c), block)]
        for index in range(3):
            np.testing.assert_allclose(np.concatenate([np.asarray(x[index]) for x in chunks]),
                                       full[index], atol=1e-15, rtol=2e-15)
    assert np.isfinite(np.asarray(full.amplitudes)).all()
    assert np.linalg.norm(np.asarray(full.amplitudes)[c == 0]) > .1
    with pytest.raises(ValueError, match='frozen calibration corpus'):
        fit_residual_chart_coefficients(anchor, f[:4])
    with pytest.raises(ValueError, match='global scalar'):
        residual_chart_amplitudes(anchor, coeff, c, d0, f, mix=np.zeros(len(c)))


def test_old_additive_ray_replay_and_complex_phase_scale_covariance(arrays):
    c, d0, f, q = arrays
    anchor = make_residual_chart_anchor(c, d0, q)
    coeff = fit_residual_chart_coefficients(anchor, f)
    conversion = additive_ray_parameters(coeff, .73)
    evaluation = residual_chart_amplitudes(anchor, coeff, c, d0, f,
        mix=conversion.mix, gain=conversion.gain)
    np.testing.assert_allclose(conversion.carrier_coefficient*evaluation.amplitudes,
                               c+f/.73, atol=2e-15, rtol=2e-15)
    # Common complex amplitude scale and unknown Q normalization change no ray.
    phase_scale = -.31+.89j
    changed_anchor = make_residual_chart_anchor(phase_scale*c, phase_scale*d0, 7.2*q)
    changed = fit_residual_chart_coefficients(changed_anchor, phase_scale*f)
    other = residual_chart_amplitudes(changed_anchor, changed,
        phase_scale*c, phase_scale*d0, phase_scale*f, mix=conversion.mix, gain=conversion.gain)
    np.testing.assert_allclose([changed.a, changed.b], [coeff.a, coeff.b], atol=4e-16)
    np.testing.assert_allclose(other.amplitudes, phase_scale*evaluation.amplitudes, atol=2e-15, rtol=2e-15)
    # Changing only the phase/scale of the reference axis must not change the
    # represented old additive ray when its coordinates are transformed.
    reference_scale = 1.7-.9j
    rotated_anchor = make_residual_chart_anchor(c, reference_scale*d0, q)
    rotated = fit_residual_chart_coefficients(rotated_anchor, f)
    rotated_ray = additive_ray_parameters(rotated, .73)
    rotated_state = residual_chart_amplitudes(rotated_anchor, rotated, c, reference_scale*d0, f,
        mix=rotated_ray.mix, gain=rotated_ray.gain)
    np.testing.assert_allclose(rotated_state.innovation, evaluation.innovation, atol=2e-15)
    np.testing.assert_allclose(rotated_state.amplitudes, evaluation.amplitudes, atol=2e-15)


def test_full_autodiff_keeps_calibration_coefficients_and_matches_finite_difference(arrays):
    c, d0, _, q = arrays
    rng = np.random.default_rng(2026100812)
    basis = jnp.asarray(rng.normal(size=(17, 3))+1j*rng.normal(size=(17, 3)))
    anchor = make_residual_chart_anchor(c[:11], d0[:11], q[:11])
    weights = jnp.asarray(np.linspace(.5, 1.7, 6))
    def loss(theta, detach=False):
        f = basis@theta+.1*theta[0]**2*basis[:, 2]
        coeff = fit_residual_chart_coefficients(anchor, f[:11])
        if detach:
            coeff = jax.tree_util.tree_map(jax.lax.stop_gradient, coeff)
        evaluated = residual_chart_amplitudes(anchor, coeff, c[11:], d0[11:], f[11:],
            mix=theta[1]+1j*theta[2], gain=1+.2j)
        return jnp.sum(weights*abs(evaluated.amplitudes)**2)
    theta = jnp.asarray([.21, -.17, .08])
    derivative = np.asarray(jax.jit(jax.grad(loss))(theta))
    epsilon = 1e-6
    finite = np.asarray([(float(loss(theta+epsilon*e))-float(loss(theta-epsilon*e)))/(2*epsilon)
                         for e in np.eye(3)])
    np.testing.assert_allclose(derivative, finite, atol=2e-8, rtol=3e-9)
    detached = np.asarray(jax.grad(lambda t: loss(t, True))(theta))
    assert np.linalg.norm(derivative-detached) > .1
    # Independent explicit projector oracle, not another call to fit/evaluate.
    ca, sa, qa = (jnp.asarray(x) for x in (c[:11], anchor.calibration_s0, q[:11]))
    s_eval = (jnp.asarray(d0[11:])-anchor.reference_overlap*jnp.asarray(c[11:]))/anchor.reference_perp_scale
    def oracle(theta):
        f = basis@theta+.1*theta[0]**2*basis[:, 2]
        a = jnp.sum(ca.conj()*f[:11]/qa)/anchor.carrier_norm
        b = jnp.sum(sa.conj()*f[:11]/qa)/anchor.carrier_norm
        psi = c[11:]+(theta[1]+1j*theta[2])*s_eval+(1+.2j)*(f[11:]-a*c[11:]-b*s_eval)
        return jnp.sum(weights*abs(psi)**2)
    np.testing.assert_allclose(derivative, jax.grad(oracle)(theta), atol=2e-14)
    print({'full_chart_gradient_FD_max': float(np.max(abs(derivative-finite))),
           'incorrect_detach_gradient_difference': float(np.linalg.norm(derivative-detached))})


def test_zero_initial_mixing_does_not_kill_innovation_tangent_or_exact_carrier(arrays):
    c, d0, direction, q = arrays
    anchor = make_residual_chart_anchor(c, d0, q)
    def amplitude(theta):
        f = theta*jnp.asarray(direction)
        coefficients = fit_residual_chart_coefficients(anchor, f)
        return residual_chart_amplitudes(anchor, coefficients, c, d0, f).amplitudes
    values, tangent = jax.jit(lambda x: jax.jvp(amplitude, (x,), (jnp.ones_like(x),)))(jnp.asarray(0.))
    np.testing.assert_array_equal(values, c)
    coeff = fit_residual_chart_coefficients(anchor, direction)
    expected = direction-coeff.a*c-coeff.b*anchor.calibration_s0
    np.testing.assert_allclose(tangent, expected, atol=8e-16)
    assert np.linalg.norm(np.asarray(tangent)) > 1.


def test_anchor_and_evaluation_reference_data_are_stop_gradient(arrays):
    c, d0, f, q = map(jnp.asarray, arrays)
    def loss(cvalue, dvalue, qvalue):
        anchor = make_residual_chart_anchor(cvalue, dvalue, qvalue)
        coeff = fit_residual_chart_coefficients(anchor, f)
        psi = residual_chart_amplitudes(anchor, coeff, cvalue, dvalue, f, mix=.1j).amplitudes
        return jnp.sum(abs(psi)**2)
    derivatives = jax.grad(loss, argnums=(0, 1, 2))(c, d0, q)
    for derivative in derivatives:
        np.testing.assert_array_equal(derivative, np.zeros(derivative.shape))


def test_global_scalar_chart_preserves_n8_singlet_and_full_dihedral_A1():
    # Independently construct singlets as products of four two-spin singlets.
    # Their coherent full D8 orbit averages supply three independent S=0,A1
    # states. No CG decoder, sampled amplitude, or Hamiltonian is evaluated.
    n = 8
    states = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.int32)
    images = np.asarray([image for shift in range(n) for image in
                         ((np.arange(n)+shift) % n, (-np.arange(n)+shift) % n)])
    indices = states[:, images]@(1 << np.arange(7, -1, -1))
    rng = np.random.default_rng(877); columns = []
    for _ in range(12):
        pairs = rng.permutation(n).reshape(-1, 2)
        vector = np.prod((states[:, pairs[:, 0]]-states[:, pairs[:, 1]])/np.sqrt(2.), axis=1)
        columns.append(vector[indices].mean(axis=1))
    basis, singular, _ = np.linalg.svd(np.column_stack(columns), full_matrices=False)
    assert singular[2] > .1 and singular[3] < 1e-14
    c = basis[:, 0]+.2j*basis[:, 1]
    d0 = basis[:, 1]+.1*basis[:, 0]
    f = .17*basis[:, 2]+(.06+.02j)*basis[:, 1]-.03j*basis[:, 0]
    # Even a non-group-invariant finite calibration metric only changes
    # GLOBAL coefficients, so cannot change the exact physical symmetry.
    selected = np.flatnonzero(states.sum(axis=1) == 4)[:37]
    q = np.exp(np.linspace(-1., 1., len(selected)))
    anchor = make_residual_chart_anchor(c[selected], d0[selected], q)
    coeff = fit_residual_chart_coefficients(anchor, f[selected])
    result = residual_chart_amplitudes(anchor, coeff, c, d0, f, mix=.12-.08j, gain=.7+.2j)
    psi = np.asarray(result.amplitudes)
    plus = np.zeros(256, dtype=complex); minus = np.zeros(256, dtype=complex)
    for label, bits in enumerate(states):
        for site in range(n):
            target = label ^ (1 << (7-site))
            if bits[site] == 0: plus[target] += psi[label]
            else: minus[target] += psi[label]
    z = (states.sum(axis=1)-4)*psi
    spin_squared = (.5*np.vdot(plus, plus).real+.5*np.vdot(minus, minus).real+np.vdot(z, z).real)/np.vdot(psi, psi).real
    a1_error = np.max(abs(psi[indices]-psi[:, None]))
    assert spin_squared < 1e-27 and a1_error < 1e-14
    print({'chart_n8_S_squared': float(spin_squared), 'chart_n8_full_A1_max': float(a1_error)})


def test_parallel_reference_near_zero_ray_and_invalid_inputs_fail_closed(arrays):
    c, d0, f, q = arrays
    with pytest.raises(ValueError, match='parallel'):
        make_residual_chart_anchor(c, (1+.2j)*c+1e-14*d0, q, relative_tolerance=1e-10)
    with pytest.raises(ValueError): make_residual_chart_anchor(np.zeros_like(c), d0, q)
    with pytest.raises(ValueError): make_residual_chart_anchor(c, d0, np.zeros_like(q))
    with pytest.raises(ValueError): make_residual_chart_anchor(c, d0[:-1], q)
    with pytest.raises(TypeError): make_residual_chart_anchor(c, d0, q.astype(complex))
    with pytest.raises(ValueError): make_residual_chart_anchor(c, d0, q, relative_tolerance=0.)
    traced_anchor = jax.jit(lambda reference: make_residual_chart_anchor(c, reference, q))(jnp.asarray(c))
    assert not bool(traced_anchor.valid)
    assert np.isnan(np.asarray(traced_anchor.calibration_s0)).all()
    coeff = ResidualChartCoefficients(jnp.asarray(-1.+1e-14+0j), jnp.asarray(.2j), jnp.asarray(True))
    with pytest.raises(ValueError, match='near-zero'):
        additive_ray_parameters(coeff, 1., overlap_tolerance=1e-10)
    failed = jax.jit(lambda a: additive_ray_parameters(
        ResidualChartCoefficients(a, jnp.asarray(.2j), jnp.asarray(True)), 1.))(jnp.asarray(-1.+0j))
    assert not bool(failed.valid) and np.isnan(np.asarray(failed.gain))
    good = fit_residual_chart_coefficients(make_residual_chart_anchor(c, d0, q), f)
    with pytest.raises(ValueError): additive_ray_parameters(good, -1.)
    with pytest.raises(ValueError): additive_ray_parameters(good, np.ones(17))
