"""Independent dense-QGT/imaginary-time checks, including carrier nodes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.importance_tangent import (make_amplitude_jacobian, tangent_system,
    direction_diagnostics, importance_sr_direction)

jax.config.update('jax_enable_x64', True)


def test_complex_amplitude_sr_matches_dense_metric_at_nodes_and_gauges():
    p = np.array([1., 0., .3j])
    h = np.array([[0, 1, 1j], [1, 2, 0], [-1j, 0, 3]], complex)
    j = np.column_stack((np.array([0, 1, 0]), np.array([0, 0, 1j]), p, 1j*p))
    # Enumerate each Hilbert row exactly with positive-Q quadrature weights.
    q = np.array([.2, 3., 1.7]); scale = np.sqrt(q)
    values, products, derivatives = p*scale, (h@p)*scale, j*scale[:, None]
    direction, info = importance_sr_direction(values, products, derivatives, q, damping=.13)
    norm = np.vdot(p, p).real
    tangent = (j-p[:, None]*(p.conj()@j)[None, :]/norm)/np.sqrt(norm)
    residual = (h@p-p*np.vdot(p, h@p)/norm)/np.sqrt(norm)
    metric = 2*(tangent.conj().T@tangent).real
    gradient = 2*(tangent.conj().T@residual).real
    expected = np.linalg.solve(metric+.13*np.eye(4), gradient)
    np.testing.assert_allclose(direction, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(direction[2:], 0, atol=2e-14)
    assert info['exact_node_records'] == 1 and info['relative_linear_residual'] < 1e-13
    assert info['descent_alignment_cosine'] > 0
    for qscale in (.01, 23.):
        changed, _ = importance_sr_direction(values, products, derivatives, qscale*q, damping=.13)
        np.testing.assert_allclose(changed, expected, atol=2e-14, rtol=2e-14)
    # Exact Hermitian H derivative independently computed by central difference.
    def energy(alpha):
        z = p-alpha*j@direction
        return np.vdot(z, h@z).real/np.vdot(z, z).real
    finite = (energy(1e-6)-energy(-1e-6))/2e-6
    assert finite == pytest.approx(info['predicted_one_sided_energy_slope_total'], abs=1e-9)


def test_born_amplitude_sr_equals_log_score_oracle_when_no_nodes():
    rng = np.random.default_rng(12)
    p = rng.normal(size=7)+1j*rng.normal(size=7)
    scores = rng.normal(size=(7,5))+1j*rng.normal(size=(7,5))
    e = rng.normal(size=7)+1j*rng.normal(size=7)
    centered = scores-scores.mean(axis=0)
    metric = 2*(centered.conj().T@centered).real/7
    gradient = 2*(centered.conj().T@(e-e.mean())).real/7
    expected = np.linalg.solve(metric+.1*np.eye(5), gradient)
    direction, _ = importance_sr_direction(p, e*p, p[:, None]*scores, abs(p)**2)
    np.testing.assert_allclose(direction, expected, atol=3e-14, rtol=3e-14)


def test_amplitude_jacobian_is_correct_at_exact_node_and_full_real_pytree():
    p = {'a': jnp.array([0., .2], dtype=jnp.float64)}
    fn = lambda p, x: p['a'][0]*x[0]+1j*p['a'][1]*(1-x[0])
    jac = make_amplitude_jacobian(fn, p, chunk_size=1)
    actual = jac(p, jnp.array([[0], [1]]))
    np.testing.assert_allclose(actual, [[0, 1j], [1, 0]], atol=1e-15)
    assert fn(p, jnp.array([1])) == 0 and actual[1,0] != 0
    with pytest.raises(ValueError): make_amplitude_jacobian(fn, p, chunk_size=False)
    with pytest.raises(ValueError): make_amplitude_jacobian(fn, {'a': jnp.ones(1, dtype=jnp.complex128)})


def test_high_energy_orthogonal_residual_can_lower_energy_through_coherence():
    h = np.array([[0., 1.], [1., 100.]])
    p, j = np.array([1.,0]), np.array([[0.],[1.]])
    direction, d = importance_sr_direction(p, h@p, j, np.ones(2))
    assert d['predicted_one_sided_energy_slope_total'] < 0
    t = -.005*j@direction
    new = p+t
    assert np.vdot(t,h@t)/np.vdot(t,t) == pytest.approx(100.)
    assert np.vdot(new,h@new)/np.vdot(new,new) < 0
    reverse = direction_diagnostics(p, h@p, j, np.ones(2), -direction)
    assert reverse['descent_alignment_cosine'] < 0


def test_invalid_positive_Q_and_null_target_fail_or_report_zero():
    p, hp, j, q = np.ones(2), np.zeros(2), np.ones((2,1)), np.ones(2)
    d, result = importance_sr_direction(p, hp, j, q)
    np.testing.assert_array_equal(d, [0.])
    assert result['descent_alignment_cosine'] == result['gradient_norm'] == 0
    for kwargs in ({'damping': 0}, {'damping': True}):
        with pytest.raises(ValueError): importance_sr_direction(p,hp,j,q,**kwargs)
    with pytest.raises(ValueError): tangent_system(p,hp,j,q*0)
    with pytest.raises(ValueError): tangent_system(p*0,hp,j,q)
    with pytest.raises(ValueError): direction_diagnostics(p,hp,j,q,np.ones(2))
