"""Independent exact physical-matrix oracles for scalar SU(2)-sector filters."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.frontier.operator_filter_learning import (
    apply_filter_controller, build_filter_sketch, filter_trace_energy,
    filtered_columns, init_filter_controller, sketch_ritz_columns,
)


def test_complex_gauge_and_seed_column_rotations_preserve_physical_projector():
    rng = np.random.default_rng(27102601)
    def unitary(n, k):
        return np.linalg.qr(rng.normal(size=(n, k))+1j*rng.normal(size=(n, k)))[0]
    u, r, v = unitary(13, 13), unitary(3, 3), unitary(13, 3)
    z = rng.normal(size=(13, 13))+1j*rng.normal(size=(13, 13))
    h = (z+z.conj().T)/2
    lo, hi = np.linalg.eigvalsh(h)[[0, -1]]+[-.1, .1]
    a = np.array([-.29, .21, -.24])
    s = build_filter_sketch(lambda x: h @ x, v, lo, hi)
    rotated = build_filter_sketch(lambda x: u.conj().T @ h @ u @ x, u.conj().T @ v @ r, lo, hi)
    w, report = filtered_columns(s, a)
    wr, _ = filtered_columns(rotated, a)
    np.testing.assert_allclose(u @ wr @ wr.conj().T @ u.conj().T, w @ w.conj().T, atol=2e-13)
    actual = np.trace(w.conj().T @ h @ w).real/3
    np.testing.assert_allclose(report['raw_trace_energy'], actual, atol=2e-13)
    np.testing.assert_allclose(s.center+s.radius*filter_trace_energy(s.gram, s.energy, a), actual, atol=2e-13)
    assert s.H_columns == 12
    assert report['minimum_filter_Gram_eigenvalue'] >= .01
    zero, _ = filtered_columns(s, np.zeros(3))
    np.testing.assert_array_equal(zero, v)
    ritz, ritz_report = sketch_ritz_columns(s)
    assert ritz_report['raw_trace_energy'] <= actual+1e-12
    np.testing.assert_allclose(ritz.conj().T @ ritz, np.eye(3), atol=2e-13)


def test_new_operator_directions_have_correct_nonzero_energy_gradient():
    h = np.array([[0., .4, .2, 0.], [.4, 1., .3, .1], [.2, .3, 2., -.2], [0., .1, -.2, 3.]])
    v = np.eye(4)[:, :1]
    s = build_filter_sketch(lambda x: h @ x, v, -1., 4.)
    f = lambda a: filter_trace_energy(jnp.asarray(s.gram), jnp.asarray(s.energy), a)
    grad = np.asarray(jax.grad(f)(jnp.zeros(3)))
    finite = np.array([(float(f(np.eye(3)[i]*1e-5))-float(f(-np.eye(3)[i]*1e-5)))/2e-5 for i in range(3)])
    np.testing.assert_allclose(grad, finite, atol=2e-10)
    assert np.linalg.norm(grad) > .01
    for a in ([-.31, 0., 0.], [np.nan, 0., 0.], [0j, 0j, 0j]):
        with pytest.raises(ValueError): filtered_columns(s, np.asarray(a))
    with pytest.raises(ValueError): build_filter_sketch(lambda x: h @ x, 2*v, -1., 4.)


@pytest.mark.parametrize('kind', ['linear', 'mlp', 'attention'])
def test_shared_controller_zero_initialization_and_q_token_permutation(kind):
    x = jax.random.normal(jax.random.PRNGKey(12), (6, 11), dtype=jnp.float64)
    p = init_filter_controller(jax.random.PRNGKey(13), 11, kind)
    np.testing.assert_array_equal(apply_filter_controller(p, x, kind), np.zeros((6, 3)))
    p['out'] = jnp.arange(p['out'].size, dtype=jnp.float64).reshape(p['out'].shape)/100
    perm = np.array([4, 0, 3, 1, 5, 2])
    a = np.asarray(apply_filter_controller(p, x, kind))
    np.testing.assert_allclose(apply_filter_controller(p, x[perm], kind), a[perm], atol=2e-14)
    assert np.max(abs(a)) < .3
