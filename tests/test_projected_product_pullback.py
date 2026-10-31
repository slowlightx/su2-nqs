"""Independent dense-H gradients, noncommuting spatial projection and nodes."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback

jax.config.update('jax_enable_x64', True)


def test_chunked_H_pullback_matches_dense_AD_and_JVP_at_nodes_and_incomplete_centers():
    basis = np.asarray([x for x in itertools.product((0, 1), repeat=4) if sum(x) == 2], np.int32)
    codes = jnp.array([8, 4, 2, 1]); perms = (tuple(range(4)), (3, 2, 1, 0))
    projector = PhysicalSpatialProjector(perms)
    bonds = ((0, 1, 1.), (1, 2, .3), (0, 3, -.4), (1, 0, .2))
    # Complex bank and nonlinear real-parameter generator test JAX conjugation.
    generator = lambda x: x*x+1j*jnp.sin(x)
    raw = lambda b, x: b[jnp.dot(x, codes)]**2
    plan, evaluate, _ = make_projected_heisenberg_products(raw, projector, bonds, chunk_size=3)
    p = jnp.arange(16, dtype=jnp.float64)/20
    p = p.at[3].set(0).at[12].set(0)
    bank, vjp = jax.vjp(generator, p)
    subset = np.array([0, 3, 5]); prepared = jax.jit(plan)(basis[subset])
    wp = jnp.array([.1+.2j, -.3j, .7]); wh = jnp.array([.4j, .2-.8j, -.6])
    value, derivative, count, padded = make_projected_product_pullback(raw, 4, chunk_size=3)(bank, prepared, wp, wh)
    gradient = vjp(derivative)[0]
    h = jnp.asarray(SzHeisenbergGraph(4, bonds, n_up=2).to_dense())
    indices = jnp.asarray(basis[:, np.asarray(perms)]@np.asarray(codes))
    def oracle(x):
        phi = jnp.mean(generator(x)[indices]**2, axis=1)
        return jnp.real(jnp.sum(wp*phi[subset]+wh*(h@phi)[subset]))
    expected = jax.grad(oracle)(p)
    np.testing.assert_allclose(gradient, expected, atol=3e-14, rtol=3e-14)
    assert float(value) == pytest.approx(float(oracle(p)), abs=3e-14)
    assert count > 0 and padded >= count
    d = jnp.cos(jnp.arange(16, dtype=jnp.float64))
    def product(x):
        hp, phi, _, _ = evaluate(generator(x), prepared)
        return jnp.real(jnp.sum(wp*phi+wh*hp))
    _, tangent = jax.jvp(product, (p,), (d,))
    assert float(gradient@d) == pytest.approx(float(tangent), abs=3e-14)
    eps = 1e-6
    finite = (oracle(p+eps*d)-oracle(p-eps*d))/(2*eps)
    assert float(gradient@d) == pytest.approx(float(finite), abs=2e-10)


def test_pullback_invalid_static_contracts_fail_before_work():
    for n, chunk in ((0, 2), (64, 2), (True, 2), (4, 0), (4, True)):
        with pytest.raises(ValueError): make_projected_product_pullback(lambda b,x: b[0], n, chunk_size=chunk)
