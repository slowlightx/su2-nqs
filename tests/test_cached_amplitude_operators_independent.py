"""Independent complete-Hilbert-space Rayleigh quotient checks for cached SR."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cached_amplitude_operators import (
    build_cached_amplitude_jacobian, solve_cached_projected_sr,
)


def test_nonlinear_all_native_columns_match_exact_rayleigh_gradient_and_F():
    """A symmetric dense H supplies both derivative terms without IS error."""
    rng=np.random.default_rng(26110107)
    linear=rng.normal(size=(8,11))
    nonlinear=rng.normal(size=(8,11))
    theta=rng.normal(size=11)
    shift=rng.normal(size=8)
    shift[3]=-(linear@theta+.03*np.sin(nonlinear@theta))[3]
    matrix=rng.normal(size=(8,8));hamiltonian=(matrix+matrix.T)/2
    def values(x):
        return linear@x+.03*np.sin(nonlinear@x)+shift
    def batch(x,rows):
        all_values=jnp.asarray(linear)@x+.03*jnp.sin(jnp.asarray(nonlinear)@x)+jnp.asarray(shift)
        return all_values[rows[:,0]]
    psi=values(theta);assert psi[3]==0
    hpsi=hamiltonian@psi
    assert abs(hpsi[3])>.01
    rows=np.arange(8,dtype=np.int32)[:,None]
    density=np.full(8,1/8,dtype=np.float64)
    cache=build_cached_amplitude_jacobian(batch,theta,rows,psi,density,
        row_batch_size=3,maximum_host_bytes=10**7,source_identity="complete-real-Hilbert-nonlinear-oracle")
    jacobian=linear+.03*np.cos(nonlinear@theta)[:,None]*nonlinear
    np.testing.assert_allclose(cache.jacobian,jacobian,atol=2e-15,rtol=2e-15)
    # Uniform exhaustive quadrature is an exact physical inner product.
    norm=psi@psi;energy=psi@hpsi/norm
    exact_gradient=2*jacobian.T@(hpsi-energy*psi)/norm
    normalized=psi/np.sqrt(norm)
    horizontal=(np.eye(8)-np.outer(normalized,normalized))@jacobian/np.sqrt(norm)
    fisher=horizontal.T@horizontal
    damping=.17
    direction,gradient,info=solve_cached_projected_sr(cache,psi,hpsi,density,
        damping=damping,parameter_chunk_size=4,maximum_device_bytes=10**7)
    np.testing.assert_allclose(gradient,exact_gradient,atol=2e-13,rtol=2e-13)
    expected=np.linalg.solve(fisher+damping*np.eye(11),exact_gradient)
    np.testing.assert_allclose(direction,expected,atol=2e-12,rtol=2e-12)
    assert np.linalg.norm(direction-np.linalg.solve(2*fisher+damping*np.eye(11),exact_gradient))>.01
    def energy_at(x):
        p=values(x);return p@hamiltonian@p/(p@p)
    epsilon=1e-6
    finite=np.array([(energy_at(theta+epsilon*np.eye(11)[i])-energy_at(theta-epsilon*np.eye(11)[i]))/(2*epsilon) for i in range(11)])
    np.testing.assert_allclose(gradient,finite,atol=3e-8,rtol=3e-8)
    # At an exact target node, dpsi*Hpsi is still part of the true gradient.
    node_contribution=2*jacobian[3]*hpsi[3]/norm
    assert np.linalg.norm(node_contribution)>1e-3
    independently_measured=np.linalg.norm((fisher+damping*np.eye(11))@direction-exact_gradient)/np.linalg.norm(exact_gradient)
    assert independently_measured<2e-12 and info["relative_linear_residual"]<2e-12


def test_constant_density_rescaling_requires_identity_refresh_but_cancels_in_SR():
    """An unknown common density scale cancels algebraically, not in provenance."""
    rng=np.random.default_rng(891)
    jacobian=rng.normal(size=(5,7));theta=rng.normal(size=7)
    psi=jacobian@theta;hpsi=rng.normal(size=5)
    rows=np.arange(5,dtype=np.int32)[:,None]
    q=np.exp(rng.normal(size=5))
    batch=lambda x,r:jnp.asarray(jacobian)[r[:,0]]@x
    make=lambda density:build_cached_amplitude_jacobian(batch,theta,rows,psi,density,
        row_batch_size=2,maximum_host_bytes=10**7,source_identity="density-scale-oracle")
    original=make(q)
    with pytest.raises(ValueError,match="does not match"):
        original.verify(theta,rows,psi,7*q,source_identity="density-scale-oracle")
    rescaled=make(7*q)
    outputs=[solve_cached_projected_sr(cache,psi,hpsi,density,damping=.31,
        parameter_chunk_size=3,maximum_device_bytes=10**7) for cache,density in ((original,q),(rescaled,7*q))]
    for index in (0,1):
        np.testing.assert_allclose(outputs[0][index],outputs[1][index],atol=2e-13,rtol=2e-13)
