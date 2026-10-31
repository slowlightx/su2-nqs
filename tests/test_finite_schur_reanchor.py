"""New physical-parent forces, unchanged nonlinear model/H AD convention."""
import itertools
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.joint_moments import residual_moment_cotangents
from su2_nqs.vmc.full_rayleigh import full_rayleigh_gradient,reverse_rayleigh_weights
from projects.clebsch_tree_nqs.workflows.spin_finite_schur_reanchor import CONFIG,configuration,retained_coefficients

jax.config.update('jax_enable_x64',True)


def test_reanchored_complete_two_sided_forces_match_dense_H_AD():
    basis=np.asarray([x for x in itertools.product((0,1),repeat=4) if sum(x)==2],np.int32)
    codes=jnp.array([8,4,2,1]);images=(tuple(range(4)),(3,2,1,0));bonds=((0,1,1.),(1,2,.3),(0,3,-.4),(1,0,.2))
    h=jnp.asarray(SzHeisenbergGraph(4,bonds,n_up=2).to_dense());ids=jnp.asarray(basis[:,np.asarray(images)]@np.asarray(codes))
    raw=lambda bank,x:bank[jnp.dot(x,codes)]**2
    gen=lambda theta:theta**2+1j*jnp.sin(theta)
    theta=jnp.arange(16,dtype=jnp.float64)/20;phi=lambda t:jnp.mean(gen(t)[ids]**2,axis=1)
    ref=phi(theta);c=jnp.array([0,.3,-.5,.2,0,.7],dtype=jnp.complex128)
    powers=np.column_stack((c,h@c,h@h@c));parent,hparent=polynomial_carrier_products(powers,[.9,-.04]);centers=np.array([0,3,5]);q=np.array([.7,2.,1.1]);scale=.17
    def state(t):return jnp.asarray(parent)+scale*(phi(t)-ref)
    def energy(t):
        p=state(t)[centers];hp=(h@state(t))[centers];return jnp.real(jnp.sum(p.conj()*hp/q)/jnp.sum(abs(p)**2/q))/4
    def residual(t):
        p=state(t)[centers];hp=(h@state(t))[centers];norm=jnp.sum(abs(p)**2/q);e=jnp.sum(p.conj()*hp/q)/norm
        return jnp.sum(abs(hp-e*p)**2/q)/norm/4
    plan,_,_=make_projected_heisenberg_products(raw,PhysicalSpatialProjector(images),bonds,chunk_size=3)
    pullback=make_projected_product_pullback(raw,4,chunk_size=3);bank,vjp=jax.vjp(gen,theta)
    p,hp=parent[centers],hparent[centers];wp,wh,_=residual_moment_cotangents(p,hp,q,4)
    _,gr,*_=pullback(bank,jax.jit(plan)(basis[centers]),scale*wp,scale*wh)
    np.testing.assert_allclose(vjp(gr)[0],jax.grad(residual)(theta),atol=4e-14,rtol=4e-14)
    _,reverse,*_=pullback(bank,jax.jit(plan)(basis[centers]),jnp.zeros(3,complex),scale*reverse_rayleigh_weights(p,q))
    jac=np.asarray(jax.jacfwd(lambda t:scale*phi(t)[centers])(theta));reverse=np.asarray(vjp(reverse)[0])
    actual=full_rayleigh_gradient(p,hp,jac,q,reverse)['gradient']/4
    np.testing.assert_allclose(actual,jax.grad(energy)(theta),atol=3e-14,rtol=3e-14)


def test_protocol_retains_checkpoint_and_removes_obsolete_ladder():
    spec=configuration(CONFIG);coef=retained_coefficients(spec)
    assert coef.shape==(2,) and coef[1]<0
    assert not any(k in spec for k in ('methods','primary_method','method_meanings','mechanism','spectral_control'))
    assert spec['steps']==1 and spec['protection']['maximum_variance_ratio']==1.
    assert spec['parent'].endswith('spin_finite_schur_cost_v1/triangular')
    assert spec['seeds']==dict(calibration=2026100411,audit=2026100412,bootstrap=2026100413)
