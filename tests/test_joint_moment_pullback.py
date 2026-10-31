"""Existing plan pullback composes the residual derivative, with TWO RHS."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.joint_moments import residual_moment_cotangents

jax.config.update('jax_enable_x64',True)


def test_two_rhs_residual_pullback_matches_independent_dense_H_AD():
    basis=np.asarray([x for x in itertools.product((0,1),repeat=4) if sum(x)==2],np.int32)
    codes=jnp.array([8,4,2,1]);perms=(tuple(range(4)),(3,2,1,0))
    bonds=((0,1,1.),(1,2,.3),(0,3,-.4),(1,0,.2))
    raw=lambda b,x:b[jnp.dot(x,codes)]**2
    gen=lambda x:x*x+1j*jnp.sin(x)
    parent=jnp.arange(16,dtype=jnp.float64)/20;parent=parent.at[3].set(0).at[12].set(0)
    indices=jnp.asarray(basis[:,np.asarray(perms)]@np.asarray(codes))
    h=jnp.asarray(SzHeisenbergGraph(4,bonds,n_up=2).to_dense())
    center=np.array([0,3,5]);q=np.array([.7,2.,1.1])
    def amplitudes(x):return jnp.mean(gen(x)[indices]**2,axis=1)
    phi=amplitudes(parent);hp=h@phi
    masks=[np.array([True,True,False]),np.ones(3,bool)]
    def moment(x,mask):
        z=amplitudes(x)[center][mask];hz=(h@amplitudes(x))[center][mask];density=jnp.asarray(q[mask])
        norm=jnp.sum(abs(z)**2/density);e=jnp.sum(z.conj()*hz/density)/norm
        return jnp.sum(abs(hz-e*z)**2/density)/norm/4
    wps=[];whs=[]
    for mask in masks:
        wp,wh,_=residual_moment_cotangents(np.asarray(phi)[center][mask],np.asarray(hp)[center][mask],q[mask],4)
        a,b=np.zeros(3,complex),np.zeros(3,complex);a[mask]=wp;b[mask]=wh;wps.append(a);whs.append(b)
    plan,_,_=make_projected_heisenberg_products(raw,PhysicalSpatialProjector(perms),bonds,chunk_size=3)
    pullback=make_projected_product_pullback(raw,4,chunk_size=3)
    bank,vjp=jax.vjp(gen,parent)
    multi=jax.jit(jax.vmap(pullback,in_axes=(None,None,0,0)))
    _,derivative,count,padded=multi(bank,jax.jit(plan)(basis[center]),jnp.asarray(wps),jnp.asarray(whs))
    for i,mask in enumerate(masks):
        actual=vjp(derivative[i])[0];expected=jax.grad(lambda x:moment(x,mask))(parent)
        np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=3e-14)
    assert count[0]==count[1] and padded[0]==padded[1]
