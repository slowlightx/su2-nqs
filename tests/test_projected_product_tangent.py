"""Nonlinear bank JVP agrees with an independent computational-basis H."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_product_tangent import make_projected_product_jvp

jax.config.update('jax_enable_x64',True)


def test_nonlinear_bank_product_jvp_keeps_exact_node_H_tangent_and_all_multiplicities():
    basis=np.asarray([x for x in itertools.product((0,1),repeat=4) if sum(x)==2],np.int32)
    codes=jnp.array([8,4,2,1]);perms=(tuple(range(4)),(3,2,1,0))
    projector=PhysicalSpatialProjector(perms)
    bonds=((0,1,1.),(1,2,.3),(0,3,-.4),(1,0,.2))
    raw=lambda p,x:p[jnp.dot(x,codes)]**2+1j*jnp.sin(p[jnp.dot(x,codes)])
    plan,evaluate,_=make_projected_heisenberg_products(raw,projector,bonds,chunk_size=3)
    p=jnp.arange(16,dtype=jnp.float64)/20;p=p.at[3].set(0).at[12].set(0)
    dp=jnp.cos(jnp.arange(16,dtype=jnp.float64));dp=dp.at[3].set(0).at[12].set(0)
    result=make_projected_product_jvp(evaluate)(p,dp,jax.jit(plan)(basis))
    hphi,phi,count,padded,hdphi,dphi=map(np.asarray,result)
    indices=basis[:,np.asarray(perms)]@np.asarray(codes)
    expected=np.mean(np.asarray(p)[indices]**2+1j*np.sin(np.asarray(p)[indices]),axis=1)
    expected_d=np.mean((2*np.asarray(p)[indices]+1j*np.cos(np.asarray(p)[indices]))*np.asarray(dp)[indices],axis=1)
    h=SzHeisenbergGraph(4,bonds,n_up=2).to_dense()
    for actual,wanted in ((phi,expected),(hphi,h@expected),(dphi,expected_d),(hdphi,h@expected_d)):
        np.testing.assert_allclose(actual,wanted,atol=2e-14,rtol=0)
    assert phi[0]==0 and dphi[0]==0 and abs(hdphi[0])>1e-4
    assert int(count)>0 and int(padded)>=int(count)
    eps=1e-6
    plus=jax.jit(evaluate)(p+eps*dp,jax.jit(plan)(basis))
    minus=jax.jit(evaluate)(p-eps*dp,jax.jit(plan)(basis))
    np.testing.assert_allclose((np.asarray(plus[0])-minus[0])/(2*eps),hdphi,atol=2e-10,rtol=2e-9)


def test_product_tangent_is_forward_AD_not_a_tangent_bank_wavefunction():
    # Nonlinear bank dependence: its derivative is not action(delta_bank).
    evaluate=lambda b,plan:(b['x']**2,b['x']**3,jnp.array(1),jnp.array(1))
    actual=make_projected_product_jvp(evaluate)({'x':jnp.array(2.)},{'x':jnp.array(3.)},None)
    assert actual[4]==12 and actual[5]==36
