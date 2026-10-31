"""Independent complex dense-H/AD validation of the reused training products."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from su2_nqs.vmc.projected_product_pullback import make_projected_product_pullback
from su2_nqs.vmc.rayleigh_cotangents import energy_moment_cotangents
from su2_nqs.vmc.joint_moments import residual_moment_cotangents

jax.config.update('jax_enable_x64',True)


def test_joint_additive_K2_gradients_match_dense_AD_incomplete_Q_with_nodes():
    states=np.array([x for x in itertools.product((0,1),repeat=4) if sum(x)==2],np.int32)
    codes=jnp.array([8,4,2,1]);perms=(tuple(range(4)),(3,2,1,0))
    group=PhysicalSpatialProjector(perms);bonds=((0,1,1.),(1,2,.3),(0,3,-.4),(1,0,.2))
    h=jnp.asarray(SzHeisenbergGraph(4,bonds,n_up=2).to_dense())
    generator=lambda x:x*x+1j*jnp.sin(x)
    raw=lambda bank,x:bank[x@codes]**2
    plan,evaluate,_=make_projected_heisenberg_products(raw,group,bonds,chunk_size=3)
    pb=make_projected_product_pullback(raw,4,chunk_size=3)
    reference=jnp.arange(16,dtype=jnp.float64)/20
    params=reference+.03*jnp.cos(jnp.arange(16));bank,vjp=jax.vjp(generator,params)
    indices=jnp.asarray(states[:,np.asarray(perms)]@np.asarray(codes))
    phi=lambda x:jnp.mean(generator(x)[indices]**2,axis=1)
    c=jnp.array([1.,0.,-.3+.1j,.3,.2j,-.4]);k=c-.03*(h@c)+.002*(h@h@c)
    scale=.7
    def physical(x):return k+scale*(phi(x)-phi(reference))
    p=np.asarray(physical(params));hp=np.asarray(h@physical(params))
    subset=np.array([0,2,5]);q=np.array([.4,1.8,.7]);prepared=jax.jit(plan)(states[subset])
    weights=[energy_moment_cotangents(p[subset],hp[subset],q,4),residual_moment_cotangents(p[subset],hp[subset],q,4)]
    multiple=jax.jit(jax.vmap(pb,in_axes=(None,None,0,0)))
    _,grads,_,_=multiple(bank,prepared,scale*jnp.stack([w[0] for w in weights]),scale*jnp.stack([w[1] for w in weights]))
    observed=np.stack([np.asarray(vjp(jax.tree.map(lambda b:b[i],grads))[0]) for i in range(2)])
    def metrics(x):
        p=physical(x)[subset];hp=(h@physical(x))[subset];norm=jnp.sum(abs(p)**2/q)
        e=jnp.sum(p.conj()*hp/q)/norm
        r=jnp.sum(abs(hp-e*p)**2/q)/norm/4
        return jnp.array([e.real/4,r])
    expected=np.asarray(jax.jacrev(metrics)(params))
    np.testing.assert_allclose(observed,expected,atol=4e-14,rtol=4e-14)


def test_fixed_protocol_preserves_architecture_and_audit_budget():
    from projects.clebsch_tree_nqs.workflows.spin_krylov_residual_training import CONFIG,frozen
    spec=frozen.read(CONFIG)
    assert (spec['multiplicity_width'],spec['maximum_q'],spec['embedding_dim'],spec['depth'])==(2,4,16,2)
    assert spec['maximum_attempts_per_arm']==8 and spec['audit']['chains']*spec['audit']['records']==128
    assert sum(spec['stage_seconds'].values())==spec['wall_limit_total_seconds']==1800
    assert spec['audit_seed']!=spec['bootstrap_seed']
