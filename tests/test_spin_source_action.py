"""Dense physical-space oracle for source/H products, phases and exact nodes."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
from su2_nqs.vmc.spin_source_action import make_spin_source_products


def test_shared_source_actions_equal_dense_H_including_imperfect_parent_and_nodes():
    n=4; bits=(np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1
    rng=np.random.default_rng(29101)
    psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n);psi[3]=0.
    psi/=np.linalg.norm(psi)
    images=np.asarray([[0,1,2,3],[1,2,3,0]],np.int32)
    bonds=((0,1,1.),(1,0,.25),(1,2,.7),(2,3,-.2),(3,0,.8))
    prob=np.ones(len(bonds))/len(bonds)
    phase=np.asarray([[1,1j,-1,-1j],[1,-1,1,-1]],complex)
    def factory(edges):
        ed=jnp.asarray(edges);powers=jnp.asarray(1<<np.arange(n-1,-1,-1))
        def query(bank,row):
            def changed(edge):
                a,b=edge; y=row.at[a].set(row[b]).at[b].set(row[a])
                return bank[jnp.dot(y,powers)]
            return bank[jnp.dot(row,powers)],jax.vmap(changed)(ed)
        return query
    action,meta=make_spin_source_products(factory,images,bonds,prob,phase,n_sites=n,projection_chunk=2)
    actual=tuple(np.asarray(x) for x in action(jnp.asarray(psi),jnp.asarray(bits)))
    unsigned=tuple(np.asarray(x) for x in action(jnp.asarray(psi),jnp.asarray(bits,dtype=jnp.uint8)))
    for a,b in zip(actual,unsigned,strict=True):
        np.testing.assert_array_equal(a,b)
    coherent=np.mean(np.stack([psi[bits[:,p]@(1<<np.arange(n-1,-1,-1))] for p in images]),axis=0)
    diagonal=(bits-.5)@phase.T/np.sqrt(n)
    target=np.column_stack((coherent,diagonal*coherent[:,None]))
    h=sum(c*spin_dot_matrix(n,a,b) for a,b,c in bonds)
    np.testing.assert_allclose(actual[0],target,rtol=1e-13,atol=3e-16)
    np.testing.assert_allclose(actual[1],h@target,rtol=1e-13,atol=4e-16)
    rawpower=np.mean([abs(psi[bits[:,p]@(1<<np.arange(n-1,-1,-1))])**2 for p in images],axis=0)
    pushed=np.zeros(2**n)
    for (a,b,_),p in zip(bonds,prob):
        changed=bits.copy();changed[:,[a,b]]=changed[:,[b,a]]
        pushed+=p*rawpower[changed@(1<<np.arange(n-1,-1,-1))]
    np.testing.assert_allclose(actual[2],rawpower,atol=2e-16)
    np.testing.assert_allclose(actual[3],pushed,atol=2e-16)
    assert meta['both_commutator_and_carrier_residual']


def test_rank_one_source_from_singlet_is_exact_triplet_not_merely_fixed_Sz():
    n=4;bits=(np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1
    # Two explicit dimers and their coherent rearrangement: not an H eigenstate.
    def covering(pairs):
        v=np.ones(2**n)
        for a,b in pairs:
            v*=np.where(bits[:,a]!=bits[:,b],(bits[:,a]-bits[:,b])/np.sqrt(2),0.)
        return v
    psi=covering(((0,1),(2,3)))+.37*covering(((0,2),(1,3)))
    psi/=np.linalg.norm(psi)
    phase=np.exp(2j*np.pi*np.arange(n)/n)
    target=((bits-.5)@phase)*psi
    s2=.75*n*np.eye(2**n)+2*sum(spin_dot_matrix(n,a,b) for a in range(n) for b in range(a+1,n))
    assert np.linalg.norm(target)>0.
    np.testing.assert_allclose(s2@psi,0.,atol=8e-16)
    np.testing.assert_allclose(s2@target,2*target,atol=1e-15)
    # A uniform source annihilates every m2=0 singlet, not a soft triplet.
    np.testing.assert_array_equal((bits-.5).sum(1)*psi,0.)


def test_invalid_source_and_bridge_are_not_repaired():
    factory=lambda e:None;images=np.arange(4)[None,:]
    for phase in (np.ones(4),np.full((1,4),np.nan)):
        with pytest.raises(ValueError):
            make_spin_source_products(factory,images,((0,1,1.),),(1.,),phase,n_sites=4,projection_chunk=1)
    with pytest.raises(ValueError):
        make_spin_source_products(factory,images,((0,1,1.),(1,2,1.)),(1.,0.),np.ones((1,4)),n_sites=4,projection_chunk=1)
