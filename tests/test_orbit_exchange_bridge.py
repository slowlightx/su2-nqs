"""Independent physical-basis orbit probabilities and node-safe moments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('image_count',[1,4])
def test_orbit_bridge_normalization_nodes_and_full_H_moments(image_count):
    n=4; powers=np.array([8,4,2,1])
    rows=((np.arange(16)[:,None]>>np.arange(3,-1,-1))&1).astype(np.int32)
    images=np.array([np.roll(np.arange(4),k) for k in range(image_count)],np.int32)
    # Not invariant under these images: must evaluate H(Ppsi), not P(Hpsi).
    bonds=[(0,1,1.),(1,0,-.25),(1,2,-.5),(2,3,.75),(3,0,.4)]
    probabilities=np.abs([g for _,_,g in bonds]);probabilities/=probabilities.sum()
    def factory(edges):
        edges=jnp.asarray(edges); pp=jnp.asarray(powers)
        def query(table,bits):
            changed=jnp.broadcast_to(bits,(len(edges),n)); indices=jnp.arange(len(edges))
            changed=changed.at[indices,edges[:,0]].set(bits[edges[:,1]])
            changed=changed.at[indices,edges[:,1]].set(bits[edges[:,0]])
            return table[bits@pp],table[changed@pp]
        return query
    products,metadata=make_orbit_exchange_bridge_products(factory,images,bonds,probabilities,
        n_sites=n,projection_chunk=1)
    # Sparse complex raw states, with exact target nodes. Norms differ.
    tables=np.zeros((2,16),complex)
    tables[0,[3,5]]=[2+.3j,-.7j];tables[1,[3,6]]=[-.4+.8j,.9-.2j]
    dense=np.zeros((16,16))
    for old,row in enumerate(rows):
        for a,b,g in bonds:
            dense[old,old]+=g*(row[a]-.5)*(row[b]-.5)
            if row[a]!=row[b]:
                changed=row.copy();changed[a],changed[b]=changed[b],changed[a]
                dense[changed@powers,old]+=.5*g
    vals=[]
    for table in tables:
        actual=tuple(map(np.asarray,products(jnp.asarray(table),jnp.asarray(rows))))
        image_values=table[rows[:,images]@powers]
        psi=image_values.mean(axis=1);qo=(abs(image_values)**2).mean(axis=1)
        qp=np.zeros(16)
        # Forward push, independently of inverse-image production indexing.
        for old,row in enumerate(rows):
            for (a,b,_),probability in zip(bonds,probabilities,strict=True):
                changed=row.copy();changed[a],changed[b]=changed[b],changed[a]
                qp[changed@powers]+=probability*qo[old]
        for x,y in zip(actual,(psi,dense@psi,qo,qp),strict=True):
            np.testing.assert_allclose(x,y,atol=2e-15,rtol=2e-14)
        norm=np.vdot(table,table).real
        assert qo.sum()==pytest.approx(norm,abs=3e-15)
        assert qp.sum()==pytest.approx(norm,abs=3e-15)
        assert np.min(qo-abs(psi)**2)>=-2e-15
        vals.append((actual,norm))
    qmix=sum((v[2]+v[3])/norm for v,norm in vals)/4
    assert qmix.sum()==pytest.approx(1.,abs=3e-15)
    positive=qmix>0
    node_energy_seen=False
    for (psi,hp,qo,qp),rawnorm in vals:
        np.testing.assert_allclose(hp[~positive],0.,atol=0,rtol=0)
        assert np.max(abs(psi[positive])**2/(rawnorm*qmix[positive]))<=4+2e-14
        z=np.sum(qmix[positive]*abs(psi[positive])**2/qmix[positive])
        h=np.sum(qmix[positive]*psi[positive].conj()*hp[positive]/qmix[positive])
        h2=np.sum(qmix[positive]*abs(hp[positive])**2/qmix[positive])
        expected=np.vdot(psi,dense@psi)/np.vdot(psi,psi)
        assert h/z==pytest.approx(expected,abs=3e-15)
        assert h2/z-abs(h/z)**2==pytest.approx(np.linalg.norm(hp-expected*psi)**2/z,abs=4e-15)
        node_energy_seen |= bool(np.any((psi==0)&(abs(hp)>1e-8)))
    assert node_energy_seen
    assert metadata['guide_square_precedes_image_mean']
    assert metadata['divide_guide_by_known_raw_norm']
