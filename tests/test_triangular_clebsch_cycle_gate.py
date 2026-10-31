"""Fused physical-cycle decoder vs three independent whole-tree amplitudes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_clebsch_cycle_gate import make_cycle_gated_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude,prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank

jax.config.update('jax_enable_x64',True)


@pytest.fixture(scope='module')
def cycle_system():
    cfg=MixedClebschConfig(6,9,3)
    params=init_mixed_clebsch_family(jax.random.PRNGKey(261027901),cfg)['ternary_free']
    bank=prepare_feature_lut_bank(mixed_clebsch_bank(params,cfg,'ternary_free'),cfg)
    tree=triangular_schur_hierarchy(6)
    sites=(tree.blocks[0].sites[3],tree.blocks[0].sites[5],tree.blocks[1].sites[4])
    edges=((sites[0],sites[1]),(sites[1],sites[2]),(sites[0],tree.blocks[2].sites[0]),
           (tree.blocks[2].sites[1],tree.blocks[3].sites[2]),(sites[1],sites[2]))
    query,_=make_cycle_gated_exchange_queries(cfg,edges,sites)
    raw=make_feature_lut_amplitude(cfg)
    r=np.arange(36);r[list(sites)]=np.asarray(sites)[[1,2,0]]
    inv=np.argsort(r)
    def reference(bank,bits):
        theta=bank['cycle_theta'];a=jnp.sin(jnp.sqrt(3.)*theta)/jnp.sqrt(3.)
        b=(1-jnp.cos(jnp.sqrt(3.)*theta))/3
        return (1-2*b)*raw(bank,bits)+(a+b)*raw(bank,bits[r])+(b-a)*raw(bank,bits[inv])
    rng=np.random.default_rng(261027902)
    rows=np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(3)],np.int32)
    return cfg,bank,sites,edges,jax.jit(query),reference,rows


def _relative(actual,expected):
    a,b=np.asarray(actual),np.asarray(expected)
    scale=max(np.max(abs(a),initial=0),np.max(abs(b),initial=0),1e-280)
    assert np.max(abs(a-b))/scale<3e-12


@pytest.mark.parametrize('theta',[0.,-.06,.06])
def test_fused_raw_and_cached_exchanges_match_full_branch_reference(cycle_system,theta):
    _,original,_,edges,query,reference,rows=cycle_system
    bank=dict(original,cycle_theta=jnp.asarray(theta))
    evaluate=jax.jit(jax.vmap(reference,in_axes=(None,0)))
    for row in rows:
        exchanged=np.repeat(row[None],len(edges),axis=0)
        for k,(i,j) in enumerate(edges):exchanged[k,[i,j]]=row[[j,i]]
        expected=evaluate(bank,np.vstack((row,exchanged)))
        raw,swapped=query(bank,row)
        _relative(np.r_[raw,swapped],expected)
        np.testing.assert_array_equal(swapped[1],swapped[-1])
    illegal=rows[0].copy();illegal[0]=2
    raw,swaps=query(bank,illegal)
    assert raw==0.;np.testing.assert_array_equal(swaps,0.)
    raw,swaps=query(bank,np.zeros(36,np.int32))
    assert raw==0.;np.testing.assert_array_equal(swaps,0.)


def test_fused_angle_and_original_W_directional_derivatives(cycle_system):
    _,original,_,_,query,reference,rows=cycle_system
    bank=dict(original,cycle_theta=jnp.asarray(.037))
    direction=jax.tree.map(lambda x:.001*jnp.cos(jnp.arange(x.size).reshape(x.shape)),bank)
    actual=jax.jit(lambda x,v:jax.jvp(lambda w:query(w,rows[0])[0],(x,),(v,))[1])(bank,direction)
    expected=jax.jit(lambda x,v:jax.jvp(lambda w:reference(w,rows[0]),(x,),(v,))[1])(bank,direction)
    _relative(actual,expected)
    theta_actual=jax.jit(jax.grad(lambda t:query(dict(original,cycle_theta=t),rows[0])[0]))(jnp.asarray(.037))
    theta_reference=jax.jit(jax.grad(lambda t:reference(dict(original,cycle_theta=t),rows[0])))(jnp.asarray(.037))
    _relative(theta_actual,theta_reference)


def test_cycle_requires_physical_cross_child_support():
    cfg=MixedClebschConfig(6,9,3);tree=triangular_schur_hierarchy(6)
    with pytest.raises(ValueError,match='cross exactly'):
        make_cycle_gated_exchange_queries(cfg,[(0,1)],tree.blocks[0].sites[:3])
    with pytest.raises(ValueError,match='cross exactly'):
        make_cycle_gated_exchange_queries(cfg,[(0,1)],(tree.blocks[0].sites[0],tree.blocks[1].sites[0],tree.blocks[2].sites[0]))


def test_generic_gate_metadata_does_not_claim_universal_family_or_A1_novelty():
    cfg=MixedClebschConfig(6,9,3);tree=triangular_schur_hierarchy(6)
    _,metadata=make_cycle_gated_exchange_queries(cfg,[(0,1)],
        (tree.blocks[0].sites[0],tree.blocks[0].sites[1],tree.blocks[1].sites[0]))
    assert metadata['may_extend_finite_compressed_family']
    assert not metadata['universal_family_extension_guaranteed']
    assert not metadata['full_A1_tangent_novelty_certified']
    assert 'changed_state_family' not in metadata
