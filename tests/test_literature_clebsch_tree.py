import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, clebsch_layers, make_clebsch_amplitudes,
    init_literature_clebsch_tree,
)
from su2_nqs.models.spin_clebsch_tree import SpinClebschTreeConfig, spin_clebsch_tree_amplitudes
from su2_nqs.su2.fusion_tree import FusionTree, Leaf
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix

jax.config.update('jax_enable_x64',True)


def test_reachable_widths_and_real_parameter_budget():
    cfg = LiteratureClebschTreeConfig(FusionTree.balanced(16),multiplicity_cap=8)
    layers = clebsch_layers(cfg)
    assert layers[0].widths[:3] == (1,0,1)
    assert layers[1].widths[:5] == (2,0,3,0,1)
    assert layers[2].widths[:9] == (8,0,8,0,8,0,7,0,1)
    p,r = init_literature_clebsch_tree(jax.random.PRNGKey(1),cfg)
    assert r['parameter_count'] == sum(np.size(a) for a in jax.tree.leaves(p))
    large = LiteratureClebschTreeConfig(FusionTree.balanced(64),multiplicity_cap=16,high_spin_cap=8)
    assert all(l.cg.nbytes < 8_000_000 for l in clebsch_layers(large))
    assert clebsch_layers(large)[-2].widths[6] == 8


@pytest.mark.parametrize('n,q,m',[(4,0,0),(6,2,2),(8,0,0)])
def test_exact_spin_and_independent_existing_contractor(n,q,m):
    cfg = LiteratureClebschTreeConfig(FusionTree.balanced(n),target_q=q,target_m2=m,
                                     multiplicity_cap=3,maximum_q=8)
    p,_ = init_literature_clebsch_tree(jax.random.PRNGKey(11),cfg)
    states = fixed_m_spin_basis(n,m)
    amp = np.asarray(make_clebsch_amplitudes(cfg)(p,jnp.asarray(states)))
    s2,ints = spin_squared_matrix(n,n_up=(n+m)//2)
    lookup = {b:i for i,b in enumerate(ints)}
    order = [lookup[int(''.join(map(str,row)),2)] for row in states]
    error = np.linalg.norm(s2[np.ix_(order,order)]@amp-q*(q+2)/4*amp)/np.linalg.norm(amp)
    assert error < 2e-12
    # Independent old dense-CG engine, with zero-padded IDENTICAL W matrices.
    merges=[]
    for bank,l in zip(p['merges'],clebsch_layers(cfg)):
        offset=0; padded=np.zeros((len(l.keys),3,1 if isinstance(l.nodes[0].left,Leaf) else 3,
                                                      1 if isinstance(l.nodes[0].right,Leaf) else 3))
        for k,shape in enumerate(l.shapes):
            size=int(np.prod(shape)); padded[k,:shape[0],:shape[1],:shape[2]]=np.asarray(bank[offset:offset+size]).reshape(shape)
            offset+=size
        merges.append(jnp.asarray(padded))
    readout=np.zeros(3); readout[:len(p['readout'])]=np.asarray(p['readout'])
    old=SpinClebschTreeConfig(cfg.tree,target_q=q,target_m2=m,feature_dim=3,maximum_intermediate_q=8)
    # Both compilers order equal-size layer banks and channels consistently.
    oldamp=spin_clebsch_tree_amplitudes(dict(merges=tuple(merges),readout=jnp.asarray(readout)),jnp.asarray(states),old)
    np.testing.assert_allclose(amp,oldamp,atol=2e-13,rtol=2e-13)


def test_positive_standardization_preserves_ray_and_gradients():
    cfg=LiteratureClebschTreeConfig(FusionTree.balanced(8),multiplicity_cap=4)
    x=np.asarray(fixed_m_spin_basis(8,0)); key=jax.random.PRNGKey(3)
    raw,_=init_literature_clebsch_tree(key,cfg)
    p,r=init_literature_clebsch_tree(key,cfg,x)
    f=jax.jit(make_clebsch_amplitudes(cfg)); a=np.asarray(f(raw,x)); b=np.asarray(f(p,x))
    np.testing.assert_allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=3e-13)
    assert max(abs(l['std_after']-1) for l in r['layers']) < 1e-12
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(jax.grad(lambda z:jnp.sum(f(z,x)**2))(p)))
    bad=x[0].copy(); bad[0]=2
    assert float(f(p,bad[None])[0]) == 0


def test_invalid_cap_fails():
    with pytest.raises(ValueError):
        LiteratureClebschTreeConfig(FusionTree.balanced(4),multiplicity_cap=0)
