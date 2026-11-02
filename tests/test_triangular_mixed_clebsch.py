"""Actual physical-state, SU2 Ward and full-A1 tests at N36."""
import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,
    make_mixed_clebsch_amplitude,make_mixed_projected_amplitude,
)
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group

jax.config.update('jax_enable_x64',True)


def test_three_parameterizations_start_at_same_singlet_and_H_action():
    c=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    family=init_mixed_clebsch_family(jax.random.PRNGKey(281),c)
    rng=np.random.default_rng(282)
    x=np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(6)],np.int32)
    raw=make_mixed_clebsch_amplitude(c)
    evaluate=jax.jit(jax.vmap(raw,in_axes=(None,0)))
    banks={mode:mixed_clebsch_bank(p,c,mode) for mode,p in family.items()}
    reference=np.asarray(evaluate(banks['ternary_c3'],x))
    for bank in banks.values():
        np.testing.assert_allclose(evaluate(bank,x),reference,atol=2e-17,rtol=2e-12)
        np.testing.assert_allclose(evaluate(bank,1-x),reference,atol=2e-17,rtol=2e-12)
    # S+psi vanishes on M=1 bras, with every allowed lowering column retained.
    bra=rng.permutation([0]*17+[1]*19)
    rows=[]
    for i in np.flatnonzero(bra):
        y=bra.copy();y[i]=0;rows.append(y)
    values=np.asarray(evaluate(banks['ternary_c3'],np.asarray(rows)))
    assert abs(values.sum())/np.linalg.norm(values)<3e-12
    bonds=tuple(triangular_j1j2_bonds(6,6,j2=.125,periodic=True))
    h=make_amplitude_heisenberg_action(raw,bonds,n_sites=36)
    _,actual=h(banks['ternary_c3'],jnp.asarray(x[:1]))
    graph=SzHeisenbergGraph(36,bonds,n_up=18)
    states,elements=graph.get_conn(tuple(x[0]))
    expected=np.dot(elements,np.asarray(evaluate(banks['ternary_c3'],np.asarray(states))))
    np.testing.assert_allclose(actual[0],expected,atol=1e-15,rtol=2e-12)


def test_complete_A1_is_coherent_and_full_parameter_gradient_has_correct_slope():
    c=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    family=init_mixed_clebsch_family(jax.random.PRNGKey(291),c)
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
    amp=make_mixed_projected_amplitude(c,images,chunk=4)
    p=family['ternary_c3']
    rng=np.random.default_rng(292)
    x=jnp.asarray(rng.permutation([0]*18+[1]*18))
    bank=mixed_clebsch_bank(p,c,'ternary_c3')
    evaluate=jax.jit(amp)
    a=evaluate(bank,x)
    b=evaluate(bank,x[images[37]])
    np.testing.assert_allclose(a,b,atol=1e-17,rtol=3e-12)
    # Use a division-free scalar to avoid conditioning this check on a node.
    loss=lambda z:amp(mixed_clebsch_bank(z,c,'ternary_c3'),x)
    direction=jax.tree.map(lambda z:jnp.ones_like(z)*.002,p)
    derivative=jax.jit(lambda z,v:jax.jvp(loss,(z,),(v,))[1])(p,direction)
    plus=jax.tree.map(lambda z,v:z+1e-4*v,p,direction)
    minus=jax.tree.map(lambda z,v:z-1e-4*v,p,direction)
    finite=(jax.jit(loss)(plus)-jax.jit(loss)(minus))/(2e-4)
    np.testing.assert_allclose(derivative,finite,atol=1e-11,rtol=2e-6)
