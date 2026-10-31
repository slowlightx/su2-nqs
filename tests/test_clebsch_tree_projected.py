import jax
import jax.numpy as jnp
import numpy as np
import pytest
from su2_nqs.models.literature_clebsch_tree import LiteratureClebschTreeConfig,init_literature_clebsch_tree
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from projects.clebsch_tree_nqs.workflows.spin import _groups
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.lattices.j1j2 import square_j1j2_bonds

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('rows,columns',[(2,2),(2,4),(4,4)])
def test_fold_is_same_projected_state_and_tangent(rows,columns):
    tree,order=square_geometry_tree(rows,columns)
    cfg=LiteratureClebschTreeConfig(tree,multiplicity_cap=4,maximum_q=8)
    p,_=init_literature_clebsch_tree(jax.random.PRNGKey(8),cfg)
    group=_groups('square',rows,columns)['gamma_a1']
    original,_=make_projected_clebsch(cfg,order,group,orbit_chunk=2,fold_root=False)
    folded,report=make_projected_clebsch(cfg,order,group,orbit_chunk=2)
    assert report['evaluation_images']*2 == report['full_group_order']
    basis=np.asarray(fixed_m_spin_basis(rows*columns,0)); x=basis if len(basis)<100 else basis[::123]
    a=np.asarray(jax.jit(original)(p,x)); b=np.asarray(jax.jit(folded)(p,x))
    np.testing.assert_allclose(a,b,rtol=3e-12,atol=2e-13)
    d=jax.tree.map(jnp.ones_like,p)
    ja=jax.jvp(lambda z:original(z,jnp.asarray(x[:2])),(p,),(d,))[1]
    jb=jax.jvp(lambda z:folded(z,jnp.asarray(x[:2])),(p,),(d,))[1]
    np.testing.assert_allclose(ja,jb,rtol=3e-12,atol=2e-13)
    if rows*columns == 8:
        bonds=square_j1j2_bonds(rows,columns,j2=.5,periodic=True)
        f=make_amplitude_heisenberg_action(lambda z,row:original(z,row[None])[0],bonds,n_sites=8)
        g=make_amplitude_heisenberg_action(lambda z,row:folded(z,row[None])[0],bonds,n_sites=8)
        for aa,bb in zip(f(p,x[:2]),g(p,x[:2]),strict=True):np.testing.assert_allclose(aa,bb,rtol=3e-12,atol=2e-13)
