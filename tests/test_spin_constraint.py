"""Independent physical S2 ratios, penalty objective and purity-bound tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse, singlet_projector_action
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.operators.spin_squared_ratios import make_spin_squared_ratios
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_full_vit_constraint import prepare_constraint

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('n,m2',[(4,0),(6,0),(6,2),(6,-2),(5,1),(4,4),(4,-4)])
def test_packed_S2_matches_independent_sparse_action(n,m2):
    matrix,basis=spin_squared_sz_sparse(n,n_up=(n+m2)//2)
    bits=jnp.asarray(basis);powers=1 << np.arange(n-1,-1,-1)
    indices=np.asarray(basis)@powers
    rng=np.random.default_rng(750+n+m2)
    values=rng.normal(size=len(basis))+1j*rng.normal(size=len(basis))
    table=np.ones(2**n,dtype=complex);table[indices]=values
    log_fn=lambda p,x:jnp.log(p[jnp.dot(x,jnp.asarray(powers))])
    for chunk in (1,4,9):
        actual=make_spin_squared_ratios(log_fn,n_sites=n,target_m2=m2,connector_batch_size=chunk)(jnp.asarray(table),bits)
        np.testing.assert_allclose(actual,(matrix@values)/values,rtol=2e-13,atol=2e-13)


def test_packed_S2_singlet_and_sector_fail_closed():
    n=6;matrix,basis=spin_squared_sz_sparse(n)
    indices=np.asarray(basis)@(1 << np.arange(n-1,-1,-1))
    columns=coupled_basis_matrix(enumerate_paths(n,0),0,0)
    values=columns@np.random.default_rng(81).normal(size=5)
    fn=lambda p,x:jnp.log(p[jnp.dot(x,1<<jnp.arange(n-1,-1,-1))]+0j)
    support=np.asarray(basis)[abs(values[indices])>1e-8]
    local=make_spin_squared_ratios(fn,n_sites=n)
    np.testing.assert_allclose(local(jnp.asarray(values),jnp.asarray(support)),0,atol=2e-13)
    assert np.isnan(np.asarray(local(jnp.asarray(values),jnp.ones((1,n),dtype=int)))).all()
    with pytest.raises(ValueError): make_spin_squared_ratios(fn,n_sites=5,target_m2=0)
    with pytest.raises(TypeError): make_spin_squared_ratios(fn,n_sites=6,target_m2=.5)


@pytest.mark.parametrize('geometry',['triangular','square'])
def test_penalty_objective_gradient_and_purity_bound(geometry):
    data=_prepare_case(dict(geometry=geometry,rows=2,columns=2,j2_over_j1=.125,exact_energy_anchor=-100.),'fixed_m0')
    model=FixedSzSpatialViTConfig(geometry=geometry,rows=2,columns=2,patch_rows=1,patch_columns=1,embedding_dim=4,head_count=2)
    p=init_fixed_sz_spatial_vit(jax.random.PRNGKey(41),model)
    problem=prepare_constraint(data,model,1.)
    state=problem['raw']['state_fn']
    physical=problem['raw']['physical'](state(p))
    loss=lambda params:jnp.vdot(state(params),jnp.asarray(problem['objective'])@state(params)).real
    expected=np.vdot(physical,(data['hamiltonian']+data['spin_squared'])@physical).real
    assert float(loss(p))==pytest.approx(expected,abs=1e-12)
    t=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.01,p)
    exact=float(jax.jvp(loss,(p,),(t,))[1])
    shifted=[jax.tree_util.tree_map(lambda x,y:x+s*1e-4*y,p,t) for s in (-1,1)]
    np.testing.assert_allclose(exact,(loss(shifted[1])-loss(shifted[0]))/2e-4,atol=2e-8)
    projection=singlet_projector_action(physical,data['spin_squared'],maximum_spin=2)
    nonsinglet=1-np.vdot(projection,projection).real
    spin=np.vdot(physical,data['spin_squared']@physical).real
    assert 0 <= nonsinglet <= spin/2+1e-12
    with pytest.raises(ValueError): prepare_constraint(data,model,-1.)
