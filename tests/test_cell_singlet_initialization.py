"""N4/N8 algebra and physical replay; no N16 refinement or large ED."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.frontier.cell_singlet_initialization import (
    local_singlet_reference, cell_product_parameters, perturb_cell_product,
    product_amplitudes, singlet_bond_diagnostics,
)
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config, amplitude_functions
from projects.clebsch_tree_nqs.workflows.spin_vap_decoder import init_decoder
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios


@pytest.mark.parametrize('diagonals',[(.5,.5),(1.,.125)])
def test_local_reference_and_N8_product_have_exact_normalization_and_independent_amplitudes(diagonals):
    bonds=((0,1,1.),(0,2,1.),(1,3,1.),(2,3,1.),(0,3,diagonals[0]),(1,2,diagonals[1]))
    ref=local_singlet_reference(bonds)
    assert ref['eigen_residual']<2e-14
    assert abs(ref['energy']-(-2+sum(diagonals)/4))<2e-14
    config=physical_config(2,4,2)
    params=cell_product_parameters(config,ref['coefficients'])
    bits=((np.arange(256)[:,None]>>np.arange(7,-1,-1))&1).astype(np.int32)
    fn=amplitude_functions(config,(tuple(range(8)),))['raw']
    actual=np.asarray(jax.jit(jax.vmap(fn,in_axes=(None,0)))(params,jnp.asarray(bits)))
    expected=product_amplitudes(bits,config.cells,ref['physical'])
    np.testing.assert_allclose(actual,expected,atol=2e-15)
    assert abs(np.vdot(actual,actual)-1)<3e-14
    assert jax.tree_util.tree_structure(params)==jax.tree_util.tree_structure(init_decoder(1,config))
    with pytest.raises(ValueError):cell_product_parameters(config,np.ones(2))


def test_seeded_full_q_product_preserves_SU2_A1_H_and_bond_energy_identities():
    data=_prepare_case(dict(geometry='square',rows=2,columns=4,j2_over_j1=.5,exact_energy_anchor=0.),'fixed_m0')
    config=physical_config(2,4,2)
    core=cell_product_parameters(config,np.array([1.,0.]))
    params=perturb_cell_product(core,seed=2026093019,scale=.05)
    repeat=perturb_cell_product(core,seed=2026093019,scale=.05)
    for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(repeat),strict=True):
        np.testing.assert_array_equal(a,b)
        assert np.all(np.asarray(a)!=0)
    funcs=amplitude_functions(config,data['groups']['gamma_a1']);basis=jnp.asarray(data['basis'])
    raw=np.asarray(jax.jit(jax.vmap(funcs['raw'],in_axes=(None,0)))(params,basis))
    fn=jax.jit(jax.vmap(funcs['a1'],in_axes=(None,0)))
    vector=np.asarray(fn(params,basis));norm=np.linalg.norm(vector)
    np.testing.assert_allclose(vector,np.mean(raw[data['tables']['gamma_a1']],axis=0),atol=2e-14)
    assert np.linalg.norm(data['spin_squared']@vector)/norm<3e-13
    assert np.min(abs(vector))>1e-12
    ratios=np.asarray(make_full_heisenberg_ratios(lambda p,x:jnp.log(funcs['a1'](p,x)),data['bonds'])(params,basis))
    expected=data['hamiltonian']@vector/vector
    np.testing.assert_allclose(ratios,expected,rtol=3e-12,atol=3e-12)
    probabilities=abs(vector/norm)**2
    diagonal=singlet_bond_diagnostics(np.asarray(basis),data['bonds'],config.cells)
    np.testing.assert_allclose(probabilities@diagonal['total'],probabilities@ratios/8,atol=2e-14)
    # Generic tangent is also a singlet; no freezing or physical-index learning.
    direction=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.01,params)
    tangent=np.asarray(jax.jvp(lambda p:fn(p,basis),(params,),(direction,))[1])
    assert np.linalg.norm(data['spin_squared']@tangent)/np.linalg.norm(tangent)<3e-13
    with pytest.raises(ValueError):perturb_cell_product(core,seed=1,scale=-1)
