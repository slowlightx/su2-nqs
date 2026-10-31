"""Small independent conventions for the bounded reduced-ViT cost probe."""
import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_viability import (
    model_config, permutation_swaps, probability_force, swap_column,
    physical_reflection, neural_bonds, neural_image,
    _plain,
)
from su2_nqs.models.many_cell_schur_vit import init_many_cell_schur_vit, many_cell_schur_log_amplitudes
from su2_nqs.su2.recoupling import tree_basis_matrix


SETTINGS=dict(lx=6,ly=6,embedding_dim=8,head_count=2,depth=1,
    direct_phase_initialization_scale=.1)


def test_ordered_cycle_swaps_implement_old_site_to_new_site_image():
    for image in ((1,2,3,0),(2,0,3,1),(1,2,0,4,5,3)):
        sites=list(range(len(image)))
        for first,second in permutation_swaps(image):
            sites[first],sites[second]=sites[second],sites[first]
        actual=tuple(sites.index(old) for old in range(len(image)))
        assert actual==image


def test_physical_swap_column_is_exact_in_fixed_singlet_basis():
    config=model_config(SETTINGS,small=True)
    states=config.layout.tree.enumerate_states(0)
    labels=np.asarray([s.labels for s in states])
    u,_=tree_basis_matrix(config.layout.tree,0,0)
    physical=np.zeros((16,16))
    for old in range(16):
        bits=list(f"{old:04b}")
        bits[1],bits[3]=bits[3],bits[1]
        physical[int(''.join(bits),2),old]=1
    reduced=np.zeros((2,2))
    lookup={tuple(row):index for index,row in enumerate(labels)}
    for column,row in enumerate(labels):
        for target,value in swap_column(config,row,(1,3)).items():
            reduced[lookup[target],column]=value
    np.testing.assert_allclose(reduced,u.conj().T@physical@u,atol=2e-13)


def test_both_sided_covariance_force_equals_exact_rayleigh_gradient():
    config=model_config(SETTINGS,small=True)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(12),config)
    labels=np.asarray([s.labels for s in config.layout.tree.enumerate_states(0)])
    h=jnp.asarray([[.8,-.4],[-.4,-.7]])
    def energy(p):
        psi=jnp.exp(many_cell_schur_log_amplitudes(p,labels,config))
        return jnp.real(jnp.vdot(psi,h@psi)/jnp.vdot(psi,psi))
    psi=jnp.exp(many_cell_schur_log_amplitudes(params,labels,config))
    cov=probability_force(params,labels,h@psi/psi,config,weights=abs(psi)**2)
    exact=jax.grad(energy)(params)
    for a,b in zip(jax.tree_util.tree_leaves(cov),jax.tree_util.tree_leaves(exact),strict=True):
        np.testing.assert_allclose(a,b,atol=3e-12,rtol=3e-12)


def test_physical_reflection_and_spatial_conjugation_are_exact_unitary_relabellings():
    image=physical_reflection(2,4)
    assert image==(0,3,2,1,4,7,6,5)
    assert tuple(image[image[i]] for i in range(8))==tuple(range(8))
    physical=tuple((i//4)*4+(i%4+1)%4 for i in range(8))
    translated=neural_image(physical,2,4)
    assert translated==tuple((i//4)*4+(i%4-1)%4 for i in range(8))
    assert neural_bonds(((0,3,.2),(1,4,1.)),2,4)==((0,1,.2),(3,4,1.))


def test_complex_array_artifacts_are_json_serializable_without_discarding_phase():
    import json
    value=np.asarray([1+2j,-3-.5j],dtype=np.complex128)
    parsed=json.loads(json.dumps(_plain({"local_energy":value})))
    assert parsed["local_energy"]==[{"real":1.,"imag":2.},{"real":-3.,"imag":-.5}]
