"""Exact mathematical checks for AFM warm parameters and coherent A1 Born law."""
from itertools import product
from dataclasses import replace

import jax
import numpy as np
import pytest

from su2_nqs.models.many_cell_schur_vit import ManyCellSchurViTConfig,init_many_cell_schur_vit,many_cell_schur_log_amplitudes
from su2_nqs.models.schur_initialization import afm_child_casimir_initialization, deepen_many_cell_schur_vit
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.recoupling import tree_basis_matrix
from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.vmc.exact_reduced_projector import exact_projected_reduced_state


def test_afm_pair_gap_bias_has_exact_boltzmann_sign_and_complete_support():
    config=ManyCellSchurViTConfig(rectangular_four_spin_layout(2,2),embedding_dim=16,head_count=2,depth=1)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(4),config)
    warmed=afm_child_casimir_initialization(params,2.)
    labels=np.asarray([[0,0,0],[2,2,0]],dtype=np.int32)
    old=abs(np.exp(np.asarray(many_cell_schur_log_amplitudes(params,labels,config))))**2
    new=abs(np.exp(np.asarray(many_cell_schur_log_amplitudes(warmed,labels,config))))**2
    np.testing.assert_allclose((new[1]/new[0])/(old[1]/old[0]),np.exp(-4.),atol=2e-14,rtol=2e-13)
    np.testing.assert_allclose(new.sum(),1.,atol=2e-14)
    assert np.all(new>0)
    assert new@[-1.5,.5] < old@[-1.5,.5]
    for key in params:
        if key!='logit_feature':
            for a,b in zip(jax.tree_util.tree_leaves(params[key]),jax.tree_util.tree_leaves(warmed[key]),strict=True):
                np.testing.assert_array_equal(a,b)
    for beta in (0.,-1.,float('nan')):
        with pytest.raises(ValueError): afm_child_casimir_initialization(params,beta)


def test_vectorized_fixed_CG_isometry_matches_complete_independent_recursion():
    layout=rectangular_four_spin_layout(4,2)
    labels=np.asarray([row.labels for row in layout.tree.enumerate_states(0)],dtype=np.int32)
    bits=np.asarray(list(product((0,1),repeat=8)),dtype=np.int32)
    order=np.asarray([3,2,1,0,7,6,5,4])
    actual=tree_overlaps_on_bits(layout.tree,labels,bits,physical_leaf_order=order)
    reference,_=tree_basis_matrix(layout.tree,0,0)
    row_index=bits[:,order]@2**np.arange(7,-1,-1)
    np.testing.assert_allclose(actual,reference[row_index],atol=2e-14,rtol=0)
    np.testing.assert_allclose(actual.T@actual,np.eye(14),atol=2e-14,rtol=0)
    with pytest.raises(ValueError,match='entry cap'):
        tree_overlaps_on_bits(layout.tree,labels,bits,maximum_entries=1)
    with pytest.raises(ValueError,match='permutation'):
        tree_overlaps_on_bits(layout.tree,labels,bits,physical_leaf_order=[False,True,2,3,4,5,6,7])
    bad=labels.copy();bad[0,-1]=2
    with pytest.raises(ValueError,match='root'):
        tree_overlaps_on_bits(layout.tree,bad,bits)


def test_exact_small_coherent_A1_sampler_matches_physical_projection_not_native_law():
    layout=rectangular_four_spin_layout(4,2)
    labels=np.asarray([row.labels for row in layout.tree.enumerate_states(0)],dtype=np.int32)
    bits=np.asarray(list(product((0,1),repeat=8)),dtype=np.int32)
    u=tree_overlaps_on_bits(layout.tree,labels,bits)
    rng=np.random.default_rng(18)
    c=rng.normal(size=14)+1j*rng.normal(size=14);c/=np.linalg.norm(c)
    group=rectangular_gamma_a1_group(2,4)
    # Physical model-leaf order for this layout is not row-major; use its
    # site-order pullback consistently when forming the reference matrix.
    u=tree_overlaps_on_bits(layout.tree,labels,bits,physical_leaf_order=layout.site_order)
    projected=exact_projected_reduced_state(labels,c,u,bits,group)
    position={tuple(row):i for i,row in enumerate(bits)}
    physical=u@c
    expected=sum(np.asarray([physical[position[tuple(row[np.asarray(image)])]] for row in bits]) for image in group)/len(group)
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(u@projected.coefficients,expected,atol=3e-14,rtol=0)
    assert np.max(abs(projected.probabilities-abs(c)**2))>.01
    draws=np.asarray(projected.sample(jax.random.PRNGKey(21),8192))
    lookup={tuple(row):i for i,row in enumerate(labels)}
    p=np.bincount([lookup[tuple(row)] for row in draws],minlength=14)/len(draws)
    assert np.max(abs(p-projected.probabilities))<.04
    replay=np.asarray(projected.sample(jax.random.PRNGKey(21),8192))
    np.testing.assert_array_equal(draws,replay)
    for tolerance in (0.,-1.,float('nan')):
        with pytest.raises(ValueError,match='tolerance'):
            exact_projected_reduced_state(labels,c,u,bits,group,tolerance=tolerance)
    with pytest.raises(TypeError,match='integer'):
        exact_projected_reduced_state(labels,c,u,bits,[tuple(float(site) for site in group[0])])


def test_depth_growth_preserves_complete_singlet_wavefunction_and_activates_new_blocks():
    source = ManyCellSchurViTConfig(rectangular_four_spin_layout(4,2),
        embedding_dim=16,head_count=2,depth=1,direct_phase_initialization_scale=.1)
    target = replace(source,depth=4)
    params = afm_child_casimir_initialization(init_many_cell_schur_vit(jax.random.PRNGKey(10),source),2.)
    grown = deepen_many_cell_schur_vit(params,source,target,jax.random.PRNGKey(11))
    labels = np.asarray([row.labels for row in source.layout.tree.enumerate_states(0)],dtype=np.int32)
    old = np.exp(np.asarray(many_cell_schur_log_amplitudes(params,labels,source)))
    new = np.exp(np.asarray(many_cell_schur_log_amplitudes(grown,labels,target)))
    np.testing.assert_allclose(new,old,atol=2e-14,rtol=0.)
    np.testing.assert_allclose(np.vdot(new,new),1.,atol=2e-14)
    assert sum(x.size for x in jax.tree_util.tree_leaves(grown)) == 17322
    gradient = jax.grad(lambda p: many_cell_schur_log_amplitudes(p,labels[:1],target)[0].real)(grown)
    assert np.linalg.norm(gradient['layers'][-1]['output']) > 1e-12
    phase_gradient = jax.grad(lambda p: many_cell_schur_log_amplitudes(p,labels[:1],target)[0].imag)(grown)
    assert np.linalg.norm(phase_gradient['global_layers'][-1]['output']) > 1e-12
    # Independent physical-basis isometry: same normalized S=0 state, not
    # merely an agreement between two samplers' frequency histograms.
    u,_ = tree_basis_matrix(source.layout.tree,0,0)
    np.testing.assert_allclose(u@new,u@old,atol=2e-14,rtol=0.)
    for bad in (source,replace(target,head_count=4)):
        with pytest.raises(ValueError):
            deepen_many_cell_schur_vit(params,source,bad,jax.random.PRNGKey(11))
