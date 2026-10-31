"""Independent mixed-root proposal and split-budget checks, not training."""
from itertools import product

import numpy as np
import pytest

from su2_nqs.su2.fusion_tree import FusionTree,FusionNode,Leaf
from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree
from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_root_refinement import (
    cost_selected_fit_count,finite_chain_rhat,norm_weight_diagnostics,
    paired_audit_difference,terminal_union_leaf_diagonals,
)


def test_raw_root_identity_mixture_matches_explicit_four_spin_oracle():
    tree=FusionTree(FusionNode(Leaf(0),Leaf(1)))
    singlet=np.asarray([0,-1,1,0])/np.sqrt(2)
    triplet=np.asarray([1,1/np.sqrt(2),1/np.sqrt(2),1])
    features=({0:singlet[:,None],2:triplet[:,None]},)*2
    state=ReducedSingletTree(tree,({0:np.eye(1),2:np.eye(1)},)*2,
        ({(0,0,0):np.ones((1,1,1))/np.sqrt(2),(2,2,0):np.ones((1,1,1))/np.sqrt(2)},),np.ones(1))
    observed,metadata=terminal_union_leaf_diagonals(state,features)
    down=np.asarray([1,0,0,0]);zero=np.asarray([0,1,1,0])/np.sqrt(2);up=down[::-1]
    vectors=[np.kron(singlet,singlet),
             (np.kron(down,up)-np.kron(zero,zero)+np.kron(up,down))/np.sqrt(3)]
    density=sum(abs(v.reshape(4,4))**2 for v in vectors)/2
    np.testing.assert_allclose(observed[0],density.sum(axis=1),atol=2e-16)
    np.testing.assert_allclose(observed[1],density.sum(axis=0),atol=2e-16)
    np.testing.assert_allclose(observed,[np.asarray([1,2,2,1])/6]*2,atol=2e-16)
    assert metadata['root_coordinates']==2
    assert metadata['root_q_probability']=={0:.5,2:.5}
    assert not metadata['extra_magnetic_degeneracy_weight']


def test_mixed_root_proposal_rejects_noncanonical_children():
    tree=FusionTree(FusionNode(Leaf(0),Leaf(1)))
    features=({1:np.eye(2,1)},)*2
    state=ReducedSingletTree(tree,({1:np.asarray([[2.]])},)*2,
        ({(1,1,0):np.ones((1,1,1))},),np.ones(1))
    with pytest.raises(ValueError):
        terminal_union_leaf_diagonals(state,features)


def test_cost_only_budget_reserves_large_untouched_audit():
    assert cost_selected_fit_count(10,1700)==32
    assert cost_selected_fit_count(1,1000)==32
    assert cost_selected_fit_count(30,1700)==0
    with pytest.raises(ValueError):
        cost_selected_fit_count(np.nan,1700)


def test_union_norm_coverage_detects_parent_only_blind_direction():
    phi=np.asarray([[1.,0],[1,0],[1,0],[0,30]])
    measured=norm_weight_diagnostics(phi,np.asarray([1.,0]),np.zeros(4))
    assert measured['parent']['effective_contributions']==3
    assert measured['root_union']['effective_contributions']<1.01


def test_paired_group_differences_cancel_identical_endpoint_and_chain_drift_is_visible():
    rng=np.random.default_rng(4110)
    phi=rng.normal(size=(64,4));hp=rng.normal(size=(64,4));c=rng.normal(size=4)
    groups=np.tile(np.arange(8),8)
    result=paired_audit_difference(phi,hp,np.zeros(64),c,c,groups,n_sites=6)
    assert result['optimized_minus_parent']==dict(energy_per_site=0.,variance_per_site=0.)
    assert result['paired_group_jackknife_SE']==dict(energy_per_site=0.,variance_per_site=0.)
    clean=rng.normal(size=(8,8));drift=clean+np.arange(8)[None,:]*10
    assert finite_chain_rhat(drift)['split']>finite_chain_rhat(clean)['split']
