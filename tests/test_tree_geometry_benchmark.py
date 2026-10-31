import numpy as np
from math import log2

from su2_nqs.lattices.geometry_optimization import (
    TreeObjectiveWeights,
    leaf_depths,
    optimize_fusion_tree,
    tree_objective,
    tree_objective_components,
    weighted_tree_distance,
)
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.operators.tree_plan import build_tree_heisenberg_plan
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse
from su2_nqs.su2.tree_trainability import tree_trainability_components


def test_geometry_optimizer_monotonically_reduces_weighted_distance():
    tree, order = square_geometry_tree(4, 4)
    physical_to_leaf = {site: leaf for leaf, site in enumerate(order)}
    physical = ((0, 1, 1.0), (0, 4, 1.0), (5, 10, 0.5), (10, 15, 0.5))
    bonds = tuple(
        (physical_to_leaf[i], physical_to_leaf[j], coupling)
        for i, j, coupling in physical
    )
    optimized, trace = optimize_fusion_tree(tree, bonds)
    costs = [float(item["cost"]) for item in trace]
    assert all(right < left for left, right in zip(costs, costs[1:]))
    assert weighted_tree_distance(optimized, bonds) <= weighted_tree_distance(tree, bonds)


def test_joint_geometry_objective_tracks_and_limits_depth():
    tree = FusionTree.balanced(8)
    bonds = ((0, 7, 1.0), (1, 6, 1.0), (2, 5, 1.0))
    weights = TreeObjectiveWeights(
        bond_distance=1.0, maximum_depth=4.0, depth_variance=2.0
    )
    optimized, trace = optimize_fusion_tree(tree, bonds, weights=weights)
    costs = [float(item["cost"]) for item in trace]
    assert all(right < left for left, right in zip(costs, costs[1:]))
    assert tree_objective(optimized, bonds, weights) <= tree_objective(
        tree, bonds, weights
    )
    assert set(("bond_distance", "maximum_depth", "depth_variance")) <= set(trace[-1])
    assert max(leaf_depths(optimized)) <= max(leaf_depths(tree)) + 1


def test_tree_objective_components_for_balanced_tree():
    tree = FusionTree.balanced(8)
    components = tree_objective_components(tree, ((0, 1, 1.0),))
    assert leaf_depths(tree) == (3,) * 8
    assert components["bond_distance"] == 2.0
    assert components["maximum_depth"] == 3.0
    assert components["depth_variance"] == 0.0


def test_trainability_entropy_chain_rule_and_tree_dependence():
    from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig

    balanced = FusionTree.balanced(8)
    left = FusionTree.left_associated(8)
    balanced_metrics = tree_trainability_components(balanced, 0)
    left_metrics = tree_trainability_components(left, 0)
    expected_entropy = log2(len(balanced.enumerate_states(0)))
    np.testing.assert_allclose(
        balanced_metrics["conditional_entropy_sum"], expected_entropy, atol=1e-14
    )
    np.testing.assert_allclose(
        left_metrics["conditional_entropy_sum"], expected_entropy, atol=1e-14
    )
    assert balanced_metrics["maximum_reachable_actions"] >= 2.0
    assert balanced_metrics["action_vocabulary"] != left_metrics["action_vocabulary"]
    assert balanced_metrics["entropy_weighted_depth"] != left_metrics["entropy_weighted_depth"]
    assert balanced_metrics["path_label_complexity"] != left_metrics["path_label_complexity"]
    assert balanced_metrics["maximum_candidate_actions"] == CompactTreeGRUConfig(
        balanced, 0
    ).action_dim
    assert left_metrics["maximum_candidate_actions"] == CompactTreeGRUConfig(
        left, 0
    ).action_dim


def test_matrix_free_tree_plan_matches_materialized_graph_operator():
    n = 6
    tree = FusionTree.balanced(n)
    bonds = ((0, 5, 0.7), (1, 4, -0.2), (2, 3, 1.0))
    plan = build_tree_heisenberg_plan(tree, bonds, target_q=0)
    left = FusionTree.left_associated(n)
    recoupling, _ = recoupling_matrix_sparse(left, tree, 0)
    reference_left = heisenberg_graph_fusion_matrix(n, bonds, target_q=0)
    reference = (recoupling @ reference_left @ recoupling.T).toarray()
    rng = np.random.default_rng(40)
    vector = rng.normal(size=reference.shape[0]) + 1j * rng.normal(size=reference.shape[0])
    np.testing.assert_allclose(plan.apply(vector), reference @ vector, atol=5e-14)


def test_compiled_batched_tree_plan_matches_numpy_reference():
    import jax

    jax.config.update("jax_enable_x64", True)
    tree = FusionTree.balanced(6)
    bonds = ((0, 5, 0.7), (1, 4, -0.2), (2, 3, 1.0))
    plan = build_tree_heisenberg_plan(tree, bonds, target_q=0)
    compiled = plan.to_jax().compiled_apply_batch()
    vectors = np.random.default_rng(41).normal(size=(3, plan.dimension))
    expected = np.stack([plan.apply(vector) for vector in vectors])
    np.testing.assert_allclose(np.asarray(compiled(vectors)), expected, atol=5e-14)
    locally_compiled = plan.to_jax().compiled_local_apply_batch()
    np.testing.assert_allclose(
        np.asarray(locally_compiled(vectors)), expected, atol=5e-14
    )
