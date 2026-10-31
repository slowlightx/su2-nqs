import numpy as np

from su2_nqs.benchmarks.tree_operator_flow import (
    ResponseTreeWeights,
    optimize_fusion_tree_response,
    source_target_sampling_metrics,
    tree_boundary_channel_metrics,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.operators.tree_plan import build_tree_heisenberg_plan
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


def test_boundary_channel_rank_is_exact_spatial_svd_rank():
    tree = FusionTree.balanced(4)
    bonds = ((0, 2, 1.0), (1, 3, 1.0))
    metrics = tree_boundary_channel_metrics(tree, bonds)
    assert metrics["maximum_spatial_channel_rank"] == 2
    assert metrics["total_spatial_channel_rank"] == 8
    by_size = {}
    for cut in metrics["cuts"]:
        by_size.setdefault(cut["region_size"], []).append(cut)
    assert len(by_size[1]) == 4
    assert len(by_size[2]) == 2
    assert all(cut["spatial_channel_rank"] == 1 for cut in by_size[1])
    assert all(cut["spatial_channel_rank"] == 2 for cut in by_size[2])
    for cut in by_size[2]:
        np.testing.assert_allclose(cut["singular_values"], (1.0, 1.0))


def test_source_target_sampling_metrics_detects_structural_mismatch():
    source = np.asarray((1.0, 0.0), dtype=np.complex128)
    target = np.asarray((0.0, 2.0j), dtype=np.complex128)
    metrics = source_target_sampling_metrics(source, target, sample_count=128)
    assert metrics["born_target_affinity_squared"] == 0.0
    assert metrics["expected_unseen_target_weight"] == 1.0
    assert metrics["target_weight_outside_source_support"] == 1.0


def test_response_tree_optimizer_is_monotone_and_preserves_physical_vectors():
    n = 6
    tree = FusionTree.balanced(n)
    bonds = square_j1j2_bonds(2, 3, j1=1.0, j2=0.6, periodic=False)
    hamiltonian = heisenberg_graph_fusion_matrix(
        n, bonds, target_q=0
    ).toarray()
    source_left = np.linalg.eigh(hamiltonian)[1][:, 0]
    perturbation = heisenberg_graph_fusion_matrix(
        n,
        square_j1j2_bonds(2, 3, j1=0.0, j2=1.0, periodic=False),
        target_q=0,
    ).toarray()
    response_left = perturbation @ source_left
    response_left -= np.vdot(source_left, response_left) * source_left
    left = FusionTree.left_associated(n)
    left_to_tree, _ = recoupling_matrix_sparse(left, tree, 0)
    source = np.asarray(left_to_tree @ source_left)
    response = np.asarray(left_to_tree @ response_left)
    optimized, transformed_source, transformed_response, trace = (
        optimize_fusion_tree_response(
            tree,
            bonds,
            source,
            response,
            sample_count=256,
            weights=ResponseTreeWeights(
                unseen_target_weight=1.0,
                maximum_leaf_depth=0.01,
            ),
            max_moves=8,
        )
    )
    costs = [float(row["cost"]) for row in trace]
    assert all(right < left for left, right in zip(costs, costs[1:]))
    np.testing.assert_allclose(np.linalg.norm(transformed_source), 1.0, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(transformed_response), np.linalg.norm(response), atol=2e-14
    )
    direct, _ = recoupling_matrix_sparse(tree, optimized, 0)
    np.testing.assert_allclose(transformed_source, direct @ source, atol=3e-14)
    np.testing.assert_allclose(transformed_response, direct @ response, atol=3e-14)


def test_square_tree_plan_is_covariant_through_n10():
    rng = np.random.default_rng(20260907)
    for rows, columns in ((2, 3), (2, 4), (2, 5)):
        n = rows * columns
        tree = FusionTree.balanced(n)
        left = FusionTree.left_associated(n)
        bonds = square_j1j2_bonds(
            rows, columns, j1=1.0, j2=0.37, periodic=False
        )
        plan = build_tree_heisenberg_plan(tree, bonds, target_q=0)
        recoupling, _ = recoupling_matrix_sparse(left, tree, 0)
        left_matrix = heisenberg_graph_fusion_matrix(
            n, bonds, target_q=0
        )
        source = rng.normal(size=plan.dimension)
        expected = recoupling @ (left_matrix @ (recoupling.T @ source))
        np.testing.assert_allclose(plan.apply(source), expected, atol=8e-14)
