import numpy as np

from su2_nqs.lattices.geometry_optimization import TreeObjectiveWeights
from su2_nqs.lattices.physical_tree_optimization import (
    PhysicalTreeWeights,
    optimize_fusion_tree_physical,
)
from su2_nqs.su2.fusion_tree import FusionTree


def test_physical_tree_optimizer_is_monotone_and_preserves_norm():
    tree = FusionTree.balanced(6)
    dimension = len(tree.enumerate_states(0))
    vector = np.random.default_rng(70).normal(size=dimension)
    optimized, transformed, trace = optimize_fusion_tree_physical(
        tree,
        ((0, 5, 1.0),),
        (vector,),
        structural_weights=TreeObjectiveWeights(
            bond_distance=1.0, maximum_depth=0.1
        ),
        physical_weights=PhysicalTreeWeights(mean_block_spin_entropy=2.0),
        max_moves=4,
    )
    del optimized
    costs = [float(item["cost"]) for item in trace]
    assert all(right < left for left, right in zip(costs, costs[1:]))
    np.testing.assert_allclose(np.linalg.norm(transformed[0]), np.linalg.norm(vector), atol=2e-14)
