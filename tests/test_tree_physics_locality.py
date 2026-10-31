import pytest

from su2_nqs.benchmarks.tree_physics import recoupling_locality_hierarchy
from su2_nqs.su2.fusion_tree import FusionTree


def test_recoupling_locality_hierarchy_is_normalized_and_ordered():
    tree = FusionTree.left_associated(4)
    result = recoupling_locality_hierarchy(
        tree, [(0, 1, 1.0), (1, 2, -2.0), (0, 3, 1.0)]
    )
    assert result["total_absolute_weight"] == pytest.approx(4.0)
    shells = result["shells"]
    assert [item["K"] for item in shells] == sorted(item["K"] for item in shells)
    assert shells[-1]["cumulative_fraction"] == pytest.approx(1.0)
    assert sum(item["fraction"] for item in shells) == pytest.approx(1.0)
