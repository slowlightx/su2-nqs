from __future__ import annotations

import pytest

from su2_nqs.benchmarks.tree_locality import (
    balanced_tree_distance,
    balanced_tree_leaf_addresses,
    joint_minimax_tree_order,
    tree_hopping_distance_score,
)
from su2_nqs.lattices.square import square_bonds
from su2_nqs.lattices.triangular import triangular_bonds


def test_balanced_six_site_tree_addresses_and_distances() -> None:
    assert balanced_tree_leaf_addresses(6) == (
        (0, 0),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0),
        (1, 1, 0),
        (1, 1, 1),
    )
    assert balanced_tree_distance(6, 1, 2) == 2
    assert balanced_tree_distance(6, 0, 5) == 5


def test_joint_square_triangular_minimax_order_is_frozen() -> None:
    square = square_bonds(2, 3)
    triangular = triangular_bonds(2, 3)
    order = joint_minimax_tree_order(6, (square, triangular))
    assert order == (1, 2, 5, 4, 0, 3)
    assert tree_hopping_distance_score(6, square, tuple(range(6))) == (26, 6, 114)
    assert tree_hopping_distance_score(6, triangular, tuple(range(6))) == (37, 6, 175)
    assert tree_hopping_distance_score(6, square, order) == (24, 5, 92)
    assert tree_hopping_distance_score(6, triangular, order) == (30, 5, 110)


def test_tree_locality_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="permutation"):
        tree_hopping_distance_score(3, ((0, 1),), (0, 0, 2))
    with pytest.raises(ValueError, match="n_sites <= 8"):
        joint_minimax_tree_order(9, (((0, 1),),))
