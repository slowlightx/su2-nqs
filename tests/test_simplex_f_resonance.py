import numpy as np
import pytest

from su2_nqs.operators.simplex_f_resonance import (
    composite_exchange_eigenvalue,
    simplex_connected_f_resonance_action,
    simplex_disjoint_four_spin_action,
    simplex_recoupling_block,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.recoupling import tree_basis_matrix
from su2_nqs.su2.tree_rotations import leaf_signature


def _full_basis_pair_action(n_sites: int):
    dimension = 1 << n_sites
    packed = np.arange(dimension, dtype=np.int64)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def pair(state: np.ndarray, first: int, second: int) -> np.ndarray:
        edge = tuple(sorted((int(first), int(second))))
        indices = cache.get(edge)
        if indices is None:
            first_mask = 1 << (n_sites - 1 - edge[0])
            second_mask = 1 << (n_sites - 1 - edge[1])
            unequal = ((packed & first_mask) != 0) != ((packed & second_mask) != 0)
            indices = np.where(unequal, packed ^ first_mask ^ second_mask, packed)
            cache[edge] = indices
        vector = np.asarray(state, dtype=np.complex128)
        return 0.5 * vector[indices] - 0.25 * vector

    return pair


def test_composite_exchange_uses_doubled_integer_labels():
    assert composite_exchange_eigenvalue(2, 2, 0) == -2.0
    assert composite_exchange_eigenvalue(2, 2, 2) == -1.0
    assert composite_exchange_eigenvalue(2, 2, 4) == 1.0
    with pytest.raises(ValueError, match="admissible"):
        composite_exchange_eigenvalue(2, 2, 1)


def test_factorized_connected_action_equals_explicit_disjoint_sum():
    generator = np.random.default_rng(1234)
    state = generator.normal(size=64) + 1.0j * generator.normal(size=64)
    pair = _full_basis_pair_action(6)
    cells = ((0, 1), (2, 3), (4, 5))
    factorized = simplex_connected_f_resonance_action(state, pair, *cells)
    explicit = simplex_disjoint_four_spin_action(state, pair, *cells)
    assert np.max(np.abs(factorized - explicit)) < 2.0e-12


def test_physical_twelve_spin_action_matches_f_move_block():
    cell_a = ((0, 1), (2, 3))
    cell_b = ((4, 5), (6, 7))
    cell_c = ((8, 9), (10, 11))
    tree = FusionTree.from_nested(((cell_a, cell_b), cell_c))
    full_basis, states = tree_basis_matrix(tree, target_q=2, target_m=2)
    signatures = tuple(leaf_signature(node) for node in tree.internal_nodes)
    required = {
        (0, 1): 0,
        (2, 3): 2,
        (0, 1, 2, 3): 2,
        (4, 5): 0,
        (6, 7): 2,
        (4, 5, 6, 7): 2,
        (8, 9): 0,
        (10, 11): 2,
        (8, 9, 10, 11): 2,
        tuple(range(12)): 2,
    }
    block = simplex_recoupling_block(2, 2, 2, 2, middle_site_count=4)
    columns = []
    for q_ab in block.left_channels:
        selected = []
        for index, state in enumerate(states):
            labels = dict(zip(signatures, state.labels, strict=True))
            if all(labels[signature] == value for signature, value in required.items()) and labels[tuple(range(8))] == q_ab:
                selected.append(index)
        assert len(selected) == 1
        columns.append(full_basis[:, selected[0]])
    physical = np.column_stack(columns)
    pair = _full_basis_pair_action(12)
    action = np.column_stack(
        [
            simplex_connected_f_resonance_action(
                physical[:, column],
                pair,
                tuple(range(4)),
                tuple(range(4, 8)),
                tuple(range(8, 12)),
            )
            for column in range(physical.shape[1])
        ]
    )
    observed = physical.conj().T @ action
    assert np.max(np.abs(block.f_matrix.T @ block.f_matrix - np.eye(3))) < 2.0e-14
    assert np.max(np.abs(observed - block.connected_resonance)) < 3.0e-12
