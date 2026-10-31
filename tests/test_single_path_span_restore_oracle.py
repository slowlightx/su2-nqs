"""Complex three-state path span restored through two ancestors without loss."""
import numpy as np

from test_hierarchy_complement import _tree
from test_su2_complement import _singlet
from test_su2_conditional_response import _physical_merge
from su2_nqs.operators.su2_complement import (
    hierarchy_singlet_complement, restore_singlet_parent,
)
from su2_nqs.operators.su2_conditional_response import _channels


def test_complex_linear_cut_span_restores_both_ancestors_in_full_N12_space() -> None:
    """A tiny analogue of the fixed-cut18 ->36 ->72 ->root export.

    The physical basis is expanded explicitly. Both the individual basis
    vectors and a complex linear combination must survive successive q-QR
    ancestor restorations; there is no physical-space H query or truncation.
    """
    tree, physical, parent, _ = _tree()
    cut = hierarchy_singlet_complement(**tree, target=0)
    assert len(cut.steps) == 2
    assert all(side == 'left' for _, side, _ in cut.steps)
    channels = _channels(tree['blocks'][0].q_widths, cut.complement.q_widths, 0)
    base = np.array([cut.coefficients[q][a, b] for q, a, _, b in channels])
    base /= np.linalg.norm(base)
    rng = np.random.default_rng(50933)
    columns = [base]
    for _ in range(2):
        vector = rng.normal(size=len(base))+1j*rng.normal(size=len(base))
        for _ in range(2):
            for column in columns:
                vector -= column*np.vdot(column, vector)
        vector /= np.linalg.norm(vector)
        columns.append(vector)
    span = np.column_stack(columns)
    lookup = {parent: (i, left, right)
              for i, (left, right, parent) in enumerate(tree['merges'])}
    environment = physical[3]
    for parent_node, _, step in cut.steps:
        _, _, sibling = lookup[parent_node]
        environment = _physical_merge(physical[sibling], environment, step.complement_maps)

    def coefficients(vector: np.ndarray) -> dict[int, np.ndarray]:
        blocks = {q: np.zeros_like(c) for q, c in cut.coefficients.items()}
        for value, (q, a, _, b) in zip(vector, channels, strict=True):
            blocks[q][a, b] = value
        return blocks

    def restored_physical_state(vector: np.ndarray) -> np.ndarray:
        blocks = coefficients(vector)
        widths = tree['blocks'][0].q_widths
        maps = {}
        for node, _, step in reversed(cut.steps):
            index, _, _ = lookup[node]
            maps[node], blocks, diagnostic = restore_singlet_parent(
                step, blocks, widths,
                parent_widths={q: value.shape[1] for q, value in tree['maps'][index].items()},
            )
            assert diagnostic['no_singular_value_truncation']
            widths = {q: value.shape[1] for q, value in maps[node].items()}
        first = _physical_merge(physical[0], physical[1], maps[4])
        second = _physical_merge(first, physical[2], maps[5])
        return _singlet(second, physical[3], blocks)

    expected_basis = np.column_stack([
        _singlet(physical[0], environment, coefficients(span[:, i])) for i in range(3)
    ])
    restored_basis = np.column_stack([restored_physical_state(span[:, i]) for i in range(3)])
    np.testing.assert_allclose(restored_basis, expected_basis, atol=1e-12, rtol=0)
    np.testing.assert_allclose(expected_basis.conj().T@expected_basis, np.eye(3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(expected_basis[:, 0], parent, atol=1e-12, rtol=0)
    combination = rng.normal(size=3)+1j*rng.normal(size=3)
    combination /= np.linalg.norm(combination)
    np.testing.assert_allclose(
        restored_physical_state(span@combination), expected_basis@combination,
        atol=1e-12, rtol=0,
    )
