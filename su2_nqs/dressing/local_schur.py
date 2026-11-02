"""Exact four-spin Schur bases built from the canonical SU(2) backend.

Every angular-momentum label is doubled.  The returned columns are physical
computational-basis vectors with bit zero denoting ``m=-1/2``, bit one
denoting ``m=+1/2``, and site zero the most significant bit.  No CG data are
implemented here: all columns are delegated to
:func:`su2_nqs.su2.recoupling.tree_coupled_state`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..su2.fusion_tree import FusionTree, FusionTreeState
from ..su2.quantum_numbers import magnetic_numbers
from ..su2.recoupling import tree_coupled_state


FOUR_SPIN_TOTAL_Q: tuple[int, int, int] = (0, 2, 4)
FOUR_SPIN_MULTIPLICITIES: dict[int, int] = {0: 2, 2: 3, 4: 1}


@dataclass(frozen=True, order=True)
class SchurLabel:
    """One coupled basis label for four spin-half sites.

    ``multiplicity`` indexes an admissible internal-label assignment inside a
    fixed total doubled spin ``q``.  ``m`` is also doubled.
    """

    q: int
    multiplicity: int
    m: int
    tree_labels: tuple[int, ...]


@dataclass(frozen=True)
class FourSpinSchurBasis:
    """Complete orthonormal Schur basis for four spin-half sites."""

    tree: FusionTree
    labels: tuple[SchurLabel, ...]
    columns: np.ndarray

    def __post_init__(self) -> None:
        if self.tree.n_leaves != 4:
            raise ValueError("a four-spin Schur basis requires exactly four leaves")
        columns = np.asarray(self.columns)
        if columns.shape != (16, 16):
            raise ValueError("four-spin Schur columns must have shape (16, 16)")
        if len(self.labels) != 16:
            raise ValueError("four-spin Schur basis requires 16 labels")

    def indices(self, q: int, *, m: int | None = None) -> tuple[int, ...]:
        """Return column indices in one total-spin block and optional m slice."""

        if q not in FOUR_SPIN_TOTAL_Q:
            raise ValueError("four-spin total q must be 0, 2, or 4")
        result = tuple(
            index
            for index, label in enumerate(self.labels)
            if label.q == q and (m is None or label.m == m)
        )
        if not result:
            raise ValueError("requested four-spin Schur slice is empty")
        return result


def _tree_key(tree: FusionTree) -> tuple[object, ...]:
    """Return a deterministic nested key without relying on object identity."""

    from ..su2.fusion_tree import FusionNode, Leaf

    def visit(node):
        if isinstance(node, Leaf):
            return ("leaf", node.index)
        if isinstance(node, FusionNode):
            return ("node", visit(node.left), visit(node.right))
        raise TypeError("unknown fusion-tree node")

    return visit(tree.root)


@lru_cache(maxsize=None)
def _four_spin_schur_basis_cached(tree_key: tuple[object, ...]) -> FourSpinSchurBasis:
    def rebuild(value: tuple[object, ...]):
        tag = value[0]
        if tag == "leaf":
            return int(value[1])
        if tag == "node":
            return (rebuild(value[1]), rebuild(value[2]))
        raise ValueError("invalid cached fusion-tree key")

    tree = FusionTree.from_nested(rebuild(tree_key))
    labels: list[SchurLabel] = []
    columns: list[np.ndarray] = []
    for q in FOUR_SPIN_TOTAL_Q:
        states = tree.enumerate_states(q)
        if len(states) != FOUR_SPIN_MULTIPLICITIES[q]:
            raise RuntimeError("four-spin multiplicity disagrees with SU(2) decomposition")
        for multiplicity, state in enumerate(states):
            for m in magnetic_numbers(q):
                labels.append(
                    SchurLabel(
                        q=q,
                        multiplicity=multiplicity,
                        m=m,
                        tree_labels=state.labels,
                    )
                )
                columns.append(tree_coupled_state(tree, state, m))
    matrix = np.column_stack(columns).astype(np.complex128, copy=False)
    return FourSpinSchurBasis(tree=tree, labels=tuple(labels), columns=matrix)


def four_spin_schur_basis(tree: FusionTree | None = None) -> FourSpinSchurBasis:
    """Return a complete four-spin Schur basis for ``tree``.

    The default tree is the repository's left-associated convention
    ``(((0,1),2),3)``.  A fresh immutable view is returned so callers cannot
    mutate the cached oracle.
    """

    selected = FusionTree.left_associated(4) if tree is None else tree
    if selected.n_leaves != 4:
        raise ValueError("four_spin_schur_basis requires four physical leaves")
    cached = _four_spin_schur_basis_cached(_tree_key(selected))
    return FourSpinSchurBasis(
        tree=cached.tree,
        labels=cached.labels,
        columns=np.array(cached.columns, copy=True),
    )


def multiplicity_change_of_basis(
    source: FourSpinSchurBasis,
    target: FourSpinSchurBasis,
    q: int,
) -> np.ndarray:
    """Return ``R[target_mult, source_mult]`` in one doubled-spin sector.

    The highest-weight magnetic slice is sufficient because an SU(2)
    recoupling transform is identical for every ``m`` in the multiplet.
    """

    if q not in FOUR_SPIN_TOTAL_Q:
        raise ValueError("four-spin total q must be 0, 2, or 4")
    if source.tree.n_leaves != target.tree.n_leaves:
        raise ValueError("source and target bases must contain the same leaves")
    source_indices = source.indices(q, m=q)
    target_indices = target.indices(q, m=q)
    source_columns = source.columns[:, source_indices]
    target_columns = target.columns[:, target_indices]
    result = target_columns.conj().T @ source_columns
    expected = (FOUR_SPIN_MULTIPLICITIES[q],) * 2
    if result.shape != expected:
        raise RuntimeError("multiplicity recoupling matrix has the wrong shape")
    return np.asarray(result, dtype=np.complex128)


__all__ = [
    "FOUR_SPIN_MULTIPLICITIES",
    "FOUR_SPIN_TOTAL_Q",
    "FourSpinSchurBasis",
    "SchurLabel",
    "four_spin_schur_basis",
    "multiplicity_change_of_basis",
]
