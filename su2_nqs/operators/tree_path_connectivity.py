"""Tree-native single-state connectivity for spin-vector pair operators.

For a bond ``(i,j)``, the two local spins are propagated as doubled-rank-two
reduced tensors from their leaves to the lowest common ancestor (LCA).  They
are contracted to the scalar ``S_i.S_j`` at that node.  Labels above the LCA
are unchanged.  No fusion basis, recoupling matrix, or computational-basis
vector is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_tree import (
    FusionNode,
    FusionTree,
    FusionTreeState,
    Leaf,
    TreeNode,
    node_q,
)
from ..su2.quantum_numbers import (
    coupling_range,
    is_admissible_triangle,
    magnetic_numbers,
    validate_q,
)
from ..su2.reduced_vector import Q_OP_VECTOR, SPHERICAL_MU_Q
from ..su2.tree_rotations import leaf_signature


@dataclass(frozen=True)
class _VectorCandidate:
    updates: tuple[tuple[FusionNode, int], ...]
    root_q: int
    reduced: float


@dataclass(frozen=True)
class _FlatVectorCandidate:
    """One vector-operator branch using postorder integer node indices."""

    updates: tuple[tuple[int, int], ...]
    root_q: int
    reduced: float


@dataclass(frozen=True)
class _FlatVectorStep:
    """One precompiled bottom-up transport step along a leaf-to-LCA path."""

    node_index: int
    active_reference: int
    spectator_reference: int
    active_is_left: bool


@dataclass(frozen=True)
class _CompiledGraphBond:
    """State-independent tree structure needed by one graph bond."""

    first: int
    second: int
    coupling: float
    lca_index: int
    left_reference: int
    right_reference: int
    left_steps: tuple[_FlatVectorStep, ...]
    right_steps: tuple[_FlatVectorStep, ...]


@dataclass(frozen=True)
class TreeHeisenbergGraphConnectivityPlan:
    """Reusable exact column plan for a Heisenberg graph on one fusion tree.

    The tree topology, bond LCAs, and leaf-to-LCA transport paths are compiled
    once into postorder integer indices.  :meth:`column` consequently validates
    a center state exactly once and performs no recursive ``FusionNode``
    hashing while applying the graph.  Magnetic labels and all returned
    connectivity still follow the public doubled-integer and
    ``H[new_path, old_path]`` conventions.

    Use :func:`compile_tree_heisenberg_graph_connectivity_plan` to construct a
    cached plan.  The historical :func:`tree_heisenberg_graph_connectivity`
    API remains unchanged and is retained as an independent reference path.
    """

    tree: FusionTree
    bonds: tuple[tuple[int, int, float], ...]
    target_q: int | None
    coefficient_tolerance: float
    _left_references: tuple[int, ...]
    _right_references: tuple[int, ...]
    _compiled_bonds: tuple[_CompiledGraphBond, ...]

    @property
    def bond_count(self) -> int:
        """Number of graph terms retained in their input order."""

        return len(self._compiled_bonds)

    def column(
        self, state: FusionTreeState | Sequence[int]
    ) -> tuple[tuple[FusionTreeState, ...], np.ndarray]:
        """Apply the precompiled graph to one fusion-tree basis state."""

        checked, labels = _validated_flat_state(
            state, self._left_references, self._right_references
        )
        if self.target_q is not None and checked.total_q != self.target_q:
            raise ValueError(
                "state does not belong to the requested total-spin sector"
            )
        combined: dict[tuple[int, ...], float] = {}
        for bond in self._compiled_bonds:
            connected, values = _flat_pair_heisenberg_connectivity(
                checked,
                labels,
                bond,
                coefficient_tolerance=self.coefficient_tolerance,
            )
            for new_state, value in zip(connected, values, strict=True):
                combined[new_state.labels] = (
                    combined.get(new_state.labels, 0.0)
                    + bond.coupling * float(value)
                )
        items = tuple(
            sorted(
                (labels_new, value)
                for labels_new, value in combined.items()
                if abs(value) > self.coefficient_tolerance
            )
        )
        return (
            tuple(FusionTreeState(labels_new) for labels_new, _ in items),
            np.asarray([value for _, value in items], dtype=np.float64),
        )


def _project_vector_reduced(q_new: int, q_old: int, magnetic_element) -> float:
    if not is_admissible_triangle(q_old, Q_OP_VECTOR, q_new):
        return 0.0
    answer = 0.0
    for m_old in magnetic_numbers(q_old):
        for mu_q in SPHERICAL_MU_Q:
            m_new = m_old + mu_q
            if abs(m_new) > q_new or (q_new - m_new) % 2:
                continue
            coefficient = clebsch_gordan(
                q_old, m_old, Q_OP_VECTOR, mu_q, q_new, m_new
            )
            if coefficient:
                answer += coefficient * magnetic_element(m_new, m_old, mu_q)
    return float(answer / math.sqrt(q_new + 1.0))


@lru_cache(maxsize=None)
def _left_vector_advance_coefficient(
    q_child_new: int,
    q_child_old: int,
    q_spectator: int,
    q_parent_new: int,
    q_parent_old: int,
) -> float:
    """Transport a unit reduced vector from the left child to its parent."""

    values = (
        q_child_new,
        q_child_old,
        q_spectator,
        q_parent_new,
        q_parent_old,
    )
    for name, value in zip(
        ("q_child_new", "q_child_old", "q_spectator", "q_parent_new", "q_parent_old"),
        values,
        strict=True,
    ):
        validate_q(value, name=name)
    if not (
        is_admissible_triangle(q_child_new, q_spectator, q_parent_new)
        and is_admissible_triangle(q_child_old, q_spectator, q_parent_old)
        and is_admissible_triangle(q_child_old, Q_OP_VECTOR, q_child_new)
        and is_admissible_triangle(q_parent_old, Q_OP_VECTOR, q_parent_new)
    ):
        return 0.0

    def element(m_parent_new: int, m_parent_old: int, mu_q: int) -> float:
        total = 0.0
        for m_child_old in magnetic_numbers(q_child_old):
            m_spectator = m_parent_old - m_child_old
            m_child_new = m_child_old + mu_q
            if (
                abs(m_spectator) > q_spectator
                or (q_spectator - m_spectator) % 2
                or abs(m_child_new) > q_child_new
                or (q_child_new - m_child_new) % 2
                or m_child_new + m_spectator != m_parent_new
            ):
                continue
            ket = clebsch_gordan(
                q_child_old,
                m_child_old,
                q_spectator,
                m_spectator,
                q_parent_old,
                m_parent_old,
            )
            bra = clebsch_gordan(
                q_child_new,
                m_child_new,
                q_spectator,
                m_spectator,
                q_parent_new,
                m_parent_new,
            )
            component = clebsch_gordan(
                q_child_old,
                m_child_old,
                Q_OP_VECTOR,
                mu_q,
                q_child_new,
                m_child_new,
            ) / math.sqrt(q_child_new + 1.0)
            total += bra * component * ket
        return total

    return _project_vector_reduced(q_parent_new, q_parent_old, element)


@lru_cache(maxsize=None)
def _right_vector_advance_coefficient(
    q_spectator: int,
    q_child_new: int,
    q_child_old: int,
    q_parent_new: int,
    q_parent_old: int,
) -> float:
    """Transport a unit reduced vector from the right child to its parent."""

    values = (
        q_spectator,
        q_child_new,
        q_child_old,
        q_parent_new,
        q_parent_old,
    )
    for name, value in zip(
        ("q_spectator", "q_child_new", "q_child_old", "q_parent_new", "q_parent_old"),
        values,
        strict=True,
    ):
        validate_q(value, name=name)
    if not (
        is_admissible_triangle(q_spectator, q_child_new, q_parent_new)
        and is_admissible_triangle(q_spectator, q_child_old, q_parent_old)
        and is_admissible_triangle(q_child_old, Q_OP_VECTOR, q_child_new)
        and is_admissible_triangle(q_parent_old, Q_OP_VECTOR, q_parent_new)
    ):
        return 0.0

    def element(m_parent_new: int, m_parent_old: int, mu_q: int) -> float:
        total = 0.0
        for m_spectator in magnetic_numbers(q_spectator):
            m_child_old = m_parent_old - m_spectator
            m_child_new = m_child_old + mu_q
            if (
                abs(m_child_old) > q_child_old
                or (q_child_old - m_child_old) % 2
                or abs(m_child_new) > q_child_new
                or (q_child_new - m_child_new) % 2
                or m_spectator + m_child_new != m_parent_new
            ):
                continue
            ket = clebsch_gordan(
                q_spectator,
                m_spectator,
                q_child_old,
                m_child_old,
                q_parent_old,
                m_parent_old,
            )
            bra = clebsch_gordan(
                q_spectator,
                m_spectator,
                q_child_new,
                m_child_new,
                q_parent_new,
                m_parent_new,
            )
            component = clebsch_gordan(
                q_child_old,
                m_child_old,
                Q_OP_VECTOR,
                mu_q,
                q_child_new,
                m_child_new,
            ) / math.sqrt(q_child_new + 1.0)
            total += bra * component * ket
        return total

    return _project_vector_reduced(q_parent_new, q_parent_old, element)


@lru_cache(maxsize=None)
def _scalar_vector_contraction_coefficient(
    q_left_new: int,
    q_right_new: int,
    q_left_old: int,
    q_right_old: int,
    q_total: int,
) -> float:
    """Contract two unit reduced vectors into ``A.B`` at fixed total q."""

    for name, value in (
        ("q_left_new", q_left_new),
        ("q_right_new", q_right_new),
        ("q_left_old", q_left_old),
        ("q_right_old", q_right_old),
        ("q_total", q_total),
    ):
        validate_q(value, name=name)
    if not (
        is_admissible_triangle(q_left_new, q_right_new, q_total)
        and is_admissible_triangle(q_left_old, q_right_old, q_total)
        and is_admissible_triangle(q_left_old, Q_OP_VECTOR, q_left_new)
        and is_admissible_triangle(q_right_old, Q_OP_VECTOR, q_right_new)
    ):
        return 0.0
    total_m = q_total
    answer = 0.0
    for m_left_old in magnetic_numbers(q_left_old):
        m_right_old = total_m - m_left_old
        if (
            abs(m_right_old) > q_right_old
            or (q_right_old - m_right_old) % 2
        ):
            continue
        ket = clebsch_gordan(
            q_left_old,
            m_left_old,
            q_right_old,
            m_right_old,
            q_total,
            total_m,
        )
        if not ket:
            continue
        for mu_q in SPHERICAL_MU_Q:
            m_left_new = m_left_old + mu_q
            m_right_new = m_right_old - mu_q
            if (
                abs(m_left_new) > q_left_new
                or (q_left_new - m_left_new) % 2
                or abs(m_right_new) > q_right_new
                or (q_right_new - m_right_new) % 2
            ):
                continue
            bra = clebsch_gordan(
                q_left_new,
                m_left_new,
                q_right_new,
                m_right_new,
                q_total,
                total_m,
            )
            left_component = clebsch_gordan(
                q_left_old,
                m_left_old,
                Q_OP_VECTOR,
                mu_q,
                q_left_new,
                m_left_new,
            ) / math.sqrt(q_left_new + 1.0)
            right_component = clebsch_gordan(
                q_right_old,
                m_right_old,
                Q_OP_VECTOR,
                -mu_q,
                q_right_new,
                m_right_new,
            ) / math.sqrt(q_right_new + 1.0)
            phase = 1.0 if mu_q == 0 else -1.0
            answer += bra * phase * left_component * right_component * ket
    return float(answer)


def _validated_state(
    tree: FusionTree, state: FusionTreeState | Sequence[int]
) -> tuple[FusionTreeState, dict[FusionNode, int]]:
    raw = state.labels if isinstance(state, FusionTreeState) else tuple(state)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError("tree-state labels must be doubled integers")
    checked = FusionTreeState(tuple(int(value) for value in raw))
    labels = checked.label_map(tree)
    for node, parent_q in zip(tree.internal_nodes, checked.labels, strict=True):
        validate_q(parent_q, name="state label")
        if parent_q not in coupling_range(
            node_q(node.left, labels), node_q(node.right, labels)
        ):
            raise ValueError("state contains an inadmissible doubled-spin coupling")
    return checked, labels


def _site_vector_candidates(
    node: TreeNode,
    labels: dict[FusionNode, int],
    site: int,
    *,
    coefficient_tolerance: float,
) -> tuple[_VectorCandidate, ...]:
    if isinstance(node, Leaf):
        if node.index != site:
            raise ValueError("requested site is outside the active subtree")
        return (_VectorCandidate((), 1, math.sqrt(1.5)),)

    parent_old = labels[node]
    if site in leaf_signature(node.left):
        active = _site_vector_candidates(
            node.left, labels, site, coefficient_tolerance=coefficient_tolerance
        )
        spectator_q = node_q(node.right, labels)
        rows = []
        for candidate in active:
            for parent_new in coupling_range(candidate.root_q, spectator_q):
                coefficient = _left_vector_advance_coefficient(
                    candidate.root_q,
                    node_q(node.left, labels),
                    spectator_q,
                    parent_new,
                    parent_old,
                )
                value = candidate.reduced * coefficient
                if abs(value) > coefficient_tolerance:
                    rows.append(
                        _VectorCandidate(
                            candidate.updates + ((node, int(parent_new)),),
                            int(parent_new),
                            value,
                        )
                    )
        return tuple(rows)
    if site in leaf_signature(node.right):
        active = _site_vector_candidates(
            node.right, labels, site, coefficient_tolerance=coefficient_tolerance
        )
        spectator_q = node_q(node.left, labels)
        rows = []
        for candidate in active:
            for parent_new in coupling_range(spectator_q, candidate.root_q):
                coefficient = _right_vector_advance_coefficient(
                    spectator_q,
                    candidate.root_q,
                    node_q(node.right, labels),
                    parent_new,
                    parent_old,
                )
                value = candidate.reduced * coefficient
                if abs(value) > coefficient_tolerance:
                    rows.append(
                        _VectorCandidate(
                            candidate.updates + ((node, int(parent_new)),),
                            int(parent_new),
                            value,
                        )
                    )
        return tuple(rows)
    raise ValueError("requested site is outside the active subtree")


def _lowest_common_ancestor(node: TreeNode, first: int, second: int) -> FusionNode:
    if isinstance(node, Leaf):
        raise RuntimeError("distinct leaves cannot have a leaf LCA")
    left = leaf_signature(node.left)
    if first in left and second in left:
        return _lowest_common_ancestor(node.left, first, second)
    right = leaf_signature(node.right)
    if first in right and second in right:
        return _lowest_common_ancestor(node.right, first, second)
    if (first in left and second in right) or (second in left and first in right):
        return node
    raise ValueError("bond endpoint lies outside the fusion tree")


def _flat_reference(
    node: TreeNode, positions_by_identity: dict[int, int]
) -> int:
    """Encode a leaf or internal node without structural dataclass hashing."""

    if isinstance(node, Leaf):
        return -node.index - 1
    return positions_by_identity[id(node)]


def _flat_reference_q(reference: int, labels: tuple[int, ...]) -> int:
    """Read a doubled spin from a flat postorder node reference."""

    return 1 if reference < 0 else labels[reference]


def _compile_flat_vector_steps(
    node: TreeNode,
    site: int,
    positions_by_identity: dict[int, int],
) -> tuple[_FlatVectorStep, ...]:
    """Compile the bottom-up transport path from ``site`` through ``node``."""

    if isinstance(node, Leaf):
        if node.index != site:
            raise ValueError("requested site is outside the active subtree")
        return ()
    if site in leaf_signature(node.left):
        return _compile_flat_vector_steps(
            node.left, site, positions_by_identity
        ) + (
            _FlatVectorStep(
                positions_by_identity[id(node)],
                _flat_reference(node.left, positions_by_identity),
                _flat_reference(node.right, positions_by_identity),
                True,
            ),
        )
    if site in leaf_signature(node.right):
        return _compile_flat_vector_steps(
            node.right, site, positions_by_identity
        ) + (
            _FlatVectorStep(
                positions_by_identity[id(node)],
                _flat_reference(node.right, positions_by_identity),
                _flat_reference(node.left, positions_by_identity),
                False,
            ),
        )
    raise ValueError("requested site is outside the active subtree")


def _validated_flat_state(
    state: FusionTreeState | Sequence[int],
    left_references: tuple[int, ...],
    right_references: tuple[int, ...],
) -> tuple[FusionTreeState, tuple[int, ...]]:
    """Validate one state against a precompiled integer tree topology."""

    raw = state.labels if isinstance(state, FusionTreeState) else tuple(state)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw
    ):
        raise TypeError("tree-state labels must be doubled integers")
    checked = FusionTreeState(tuple(int(value) for value in raw))
    if len(checked.labels) != len(left_references):
        raise ValueError("state label count does not match the tree")
    for index, parent_q in enumerate(checked.labels):
        validate_q(parent_q, name="state label")
        if parent_q not in coupling_range(
            _flat_reference_q(left_references[index], checked.labels),
            _flat_reference_q(right_references[index], checked.labels),
        ):
            raise ValueError(
                "state contains an inadmissible doubled-spin coupling"
            )
    return checked, checked.labels


def _flat_site_vector_candidates(
    labels: tuple[int, ...],
    steps: tuple[_FlatVectorStep, ...],
    *,
    coefficient_tolerance: float,
) -> tuple[_FlatVectorCandidate, ...]:
    """Propagate one leaf vector through a precompiled integer path."""

    active = (_FlatVectorCandidate((), 1, math.sqrt(1.5)),)
    for step in steps:
        parent_old = labels[step.node_index]
        child_old = _flat_reference_q(step.active_reference, labels)
        spectator_q = _flat_reference_q(step.spectator_reference, labels)
        rows: list[_FlatVectorCandidate] = []
        for candidate in active:
            parent_range = (
                coupling_range(candidate.root_q, spectator_q)
                if step.active_is_left
                else coupling_range(spectator_q, candidate.root_q)
            )
            for parent_new in parent_range:
                if step.active_is_left:
                    coefficient = _left_vector_advance_coefficient(
                        candidate.root_q,
                        child_old,
                        spectator_q,
                        parent_new,
                        parent_old,
                    )
                else:
                    coefficient = _right_vector_advance_coefficient(
                        spectator_q,
                        candidate.root_q,
                        child_old,
                        parent_new,
                        parent_old,
                    )
                value = candidate.reduced * coefficient
                if abs(value) > coefficient_tolerance:
                    rows.append(
                        _FlatVectorCandidate(
                            candidate.updates
                            + ((step.node_index, int(parent_new)),),
                            int(parent_new),
                            value,
                        )
                    )
        active = tuple(rows)
    return active


def _flat_pair_heisenberg_connectivity(
    checked: FusionTreeState,
    labels: tuple[int, ...],
    bond: _CompiledGraphBond,
    *,
    coefficient_tolerance: float,
) -> tuple[tuple[FusionTreeState, ...], np.ndarray]:
    """Apply one precompiled bond to an already validated center state."""

    left_candidates = _flat_site_vector_candidates(
        labels,
        bond.left_steps,
        coefficient_tolerance=coefficient_tolerance,
    )
    right_candidates = _flat_site_vector_candidates(
        labels,
        bond.right_steps,
        coefficient_tolerance=coefficient_tolerance,
    )
    q_left_old = _flat_reference_q(bond.left_reference, labels)
    q_right_old = _flat_reference_q(bond.right_reference, labels)
    q_total = labels[bond.lca_index]
    combined: dict[tuple[int, ...], float] = {}
    for left in left_candidates:
        for right in right_candidates:
            scalar = _scalar_vector_contraction_coefficient(
                left.root_q,
                right.root_q,
                q_left_old,
                q_right_old,
                q_total,
            )
            value = left.reduced * right.reduced * scalar
            if abs(value) <= coefficient_tolerance:
                continue
            output = list(checked.labels)
            for node_index, q_new in left.updates + right.updates:
                output[node_index] = int(q_new)
            key = tuple(output)
            combined[key] = combined.get(key, 0.0) + value
    items = tuple(
        sorted(
            (labels_new, value)
            for labels_new, value in combined.items()
            if abs(value) > coefficient_tolerance
        )
    )
    return (
        tuple(FusionTreeState(labels_new) for labels_new, _ in items),
        np.asarray([value for _, value in items], dtype=np.float64),
    )


def _normalize_graph_bonds(
    tree: FusionTree,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
) -> tuple[tuple[int, int, float], ...]:
    """Validate graph terms once while retaining their original order."""

    normalized: list[tuple[int, int, float]] = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        if isinstance(coupling, bool) or not isinstance(coupling, Real):
            raise TypeError("bond coupling must be real")
        coupling = float(coupling)
        if not math.isfinite(coupling):
            raise ValueError("bond coupling must be finite")
        if any(
            isinstance(site, bool) or not isinstance(site, Integral)
            for site in (first, second)
        ):
            raise TypeError("bond endpoints must be integers")
        first, second = int(first), int(second)
        if (
            first == second
            or not 0 <= first < tree.n_leaves
            or not 0 <= second < tree.n_leaves
        ):
            raise ValueError(
                "bond endpoints must be distinct leaves inside the tree"
            )
        normalized.append((first, second, coupling))
    return tuple(normalized)


@lru_cache(maxsize=128)
def _cached_tree_heisenberg_graph_connectivity_plan(
    tree: FusionTree,
    bonds: tuple[tuple[int, int, float], ...],
    target_q: int | None,
    coefficient_tolerance: float,
) -> TreeHeisenbergGraphConnectivityPlan:
    nodes = tree.internal_nodes
    positions_by_identity = {id(node): index for index, node in enumerate(nodes)}
    left_references = tuple(
        _flat_reference(node.left, positions_by_identity) for node in nodes
    )
    right_references = tuple(
        _flat_reference(node.right, positions_by_identity) for node in nodes
    )
    compiled_bonds: list[_CompiledGraphBond] = []
    for first, second, coupling in bonds:
        lca = _lowest_common_ancestor(tree.root, first, second)
        left_sites = leaf_signature(lca.left)
        left_site, right_site = (
            (first, second) if first in left_sites else (second, first)
        )
        compiled_bonds.append(
            _CompiledGraphBond(
                first,
                second,
                coupling,
                positions_by_identity[id(lca)],
                _flat_reference(lca.left, positions_by_identity),
                _flat_reference(lca.right, positions_by_identity),
                _compile_flat_vector_steps(
                    lca.left, left_site, positions_by_identity
                ),
                _compile_flat_vector_steps(
                    lca.right, right_site, positions_by_identity
                ),
            )
        )
    return TreeHeisenbergGraphConnectivityPlan(
        tree,
        bonds,
        target_q,
        coefficient_tolerance,
        left_references,
        right_references,
        tuple(compiled_bonds),
    )


def compile_tree_heisenberg_graph_connectivity_plan(
    tree: FusionTree,
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    target_q: int | None = None,
    coefficient_tolerance: float = 1.0e-14,
) -> TreeHeisenbergGraphConnectivityPlan:
    """Compile and cache reusable structural data for exact graph columns.

    Compilation is independent of the center labels.  Repeating this call with
    an equal immutable tree and normalized bond sequence returns the same
    cached plan, so callers may either retain the object explicitly or rely on
    the bounded module cache.
    """

    if target_q is not None:
        validate_q(target_q, name="target_q")
        target_q = int(target_q)
    if (
        isinstance(coefficient_tolerance, bool)
        or not isinstance(coefficient_tolerance, Real)
        or not math.isfinite(float(coefficient_tolerance))
        or coefficient_tolerance < 0.0
    ):
        raise ValueError(
            "coefficient_tolerance must be finite and non-negative"
        )
    normalized = _normalize_graph_bonds(tree, bonds)
    return _cached_tree_heisenberg_graph_connectivity_plan(
        tree, normalized, target_q, float(coefficient_tolerance)
    )


def tree_pair_heisenberg_connectivity(
    tree: FusionTree,
    state: FusionTreeState | Sequence[int],
    first: int,
    second: int,
    *,
    coefficient_tolerance: float = 1.0e-14,
) -> tuple[tuple[FusionTreeState, ...], np.ndarray]:
    """Apply one ``S_first.S_second`` column in an arbitrary tree basis."""

    if any(
        isinstance(site, bool) or not isinstance(site, Integral)
        for site in (first, second)
    ):
        raise TypeError("bond endpoints must be integers")
    first, second = int(first), int(second)
    if first == second or not 0 <= first < tree.n_leaves or not 0 <= second < tree.n_leaves:
        raise ValueError("bond endpoints must be distinct leaves inside the tree")
    if (
        isinstance(coefficient_tolerance, bool)
        or not isinstance(coefficient_tolerance, Real)
        or not math.isfinite(float(coefficient_tolerance))
        or coefficient_tolerance < 0.0
    ):
        raise ValueError("coefficient_tolerance must be finite and non-negative")
    tolerance = float(coefficient_tolerance)
    checked, labels = _validated_state(tree, state)
    lca = _lowest_common_ancestor(tree.root, first, second)
    left_sites = leaf_signature(lca.left)
    left_site, right_site = (
        (first, second) if first in left_sites else (second, first)
    )
    left_candidates = _site_vector_candidates(
        lca.left, labels, left_site, coefficient_tolerance=tolerance
    )
    right_candidates = _site_vector_candidates(
        lca.right, labels, right_site, coefficient_tolerance=tolerance
    )
    q_left_old = node_q(lca.left, labels)
    q_right_old = node_q(lca.right, labels)
    q_total = labels[lca]
    positions = {node: index for index, node in enumerate(tree.internal_nodes)}
    combined: dict[tuple[int, ...], float] = {}
    for left in left_candidates:
        for right in right_candidates:
            scalar = _scalar_vector_contraction_coefficient(
                left.root_q,
                right.root_q,
                q_left_old,
                q_right_old,
                q_total,
            )
            value = left.reduced * right.reduced * scalar
            if abs(value) <= tolerance:
                continue
            output = list(checked.labels)
            for node, q_new in left.updates + right.updates:
                output[positions[node]] = int(q_new)
            key = tuple(output)
            combined[key] = combined.get(key, 0.0) + value
    items = tuple(
        sorted(
            (labels_new, value)
            for labels_new, value in combined.items()
            if abs(value) > tolerance
        )
    )
    return (
        tuple(FusionTreeState(labels_new) for labels_new, _ in items),
        np.asarray([value for _, value in items], dtype=np.float64),
    )


def tree_heisenberg_graph_connectivity(
    tree: FusionTree,
    state: FusionTreeState | Sequence[int],
    bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    *,
    target_q: int | None = None,
    coefficient_tolerance: float = 1.0e-14,
) -> tuple[tuple[FusionTreeState, ...], np.ndarray]:
    """Apply ``sum_ij J_ij S_i.S_j`` with ``H[new,old]`` connectivity."""

    checked, _ = _validated_state(tree, state)
    if target_q is not None:
        validate_q(target_q, name="target_q")
        if checked.total_q != int(target_q):
            raise ValueError("state does not belong to the requested total-spin sector")
    combined: dict[tuple[int, ...], float] = {}
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        if isinstance(coupling, bool) or not isinstance(coupling, Real):
            raise TypeError("bond coupling must be real")
        coupling = float(coupling)
        if not math.isfinite(coupling):
            raise ValueError("bond coupling must be finite")
        connected, values = tree_pair_heisenberg_connectivity(
            tree,
            checked,
            first,
            second,
            coefficient_tolerance=coefficient_tolerance,
        )
        for new_state, value in zip(connected, values, strict=True):
            combined[new_state.labels] = (
                combined.get(new_state.labels, 0.0) + coupling * float(value)
            )
    tolerance = float(coefficient_tolerance)
    items = tuple(
        sorted(
            (labels_new, value)
            for labels_new, value in combined.items()
            if abs(value) > tolerance
        )
    )
    return (
        tuple(FusionTreeState(labels_new) for labels_new, _ in items),
        np.asarray([value for _, value in items], dtype=np.float64),
    )


__all__ = [
    "TreeHeisenbergGraphConnectivityPlan",
    "compile_tree_heisenberg_graph_connectivity_plan",
    "tree_heisenberg_graph_connectivity",
    "tree_pair_heisenberg_connectivity",
]
