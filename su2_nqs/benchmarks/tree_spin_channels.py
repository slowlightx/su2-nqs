"""Exact physical projectors for intermediate-spin channels of a tree.

An internal fusion-tree label is the doubled total spin of the physical
leaves below that node.  The non-root internal supports of any binary tree
form a laminar family: every pair is nested or disjoint.  Their block-spin
Casimirs therefore commute, making a product of ``q <= qmax`` projectors an
unambiguous orthogonal projector onto the corresponding single-tree
channel-truncated Hilbert subspace.

These routines are exact-small references.  They materialize a full
computational-basis vector and dense subsystem spin eigenvectors; scalable
training code must not call them at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..su2.fusion_tree import FusionTree, Leaf, TreeNode
from .block_spin_fcs import subsystem_spin_eigenbasis


def _node_leaf_indices(node: TreeNode) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (int(node.index),)
    return _node_leaf_indices(node.left) + _node_leaf_indices(node.right)


def physical_internal_supports(
    tree: FusionTree, site_order: Sequence[int]
) -> tuple[dict[str, object], ...]:
    """Return every non-root internal physical support in tree postorder.

    ``site_order[leaf]`` is the physical site assigned to that tree leaf.
    Leaves themselves and the fixed-total-spin root are intentionally absent.
    """

    order = tuple(int(site) for site in site_order)
    if sorted(order) != list(range(tree.n_leaves)):
        raise ValueError("site_order must contain every physical site once")
    rows = []
    for node_index, node in enumerate(tree.internal_nodes[:-1]):
        leaves = _node_leaf_indices(node)
        support = tuple(sorted(order[leaf] for leaf in leaves))
        rows.append(
            {
                "node_postorder_index": node_index,
                "leaf_indices": list(leaves),
                "physical_support": list(support),
                "support_size": len(support),
            }
        )
    return tuple(rows)


def laminar_family_audit(
    support_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Check that physical supports are unique and nested-or-disjoint."""

    supports = [
        frozenset(map(int, row["physical_support"])) for row in support_rows
    ]
    crossing_pairs = []
    for left_index, left in enumerate(supports):
        for right_index in range(left_index + 1, len(supports)):
            right = supports[right_index]
            if left & right and not (left <= right or right <= left):
                crossing_pairs.append([left_index, right_index])
    duplicates = len(supports) - len(set(supports))
    return {
        "support_count": len(supports),
        "duplicate_support_count": duplicates,
        "crossing_support_pairs": crossing_pairs,
        "is_laminar": not crossing_pairs and duplicates == 0,
    }


def apply_q_leq_projector(
    wavefunction: np.ndarray,
    support: Sequence[int],
    qmax: int,
) -> np.ndarray:
    """Apply ``sum_{q<=qmax} P_q`` on one physical support.

    The wavefunction uses the full computational basis with site zero as the
    most significant bit.  ``qmax`` and every spectral label are doubled
    spins.  The returned vector is not normalized.
    """

    state = np.asarray(wavefunction, dtype=np.complex128)
    n_sites = int(round(np.log2(state.size)))
    if 2**n_sites != state.size:
        raise ValueError("wavefunction length must be a power of two")
    subset = tuple(sorted(int(site) for site in support))
    if not subset or len(set(subset)) != len(subset):
        raise ValueError("support must contain distinct sites")
    if any(not 0 <= site < n_sites for site in subset):
        raise ValueError("support contains an out-of-range site")
    if (
        isinstance(qmax, bool)
        or not isinstance(qmax, (int, np.integer))
        or qmax < 0
    ):
        raise ValueError("qmax must be a non-negative doubled integer")
    labels, eigenvectors = subsystem_spin_eigenbasis(len(subset))
    selected = eigenvectors[:, labels <= int(qmax)]
    if selected.shape[1] == len(labels):
        return state.copy()
    remainder = tuple(site for site in range(n_sites) if site not in subset)
    order = subset + remainder
    inverse = tuple(int(value) for value in np.argsort(order))
    tensor = state.reshape((2,) * n_sites).transpose(order)
    matrix = tensor.reshape(2 ** len(subset), -1)
    projected = selected @ (selected.conj().T @ matrix)
    return projected.reshape((2,) * n_sites).transpose(inverse).reshape(-1)


def project_laminar_q_channels(
    wavefunction: np.ndarray,
    support_rows: Sequence[Mapping[str, object]],
    qmax: int,
    *,
    reverse: bool = False,
) -> np.ndarray:
    """Project every nontrivial support onto doubled spin ``q <= qmax``.

    ``reverse`` exists as an exact-small commutativity audit.  Forward and
    reverse answers must agree for a genuine laminar family.
    """

    if isinstance(qmax, bool) or not isinstance(qmax, (int, np.integer)) or qmax < 0:
        raise ValueError("qmax must be a non-negative doubled integer")
    checked_qmax = int(qmax)
    audit = laminar_family_audit(support_rows)
    if not audit["is_laminar"]:
        raise ValueError("channel projectors require a laminar support family")
    rows = list(reversed(support_rows)) if reverse else list(support_rows)
    projected = np.asarray(wavefunction, dtype=np.complex128)
    for row in rows:
        # The maximum doubled spin of n spin-halves is n.
        if int(row["support_size"]) > checked_qmax:
            projected = apply_q_leq_projector(
                projected, row["physical_support"], checked_qmax
            )
    return projected


def project_laminar_q_channels_by_node(
    wavefunction: np.ndarray,
    support_rows: Sequence[Mapping[str, object]],
    qmax_by_node: Mapping[int, int],
    *,
    reverse: bool = False,
) -> np.ndarray:
    """Project a laminar tree with one doubled-spin cutoff per node.

    Keys are the ``node_postorder_index`` values returned by
    :func:`physical_internal_supports`.  Every non-root internal node must be
    present exactly once.  A cutoff always retains an entire SU(2) multiplet;
    it never truncates magnetic projections within a fixed ``q`` sector.

    This exact-small routine is intentionally separate from trainable tensor
    layouts.  It defines the physical subspace selected by node-dependent
    intermediate-spin cutoffs without assuming any feature multiplicity.
    """

    audit = laminar_family_audit(support_rows)
    if not audit["is_laminar"]:
        raise ValueError("channel projectors require a laminar support family")
    expected = {int(row["node_postorder_index"]) for row in support_rows}
    observed = set(qmax_by_node)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"qmax_by_node must cover every support exactly; missing={missing}, extra={extra}"
        )
    checked: dict[int, int] = {}
    for index, raw in qmax_by_node.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, np.integer))
            or int(raw) < 0
        ):
            raise ValueError("every node cutoff must be a non-negative doubled integer")
        checked[int(index)] = int(raw)
    rows = list(reversed(support_rows)) if reverse else list(support_rows)
    projected = np.asarray(wavefunction, dtype=np.complex128)
    for row in rows:
        index = int(row["node_postorder_index"])
        if int(row["support_size"]) > checked[index]:
            projected = apply_q_leq_projector(
                projected, row["physical_support"], checked[index]
            )
    return projected


def tree_q_channel_cost_proxy(
    tree: FusionTree,
    qmax_by_node: Mapping[int, int],
    *,
    target_q: int = 0,
) -> dict[str, object]:
    """Return auditable reduced-channel and CG-work proxies for one tree.

    The proxy assigns unit multiplicity to every retained irrep.  Its packed
    channel count is the sum of retained parent irreps on non-root internal
    nodes.  Its CG work counts one coefficient for every child magnetic-state
    pair in every retained coupling triple,

    ``sum_(qL,qR,q) (qL+1)(qR+1)``.

    It is not a hardware FLOP measurement and deliberately excludes learned
    feature widths.  It is nevertheless exact and topology-aware for comparing
    node-specific ``q`` allocations on the same unit-multiplicity backbone.
    """

    if (
        isinstance(target_q, bool)
        or not isinstance(target_q, (int, np.integer))
        or int(target_q) < 0
    ):
        raise ValueError("target_q must be a non-negative doubled integer")
    nodes = tree.internal_nodes
    expected = set(range(max(len(nodes) - 1, 0)))
    observed = set(qmax_by_node)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"qmax_by_node must cover non-root nodes exactly; missing={missing}, extra={extra}"
        )
    checked: dict[int, int] = {}
    for index, raw in qmax_by_node.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, np.integer))
            or int(raw) < 0
        ):
            raise ValueError("every node cutoff must be a non-negative doubled integer")
        checked[int(index)] = int(raw)
    node_indices = {node: index for index, node in enumerate(nodes)}
    rows: list[dict[str, object]] = []

    def visit(node: TreeNode) -> tuple[int, ...]:
        if isinstance(node, Leaf):
            return (1,)
        left_values = visit(node.left)
        right_values = visit(node.right)
        index = node_indices[node]
        is_root = index == len(nodes) - 1
        cutoff = int(target_q) if is_root else checked[index]
        triples = []
        parents: set[int] = set()
        cg_coefficients = 0
        for q_left in left_values:
            for q_right in right_values:
                for q_parent in range(
                    abs(q_left - q_right), q_left + q_right + 1, 2
                ):
                    if q_parent > cutoff or (is_root and q_parent != int(target_q)):
                        continue
                    triples.append((q_left, q_right, q_parent))
                    parents.add(q_parent)
                    cg_coefficients += (q_left + 1) * (q_right + 1)
        if not parents:
            raise ValueError(
                f"node {index} has no reachable retained q sector under the cutoffs"
            )
        rows.append(
            {
                "node_postorder_index": index,
                "is_root": is_root,
                "qmax": cutoff,
                "retained_parent_q": sorted(parents),
                "retained_coupling_triples": [list(value) for value in triples],
                "reduced_coupling_count": len(triples),
                "cg_coefficient_count": cg_coefficients,
            }
        )
        return tuple(sorted(parents))

    root_values = visit(tree.root)
    if root_values != (int(target_q),):
        raise RuntimeError("tree cost recursion did not enforce the target irrep")
    nonroot = [row for row in rows if not row["is_root"]]
    return {
        "target_q": int(target_q),
        "unit_multiplicity_contract": True,
        "packed_reduced_channel_count": sum(
            len(row["retained_parent_q"]) for row in nonroot
        ),
        "packed_magnetic_state_equivalent": sum(
            sum(int(q) + 1 for q in row["retained_parent_q"])
            for row in nonroot
        ),
        "reduced_coupling_count": sum(
            int(row["reduced_coupling_count"]) for row in rows
        ),
        "cg_coefficient_flop_proxy": sum(
            int(row["cg_coefficient_count"]) for row in rows
        ),
        "nodes": sorted(rows, key=lambda row: int(row["node_postorder_index"])),
    }


__all__ = [
    "apply_q_leq_projector",
    "laminar_family_audit",
    "physical_internal_supports",
    "project_laminar_q_channels",
    "project_laminar_q_channels_by_node",
    "tree_q_channel_cost_proxy",
]
