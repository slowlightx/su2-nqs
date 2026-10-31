"""Exact-small utilities for physical multiplet-flux channel allocation.

The functions in this module know about binary fusion trees, physical cut
boundaries, and complete SU(2) multiplets.  They contain no Hamiltonian
construction, optimizer, filesystem I/O, or exact-state selection logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np

from ..su2.fusion_tree import FusionTree
from .tree_spin_channels import tree_q_channel_cost_proxy


def weighted_boundary_score(
    support: Sequence[int],
    bonds: Sequence[Sequence[float | int]],
) -> float:
    """Return ``sum |J_ij|`` over Hamiltonian bonds crossing a support.

    Each bond row must begin with ``(i, j, J_ij)``.  Absolute couplings make
    this a geometry/workload feature rather than a cancellation-sensitive
    energy estimator.
    """

    selected = set(map(int, support))
    if not selected:
        raise ValueError("support must be non-empty")
    total = 0.0
    for raw in bonds:
        if len(raw) < 3:
            raise ValueError("bond rows must contain i, j, and coupling")
        first, second = int(raw[0]), int(raw[1])
        coupling = float(raw[2])
        if not np.isfinite(coupling):
            raise ValueError("bond couplings must be finite")
        if (first in selected) != (second in selected):
            total += abs(coupling)
    return float(total)


def cutoff_options_by_node(
    support_rows: Sequence[Mapping[str, object]],
    *,
    baseline_qmax: int = 2,
) -> dict[int, tuple[int, ...]]:
    """Return all complete-multiplet cutoffs above a uniform baseline.

    For ``n`` spin-half leaves, allowed doubled spins have parity ``n`` and
    run to ``n``.  The first option is the largest physical label not exceeding
    ``baseline_qmax``; subsequent options add one full spin unit (``Delta q=2``)
    until no truncation remains.
    """

    if (
        isinstance(baseline_qmax, bool)
        or not isinstance(baseline_qmax, (int, np.integer))
        or int(baseline_qmax) < 0
    ):
        raise ValueError("baseline_qmax must be a non-negative doubled integer")
    output: dict[int, tuple[int, ...]] = {}
    for row in support_rows:
        index = int(row["node_postorder_index"])
        size = int(row["support_size"])
        if size < 2:
            raise ValueError("only non-root internal supports are accepted")
        allowed = tuple(range(size % 2, size + 1, 2))
        baseline = max(value for value in allowed if value <= int(baseline_qmax))
        output[index] = tuple(value for value in allowed if value >= baseline)
    if len(output) != len(support_rows):
        raise ValueError("node_postorder_index values must be unique")
    return output


def uniform_q_cutoffs(
    support_rows: Sequence[Mapping[str, object]], qmax: int
) -> dict[int, int]:
    """Convert one global doubled cutoff into physical per-node labels."""

    if (
        isinstance(qmax, bool)
        or not isinstance(qmax, (int, np.integer))
        or int(qmax) < 0
    ):
        raise ValueError("qmax must be a non-negative doubled integer")
    result = {}
    for row in support_rows:
        size = int(row["support_size"])
        allowed = tuple(range(size % 2, size + 1, 2))
        kept = [value for value in allowed if value <= int(qmax)]
        if not kept:
            raise ValueError(f"global qmax={qmax} removes every q sector at node")
        result[int(row["node_postorder_index"])] = max(kept)
    return result


def boundary_allocation_loss_table(
    support_rows: Sequence[Mapping[str, object]],
    options_by_node: Mapping[int, Sequence[int]],
    bonds: Sequence[Sequence[float | int]],
) -> dict[int, dict[int, float]]:
    """Build a Hamiltonian-only loss table for node-wise channel allocation.

    The loss is boundary weight times the unresolved fraction of the physical
    spin range.  It is a deliberately simple blind control, not a fitted
    observable model.
    """

    output: dict[int, dict[int, float]] = {}
    for row in support_rows:
        index = int(row["node_postorder_index"])
        options = tuple(map(int, options_by_node[index]))
        highest = max(options)
        lowest = min(options)
        boundary = weighted_boundary_score(row["physical_support"], bonds)
        denominator = max(highest - lowest, 1)
        output[index] = {
            cutoff: boundary * (highest - cutoff) / denominator
            for cutoff in options
        }
    return output


def fcs_allocation_loss_table(
    support_rows: Sequence[Mapping[str, object]],
    options_by_node: Mapping[int, Sequence[int]],
    distributions_by_node: Mapping[int, Mapping[int, float]],
) -> dict[int, dict[int, float]]:
    """Return the sum-of-single-cut discarded-weight surrogate by node.

    Distributions must be physical subsystem total-spin FCS, not latent tree
    channel weights.  The resulting additive loss selects an allocation; the
    actual joint retained norm still has to be evaluated with the commuting
    laminar projector product.
    """

    expected = {int(row["node_postorder_index"]) for row in support_rows}
    if set(distributions_by_node) != expected:
        raise ValueError("one physical FCS distribution is required per node")
    output: dict[int, dict[int, float]] = {}
    for index in sorted(expected):
        distribution = {
            int(q): float(probability)
            for q, probability in distributions_by_node[index].items()
        }
        if any(value < -1.0e-13 for value in distribution.values()):
            raise ValueError("FCS probabilities must be non-negative")
        if abs(sum(distribution.values()) - 1.0) > 2.0e-10:
            raise ValueError("FCS probabilities must sum to one")
        output[index] = {
            int(cutoff): float(
                sum(value for q, value in distribution.items() if q > int(cutoff))
            )
            for cutoff in options_by_node[index]
        }
    return output


def select_cutoffs_under_budget(
    tree: FusionTree,
    options_by_node: Mapping[int, Sequence[int]],
    loss_by_node_and_cutoff: Mapping[int, Mapping[int, float]],
    *,
    maximum_cg_flop_proxy: int,
) -> dict[str, object]:
    """Exhaustively minimize an additive loss under an exact q-only cost.

    This routine is intended for exact-small catalogs.  Ties are resolved by
    larger utilized CG cost and then lexicographically smaller cutoff tuples,
    making selection deterministic and independent of dictionary order.
    """

    if maximum_cg_flop_proxy < 1:
        raise ValueError("maximum_cg_flop_proxy must be positive")
    indices = tuple(sorted(options_by_node))
    if set(loss_by_node_and_cutoff) != set(indices):
        raise ValueError("loss and cutoff tables must contain the same nodes")
    choices = []
    for index in indices:
        node_options = tuple(sorted(set(map(int, options_by_node[index]))))
        if not node_options:
            raise ValueError("every node needs at least one cutoff option")
        if set(loss_by_node_and_cutoff[index]) != set(node_options):
            raise ValueError("every cutoff option needs a registered loss")
        choices.append(node_options)
    best: tuple[float, int, tuple[int, ...], dict[str, object]] | None = None
    feasible = 0
    for values in product(*choices):
        allocation = dict(zip(indices, values, strict=True))
        cost = tree_q_channel_cost_proxy(tree, allocation, target_q=0)
        cg_cost = int(cost["cg_coefficient_flop_proxy"])
        if cg_cost > maximum_cg_flop_proxy:
            continue
        feasible += 1
        loss = float(
            sum(loss_by_node_and_cutoff[index][cutoff]
                for index, cutoff in allocation.items())
        )
        rank = (loss, -cg_cost, values, cost)
        if best is None or rank[:3] < best[:3]:
            best = rank
    if best is None:
        raise ValueError("no cutoff allocation satisfies the registered budget")
    loss, negative_cost, values, cost = best
    return {
        "qmax_by_node": {
            str(index): int(value)
            for index, value in zip(indices, values, strict=True)
        },
        "selection_surrogate_loss": loss,
        "maximum_cg_flop_proxy": int(maximum_cg_flop_proxy),
        "used_cg_flop_proxy": -negative_cost,
        "feasible_allocation_count": feasible,
        "cost": cost,
    }


__all__ = [
    "boundary_allocation_loss_table",
    "cutoff_options_by_node",
    "fcs_allocation_loss_table",
    "select_cutoffs_under_budget",
    "uniform_q_cutoffs",
    "weighted_boundary_score",
]
