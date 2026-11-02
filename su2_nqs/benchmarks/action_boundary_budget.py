"""State-independent SU(2) channel budgets aligned with fusion-action cuts."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable, Mapping

import numpy as np

from .boundary_operator_channels import (
    BoundaryBond,
    compile_boundary_recoupling_plan,
    normalize_boundary_bonds,
    partition_boundary_bonds,
)


BondLike = (
    BoundaryBond
    | tuple[int, int]
    | tuple[int, int, float]
    | tuple[int, int, float, int]
)


def _capture_rank(singular_values: np.ndarray, fraction: float) -> int:
    if len(singular_values) == 0:
        return 0
    squared = singular_values**2
    cumulative = np.cumsum(squared) / np.sum(squared)
    return int(np.searchsorted(cumulative, fraction, side="left") + 1)


def _validate_cut(n_sites: int, prefix_length: int) -> tuple[int, int]:
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, Integral):
        raise TypeError("prefix_length must be an integer")
    if not 0 <= prefix_length <= n_sites - 1:
        raise ValueError("prefix_length lies outside the compact action sequence")
    return int(n_sites), int(prefix_length)


def _validate_shells(
    shell_bonds: Mapping[str, Iterable[BondLike]],
    shell_weights: Mapping[str, float],
) -> tuple[tuple[str, tuple[BoundaryBond, ...], float], ...]:
    if not shell_bonds:
        raise ValueError("shell_bonds must not be empty")
    if set(shell_bonds) != set(shell_weights):
        raise ValueError("shell_bonds and shell_weights must have identical keys")
    rows = []
    for name, bonds in shell_bonds.items():
        if not isinstance(name, str) or not name:
            raise ValueError("shell names must be nonempty strings")
        weight = float(shell_weights[name])
        if not math.isfinite(weight):
            raise ValueError("shell weights must be finite")
        normalized = normalize_boundary_bonds(bonds)
        rows.append((name, normalized, weight))
    return tuple(rows)


def fusion_action_boundary_budget(
    n_sites: int,
    shell_bonds: Mapping[str, Iterable[BondLike]],
    shell_weights: Mapping[str, float],
    prefix_length: int,
    *,
    rank_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Return exact vector/scalar channel counts after an action prefix.

    ``prefix_length=0`` is the pre-decision cut with all bonds hidden.
    For positive prefix length, compact action ``d-1`` has just been revealed,
    so it maps to :func:`compile_boundary_recoupling_plan` decision ``d-1``.
    The final compact action resolves the last two physical leaves together.
    """

    n_sites, prefix_length = _validate_cut(n_sites, prefix_length)
    if not math.isfinite(rank_tolerance) or rank_tolerance < 0.0:
        raise ValueError("rank_tolerance must be finite and non-negative")
    shells = _validate_shells(shell_bonds, shell_weights)
    active = tuple(row for row in shells if row[2] != 0.0)

    for _, bonds, _ in shells:
        for bond in bonds:
            if not (0 <= bond.first < n_sites and 0 <= bond.second < n_sites):
                raise ValueError("bond endpoint lies outside the system")

    if prefix_length == 0:
        hidden_by_shell = {
            name: len(bonds) if weight != 0.0 else 0
            for name, bonds, weight in shells
        }
        return {
            "prefix_length": 0,
            "decision": None,
            "hidden_site_count": n_sites,
            "prefix_site_count": 0,
            "combined_vector_rank": 0,
            "shell_vector_ranks": {name: 0 for name, _, _ in shells},
            "shell_separated_vector_rank": 0,
            "endpoint_gauge_channel_count": 0,
            "ordinary_magnetic_rank": 0,
            "crossing_bond_count": 0,
            "crossing_bond_count_by_shell": {name: 0 for name, _, _ in shells},
            "hidden_scalar_bond_count": int(sum(hidden_by_shell.values())),
            "hidden_scalar_bond_count_by_shell": hidden_by_shell,
            "combined_hidden_scalar_channel_count": int(
                any(count > 0 for count in hidden_by_shell.values())
            ),
            "shell_resolved_hidden_scalar_channel_count": sum(
                count > 0 for count in hidden_by_shell.values()
            ),
            "combined_singular_values": [],
            "combined_rank_r90": 0,
            "combined_rank_r99": 0,
            "combined_rank_r999": 0,
        }

    decision = prefix_length - 1
    combined_bonds = tuple(
        BoundaryBond(
            bond.first,
            bond.second,
            weight * bond.coupling,
            shell_index,
        )
        for shell_index, (_, bonds, weight) in enumerate(shells)
        if weight != 0.0
        for bond in bonds
    )
    combined = compile_boundary_recoupling_plan(
        n_sites,
        combined_bonds,
        decision,
        rank_tolerance=rank_tolerance,
    )
    shell_ranks: dict[str, int] = {}
    crossing_by_shell: dict[str, int] = {}
    hidden_by_shell: dict[str, int] = {}
    for name, bonds, weight in shells:
        if weight == 0.0:
            shell_ranks[name] = 0
            crossing_by_shell[name] = 0
            hidden_by_shell[name] = 0
            continue
        weighted = tuple(
            BoundaryBond(
                bond.first, bond.second, weight * bond.coupling, bond.channel
            )
            for bond in bonds
        )
        plan = compile_boundary_recoupling_plan(
            n_sites, weighted, decision, rank_tolerance=rank_tolerance
        )
        parts = partition_boundary_bonds(n_sites, weighted, decision)
        shell_ranks[name] = plan.channel_rank
        crossing_by_shell[name] = len(parts["crossing"])
        hidden_by_shell[name] = len(parts["hidden"])

    singular = np.asarray(combined.singular_values, dtype=np.float64)
    return {
        "prefix_length": prefix_length,
        "decision": decision,
        "hidden_site_count": len(combined.hidden_sites),
        "prefix_site_count": len(combined.prefix_sites),
        "combined_vector_rank": combined.channel_rank,
        "shell_vector_ranks": shell_ranks,
        "shell_separated_vector_rank": int(sum(shell_ranks.values())),
        "endpoint_gauge_channel_count": combined.endpoint_channel_count,
        "ordinary_magnetic_rank": combined.ordinary_magnetic_schmidt_rank,
        "crossing_bond_count": int(sum(crossing_by_shell.values())),
        "crossing_bond_count_by_shell": crossing_by_shell,
        "hidden_scalar_bond_count": int(sum(hidden_by_shell.values())),
        "hidden_scalar_bond_count_by_shell": hidden_by_shell,
        "combined_hidden_scalar_channel_count": combined.hidden_scalar_channel_count,
        "shell_resolved_hidden_scalar_channel_count": (
            combined.shell_resolved_hidden_scalar_channel_count
        ),
        "combined_singular_values": [float(value) for value in singular],
        "combined_rank_r90": _capture_rank(singular, 0.9),
        "combined_rank_r99": _capture_rank(singular, 0.99),
        "combined_rank_r999": _capture_rank(singular, 0.999),
    }


def fusion_action_boundary_budgets(
    n_sites: int,
    shell_bonds: Mapping[str, Iterable[BondLike]],
    shell_weights: Mapping[str, float],
    *,
    rank_tolerance: float = 1.0e-12,
) -> tuple[dict[str, object], ...]:
    """Return state-independent budgets at all compact action cuts."""

    return tuple(
        fusion_action_boundary_budget(
            n_sites,
            shell_bonds,
            shell_weights,
            prefix_length,
            rank_tolerance=rank_tolerance,
        )
        for prefix_length in range(n_sites)
    )


__all__ = [
    "fusion_action_boundary_budget",
    "fusion_action_boundary_budgets",
]
