"""Target-independent graph locality scores for deterministic binary trees."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations


Bond = tuple[int, int]


def balanced_tree_leaf_addresses(n_sites: int) -> tuple[tuple[int, ...], ...]:
    """Return binary root-to-leaf addresses for the model's balanced tree."""

    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    addresses: list[tuple[int, ...] | None] = [None] * n_sites

    def visit(start: int, stop: int, prefix: tuple[int, ...]) -> None:
        if stop - start == 1:
            addresses[start] = prefix
            return
        middle = start + (stop - start) // 2
        visit(start, middle, prefix + (0,))
        visit(middle, stop, prefix + (1,))

    visit(0, n_sites, ())
    return tuple(address for address in addresses if address is not None)


def balanced_tree_distance(n_sites: int, first_position: int, second_position: int) -> int:
    """Count tree edges between two leaf positions."""

    if not 0 <= first_position < n_sites or not 0 <= second_position < n_sites:
        raise ValueError("leaf position is outside the tree")
    addresses = balanced_tree_leaf_addresses(n_sites)
    first = addresses[first_position]
    second = addresses[second_position]
    shared = 0
    while (
        shared < min(len(first), len(second))
        and first[shared] == second[shared]
    ):
        shared += 1
    return len(first) + len(second) - 2 * shared


def tree_hopping_distance_score(
    n_sites: int,
    bonds: Sequence[Bond],
    site_order: Sequence[int],
) -> tuple[int, int, int]:
    """Return sum, maximum and squared-sum hopping distances on the tree."""

    order = tuple(site_order)
    if tuple(sorted(order)) != tuple(range(n_sites)):
        raise ValueError("site_order must be a permutation of all sites")
    positions = {site: position for position, site in enumerate(order)}
    distances = tuple(
        balanced_tree_distance(n_sites, positions[first], positions[second])
        for first, second in bonds
    )
    if not distances:
        return 0, 0, 0
    return sum(distances), max(distances), sum(value * value for value in distances)


def joint_minimax_tree_order(
    n_sites: int,
    bond_sets: Sequence[Sequence[Bond]],
) -> tuple[int, ...]:
    """Select a target-blind order by a frozen cross-geometry minimax rule.

    The ordering key is maximum total distance across geometries, then the
    sum of those totals, then the tuple of per-geometry totals, and finally
    lexicographic site order.  Exhaustive permutation search is intentionally
    restricted to exact-small systems.
    """

    if n_sites > 8:
        raise ValueError("exhaustive locality ordering is restricted to n_sites <= 8")
    if not bond_sets:
        raise ValueError("at least one bond set is required")
    candidates = []
    for order in permutations(range(n_sites)):
        totals = tuple(
            tree_hopping_distance_score(n_sites, bonds, order)[0]
            for bonds in bond_sets
        )
        candidates.append((max(totals), sum(totals), totals, order))
    return min(candidates)[-1]


__all__ = [
    "Bond",
    "balanced_tree_distance",
    "balanced_tree_leaf_addresses",
    "joint_minimax_tree_order",
    "tree_hopping_distance_score",
]
