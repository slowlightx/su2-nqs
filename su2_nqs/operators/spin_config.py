"""On-demand Heisenberg connectivity on physical spin configurations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


SpinBond = tuple[int, int] | tuple[int, int, float]


def _checked_spin_bonds(
    n_sites: int, bonds: Iterable[SpinBond]
) -> tuple[tuple[int, int, float], ...]:
    checked = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid spin bond: {bond!r}")
        first, second = int(first), int(second)
        if first == second or not (
            0 <= first < n_sites and 0 <= second < n_sites
        ):
            raise ValueError(f"invalid spin bond: {bond!r}")
        checked.append((first, second, float(coupling)))
    return tuple(checked)


def heisenberg_spin_connectivity(
    spins: Sequence[int],
    bonds: Iterable[SpinBond],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Return combined physical states and ``H[new, old]`` elements.

    Bits follow the repository convention ``0=down`` and ``1=up``.  Repeated
    graph edges are combined before returning, so the routine remains correct
    for small periodic cells whose lattice generator merges multiple images.
    """

    old = tuple(int(value) for value in spins)
    if not old or any(value not in (0, 1) for value in old):
        raise ValueError("spins must be a nonempty sequence of zero/one bits")
    diagonal = 0.0
    elements: dict[tuple[int, ...], float] = {}
    for first, second, coupling in _checked_spin_bonds(len(old), bonds):
        if old[first] == old[second]:
            diagonal += 0.25 * coupling
            continue
        diagonal -= 0.25 * coupling
        flipped = list(old)
        flipped[first], flipped[second] = flipped[second], flipped[first]
        new = tuple(flipped)
        elements[new] = elements.get(new, 0.0) + 0.5 * coupling
    elements[old] = elements.get(old, 0.0) + diagonal
    states = tuple(sorted(elements))
    values = np.asarray([elements[state] for state in states], dtype=np.float64)
    return states, values


__all__ = ["SpinBond", "heisenberg_spin_connectivity"]
