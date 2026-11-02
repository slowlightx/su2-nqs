"""Separate physical local-q transfer from cell multiplicity rearrangement.

The decomposition is defined BEFORE space-group averaging. A lattice
projector need not commute with a chosen cell's Casimir, so a projected
q-preserving operator need not preserve q for that one fixed partition.
"""

from collections.abc import Callable, Sequence
from itertools import product
from numbers import Integral

import numpy as np

from .cell_spin_channels import PairAction, _cell, cell_spin_projector_action

ArrayAction = Callable[[np.ndarray], np.ndarray]


def cell_spin_preserving_action(
    state: np.ndarray,
    operator_action: ArrayAction,
    pair_action: PairAction,
    cells: Sequence[Sequence[int]],
) -> np.ndarray:
    """Apply ``sum_Q P_Q O P_Q``, Q being a tuple of doubled cell spins.

    The input operator must be a global scalar. Spectral pinching then
    retains precisely its matrix blocks preserving all specified cell q.
    It does not isolate multiplicity transitions from magnetic exchange.
    The complementary action ``O - pinching(O)`` has zero same-q blocks.
    """
    checked = tuple(_cell(c) for c in cells)
    if not checked:
        raise ValueError("at least one cell is required")
    all_sites = [site for c in checked for site in c]
    if len(set(all_sites)) != len(all_sites):
        raise ValueError("cells must be disjoint")
    labels = [tuple(range(len(c) % 2, len(c) + 1, 2)) for c in checked]
    vector = np.asarray(state, dtype=np.complex128)
    result = np.zeros_like(vector)
    for qs in product(*labels):
        selected = vector
        for c, q in zip(checked, qs, strict=True):
            selected = cell_spin_projector_action(selected, pair_action, c, q)
        selected = operator_action(selected)
        for c, q in zip(checked, qs, strict=True):
            selected = cell_spin_projector_action(selected, pair_action, c, q)
        result += selected
    return result


def four_site_dimer_action(
    state: np.ndarray, pair_action: PairAction, sites: Sequence[int], mode: int
) -> np.ndarray:
    """Apply a traceless reduced scalar to the multiplicities of a 4-spin cell.

    Write the three perfect-matching bond sums as D0,D1,D2, in site order
    (01)(23), (02)(13), (03)(12). Modes are (D0-D1)/sqrt(2) and
    (D0+D1-2D2)/sqrt(6). They commute with every cell spin component,
    have zero trace in each q block and act only on multiplicity indices.
    'Changing multiplicity' refers to nontrivial reduced action, not exclusively
    off-diagonal entries in any one gauge-dependent choice of a basis.
    """
    cell = _cell(sites)
    if len(cell) != 4:
        raise ValueError("a dimer mode requires four distinct sites")
    if isinstance(mode, bool) or not isinstance(mode, Integral) or mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    coefficients = (
        np.asarray((1., -1., 0.)) / np.sqrt(2.)
        if mode == 0 else np.asarray((1., 1., -2.)) / np.sqrt(6.)
    )
    vector = np.asarray(state, dtype=np.complex128)
    result = np.zeros_like(vector)
    for weight, pairs in zip(coefficients, (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))), strict=True):
        if weight:
            for i, j in pairs:
                result += weight * pair_action(vector, cell[i], cell[j])
    return result
