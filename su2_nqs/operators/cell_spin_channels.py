"""Local spin-channel transitions acting on physical carrier amplitudes.

Uniform cell modes are total-spin generators and cannot change the cell
irrep. Nonuniform modes are rank-one SU(2) tensors, not scalar functions of
magnetic configurations. Contracting two such modes gives a global scalar
which *can* change local doubled-spin labels ``q``. Spectator-cell Casimir
projectors make their coefficients conditional without breaking SU(2).
"""

from collections.abc import Callable, Sequence
from numbers import Integral

import numpy as np

from ..su2.quantum_numbers import validate_q

PairAction = Callable[[np.ndarray, int, int], np.ndarray]


def _cell(sites: Sequence[int]) -> tuple[int, ...]:
    if not sites or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in sites):
        raise ValueError("cell sites must be nonnegative integers")
    result = tuple(int(i) for i in sites)
    if len(set(result)) != len(result):
        raise ValueError("cell sites must be distinct")
    return result


def cell_casimir_action(state: np.ndarray, pair_action: PairAction, sites: Sequence[int]) -> np.ndarray:
    """Apply ``Q_cell^2`` to a vector or a batch of column vectors."""
    cell = _cell(sites)
    vector = np.asarray(state)
    result = 0.75 * len(cell) * vector
    for k, i in enumerate(cell):
        for j in cell[k + 1:]:
            result = result + 2.0 * pair_action(vector, i, j)
    return result


def cell_spin_projector_action(
    state: np.ndarray, pair_action: PairAction, sites: Sequence[int], q: int
) -> np.ndarray:
    """Apply the exact Casimir polynomial projector onto doubled spin ``q``.

    All multiplicity copies of the requested irrep are retained. The action
    needs only local pair exchanges; it never constructs a fusion basis.
    """
    cell = _cell(sites)
    validate_q(q)
    allowed = tuple(range(len(cell) % 2, len(cell) + 1, 2))
    if q not in allowed:
        raise ValueError("q is not an allowed doubled cell spin")
    eigenvalue = q * (q + 2) / 4.0
    result = np.asarray(state)
    for other in allowed:
        if other != q:
            other_value = other * (other + 2) / 4.0
            result = (cell_casimir_action(result, pair_action, cell) - other_value * result) / (eigenvalue - other_value)
    return result


def weighted_cell_exchange_action(
    state: np.ndarray,
    pair_action: PairAction,
    cell_a: Sequence[int],
    cell_b: Sequence[int],
    weights_a: Sequence[float],
    weights_b: Sequence[float],
) -> np.ndarray:
    """Apply ``(sum_i f_i S_i).(sum_j g_j S_j)`` between disjoint cells.

    Real form factors make the operator Hermitian. Nonuniform ``f,g`` permit
    transitions between cell irreps while total spin is conserved exactly.
    """
    a, b = _cell(cell_a), _cell(cell_b)
    if set(a) & set(b):
        raise ValueError("active cells must be disjoint")
    fa, fb = np.asarray(weights_a, dtype=float), np.asarray(weights_b, dtype=float)
    if fa.shape != (len(a),) or fb.shape != (len(b),) or not np.all(np.isfinite(fa)) or not np.all(np.isfinite(fb)):
        raise ValueError("form factors must match cells and be finite")
    vector = np.asarray(state)
    result = np.zeros_like(vector, dtype=np.result_type(vector, float))
    for i, f in zip(a, fa, strict=True):
        for j, g in zip(b, fb, strict=True):
            result = result + f * g * pair_action(vector, i, j)
    return result


def four_site_spin_modes() -> np.ndarray:
    """Orthonormal real form factors: uniform, two dipoles, checkerboard."""
    return np.asarray(((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)), dtype=float) / 2.0
