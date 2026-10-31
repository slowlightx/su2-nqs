"""Physical-amplitude form of a local SU(2) simplex recoupling operator.

The scalable object in this module is an *operator*, not a change of basis for
the variational state.  A carrier may therefore stay in the computational
basis (for example a projected parton determinant or a Krylov-improved
state), while the recoupling physics is applied as a local scalar action.

For three disjoint cells ``A,B,C`` define

``X_AB = Q_A . Q_B = sum_(i in A,j in B) S_i . S_j``.

In ``((q_A q_B)q_AB,q_C)Q`` coordinates, ``X_AB`` is diagonal and ``X_BC``
is ``F.T @ D_BC @ F``.  The Hermitian Jordan product contains a lower-body
piece because for spin one half

``{h_ij,h_jk}/2 = h_ik/4``.

Removing that piece gives the genuinely four-spin resonance

``R_ABC = {X_AB,X_BC}/2 - |B| X_AC/4``.

Equivalently, ``R_ABC`` is the sum of ``h_ij h_kl`` over terms whose two
bonds use *different* sites ``j != k`` in the middle cell.  This identity is
the bridge between the reduced F-move picture and direct physical-amplitude
evaluation.  No global fusion coefficient vector is constructed here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..su2.quantum_numbers import is_admissible_triangle, validate_q
from ..su2.wigner6j import f_move


PairAction = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class SimplexRecouplingBlock:
    """One fixed-``(q_A,q_B,q_C,Q)`` reduced recoupling block.

    All angular momenta are doubled integers.  Matrices use the left-coupled
    ``q_AB`` ordering recorded in :attr:`left_channels`.
    """

    left_channels: tuple[int, ...]
    right_channels: tuple[int, ...]
    f_matrix: np.ndarray
    x_ab: np.ndarray
    x_bc: np.ndarray
    x_ac: np.ndarray
    connected_resonance: np.ndarray


def composite_exchange_eigenvalue(q_left: int, q_right: int, q_pair: int) -> float:
    """Return the eigenvalue of ``Q_left . Q_right`` for doubled labels."""

    for name, value in (
        ("q_left", q_left),
        ("q_right", q_right),
        ("q_pair", q_pair),
    ):
        validate_q(value, name=name)
    if not is_admissible_triangle(q_left, q_right, q_pair):
        raise ValueError("q_pair is not an admissible coupling channel")
    # j(j+1)=q(q+2)/4 and J_l.J_r=(J_pair^2-J_l^2-J_r^2)/2.
    return float(
        (
            q_pair * (q_pair + 2)
            - q_left * (q_left + 2)
            - q_right * (q_right + 2)
        )
        / 8.0
    )


def simplex_recoupling_block(
    q_a: int,
    q_b: int,
    q_c: int,
    q_total: int,
    *,
    middle_site_count: int,
) -> SimplexRecouplingBlock:
    """Construct the F-derived connected resonance in the left basis.

    ``middle_site_count`` counts spin-half sites in cell ``B``.  It appears
    only in the exact subtraction of shared-site two-body terms.
    """

    for name, value in (
        ("q_a", q_a),
        ("q_b", q_b),
        ("q_c", q_c),
        ("q_total", q_total),
    ):
        validate_q(value, name=name)
    if (
        isinstance(middle_site_count, bool)
        or not isinstance(middle_site_count, Integral)
        or middle_site_count < 1
    ):
        raise ValueError("middle_site_count must be a positive integer")

    f_matrix, left, right = f_move(q_a, q_b, q_c, q_total)
    x_ab_values = np.asarray(
        [composite_exchange_eigenvalue(q_a, q_b, q_ab) for q_ab in left],
        dtype=np.float64,
    )
    x_bc_values = np.asarray(
        [composite_exchange_eigenvalue(q_b, q_c, q_bc) for q_bc in right],
        dtype=np.float64,
    )
    x_ab = np.diag(x_ab_values)
    # Repository convention: F[q_BC,q_AB]=<right|left>.
    x_bc = f_matrix.T @ np.diag(x_bc_values) @ f_matrix
    total_constant = float(
        (
            q_total * (q_total + 2)
            - q_a * (q_a + 2)
            - q_b * (q_b + 2)
            - q_c * (q_c + 2)
        )
        / 8.0
    )
    identity = np.eye(len(left), dtype=np.float64)
    x_ac = total_constant * identity - x_ab - x_bc
    jordan = 0.5 * (x_ab @ x_bc + x_bc @ x_ab)
    connected = jordan - float(middle_site_count) * x_ac / 4.0
    return SimplexRecouplingBlock(
        left_channels=left,
        right_channels=right,
        f_matrix=f_matrix,
        x_ab=x_ab,
        x_bc=x_bc,
        x_ac=x_ac,
        connected_resonance=0.5 * (connected + connected.T),
    )


def _checked_cell(
    sites: Sequence[int], *, name: str, n_sites: int | None = None
) -> tuple[int, ...]:
    checked = tuple(int(site) for site in sites)
    if not checked or len(set(checked)) != len(checked) or min(checked) < 0:
        raise ValueError(f"{name} must contain distinct non-negative sites")
    if n_sites is not None and max(checked) >= n_sites:
        raise ValueError(f"{name} contains a site outside the state")
    return checked


def validate_simplex_cells(
    cell_a: Sequence[int],
    cell_b: Sequence[int],
    cell_c: Sequence[int],
    *,
    n_sites: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate three nonempty, pairwise-disjoint physical cells."""

    cells = tuple(
        _checked_cell(cell, name=name, n_sites=n_sites)
        for cell, name in zip(
            (cell_a, cell_b, cell_c), ("cell_a", "cell_b", "cell_c"), strict=True
        )
    )
    if any(
        set(cells[first]) & set(cells[second])
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("simplex cells must be pairwise disjoint")
    return cells[0], cells[1], cells[2]


def cell_exchange_action(
    state: np.ndarray,
    pair_action: PairAction,
    cell_left: Sequence[int],
    cell_right: Sequence[int],
) -> np.ndarray:
    """Apply ``Q_left . Q_right`` without changing state representation."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    left = _checked_cell(cell_left, name="cell_left")
    right = _checked_cell(cell_right, name="cell_right")
    if set(left) & set(right):
        raise ValueError("the two cells must be disjoint")
    answer = np.zeros_like(vector)
    for first in left:
        for second in right:
            contribution = np.asarray(
                pair_action(vector, first, second), dtype=np.complex128
            )
            if contribution.shape != vector.shape:
                raise ValueError("pair_action returned the wrong state shape")
            answer += contribution
    return answer


def simplex_connected_f_resonance_action(
    state: np.ndarray,
    pair_action: PairAction,
    cell_a: Sequence[int],
    cell_b: Sequence[int],
    cell_c: Sequence[int],
) -> np.ndarray:
    """Apply the connected four-spin part of the simplex Jordan product.

    This factorized implementation costs two cell-exchange actions plus one
    lower-body subtraction.  It is algebraically identical to explicitly
    summing ``h_ij h_kl`` over ``j != k`` in the middle cell.
    """

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    a, b, c = validate_simplex_cells(cell_a, cell_b, cell_c)
    x_bc = cell_exchange_action(vector, pair_action, b, c)
    forward = cell_exchange_action(x_bc, pair_action, a, b)
    x_ab = cell_exchange_action(vector, pair_action, a, b)
    reverse = cell_exchange_action(x_ab, pair_action, b, c)
    lower_body = cell_exchange_action(vector, pair_action, a, c)
    return 0.5 * (forward + reverse) - len(b) * lower_body / 4.0


def simplex_disjoint_four_spin_action(
    state: np.ndarray,
    pair_action: PairAction,
    cell_a: Sequence[int],
    cell_b: Sequence[int],
    cell_c: Sequence[int],
) -> np.ndarray:
    """Reference sum of disjoint ``h_ij h_kl`` terms for exact tests."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    a, b, c = validate_simplex_cells(cell_a, cell_b, cell_c)
    answer = np.zeros_like(vector)
    for first in a:
        for middle_left in b:
            for middle_right in b:
                if middle_left == middle_right:
                    continue
                for last in c:
                    image = np.asarray(
                        pair_action(vector, middle_right, last),
                        dtype=np.complex128,
                    )
                    image = np.asarray(
                        pair_action(image, first, middle_left),
                        dtype=np.complex128,
                    )
                    if image.shape != vector.shape:
                        raise ValueError("pair_action returned the wrong state shape")
                    answer += image
    return answer


__all__ = [
    "PairAction",
    "SimplexRecouplingBlock",
    "cell_exchange_action",
    "composite_exchange_eigenvalue",
    "simplex_connected_f_resonance_action",
    "simplex_disjoint_four_spin_action",
    "simplex_recoupling_block",
    "validate_simplex_cells",
]
