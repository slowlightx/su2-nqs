"""Wigner-6j symbols and SU(2) F-matrices for doubled integer spins."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .quantum_numbers import coupling_range, is_admissible_triangle, validate_q


def _delta(qa: int, qb: int, qc: int) -> float:
    if not is_admissible_triangle(qa, qb, qc):
        return 0.0
    x = (qa + qb - qc) // 2
    y = (qa - qb + qc) // 2
    z = (-qa + qb + qc) // 2
    denominator = (qa + qb + qc) // 2 + 1
    return math.sqrt(
        math.factorial(x)
        * math.factorial(y)
        * math.factorial(z)
        / math.factorial(denominator)
    )


@lru_cache(maxsize=None)
def wigner_6j(qa: int, qb: int, qc: int, qd: int, qe: int, qf: int) -> float:
    """Evaluate ``{a b c; d e f}`` by the Racah factorial formula."""

    for name, value in zip("abcdef", (qa, qb, qc, qd, qe, qf)):
        validate_q(value, name=f"q{name}")
    triangles = (
        (qa, qb, qc),
        (qa, qe, qf),
        (qd, qb, qf),
        (qd, qe, qc),
    )
    if not all(is_admissible_triangle(*triangle) for triangle in triangles):
        return 0.0

    lower = (
        (qa + qb + qc) // 2,
        (qa + qe + qf) // 2,
        (qd + qb + qf) // 2,
        (qd + qe + qc) // 2,
    )
    upper = (
        (qa + qb + qd + qe) // 2,
        (qa + qc + qd + qf) // 2,
        (qb + qc + qe + qf) // 2,
    )
    z_min, z_max = max(lower), min(upper)
    series = 0.0
    for z in range(z_min, z_max + 1):
        denominator = 1
        for value in lower:
            denominator *= math.factorial(z - value)
        for value in upper:
            denominator *= math.factorial(value - z)
        term = math.factorial(z + 1) / denominator
        series += -term if z % 2 else term

    prefactor = 1.0
    for triangle in triangles:
        prefactor *= _delta(*triangle)
    return prefactor * series


def f_move(
    q1: int,
    q2: int,
    q3: int,
    q_total: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Recouple ``((q1,q2)q12,q3)q`` to ``(q1,(q2,q3)q23)q``.

    The returned matrix has right-associated channels ``q23`` as rows and
    left-associated channels ``q12`` as columns:

    ``F[q23, q12] = <q1,(q2,q3)q23;q | (q1,q2)q12,q3;q>``.
    """

    for name, value in (("q1", q1), ("q2", q2), ("q3", q3), ("q_total", q_total)):
        validate_q(value, name=name)
    left = tuple(
        q12
        for q12 in coupling_range(q1, q2)
        if is_admissible_triangle(q12, q3, q_total)
    )
    right = tuple(
        q23
        for q23 in coupling_range(q2, q3)
        if is_admissible_triangle(q1, q23, q_total)
    )
    if not left or len(left) != len(right):
        raise ValueError(
            f"no compatible recoupling sector for ({q1}, {q2}, {q3}) -> {q_total}"
        )
    exponent_numerator = q1 + q2 + q3 + q_total
    if exponent_numerator % 2:
        raise ValueError("the F-move phase exponent is not integral")
    phase = -1.0 if (exponent_numerator // 2) % 2 else 1.0
    matrix = np.empty((len(right), len(left)), dtype=np.float64)
    for row, q23 in enumerate(right):
        for column, q12 in enumerate(left):
            matrix[row, column] = (
                phase
                * math.sqrt((q12 + 1) * (q23 + 1))
                * wigner_6j(q1, q2, q12, q3, q_total, q23)
            )
    return matrix, left, right


def pair_heisenberg_block(q_left: int, q_total: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Matrix of a neighboring spin-1/2 pair in a left-associated sector."""

    matrix, left_channels, pair_channels = f_move(q_left, 1, 1, q_total)
    eigenvalues = np.asarray([-0.75 if q_pair == 0 else 0.25 for q_pair in pair_channels])
    return matrix.T @ np.diag(eigenvalues) @ matrix, left_channels

