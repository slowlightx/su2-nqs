"""The real :math:`E_2` orientation irrep of the triangular point group.

The three columns refer to the unoriented nearest-neighbour axes in
``J1_POSITIVE`` order.  The uniform ``A1`` direction is removed exactly, so
the two rows form an orthonormal, convention-fixed basis for bond nematicity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .triangular_hnf import (
    IntMatrix,
    IntVector,
    J1_POSITIVE,
    J2_POSITIVE,
    matrix_vector,
)


E2_ORIENTATION_FRAME = np.asarray(
    (
        (2.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0)),
        (0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)),
    ),
    dtype=np.float64,
)

# The J2 axes are rotated by 30 physical degrees relative to J1.  An E2
# object therefore rotates by 60 degrees.  This alignment makes the J1 and J2
# doublets transform under exactly the same matrices, so their scalar product
# is an unambiguous A1 cross-shell carrier.
_ROTATE_E2_60 = np.asarray(
    ((0.5, -np.sqrt(3.0) / 2.0), (np.sqrt(3.0) / 2.0, 0.5)),
    dtype=np.float64,
)
J2_E2_ORIENTATION_FRAME = _ROTATE_E2_60 @ E2_ORIENTATION_FRAME


def _canonical_axis(vector: IntVector) -> IntVector:
    """Choose one sign for an unoriented primitive-lattice axis."""

    first, second = (int(vector[0]), int(vector[1]))
    if first == 0 and second == 0:
        raise ValueError("an orientation axis must be nonzero")
    return (first, second) if (first > 0 or (first == 0 and second > 0)) else (-first, -second)


def orientation_axis_permutation(
    operation: IntMatrix,
    axes: Sequence[IntVector] = J1_POSITIVE,
) -> tuple[int, ...]:
    """Return the active old-axis to new-axis permutation.

    Bond operators are unoriented, hence ``delta`` and ``-delta`` identify the
    same axis.  The supplied axes must be closed under ``operation`` modulo
    this sign.
    """

    checked = tuple(_canonical_axis(axis) for axis in axes)
    if len(set(checked)) != len(checked) or not checked:
        raise ValueError("axes must be distinct nonzero unoriented directions")
    lookup = {axis: index for index, axis in enumerate(checked)}
    answer = []
    for axis in axes:
        transformed = _canonical_axis(matrix_vector(operation, axis))
        if transformed not in lookup:
            raise ValueError("operation does not preserve the supplied axes")
        answer.append(lookup[transformed])
    if tuple(sorted(answer)) != tuple(range(len(checked))):
        raise RuntimeError("orientation action is not a permutation")
    return tuple(answer)


def e2_orientation_representation(
    operation: IntMatrix,
    *,
    axes: Sequence[IntVector] = J1_POSITIVE,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    """Return the real two-dimensional matrix acting on ``E2`` components."""

    checked_frame = (
        E2_ORIENTATION_FRAME
        if frame is None and tuple(axes) == J1_POSITIVE
        else J2_E2_ORIENTATION_FRAME
        if frame is None and tuple(axes) == J2_POSITIVE
        else np.asarray(frame, dtype=np.float64)
    )
    if checked_frame.shape != (2, 3):
        raise ValueError("orientation frame must have shape (2,3)")
    mapping = orientation_axis_permutation(operation, axes)
    permutation = np.zeros((3, 3), dtype=np.float64)
    for old, new in enumerate(mapping):
        permutation[new, old] = 1.0
    # If B_delta -> B_{g delta}, E B -> E P.T B.
    return checked_frame @ permutation.T @ checked_frame.T


def e2_a1_clebsch_gordan() -> np.ndarray:
    """Return the normalized ``E2 x E2 -> A1`` Clebsch--Gordan tensor."""

    return np.eye(2, dtype=np.float64) / np.sqrt(2.0)


def e2_a1_contract(first: Sequence[float], second: Sequence[float]) -> float:
    """Contract two real ``E2`` vectors into the normalized ``A1`` scalar."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("E2 vectors must have shape (2,)")
    return float(left @ e2_a1_clebsch_gordan() @ right)


__all__ = [
    "E2_ORIENTATION_FRAME",
    "J2_E2_ORIENTATION_FRAME",
    "e2_a1_clebsch_gordan",
    "e2_a1_contract",
    "e2_orientation_representation",
    "orientation_axis_permutation",
]
