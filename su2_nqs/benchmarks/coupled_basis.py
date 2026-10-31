"""Reference conversion from left-fusion paths to the computational basis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..su2.clebsch_gordan import spin_half_cg
from ..su2.fusion_path import FusionPath, is_valid_path
from ..su2.quantum_numbers import validate_m


def coupled_state_to_sz(
    path: Sequence[int],
    total_q: int | None = None,
    total_m: int | None = None,
) -> np.ndarray:
    """Expand a left-associated coupled state as a length-``2**N`` vector.

    Basis bit ``0`` denotes ``m=-1/2`` and bit ``1`` denotes ``m=+1/2``;
    the first physical spin is the most significant bit.  All quantum-number
    arguments are doubled integers.
    """

    path = tuple(path)
    if total_q is None:
        total_q = path[-1] if path else -1
    if not is_valid_path(path, total_q):
        raise ValueError(f"invalid fusion path for q={total_q}: {path!r}")
    if total_m is None:
        total_m = total_q
    validate_m(total_q, total_m, name="total_m")

    # (running m, computational index) -> amplitude
    amplitudes: dict[tuple[int, int], complex] = {
        (-1, 0): 1.0,
        (1, 1): 1.0,
    }
    q_left = 1
    for q_total_at_step in path[1:]:
        updated: dict[tuple[int, int], complex] = {}
        for (m_left, index), amplitude in amplitudes.items():
            for bit, m_phys in ((0, -1), (1, 1)):
                coefficient = spin_half_cg(q_left, m_left, m_phys, q_total_at_step)
                if coefficient:
                    key = (m_left + m_phys, (index << 1) | bit)
                    updated[key] = updated.get(key, 0.0) + amplitude * coefficient
        amplitudes = updated
        q_left = q_total_at_step

    vector = np.zeros(2 ** len(path), dtype=np.complex128)
    for (m, index), amplitude in amplitudes.items():
        if m == total_m:
            vector[index] = amplitude
    return vector


def coupled_basis_matrix(
    paths: Sequence[FusionPath],
    total_q: int | None = None,
    total_m: int | None = None,
) -> np.ndarray:
    """Stack coupled states as orthonormal computational-basis columns."""

    paths = tuple(tuple(path) for path in paths)
    if not paths:
        return np.empty((0, 0), dtype=np.complex128)
    if total_q is None:
        total_q = paths[0][-1]
    if total_m is None:
        total_m = total_q
    return np.column_stack(
        [coupled_state_to_sz(path, total_q, total_m) for path in paths]
    )

