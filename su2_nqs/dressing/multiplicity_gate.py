"""Real orthogonal rotations in four-spin SU(2) multiplicity spaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from .local_schur import (
    FOUR_SPIN_MULTIPLICITIES,
    FOUR_SPIN_TOTAL_Q,
    FourSpinSchurBasis,
    four_spin_schur_basis,
    multiplicity_change_of_basis,
)


def _rotation(size: int, first: int, second: int, angle: float) -> np.ndarray:
    if not math.isfinite(angle):
        raise ValueError("multiplicity rotation angles must be finite")
    result = np.eye(size, dtype=np.float64)
    cosine, sine = math.cos(angle), math.sin(angle)
    result[first, first] = cosine
    result[first, second] = -sine
    result[second, first] = sine
    result[second, second] = cosine
    return result


def plaquette_multiplicity_blocks(
    angles: Sequence[float],
) -> dict[int, np.ndarray]:
    """Return the canonical ``SO(2) x SO(3) x SO(1)`` block family.

    The four angles are ordered ``(q0_01, q2_01, q2_02, q2_12)``.  The
    ``q=2`` convention is the ordered Givens product
    ``R12 @ R02 @ R01``.  All labels are doubled, so q=2 denotes local S=1.
    """

    values = np.asarray(tuple(angles), dtype=np.float64)
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("plaquette gate requires four finite real angles")
    q0 = _rotation(2, 0, 1, float(values[0]))
    q2 = (
        _rotation(3, 1, 2, float(values[3]))
        @ _rotation(3, 0, 2, float(values[2]))
        @ _rotation(3, 0, 1, float(values[1]))
    )
    return {0: q0, 2: q2, 4: np.ones((1, 1), dtype=np.float64)}


def plaquette_multiplicity_generators() -> tuple[dict[int, np.ndarray], ...]:
    """Return the four identity-tangent generators of the real gate family.

    The generator order is identical to :func:`plaquette_multiplicity_blocks`.
    Each block is real antisymmetric (and hence anti-Hermitian), and the four
    generators span ``so(2) + so(3)`` in the q=0 and q=2 multiplicity spaces.
    The q=4 block vanishes because that sector has multiplicity one.
    """

    def generator(size: int, first: int, second: int) -> np.ndarray:
        result = np.zeros((size, size), dtype=np.float64)
        result[first, second] = -1.0
        result[second, first] = 1.0
        return result

    zero_1 = np.zeros((1, 1), dtype=np.float64)
    zero_2 = np.zeros((2, 2), dtype=np.float64)
    zero_3 = np.zeros((3, 3), dtype=np.float64)
    return (
        {0: generator(2, 0, 1), 2: zero_3.copy(), 4: zero_1.copy()},
        {0: zero_2.copy(), 2: generator(3, 0, 1), 4: zero_1.copy()},
        {0: zero_2.copy(), 2: generator(3, 0, 2), 4: zero_1.copy()},
        {0: zero_2.copy(), 2: generator(3, 1, 2), 4: zero_1.copy()},
    )


def _validated_blocks(
    blocks: Mapping[int, np.ndarray], *, unitary: bool = True
) -> dict[int, np.ndarray]:
    if set(blocks) != set(FOUR_SPIN_TOTAL_Q):
        raise ValueError("multiplicity blocks must contain exactly q=0,2,4")
    answer: dict[int, np.ndarray] = {}
    for q in FOUR_SPIN_TOTAL_Q:
        value = np.asarray(blocks[q], dtype=np.complex128)
        dimension = FOUR_SPIN_MULTIPLICITIES[q]
        if value.shape != (dimension, dimension) or np.any(~np.isfinite(value)):
            raise ValueError(f"q={q} multiplicity block has invalid shape or values")
        if unitary and (
            np.max(np.abs(value.conj().T @ value - np.eye(dimension))) > 1.0e-12
        ):
            raise ValueError(f"q={q} multiplicity block is not unitary")
        answer[q] = value
    return answer


def multiplicity_block_operator(
    basis: FourSpinSchurBasis,
    blocks: Mapping[int, np.ndarray],
    *,
    unitary: bool = True,
) -> np.ndarray:
    """Return the gate matrix in ``basis`` coordinates.

    A block acts on multiplicity labels and is repeated identically for every
    magnetic component.  Matrix orientation is ``U[new, old]``.
    """

    checked = _validated_blocks(blocks, unitary=unitary)
    result = np.zeros((16, 16), dtype=np.complex128)
    positions = {
        (label.q, label.m, label.multiplicity): index
        for index, label in enumerate(basis.labels)
    }
    for q in FOUR_SPIN_TOTAL_Q:
        for m in range(-q, q + 1, 2):
            for new in range(FOUR_SPIN_MULTIPLICITIES[q]):
                for old in range(FOUR_SPIN_MULTIPLICITIES[q]):
                    result[
                        positions[(q, m, new)], positions[(q, m, old)]
                    ] = checked[q][new, old]
    return result


def plaquette_unitary(
    angles: Sequence[float],
    *,
    basis: FourSpinSchurBasis | None = None,
) -> np.ndarray:
    """Return the 16x16 physical computational-basis plaquette gate."""

    selected = four_spin_schur_basis() if basis is None else basis
    reduced = multiplicity_block_operator(
        selected, plaquette_multiplicity_blocks(angles)
    )
    return np.asarray(
        selected.columns @ reduced @ selected.columns.conj().T,
        dtype=np.complex128,
    )


def plaquette_generators(
    *,
    basis: FourSpinSchurBasis | None = None,
) -> tuple[np.ndarray, ...]:
    """Return the four 16x16 physical anti-Hermitian gate generators."""

    selected = four_spin_schur_basis() if basis is None else basis
    answer = []
    for blocks in plaquette_multiplicity_generators():
        reduced = multiplicity_block_operator(selected, blocks, unitary=False)
        answer.append(
            np.asarray(
                selected.columns @ reduced @ selected.columns.conj().T,
                dtype=np.complex128,
            )
        )
    return tuple(answer)


def recouple_multiplicity_blocks(
    blocks: Mapping[int, np.ndarray],
    *,
    source: FourSpinSchurBasis,
    target: FourSpinSchurBasis,
) -> dict[int, np.ndarray]:
    """Express one physical multiplicity gate in a second fusion tree.

    If ``R = <target|source>``, the target-coordinate block is
    ``R @ O_source @ R^dagger``.  Reconstructing the physical gate from the
    returned blocks must therefore be exactly tree covariant.
    """

    checked = _validated_blocks(blocks)
    result: dict[int, np.ndarray] = {}
    for q in FOUR_SPIN_TOTAL_Q:
        transform = multiplicity_change_of_basis(source, target, q)
        result[q] = transform @ checked[q] @ transform.conj().T
    return result


__all__ = [
    "multiplicity_block_operator",
    "plaquette_generators",
    "plaquette_multiplicity_blocks",
    "plaquette_multiplicity_generators",
    "plaquette_unitary",
    "recouple_multiplicity_blocks",
]
