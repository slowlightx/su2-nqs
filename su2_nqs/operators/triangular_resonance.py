"""Symmetry-labelled triangular bond-nematic resonance actions.

``N_a`` transforms in the real ``E2`` irrep, while
``K=(N_1^2+N_2^2)/sqrt(2)`` is its normalized ``A1`` pair contraction.  The
module accepts a pair-action callback and therefore remains independent of a
particular computational or fusion basis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from ..lattices.triangular_e2 import (
    E2_ORIENTATION_FRAME,
    J2_E2_ORIENTATION_FRAME,
)
from ..lattices.triangular_hnf import (
    IntMatrix,
    IntVector,
    J1_POSITIVE,
    J2_POSITIVE,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)


PairAction = Callable[[np.ndarray, int, int], np.ndarray]


def _oriented_bond_families(
    supercell: IntMatrix,
    displacements: tuple[IntVector, IntVector, IntVector],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    families = []
    for displacement in displacements:
        family = []
        for first, site in enumerate(sites):
            target = reduce_hnf(
                (site[0] + displacement[0], site[1] + displacement[1]), matrix
            )
            second = positions[target]
            if first == second:
                raise ValueError("nearest-neighbour direction aliases a site to itself")
            family.append((first, second))
        families.append(tuple(family))
    return tuple(families)


def triangular_e2_actions(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
    frame: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the two extensive-normalized ``E2`` bond operators ``N_a``."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    n_sites = len(hnf_sites(validate_hnf(supercell)))
    orientation_images = []
    checked_frame = (
        E2_ORIENTATION_FRAME
        if frame is None and displacements == J1_POSITIVE
        else J2_E2_ORIENTATION_FRAME
        if frame is None and displacements == J2_POSITIVE
        else np.asarray(frame, dtype=np.float64)
    )
    if checked_frame.shape != (2, 3):
        raise ValueError("orientation frame must have shape (2,3)")
    for family in _oriented_bond_families(supercell, displacements):
        image = np.zeros_like(vector)
        for first, second in family:
            contribution = np.asarray(
                pair_action(vector, first, second), dtype=np.complex128
            )
            if contribution.shape != vector.shape:
                raise ValueError("pair action returned the wrong state shape")
            image += contribution
        orientation_images.append(image)
    matrix = np.stack(orientation_images, axis=0)
    result = checked_frame @ matrix / math.sqrt(n_sites)
    return result[0], result[1]


def triangular_a1_bond_action(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
) -> np.ndarray:
    """Apply the normalized orientation-uniform ``A1`` bond operator.

    The coefficient vector ``(1,1,1)/sqrt(3)`` has the same norm as either
    row of the E2 frame.  The additional ``1/sqrt(N)`` therefore makes this a
    same-shell, same-extensivity control for one E2 component.
    """

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite vector")
    n_sites = len(hnf_sites(validate_hnf(supercell)))
    answer = np.zeros_like(vector)
    for family in _oriented_bond_families(supercell, displacements):
        for first, second in family:
            contribution = np.asarray(
                pair_action(vector, first, second), dtype=np.complex128
            )
            if contribution.shape != vector.shape:
                raise ValueError("pair action returned the wrong state shape")
            answer += contribution
    return answer / math.sqrt(3.0 * n_sites)


def triangular_a1_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    first_displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
    second_displacements: tuple[IntVector, IntVector, IntVector] | None = None,
) -> np.ndarray:
    """Apply the Hermitian ``A1 x A1 -> A1`` same-work pair control."""

    second_displacements = (
        first_displacements if second_displacements is None else second_displacements
    )
    first = triangular_a1_bond_action(
        state,
        pair_action,
        supercell,
        displacements=first_displacements,
    )
    answer = triangular_a1_bond_action(
        first,
        pair_action,
        supercell,
        displacements=second_displacements,
    )
    if second_displacements != first_displacements:
        second = triangular_a1_bond_action(
            state,
            pair_action,
            supercell,
            displacements=second_displacements,
        )
        reverse = triangular_a1_bond_action(
            second,
            pair_action,
            supercell,
            displacements=first_displacements,
        )
        answer = 0.5 * (answer + reverse)
    return np.asarray(answer, dtype=np.complex128)


def triangular_e2_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    first_displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
    second_displacements: tuple[IntVector, IntVector, IntVector] | None = None,
) -> np.ndarray:
    """Apply a Hermitian ``E2 x E2 -> A1`` shell-pair contraction.

    Distinct shells are anticommuted before the normalized scalar contraction.
    """

    second_displacements = (
        first_displacements if second_displacements is None else second_displacements
    )
    first_actions = triangular_e2_actions(
        state,
        pair_action,
        supercell,
        displacements=first_displacements,
    )
    answer = np.zeros_like(np.asarray(state, dtype=np.complex128))
    for index, image in enumerate(first_actions):
        answer += triangular_e2_actions(
            image,
            pair_action,
            supercell,
            displacements=second_displacements,
        )[index]
    if second_displacements != first_displacements:
        second_actions = triangular_e2_actions(
            state,
            pair_action,
            supercell,
            displacements=second_displacements,
        )
        reverse = np.zeros_like(answer)
        for index, image in enumerate(second_actions):
            reverse += triangular_e2_actions(
                image,
                pair_action,
                supercell,
                displacements=first_displacements,
            )[index]
        answer = 0.5 * (answer + reverse)
    return np.asarray(answer, dtype=np.complex128) / math.sqrt(2.0)


def connected_triangular_e2_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    first_displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
    second_displacements: tuple[IntVector, IntVector, IntVector] | None = None,
) -> tuple[np.ndarray, complex]:
    """Return ``(K-<K>)|psi>`` and ``<K>`` for normalized ``psi``."""

    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("state has zero norm")
    vector = vector / norm
    image = triangular_e2_pair_action(
        vector,
        pair_action,
        supercell,
        first_displacements=first_displacements,
        second_displacements=second_displacements,
    )
    mean = complex(np.vdot(vector, image))
    return image - mean * vector, mean


def connected_triangular_a1_pair_action(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    *,
    first_displacements: tuple[IntVector, IntVector, IntVector] = J1_POSITIVE,
    second_displacements: tuple[IntVector, IntVector, IntVector] | None = None,
) -> tuple[np.ndarray, complex]:
    """Return the connected orientation-uniform pair control and its mean."""

    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError("state has zero norm")
    vector = vector / norm
    image = triangular_a1_pair_action(
        vector,
        pair_action,
        supercell,
        first_displacements=first_displacements,
        second_displacements=second_displacements,
    )
    mean = complex(np.vdot(vector, image))
    return image - mean * vector, mean


@dataclass(frozen=True)
class E2Moments:
    """Equal-time weight and energy-weighted ``E2`` sum-rule diagnostics."""

    m0: float
    m1: float
    sma_gap: float
    component_m0: tuple[float, float]


def triangular_e2_moments(
    state: np.ndarray,
    pair_action: PairAction,
    supercell: IntMatrix,
    hamiltonian_action: Callable[[np.ndarray], np.ndarray],
) -> E2Moments:
    """Return ``m0``, ``m1`` and the two-component SMA gap ``m1/m0``."""

    vector = np.asarray(state, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    h_state = np.asarray(hamiltonian_action(vector), dtype=np.complex128)
    energy = float(np.vdot(vector, h_state).real)
    weights = []
    first_moment = 0.0
    for image in triangular_e2_actions(vector, pair_action, supercell):
        mean = np.vdot(vector, image)
        centered = image - mean * vector
        weight = float(np.vdot(centered, centered).real)
        weights.append(weight)
        first_moment += float(
            np.vdot(centered, hamiltonian_action(centered)).real - energy * weight
        )
    m0 = float(sum(weights))
    return E2Moments(
        m0=m0,
        m1=float(first_moment),
        sma_gap=float(first_moment / m0) if m0 > 1.0e-14 else math.inf,
        component_m0=(float(weights[0]), float(weights[1])),
    )


__all__ = [
    "E2Moments",
    "connected_triangular_a1_pair_action",
    "connected_triangular_e2_pair_action",
    "triangular_a1_bond_action",
    "triangular_a1_pair_action",
    "triangular_e2_actions",
    "triangular_e2_moments",
    "triangular_e2_pair_action",
]
