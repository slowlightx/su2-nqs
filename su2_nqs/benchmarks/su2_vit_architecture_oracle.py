"""Exact-small oracle for physical-input SU(2)-equivariant architectures.

An invariant nonlinear function of representation-valued *inputs* is not, by
itself, an invariant quantum coefficient tensor.  A quantum spin rotation
maps one computational-basis configuration to a superposition of
configurations, whereas an ordinary equivariant neural-network statement
only constrains the value of the function on a rotated feature vector.  The
two statements agree when the scalar map is multilinear in every physical
input leg, but need not agree after attention, self-products, or another
input-dependent nonlinearity.

This module records the minimal two-spin counterexample in the repository's
Condon--Shortley convention.  The Clebsch--Gordan singlet contraction

``a(z1, z2) = sum_(m1,m2) <1/2,m1;1/2,m2|0,0> z1[m1] z2[m2]``

is both SU(2)-invariant and multilinear.  Its square ``a**2`` remains an
SU(2)-invariant classical scalar function, but its values on computational
basis inputs form the triplet ``M=0`` coefficient state rather than a
singlet.  This is an architecture oracle, not a trainable ansatz and not a
ViT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from .exact_diagonalization import spin_squared_matrix
from .spin_sector import spin_purity_metrics


@dataclass(frozen=True)
class SU2ViTArchitectureOracle:
    """Numerical certificate separating scalar invariance from spin purity."""

    singlet_rotation_invariance_error: float
    squared_rotation_invariance_error: float
    singlet_wavefunction_rotation_error: float
    squared_wavefunction_rotation_error: float
    singlet_multi_affinity_defect: float
    squared_multi_affinity_defect: float
    singlet_coefficients: tuple[complex, ...]
    squared_coefficients: tuple[complex, ...]
    singlet_spin_squared_residual_ratio: float
    squared_spin_squared_residual_ratio: float
    singlet_weight: float
    squared_singlet_weight: float


def two_spin_cg_singlet_tensor() -> np.ndarray:
    """Return ``C[m1,m2]=<1/2,m1;1/2,m2|0,0>`` in down/up order."""

    magnetic = (-1, 1)
    tensor = np.asarray(
        [
            [
                clebsch_gordan(1, m1, 1, m2, 0, 0)
                for m2 in magnetic
            ]
            for m1 in magnetic
        ],
        dtype=np.complex128,
    )
    tensor.setflags(write=False)
    return tensor


def spin_half_su2_rotation(
    axis: np.ndarray | tuple[float, float, float], angle: Real
) -> np.ndarray:
    """Return a spin-half SU(2) rotation in repository ``(down, up)`` order."""

    direction = np.asarray(axis, dtype=np.float64)
    if direction.shape != (3,) or np.any(~np.isfinite(direction)):
        raise ValueError("axis must contain three finite real components")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("axis must have nonzero norm")
    if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, Real):
        raise TypeError("angle must be a real scalar")
    theta = float(angle)
    if not math.isfinite(theta):
        raise ValueError("angle must be finite")
    nx, ny, nz = direction / norm
    # Pauli matrices in basis (|down>, |up>).  In this ordering sigma_y has
    # the opposite displayed sign from the common (|up>, |down>) ordering.
    sigma_x = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
    sigma_y = np.asarray(((0.0, 1.0j), (-1.0j, 0.0)), dtype=np.complex128)
    sigma_z = np.asarray(((-1.0, 0.0), (0.0, 1.0)), dtype=np.complex128)
    generator = nx * sigma_x + ny * sigma_y + nz * sigma_z
    rotation = (
        math.cos(theta / 2.0) * np.eye(2, dtype=np.complex128)
        - 1.0j * math.sin(theta / 2.0) * generator
    )
    return rotation


def two_spin_invariant_scalar(
    first: np.ndarray,
    second: np.ndarray,
    *,
    power: int = 1,
) -> complex:
    """Evaluate the CG singlet contraction, optionally raised to a power.

    ``power=1`` is a multilinear invariant.  ``power=2`` is the minimal
    invariant-but-not-multilinear counterexample relevant to nonlinear
    equivariant feature networks.
    """

    if isinstance(power, bool) or not isinstance(power, Integral) or power < 1:
        raise ValueError("power must be a positive integer")
    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("each spinor must have shape (2,) in down/up order")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("spinors must be finite")
    value = complex(left @ two_spin_cg_singlet_tensor() @ right)
    return value**int(power)


def two_spin_basis_coefficients(*, power: int = 1) -> np.ndarray:
    """Evaluate the invariant scalar on all four computational basis pairs."""

    basis = np.eye(2, dtype=np.complex128)
    values = np.asarray(
        [
            two_spin_invariant_scalar(basis[first], basis[second], power=power)
            for first in range(2)
            for second in range(2)
        ],
        dtype=np.complex128,
    )
    values.setflags(write=False)
    return values


def _rotation_invariance_error(*, power: int) -> float:
    rng = np.random.default_rng(20260916 + power)
    maximum = 0.0
    for _ in range(16):
        axis = rng.normal(size=3)
        angle = rng.uniform(-2.0 * np.pi, 2.0 * np.pi)
        rotation = spin_half_su2_rotation(axis, angle)
        first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
        second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
        original = two_spin_invariant_scalar(first, second, power=power)
        rotated = two_spin_invariant_scalar(
            rotation @ first,
            rotation @ second,
            power=power,
        )
        scale = max(1.0, abs(original), abs(rotated))
        maximum = max(maximum, abs(rotated - original) / scale)
    return float(maximum)


def two_spin_wavefunction_rotation_error(coefficients: np.ndarray) -> float:
    """Return the maximum ``U tensor U`` coefficient-state invariance error.

    This is the quantum-state test that classical scalar invariance does not
    imply.  Coefficients use computational order ``down-down, down-up,
    up-down, up-up``.
    """

    state = np.asarray(coefficients, dtype=np.complex128)
    if state.shape != (4,) or np.any(~np.isfinite(state)):
        raise ValueError("coefficients must be a finite length-four vector")
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("coefficient state must have nonzero norm")
    rng = np.random.default_rng(20260921)
    maximum = 0.0
    for _ in range(16):
        rotation = spin_half_su2_rotation(
            rng.normal(size=3),
            rng.uniform(-2.0 * np.pi, 2.0 * np.pi),
        )
        transformed = np.kron(rotation, rotation) @ state
        maximum = max(maximum, float(np.linalg.norm(transformed - state) / norm))
    return maximum


def _multi_affinity_defect(*, power: int) -> float:
    """Return the maximum normalized additivity defect over both input legs."""

    down, up = np.eye(2, dtype=np.complex128)
    probe = np.asarray((1.0 + 0.2j, 1.7 - 0.4j), dtype=np.complex128)
    alpha = 0.7 + 0.2j
    beta = -0.3 + 0.5j

    def defect(slot: int) -> float:
        if slot == 0:
            combined = two_spin_invariant_scalar(
                alpha * down + beta * up, probe, power=power
            )
            separate = (
                alpha * two_spin_invariant_scalar(down, probe, power=power)
                + beta * two_spin_invariant_scalar(up, probe, power=power)
            )
        else:
            combined = two_spin_invariant_scalar(
                probe, alpha * down + beta * up, power=power
            )
            separate = (
                alpha * two_spin_invariant_scalar(probe, down, power=power)
                + beta * two_spin_invariant_scalar(probe, up, power=power)
            )
        return float(
            abs(combined - separate)
            / max(1.0, abs(combined), abs(separate))
        )

    return max(defect(0), defect(1))


def _spin_metrics(coefficients: np.ndarray) -> tuple[float, float]:
    """Return ``(||S^2 psi||/||psi||, singlet weight)`` in ``M=0``."""

    state = np.asarray(coefficients, dtype=np.complex128)
    if state.shape != (4,):
        raise ValueError("two-spin coefficients must have length four")
    spin_squared, basis = spin_squared_matrix(2, n_up=1)
    sector = state[np.asarray(basis, dtype=np.int32)]
    norm = float(np.linalg.norm(sector))
    if norm <= 0.0:
        raise ValueError("coefficient state has zero M=0 norm")
    residual_ratio = float(np.linalg.norm(spin_squared @ sector) / norm)
    metrics = spin_purity_metrics(
        sector,
        spin_squared,
        maximum_spin=1,
        tolerance=1.0e-13,
    )
    return residual_ratio, metrics.singlet_weight


def su2_vit_architecture_oracle() -> SU2ViTArchitectureOracle:
    """Build the deterministic two-spin exactness/counterexample certificate."""

    singlet = two_spin_basis_coefficients(power=1)
    squared = two_spin_basis_coefficients(power=2)
    singlet_residual, singlet_weight = _spin_metrics(singlet)
    squared_residual, squared_weight = _spin_metrics(squared)
    return SU2ViTArchitectureOracle(
        singlet_rotation_invariance_error=_rotation_invariance_error(power=1),
        squared_rotation_invariance_error=_rotation_invariance_error(power=2),
        singlet_wavefunction_rotation_error=two_spin_wavefunction_rotation_error(
            singlet
        ),
        squared_wavefunction_rotation_error=two_spin_wavefunction_rotation_error(
            squared
        ),
        singlet_multi_affinity_defect=_multi_affinity_defect(power=1),
        squared_multi_affinity_defect=_multi_affinity_defect(power=2),
        singlet_coefficients=tuple(complex(value) for value in singlet),
        squared_coefficients=tuple(complex(value) for value in squared),
        singlet_spin_squared_residual_ratio=singlet_residual,
        squared_spin_squared_residual_ratio=squared_residual,
        singlet_weight=singlet_weight,
        squared_singlet_weight=squared_weight,
    )


__all__ = [
    "SU2ViTArchitectureOracle",
    "spin_half_su2_rotation",
    "su2_vit_architecture_oracle",
    "two_spin_basis_coefficients",
    "two_spin_cg_singlet_tensor",
    "two_spin_invariant_scalar",
    "two_spin_wavefunction_rotation_error",
]
