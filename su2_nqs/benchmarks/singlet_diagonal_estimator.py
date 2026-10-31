"""Exact-small oracle for diagonal estimators inside the singlet manifold.

For an isotropic spin-half Heisenberg Hamiltonian

``H = sum_ij J_ij (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j)``

rotational invariance of singlet bra and ket states gives

``P0 H P0 = 3 P0 Hz P0``,  ``Hz = sum_ij J_ij Sz_i Sz_j``.

Consequently both the variational energy and every force along a
singlet-preserving tangent can be evaluated with the diagonal ``3 Hz``
operator.  This does *not* make ``3 Hz`` equal to ``H`` before projection.
Its action generally has nonsinglet components, so its Born local-energy
variance remains nonzero even when the full-Hamiltonian local energy has the
zero-variance property on an exact eigenstate.

The routines below build the independent coupled-basis isometry and dense
fixed-magnetization Hamiltonian only for ``N=4``, ``N=6``, or ``N=8`` exact-small
audits.  They are benchmark/oracle utilities, not a scalable estimator or a
training workflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..operators.spin_config import SpinBond, heisenberg_spin_connectivity
from ..su2.fusion_path import enumerate_paths
from .coupled_basis import coupled_basis_matrix
from .exact_diagonalization import heisenberg_sz_matrix


@dataclass(frozen=True)
class SingletDiagonalEstimatorReport:
    """Operator, force, and Born-variance certificate for one exact-small graph."""

    n_sites: int
    sz_dimension: int
    singlet_dimension: int
    singlet_isometry_error: float
    projected_operator_maximum_error: float
    random_singlet_energy_full: float
    random_singlet_energy_diagonal: float
    random_singlet_energy_difference: float
    random_singlet_force_full: float
    random_singlet_force_diagonal: float
    random_singlet_force_difference: float
    singlet_ground_energy: float
    full_ground_local_energy_variance: float
    diagonal_ground_local_energy_variance: float
    diagonal_outside_singlet_norm_squared: float
    diagonal_variance_outside_singlet_error: float


def _checked_size(n_sites: int) -> int:
    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or int(n_sites) not in (4, 6, 8)
    ):
        raise ValueError("the exact-small singlet oracle supports N=4, N=6, or N=8")
    return int(n_sites)


def _bits(state: int, n_sites: int) -> tuple[int, ...]:
    return tuple(
        (int(state) >> (n_sites - site - 1)) & 1
        for site in range(n_sites)
    )


def singlet_isometry_in_sz(n_sites: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return computational-``Sz=0`` rows by singlet multiplicity columns."""

    n_sites = _checked_size(n_sites)
    paths = enumerate_paths(n_sites, 0)
    full = coupled_basis_matrix(paths, total_q=0, total_m=0)
    # An empty Hamiltonian call supplies the repository's deterministic
    # integer basis ordering without introducing a second basis convention.
    _, basis = heisenberg_sz_matrix(n_sites, (), n_up=n_sites // 2)
    isometry = np.asarray(full[np.asarray(basis, dtype=np.int64)], dtype=np.complex128)
    expected = np.eye(len(paths), dtype=np.complex128)
    error = float(np.max(np.abs(isometry.conj().T @ isometry - expected)))
    if error > 2.0e-13:
        raise RuntimeError("coupled-basis singlet columns are not orthonormal")
    isometry.setflags(write=False)
    return isometry, basis


def heisenberg_z_diagonal(
    n_sites: int,
    bonds: Iterable[SpinBond],
    *,
    basis: Sequence[int] | None = None,
) -> np.ndarray:
    """Return diagonal ``Hz=sum J_ij Sz_i Sz_j`` in integer basis order."""

    n_sites = _checked_size(n_sites)
    checked_bonds = tuple(bonds)
    if basis is None:
        _, selected_basis = heisenberg_sz_matrix(
            n_sites, checked_bonds, n_up=n_sites // 2
        )
    else:
        selected_basis = tuple(int(state) for state in basis)
        if not selected_basis or len(set(selected_basis)) != len(selected_basis):
            raise ValueError("basis must be nonempty and duplicate free")
        if any(
            state < 0
            or state >= (1 << n_sites)
            or state.bit_count() != n_sites // 2
            for state in selected_basis
        ):
            raise ValueError("basis states must belong to the Sz=0 sector")
    diagonal = []
    for state in selected_basis:
        configuration = _bits(state, n_sites)
        connected, elements = heisenberg_spin_connectivity(
            configuration, checked_bonds
        )
        try:
            index = connected.index(configuration)
        except ValueError:
            value = 0.0
        else:
            value = float(elements[index])
        diagonal.append(value)
    result = np.asarray(diagonal, dtype=np.float64)
    result.setflags(write=False)
    return result


def normalized_energy(
    state: np.ndarray,
    operator: np.ndarray,
) -> float:
    """Return the real normalized expectation of one Hermitian operator."""

    vector = np.asarray(state, dtype=np.complex128)
    matrix = np.asarray(operator, dtype=np.complex128)
    if vector.ndim != 1 or matrix.shape != (len(vector), len(vector)):
        raise ValueError("state and operator dimensions disagree")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite positive norm")
    expectation = complex(np.vdot(vector, matrix @ vector) / norm)
    if abs(expectation.imag) > 2.0e-12:
        raise RuntimeError("Hermitian expectation acquired an imaginary part")
    return float(expectation.real)


def singlet_variational_force(
    state: np.ndarray,
    tangent: np.ndarray,
    operator: np.ndarray,
) -> float:
    """Return the centered real-parameter energy derivative along ``tangent``."""

    vector = np.asarray(state, dtype=np.complex128)
    derivative = np.asarray(tangent, dtype=np.complex128)
    matrix = np.asarray(operator, dtype=np.complex128)
    if (
        vector.ndim != 1
        or derivative.shape != vector.shape
        or matrix.shape != (len(vector), len(vector))
    ):
        raise ValueError("state, tangent, and operator dimensions disagree")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite positive norm")
    action = matrix @ vector
    energy = np.vdot(vector, action) / norm
    centered = (
        np.vdot(derivative, action)
        - np.vdot(derivative, vector) * energy
    ) / norm
    return float(2.0 * centered.real)


def born_local_energy_variance(
    state: np.ndarray,
    operator: np.ndarray,
) -> float:
    """Return ``Var_|psi|^2[(A psi)(x)/psi(x)]`` without dividing by zeros."""

    vector = np.asarray(state, dtype=np.complex128)
    matrix = np.asarray(operator, dtype=np.complex128)
    if vector.ndim != 1 or matrix.shape != (len(vector), len(vector)):
        raise ValueError("state and operator dimensions disagree")
    norm = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite positive norm")
    action = matrix @ vector
    mean = np.vdot(vector, action) / norm
    # Born weights cancel the local-energy denominator exactly:
    # sum_x p(x)|Apsi(x)/psi(x)-<A>|^2 = ||(A-<A>)psi||^2/||psi||^2.
    residual = action - mean * vector
    variance = float(np.vdot(residual, residual).real / norm)
    if variance < -2.0e-13:
        raise RuntimeError("local-energy variance became negative")
    return max(0.0, variance)


def singlet_diagonal_estimator_benchmark(
    n_sites: int,
    bonds: Iterable[SpinBond],
    *,
    seed: int = 20260916,
) -> SingletDiagonalEstimatorReport:
    """Return a deterministic ``P0 H P0 = 3 P0 Hz P0`` certificate."""

    n_sites = _checked_size(n_sites)
    checked_bonds = tuple(bonds)
    hamiltonian, basis = heisenberg_sz_matrix(
        n_sites, checked_bonds, n_up=n_sites // 2
    )
    singlets, singlet_basis = singlet_isometry_in_sz(n_sites)
    if basis != singlet_basis:
        raise RuntimeError("ED and coupled-basis Sz orderings disagree")
    identity = np.eye(singlets.shape[1], dtype=np.complex128)
    isometry_error = float(
        np.max(np.abs(singlets.conj().T @ singlets - identity))
    )
    hz_diagonal = heisenberg_z_diagonal(
        n_sites, checked_bonds, basis=basis
    )
    if np.max(np.abs(np.diag(hamiltonian) - hz_diagonal), initial=0.0) > 2.0e-14:
        raise RuntimeError("physical connectivity and dense Hz diagonals disagree")
    diagonal_operator = np.diag(3.0 * hz_diagonal)
    projected_hamiltonian = singlets.conj().T @ hamiltonian @ singlets
    projected_diagonal = singlets.conj().T @ diagonal_operator @ singlets
    projected_error = float(
        np.max(np.abs(projected_hamiltonian - projected_diagonal))
    )

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed) + n_sites)
    coefficients = rng.normal(size=singlets.shape[1]) + 1.0j * rng.normal(
        size=singlets.shape[1]
    )
    tangent_coefficients = rng.normal(
        size=singlets.shape[1]
    ) + 1.0j * rng.normal(size=singlets.shape[1])
    state = singlets @ coefficients
    tangent = singlets @ tangent_coefficients
    energy_full = normalized_energy(state, hamiltonian)
    energy_diagonal = normalized_energy(state, diagonal_operator)
    force_full = singlet_variational_force(state, tangent, hamiltonian)
    force_diagonal = singlet_variational_force(
        state, tangent, diagonal_operator
    )

    eigenvalues, eigenvectors = np.linalg.eigh(projected_hamiltonian)
    ground_energy = float(eigenvalues[0])
    ground_state = singlets @ eigenvectors[:, 0]
    full_variance = born_local_energy_variance(ground_state, hamiltonian)
    diagonal_variance = born_local_energy_variance(
        ground_state, diagonal_operator
    )
    projector = singlets @ singlets.conj().T
    outside = (np.eye(len(basis)) - projector) @ (
        diagonal_operator @ ground_state
    )
    outside_norm_squared = float(np.vdot(outside, outside).real)

    return SingletDiagonalEstimatorReport(
        n_sites=n_sites,
        sz_dimension=len(basis),
        singlet_dimension=singlets.shape[1],
        singlet_isometry_error=isometry_error,
        projected_operator_maximum_error=projected_error,
        random_singlet_energy_full=energy_full,
        random_singlet_energy_diagonal=energy_diagonal,
        random_singlet_energy_difference=abs(energy_full - energy_diagonal),
        random_singlet_force_full=force_full,
        random_singlet_force_diagonal=force_diagonal,
        random_singlet_force_difference=abs(force_full - force_diagonal),
        singlet_ground_energy=ground_energy,
        full_ground_local_energy_variance=full_variance,
        diagonal_ground_local_energy_variance=diagonal_variance,
        diagonal_outside_singlet_norm_squared=outside_norm_squared,
        diagonal_variance_outside_singlet_error=abs(
            diagonal_variance - outside_norm_squared
        ),
    )


__all__ = [
    "SingletDiagonalEstimatorReport",
    "born_local_energy_variance",
    "heisenberg_z_diagonal",
    "normalized_energy",
    "singlet_diagonal_estimator_benchmark",
    "singlet_isometry_in_sz",
    "singlet_variational_force",
]
