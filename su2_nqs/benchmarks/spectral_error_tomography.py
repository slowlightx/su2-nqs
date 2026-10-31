"""Exact-small spectral and matched-rank subspace diagnostics.

These helpers do not construct a Hamiltonian or optimize a state.  They take
an already sealed Hermitian matrix and normalized candidate vectors and ask a
specific question: does a neural state contribute a low-energy direction
outside a matched polynomial Krylov space, or merely relearn spectral
filtering of the same carrier?
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalized_state(state: Sequence[complex]) -> np.ndarray:
    """Return a finite normalized complex128 state vector."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a nonempty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError("state contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).tiny:
        raise ValueError("state has zero norm")
    return vector / norm


def orthonormal_column_space(
    columns: Sequence[Sequence[complex]] | np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return a deterministic SVD basis, rank, and singular values."""

    matrix = np.asarray(columns, dtype=np.complex128)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("columns must be a nonempty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("columns contain non-finite values")
    if not 0.0 < float(relative_tolerance) < 1.0:
        raise ValueError("relative_tolerance must lie in (0,1)")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    cutoff = max(
        float(singular[0]) * float(relative_tolerance),
        100.0 * np.finfo(float).eps,
    )
    rank = int(np.count_nonzero(singular > cutoff))
    if rank == 0:
        raise ValueError("column space has zero numerical rank")
    return left[:, :rank], rank, singular


def krylov_columns(
    carrier: Sequence[complex],
    hamiltonian: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Return ``[psi, H psi, ..., H**degree psi]`` without orthogonalization."""

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    state = normalized_state(carrier)
    if matrix.ndim != 2 or matrix.shape != (state.size, state.size):
        raise ValueError("Hamiltonian shape and state dimension disagree")
    if not np.allclose(matrix, matrix.conj().T, atol=2.0e-12, rtol=0.0):
        raise ValueError("Hamiltonian must be Hermitian")
    result = [state]
    for _ in range(degree):
        result.append(matrix @ result[-1])
    return np.column_stack(result)


def lowest_subspace_state(
    columns: Sequence[Sequence[complex]] | np.ndarray,
    hamiltonian: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, int]:
    """Return the lowest Ritz vector in an arbitrary supplied column space."""

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    basis, rank, _ = orthonormal_column_space(
        columns, relative_tolerance=relative_tolerance
    )
    if matrix.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("Hamiltonian shape and column dimension disagree")
    reduced = basis.conj().T @ matrix @ basis
    reduced = 0.5 * (reduced + reduced.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    state = basis @ eigenvectors[:, int(np.argmin(eigenvalues))]
    return normalized_state(state), rank


def state_metrics(state: Sequence[complex], hamiltonian: np.ndarray) -> dict[str, float]:
    """Return total energy and variance for one normalized state."""

    vector = normalized_state(state)
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("Hamiltonian shape and state dimension disagree")
    action = matrix @ vector
    energy = float(np.vdot(vector, action).real)
    variance = float(np.vdot(action, action).real - energy * energy)
    return {"energy_total": energy, "variance_total": max(variance, 0.0)}


def spectral_state_profile(
    state: Sequence[complex], hamiltonian: np.ndarray
) -> dict[str, object]:
    """Decompose energy error and variance over the exact finite spectrum."""

    vector = normalized_state(state)
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.shape != (vector.size, vector.size):
        raise ValueError("Hamiltonian shape and state dimension disagree")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    weights = np.abs(eigenvectors.conj().T @ vector) ** 2
    weights = np.asarray(weights.real, dtype=np.float64)
    weights /= np.sum(weights)
    energy = float(weights @ eigenvalues)
    ground = float(eigenvalues[0])
    energy_terms = weights * (eigenvalues - ground)
    variance_terms = weights * (eigenvalues - energy) ** 2
    energy_error = float(np.sum(energy_terms))
    variance = float(np.sum(variance_terms))
    excited_weight = float(1.0 - weights[0])

    def count_for_fraction(terms: np.ndarray, fraction: float) -> int:
        total = float(np.sum(terms[1:]))
        if total <= np.finfo(float).tiny:
            return 0
        cumulative = np.cumsum(terms[1:]) / total
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    excited = weights[1:]
    leading_offset = int(np.argmax(excited)) if len(excited) else 0
    leading_index = leading_offset + 1 if len(excited) else 0
    upper_start = max(1, len(weights) // 2)
    return {
        "dimension": int(vector.size),
        "energy_total": energy,
        "variance_total": variance,
        "ground_energy_total": ground,
        "energy_error_total": energy_error,
        "ground_state_fidelity": float(weights[0]),
        "total_excited_weight": excited_weight,
        "effective_excitation_energy": (
            energy_error / excited_weight
            if excited_weight > np.finfo(float).tiny
            else 0.0
        ),
        "spectral_participation_number": float(1.0 / np.sum(weights**2)),
        "leading_excited_index": leading_index,
        "leading_excited_gap": (
            float(eigenvalues[leading_index] - ground) if leading_index else 0.0
        ),
        "leading_excited_weight": (
            float(weights[leading_index]) if leading_index else 0.0
        ),
        "lowest_excited_states_for_90pct_energy_error": count_for_fraction(
            energy_terms, 0.9
        ),
        "lowest_excited_states_for_90pct_variance": count_for_fraction(
            variance_terms, 0.9
        ),
        "upper_half_energy_error_fraction": (
            float(np.sum(energy_terms[upper_start:]) / energy_error)
            if energy_error > np.finfo(float).tiny
            else 0.0
        ),
        "upper_half_variance_fraction": (
            float(np.sum(variance_terms[upper_start:]) / variance)
            if variance > np.finfo(float).tiny
            else 0.0
        ),
        "eigenvalues_total": eigenvalues.real.tolist(),
        "spectral_weights": weights.tolist(),
        "energy_error_contributions": energy_terms.tolist(),
        "variance_contributions": variance_terms.tolist(),
    }


def orthogonal_weight(
    state: Sequence[complex], columns: Sequence[Sequence[complex]] | np.ndarray
) -> float:
    """Return squared norm outside the numerical span of ``columns``."""

    vector = normalized_state(state)
    basis, _, _ = orthonormal_column_space(columns)
    if basis.shape[0] != vector.size:
        raise ValueError("state and column dimensions disagree")
    projected = basis @ (basis.conj().T @ vector)
    return float(max(0.0, 1.0 - np.vdot(projected, projected).real))


__all__ = [
    "krylov_columns",
    "lowest_subspace_state",
    "normalized_state",
    "orthogonal_weight",
    "orthonormal_column_space",
    "spectral_state_profile",
    "state_metrics",
]
