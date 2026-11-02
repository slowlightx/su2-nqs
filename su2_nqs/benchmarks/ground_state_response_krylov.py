"""Response spectral quadrature from a saved state and an exact H action.

This finite-vector calibration does not optimize a ground state. Spin and
momentum labels belong to the supplied source and must be checked separately.
For an approximate parent, Ritz energies relative to its energy are not gaps.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ResponseKrylov:
    """Orthonormal source Krylov basis and its physical spectral quadrature."""

    basis: np.ndarray
    h_basis: np.ndarray
    energies: np.ndarray
    source_weights: np.ndarray
    residual_norms: np.ndarray
    orthonormality_error: float
    projected_hermiticity_error: float


def response_krylov(
    h_action: Callable[[np.ndarray], np.ndarray],
    source: np.ndarray,
    *,
    max_dimension: int,
    breakdown_tolerance: float = 1e-12,
) -> ResponseKrylov:
    """Build K_m(H, source) with two-pass full reorthogonalization.

    Absolute Ritz energies are upper bounds within the source-accessible
    Hilbert subspace for a Hermitian exact H action. Quadrature weights are
    overlaps with the normalized source; no artificial broadening is used.
    A small projected eigenvalue is only resolved if its physical residual
    and stability with dimension are also satisfactory.
    """
    vector = np.asarray(source, dtype=np.complex128)
    if (vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0):
        raise ValueError("a finite nonzero source vector is required")
    if (isinstance(max_dimension, bool) or not isinstance(max_dimension, Integral)
            or max_dimension < 1 or max_dimension > len(vector)):
        raise ValueError("valid positive Krylov dimension required")
    if not np.isfinite(breakdown_tolerance) or breakdown_tolerance <= 0:
        raise ValueError("positive finite breakdown tolerance required")
    columns = [vector / np.linalg.norm(vector)]
    images = []
    for index in range(int(max_dimension)):
        image = np.asarray(h_action(columns[index]), dtype=np.complex128)
        if image.shape != vector.shape or not np.isfinite(image).all():
            raise ValueError("H action returned an invalid vector")
        images.append(image)
        if index + 1 == max_dimension:
            break
        basis = np.column_stack(columns)
        candidate = image.copy()
        for _ in range(2):
            candidate -= basis @ (basis.conj().T @ candidate)
        length = np.linalg.norm(candidate)
        if length <= breakdown_tolerance * max(1., np.linalg.norm(image)):
            break
        columns.append(candidate / length)
    basis = np.column_stack(columns)
    h_basis = np.column_stack(images)
    projected = basis.conj().T @ h_basis
    hermiticity = float(np.max(abs(projected - projected.conj().T)))
    if hermiticity > 1e-10 * max(1., np.max(abs(projected))):
        raise ValueError("H action is not Hermitian on the response subspace")
    energies, coefficients = np.linalg.eigh((projected + projected.conj().T) / 2)
    weights = abs(coefficients[0])**2
    residual = h_basis @ coefficients - (basis @ coefficients)*energies[None, :]
    return ResponseKrylov(
        basis=basis, h_basis=h_basis, energies=energies,
        source_weights=weights, residual_norms=np.linalg.norm(residual, axis=0),
        orthonormality_error=float(np.max(abs(basis.conj().T @ basis - np.eye(len(columns))))),
        projected_hermiticity_error=hermiticity,
    )


def one_step_response_from_moments(moments: np.ndarray) -> dict[str, np.ndarray]:
    """Two-dimensional response Ritz quadrature from m0 through m3.

    Moments are <phi|D^r|phi> for a Hermitian D=H-E_shift. m3, absent from a
    centroid/width readout, fixes the second diagonal of the Lanczos matrix.
    Fail on an unresolved/nonpositive response variance; do not silently
    truncate a noisy moment matrix. Exact eigen-sources need no second vector.
    """
    m = np.asarray(moments)
    if (m.shape != (4,) or np.iscomplexobj(m) or not np.isfinite(m).all()
            or m[0] <= 0):
        raise ValueError("four finite real moments with positive m0 required")
    normalized = m / m[0]
    mean = normalized[1]
    variance = normalized[2] - mean**2
    if variance <= 0:
        raise ValueError("positive resolved source variance required for a second vector")
    next_diagonal = (normalized[3] - 2*mean*normalized[2] + mean**3) / variance
    matrix = np.array([[mean, np.sqrt(variance)], [np.sqrt(variance), next_diagonal]])
    energies, eigenvectors = np.linalg.eigh(matrix)
    return {"matrix": matrix, "energies": energies, "source_weights": abs(eigenvectors[0])**2}
