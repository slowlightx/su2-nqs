"""Cross-fitted ridge selection for small sampled scalar-operator spaces.

Every candidate space must contain an explicitly represented three-dimensional
``I,H,H^2`` control.  The ridge penalizes only the physical state norm outside
that control, so its infinite-strength limit is exactly two-Lanczos even when
the scalar tokens are centered, rescaled, or not aligned with coordinate
axes.  Selection is grouped by persistent Markov chain so no chain contributes
to both sides of a cross-validation fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


KRYLOV_INDICES = (0, 1, 3)
RESONANCE_INDICES = (2, 4, 5, 6)


@dataclass(frozen=True)
class RegularizedRitzFit:
    """Coefficients and diagnostics for one regularized generalized Ritz fit."""

    coefficients: np.ndarray
    training_energy: float
    gram_rank: int
    gram_condition_number: float
    resonance_norm_squared: float


def _checked_matrices(
    components: np.ndarray, hamiltonian_components: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(components, dtype=np.complex128)
    hx = np.asarray(hamiltonian_components, dtype=np.complex128)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape != hx.shape:
        raise ValueError("component and Hamiltonian matrices must share a rank-two shape")
    if x.shape[1] < 4:
        raise ValueError("the candidate space must contain at least four columns")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(hx)):
        raise FloatingPointError("operator matrices contain non-finite values")
    return x, hx


def _checked_control_basis(
    control_basis: np.ndarray | None, component_count: int
) -> np.ndarray:
    if control_basis is None:
        if component_count != 7:
            raise ValueError(
                "an explicit control_basis is required outside the legacy "
                "seven-column H+R ordering"
            )
        basis = np.zeros((7, 3), dtype=np.float64)
        basis[np.asarray(KRYLOV_INDICES), np.arange(3)] = 1.0
        return basis
    basis = np.asarray(control_basis, dtype=np.float64)
    if basis.shape != (component_count, 3) or np.any(~np.isfinite(basis)):
        raise ValueError(
            f"control_basis must have shape ({component_count}, 3) and be finite"
        )
    if np.linalg.matrix_rank(basis) != 3:
        raise ValueError("control_basis must have rank three")
    return basis


def self_normalized_energy(
    coefficients: np.ndarray,
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
) -> float:
    """Evaluate the carrier-sampled Rayleigh quotient for fixed coefficients."""

    x, hx = _checked_matrices(components, hamiltonian_components)
    vector = np.asarray(coefficients, dtype=np.float64)
    if vector.shape != (x.shape[1],) or np.any(~np.isfinite(vector)):
        raise ValueError(
            f"coefficients must contain {x.shape[1]} finite real values"
        )
    state = x @ vector
    action = hx @ vector
    denominator = float(np.sum(np.abs(state) ** 2))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise FloatingPointError("trial state has invalid sampled norm")
    energy = complex(np.sum(np.conj(state) * action) / denominator)
    if not np.isfinite(energy):
        raise FloatingPointError("trial state has non-finite sampled energy")
    return float(energy.real)


def regularized_generalized_ritz(
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
    *,
    ridge: float,
    gram_relative_cutoff: float = 1.0e-10,
    control_basis: np.ndarray | None = None,
) -> RegularizedRitzFit:
    """Solve the real-coefficient generalized Ritz problem with resonance ridge."""

    x, hx = _checked_matrices(components, hamiltonian_components)
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not 0.0 < gram_relative_cutoff < 1.0:
        raise ValueError("gram_relative_cutoff must lie between zero and one")
    gram = np.real(x.conj().T @ x / len(x))
    hamiltonian = np.real(x.conj().T @ hx / len(x))
    gram = (gram + gram.T) / 2.0
    hamiltonian = (hamiltonian + hamiltonian.T) / 2.0
    # Penalize the *state norm* outside the nested Krylov span, not the
    # Euclidean length of basis-dependent coefficients.  For K selecting
    # I,H,H^2, c.T @ P @ c is the minimum Gram-norm residual after projecting
    # Xc onto XK.  It is invariant under centering/scaling changes that retain
    # the two nested spans.
    basis = _checked_control_basis(control_basis, x.shape[1])
    krylov_gram = basis.T @ gram @ basis
    penalty = gram - gram @ basis @ np.linalg.pinv(
        krylov_gram, rcond=gram_relative_cutoff
    ) @ basis.T @ gram
    penalty = (penalty + penalty.T) / 2.0
    objective = hamiltonian + ridge * penalty
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    cutoff = max(1.0e-13, float(eigenvalues[-1]) * gram_relative_cutoff)
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("sampled component Gram matrix has zero rank")
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitener.T @ objective @ whitener
    _, vectors = np.linalg.eigh((reduced + reduced.T) / 2.0)
    coefficients = np.asarray(whitener @ vectors[:, 0], dtype=np.float64)
    norm = float(coefficients @ gram @ coefficients)
    coefficients /= np.sqrt(norm)
    energy = self_normalized_energy(coefficients, x, hx)
    resonance_norm_squared = float(max(0.0, coefficients @ penalty @ coefficients))
    return RegularizedRitzFit(
        coefficients=coefficients,
        training_energy=energy,
        gram_rank=int(np.count_nonzero(keep)),
        gram_condition_number=float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
        resonance_norm_squared=resonance_norm_squared,
    )


def krylov_control_fit(
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
    *,
    control_basis: np.ndarray | None = None,
) -> RegularizedRitzFit:
    """Fit the exactly nested ``I,H,H^2`` real-coefficient control."""

    x, hx = _checked_matrices(components, hamiltonian_components)
    basis = _checked_control_basis(control_basis, x.shape[1])
    local_x, local_hx = x @ basis, hx @ basis
    gram = np.real(local_x.conj().T @ local_x / len(x))
    hamiltonian = np.real(local_x.conj().T @ local_hx / len(x))
    gram = (gram + gram.T) / 2.0
    hamiltonian = (hamiltonian + hamiltonian.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    cutoff = max(1.0e-13, float(eigenvalues[-1]) * 1.0e-10)
    keep = eigenvalues > cutoff
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitener.T @ hamiltonian @ whitener
    _, vectors = np.linalg.eigh((reduced + reduced.T) / 2.0)
    local_coefficients = whitener @ vectors[:, 0]
    coefficients = basis @ local_coefficients
    coefficients /= np.sqrt(float(coefficients @ np.real(x.conj().T @ x / len(x)) @ coefficients))
    return RegularizedRitzFit(
        coefficients=coefficients,
        training_energy=self_normalized_energy(coefficients, x, hx),
        gram_rank=int(np.count_nonzero(keep)),
        gram_condition_number=float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
        resonance_norm_squared=0.0,
    )


def _chain_fold_indices(
    sample_count: int, *, chain_count: int, fold_count: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if chain_count < 2 or sample_count % chain_count != 0:
        raise ValueError("sample_count must be a multiple of at least two chains")
    if fold_count < 2 or fold_count > chain_count:
        raise ValueError("fold_count must lie between two and chain_count")
    indices = np.arange(sample_count, dtype=np.int64)
    chain_ids = indices % chain_count
    folds = []
    for fold in range(fold_count):
        held_chains = np.arange(chain_count)[np.arange(chain_count) % fold_count == fold]
        held = indices[np.isin(chain_ids, held_chains)]
        training = indices[~np.isin(chain_ids, held_chains)]
        folds.append((training, held))
    return tuple(folds)


def crossfit_select_ridge(
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
    *,
    chain_count: int,
    fold_count: int,
    ridge_grid: Sequence[float],
    control_basis: np.ndarray | None = None,
) -> tuple[RegularizedRitzFit, dict[str, object]]:
    """Select ridge by grouped cross-fit and the conservative one-SE rule.

    The score is the held-fold energy difference relative to a two-Lanczos fit
    trained on exactly the same complementary chains.  Among candidates whose
    mean score is within one standard error of the best score, the largest
    ridge is selected.  Thus weak evidence automatically shrinks toward the
    nested Krylov control.
    """

    x, hx = _checked_matrices(components, hamiltonian_components)
    ridges = np.asarray(tuple(float(value) for value in ridge_grid), dtype=np.float64)
    if ridges.ndim != 1 or len(ridges) < 2 or np.any(~np.isfinite(ridges)):
        raise ValueError("ridge_grid must contain at least two finite values")
    if np.any(ridges < 0.0) or np.any(np.diff(ridges) <= 0.0):
        raise ValueError("ridge_grid must be strictly increasing and non-negative")
    folds = _chain_fold_indices(len(x), chain_count=chain_count, fold_count=fold_count)
    basis = _checked_control_basis(control_basis, x.shape[1])
    rows = []
    for ridge in ridges:
        differences = []
        resonance_norms = []
        for training, held in folds:
            fit = regularized_generalized_ritz(
                x[training],
                hx[training],
                ridge=float(ridge),
                control_basis=basis,
            )
            control = krylov_control_fit(
                x[training], hx[training], control_basis=basis
            )
            differences.append(
                self_normalized_energy(fit.coefficients, x[held], hx[held])
                - self_normalized_energy(control.coefficients, x[held], hx[held])
            )
            resonance_norms.append(fit.resonance_norm_squared)
        mean = float(np.mean(differences))
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        rows.append(
            {
                "ridge": float(ridge),
                "fold_energy_differences": differences,
                "mean_energy_difference": mean,
                "fold_standard_error": standard_error,
                "mean_resonance_norm_squared": float(np.mean(resonance_norms)),
            }
        )
    best = min(rows, key=lambda row: float(row["mean_energy_difference"]))
    threshold = float(best["mean_energy_difference"]) + float(best["fold_standard_error"])
    eligible = [row for row in rows if float(row["mean_energy_difference"]) <= threshold]
    selected = max(eligible, key=lambda row: float(row["ridge"]))
    final_fit = regularized_generalized_ritz(
        x, hx, ridge=float(selected["ridge"]), control_basis=basis
    )
    return final_fit, {
        "fold_count": fold_count,
        "chain_count": chain_count,
        "selection_rule": "largest_ridge_within_one_standard_error_of_best_paired_cv_score",
        "best_mean_ridge": float(best["ridge"]),
        "one_standard_error_threshold": threshold,
        "selected_ridge": float(selected["ridge"]),
        "grid": rows,
    }


__all__ = [
    "KRYLOV_INDICES",
    "RESONANCE_INDICES",
    "RegularizedRitzFit",
    "crossfit_select_ridge",
    "krylov_control_fit",
    "regularized_generalized_ritz",
    "self_normalized_energy",
]
