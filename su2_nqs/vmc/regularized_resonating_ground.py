"""Cross-fitted regularisation for a nested direct-amplitude ground-state bank.

Column zero is an immutable strong carrier.  Remaining columns are exact-
sector nonlinear resonances.  A positive ridge is applied only to the
resonance coordinates after empirical column normalisation.  The
``ridge -> infinity`` limit therefore recovers the carrier instead of a zero
state.  Persistent chains, rather than individual configurations, define the
cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegularizedGroundFit:
    ridge: float
    energy: float
    variance: float
    coefficients: np.ndarray
    overlap_condition_number: float
    resonance_norm_fraction: float
    sample_count: int


def _checked_rows(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    proposal_probabilities: np.ndarray | Sequence[float],
):
    phi = np.asarray(amplitudes, dtype=np.complex128)
    hphi = np.asarray(actions, dtype=np.complex128)
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] < 3 or phi.shape[1] < 2:
        raise ValueError("amplitudes must have shape (at_least_three,at_least_two)")
    if hphi.shape != phi.shape:
        raise ValueError("actions must match amplitudes")
    if proposal.shape != (len(phi),) or np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal probabilities must be finite and positive")
    if np.any(~np.isfinite(phi)) or np.any(~np.isfinite(hphi)):
        raise FloatingPointError("direct amplitudes and actions must be finite")
    measure = 1.0 / proposal
    measure /= np.max(measure)
    return phi, hphi, measure


def fit_regularized_resonating_ground(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    proposal_probabilities: np.ndarray | Sequence[float],
    *,
    ridge: float,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float = 1.0e9,
) -> RegularizedGroundFit:
    """Fit the lowest ridge-regularized Ritz root on supplied rows."""

    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    phi, hphi, measure = _checked_rows(amplitudes, actions, proposal_probabilities)
    column_norms = np.sqrt(
        np.real(np.einsum("x,xa,xa->a", measure, phi.conj(), phi, optimize=True))
    )
    if np.any(~np.isfinite(column_norms)) or np.any(column_norms <= 0.0):
        raise FloatingPointError("carrier bank contains a zero or invalid column")
    scaled = phi / column_norms[None, :]
    scaled_h = hphi / column_norms[None, :]
    overlap = np.einsum("x,xa,xb->ab", measure, scaled.conj(), scaled, optimize=True)
    hamiltonian = np.einsum(
        "x,xa,xb->ab", measure, scaled.conj(), scaled_h, optimize=True
    )
    overlap = 0.5 * (overlap + overlap.conj().T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    values, vectors = np.linalg.eigh(overlap)
    cutoff = max(float(values[-1]) * relative_rank_tolerance, 1.0e-14)
    keep = values > cutoff
    if int(np.count_nonzero(keep)) != phi.shape[1]:
        raise FloatingPointError("regularized carrier bank is rank deficient")
    condition = float(values[-1] / values[0])
    if condition > maximum_condition_number:
        raise FloatingPointError("regularized carrier overlap exceeds condition gate")
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    penalty = np.diag(np.r_[0.0, np.ones(phi.shape[1] - 1)])
    effective = whitener.conj().T @ (hamiltonian + float(ridge) * penalty) @ whitener
    effective = 0.5 * (effective + effective.conj().T)
    _, roots = np.linalg.eigh(effective)
    scaled_coefficients = whitener @ roots[:, 0]
    coefficients = scaled_coefficients / column_norms
    state = phi @ coefficients
    action = hphi @ coefficients
    norm = float(np.sum(measure * np.abs(state) ** 2))
    energy = complex(np.sum(measure * np.conj(state) * action) / norm)
    variance = float(
        np.sum(measure * np.abs(action - energy * state) ** 2) / norm
    )
    resonance = float(np.sum(np.abs(scaled_coefficients[1:]) ** 2))
    total = float(np.sum(np.abs(scaled_coefficients) ** 2))
    return RegularizedGroundFit(
        ridge=float(ridge),
        energy=float(energy.real),
        variance=variance,
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        overlap_condition_number=condition,
        resonance_norm_fraction=resonance / total,
        sample_count=len(phi),
    )


def evaluate_resonating_ground(
    coefficients: np.ndarray | Sequence[complex],
    amplitudes: np.ndarray,
    actions: np.ndarray,
    proposal_probabilities: np.ndarray | Sequence[float],
) -> dict[str, object]:
    """Evaluate frozen coefficients by a self-normalized ratio estimator."""

    phi, hphi, measure = _checked_rows(amplitudes, actions, proposal_probabilities)
    values = np.asarray(coefficients, dtype=np.complex128)
    if values.shape != (phi.shape[1],):
        raise ValueError("coefficients have the wrong shape")
    state, action = phi @ values, hphi @ values
    norm = float(np.sum(measure * np.abs(state) ** 2))
    energy = complex(np.sum(measure * np.conj(state) * action) / norm)
    variance = float(np.sum(measure * np.abs(action - energy * state) ** 2) / norm)
    weights = measure * np.abs(state) ** 2
    ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
    return {
        "energy": float(energy.real),
        "variance": variance,
        "ess_fraction": ess / len(phi),
        "state": state,
        "action": action,
    }


def crossfit_resonating_ridge(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    proposal_probabilities: np.ndarray | Sequence[float],
    *,
    chain_count: int,
    fold_count: int,
    ridge_grid: Sequence[float],
) -> dict[str, object]:
    """Select ridge by grouped persistent-chain cross-validation.

    The one-standard-error rule chooses the *largest* ridge statistically
    compatible with the best mean held-fold energy, biasing toward the nested
    carrier when the extra resonance columns are not reproducibly useful.
    """

    phi, hphi, _ = _checked_rows(amplitudes, actions, proposal_probabilities)
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if (
        isinstance(chain_count, bool)
        or not isinstance(chain_count, Integral)
        or int(chain_count) < 2
        or len(phi) % int(chain_count)
    ):
        raise ValueError("sample count must be a multiple of chain_count >= 2")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, Integral)
        or int(fold_count) < 2
        or int(fold_count) > int(chain_count)
    ):
        raise ValueError("fold_count must lie between two and chain_count")
    ridges = tuple(float(value) for value in ridge_grid)
    if not ridges or any(not np.isfinite(value) or value < 0.0 for value in ridges):
        raise ValueError("ridge_grid must be finite and nonnegative")
    if tuple(sorted(set(ridges))) != ridges:
        raise ValueError("ridge_grid must be strictly increasing")
    chain_ids = np.arange(len(phi)) % int(chain_count)
    rows = []
    for ridge in ridges:
        held = []
        for fold in range(int(fold_count)):
            validation = chain_ids % int(fold_count) == fold
            training = ~validation
            fit = fit_regularized_resonating_ground(
                phi[training], hphi[training], proposal[training], ridge=ridge
            )
            observed = evaluate_resonating_ground(
                fit.coefficients,
                phi[validation],
                hphi[validation],
                proposal[validation],
            )
            held.append(float(observed["energy"]))
        mean = float(np.mean(held))
        standard_error = float(np.std(held, ddof=1) / np.sqrt(len(held)))
        rows.append(
            {
                "ridge": ridge,
                "held_fold_energies": held,
                "mean_held_energy": mean,
                "fold_standard_error": standard_error,
            }
        )
    best = min(rows, key=lambda row: row["mean_held_energy"])
    threshold = best["mean_held_energy"] + best["fold_standard_error"]
    eligible = [row for row in rows if row["mean_held_energy"] <= threshold]
    selected = max(eligible, key=lambda row: row["ridge"])
    fit = fit_regularized_resonating_ground(
        phi, hphi, proposal, ridge=float(selected["ridge"])
    )
    return {
        "grid": rows,
        "minimum_mean_ridge": best["ridge"],
        "one_standard_error_threshold": threshold,
        "selected_ridge": selected["ridge"],
        "selection_rule": "largest_ridge_within_one_fold_standard_error_of_minimum",
        "fit": fit,
    }


__all__ = [
    "RegularizedGroundFit",
    "crossfit_resonating_ridge",
    "evaluate_resonating_ground",
    "fit_regularized_resonating_ground",
]
