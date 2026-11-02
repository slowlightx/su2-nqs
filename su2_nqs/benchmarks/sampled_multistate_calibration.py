"""Exact-small statistical calibration for sampled multistate Ritz estimates.

The production three-root estimator necessarily learns a generalized Ritz
basis from Monte Carlo data.  Reusing those rows to advertise a residual is
optimistically biased.  This module therefore keeps two independent,
equal-stratum samples:

``fit``
    fixes the sampled ``S/H/K`` matrices and the generalized Ritz vectors;

``held_out``
    evaluates the frozen vectors and energies without rediagonalizing.

The exhaustive Hilbert space is used only in small-system calibration.  It
provides the exact Ritz result and, for every *sample-fitted* vector, its exact
Rayleigh quotient and normalized direct residual.  This distinction avoids
root-matching ambiguities when two Ritz values are close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vmc.sampled_multistate import (
    FailClosedGeneralizedRitz,
    SampledSubspaceMoments,
    fail_closed_generalized_ritz,
    sampled_subspace_moments,
)


@dataclass(frozen=True)
class EqualStratumSample:
    """Indices drawn independently from every normalized root Born law."""

    basis_indices: np.ndarray
    root_ids: np.ndarray
    samples_per_root: int
    root_count: int

    @property
    def sample_count(self) -> int:
        """Total number of proposal observations."""

        return int(len(self.basis_indices))


@dataclass(frozen=True)
class HeldOutMultistateCalibration:
    """One fit/held-out replicate with exhaustive small-system references."""

    samples_per_root: int
    root_count: int
    fit_moments: SampledSubspaceMoments
    held_out_moments: SampledSubspaceMoments
    fit_ritz: FailClosedGeneralizedRitz
    exact_ritz: FailClosedGeneralizedRitz
    exact_ritz_energies: np.ndarray
    fit_ritz_energy_errors: np.ndarray
    maximum_fit_ritz_energy_error_per_site: float
    held_out_rayleigh_energies: np.ndarray
    exact_fitted_rayleigh_energies: np.ndarray
    held_out_rayleigh_energy_errors: np.ndarray
    maximum_held_out_energy_error_per_site: float
    held_out_normalized_residuals: np.ndarray
    exact_fitted_normalized_residuals: np.ndarray
    held_out_residual_absolute_errors: np.ndarray
    held_out_residual_relative_errors: np.ndarray
    maximum_held_out_residual_absolute_error_per_sqrt_site: float
    maximum_held_out_residual_relative_error: float
    held_out_hamiltonian_antihermiticity_relative: float
    fit_overlap_condition_number: float
    held_out_overlap_condition_number: float
    minimum_root_ratio_kish_ess_fraction: float
    maximum_root_importance_ratio: float


@dataclass(frozen=True)
class ScalarMetricSummary:
    """Deterministic replicate summary with percentile-bootstrap mean CI."""

    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    mean_ci95_lower: float
    mean_ci95_upper: float
    minimum: float
    q05: float
    median: float
    q95: float
    maximum: float


@dataclass(frozen=True)
class ForceComparison:
    """Scale and direction error between sampled and exhaustive forces."""

    exact_norm: float
    sampled_norm: float
    absolute_l2_error: float
    relative_l2_error: float
    cosine_similarity: float


def _validated_root_probabilities(root_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(root_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or min(probabilities.shape) < 1:
        raise ValueError("root_probabilities must have shape (basis, roots)")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("root probabilities must be finite and nonnegative")
    totals = np.sum(probabilities, axis=0)
    if np.any(totals <= 0.0):
        raise ValueError("every root probability must have positive mass")
    probabilities = probabilities / totals[None, :]
    return probabilities


def draw_equal_stratum_sample(
    root_probabilities: np.ndarray,
    samples_per_root: int,
    rng: np.random.Generator,
) -> EqualStratumSample:
    """Draw an equal number of categorical observations from every root.

    Equal root counts implement the uniform mixture expectation with uniform
    observation weights, while retaining exact independent streams per root.
    """

    probabilities = _validated_root_probabilities(root_probabilities)
    if isinstance(samples_per_root, bool) or int(samples_per_root) < 1:
        raise ValueError("samples_per_root must be a positive integer")
    samples_per_root = int(samples_per_root)
    basis_size, root_count = probabilities.shape
    indices = np.concatenate(
        [
            rng.choice(basis_size, size=samples_per_root, p=probabilities[:, root])
            for root in range(root_count)
        ]
    ).astype(np.int64, copy=False)
    root_ids = np.repeat(
        np.arange(root_count, dtype=np.int32), samples_per_root
    )
    return EqualStratumSample(
        basis_indices=indices,
        root_ids=root_ids,
        samples_per_root=samples_per_root,
        root_count=int(root_count),
    )


def _mixture_and_ratios(
    root_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _validated_root_probabilities(root_probabilities)
    mixture = np.mean(probabilities, axis=1)
    if np.any(mixture <= 0.0):
        raise ValueError("the common Born mixture must be strictly positive")
    ratios = probabilities / mixture[:, None]
    return mixture, ratios


def _sampled_rows(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    mixture: np.ndarray,
    sample: EqualStratumSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = sample.basis_indices
    return amplitudes[indices], actions[indices], mixture[indices]


def _condition_number(matrix: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    if values[0] <= 0.0:
        return float("inf")
    return float(values[-1] / values[0])


def _normalized_direct_residuals(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    proposal: np.ndarray,
    coefficients: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inverse_sqrt = 1.0 / np.sqrt(proposal)
    represented = (amplitudes * inverse_sqrt[:, None]) @ coefficients
    represented_actions = (actions * inverse_sqrt[:, None]) @ coefficients
    residual = represented_actions - represented * energies[None, :]
    norms_squared = np.mean(np.abs(represented) ** 2, axis=0)
    residuals_squared = np.mean(np.abs(residual) ** 2, axis=0)
    if np.any(norms_squared <= 0.0):
        raise FloatingPointError("a held-out fitted Ritz state has zero norm")
    return (
        np.sqrt(np.maximum(residuals_squared, 0.0) / norms_squared),
        norms_squared,
    )


def _exact_fitted_statistics(
    amplitudes: np.ndarray,
    actions: np.ndarray,
    coefficients: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    represented = amplitudes @ coefficients
    represented_actions = actions @ coefficients
    norms_squared = np.sum(np.abs(represented) ** 2, axis=0)
    if np.any(norms_squared <= 0.0):
        raise FloatingPointError("a fitted Ritz state has zero exhaustive norm")
    rayleigh = np.real(
        np.sum(represented.conj() * represented_actions, axis=0)
        / norms_squared
    )
    residual = represented_actions - represented * energies[None, :]
    normalized_residual = np.sqrt(
        np.maximum(np.sum(np.abs(residual) ** 2, axis=0), 0.0)
        / norms_squared
    )
    return rayleigh, normalized_residual


def _held_out_rayleigh_statistics(
    moments: SampledSubspaceMoments,
    coefficients: np.ndarray,
) -> np.ndarray:
    numerator = np.diag(
        coefficients.conj().T @ moments.hamiltonian_raw @ coefficients
    )
    denominator = np.diag(
        coefficients.conj().T @ moments.overlap @ coefficients
    )
    if np.any(np.real(denominator) <= 0.0):
        raise FloatingPointError("a held-out fitted Ritz norm is nonpositive")
    # A finite sample can leave a small imaginary diagonal because H_raw is
    # intentionally not Hermitized.  The real part is the unbiased Rayleigh
    # estimator; the discarded component remains covered by the separately
    # reported anti-Hermiticity diagnostic.
    return np.real(numerator / denominator)


def calibrate_fit_and_held_out_multistate(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    fit_sample: EqualStratumSample,
    held_out_sample: EqualStratumSample,
    *,
    n_sites: int,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float | None = None,
    residual_relative_floor: float = 1.0e-8,
) -> HeldOutMultistateCalibration:
    """Fit sampled Ritz roots, then audit them on independent observations."""

    states = np.asarray(amplitudes, dtype=np.complex128)
    actions = np.asarray(hamiltonian_actions, dtype=np.complex128)
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError("amplitudes and actions must share shape (basis, roots)")
    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    if fit_sample.root_count != states.shape[1] or held_out_sample.root_count != states.shape[1]:
        raise ValueError("sample root counts do not match the states")
    if fit_sample.samples_per_root != held_out_sample.samples_per_root:
        raise ValueError("fit and held-out samples must have equal stratum sizes")
    root_probabilities = np.abs(states) ** 2
    mixture, ratios = _mixture_and_ratios(root_probabilities)

    exhaustive = sampled_subspace_moments(
        states,
        actions,
        mixture,
        observation_weights=mixture,
    )
    exact_ritz = fail_closed_generalized_ritz(
        exhaustive,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
    )
    fit_rows = _sampled_rows(states, actions, mixture, fit_sample)
    held_rows = _sampled_rows(states, actions, mixture, held_out_sample)
    fit_moments = sampled_subspace_moments(*fit_rows)
    held_moments = sampled_subspace_moments(*held_rows)
    fit_ritz = fail_closed_generalized_ritz(
        fit_moments,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
    )

    exact_rayleigh, exact_residual = _exact_fitted_statistics(
        states,
        actions,
        fit_ritz.coefficients,
        fit_ritz.energies,
    )
    held_rayleigh = _held_out_rayleigh_statistics(
        held_moments, fit_ritz.coefficients
    )
    held_residual, _ = _normalized_direct_residuals(
        *held_rows,
        fit_ritz.coefficients,
        fit_ritz.energies,
    )
    residual_absolute_error = np.abs(held_residual - exact_residual)
    residual_relative_error = residual_absolute_error / np.maximum(
        exact_residual, residual_relative_floor
    )
    fit_energy_error = np.abs(fit_ritz.energies - exact_ritz.energies)
    held_energy_error = np.abs(held_rayleigh - exact_rayleigh)

    sampled_ratios = ratios[held_out_sample.basis_indices]
    ratio_sum = np.sum(sampled_ratios, axis=0)
    ratio_square_sum = np.sum(sampled_ratios**2, axis=0)
    sample_count = held_out_sample.sample_count
    ess_fraction = np.divide(
        ratio_sum**2,
        sample_count * ratio_square_sum,
        out=np.zeros_like(ratio_sum),
        where=ratio_square_sum > 0.0,
    )
    return HeldOutMultistateCalibration(
        samples_per_root=fit_sample.samples_per_root,
        root_count=fit_sample.root_count,
        fit_moments=fit_moments,
        held_out_moments=held_moments,
        fit_ritz=fit_ritz,
        exact_ritz=exact_ritz,
        exact_ritz_energies=np.asarray(exact_ritz.energies),
        fit_ritz_energy_errors=np.asarray(fit_energy_error),
        maximum_fit_ritz_energy_error_per_site=float(
            np.max(fit_energy_error) / n_sites
        ),
        held_out_rayleigh_energies=np.asarray(held_rayleigh),
        exact_fitted_rayleigh_energies=np.asarray(exact_rayleigh),
        held_out_rayleigh_energy_errors=np.asarray(held_energy_error),
        maximum_held_out_energy_error_per_site=float(
            np.max(held_energy_error) / n_sites
        ),
        held_out_normalized_residuals=np.asarray(held_residual),
        exact_fitted_normalized_residuals=np.asarray(exact_residual),
        held_out_residual_absolute_errors=np.asarray(residual_absolute_error),
        held_out_residual_relative_errors=np.asarray(residual_relative_error),
        maximum_held_out_residual_absolute_error_per_sqrt_site=float(
            np.max(residual_absolute_error) / np.sqrt(n_sites)
        ),
        maximum_held_out_residual_relative_error=float(
            np.max(residual_relative_error)
        ),
        held_out_hamiltonian_antihermiticity_relative=float(
            held_moments.hamiltonian_hermiticity.relative_frobenius_error
        ),
        fit_overlap_condition_number=float(
            fit_ritz.overlap_condition_number
        ),
        held_out_overlap_condition_number=_condition_number(
            held_moments.overlap
        ),
        minimum_root_ratio_kish_ess_fraction=float(np.min(ess_fraction)),
        maximum_root_importance_ratio=float(np.max(sampled_ratios)),
    )


def compare_flat_forces(
    exact_force: np.ndarray,
    sampled_force: np.ndarray,
    *,
    relative_floor: float = 1.0e-12,
) -> ForceComparison:
    """Compare real flattened forces without hiding a near-zero reference."""

    exact = np.asarray(exact_force, dtype=np.float64).reshape(-1)
    sampled = np.asarray(sampled_force, dtype=np.float64).reshape(-1)
    if exact.shape != sampled.shape or exact.size < 1:
        raise ValueError("force arrays must be nonempty and have equal shape")
    if np.any(~np.isfinite(exact)) or np.any(~np.isfinite(sampled)):
        raise ValueError("force arrays must be finite")
    exact_norm = float(np.linalg.norm(exact))
    sampled_norm = float(np.linalg.norm(sampled))
    error = float(np.linalg.norm(sampled - exact))
    denominator = max(exact_norm, float(relative_floor))
    if exact_norm <= relative_floor or sampled_norm <= relative_floor:
        cosine = float("nan")
    else:
        cosine = float(
            np.clip(np.dot(exact, sampled) / (exact_norm * sampled_norm), -1.0, 1.0)
        )
    return ForceComparison(
        exact_norm=exact_norm,
        sampled_norm=sampled_norm,
        absolute_l2_error=error,
        relative_l2_error=error / denominator,
        cosine_similarity=cosine,
    )


def summarize_scalar_metric(
    values: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = 4000,
) -> ScalarMetricSummary:
    """Summarize independent replicates and bootstrap the mean deterministically."""

    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 2 or np.any(~np.isfinite(data)):
        raise ValueError("at least two finite replicate values are required")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    rng = np.random.default_rng(int(bootstrap_seed))
    resampled = rng.choice(data, size=(bootstrap_resamples, data.size), replace=True)
    bootstrap_means = np.mean(resampled, axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    q05, median, q95 = np.quantile(data, (0.05, 0.5, 0.95))
    standard_deviation = float(np.std(data, ddof=1))
    return ScalarMetricSummary(
        count=int(data.size),
        mean=float(np.mean(data)),
        standard_deviation=standard_deviation,
        standard_error=standard_deviation / np.sqrt(data.size),
        mean_ci95_lower=float(lower),
        mean_ci95_upper=float(upper),
        minimum=float(np.min(data)),
        q05=float(q05),
        median=float(median),
        q95=float(q95),
        maximum=float(np.max(data)),
    )


__all__ = [
    "EqualStratumSample",
    "ForceComparison",
    "HeldOutMultistateCalibration",
    "ScalarMetricSummary",
    "calibrate_fit_and_held_out_multistate",
    "compare_flat_forces",
    "draw_equal_stratum_sample",
    "summarize_scalar_metric",
]
