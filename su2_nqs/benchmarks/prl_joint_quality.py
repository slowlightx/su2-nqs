"""Fail-closed quality and sampled-variance metrics for the P8 route.

The public helpers in this module are deliberately model agnostic.  They keep
the physical target (energy, variance, scalar observables, and collective-spin
FCS) separate from symmetry certification.  This distinction prevents an
exactly constrained SU(2) state from winning a compute comparison solely
because its non-singlet weight is identically zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointQualityThresholds:
    """Positive denominators defining the two P8 joint targets."""

    energy_error_per_site: float = 0.02
    v_score: float = 0.2
    scalar_normalized_error: float = 0.02
    fcs_total_variation: float = 0.05
    fcs_hellinger: float = 0.05
    fcs_casimir_normalized_error: float = 0.02
    non_singlet_weight: float = 0.05
    wrong_spatial_sector_weight: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            _positive_finite(value, name)


@dataclass(frozen=True)
class VScoreEstimate:
    """One iid plug-in estimate of ``N Var(H) / E**2``.

    ``imaginary_energy`` is retained as an audit quantity rather than forced
    to zero.  Even when the exact expectation of a Hermitian Hamiltonian is
    real, a finite iid sample of generally complex local energies has a
    fluctuating imaginary sample mean.
    """

    sample_count: int
    energy: float
    variance: float
    v_score: float
    standard_error: float
    imaginary_energy: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly representation."""

        return {
            "sample_count": self.sample_count,
            "energy": self.energy,
            "variance": self.variance,
            "v_score": self.v_score,
            "standard_error": self.standard_error,
            "imaginary_energy": self.imaginary_energy,
        }


@dataclass(frozen=True)
class ImportanceVScoreEstimate:
    """Self-normalized importance estimate of the P8 V-score.

    The proposal-to-target correction is supplied as ``p / q``.  The
    effective sample size and weight extrema are audit quantities; they do
    not replace exact-small closure tests.
    """

    sample_count: int
    energy: float
    variance: float
    v_score: float
    standard_error: float
    imaginary_energy: float
    mean_importance_weight: float
    effective_sample_size: float
    minimum_importance_weight: float
    maximum_importance_weight: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly representation."""

        return {
            "sample_count": self.sample_count,
            "energy": self.energy,
            "variance": self.variance,
            "v_score": self.v_score,
            "standard_error": self.standard_error,
            "imaginary_energy": self.imaginary_energy,
            "mean_importance_weight": self.mean_importance_weight,
            "effective_sample_size": self.effective_sample_size,
            "minimum_importance_weight": self.minimum_importance_weight,
            "maximum_importance_weight": self.maximum_importance_weight,
        }


def _positive_finite(value: Real, name: str) -> float:
    answer = float(value)
    if not np.isfinite(answer) or answer <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return answer


def _nonnegative_finite(value: Real, name: str) -> float:
    answer = float(value)
    if not np.isfinite(answer) or answer < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return answer


def complete_joint_quality(
    *,
    energy_error_per_site: Real,
    v_score: Real,
    spectral_normalized_scalar_errors: Mapping[str, Real],
    fcs_total_variation: Real,
    fcs_hellinger: Real,
    spectral_normalized_fcs_casimir_error: Real,
    native_non_singlet_weight: Real,
    native_wrong_spatial_sector_weight: Real = 0.0,
    thresholds: JointQualityThresholds | None = None,
) -> dict[str, Any]:
    """Return non-compensating physical and certified joint scores.

    ``Q_phys`` contains no hard-symmetry term.  ``Q_cert`` is the maximum of
    ``Q_phys`` and the native total-spin and spatial-sector leakage scores.
    Every scalar channel is retained separately so an RMS average cannot hide
    a failed observable.
    """

    limits = JointQualityThresholds() if thresholds is None else thresholds
    if not isinstance(limits, JointQualityThresholds):
        raise TypeError("thresholds must be JointQualityThresholds")
    if not spectral_normalized_scalar_errors:
        raise ValueError("at least one scalar observable error is required")

    scalar_scores = {
        str(name): _nonnegative_finite(value, f"scalar error {name!r}")
        / limits.scalar_normalized_error
        for name, value in spectral_normalized_scalar_errors.items()
    }
    if len(scalar_scores) != len(spectral_normalized_scalar_errors):
        raise ValueError("scalar observable names must be unique as strings")

    physical_components = {
        "energy": _nonnegative_finite(
            energy_error_per_site, "energy_error_per_site"
        )
        / limits.energy_error_per_site,
        "V": _nonnegative_finite(v_score, "v_score") / limits.v_score,
        **{f"scalar:{name}": value for name, value in scalar_scores.items()},
        "FCS_TV": _nonnegative_finite(
            fcs_total_variation, "fcs_total_variation"
        )
        / limits.fcs_total_variation,
        "FCS_Hellinger": _nonnegative_finite(fcs_hellinger, "fcs_hellinger")
        / limits.fcs_hellinger,
        "FCS_Casimir": _nonnegative_finite(
            spectral_normalized_fcs_casimir_error,
            "spectral_normalized_fcs_casimir_error",
        )
        / limits.fcs_casimir_normalized_error,
    }
    certification_components = {
        "non_singlet_weight": _nonnegative_finite(
            native_non_singlet_weight, "native_non_singlet_weight"
        )
        / limits.non_singlet_weight,
        "wrong_spatial_sector_weight": _nonnegative_finite(
            native_wrong_spatial_sector_weight,
            "native_wrong_spatial_sector_weight",
        )
        / limits.wrong_spatial_sector_weight,
    }
    q_phys = max(physical_components.values())
    q_cert = max(q_phys, *certification_components.values())
    return {
        "Q_phys": float(q_phys),
        "Q_cert": float(q_cert),
        "physical_components": physical_components,
        "certification_components": certification_components,
        "worst_physical_terms": sorted(
            name
            for name, value in physical_components.items()
            if np.isclose(value, q_phys, rtol=0.0, atol=1.0e-15)
        ),
        "worst_certified_terms": sorted(
            [
                *(name for name, value in physical_components.items() if np.isclose(value, q_cert, rtol=0.0, atol=1.0e-15)),
                *(name for name, value in certification_components.items() if np.isclose(value, q_cert, rtol=0.0, atol=1.0e-15)),
            ]
        ),
    }


def confirmed_quality_crossing(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: Real = 1.0,
    confirmation_points: int = 3,
    step_key: str = "step",
) -> dict[str, int | float] | None:
    """Return the first confirmed consecutive crossing without interpolation."""

    limit = _positive_finite(threshold, "threshold")
    if isinstance(confirmation_points, bool) or not isinstance(
        confirmation_points, Integral
    ):
        raise TypeError("confirmation_points must be an integer")
    count = int(confirmation_points)
    if count < 1:
        raise ValueError("confirmation_points must be positive")
    if not trajectory:
        return None

    run_start: int | None = None
    previous_step: float | None = None
    for index, row in enumerate(trajectory):
        if score_key not in row or step_key not in row:
            raise KeyError(f"trajectory rows require {score_key!r} and {step_key!r}")
        score = _nonnegative_finite(row[score_key], score_key)
        step = float(row[step_key])
        if not np.isfinite(step) or (previous_step is not None and step <= previous_step):
            raise ValueError("trajectory steps must be finite and strictly increasing")
        previous_step = step
        if score <= limit:
            run_start = index if run_start is None else run_start
            if index - run_start + 1 == count:
                first = trajectory[run_start]
                return {
                    "first_qualifying_index": run_start,
                    "confirmation_index": index,
                    "first_qualifying_step": float(first[step_key]),
                    "confirmation_step": step,
                }
        else:
            run_start = None
    return None


def _validated_local_energy_distribution(
    probabilities: Sequence[Real], local_energies: Sequence[complex]
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(probabilities, dtype=np.float64)
    values = np.asarray(local_energies, dtype=np.complex128)
    if weights.ndim != 1 or values.shape != weights.shape or len(weights) < 2:
        raise ValueError("probabilities and local energies must be matching vectors")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if np.any(~np.isfinite(values)):
        raise ValueError("local energies must be finite")
    if abs(float(np.sum(weights)) - 1.0) > 1.0e-10:
        raise ValueError("probabilities must be normalized")
    if np.count_nonzero(weights > 0.0) < 2:
        raise ValueError("the distribution must have at least two active outcomes")
    return weights, values


def exact_v_score(
    probabilities: Sequence[Real],
    local_energies: Sequence[complex],
    *,
    n_sites: int,
) -> dict[str, float]:
    """Evaluate the exact distributional V-score ``N Var(H) / E**2``."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    weights, values = _validated_local_energy_distribution(
        probabilities, local_energies
    )
    complex_energy = complex(weights @ values)
    if abs(complex_energy.imag) > 1.0e-10:
        raise ValueError("the local-energy expectation must be real")
    energy = float(complex_energy.real)
    if abs(energy) <= 1.0e-14:
        raise ValueError("V-score is undefined at zero energy")
    variance = float(weights @ np.abs(values - complex_energy) ** 2)
    return {
        "energy": energy,
        "variance": variance,
        "v_score": float(int(n_sites) * variance / energy**2),
    }


def sampled_v_score(
    local_energy_samples: Sequence[complex], *, n_sites: int
) -> VScoreEstimate:
    """Estimate a V-score and iid delta-method standard error from samples."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    values = np.asarray(local_energy_samples, dtype=np.complex128)
    if values.ndim != 1 or len(values) < 3 or np.any(~np.isfinite(values)):
        raise ValueError("at least three finite local-energy samples are required")
    complex_energy = complex(np.mean(values))
    energy = float(complex_energy.real)
    if abs(energy) <= 1.0e-14:
        raise ValueError("V-score is undefined at zero sampled energy")
    # Center on the real energy estimator.  The imaginary part of a complex
    # local energy is stochastic at finite sample size and contributes to the
    # physical second central moment; centering on the complex sample mean
    # would remove that fluctuation and bias the variance downward.
    squared = np.abs(values - energy) ** 2
    variance = float(np.mean(squared))
    size = int(n_sites)
    estimate = float(size * variance / energy**2)
    influence = size * (
        (squared - variance) / energy**2
        - 2.0 * variance * (values.real - energy) / energy**3
    )
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(values)))
    return VScoreEstimate(
        sample_count=len(values),
        energy=energy,
        variance=variance,
        v_score=estimate,
        standard_error=standard_error,
        imaginary_energy=float(complex_energy.imag),
    )


def draw_sampled_v_score(
    probabilities: Sequence[Real],
    local_energies: Sequence[complex],
    *,
    n_sites: int,
    n_samples: int,
    seed: int,
) -> VScoreEstimate:
    """Draw iid support indices and return :func:`sampled_v_score`."""

    weights, values = _validated_local_energy_distribution(
        probabilities, local_energies
    )
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer")
    if int(n_samples) < 3:
        raise ValueError("n_samples must be at least three")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(weights), size=int(n_samples), p=weights)
    return sampled_v_score(values[indices], n_sites=n_sites)


def importance_sampled_v_score(
    local_energy_samples: Sequence[complex],
    importance_weights: Sequence[Real],
    *,
    n_sites: int,
) -> ImportanceVScoreEstimate:
    """Estimate the V-score from proposal samples and exact ``p / q`` weights.

    Self-normalization makes an exact eigenstate return zero variance even
    when proposal weights fluctuate.  The standard error is the iid
    delta-method estimate for the corresponding ratio estimators.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    values = np.asarray(local_energy_samples, dtype=np.complex128)
    weights = np.asarray(importance_weights, dtype=np.float64)
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or len(values) < 3
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            "at least three finite local energies and matching non-negative "
            "importance weights with positive total weight are required"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        total_weight = float(np.sum(weights))
        squared_weight_sum = float(np.sum(weights**2))
    if (
        not np.isfinite(total_weight)
        or not np.isfinite(squared_weight_sum)
        or total_weight <= 0.0
        or squared_weight_sum <= 0.0
    ):
        raise ValueError("importance-weight sums must be positive and finite")
    normalized = weights / total_weight
    complex_energy = complex(normalized @ values)
    energy = float(complex_energy.real)
    if abs(energy) <= 1.0e-14:
        raise ValueError("V-score is undefined at zero sampled energy")
    squared = np.abs(values - energy) ** 2
    variance = float(normalized @ squared)
    size = int(n_sites)
    estimate = float(size * variance / energy**2)
    mean_weight = float(np.mean(weights))
    second_moment = variance + energy**2
    influence = size * (weights / mean_weight) * (
        (np.abs(values) ** 2 - second_moment) / energy**2
        - 2.0 * second_moment * (values.real - energy) / energy**3
    )
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(values)))
    effective_size = float((total_weight / np.sqrt(squared_weight_sum)) ** 2)
    if not all(
        np.isfinite(value)
        for value in (energy, variance, estimate, standard_error, effective_size)
    ):
        raise FloatingPointError("importance V-score statistics are not finite")
    return ImportanceVScoreEstimate(
        sample_count=len(values),
        energy=energy,
        variance=variance,
        v_score=estimate,
        standard_error=standard_error,
        imaginary_energy=float(complex_energy.imag),
        mean_importance_weight=mean_weight,
        effective_sample_size=effective_size,
        minimum_importance_weight=float(np.min(weights)),
        maximum_importance_weight=float(np.max(weights)),
    )


def draw_uniform_mixture_v_score(
    probabilities: Sequence[Real],
    local_energies: Sequence[complex],
    *,
    n_sites: int,
    n_samples: int,
    seed: int,
    uniform_fraction: Real,
) -> ImportanceVScoreEstimate:
    """Draw from ``q=(1-eta)*p+eta*Uniform`` on an exact finite support.

    This exact-small implementation is intended to validate the tail-aware
    estimator.  A scalable runner must independently implement uniform draws
    over the constrained support and evaluate the same known proposal mass.
    """

    weights, values = _validated_local_energy_distribution(
        probabilities, local_energies
    )
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer")
    if int(n_samples) < 3:
        raise ValueError("n_samples must be at least three")
    eta = float(uniform_fraction)
    if not np.isfinite(eta) or eta < 0.0 or eta > 1.0:
        raise ValueError("uniform_fraction must lie in [0, 1]")
    proposal = (1.0 - eta) * weights + eta / len(weights)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(weights), size=int(n_samples), p=proposal)
    ratios = weights[indices] / proposal[indices]
    return importance_sampled_v_score(
        values[indices], ratios, n_sites=n_sites
    )


def v_score_closure(
    *,
    exact: Real,
    estimate: Real,
    standard_error: Real,
    sigma: Real = 2.0,
    relative_tolerance: Real = 0.05,
    zero_absolute_tolerance: Real = 1.0e-12,
) -> dict[str, float | bool]:
    """Apply the conjunctive P8D0 ``sigma`` and relative-error closure gate."""

    truth = _nonnegative_finite(exact, "exact")
    sampled = _nonnegative_finite(estimate, "estimate")
    error = _nonnegative_finite(standard_error, "standard_error")
    sigma_value = _positive_finite(sigma, "sigma")
    relative_limit = _positive_finite(relative_tolerance, "relative_tolerance")
    zero_limit = _positive_finite(zero_absolute_tolerance, "zero_absolute_tolerance")
    difference = abs(sampled - truth)
    sigma_pass = difference <= sigma_value * error
    if truth <= zero_limit:
        relative_error = 0.0 if difference <= zero_limit else np.inf
        relative_pass = difference <= zero_limit
    else:
        relative_error = difference / truth
        relative_pass = relative_error <= relative_limit
    return {
        "absolute_error": float(difference),
        "relative_error": float(relative_error),
        "sigma_error": float(difference / error) if error > 0.0 else (0.0 if difference == 0.0 else np.inf),
        "sigma_pass": bool(sigma_pass),
        "relative_pass": bool(relative_pass),
        "pass": bool(sigma_pass and relative_pass),
    }


__all__ = [
    "ImportanceVScoreEstimate",
    "JointQualityThresholds",
    "VScoreEstimate",
    "complete_joint_quality",
    "confirmed_quality_crossing",
    "draw_sampled_v_score",
    "draw_uniform_mixture_v_score",
    "exact_v_score",
    "importance_sampled_v_score",
    "sampled_v_score",
    "v_score_closure",
]
