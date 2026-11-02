"""Casimir-resolved error directions for block-spin distributions.

For an exact distribution ``p`` and a candidate ``q`` on the same physical
support, the signed error ``delta=q-p`` is decomposed in the Pearson/Fisher
metric.  The first component is the unique tangent direction that reproduces
all errors in the first moments of the measured block Casimirs.  The
orthogonal component changes only correlations and higher moments of the
same Casimir-valued full-counting distribution.

This is an error diagnostic, not a phase classifier.  In particular,
"Casimir-orthogonal" never means that the full FCS contains a non-Casimir
observable; every outcome is still labelled by block total spin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


FCSKey = tuple[int, ...]


def block_casimir_features(outcomes: Sequence[FCSKey]) -> np.ndarray:
    """Return one ``j(j+1)`` feature per measured doubled-spin label."""

    keys = tuple(tuple(int(q) for q in outcome) for outcome in outcomes)
    if not keys:
        raise ValueError("outcomes must be nonempty")
    arity = len(keys[0])
    if arity < 1 or any(len(key) != arity for key in keys):
        raise ValueError("all FCS outcomes must have the same positive arity")
    if any(q < 0 for key in keys for q in key):
        raise ValueError("doubled spins must be nonnegative")
    return np.asarray(
        [[q * (q + 2) / 4.0 for q in key] for key in keys],
        dtype=np.float64,
    )


def _distribution_vector(
    distribution: Mapping[FCSKey, float], outcomes: Sequence[FCSKey], name: str
) -> np.ndarray:
    keys = tuple(outcomes)
    if set(distribution) - set(keys):
        raise ValueError(f"{name} contains outcomes outside the requested support")
    values = np.asarray([float(distribution.get(key, 0.0)) for key in keys])
    if np.any(~np.isfinite(values)) or np.min(values) < -1.0e-14:
        raise ValueError(f"{name} must be finite and nonnegative")
    values = np.clip(values, 0.0, None)
    if abs(float(np.sum(values)) - 1.0) > 2.0e-10:
        raise ValueError(f"{name} must be normalized")
    return values / np.sum(values)


def casimir_error_tomography(
    reference: Mapping[FCSKey, float],
    candidate: Mapping[FCSKey, float],
    *,
    outcomes: Sequence[FCSKey] | None = None,
    probability_floor: float = 0.0,
    pseudoinverse_rcond: float = 1.0e-12,
) -> dict[str, object]:
    """Orthogonally split a finite FCS error into moment and shape parts.

    Outcomes with exact reference probability at or below ``probability_floor``
    are excluded and their candidate mass is reported.  A primary analysis
    should use a mathematically fixed support and floor zero; nonzero floors
    are sensitivity controls only.
    """

    if probability_floor < 0.0:
        raise ValueError("probability_floor must be nonnegative")
    if not 0.0 < pseudoinverse_rcond < 1.0:
        raise ValueError("pseudoinverse_rcond must lie in (0,1)")
    keys = tuple(
        sorted(set(reference) | set(candidate))
        if outcomes is None
        else tuple(tuple(int(q) for q in key) for key in outcomes)
    )
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("the requested support must be nonempty and unique")
    p_full = _distribution_vector(reference, keys, "reference")
    q_full = _distribution_vector(candidate, keys, "candidate")
    active = p_full > float(probability_floor)
    if not np.any(active):
        raise ValueError("probability floor removed the complete reference support")
    inactive_candidate_mass = float(np.sum(q_full[~active]))
    if inactive_candidate_mass > 1.0e-10:
        raise ValueError("candidate has material mass outside the active reference support")

    p = p_full[active]
    q = q_full[active]
    # Renormalization changes only roundoff when the support is physical and
    # complete.  It keeps the tangent sum rule explicit under sensitivity
    # floors that remove negligible reference bins.
    p /= np.sum(p)
    q /= np.sum(q)
    delta = q - p
    features = block_casimir_features(keys)[active]
    mean_features = p @ features
    centered = features - mean_features
    covariance = (centered.T * p) @ centered
    moment_error = delta @ features
    inverse = np.linalg.pinv(covariance, rcond=pseudoinverse_rcond)
    coefficients = inverse @ moment_error
    moment_direction = p * (centered @ coefficients)
    shape_direction = delta - moment_direction

    def fisher_norm_squared(vector: np.ndarray) -> float:
        return float(np.sum(vector**2 / p))

    total_information = fisher_norm_squared(delta)
    moment_information = fisher_norm_squared(moment_direction)
    shape_information = fisher_norm_squared(shape_direction)
    cross = float(np.sum(moment_direction * shape_direction / p))
    shape_moments = shape_direction @ features
    covariance_eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    rank_threshold = pseudoinverse_rcond * max(
        float(np.max(np.abs(covariance_eigenvalues), initial=0.0)), 1.0
    )
    return {
        "outcomes": [list(key) for key, keep in zip(keys, active, strict=True) if keep],
        "active_support_size": int(np.count_nonzero(active)),
        "inactive_support_size": int(np.count_nonzero(~active)),
        "inactive_candidate_mass": inactive_candidate_mass,
        "probability_floor": float(probability_floor),
        "pseudoinverse_rcond": float(pseudoinverse_rcond),
        "feature_rank": int(np.count_nonzero(covariance_eigenvalues > rank_threshold)),
        "reference_feature_means": [float(value) for value in mean_features],
        "candidate_minus_reference_feature_means": [
            float(value) for value in moment_error
        ],
        "total_pearson_information": total_information,
        "first_casimir_moment_information": moment_information,
        "casimir_orthogonal_shape_information": shape_information,
        "shape_fraction": shape_information / total_information
        if total_information > 0.0
        else 0.0,
        "shape_total_variation_tangent": float(0.5 * np.sum(np.abs(shape_direction))),
        "maximum_probability_reconstruction_error": float(
            np.max(np.abs(delta - moment_direction - shape_direction), initial=0.0)
        ),
        "normalization_error": float(abs(np.sum(delta))),
        "shape_normalization_error": float(abs(np.sum(shape_direction))),
        "maximum_shape_casimir_moment_error": float(
            np.max(np.abs(shape_moments), initial=0.0)
        ),
        "fisher_pythagorean_error": float(
            abs(total_information - moment_information - shape_information)
        ),
        "fisher_cross_term": cross,
        "signed_error": [float(value) for value in delta],
        "first_moment_direction": [float(value) for value in moment_direction],
        "casimir_orthogonal_shape_direction": [float(value) for value in shape_direction],
    }


__all__ = ["block_casimir_features", "casimir_error_tomography"]
