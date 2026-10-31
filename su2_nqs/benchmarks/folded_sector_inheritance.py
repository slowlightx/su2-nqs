"""Fail-closed sector inheritance for residual and folded-spectrum targets.

This module answers a narrow question relevant to an unprojected ``q=0``
calculation: when may an independently known exact spectral window transfer
its ``Gamma.A1`` label to an approximate variational state?  Energy targeting
alone never transfers the label.  The window must be isolated from *all*
other ``q=0`` sectors and every exact eigenvector in the certified window must
carry the requested sector.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def residual_sector_inheritance_audit(
    *,
    residual_norm_upper: float,
    complement_distance_lower: float,
    target_window_all_in_requested_sector: bool,
    target_window_dimension: int,
    requested_sector: str = "Gamma.A1",
    requested_minimum_weight: float = 0.9,
) -> dict[str, object]:
    """Certify sector weight from a residual-gap spectral-window bound.

    For normalized ``psi`` with ``E=<H>`` and
    ``r=||(H-E)psi||``, let ``d`` be a rigorous lower bound on the distance
    from ``E`` to every eigenvalue outside the target window.  Then
    ``<psi|P_window|psi> >= 1-(r/d)^2``.  Sector inheritance follows only if
    the entire target window belongs to ``requested_sector``.
    """

    residual = float(residual_norm_upper)
    distance = float(complement_distance_lower)
    threshold = float(requested_minimum_weight)
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("residual_norm_upper must be finite and nonnegative")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("complement_distance_lower must be finite and positive")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("requested_minimum_weight must lie in (0,1]")
    if (
        isinstance(target_window_dimension, bool)
        or not isinstance(target_window_dimension, Integral)
        or target_window_dimension < 1
    ):
        raise ValueError("target_window_dimension must be a positive integer")
    ratio = residual / distance
    lower = max(0.0, 1.0 - ratio**2)
    sector_pure = bool(target_window_all_in_requested_sector)
    return {
        "certificate": "residual_gap_window",
        "requested_sector": str(requested_sector),
        "residual_norm_upper": residual,
        "complement_distance_lower": distance,
        "residual_to_gap_ratio_upper": ratio,
        "target_window_dimension": int(target_window_dimension),
        "target_window_all_in_requested_sector": sector_pure,
        "target_window_weight_lower": lower,
        "requested_sector_weight_lower": lower if sector_pure else 0.0,
        "sector_inheritance_pass": sector_pure and lower >= threshold,
        "individual_root_ownership_pass": (
            sector_pure and int(target_window_dimension) == 1 and lower >= threshold
        ),
        "interpretation": (
            "individual_eigenroot"
            if sector_pure and int(target_window_dimension) == 1
            else "same-sector_subspace_only"
            if sector_pure
            else "energy_window_contains_or_may_contain_other_sectors"
        ),
    }


def folded_spectrum_sector_inheritance_audit(
    *,
    folded_expectation_upper: float,
    complement_shift_distance_lower: float,
    target_shift_distance_upper: float,
    target_window_all_in_requested_sector: bool,
    target_window_dimension: int,
    requested_sector: str = "Gamma.A1",
    requested_minimum_weight: float = 0.9,
) -> dict[str, object]:
    """Audit a target selected with ``<(H-sigma)^2>``.

    ``complement_shift_distance_lower`` must exclude every eigenvalue outside
    the target window across the complete working ``q`` sector, not merely
    eigenvalues in ``Gamma.A1``.  The folded objective orders the requested
    window ahead of its complement only when the largest target distance from
    ``sigma`` is strictly smaller than the smallest complement distance.

    The rigorous bound used here is
    ``weight(window) >= 1-F_upper/d_out_lower^2``.  A degenerate window can
    certify its subspace but cannot assign an individual root.
    """

    folded = float(folded_expectation_upper)
    outside = float(complement_shift_distance_lower)
    inside = float(target_shift_distance_upper)
    threshold = float(requested_minimum_weight)
    if not np.isfinite(folded) or folded < 0.0:
        raise ValueError("folded_expectation_upper must be finite and nonnegative")
    if not np.isfinite(outside) or outside <= 0.0:
        raise ValueError(
            "complement_shift_distance_lower must be finite and positive"
        )
    if not np.isfinite(inside) or inside < 0.0:
        raise ValueError("target_shift_distance_upper must be finite and nonnegative")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("requested_minimum_weight must lie in (0,1]")
    if (
        isinstance(target_window_dimension, bool)
        or not isinstance(target_window_dimension, Integral)
        or target_window_dimension < 1
    ):
        raise ValueError("target_window_dimension must be a positive integer")
    spectral_ordering = inside < outside
    lower = max(0.0, 1.0 - folded / outside**2)
    sector_pure = bool(target_window_all_in_requested_sector)
    passed = spectral_ordering and sector_pure and lower >= threshold
    return {
        "certificate": "folded_spectrum_global_q_sector_window",
        "requested_sector": str(requested_sector),
        "folded_expectation_upper": folded,
        "complement_shift_distance_lower": outside,
        "target_shift_distance_upper": inside,
        "target_is_strictly_closest_window": spectral_ordering,
        "target_window_dimension": int(target_window_dimension),
        "target_window_all_in_requested_sector": sector_pure,
        "target_window_weight_lower": lower,
        "requested_sector_weight_lower": lower if sector_pure else 0.0,
        "sector_inheritance_pass": passed,
        "individual_root_ownership_pass": (
            passed and int(target_window_dimension) == 1
        ),
        "requires_full_working_sector_exclusion": True,
        "interpretation": (
            "individual_eigenroot"
            if passed and int(target_window_dimension) == 1
            else "same-sector_subspace_only"
            if passed
            else "no_sector_inheritance"
        ),
    }


def exact_diagonal_folded_quantities(
    eigenvalues: np.ndarray, amplitudes: np.ndarray, shift: float
) -> dict[str, float]:
    """Independent diagonal reference used by exact-small tests."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    state = np.asarray(amplitudes, dtype=np.complex128)
    if values.ndim != 1 or state.shape != values.shape:
        raise ValueError("eigenvalues and amplitudes must be equal-length vectors")
    norm = float(np.real(np.vdot(state, state)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("amplitudes must have finite nonzero norm")
    probability = np.abs(state) ** 2 / norm
    energy = float(np.sum(probability * values))
    residual_squared = float(np.sum(probability * (values - energy) ** 2))
    folded = float(np.sum(probability * (values - float(shift)) ** 2))
    return {
        "energy": energy,
        "residual_norm": float(np.sqrt(max(residual_squared, 0.0))),
        "folded_expectation": folded,
    }


__all__ = [
    "exact_diagonal_folded_quantities",
    "folded_spectrum_sector_inheritance_audit",
    "residual_sector_inheritance_audit",
]
