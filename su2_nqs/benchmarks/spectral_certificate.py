"""Residual-gap certificates for variational spectral subspaces.

These helpers do not compute an exact spectrum.  They turn an independently
established lower bound on the distance to every level outside a target
window into a conservative lower bound on the variational state's weight in
that window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResidualGapCertificate:
    """Conservative target-window weight certificate."""

    residual_upper: float
    complement_gap_lower: float
    residual_to_gap_upper: float
    raw_weight_lower_bound: float
    physical_weight_lower_bound: float
    required_weight: float
    passes: bool


def residual_gap_reference_weight_certificate(
    residual_upper: float,
    complement_gap_lower: float,
    *,
    required_weight: float = 0.9,
) -> ResidualGapCertificate:
    """Certify target-window weight from conservative residual/gap bounds.

    For a normalized state with ``E=<H>``, let ``W`` be an exact spectral
    window, ``r=||(H-E)psi||``, and ``d=min_{lambda outside W}|lambda-E|``.
    Then ``<psi|P_W|psi> >= 1-r**2/d**2``.  Callers must supply an upper
    confidence bound for ``r`` and a lower confidence bound for ``d``.
    """

    residual = float(residual_upper)
    gap = float(complement_gap_lower)
    threshold = float(required_weight)
    if not math.isfinite(residual) or residual < 0.0:
        raise ValueError("residual_upper must be finite and non-negative")
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("complement_gap_lower must be finite and positive")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("required_weight must lie in [0, 1]")
    ratio = residual / gap
    raw = 1.0 - ratio * ratio
    physical = min(1.0, max(0.0, raw))
    return ResidualGapCertificate(
        residual_upper=residual,
        complement_gap_lower=gap,
        residual_to_gap_upper=ratio,
        raw_weight_lower_bound=raw,
        physical_weight_lower_bound=physical,
        required_weight=threshold,
        passes=raw >= threshold,
    )


__all__ = [
    "ResidualGapCertificate",
    "residual_gap_reference_weight_certificate",
]
