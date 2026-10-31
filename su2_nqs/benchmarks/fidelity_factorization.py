"""Exact-small fidelity budgets in a common computational basis.

The routines in this module separate a state overlap into a Born-magnitude
factor and an amplitude-weighted relative-phase coherence.  They are basis
dependent diagnostics, not a sign-problem measure.  Sector projection is
kept explicit so that a raw fixed-``Sz`` state has the exact product

``F_raw = w_sector * F_amplitude * C_phase``.

All input states are normalized internally in ``complex128``.  A zero factor
is represented as a genuinely non-evaluable logarithm; no numerical floor is
inserted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FidelityFactors:
    """Magnitude/phase factorization of one normalized state overlap."""

    fidelity: float
    bhattacharyya_coefficient: float
    amplitude_fidelity: float
    phase_coherence: float | None
    weighted_phase_coherence: float | None
    phase_ratio_weighted_error: float | None
    absolute_closure_error: float | None
    relative_closure_error: float | None


@dataclass(frozen=True)
class SectorFidelityBudget:
    """Raw-to-sector fidelity factorization and its additive log budget."""

    sector_weight: float
    projected_state: np.ndarray | None
    raw_fidelity: float
    conditional_fidelity: float | None
    factors: FidelityFactors | None
    raw_conditional_closure_error: float | None
    raw_product_absolute_error: float | None
    raw_product_relative_error: float | None
    log_evaluable: bool
    negative_log_raw_fidelity: float | None
    negative_log_sector_weight: float | None
    negative_log_amplitude_fidelity: float | None
    negative_log_phase_coherence: float | None
    log_closure_error: float | None
    non_evaluable_reason: str | None


@dataclass(frozen=True)
class HybridStates:
    """Computational-basis amplitude/phase counterfactual square.

    Pointwise hybrids need not remain in the same total-spin sector even when
    both endpoints do.  The caller must therefore audit their ``S^2`` and
    sector weight before interpreting an energy.
    """

    evaluable: bool
    exact: np.ndarray
    candidate: np.ndarray
    candidate_amplitude_exact_phase: np.ndarray | None
    exact_amplitude_candidate_phase: np.ndarray | None
    candidate_mass_on_exact_phase_zeros: float
    exact_mass_on_candidate_phase_zeros: float
    non_evaluable_reason: str | None


def normalize_state(
    state: Sequence[complex] | np.ndarray,
    *,
    name: str = "state",
) -> np.ndarray:
    """Return a finite normalized ``complex128`` vector."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be one nonempty vector")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm_squared = float(np.real(np.vdot(vector, vector)))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError(f"{name} must have positive finite norm")
    return vector / math.sqrt(norm_squared)


def _relative_error(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) / scale if scale > 0.0 else 0.0


def fidelity_factors(
    reference: Sequence[complex] | np.ndarray,
    candidate: Sequence[complex] | np.ndarray,
) -> FidelityFactors:
    """Factor fidelity into Born-magnitude overlap and phase coherence.

    For normalized vectors ``g`` and ``psi`` in the same basis,

    ``B = sum_x |g_x| |psi_x|``, ``F_amp = B**2``, and
    ``C_phase = |<g|psi>|**2 / F_amp`` when ``F_amp > 0``.

    ``C_phase`` is independently reconstructed as the squared magnitude of
    the relative-phase average weighted by ``|g_x||psi_x|/B``.  Components
    with zero product weight require no phase convention.  If ``F_amp`` is
    exactly zero, phase coherence is undefined and returned as ``None``.
    """

    ground = normalize_state(reference, name="reference")
    state = normalize_state(candidate, name="candidate")
    if ground.shape != state.shape:
        raise ValueError("reference and candidate dimensions differ")

    overlap = np.vdot(ground, state)
    fidelity = float(abs(overlap) ** 2)
    magnitude_products = np.abs(ground) * np.abs(state)
    coefficient = float(math.fsum(float(value) for value in magnitude_products))
    amplitude_fidelity = float(coefficient * coefficient)
    if amplitude_fidelity == 0.0:
        return FidelityFactors(
            fidelity=fidelity,
            bhattacharyya_coefficient=coefficient,
            amplitude_fidelity=amplitude_fidelity,
            phase_coherence=None,
            weighted_phase_coherence=None,
            phase_ratio_weighted_error=None,
            absolute_closure_error=None,
            relative_closure_error=None,
        )

    phase_coherence = float(fidelity / amplitude_fidelity)
    active = magnitude_products > 0.0
    relative_units = np.conjugate(ground[active]) * state[active]
    relative_units /= magnitude_products[active]
    real_average = math.fsum(
        float(weight * unit.real)
        for weight, unit in zip(
            magnitude_products[active], relative_units, strict=True
        )
    ) / coefficient
    imaginary_average = math.fsum(
        float(weight * unit.imag)
        for weight, unit in zip(
            magnitude_products[active], relative_units, strict=True
        )
    ) / coefficient
    weighted_phase = float(real_average**2 + imaginary_average**2)
    reconstructed = float(amplitude_fidelity * weighted_phase)
    return FidelityFactors(
        fidelity=fidelity,
        bhattacharyya_coefficient=coefficient,
        amplitude_fidelity=amplitude_fidelity,
        phase_coherence=phase_coherence,
        weighted_phase_coherence=weighted_phase,
        phase_ratio_weighted_error=abs(phase_coherence - weighted_phase),
        absolute_closure_error=abs(fidelity - reconstructed),
        relative_closure_error=_relative_error(fidelity, reconstructed),
    )


def sector_fidelity_budget(
    reference_in_sector: Sequence[complex] | np.ndarray,
    raw_candidate: Sequence[complex] | np.ndarray,
    sector_columns: Sequence[Sequence[complex]] | np.ndarray,
) -> SectorFidelityBudget:
    """Return ``F_raw = w * F_amp * C_phase`` for an orthonormal sector basis.

    ``sector_columns`` has shape ``(basis_dimension, sector_dimension)``.  The
    reference must lie in its column space.  If the projected weight is zero,
    the conditional state and logarithmic budget are not evaluable.
    Positive factors are logged exactly as represented; no floor is used.
    """

    ground = normalize_state(reference_in_sector, name="reference_in_sector")
    raw = normalize_state(raw_candidate, name="raw_candidate")
    columns = np.asarray(sector_columns, dtype=np.complex128)
    if columns.ndim != 2 or columns.shape[0] != ground.size:
        raise ValueError("sector columns and state dimensions differ")
    if raw.shape != ground.shape:
        raise ValueError("reference and raw candidate dimensions differ")
    coordinates = columns.conj().T @ raw
    sector_weight = float(np.real(np.vdot(coordinates, coordinates)))
    raw_fidelity = float(abs(np.vdot(ground, raw)) ** 2)
    if sector_weight <= 0.0:
        return SectorFidelityBudget(
            sector_weight=max(0.0, sector_weight),
            projected_state=None,
            raw_fidelity=raw_fidelity,
            conditional_fidelity=None,
            factors=None,
            raw_conditional_closure_error=None,
            raw_product_absolute_error=None,
            raw_product_relative_error=None,
            log_evaluable=False,
            negative_log_raw_fidelity=None,
            negative_log_sector_weight=None,
            negative_log_amplitude_fidelity=None,
            negative_log_phase_coherence=None,
            log_closure_error=None,
            non_evaluable_reason="zero_sector_weight",
        )
    projected = columns @ coordinates / math.sqrt(sector_weight)
    projected = normalize_state(projected, name="projected_candidate")
    factors = fidelity_factors(ground, projected)
    conditional_fidelity = factors.fidelity
    raw_conditional_product = float(sector_weight * conditional_fidelity)
    if factors.phase_coherence is None:
        raw_factor_product = None
    else:
        raw_factor_product = float(
            sector_weight
            * factors.amplitude_fidelity
            * factors.weighted_phase_coherence
        )

    factor_values = (
        sector_weight,
        factors.amplitude_fidelity,
        factors.phase_coherence,
        raw_fidelity,
    )
    log_evaluable = all(value is not None and value > 0.0 for value in factor_values)
    if log_evaluable:
        log_raw = -math.log(raw_fidelity)
        log_sector = -math.log(sector_weight)
        log_amplitude = -math.log(factors.amplitude_fidelity)
        log_phase = -math.log(float(factors.phase_coherence))
        log_residual = abs(log_raw - (log_sector + log_amplitude + log_phase))
        reason = None
    else:
        log_raw = log_sector = log_amplitude = log_phase = log_residual = None
        zero_names = [
            name
            for name, value in zip(
                ("sector_weight", "amplitude_fidelity", "phase_coherence", "raw_fidelity"),
                factor_values,
                strict=True,
            )
            if value is None or value <= 0.0
        ]
        reason = "nonpositive_log_factor:" + ",".join(zero_names)
    return SectorFidelityBudget(
        sector_weight=sector_weight,
        projected_state=projected,
        raw_fidelity=raw_fidelity,
        conditional_fidelity=conditional_fidelity,
        factors=factors,
        raw_conditional_closure_error=abs(raw_fidelity - raw_conditional_product),
        raw_product_absolute_error=(
            abs(raw_fidelity - raw_factor_product)
            if raw_factor_product is not None
            else None
        ),
        raw_product_relative_error=(
            _relative_error(raw_fidelity, raw_factor_product)
            if raw_factor_product is not None
            else None
        ),
        log_evaluable=log_evaluable,
        negative_log_raw_fidelity=log_raw,
        negative_log_sector_weight=log_sector,
        negative_log_amplitude_fidelity=log_amplitude,
        negative_log_phase_coherence=log_phase,
        log_closure_error=log_residual,
        non_evaluable_reason=reason,
    )


def _phase_with_covariant_completion(
    donor: np.ndarray,
    recipient: np.ndarray,
    *,
    coefficient_zero_tolerance: float,
) -> tuple[np.ndarray, float]:
    donor_magnitude = np.abs(donor)
    recipient_magnitude = np.abs(recipient)
    donor_zero = donor_magnitude <= coefficient_zero_tolerance
    ambiguity_mass = float(np.sum(recipient_magnitude[donor_zero] ** 2))
    phases = np.ones_like(donor)
    donor_active = ~donor_zero
    phases[donor_active] = donor[donor_active] / donor_magnitude[donor_active]
    fallback = donor_zero & (recipient_magnitude > 0.0)
    phases[fallback] = recipient[fallback] / recipient_magnitude[fallback]
    return phases, ambiguity_mass


def amplitude_phase_hybrids(
    exact: Sequence[complex] | np.ndarray,
    candidate: Sequence[complex] | np.ndarray,
    *,
    coefficient_zero_tolerance: float,
    maximum_ambiguity_mass: float,
) -> HybridStates:
    """Construct the two pointwise amplitude/phase swapped states.

    At a donor coefficient below ``coefficient_zero_tolerance`` its phase is
    undefined.  The affected recipient Born mass is reported.  A covariant
    completion uses the recipient phase only when this mass does not exceed
    ``maximum_ambiguity_mass``; otherwise both hybrids are ``None`` and the
    secondary diagnostic is not evaluable.
    """

    if coefficient_zero_tolerance < 0.0 or maximum_ambiguity_mass < 0.0:
        raise ValueError("hybrid zero tolerances must be nonnegative")
    ground = normalize_state(exact, name="exact")
    state = normalize_state(candidate, name="candidate")
    if ground.shape != state.shape:
        raise ValueError("exact and candidate dimensions differ")
    exact_phase, candidate_mass = _phase_with_covariant_completion(
        ground,
        state,
        coefficient_zero_tolerance=coefficient_zero_tolerance,
    )
    candidate_phase, exact_mass = _phase_with_covariant_completion(
        state,
        ground,
        coefficient_zero_tolerance=coefficient_zero_tolerance,
    )
    if max(candidate_mass, exact_mass) > maximum_ambiguity_mass:
        return HybridStates(
            evaluable=False,
            exact=ground,
            candidate=state,
            candidate_amplitude_exact_phase=None,
            exact_amplitude_candidate_phase=None,
            candidate_mass_on_exact_phase_zeros=candidate_mass,
            exact_mass_on_candidate_phase_zeros=exact_mass,
            non_evaluable_reason="donor_zero_ambiguity_mass_exceeds_tolerance",
        )
    amplitude_only = normalize_state(
        np.abs(state) * exact_phase,
        name="candidate_amplitude_exact_phase",
    )
    phase_only = normalize_state(
        np.abs(ground) * candidate_phase,
        name="exact_amplitude_candidate_phase",
    )
    return HybridStates(
        evaluable=True,
        exact=ground,
        candidate=state,
        candidate_amplitude_exact_phase=amplitude_only,
        exact_amplitude_candidate_phase=phase_only,
        candidate_mass_on_exact_phase_zeros=candidate_mass,
        exact_mass_on_candidate_phase_zeros=exact_mass,
        non_evaluable_reason=None,
    )


def local_z2_configuration_diagonal(
    basis: Sequence[Sequence[int]] | np.ndarray,
    site_signs: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return ``D_x=prod_i z_i**n_i`` for a computational-bit basis.

    Bit ``0`` is spin down, bit ``1`` is spin up, and the input site order is
    preserved.  This is a coordinate gauge, not a physical spin rotation.
    """

    configurations = np.asarray(basis, dtype=np.int8)
    signs = np.asarray(site_signs, dtype=np.int8)
    if configurations.ndim != 2 or signs.shape != (configurations.shape[1],):
        raise ValueError("basis and site-sign dimensions differ")
    if np.any((configurations != 0) & (configurations != 1)):
        raise ValueError("computational configurations must contain only bits")
    if np.any((signs != -1) & (signs != 1)):
        raise ValueError("site signs must be +/-1")
    selected = np.where(configurations == 1, signs[None, :], 1)
    return np.prod(selected, axis=1, dtype=np.int8)


def factors_to_dict(factors: FidelityFactors) -> dict[str, float | None]:
    """Serialize :class:`FidelityFactors` without non-finite sentinels."""

    return {
        "fidelity": factors.fidelity,
        "bhattacharyya_coefficient": factors.bhattacharyya_coefficient,
        "amplitude_fidelity": factors.amplitude_fidelity,
        "phase_coherence": factors.phase_coherence,
        "weighted_phase_coherence": factors.weighted_phase_coherence,
        "phase_ratio_weighted_error": factors.phase_ratio_weighted_error,
        "absolute_closure_error": factors.absolute_closure_error,
        "relative_closure_error": factors.relative_closure_error,
    }


def budget_to_dict(
    budget: SectorFidelityBudget,
) -> Mapping[str, object]:
    """Serialize a sector budget while excluding its projected state vector."""

    return {
        "sector_weight": budget.sector_weight,
        "raw_fidelity": budget.raw_fidelity,
        "conditional_fidelity": budget.conditional_fidelity,
        "factors": factors_to_dict(budget.factors) if budget.factors else None,
        "raw_conditional_closure_error": budget.raw_conditional_closure_error,
        "raw_product_absolute_error": budget.raw_product_absolute_error,
        "raw_product_relative_error": budget.raw_product_relative_error,
        "log_evaluable": budget.log_evaluable,
        "negative_log_raw_fidelity": budget.negative_log_raw_fidelity,
        "negative_log_sector_weight": budget.negative_log_sector_weight,
        "negative_log_amplitude_fidelity": budget.negative_log_amplitude_fidelity,
        "negative_log_phase_coherence": budget.negative_log_phase_coherence,
        "log_closure_error": budget.log_closure_error,
        "non_evaluable_reason": budget.non_evaluable_reason,
    }
