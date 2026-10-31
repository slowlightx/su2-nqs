from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.fidelity_factorization import (
    amplitude_phase_hybrids,
    fidelity_factors,
    local_z2_configuration_diagonal,
    normalize_state,
    sector_fidelity_budget,
)


def test_fidelity_factorization_matches_independent_weighted_phase_average() -> None:
    reference = np.asarray([1.0, 1.0j, -1.0, 0.5j], dtype=np.complex128)
    candidate = np.asarray([0.8j, 0.2, -0.3j, -1.1], dtype=np.complex128)
    factors = fidelity_factors(reference, candidate)
    assert 0.0 <= factors.fidelity <= factors.amplitude_fidelity <= 1.0
    assert factors.phase_coherence is not None
    assert factors.weighted_phase_coherence is not None
    assert 0.0 <= factors.phase_coherence <= 1.0
    assert factors.phase_ratio_weighted_error < 1.0e-15
    assert factors.absolute_closure_error < 1.0e-15
    assert factors.relative_closure_error < 1.0e-14


def test_zero_amplitude_overlap_has_no_invented_phase_or_log_floor() -> None:
    factors = fidelity_factors([1.0, 0.0], [0.0, 1.0])
    assert factors.fidelity == 0.0
    assert factors.amplitude_fidelity == 0.0
    assert factors.phase_coherence is None
    assert factors.weighted_phase_coherence is None
    columns = np.asarray([[1.0], [0.0]], dtype=np.complex128)
    budget = sector_fidelity_budget([1.0, 0.0], [0.0, 1.0], columns)
    assert budget.sector_weight == 0.0
    assert budget.projected_state is None
    assert not budget.log_evaluable
    assert budget.negative_log_raw_fidelity is None
    assert budget.non_evaluable_reason == "zero_sector_weight"


def test_sector_weight_amplitude_phase_log_budget_is_exact() -> None:
    # The first two coordinates span the registered sector.  The raw state has
    # a known out-of-sector component, while the conditional state has both a
    # magnitude and a relative-phase error.
    columns = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.complex128
    )
    reference = normalize_state([1.0, 1.0j, 0.0])
    raw = normalize_state([2.0, -0.5j, 1.0])
    budget = sector_fidelity_budget(reference, raw, columns)
    assert budget.projected_state is not None
    assert np.isclose(budget.sector_weight, 4.25 / 5.25)
    assert budget.raw_conditional_closure_error < 1.0e-15
    assert budget.raw_product_absolute_error is not None
    assert budget.raw_product_absolute_error < 1.0e-15
    assert budget.raw_product_relative_error is not None
    assert budget.raw_product_relative_error < 1.0e-14
    assert budget.log_evaluable
    assert budget.log_closure_error is not None
    assert budget.log_closure_error < 1.0e-14


def test_factorization_is_invariant_to_independent_global_phases() -> None:
    reference = normalize_state([1.0, 2.0j, -0.4])
    candidate = normalize_state([0.3j, -0.7, 1.2j])
    original = fidelity_factors(reference, candidate)
    transformed = fidelity_factors(1.0j * reference, (-0.6 + 0.8j) * candidate)
    assert np.isclose(original.fidelity, transformed.fidelity, atol=1.0e-15)
    assert np.isclose(
        original.amplitude_fidelity,
        transformed.amplitude_fidelity,
        atol=1.0e-15,
    )
    assert np.isclose(
        original.phase_coherence,
        transformed.phase_coherence,
        atol=1.0e-15,
    )


def test_hybrids_swap_only_named_factor_and_are_diagonal_gauge_covariant() -> None:
    exact = normalize_state([1.0, 2.0j, -0.5, 0.7j])
    candidate = normalize_state([0.2j, -0.8, 1.1j, 0.4])
    hybrids = amplitude_phase_hybrids(
        exact,
        candidate,
        coefficient_zero_tolerance=0.0,
        maximum_ambiguity_mass=0.0,
    )
    assert hybrids.evaluable
    amplitude_only = hybrids.candidate_amplitude_exact_phase
    phase_only = hybrids.exact_amplitude_candidate_phase
    assert amplitude_only is not None and phase_only is not None
    assert np.allclose(np.abs(amplitude_only), np.abs(candidate))
    assert np.allclose(np.abs(phase_only), np.abs(exact))

    diagonal = np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.complex128)
    gauged = amplitude_phase_hybrids(
        diagonal * exact,
        diagonal * candidate,
        coefficient_zero_tolerance=0.0,
        maximum_ambiguity_mass=0.0,
    )
    assert gauged.evaluable
    assert np.allclose(
        gauged.candidate_amplitude_exact_phase,
        diagonal * amplitude_only,
        atol=1.0e-15,
    )
    assert np.allclose(
        gauged.exact_amplitude_candidate_phase,
        diagonal * phase_only,
        atol=1.0e-15,
    )


def test_hybrid_donor_zero_ambiguity_fails_closed() -> None:
    hybrids = amplitude_phase_hybrids(
        [1.0, 0.0],
        [1.0, 1.0],
        coefficient_zero_tolerance=1.0e-14,
        maximum_ambiguity_mass=1.0e-3,
    )
    assert not hybrids.evaluable
    assert hybrids.candidate_amplitude_exact_phase is None
    assert np.isclose(hybrids.candidate_mass_on_exact_phase_zeros, 0.5)
    assert hybrids.non_evaluable_reason == (
        "donor_zero_ambiguity_mass_exceeds_tolerance"
    )


def test_local_z2_diagonal_uses_bit_one_and_preserves_site_order() -> None:
    basis = ((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0))
    diagonal = local_z2_configuration_diagonal(basis, (1, -1, -1))
    assert diagonal.tolist() == [1, -1, 1, -1]
    with pytest.raises(ValueError):
        local_z2_configuration_diagonal(basis, (1, 0, -1))
