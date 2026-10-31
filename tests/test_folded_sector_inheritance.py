import numpy as np

from su2_nqs.benchmarks.folded_sector_inheritance import (
    exact_diagonal_folded_quantities,
    folded_spectrum_sector_inheritance_audit,
    residual_sector_inheritance_audit,
)


def test_residual_gap_bound_is_conservative_and_inherits_pure_sector():
    eigenvalues = np.asarray((-2.0, -0.8, 0.4))
    state = np.sqrt(np.asarray((0.98, 0.015, 0.005)))
    exact = exact_diagonal_folded_quantities(eigenvalues, state, shift=-2.0)
    distance = min(abs(eigenvalues[1:] - exact["energy"]))
    audit = residual_sector_inheritance_audit(
        residual_norm_upper=exact["residual_norm"],
        complement_distance_lower=float(distance),
        target_window_all_in_requested_sector=True,
        target_window_dimension=1,
        requested_minimum_weight=0.9,
    )
    assert audit["target_window_weight_lower"] <= 0.98 + 1.0e-14
    assert audit["sector_inheritance_pass"]
    assert audit["individual_root_ownership_pass"]


def test_energy_degeneracy_with_another_irrep_forbids_sector_inheritance():
    audit = residual_sector_inheritance_audit(
        residual_norm_upper=1.0e-8,
        complement_distance_lower=0.2,
        target_window_all_in_requested_sector=False,
        target_window_dimension=2,
    )
    assert audit["target_window_weight_lower"] > 0.999999
    assert audit["requested_sector_weight_lower"] == 0.0
    assert not audit["sector_inheritance_pass"]
    assert audit["interpretation"] == "energy_window_contains_or_may_contain_other_sectors"


def test_folded_spectrum_requires_global_closest_window_not_gamma_only_gap():
    audit = folded_spectrum_sector_inheritance_audit(
        folded_expectation_upper=1.0e-6,
        complement_shift_distance_lower=0.01,
        target_shift_distance_upper=0.02,
        target_window_all_in_requested_sector=True,
        target_window_dimension=1,
    )
    assert not audit["target_is_strictly_closest_window"]
    assert not audit["sector_inheritance_pass"]


def test_folded_near_degenerate_same_sector_certifies_subspace_not_root():
    audit = folded_spectrum_sector_inheritance_audit(
        folded_expectation_upper=2.0e-5,
        complement_shift_distance_lower=0.1,
        target_shift_distance_upper=0.004,
        target_window_all_in_requested_sector=True,
        target_window_dimension=2,
        requested_minimum_weight=0.99,
    )
    assert audit["target_window_weight_lower"] == 0.998
    assert audit["sector_inheritance_pass"]
    assert not audit["individual_root_ownership_pass"]
    assert audit["interpretation"] == "same-sector_subspace_only"

