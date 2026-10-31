from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.rvb_covering_frontier import rvb_covering_frontier


def test_rvb_covering_oracle_finds_non_krylov_directions_in_both_geometries() -> None:
    result = rvb_covering_frontier()
    assert result["exact_sector_gate_passed"]
    assert result["rvb_direction_beyond_two_lanczos_present_both_cases"]
    assert result["architecture"]["forward_reynolds_sum_required"] is False
    assert result["architecture"]["neural_optimization_used"] is False

    triangular = result["triangular_n12"]
    assert triangular["matching_count"] == 348
    assert triangular["matching_orbit_count"] == 7
    assert triangular["nonzero_a1_orbit_count"] == 7
    assert triangular["rvb_a1_oracle"]["rank"] == 4
    assert triangular["rvb_increment_beyond_two_lanczos_per_site"] < -6.0e-3
    assert triangular["rvb_a1_oracle"]["variance_total"] < 1.0e-20

    square = result["square_n16"]
    assert square["matching_count"] == 272
    assert square["matching_orbit_count"] == 13
    assert square["nonzero_a1_orbit_count"] == 13
    assert square["rvb_a1_oracle"]["rank"] == 13
    assert square["rvb_increment_beyond_two_lanczos_per_site"] < -4.0e-3
    assert square["two_lanczos_plus_rvb_a1_oracle"]["variance_total"] < (
        0.25 * square["two_lanczos"]["variance_total"]
    )
    np.testing.assert_allclose(
        square["carrier_gamma_a1_projection_weight"],
        0.8299693085493866,
        atol=2.0e-13,
    )

