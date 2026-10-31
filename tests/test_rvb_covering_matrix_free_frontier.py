from __future__ import annotations

from su2_nqs.benchmarks.rvb_covering_matrix_free_frontier import (
    exact_small_overlap_sign_frustration,
    exact_small_transition_loop_gate,
)


def test_exact_small_transition_loop_gate_is_at_machine_precision() -> None:
    result = exact_small_transition_loop_gate()
    assert result["passed"]
    assert result["maximum_error"] < 1.0e-12


def test_overlap_sign_frustration_distinguishes_triangle_and_square() -> None:
    result = exact_small_overlap_sign_frustration()
    triangle = result["triangular_n12"]
    square = result["square_n16"]
    assert triangle["minimum_negative_edge_count_over_real_gauges"] == 2
    assert triangle["gauge_invariant_frustrated_triangle_count"] == 8
    assert triangle["stoquastic_covering_gauge_exists"] is False
    assert square["minimum_negative_edge_count_over_real_gauges"] == 0
    assert square["gauge_invariant_frustrated_triangle_count"] == 0
    assert square["stoquastic_covering_gauge_exists"] is True
