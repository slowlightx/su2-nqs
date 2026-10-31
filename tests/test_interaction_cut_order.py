"""Exact tests for graph-only interaction-cut ordering and sparse leakage."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.interaction_cut_order import (
    exact_minimax_cut_order,
    exposure_quartile_contrast,
    first_connected_random_orders,
    normalized_absolute_edges,
    order_cut_exposures,
    physical_prefix_signature,
    sparse_leakage_and_graph_metrics,
    spearman_correlation,
    subset_is_connected,
)


def test_duplicate_signed_edges_are_combined_before_absolute_value() -> None:
    edges = normalized_absolute_edges(
        3,
        (
            (0, 1, 2.0),
            (1, 0, -0.5),
            (0, 2, 0.75),
            (2, 0, -0.75),
        ),
    )
    assert edges == ((0, 1, 1.5),)
    with pytest.raises(ValueError, match="nonzero edge"):
        normalized_absolute_edges(2, ((0, 1, 1.0), (1, 0, -1.0)))


def test_dynamic_program_matches_exhaustive_lexicographic_objective() -> None:
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.5),
        (2, 3, 1.25),
        (3, 4, 0.75),
        (0, 4, 0.25),
    )
    exhaustive = []
    for order in permutations(range(5)):
        exposure = order_cut_exposures(order, bonds)
        exhaustive.append((max(exposure), sum(exposure), order))
    expected = min(exhaustive)
    actual = exact_minimax_cut_order(5, bonds, tolerance=0.0)
    assert actual["maximum_cut_exposure"] == pytest.approx(expected[0])
    assert actual["summed_cut_exposure"] == pytest.approx(expected[1])
    assert actual["order"] == expected[2]


def test_two_stage_dp_preserves_global_lex_tie_after_cutwidth_bottleneck() -> None:
    # A one-label-per-subset joint (partial max, partial sum, lex) recurrence
    # incorrectly returns (1,3,4,2,0) here.  Both complete orders have width 4
    # and sum 11, so the globally correct third-stage tie break starts at 0.
    bonds = (
        (0, 4, 1.0),
        (1, 3, 3.0),
        (1, 4, 1.0),
        (2, 4, 3.0),
        (3, 4, 1.0),
    )
    actual = exact_minimax_cut_order(5, bonds, tolerance=0.0)
    assert actual["maximum_cut_exposure"] == pytest.approx(4.0)
    assert actual["summed_cut_exposure"] == pytest.approx(11.0)
    assert actual["order"] == (0, 2, 4, 1, 3)


def test_prefix_signature_and_connected_controls_are_physical_and_reproducible() -> None:
    bonds = tuple((site, (site + 1) % 8, 1.0) for site in range(8))
    named = ((0, 1, 2, 3, 4, 5, 6, 7),)
    exclusions = {physical_prefix_signature(named[0], (2, 4))}
    first = first_connected_random_orders(
        8,
        bonds,
        seed=42,
        count=3,
        signature_cuts=(2, 4),
        connectivity_cuts=(2, 4),
        excluded_signatures=exclusions,
    )
    second = first_connected_random_orders(
        8,
        bonds,
        seed=42,
        count=3,
        signature_cuts=(2, 4),
        connectivity_cuts=(2, 4),
        excluded_signatures=exclusions,
    )
    assert first == second
    signatures = {physical_prefix_signature(order, (2, 4)) for order in first}
    assert len(signatures) == 3
    assert not signatures & exclusions
    for order in first:
        for size in (2, 4):
            assert subset_is_connected(order[:size], 8, bonds)
            assert subset_is_connected(order[size:], 8, bonds)


def test_sparse_leakage_matches_direct_dense_blocks() -> None:
    dense = np.asarray(
        [
            [-1.0, 0.2, 0.3, 0.0],
            [0.2, -0.5, 0.4, 0.1],
            [0.3, 0.4, 0.7, 0.6],
            [0.0, 0.1, 0.6, 1.1],
        ]
    )
    retained = (0, 2)
    result = sparse_leakage_and_graph_metrics(sparse.csr_matrix(dense), retained)
    complement = (1, 3)
    expected = np.linalg.norm(dense[np.ix_(complement, retained)]) / np.linalg.norm(dense)
    assert result["leakage_ratio"] == pytest.approx(expected)
    assert result["projected_offdiagonal_nnz"] == 2
    assert result["projected_mean_offdiagonal_degree"] == pytest.approx(1.0)
    assert result["projected_Hermiticity_error"] == 0.0


def test_tie_aware_spearman_and_registered_quartiles() -> None:
    assert spearman_correlation((0, 1, 1, 2), (3, 2, 2, 0)) == pytest.approx(-1.0)
    rows = [
        {
            "order_id": f"c{index}",
            "registered_cut_exposure_sum": float(index),
            "projection_fidelity": 1.0 - 0.02 * index,
            "Ritz_error_over_bandwidth": 0.01 * (index + 1),
            "leakage_ratio": 0.02 * (index + 1),
        }
        for index in range(8)
    ]
    result = exposure_quartile_contrast(rows)
    assert result["quartile_count"] == 2
    assert result["low_order_ids"] == ["c0", "c1"]
    assert result["high_order_ids"] == ["c6", "c7"]
    assert result["fidelity_low_minus_high"] == pytest.approx(0.12)
    assert result["Ritz_error_reduction"] == pytest.approx(0.8)
    assert result["leakage_reduction"] == pytest.approx(0.8)


def test_quartile_reductions_fail_closed_on_zero_denominator() -> None:
    rows = [
        {
            "order_id": f"c{index}",
            "registered_cut_exposure_sum": float(index),
            "projection_fidelity": 1.0,
            "Ritz_error_over_bandwidth": 0.0,
            "leakage_ratio": 0.0,
        }
        for index in range(4)
    ]
    result = exposure_quartile_contrast(rows)
    assert result["Ritz_error_reduction"] is None
    assert result["leakage_reduction"] is None
