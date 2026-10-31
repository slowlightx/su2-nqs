import math

import pytest

from su2_nqs.benchmarks.spin_benchmark_synthesis import (
    log_correlation_table,
    paired_basis_rows,
    summarize_paired_basis,
)


def _row(
    *,
    lattice: str,
    n_sites: int,
    ordering: str,
    basis: str,
    scale: float,
) -> dict[str, object]:
    return {
        "lattice": lattice,
        "n_sites": n_sites,
        "ordering": ordering,
        "basis_kind": basis,
        "maximum_cut_width": n_sites,
        "dimension": scale,
        "condition_number": scale**2,
        "exact_samples_for_99_percent_target_weight": scale**3,
        "exact_tangent_residual_fraction": scale**4,
        "born_worst_tangent_error": scale**5,
        "cutoff_worst_tangent_error": scale**6,
    }


def test_log_correlations_use_log_scale_and_tie_safe_ranks() -> None:
    rows = [
        _row(
            lattice="square",
            n_sites=index + 4,
            ordering="row_major",
            basis="fusion" if index % 2 == 0 else "sz_zero",
            scale=float(index + 1),
        )
        for index in range(4)
    ]
    table = log_correlation_table(rows)
    assert len(table) == 8
    for row in table:
        assert math.isclose(row["log_pearson"], 1.0, abs_tol=1.0e-14)
        assert math.isclose(row["spearman"], 1.0, abs_tol=1.0e-14)


def test_paired_ratios_are_sz_over_fusion() -> None:
    fusion = _row(
        lattice="triangular",
        n_sites=6,
        ordering="morton",
        basis="fusion",
        scale=2.0,
    )
    sz_zero = _row(
        lattice="triangular",
        n_sites=6,
        ordering="morton",
        basis="sz_zero",
        scale=4.0,
    )
    paired = paired_basis_rows([sz_zero, fusion])
    assert len(paired) == 1
    assert paired[0]["dimension_compression"] == 2.0
    assert paired[0]["condition_gain"] == 4.0
    assert paired[0]["target_tail_n99_gain"] == 8.0
    assert paired[0]["born_worst_error_gain"] == 32.0
    summary = summarize_paired_basis(paired)
    assert summary["fusion_better_counts"]["target_tail_n99_gain"] == 1


def test_basis_pair_must_be_complete() -> None:
    fusion = _row(
        lattice="square",
        n_sites=6,
        ordering="row_major",
        basis="fusion",
        scale=2.0,
    )
    with pytest.raises(ValueError, match="incomplete basis pair"):
        paired_basis_rows([fusion])
