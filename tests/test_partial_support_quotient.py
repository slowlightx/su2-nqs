import numpy as np
import pytest

from su2_nqs.benchmarks.partial_support_quotient import (
    complex_centered_rank_ceiling,
    innovation_rank_stability,
)


def test_complex_centered_rank_ceiling_counts_probability_and_phase_gauges() -> None:
    assert complex_centered_rank_ceiling(1) == 0
    assert complex_centered_rank_ceiling(42) == 82
    assert complex_centered_rank_ceiling(100, full_state_count=132) == 198
    assert complex_centered_rank_ceiling(132, full_state_count=132) == 262
    with pytest.raises(ValueError, match="exceed"):
        complex_centered_rank_ceiling(133, full_state_count=132)


def test_innovation_rank_stability_normalizes_different_total_ranks() -> None:
    result = innovation_rank_stability(
        ((0, 2, 2, 6, 10), (0, 4, 4, 12, 20))
    )
    np.testing.assert_allclose(result.normalized_cumulative_profiles[0], (
        0.0,
        0.1,
        0.2,
        0.5,
        1.0,
    ))
    np.testing.assert_allclose(
        result.normalized_cumulative_profiles[0],
        result.normalized_cumulative_profiles[1],
    )
    assert result.maximum_profile_range == pytest.approx(0.0)
    assert result.peak_decision_consistent
    assert result.final_ranks == (20, 40)


def test_innovation_rank_stability_detects_peak_and_profile_changes() -> None:
    result = innovation_rank_stability(((0, 1, 5, 2), (0, 1, 2, 5)))
    assert not result.peak_decision_consistent
    assert result.maximum_profile_range > 0.3
    with pytest.raises(ValueError, match="positive total"):
        innovation_rank_stability(((0, 0),))
