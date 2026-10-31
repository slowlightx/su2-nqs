import numpy as np
import pytest

from su2_nqs.benchmarks.square_sector_spectroscopy import (
    lowest_square_sector_minimum,
    square_sector_shells,
)


@pytest.mark.parametrize("target_q", (0, 2, 4))
def test_small_square_sector_minimum_is_exact_and_labeled(target_q: int) -> None:
    h1, h2 = square_sector_shells(2, target_q)
    result = lowest_square_sector_minimum(
        h1,
        h2,
        linear_size=2,
        target_q=target_q,
        j2_over_j1=0.2,
        low_modes=2,
    )

    assert np.isfinite(result.energy)
    assert result.next_level_gap > 0.0
    assert result.momentum_label in ("Gamma", "M", "X", "Y")
    assert result.audit["ground_eigen_residual"] < 1e-10
    assert result.audit["maximum_space_group_error"] < 1e-10
