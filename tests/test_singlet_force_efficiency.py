from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.singlet_force_efficiency import (
    singlet_force_efficiency_benchmark,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds


def test_force_efficiency_benchmark_is_exact_and_exposes_crossover() -> None:
    report = singlet_force_efficiency_benchmark(
        square_j1j2_bonds(2, 4, j1=1.0, j2=0.5, periodic=False),
        overlaps=(0.0, 1.0),
        sample_counts=(64,),
        replicates=32,
        tangent_count=4,
        seed=19,
    )
    assert report["singlet_dimension"] == 14
    assert report["exact_force_maximum_difference"] < 3.0e-12
    rows = report["rows"]
    assert len(rows) == 2
    initial, ground = rows
    assert initial["full_amplitude_work_per_sample"] > 2.0
    assert initial["diagonal_amplitude_work_per_sample"] == 2.0
    assert np.isfinite(initial["diagonal_over_full_work_adjusted_mse"])
    assert np.isfinite(initial["diagonal_over_full_asymptotic_work_adjusted_mse"])
    # The full local energy has exact zero variance at the eigenstate while
    # the diagonal estimator deliberately does not.
    assert ground["full_local_energy_variance"] < 1.0e-25
    assert ground["diagonal_local_energy_variance"] > 1.0e-4
    assert ground["diagonal_over_full_force_mse"] is None
    assert ground["diagonal_over_full_asymptotic_mse"] is None
