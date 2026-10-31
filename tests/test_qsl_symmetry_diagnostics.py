from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.qsl_symmetry_diagnostics import (
    triangular_qsl_symmetry_diagnostics,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_pi_flux_operator_components,
)


def test_identical_reference_has_zero_symmetry_observable_error() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    result = triangular_qsl_symmetry_diagnostics(
        data,
        {"copy": data.carrier.copy()},
        data.carrier,
    )
    row = result["variational"]["copy"]
    assert np.isclose(row["exact_fidelity"], 1.0, atol=2.0e-14)
    assert row["median_symmetry_observable_relative_error"] < 2.0e-14
    assert max(row["relative_errors_to_exact"].values()) < 2.0e-14
    assert max(row["ordering_relative_errors_to_exact"].values()) < 2.0e-14
    assert result["diagnostic_contract"]["qsl_identification_from_4x4_is_forbidden"]
