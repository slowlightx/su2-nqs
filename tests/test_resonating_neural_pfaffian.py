from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.resonating_neural_pfaffian import lowest_ritz_bank


def test_lowest_ritz_bank_handles_redundant_columns() -> None:
    hamiltonian = np.diag((-2.0, -1.0, 0.5, 1.0))
    columns = np.column_stack(
        (np.asarray((1.0, 1.0, 0.0, 0.0)), np.asarray((2.0, 2.0, 0.0, 0.0)))
    )
    state, row = lowest_ritz_bank(columns, hamiltonian)
    assert row["rank"] == 1
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1.0e-14)
