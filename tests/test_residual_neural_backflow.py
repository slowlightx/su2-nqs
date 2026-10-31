from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.neural_backflow_pfaffian import build_neural_backflow_data
from su2_nqs.benchmarks.residual_neural_backflow import residual_backflow_state


def test_zero_scale_residual_reproduces_anchor_exactly() -> None:
    states = []
    for occupied in list(combinations(range(16), 8))[:80]:
        row = np.zeros(16, dtype=np.int8)
        row[list(occupied)] = 1
        states.append(row)
    states = np.asarray(states)
    basis, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(80, 5)))
    data = build_neural_backflow_data(
        states, basis, np.diag(np.arange(5.0)), kind="compact_j1"
    )
    anchor = np.asarray((1.0, 2.0, -1.0, 0.5, 0.25))
    parameters = jnp.asarray((0.2, -0.1, 0.3, -0.2, 0.0))
    observed = np.asarray(residual_backflow_state(parameters, data, jnp.asarray(anchor)))
    expected = anchor / np.linalg.norm(anchor)
    np.testing.assert_allclose(observed, expected, atol=2.0e-13)
