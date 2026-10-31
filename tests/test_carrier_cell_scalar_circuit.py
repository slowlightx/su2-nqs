from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.carrier_cell_scalar_circuit import (
    serial_scalar_circuit_state,
)


def test_zero_weight_scalar_circuit_exactly_replays_carrier() -> None:
    carrier = jnp.asarray([1.0, 0.0, 0.0])
    operators = jnp.asarray(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    state = serial_scalar_circuit_state(
        jnp.zeros((2, 2), dtype=jnp.float64), carrier, operators
    )
    np.testing.assert_allclose(state, carrier, atol=0.0, rtol=0.0)


def test_scalar_circuit_layers_are_normalized_and_order_sensitive() -> None:
    carrier = jnp.asarray([1.0, 0.0, 0.0])
    operators = jnp.asarray(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ]
    )
    forward = serial_scalar_circuit_state(
        jnp.asarray([[0.4, 0.0], [0.0, 0.3]]), carrier, operators
    )
    reverse = serial_scalar_circuit_state(
        jnp.asarray([[0.0, 0.3], [0.4, 0.0]]), carrier, operators
    )
    np.testing.assert_allclose(np.linalg.norm(forward), 1.0, atol=2.0e-15)
    assert not np.allclose(forward, reverse)
