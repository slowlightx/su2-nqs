from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.neural_backflow_pfaffian import (
    build_neural_backflow_data,
    neural_backflow_pairing_batch,
    neural_backflow_reduced_state,
)


def _toy_support():
    states = []
    for occupied in list(combinations(range(16), 8))[:80]:
        row = np.zeros(16, dtype=np.int8)
        row[list(occupied)] = 1
        states.append(row)
    states = np.asarray(states)
    raw = np.random.default_rng(4).normal(size=(len(states), 5))
    basis, _ = np.linalg.qr(raw)
    hamiltonian = np.diag(np.linspace(-2.0, 1.0, 5))
    return states, basis, hamiltonian


def test_backflow_pairing_is_symmetric_and_configuration_dependent() -> None:
    states, basis, hamiltonian = _toy_support()
    data = build_neural_backflow_data(
        states, basis, hamiltonian, kind="compact_j1"
    )
    pair = np.asarray(
        neural_backflow_pairing_batch(jnp.asarray((0.4, -0.3, 0.2, 0.1)), data)
    )
    np.testing.assert_allclose(pair, pair.transpose(0, 2, 1), atol=1.0e-14)
    np.testing.assert_allclose(np.diagonal(pair, axis1=1, axis2=2), 0.0)
    assert np.linalg.norm(pair[0] - pair[-1]) > 1.0e-6


def test_exact_projected_backflow_state_is_normalized() -> None:
    states, basis, hamiltonian = _toy_support()
    data = build_neural_backflow_data(
        states, basis, hamiltonian, kind="extended_j1j2"
    )
    state = np.asarray(
        neural_backflow_reduced_state(jnp.linspace(-0.2, 0.3, 6), data)
    )
    assert state.shape == (5,)
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=2.0e-13)


def test_scrambled_relation_control_has_identical_parameter_shape() -> None:
    states, basis, hamiltonian = _toy_support()
    physical = build_neural_backflow_data(
        states, basis, hamiltonian, kind="compact_j1"
    )
    scrambled = build_neural_backflow_data(
        states,
        basis,
        hamiltonian,
        kind="compact_j1",
        relation_site_permutation=np.random.default_rng(9).permutation(16),
    )
    assert physical.parameter_count == scrambled.parameter_count == 4
    assert not np.allclose(physical.j1_relation, scrambled.j1_relation)
