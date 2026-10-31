from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.qsl_three_arm import (
    triangular_pi_flux_operator_components,
)
from su2_nqs.benchmarks.operator_tangent_response import (
    triangular_a1_shell_word_local_ratios,
    triangular_a1_word_actions,
)
from su2_nqs.benchmarks.scalar_operator_frontier import (
    triangular_a1_resonance_operators,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    fixed_sz_sparse_hamiltonian,
)
from su2_nqs.operators.triangular_a1_scalar import (
    TRIANGULAR_A1_SCALAR_LABELS,
    triangular_a1_scalar_connectivity,
    triangular_a1_scalar_token_connectivity,
)


def test_on_demand_scalar_tokens_match_exact_sparse_columns() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    case = SMALL_CASES["square4"]
    h1, basis_h1 = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, basis_h_one = fixed_sz_sparse_hamiltonian(case, 1.0)
    assert basis_h1 == data.integer_basis == basis_h_one
    resonance = triangular_a1_resonance_operators(case, data.integer_basis)
    operators = (h1, h_at_one - h1, *resonance.values())
    positions = {
        tuple(configuration): index
        for index, configuration in enumerate(data.configurations)
    }
    for label, operator in zip(
        TRIANGULAR_A1_SCALAR_LABELS, operators, strict=True
    ):
        for column in (0, 1234, 12869):
            tokens = triangular_a1_scalar_token_connectivity(
                data.configurations[column], case.supercell
            )
            states, values = tokens[label]
            observed = np.zeros((len(data.integer_basis),), dtype=np.float64)
            observed[[positions[state] for state in states]] = values
            expected = operator.getcol(column).toarray().ravel()
            assert np.allclose(observed, expected, atol=2.0e-13)


def test_combined_scalar_connectivity_is_linear_and_preserves_magnetization() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    configuration = tuple(int(value) for value in data.configurations[4321])
    weights = np.asarray((0.3, -0.2, 0.1, -0.07, 0.04))
    states, values = triangular_a1_scalar_connectivity(
        configuration, SMALL_CASES["square4"].supercell, weights
    )
    assert all(sum(state) == sum(configuration) for state in states)
    tokens = triangular_a1_scalar_token_connectivity(
        configuration, SMALL_CASES["square4"].supercell
    )
    expected: dict[tuple[int, ...], float] = {}
    for weight, label in zip(weights, TRIANGULAR_A1_SCALAR_LABELS, strict=True):
        token_states, token_values = tokens[label]
        for state, value in zip(token_states, token_values, strict=True):
            expected[state] = expected.get(state, 0.0) + weight * value
    expected = {
        state: value for state, value in expected.items() if abs(value) > 1.0e-14
    }
    assert states == tuple(sorted(expected))
    assert np.allclose(values, [expected[state] for state in states])


def test_on_demand_depth_two_words_match_sparse_operator_products() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    case = SMALL_CASES["square4"]
    h1, _ = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, _ = fixed_sz_sparse_hamiltonian(case, 1.0)
    resonance = triangular_a1_resonance_operators(case, data.integer_basis)
    operators = (h1, h_at_one - h1, *resonance.values())
    positions = {
        tuple(configuration): index
        for index, configuration in enumerate(data.configurations)
    }

    def amplitudes(states: np.ndarray) -> np.ndarray:
        return np.asarray([data.carrier[positions[tuple(row)]] for row in states])

    support = np.flatnonzero(np.abs(data.carrier) > 1.0e-12)
    selected = support[[0, len(support) // 2, -1]]
    observed, _ = triangular_a1_shell_word_local_ratios(
        amplitudes,
        data.configurations[selected],
        data.carrier[selected],
        case.supercell,
        amplitude_batch_size=64,
    )
    for row, basis_index in enumerate(selected):
        for shell in range(2):
            for token in range(5):
                expected = (
                    operators[shell] @ (operators[token] @ data.carrier)
                )[basis_index] / data.carrier[basis_index]
                assert np.isclose(observed[row, shell, token], expected, atol=2e-11)


def test_generic_depth_three_words_match_sparse_operator_products() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    case = SMALL_CASES["square4"]
    h1, _ = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, _ = fixed_sz_sparse_hamiltonian(case, 1.0)
    resonance = triangular_a1_resonance_operators(case, data.integer_basis)
    operators = (h1, h_at_one - h1, *resonance.values())
    positions = {
        tuple(configuration): index
        for index, configuration in enumerate(data.configurations)
    }

    def amplitudes(states: np.ndarray) -> np.ndarray:
        return np.asarray([data.carrier[positions[tuple(row)]] for row in states])

    selected = np.asarray([4321], dtype=np.int64)
    words = (
        (),
        (TRIANGULAR_A1_SCALAR_LABELS[0],),
        (
            TRIANGULAR_A1_SCALAR_LABELS[2],
            TRIANGULAR_A1_SCALAR_LABELS[1],
        ),
        (
            TRIANGULAR_A1_SCALAR_LABELS[0],
            TRIANGULAR_A1_SCALAR_LABELS[2],
            TRIANGULAR_A1_SCALAR_LABELS[1],
        ),
    )
    observed, ledger = triangular_a1_word_actions(
        amplitudes,
        data.configurations[selected],
        case.supercell,
        words,
        amplitude_batch_size=128,
    )
    expected = (
        data.carrier,
        operators[0] @ data.carrier,
        operators[2] @ (operators[1] @ data.carrier),
        operators[0] @ (operators[2] @ (operators[1] @ data.carrier)),
    )
    np.testing.assert_allclose(
        observed[0],
        [vector[selected[0]] for vector in expected],
        atol=3.0e-10,
    )
    assert ledger["maximum_word_depth"] == 3
    assert ledger["word_count"] == len(words)
