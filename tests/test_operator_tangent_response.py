from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import (
    cross_validated_one_layer_operator_ritz,
    cross_validated_resonance_channel_ritz,
    one_lanczos_and_resonance_state_actions,
    sampled_resonance_channel_ritz,
    sampled_one_layer_operator_ritz,
    scalar_operator_tangent_statistics,
)


def test_operator_tangent_force_matches_exact_finite_difference() -> None:
    generator = np.random.default_rng(817)
    dimension = 17
    state = generator.normal(size=dimension)
    state /= np.linalg.norm(state)
    operators = []
    for _ in range(5):
        matrix = generator.normal(size=(dimension, dimension))
        operators.append((matrix + matrix.T) / 2.0)
    responses = np.column_stack([operator @ state for operator in operators])
    ratios = responses / state[:, None]
    probabilities = state**2
    coupling = 0.31
    report = scalar_operator_tangent_statistics(
        ratios, (coupling,), weights=probabilities
    )
    row = report["couplings"][0]
    force = np.asarray(row["energy_force"])
    hamiltonian = operators[0] + coupling * operators[1]

    for token in range(5):
        epsilon = 1.0e-6

        def energy(value: float) -> float:
            trial = state + value * responses[:, token]
            return float(trial @ hamiltonian @ trial / (trial @ trial))

        finite_difference = (energy(epsilon) - energy(-epsilon)) / (2 * epsilon)
        assert np.isclose(force[token], finite_difference, atol=2.0e-9)


def test_full_operator_space_cannot_have_smaller_response_than_subspaces() -> None:
    generator = np.random.default_rng(29)
    ratios = generator.normal(size=(101, 5)) + 0.2j * generator.normal(
        size=(101, 5)
    )
    report = scalar_operator_tangent_statistics(ratios, (0.0, 0.125, 0.5))
    for row in report["couplings"]:
        full = row["full_five_token_subspace"][
            "squared_energy_slope_per_unit_fs_norm"
        ]
        shell = row["two_shell_subspace"][
            "squared_energy_slope_per_unit_fs_norm"
        ]
        hamiltonian = row["hamiltonian_lanczos_direction"][
            "squared_energy_slope_per_unit_fs_norm"
        ]
        assert full >= shell - 1.0e-12
        assert shell >= hamiltonian - 1.0e-12


def test_sampled_operator_ritz_contains_one_lanczos_space() -> None:
    generator = np.random.default_rng(907)
    dimension = 31
    state = generator.normal(size=dimension)
    state /= np.linalg.norm(state)
    operators = []
    for _ in range(5):
        matrix = generator.normal(size=(dimension, dimension))
        operators.append((matrix + matrix.T) / 2.0)
    responses = np.column_stack([operator @ state for operator in operators])
    local = responses / state[:, None]
    words = np.empty((dimension, 2, 5), dtype=np.complex128)
    for shell in range(2):
        for token in range(5):
            words[:, shell, token] = (
                operators[shell] @ responses[:, token]
            ) / state
    report = sampled_one_layer_operator_ritz(
        local, words, (0.27,), weights=state**2
    )
    row = report["couplings"][0]
    operator = row["one_layer_five_token_operator"]["energy_total"]
    krylov = row["one_lanczos_krylov"]["energy_total"]
    assert operator <= krylov + 2.0e-12
    channels = sampled_resonance_channel_ritz(
        local, words, (0.27,), weights=state**2
    )["couplings"][0]
    for channel in channels["resonance_channels"]:
        augmented = channel["augmented_ritz"]["energy_total"]
        assert operator <= augmented + 2.0e-12
        assert augmented <= krylov + 2.0e-12


def test_cross_validated_ritz_uses_each_independent_replicate_once() -> None:
    generator = np.random.default_rng(612)
    local = [generator.normal(size=(40, 5)) for _ in range(3)]
    words = [generator.normal(size=(40, 2, 5)) for _ in range(3)]
    report = cross_validated_one_layer_operator_ritz(local, words, (0.125,))
    folds = report["couplings"][0]["folds"]
    assert [fold["held_out_replicate"] for fold in folds] == [0, 1, 2]
    assert all(fold["training_sample_count"] == 80 for fold in folds)
    assert all(fold["audit_sample_count"] == 40 for fold in folds)


def test_single_channel_cross_validation_reports_all_physical_resonances() -> None:
    generator = np.random.default_rng(938)
    local = [generator.normal(size=(32, 5)) for _ in range(3)]
    words = [generator.normal(size=(32, 2, 5)) for _ in range(3)]
    report = cross_validated_resonance_channel_ritz(local, words, (0.0,))
    channels = report["couplings"][0]["resonance_channels"]
    assert [channel["token_label"] for channel in channels] == [
        "rhombus_ring_exchange",
        "rhombus_dimer_resonance",
        "six_site_loop_rearrangement",
    ]
    assert all(len(channel["folds"]) == 3 for channel in channels)


def test_linear_channel_state_actions_match_direct_dense_algebra() -> None:
    generator = np.random.default_rng(404)
    dimension = 19
    carrier = generator.normal(size=dimension)
    operators = []
    for _ in range(5):
        matrix = generator.normal(size=(dimension, dimension))
        operators.append((matrix + matrix.T) / 2.0)
    actions = np.column_stack([operator @ carrier for operator in operators])
    words = np.empty((dimension, 2, 5))
    for shell in range(2):
        for token in range(5):
            words[:, shell, token] = operators[shell] @ actions[:, token]
    coupling = 0.173
    hamiltonian = operators[0] + coupling * operators[1]
    krylov = np.asarray((0.8, -0.13))
    resonance = np.asarray((0.7, -0.11, 0.025))
    observed = one_lanczos_and_resonance_state_actions(
        carrier, actions, words, coupling, 3, krylov, resonance
    )
    expected_krylov = krylov[0] * carrier + krylov[1] * (hamiltonian @ carrier)
    expected_resonance = (
        resonance[0] * carrier
        + resonance[1] * (hamiltonian @ carrier)
        + resonance[2] * (operators[3] @ carrier)
    )
    np.testing.assert_allclose(observed[0], expected_krylov, atol=2.0e-13)
    np.testing.assert_allclose(observed[1], hamiltonian @ expected_krylov, atol=2.0e-13)
    np.testing.assert_allclose(observed[2], expected_resonance, atol=2.0e-13)
    np.testing.assert_allclose(observed[3], hamiltonian @ expected_resonance, atol=2.0e-13)
