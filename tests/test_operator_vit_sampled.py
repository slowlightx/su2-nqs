from __future__ import annotations

import jax
import numpy as np

from su2_nqs.benchmarks.operator_vit_sampled import (
    centered_component_coefficients_to_raw_words,
    centered_operator_vit_matrices,
    centered_scalar_component_matrices,
    effective_word_actions,
    expanded_base_words,
    krylov_control_basis,
    operator_vit_component_words,
    operator_vit_required_words,
    sampled_operator_vit_energy,
    sampled_operator_vit_summary,
    scalar_component_words,
)
from su2_nqs.models.su2_a1_operator_vit import (
    SU2A1OperatorViTConfig,
    init_su2_a1_operator_vit,
    su2_a1_operator_vit_component_weights,
)
from su2_nqs.operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS


def test_sampled_centered_components_match_direct_operator_algebra() -> None:
    generator = np.random.default_rng(9021)
    dimension = 23
    carrier = generator.normal(size=dimension)
    carrier /= np.linalg.norm(carrier)
    operators = []
    for _ in TRIANGULAR_A1_SCALAR_LABELS:
        matrix = generator.normal(size=(dimension, dimension))
        operators.append((matrix + matrix.T) / 2.0)
    labels = {label: operators[index] for index, label in enumerate(TRIANGULAR_A1_SCALAR_LABELS)}
    words = operator_vit_required_words()

    def action(word):
        value = carrier
        for label in reversed(word):
            value = labels[label] @ value
        return value

    raw = np.column_stack([action(word) for word in words])
    means = np.asarray([carrier @ (operator @ carrier) for operator in operators])
    scales = np.asarray(
        [np.linalg.norm(operator @ carrier - mean * carrier) for operator, mean in zip(operators, means, strict=True)]
    )
    coupling = 0.125
    components, h_components = centered_operator_vit_matrices(
        raw, words, coupling=coupling, means=means, scales=scales
    )
    centered = [
        (operator - mean * np.eye(dimension)) / scale
        for operator, mean, scale in zip(operators, means, scales, strict=True)
    ]
    expected = [carrier]
    expected.extend(operator @ carrier for operator in centered)
    expected.extend(
        centered[outer] @ (centered[inner] @ carrier)
        for outer in range(5)
        for inner in range(5)
    )
    hamiltonian = operators[0] + coupling * operators[1]
    np.testing.assert_allclose(components, np.column_stack(expected), atol=3e-12)
    np.testing.assert_allclose(
        h_components,
        np.column_stack([hamiltonian @ value for value in expected]),
        atol=5e-11,
    )
    assert len(operator_vit_component_words()) == 31
    assert max(map(len, words)) == 3


def test_sampled_operator_vit_energy_and_variance_match_dense_state() -> None:
    generator = np.random.default_rng(194)
    component_matrix = generator.normal(size=(29, 31))
    hamiltonian = generator.normal(size=(29, 29))
    hamiltonian = (hamiltonian + hamiltonian.T) / 2.0
    hamiltonian_matrix = hamiltonian @ component_matrix
    config = SU2A1OperatorViTConfig(
        operator_labels=TRIANGULAR_A1_SCALAR_LABELS,
        operator_descriptors=tuple((float(index), 1.0) for index in range(5)),
        embedding_dim=6,
        depth=2,
    )
    params = init_su2_a1_operator_vit(jax.random.PRNGKey(7), config)
    weights = np.asarray(su2_a1_operator_vit_component_weights(params, config))
    state = component_matrix @ weights
    action = hamiltonian @ state
    expected = float(np.vdot(state, action).real / np.vdot(state, state).real)
    observed = float(
        sampled_operator_vit_energy(
            params, component_matrix, hamiltonian_matrix, config
        )
    )
    summary = sampled_operator_vit_summary(
        params, component_matrix, hamiltonian_matrix, config
    )
    assert np.isclose(observed, expected, atol=2e-12)
    assert np.isclose(summary["energy_total"], expected, atol=2e-12)
    assert summary["variance_total"] >= 0.0
    assert 0.0 < summary["importance_ess_fraction"] <= 1.0


def test_effective_hamiltonian_and_resonance_token_expansion() -> None:
    generator = np.random.default_rng(552)
    dimension = 13
    state = generator.normal(size=dimension)
    base = {}
    for label in TRIANGULAR_A1_SCALAR_LABELS[:3]:
        matrix = generator.normal(size=(dimension, dimension))
        base[label] = (matrix + matrix.T) / 2.0
    coupling = 0.2
    expansions = {
        "H": {
            TRIANGULAR_A1_SCALAR_LABELS[0]: 1.0,
            TRIANGULAR_A1_SCALAR_LABELS[1]: coupling,
        },
        "R": {TRIANGULAR_A1_SCALAR_LABELS[2]: 1.0},
    }
    effective_components = scalar_component_words(("H", "R"))
    effective_required = tuple(
        dict.fromkeys(
            (
                *effective_components,
                *(("H", *word) for word in effective_components),
            )
        )
    )
    base_words = expanded_base_words(effective_required, expansions)

    def action(word):
        value = state
        for label in reversed(word):
            value = base[label] @ value
        return value

    base_actions = np.column_stack([action(word) for word in base_words])
    effective_actions = effective_word_actions(
        base_actions, base_words, effective_required, expansions
    )
    hamiltonian = base[TRIANGULAR_A1_SCALAR_LABELS[0]] + coupling * base[
        TRIANGULAR_A1_SCALAR_LABELS[1]
    ]
    resonance = base[TRIANGULAR_A1_SCALAR_LABELS[2]]
    direct = {"H": hamiltonian, "R": resonance}
    expected = []
    for word in effective_required:
        value = state
        for label in reversed(word):
            value = direct[label] @ value
        expected.append(value)
    np.testing.assert_allclose(
        effective_actions, np.column_stack(expected), atol=2e-11
    )

    means = np.asarray((state @ (hamiltonian @ state), state @ (resonance @ state)))
    scales = np.asarray(
        (
            np.linalg.norm(hamiltonian @ state - means[0] * state),
            np.linalg.norm(resonance @ state - means[1] * state),
        )
    )
    components, h_components = centered_scalar_component_matrices(
        effective_actions,
        effective_required,
        labels=("H", "R"),
        hamiltonian_label="H",
        means=means,
        scales=scales,
    )
    centered = (
        (hamiltonian - means[0] * np.eye(dimension)) / scales[0],
        (resonance - means[1] * np.eye(dimension)) / scales[1],
    )
    direct_components = [state]
    direct_components.extend(operator @ state for operator in centered)
    direct_components.extend(
        centered[outer] @ (centered[inner] @ state)
        for outer in range(2)
        for inner in range(2)
    )
    np.testing.assert_allclose(
        components, np.column_stack(direct_components), atol=5e-11
    )
    np.testing.assert_allclose(
        h_components,
        np.column_stack([hamiltonian @ value for value in direct_components]),
        atol=1e-9,
    )


def test_shell_resolved_component_basis_contains_exact_two_lanczos_control() -> None:
    generator = np.random.default_rng(812)
    dimension = 17
    state = generator.normal(size=dimension)
    state /= np.linalg.norm(state)
    operators = []
    for _ in range(2):
        matrix = generator.normal(size=(dimension, dimension))
        operators.append((matrix + matrix.T) / 2.0)
    coupling = 0.125
    means = np.asarray([state @ (operator @ state) for operator in operators])
    scales = np.asarray(
        [
            np.linalg.norm(operator @ state - mean * state)
            for operator, mean in zip(operators, means, strict=True)
        ]
    )
    centered = [
        (operator - mean * np.eye(dimension)) / scale
        for operator, mean, scale in zip(operators, means, scales, strict=True)
    ]
    components = [state]
    components.extend(operator @ state for operator in centered)
    components.extend(
        centered[outer] @ (centered[inner] @ state)
        for outer in range(2)
        for inner in range(2)
    )
    component_matrix = np.column_stack(components)
    basis = krylov_control_basis(
        hamiltonian_weights=(1.0, coupling), means=means, scales=scales
    )
    hamiltonian = operators[0] + coupling * operators[1]
    expected = np.column_stack(
        (state, hamiltonian @ state, hamiltonian @ (hamiltonian @ state))
    )
    np.testing.assert_allclose(component_matrix @ basis, expected, atol=2e-11)

    labels = ("J1", "J2")
    component_words = scalar_component_words(labels)
    required_words = tuple(
        dict.fromkeys(
            (
                *component_words,
                *((label, *word) for label in labels for word in component_words),
            )
        )
    )
    lookup = dict(zip(labels, operators, strict=True))

    def action(word):
        value = state
        for label in reversed(word):
            value = lookup[label] @ value
        return value

    raw = np.column_stack([action(word) for word in required_words])
    observed_components, observed_h = centered_scalar_component_matrices(
        raw,
        required_words,
        labels=labels,
        hamiltonian_weights=(1.0, coupling),
        means=means,
        scales=scales,
    )
    np.testing.assert_allclose(observed_components, component_matrix, atol=2e-11)
    np.testing.assert_allclose(observed_h, hamiltonian @ component_matrix, atol=2e-10)


def test_centered_polynomial_converts_to_identical_ordered_raw_words() -> None:
    generator = np.random.default_rng(2209)
    sample_count = 41
    labels = ("J1", "J2", "R")
    component_words = scalar_component_words(labels)
    raw = generator.normal(size=(sample_count, len(component_words)))
    means = generator.normal(size=3)
    scales = np.exp(generator.normal(size=3))
    coefficients = generator.normal(size=13)
    # Add the Hamiltonian-prefixed columns required by the generic constructor;
    # only the component half is used in this conversion identity.
    words = tuple(
        dict.fromkeys(
            (
                *component_words,
                *((label, *word) for label in labels for word in component_words),
            )
        )
    )
    extended = generator.normal(size=(sample_count, len(words)))
    positions = {word: index for index, word in enumerate(words)}
    for index, word in enumerate(component_words):
        extended[:, positions[word]] = raw[:, index]
    components, _ = centered_scalar_component_matrices(
        extended,
        words,
        labels=labels,
        hamiltonian_weights=(1.0, 0.125, 0.0),
        means=means,
        scales=scales,
    )
    physical = centered_component_coefficients_to_raw_words(
        coefficients, labels=labels, means=means, scales=scales
    )
    reconstructed = sum(
        value * extended[:, positions[word]] for word, value in physical.items()
    )
    np.testing.assert_allclose(reconstructed, components @ coefficients, atol=2e-11)
