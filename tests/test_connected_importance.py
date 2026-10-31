import numpy as np
import pytest
from scipy import sparse

from su2_nqs.benchmarks.connected_importance import (
    connected_proposal_distribution,
    hamiltonian_connected_transition,
    mixed_connected_probabilities_and_base_at_indices,
    mixed_connected_probabilities_at_indices,
    mixed_connected_proposal_distribution,
    proposal_importance_diagnostics,
    sample_mixed_connected_indices,
    sample_stratified_connected_indices,
    stratified_connected_proposal_distribution,
)


def test_connected_kernel_uses_offdiagonal_hamiltonian_weights_and_fallback():
    # H[new, old]; column two has no off-diagonal connection and must self-loop.
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [3.0, 1.0, 0.0],
                [2.0, -4.0, 0.0],
                [1.0, 3.0, 5.0],
            ]
        )
    )
    transition = hamiltonian_connected_transition(matrix, amplitude_power=1.0)
    assert np.asarray(transition.sum(axis=0)).ravel() == pytest.approx(1.0)
    assert transition.toarray() == pytest.approx(
        np.asarray(
            [
                [0.0, 0.25, 0.0],
                [2.0 / 3.0, 0.0, 0.0],
                [1.0 / 3.0, 0.75, 1.0],
            ]
        )
    )


def test_connected_mixture_is_normalized_and_covers_one_action_teacher_support():
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
    )
    transition = hamiltonian_connected_transition(matrix)
    base = np.asarray([1.0, 0.0, 0.0])
    connected = connected_proposal_distribution(base, transition)
    proposal = mixed_connected_proposal_distribution(
        base, transition, mixture_weight=0.25
    )
    assert connected == pytest.approx([0.0, 1.0, 0.0])
    assert proposal == pytest.approx([0.75, 0.25, 0.0])
    teacher = np.asarray([0.8, 0.2, 0.0])
    diagnostics = proposal_importance_diagnostics(
        proposal, teacher, sample_count=100
    )
    assert diagnostics["target_weight_outside_proposal_support"] == 0.0
    assert diagnostics["effective_sample_size_fraction"] > 0.0


def test_mixed_connected_sampler_matches_exact_marginal():
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 4.0],
                [3.0, 1.0, 0.0],
            ]
        )
    )
    transition = hamiltonian_connected_transition(matrix, amplitude_power=2.0)
    base = np.asarray([0.2, 0.3, 0.5])
    eta = 0.4
    expected = mixed_connected_proposal_distribution(
        base, transition, mixture_weight=eta
    )
    rng = np.random.default_rng(17)
    parents = rng.choice(3, size=200_000, p=base)
    samples = sample_mixed_connected_indices(
        parents, transition, mixture_weight=eta, rng=rng
    )
    observed = np.bincount(samples, minlength=3) / len(samples)
    assert observed == pytest.approx(expected, abs=3e-3)


def test_reverse_local_probabilities_match_full_connected_distribution():
    matrix = sparse.csc_matrix(
        np.asarray(
            [
                [0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 4.0, 0.0],
                [3.0, 1.0, 0.0, 2.0],
                [0.0, 2.0, 1.0, 0.0],
            ]
        )
    )
    transition = hamiltonian_connected_transition(matrix, amplitude_power=2.0)
    base = np.asarray([0.1, 0.2, 0.3, 0.4])
    targets = np.asarray([3, 1, 3, 0], dtype=np.int64)
    expected = mixed_connected_proposal_distribution(
        base, transition, mixture_weight=0.5
    )[targets]
    observed, diagnostics = mixed_connected_probabilities_at_indices(
        targets,
        transition,
        lambda indices: base[indices],
        mixture_weight=0.5,
    )
    assert observed == pytest.approx(expected, abs=1e-15)
    assert diagnostics["unique_target_count"] == 3
    assert diagnostics["unique_probability_evaluation_count"] <= 4
    assert diagnostics["reverse_connection_count"] > 0

    shared_observed, shared_base, shared_diagnostics = (
        mixed_connected_probabilities_and_base_at_indices(
            targets,
            transition,
            lambda indices: base[indices],
            mixture_weight=0.5,
        )
    )
    assert shared_observed == pytest.approx(expected, abs=1e-15)
    assert shared_base == pytest.approx(base[targets], abs=1e-15)
    assert shared_diagnostics == diagnostics


def test_stratified_connected_sampler_preserves_channel_mixture():
    first = hamiltonian_connected_transition(
        sparse.csc_matrix(
            np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        )
    )
    second = hamiltonian_connected_transition(
        sparse.csc_matrix(
            np.asarray([[0.0, 0.0, 1.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]])
        )
    )
    base = np.asarray([0.2, 0.3, 0.5])
    weights = (0.2, 0.35)
    expected = stratified_connected_proposal_distribution(
        base, (first, second), mixture_weights=weights
    )
    rng = np.random.default_rng(103)
    parents = rng.choice(3, size=200_000, p=base)
    samples = sample_stratified_connected_indices(
        parents, (first, second), mixture_weights=weights, rng=rng
    )
    observed = np.bincount(samples, minlength=3) / len(samples)
    assert observed == pytest.approx(expected, abs=3e-3)
    with pytest.raises(ValueError):
        stratified_connected_proposal_distribution(
            base, (first, second), mixture_weights=(0.8, 0.3)
        )


def test_proposal_diagnostics_expose_unseen_rare_tail():
    proposal = np.asarray([1.0 - 1e-9, 1e-9])
    target = np.asarray([0.5, 0.5])
    diagnostics = proposal_importance_diagnostics(
        proposal, target, sample_count=32_768
    )
    assert diagnostics["effective_sample_size_fraction"] < 1e-8
    assert diagnostics["maximum_weight_index"] == 1
    assert diagnostics["expected_maximum_weight_state_hits"] == pytest.approx(
        32_768e-9
    )
    assert diagnostics["expected_unseen_target_mass"] > 0.49


def test_connected_proposal_validates_parameters():
    matrix = sparse.eye(2)
    with pytest.raises(ValueError):
        hamiltonian_connected_transition(matrix, amplitude_power=0.0)
    transition = hamiltonian_connected_transition(matrix)
    with pytest.raises(ValueError):
        mixed_connected_proposal_distribution(
            np.asarray([0.5, 0.5]), transition, mixture_weight=1.1
        )
