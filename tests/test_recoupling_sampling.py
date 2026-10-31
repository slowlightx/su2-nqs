import numpy as np
import pytest

from su2_nqs.benchmarks.recoupling_sampling import (
    local_choice_recoupling_moments,
)
from su2_nqs.operators.permutation import (
    adjacent_swap_path_connectivity,
    site_permutation_to_adjacent_swap_word,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _brute_history_moments(path, image, amplitude, target_q):
    walkers = [(path, 1.0, 1.0 + 0.0j)]
    for bond in reversed(site_permutation_to_adjacent_swap_word(image)):
        updated = []
        for current, probability, weight in walkers:
            connected, values = adjacent_swap_path_connectivity(
                current, bond, target_q=target_q
            )
            probabilities = np.abs(values) ** 2
            probabilities /= np.sum(probabilities)
            for candidate, coefficient, local_probability in zip(
                connected, values, probabilities, strict=True
            ):
                updated.append(
                    (
                        candidate,
                        probability * local_probability,
                        weight * coefficient / local_probability,
                    )
                )
        walkers = updated
    samples = np.asarray(
        [weight * amplitude(source) for source, _, weight in walkers]
    )
    probabilities = np.asarray([probability for _, probability, _ in walkers])
    return (
        np.sum(probabilities * samples),
        float(np.sum(probabilities * np.abs(samples) ** 2)),
        len(walkers),
    )


def test_dynamic_recoupling_moments_match_explicit_histories() -> None:
    n, target_q = 6, 2
    image = (2, 0, 5, 1, 4, 3)
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    rng = np.random.default_rng(20260912)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    amplitude = lambda path: state[positions[path]]

    for path in basis[:: max(len(basis) // 4, 1)]:
        expected_mean, expected_second, history_count = _brute_history_moments(
            path, image, amplitude, target_q
        )
        actual = local_choice_recoupling_moments(
            path, image, amplitude, target_q=target_q
        )
        assert actual["mean"] == pytest.approx(expected_mean, abs=2.0e-13)
        assert actual["second_moment"] == pytest.approx(
            expected_second, abs=2.0e-12
        )
        assert actual["total_history_count"] == history_count
        assert actual["maximum_local_normalization_error"] < 2.0e-14


def test_identity_recoupling_estimator_has_zero_variance() -> None:
    path = (1, 0, 1, 0)
    value = 0.3 - 0.7j
    result = local_choice_recoupling_moments(
        path,
        (0, 1, 2, 3),
        lambda _: value,
        target_q=0,
    )
    assert result["mean"] == pytest.approx(value)
    assert result["variance"] == pytest.approx(0.0)
    assert result["terminal_support"] == 1
    assert result["total_history_count"] == 1
