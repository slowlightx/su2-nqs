import numpy as np
import pytest

from su2_nqs.benchmarks.hamiltonian_connected_proposal import (
    build_hamiltonian_connected_kernel,
    connected_mixture_probabilities,
)


def test_connected_kernel_is_column_stochastic_and_pushes_probability() -> None:
    hamiltonian = np.asarray(
        (
            (1.0, -1.0, 0.0, 0.0),
            (-1.0, 2.0, -0.5, 0.0),
            (0.0, -0.5, 3.0, 0.0),
            (0.0, 0.0, 0.0, 4.0),
        )
    )
    kernel = build_hamiltonian_connected_kernel(hamiltonian)
    np.testing.assert_array_equal(kernel.degrees, (1, 2, 1, 1))
    np.testing.assert_allclose(
        np.asarray(kernel.transition.sum(axis=0)).reshape(-1), 1.0
    )
    base = np.asarray((0.5, 0.3, 0.2, 0.0))
    np.testing.assert_allclose(
        kernel.distribution(base), (0.15, 0.7, 0.15, 0.0)
    )


def test_connected_sampler_matches_exact_transition_distribution() -> None:
    hamiltonian = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )
    kernel = build_hamiltonian_connected_kernel(hamiltonian)
    sources = np.full(40_000, 1, dtype=np.int32)
    draws = kernel.sample(sources, np.random.default_rng(1701))
    frequencies = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(frequencies, (0.5, 0.0, 0.5), atol=0.01)


def test_connected_mixture_has_full_support_and_exact_weights() -> None:
    hamiltonian = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )
    kernel = build_hamiltonian_connected_kernel(hamiltonian)
    proposal = connected_mixture_probabilities(
        np.asarray((1.0, 0.0, 0.0)),
        kernel,
        uniform_fraction=0.25,
        connected_fraction=0.25,
    )
    np.testing.assert_allclose(proposal, (7.0 / 12.0, 1.0 / 3.0, 1.0 / 12.0))
    assert np.all(proposal > 0.0)
    with pytest.raises(ValueError, match="sum below one"):
        connected_mixture_probabilities(
            np.ones(3), kernel, uniform_fraction=0.5, connected_fraction=0.5
        )
