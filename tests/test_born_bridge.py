import numpy as np

from su2_nqs.vmc.born_bridge import born_mixture_bridge_observations


def test_born_mixture_bridge_is_exact_and_bounded_on_complete_support():
    candidate_probability = np.asarray([0.8, 0.2])
    current_probability = np.asarray([0.3, 0.7])
    mixture = 0.5 * (candidate_probability + current_probability)
    diagonal_energy = np.asarray([-1.0, 2.0], dtype=np.complex128)
    observed = born_mixture_bridge_observations(
        diagonal_energy,
        diagonal_energy,
        np.log(candidate_probability),
        np.log(current_probability),
    )

    candidate_energy = float(candidate_probability @ np.real(diagonal_energy))
    current_energy = float(current_probability @ np.real(diagonal_energy))
    np.testing.assert_allclose(
        mixture @ observed.energy_difference,
        candidate_energy - current_energy,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mixture @ observed.forward_kl,
        np.sum(candidate_probability * np.log(candidate_probability / current_probability)),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        mixture @ observed.reverse_kl,
        np.sum(current_probability * np.log(current_probability / candidate_probability)),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        observed.candidate_weight + observed.current_weight, 2.0, atol=1.0e-14
    )
    assert np.max(observed.candidate_weight) <= 2.0
    assert np.max(observed.current_weight) <= 2.0


def test_born_mixture_bridge_rejects_mismatched_inputs():
    with np.testing.assert_raises_regex(ValueError, "matching vectors"):
        born_mixture_bridge_observations(
            np.ones(2), np.ones(1), np.zeros(2), np.zeros(2)
        )
