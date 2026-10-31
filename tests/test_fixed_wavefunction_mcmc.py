from __future__ import annotations

import numpy as np

from su2_nqs.vmc.fixed_wavefunction_mcmc import (
    chain_mean_standard_error,
    fixed_wavefunction_local_energies,
    sample_fixed_wavefunction_exchange,
)


def _constant_sz_amplitudes(states: np.ndarray) -> np.ndarray:
    array = np.asarray(states)
    return np.where(np.sum(array, axis=1) == 2, 1.0, 0.0)


def test_fixed_exchange_sampler_preserves_sector_and_uniform_state() -> None:
    sample = sample_fixed_wavefunction_exchange(
        _constant_sz_amplitudes,
        n_sites=4,
        n_up=2,
        chain_count=16,
        burn_sweeps=20,
        measurement_blocks=10,
        decorrelation_sweeps=3,
        seed=7,
    )
    assert sample.configurations.shape == (160, 4)
    np.testing.assert_array_equal(np.sum(sample.configurations, axis=1), 2)
    assert sample.acceptance_fraction == 1.0
    assert sample.burn_acceptance_fraction == 1.0


def test_local_energy_matches_constant_state_on_four_site_ring() -> None:
    states = np.asarray(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=np.int8
    )
    bonds = ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0))
    local = fixed_wavefunction_local_energies(
        _constant_sz_amplitudes, states, bonds, batch_size=2
    )
    # A constant fixed-Sz wavefunction is the fully symmetric S=2, M=0 state.
    np.testing.assert_allclose(local, 1.0, atol=1.0e-14)


def test_chain_mean_standard_error_uses_chain_trajectories() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 4.0])
    # Two blocks by two chains give chain means (1.5, 3.5).
    assert chain_mean_standard_error(
        values, chain_count=2, measurement_blocks=2
    ) == 1.0
