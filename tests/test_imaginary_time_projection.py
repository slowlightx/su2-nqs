from __future__ import annotations

import numpy as np

from su2_nqs.vmc.imaginary_time_projection import (
    exact_state_distance,
    imaginary_time_euler_target,
    select_imaginary_time_euler_step,
)


def _energy_and_variance(state: np.ndarray, hamiltonian: np.ndarray) -> tuple[float, float]:
    vector = state / np.linalg.norm(state)
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def test_exact_state_distance_ignores_global_phase_and_normalization() -> None:
    state = np.asarray([1.0, 2.0j, -0.3], dtype=np.complex128)
    distance = exact_state_distance(state, 4.2 * np.exp(0.7j) * state)
    assert distance.fubini_study_distance < 2.0e-8
    assert distance.symmetric_kl < 2.0e-15
    assert abs(distance.importance_effective_sample_fraction - 1.0) < 2.0e-15


def test_exact_state_distance_handles_extreme_born_ratios_without_overflow() -> None:
    old = np.asarray([1.0, 1.0e-160], dtype=np.complex128)
    new = np.asarray([1.0e-160, 1.0], dtype=np.complex128)
    with np.errstate(all="raise"):
        distance = exact_state_distance(old, new)
    assert np.isfinite(distance.symmetric_kl)
    assert distance.symmetric_kl > 700.0
    assert distance.importance_effective_sample_fraction == 0.0


def test_imaginary_time_euler_step_lowers_energy_at_small_tau() -> None:
    hamiltonian = np.asarray([[0.4, 0.3], [0.3, -0.7]], dtype=np.float64)
    state = np.asarray([1.0, 0.2j], dtype=np.complex128)
    action = hamiltonian @ (state / np.linalg.norm(state))
    target, shift = imaginary_time_euler_target(state, action, 0.02)
    initial_energy, _ = _energy_and_variance(state, hamiltonian)
    target_energy, _ = _energy_and_variance(target, hamiltonian)
    assert abs(shift - initial_energy) < 2.0e-15
    assert target_energy < initial_energy


def test_euler_selector_saturates_exact_fs_radius() -> None:
    hamiltonian = np.asarray([[0.6, -0.4], [-0.4, -0.2]], dtype=np.float64)
    state = np.asarray([np.sqrt(0.7), np.sqrt(0.3)], dtype=np.complex128)
    normalized = state / np.linalg.norm(state)
    selected = select_imaginary_time_euler_step(
        state,
        hamiltonian @ normalized,
        maximum_tau=0.5,
        maximum_fubini_study_distance=0.03,
        maximum_symmetric_kl=0.02,
        minimum_importance_effective_sample_fraction=0.8,
    )
    assert 0.0 < selected.tau < 0.5
    assert selected.distance.fubini_study_distance <= 0.03 * (1.0 + 2.0e-12)
    assert selected.distance.fubini_study_distance >= 0.03 * (1.0 - 2.0e-10)
    assert selected.distance.symmetric_kl <= 0.02
    assert selected.distance.importance_effective_sample_fraction >= 0.8
