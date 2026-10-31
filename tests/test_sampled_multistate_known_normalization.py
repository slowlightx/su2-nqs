from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.sampled_multistate_known_normalization import (
    per_observation_hermitian_pair_hamiltonian,
    sampled_moments_with_known_normalization,
)
from su2_nqs.vmc.sampled_multistate import sampled_subspace_moments


def _problem(seed: int = 95600):
    rng = np.random.default_rng(seed)
    dimension, roots = 19, 3
    states = rng.normal(size=(dimension, roots)) + 1j * rng.normal(
        size=(dimension, roots)
    )
    states /= np.linalg.norm(states, axis=0, keepdims=True)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    hamiltonian = 0.5 * (raw + raw.conj().T)
    actions = hamiltonian @ states
    proposal = np.mean(np.abs(states) ** 2, axis=1)
    return states, actions, proposal


def test_per_observation_pair_equals_post_mean_hermitization() -> None:
    states, actions, proposal = _problem()
    indices = np.asarray([0, 0, 2, 5, 7, 11, 18])
    ordinary = sampled_subspace_moments(
        states[indices], actions[indices], proposal[indices]
    )
    paired = per_observation_hermitian_pair_hamiltonian(
        states[indices], actions[indices], proposal[indices]
    )

    np.testing.assert_allclose(paired, ordinary.hamiltonian, atol=3.0e-15)
    np.testing.assert_allclose(paired, paired.conj().T, atol=0.0)
    assert ordinary.hamiltonian_hermiticity.relative_frobenius_error > 1.0e-3


def test_known_normalization_replaces_only_overlap_diagonal() -> None:
    states, actions, proposal = _problem(95601)
    indices = np.asarray([0, 1, 1, 3, 4, 8, 10, 13, 17])
    ordinary = sampled_subspace_moments(
        states[indices], actions[indices], proposal[indices]
    )
    audited = sampled_moments_with_known_normalization(
        states[indices], actions[indices], proposal[indices]
    )

    np.testing.assert_allclose(np.diag(audited.moments.overlap), 1.0, atol=0.0)
    off_diagonal = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(
        audited.moments.overlap[off_diagonal],
        ordinary.overlap[off_diagonal],
        atol=0.0,
    )
    np.testing.assert_allclose(
        audited.moments.hamiltonian, ordinary.hamiltonian, atol=0.0
    )
    np.testing.assert_allclose(
        audited.moments.action_gram, ordinary.action_gram, atol=0.0
    )
    assert audited.paired_vs_post_mean_maximum_error < 3.0e-15


def test_exhaustive_known_norm_control_is_unbiased() -> None:
    states, actions, proposal = _problem(95602)
    audited = sampled_moments_with_known_normalization(
        states,
        actions,
        proposal,
        observation_weights=proposal,
    )
    np.testing.assert_allclose(
        audited.moments.overlap,
        states.conj().T @ states,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        audited.paired_hamiltonian,
        states.conj().T @ actions,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
