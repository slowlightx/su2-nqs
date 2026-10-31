from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.vmc.transition_moments import (
    estimate_transition_moments,
    flatten_complex_moments,
    hermitize_transition_moments,
    joint_block_bootstrap,
    stratified_observation_weights,
    unflatten_complex_moments,
)


def _orthonormal_states(seed: int, dimension: int = 9, count: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, count)) + 1.0j * rng.normal(
        size=(dimension, count)
    )
    states, _ = np.linalg.qr(matrix)
    return np.asarray(states, dtype=np.complex128)


def test_exhaustive_transition_moments_have_bra_ket_orientation() -> None:
    states = _orthonormal_states(771)
    rng = np.random.default_rng(772)
    raw = rng.normal(size=(2, len(states), len(states))) + 1.0j * rng.normal(
        size=(2, len(states), len(states))
    )
    operators = raw + raw.conj().transpose(0, 2, 1)
    actions = np.einsum("oxy,ya->xoa", operators, states, optimize=True)
    proposal = np.linspace(1.0, 2.0, len(states), dtype=np.float64)
    proposal /= np.sum(proposal)
    estimate = estimate_transition_moments(
        states,
        actions,
        proposal,
        observation_weights=proposal,
    )
    exact = np.einsum("xb,xoa->oba", states.conj(), actions, optimize=True)
    np.testing.assert_allclose(estimate.moments, exact, atol=3.0e-14)
    projected = np.einsum(
        "xb,oxy,ya->oba", states.conj(), operators, states, optimize=True
    )
    np.testing.assert_allclose(exact, projected, atol=3.0e-14)
    assert estimate.moments.shape == (2, 3, 3)
    assert np.min(estimate.root_ratio_ess.fractions) > 0.0
    assert np.max(estimate.root_ratio_ess.fractions) <= 1.0


def test_stratified_design_weights_are_not_self_normalized() -> None:
    strata = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int64)
    weights = stratified_observation_weights(strata, (0.25, 0.75))
    np.testing.assert_allclose(
        weights, (0.0625, 0.375, 0.0625, 0.0625, 0.375, 0.0625)
    )
    assert np.sum(weights) == pytest.approx(1.0, abs=2.0e-15)

    state = np.ones((len(strata), 1), dtype=np.complex128)
    action = state[:, None, :]
    proposal = np.ones(len(strata), dtype=np.float64)
    with pytest.raises(ValueError, match="not self-normalized"):
        estimate_transition_moments(
            state,
            action,
            proposal,
            observation_weights=0.9 * weights,
        )


def test_hermitization_and_complex_flattening_are_exact_inverses() -> None:
    rng = np.random.default_rng(901)
    moments = rng.normal(size=(3, 2, 2)) + 1.0j * rng.normal(size=(3, 2, 2))
    hermitian = hermitize_transition_moments(moments)
    np.testing.assert_allclose(
        hermitian, hermitian.conj().transpose(0, 2, 1), atol=2.0e-15
    )
    flattened = flatten_complex_moments(moments)
    np.testing.assert_allclose(
        unflatten_complex_moments(flattened, moments.shape), moments, atol=0.0
    )


def test_joint_bootstrap_preserves_full_covariance_and_nonlinear_replicates() -> None:
    rng = np.random.default_rng(1729)
    block_count = 8
    draws_per_block_per_stratum = 5
    indices: list[int] = []
    strata: list[int] = []
    blocks: list[int] = []
    states = _orthonormal_states(1730, dimension=7, count=2)
    probabilities = np.abs(states) ** 2
    proposal_full = np.mean(probabilities, axis=1)
    operator = np.diag(np.linspace(-0.7, 0.9, len(states))).astype(np.complex128)
    actions_full = np.stack((states, operator @ states), axis=1)
    for block in range(block_count):
        for stratum in range(2):
            draw = rng.choice(
                len(states),
                size=draws_per_block_per_stratum,
                p=probabilities[:, stratum],
            )
            indices.extend(int(value) for value in draw)
            strata.extend([stratum] * len(draw))
            blocks.extend([block] * len(draw))
    index = np.asarray(indices, dtype=np.int64)
    weights = stratified_observation_weights(np.asarray(strata), (0.5, 0.5))
    estimate = estimate_transition_moments(
        states[index],
        actions_full[index],
        proposal_full[index],
        observation_weights=weights,
    )

    def nonlinear(value: np.ndarray) -> np.ndarray:
        hermitian = hermitize_transition_moments(value)
        return np.asarray(
            [abs(hermitian[0, 1, 0]), np.linalg.norm(hermitian[1])],
            dtype=np.float64,
        )

    bootstrap = joint_block_bootstrap(
        estimate,
        np.asarray(blocks),
        bootstrap_resamples=300,
        seed=1731,
        nonlinear_transform=nonlinear,
        nonlinear_names=("overlap_null", "operator_frobenius"),
    )
    parameter_count = 2 * np.prod(estimate.moments.shape)
    assert bootstrap.covariance.shape == (parameter_count, parameter_count)
    assert bootstrap.replicate_estimates.shape == (300, 2, 2, 2)
    assert bootstrap.block_count == block_count
    np.testing.assert_allclose(
        bootstrap.block_weights,
        np.full(block_count, 1.0 / block_count),
        atol=2.0e-15,
    )
    assert bootstrap.nonlinear is not None
    assert bootstrap.nonlinear.names == ("overlap_null", "operator_frobenius")
    assert bootstrap.nonlinear.replicate_values.shape == (300, 2)
    assert np.min(np.linalg.eigvalsh(bootstrap.covariance)) > -1.0e-14


def test_bootstrap_fails_closed_on_unbalanced_design_blocks() -> None:
    states = np.ones((4, 1), dtype=np.complex128) / 2.0
    actions = states[:, None, :]
    proposal = np.full(4, 0.25)
    estimate = estimate_transition_moments(
        states,
        actions,
        proposal,
        observation_weights=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    with pytest.raises(ValueError, match="equal design weight"):
        joint_block_bootstrap(
            estimate,
            np.asarray([0, 0, 1, 1]),
            bootstrap_resamples=20,
            seed=12,
        )
