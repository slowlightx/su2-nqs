from __future__ import annotations

from itertools import combinations
import math

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.spin_symmetry_mixture import (
    d6_complete_dimer_matchings,
    d6_point_group_images,
    dimer_uniform_log_probabilities,
    metropolis_spin_symmetry_mixture_step,
    run_spin_symmetry_mixture_sweeps,
    sample_dimer_uniform_proposal,
)


jax.config.update("jax_enable_x64", True)


SMALL_MATCHINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def test_n36_d6_and_dimer_orbit_are_complete() -> None:
    images = d6_point_group_images(((6, 0), (0, 6)))
    assert len(images) == 12
    identity = tuple(range(36))
    image_set = set(images)
    assert identity in image_set
    for image in images:
        inverse = tuple(np.argsort(np.asarray(image)))
        assert inverse in image_set

    matchings = d6_complete_dimer_matchings(6, 6)
    assert len(matchings) == 12
    j1_edges = {
        tuple(sorted((first, second)))
        for first, second, _weight in triangular_hnf_j1j2_bonds(
            ((6, 0), (0, 6)), j1=1.0, j2=0.0
        )
    }
    for matching in matchings:
        assert len(matching) == 18
        assert {site for pair in matching for site in pair} == set(range(36))
        assert set(matching) <= j1_edges


def test_dimer_uniform_density_is_normalized_and_strictly_positive() -> None:
    basis = np.asarray(enumerate_sz_configs(4, 2), dtype=np.int32)
    log_q = np.asarray(
        dimer_uniform_log_probabilities(
            jnp.asarray(basis), SMALL_MATCHINGS, uniform_floor=0.05
        )
    )
    probabilities = np.exp(log_q)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=2.0e-15)
    assert np.min(probabilities) > 0.0

    outside = jnp.asarray(((0, 0, 0, 0), (1, 1, 1, 1)), dtype=jnp.int32)
    assert np.all(np.isneginf(np.asarray(
        dimer_uniform_log_probabilities(outside, SMALL_MATCHINGS, 0.05)
    )))


def test_dimer_uniform_sampling_matches_analytic_density() -> None:
    basis = tuple(enumerate_sz_configs(4, 2))
    positions = {state: index for index, state in enumerate(basis)}
    states, used_uniform = sample_dimer_uniform_proposal(
        jax.random.PRNGKey(11), 65_536, SMALL_MATCHINGS, 0.08
    )
    counts = np.zeros(len(basis), dtype=np.float64)
    for row in np.asarray(states):
        counts[positions[tuple(int(value) for value in row)]] += 1.0
    empirical = counts / counts.sum()
    exact = np.exp(np.asarray(dimer_uniform_log_probabilities(
        jnp.asarray(basis, dtype=jnp.int32), SMALL_MATCHINGS, 0.08
    )))
    assert 0.5 * np.sum(np.abs(empirical - exact)) < 0.012
    assert abs(float(np.mean(np.asarray(used_uniform))) - 0.08) < 0.005


def test_independence_hastings_kernel_obeys_pairwise_detailed_balance() -> None:
    basis = np.asarray(enumerate_sz_configs(4, 2), dtype=np.int32)
    q = np.exp(np.asarray(dimer_uniform_log_probabilities(
        jnp.asarray(basis), SMALL_MATCHINGS, 0.07
    )))
    fields = np.asarray((-0.8, -0.1, 0.35, 0.9))
    pi = np.exp(0.7 * basis @ fields)
    pi /= pi.sum()
    for first, second in combinations(range(len(basis)), 2):
        forward = q[second] * min(
            1.0, pi[second] * q[first] / (pi[first] * q[second])
        )
        backward = q[first] * min(
            1.0, pi[first] * q[second] / (pi[second] * q[first])
        )
        np.testing.assert_allclose(
            pi[first] * forward, pi[second] * backward, atol=2.0e-16
        )


def test_a1_orbit_moves_accept_and_indexed_replay_is_chunk_invariant() -> None:
    images = d6_point_group_images(((2, 0), (0, 2)))
    basis = jnp.asarray(enumerate_sz_configs(4, 2), dtype=jnp.int32)
    # This amplitude depends only on total magnetization and is therefore
    # exactly invariant under every site permutation.
    def invariant_amplitude(_params, state):
        return jnp.exp(0.1 * jnp.sum(state))

    amplitudes = jax.vmap(lambda row: invariant_amplitude(None, row))(basis)
    updated = metropolis_spin_symmetry_mixture_step(
        None,
        jax.random.PRNGKey(21),
        basis,
        amplitudes,
        amplitude_fn=invariant_amplitude,
        exchange_edges=((0, 1), (1, 3), (2, 3), (0, 2)),
        kernel_weights=(0.0, 1.0, 0.0),
        d6_images=images,
        dimer_matchings=SMALL_MATCHINGS,
        uniform_floor=0.05,
    )
    _states, _amplitudes, accepted, nontrivial, kernel = updated
    np.testing.assert_array_equal(np.asarray(kernel), 1)
    assert np.all(np.asarray(accepted)[np.asarray(nontrivial)])

    weights = jnp.asarray((-0.7, -0.2, 0.35, 0.8), dtype=jnp.float64)

    def amplitude(params, state):
        return jnp.exp(params * jnp.dot(weights, state))

    start = basis
    kwargs = {
        "amplitude_fn": amplitude,
        "exchange_edges": ((0, 1), (1, 3), (2, 3), (0, 2)),
        "kernel_weights": (0.6, 0.2, 0.2),
        "d6_images": images,
        "dimer_matchings": SMALL_MATCHINGS,
        "uniform_floor": 0.05,
        "bond_proposal_probability": 0.8,
    }
    master = jax.random.PRNGKey(22)
    full_states, full_values, full_index, _ = run_spin_symmetry_mixture_sweeps(
        jnp.asarray(0.55), master, 0, start, sweeps=20, **kwargs
    )
    first_states, first_values, first_index, _ = run_spin_symmetry_mixture_sweeps(
        jnp.asarray(0.55), master, 0, start, sweeps=7, **kwargs
    )
    resumed_states, resumed_values, resumed_index, _ = run_spin_symmetry_mixture_sweeps(
        jnp.asarray(0.55),
        master,
        first_index,
        first_states,
        amplitudes=first_values,
        sweeps=13,
        **kwargs,
    )
    assert full_index == resumed_index == 20
    np.testing.assert_array_equal(np.asarray(full_states), np.asarray(resumed_states))
    np.testing.assert_array_equal(np.asarray(full_values), np.asarray(resumed_values))
