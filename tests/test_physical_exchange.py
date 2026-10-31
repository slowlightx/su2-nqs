from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.physical_exchange import (
    canonical_exchange_edges,
    hubbard_bond_exchange_proposals,
    hubbard_exchange_proposals,
    initialize_hubbard_exchange_chains,
    initialize_spin_exchange_chains,
    metropolis_exchange_step,
    run_exchange_sweeps,
    spin_bond_exchange_proposals,
    spin_exchange_proposals,
)


jax.config.update("jax_enable_x64", True)


def test_exchange_proposals_preserve_fixed_sectors() -> None:
    spin = initialize_spin_exchange_chains(
        jax.random.PRNGKey(1), n_chains=64, n_sites=8, n_up=3
    )
    spin_proposed = spin_exchange_proposals(jax.random.PRNGKey(2), spin)
    np.testing.assert_array_equal(np.sum(spin_proposed, axis=1), 3)

    hubbard = initialize_hubbard_exchange_chains(
        jax.random.PRNGKey(3),
        n_chains=64,
        n_sites=8,
        n_up=3,
        n_down=4,
    )
    proposed = hubbard_exchange_proposals(jax.random.PRNGKey(4), hubbard)
    np.testing.assert_array_equal(np.sum(np.asarray(proposed) & 1, axis=1), 3)
    np.testing.assert_array_equal(
        np.sum((np.asarray(proposed) >> 1) & 1, axis=1), 4
    )

    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    spin_bond = spin_bond_exchange_proposals(
        jax.random.PRNGKey(5), spin[:, :4], edges
    )
    np.testing.assert_array_equal(
        np.sum(spin_bond, axis=1), np.sum(np.asarray(spin[:, :4]), axis=1)
    )
    hubbard_bond = hubbard_bond_exchange_proposals(
        jax.random.PRNGKey(6), hubbard[:, :4], edges
    )
    np.testing.assert_array_equal(
        np.sum(np.asarray(hubbard_bond) & 1, axis=1),
        np.sum(np.asarray(hubbard[:, :4]) & 1, axis=1),
    )
    np.testing.assert_array_equal(
        np.sum((np.asarray(hubbard_bond) >> 1) & 1, axis=1),
        np.sum((np.asarray(hubbard[:, :4]) >> 1) & 1, axis=1),
    )


def test_canonical_exchange_edges_are_undirected_and_checked() -> None:
    assert canonical_exchange_edges(((1, 0), (0, 1), (2, 3)), n_sites=4) == (
        (0, 1),
        (2, 3),
    )


def test_spin_exchange_stationary_distribution_matches_amplitude_squared() -> None:
    n_chains = 8192
    states = initialize_spin_exchange_chains(
        jax.random.PRNGKey(10), n_chains=n_chains, n_sites=4, n_up=2
    )
    weights = jnp.asarray([-0.7, -0.2, 0.35, 0.8], dtype=jnp.float64)

    def amplitude(params, state):
        return jnp.exp(params * jnp.dot(weights, state))

    alpha = jnp.asarray(0.55)
    advance = jax.jit(
        lambda key, current: run_exchange_sweeps(
            alpha,
            key,
            current,
            amplitude_fn=amplitude,
            particle_kind="spin",
            sweeps=160,
        )
    )
    sampled, amplitudes, _, diagnostics = advance(jax.random.PRNGKey(11), states)
    assert np.all(np.isfinite(np.asarray(amplitudes)))
    assert 0.05 < float(diagnostics["accepted_move_fraction"]) < 0.8

    basis = enumerate_sz_configs(4, 2)
    positions = {state: index for index, state in enumerate(basis)}
    counts = np.zeros(len(basis), dtype=np.float64)
    for row in np.asarray(sampled):
        counts[positions[tuple(int(value) for value in row)]] += 1
    empirical = counts / counts.sum()
    exact_amplitudes = np.asarray(
        [amplitude(alpha, jnp.asarray(row)) for row in basis], dtype=np.float64
    )
    exact = exact_amplitudes**2
    exact /= exact.sum()
    assert 0.5 * np.sum(np.abs(empirical - exact)) < 0.025


def test_spin_bond_exchange_stationary_distribution_matches_amplitude_squared() -> None:
    n_chains = 8192
    states = initialize_spin_exchange_chains(
        jax.random.PRNGKey(30), n_chains=n_chains, n_sites=4, n_up=2
    )
    weights = jnp.asarray([-0.7, -0.2, 0.35, 0.8], dtype=jnp.float64)

    def amplitude(params, state):
        return jnp.exp(params * jnp.dot(weights, state))

    edges = ((0, 1), (1, 2), (2, 3), (0, 3))
    alpha = jnp.asarray(0.55)
    advance = jax.jit(
        lambda key, current: run_exchange_sweeps(
            alpha,
            key,
            current,
            amplitude_fn=amplitude,
            particle_kind="spin",
            sweeps=240,
            exchange_edges=edges,
        )
    )
    sampled, _, _, diagnostics = advance(jax.random.PRNGKey(31), states)
    assert 0.05 < float(diagnostics["accepted_move_fraction"]) < 0.8
    basis = enumerate_sz_configs(4, 2)
    positions = {state: index for index, state in enumerate(basis)}
    counts = np.zeros(len(basis), dtype=np.float64)
    for row in np.asarray(sampled):
        counts[positions[tuple(int(value) for value in row)]] += 1
    empirical = counts / counts.sum()
    exact_amplitudes = np.asarray(
        [amplitude(alpha, jnp.asarray(row)) for row in basis], dtype=np.float64
    )
    exact = exact_amplitudes**2
    exact /= exact.sum()
    assert 0.5 * np.sum(np.abs(empirical - exact)) < 0.025


def test_hubbard_exchange_sweeps_preserve_charge_and_spin() -> None:
    states = initialize_hubbard_exchange_chains(
        jax.random.PRNGKey(20),
        n_chains=128,
        n_sites=6,
        n_up=2,
        n_down=3,
    )

    def amplitude(_params, state):
        doublons = jnp.sum(state == 3)
        return jnp.exp(-0.2 * doublons)

    advance = jax.jit(
        lambda key, current: run_exchange_sweeps(
            None,
            key,
            current,
            amplitude_fn=amplitude,
            particle_kind="hubbard",
            sweeps=80,
        )
    )
    sampled, _, _, diagnostics = advance(jax.random.PRNGKey(21), states)
    concrete = np.asarray(sampled)
    np.testing.assert_array_equal(np.sum(concrete & 1, axis=1), 2)
    np.testing.assert_array_equal(np.sum((concrete >> 1) & 1, axis=1), 3)
    assert float(diagnostics["zero_amplitude_chain_fraction"]) == 0.0


def test_hubbard_mixed_bond_kernel_has_the_born_stationary_distribution() -> None:
    n_sites = 3
    n_chains = 8192
    states = initialize_hubbard_exchange_chains(
        jax.random.PRNGKey(40),
        n_chains=n_chains,
        n_sites=n_sites,
        n_up=1,
        n_down=1,
    )
    up_weights = jnp.asarray([-0.8, 0.15, 0.55], dtype=jnp.float64)
    down_weights = jnp.asarray([0.35, -0.45, 0.7], dtype=jnp.float64)

    def amplitude(params, state):
        up = state & 1
        down = (state >> 1) & 1
        doublons = jnp.sum((up == 1) & (down == 1))
        return jnp.exp(
            params
            * (
                jnp.dot(up_weights, up)
                + jnp.dot(down_weights, down)
                - 0.3 * doublons
            )
        )

    alpha = jnp.asarray(0.6)
    edges = ((0, 1), (1, 2), (0, 2))
    advance = jax.jit(
        lambda key, current: run_exchange_sweeps(
            alpha,
            key,
            current,
            amplitude_fn=amplitude,
            particle_kind="hubbard",
            sweeps=260,
            exchange_edges=edges,
            bond_proposal_probability=0.65,
            lazy_probability=0.25,
        )
    )
    sampled, _, _, diagnostics = advance(jax.random.PRNGKey(41), states)
    assert 0.02 < float(diagnostics["accepted_move_fraction"]) < 0.7

    basis = []
    for up_sites in combinations(range(n_sites), 1):
        for down_sites in combinations(range(n_sites), 1):
            row = np.zeros(n_sites, dtype=np.int32)
            row[list(up_sites)] += 1
            row[list(down_sites)] += 2
            basis.append(tuple(int(value) for value in row))
    positions = {state: index for index, state in enumerate(basis)}
    counts = np.zeros(len(basis), dtype=np.float64)
    for row in np.asarray(sampled):
        counts[positions[tuple(int(value) for value in row)]] += 1
    empirical = counts / counts.sum()
    exact_amplitudes = np.asarray(
        [amplitude(alpha, jnp.asarray(row)) for row in basis], dtype=np.float64
    )
    exact = exact_amplitudes**2
    exact /= exact.sum()
    assert 0.5 * np.sum(np.abs(empirical - exact)) < 0.025


def test_mixture_probability_is_checked() -> None:
    states = jnp.asarray([[1, 0]], dtype=jnp.int32)

    def amplitude(_params, _state):
        return jnp.asarray(1.0, dtype=jnp.float64)

    with np.testing.assert_raises_regex(ValueError, "bond_proposal_probability"):
        metropolis_exchange_step(
            None,
            jax.random.PRNGKey(42),
            states,
            jnp.ones((1,), dtype=jnp.float64),
            amplitude_fn=amplitude,
            particle_kind="spin",
            exchange_edges=((0, 1),),
            bond_proposal_probability=1.1,
        )


def test_metropolis_log_ratio_survives_squared_weight_underflow() -> None:
    states = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.int32)

    def amplitude(_params, state):
        # Both squared magnitudes underflow in float64, while their ratio is
        # finite and strongly favours the configuration with spin-up at site 1.
        return jnp.where(state[1] == 1, 1.0e-200, 1.0e-220)

    cached = jax.vmap(lambda row: amplitude(None, row))(states)
    updated, values, accepted, nontrivial = metropolis_exchange_step(
        None,
        jax.random.PRNGKey(123),
        states,
        cached,
        amplitude_fn=amplitude,
        particle_kind="spin",
    )
    concrete = np.asarray(updated)
    # With two sites every proposal swaps the occupations.  The first chain
    # moves uphill by 40 log-units of Born weight and must be accepted; all
    # returned amplitudes remain finite and nonzero.
    assert bool(np.asarray(nontrivial)[0])
    assert bool(np.asarray(accepted)[0])
    np.testing.assert_array_equal(concrete[0], np.asarray([0, 1]))
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.asarray(values) > 0.0)


def test_explicit_lazy_kernel_breaks_two_state_periodicity() -> None:
    states = jnp.tile(jnp.asarray([[1, 0]], dtype=jnp.int32), (4096, 1))

    def amplitude(_params, _state):
        return jnp.asarray(1.0, dtype=jnp.float64)

    cached = jnp.ones((states.shape[0],), dtype=jnp.float64)
    updated, _, accepted, nontrivial = metropolis_exchange_step(
        None,
        jax.random.PRNGKey(212),
        states,
        cached,
        amplitude_fn=amplitude,
        particle_kind="spin",
        exchange_edges=((0, 1),),
        lazy_probability=0.5,
    )
    moved = np.mean(np.asarray(updated)[:, 0] == 0)
    assert np.all(np.asarray(nontrivial))
    assert 0.45 < moved < 0.55
    assert np.mean(np.asarray(accepted)) == moved
