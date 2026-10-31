"""Independent dense-energy, ridge, and trajectory checks for the local pilot.

These tests concern the model-independent optimizer.  N=6 fixed-Sz states
provide an exhaustive physical Hamiltonian oracle, independently of the N=36
Clebsch-tree contraction and its exact SU(2) tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from projects.clebsch_tree_nqs.workflows.triangle_dictionary_vmc_local import (
    make_record_sampler,
    make_prepared_sr_update,
    summarize_born_audit,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains


jax.config.update("jax_enable_x64", True)


def _unconstrained_update(prepare, log_prepared, params, **kwargs):
    """Remove empirical step caps to expose the independently solved ridge."""
    return make_prepared_sr_update(
        prepare,
        log_prepared,
        params,
        damping=0.13,
        step_size=1.0e-4,
        fisher_trust_radius=100.0,
        maximum_overlap_distance=1.0,
        minimum_ess_fraction=0.01,
        **kwargs,
    )


@pytest.mark.parametrize("score_chunk_size", [1, 3])
@pytest.mark.parametrize("parameter_chunk_size", [None, 3])
def test_prepared_pullback_ridge_matches_dense_complex_original_parameter_metric(
    score_chunk_size, parameter_chunk_size,
):
    rng = np.random.default_rng(2026101901)
    params = {
        "network": jnp.asarray(rng.normal(size=3)),
        "dictionary": jnp.asarray(rng.normal(size=4)),
    }
    theta, unravel = ravel_pytree(params)
    first = jnp.asarray(rng.normal(size=(5, 7)))
    second = jnp.asarray(rng.normal(size=(5, 7)))
    coefficients = jnp.asarray(
        rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    )

    def prepare(candidate):
        flat, _ = ravel_pytree(candidate)
        return {"coefficients": first @ flat + 0.2 * (second @ flat) ** 2}

    def log_prepared(prepared, row):
        return coefficients[row[0]] @ prepared["coefficients"]

    samples = jnp.arange(7, dtype=jnp.int32)[:, None]
    energies = jnp.asarray(rng.normal(size=7) + 1j * rng.normal(size=7))
    update = _unconstrained_update(
        prepare, log_prepared, params, score_chunk_size=score_chunk_size,
        parameter_chunk_size=parameter_chunk_size,
    )
    changed, diagnostic = update(params, samples, energies)
    # Analytic chain rule through the nontrivial prepared cache, with no AD
    # or prepared-coordinate metric in the independent NumPy oracle.
    derivative = np.asarray(first) + (
        0.4 * np.asarray(second @ theta)
    )[:, None] * np.asarray(second)
    scores = np.asarray(coefficients) @ derivative
    centered = scores - scores.mean(axis=0)
    residual = np.asarray(energies - energies.mean())
    gradient = 2 * np.real(centered.conj().T @ residual) / len(samples)
    metric = 2 * np.real(centered.conj().T @ centered) / len(samples)
    direction = np.linalg.solve(metric + 0.13 * np.eye(len(theta)), gradient)
    expected = np.asarray(theta) - 1.0e-4 * direction
    actual, _ = ravel_pytree(changed)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert bool(diagnostic["accepted"])
    assert float(diagnostic["relative_linear_residual"]) < 1e-10
    assert float(diagnostic["gradient_norm"]) == pytest.approx(
        np.linalg.norm(gradient), rel=2e-12, abs=2e-12
    )
    # Every original component enters the pullback, including the network.
    assert all(
        np.linalg.norm(np.asarray(a - b)) > 0
        for a, b in zip(jax.tree.leaves(changed), jax.tree.leaves(params), strict=True)
    )


def test_N6_full_heisenberg_Born_gradient_matches_exact_rayleigh_derivative():
    graph = SzHeisenbergGraph(
        6, triangular_j1j2_bonds(2, 3, j2=0.0, periodic=True), n_up=3
    )
    matrix = jnp.asarray(graph.to_dense())
    basis = np.asarray(graph.basis)
    rng = np.random.default_rng(2026101902)
    features = jnp.asarray(
        rng.normal(size=(len(basis), 4)) + 1j * rng.normal(size=(len(basis), 4))
    )
    signs = jnp.asarray(rng.choice([-1.0, 1.0], size=len(basis)))
    phases = jnp.where(signs < 0, jnp.pi, 0.0)
    params = jnp.zeros(4, dtype=jnp.float64)
    samples = jnp.arange(len(basis), dtype=jnp.int32)[:, None]

    def log_prepared(prepared, row):
        return features[row[0]] @ prepared + 1j * phases[row[0]]

    def energy(candidate):
        values = jnp.exp(features @ candidate + 1j * phases)
        return jnp.real(jnp.vdot(values, matrix @ values) / jnp.vdot(values, values))

    # At theta=0 all 20 fixed-Sz configurations have exactly equal Born
    # weight; exhaustive unweighted samples therefore give the exact force.
    values = jnp.exp(1j * phases)
    local_energies = matrix @ values / values
    centered = np.asarray(features - features.mean(axis=0))
    force = (
        2
        * np.real(centered.conj().T @ np.asarray(local_energies - local_energies.mean()))
        / len(samples)
    )
    exact_gradient = np.asarray(jax.grad(energy)(params))
    np.testing.assert_allclose(force, exact_gradient, atol=3e-13, rtol=3e-13)
    metric = 2 * np.real(centered.conj().T @ centered) / len(samples)
    direction = np.linalg.solve(metric + 0.13 * np.eye(4), exact_gradient)
    update = _unconstrained_update(
        lambda candidate: candidate, log_prepared, params, score_chunk_size=3
    )
    changed, diagnostic = update(params, samples, local_energies)
    np.testing.assert_allclose(changed, -1.0e-4 * direction, atol=2e-13, rtol=2e-11)
    assert float(energy(changed)) < float(energy(params))
    assert float(diagnostic["gradient_norm"]) == pytest.approx(
        np.linalg.norm(exact_gradient), rel=3e-13, abs=3e-13
    )
    epsilon = 1.0e-5
    finite_difference = float((energy(-epsilon * direction) - energy(epsilon * direction)) / (2 * epsilon))
    assert finite_difference == pytest.approx(
        -float(exact_gradient @ direction), rel=3e-9, abs=3e-10
    )


def test_checkpoint_replays_next_sampler_and_original_parameter_update(tmp_path):
    params = {"network": jnp.asarray([0.1, -0.03]), "dictionary": jnp.asarray([0.07])}
    first = jnp.asarray(np.random.default_rng(2026101903).normal(size=(6, 3)))

    def prepare(candidate):
        theta, _ = ravel_pytree(candidate)
        return {"field": first @ theta + 0.01 * (first @ theta) ** 2}

    def log_prepared(prepared, row):
        return (row @ prepared["field"]).astype(jnp.complex128)

    sampler = make_record_sampler(
        lambda prepared, row: jnp.exp(log_prepared(prepared, row)).real,
        records=2,
        burn_proposals=7,
        proposals_between=3,
        exchange_edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
    )
    update = _unconstrained_update(
        prepare, log_prepared, params, score_chunk_size=2
    )
    key = jax.random.PRNGKey(2026101904)
    states = initialize_spin_exchange_chains(
        key, n_chains=8, n_sites=6, n_up=3
    )
    states, key, rows, _ = sampler(prepare(params), key, states)
    samples = rows.reshape(-1, 6)
    energies = samples @ jnp.asarray([0.1, -0.3, 0.8, -1.2, 0.7, 0.4])
    params, _ = update(params, samples, energies)
    path = tmp_path / "checkpoint.pkl"
    save_checkpoint(path, params, metadata={"rng": key, "chains": states, "step": 1})
    saved = load_checkpoint(path)

    def next_step(candidate, rng_key, chains):
        chains, rng_key, rows, sampling = sampler(prepare(candidate), rng_key, chains)
        samples = rows.reshape(-1, 6)
        local = samples @ jnp.asarray([0.1, -0.3, 0.8, -1.2, 0.7, 0.4])
        candidate, diagnostic = update(candidate, samples, local)
        return candidate, rng_key, chains, rows, sampling, diagnostic

    expected = next_step(params, key, states)
    restored = next_step(
        saved["params"], saved["metadata"]["rng"], saved["metadata"]["chains"]
    )
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)


def test_prepared_score_centering_removes_normalization_and_phase_gauges():
    params = jnp.asarray([0.1, -0.03, 0.07], dtype=jnp.float64)
    samples = jnp.arange(6)[:, None]

    def log_prepared(prepared, row):
        x = row[0] - 2.5
        return x * prepared[0] + (1 + 2j) * prepared[1] + 1j * x * x * prepared[2]

    update = _unconstrained_update(
        lambda candidate: candidate, log_prepared, params, score_chunk_size=2
    )
    energy = jnp.asarray([0.4, -1.1, 0.7, -0.2, 0.8, 0.3]) + 0.03j * jnp.arange(6)
    first, _ = update(params, samples, energy)
    second, _ = update(params, samples, energy + 7 + 2j)
    np.testing.assert_allclose(first, second, atol=2e-13, rtol=2e-13)
    assert float(first[1]) == pytest.approx(float(params[1]), abs=2e-13)


def test_born_audit_preserves_raw_complex_second_moment_and_chain_blocks():
    values = np.asarray(
        [[1 + 2j, 3 + 4j], [5 - 1j, 7 + 6j], [9 + 0j, 1.0e5 + 7j]],
        dtype=np.complex128,
    )
    rows = np.zeros((3, 2, 6), dtype=np.int32)
    rows[..., :3] = 1
    summary = summarize_born_audit(values, rows, n_sites=6)
    expected_variance = (np.mean(abs(values) ** 2) - abs(np.mean(values)) ** 2) / 6
    expected_se = values.real.mean(axis=0).std(ddof=1) / np.sqrt(2) / 6
    assert summary["variance_per_site"] == pytest.approx(expected_variance, rel=3e-15)
    assert summary["chain_block_standard_error_per_site"] == pytest.approx(expected_se)
    assert summary["energy_per_site"] == pytest.approx(values.real.mean() / 6)
    assert summary["local_energy_range_per_site"][1] == pytest.approx(1.0e5 / 6)
    assert summary["sample_records"] == 6
    assert summary["independent_chains"] == 2
    assert not summary["clipping"]
    assert not summary["importance_reweighting"]
