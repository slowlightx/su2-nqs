"""Tests for the optimizer used in matched exact-small tree controls."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.exact_state_sr import (
    train_exact_state_sr,
    train_exact_state_sr_trust_region,
    train_product_state_sr_trust_region,
)


jax.config.update("jax_enable_x64", True)


def test_exact_state_sr_lowers_two_level_energy() -> None:
    params = {"amplitudes": jnp.asarray([1.0, 1.0])}
    trained, report = train_exact_state_sr(
        params,
        state_fn=lambda values: values["amplitudes"],
        hamiltonian=jnp.diag(jnp.asarray([0.0, 1.0])),
        steps=30,
        learning_rate=0.1,
        damping=1.0e-2,
        record_every=10,
    )
    normalized = np.array(trained["amplitudes"])
    normalized /= np.linalg.norm(normalized)
    assert report["final_energy"] < 1.0e-5
    assert abs(normalized[0]) > 0.99999
    assert report["history"][-1]["step"] == 30.0


def test_exact_state_sr_accepts_matrix_free_hamiltonian_action() -> None:
    params = {"amplitudes": jnp.asarray([1.0, 1.0])}
    diagonal = jnp.asarray([0.0, 1.0])
    trained, report = train_exact_state_sr(
        params,
        state_fn=lambda values: values["amplitudes"],
        hamiltonian_action=lambda state: diagonal * state,
        steps=30,
        learning_rate=0.1,
        damping=1.0e-2,
        record_every=10,
    )
    normalized = np.array(trained["amplitudes"])
    normalized /= np.linalg.norm(normalized)
    assert report["final_energy"] < 1.0e-5
    assert abs(normalized[0]) > 0.99999


@pytest.mark.parametrize(
    ("hamiltonian", "hamiltonian_action"),
    [(None, None), (jnp.eye(2), lambda state: state)],
)
def test_exact_state_sr_requires_exactly_one_hamiltonian_interface(
    hamiltonian, hamiltonian_action
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        train_exact_state_sr(
            {"amplitudes": jnp.asarray([1.0, 1.0])},
            state_fn=lambda values: values["amplitudes"],
            hamiltonian=hamiltonian,
            hamiltonian_action=hamiltonian_action,
            steps=1,
            learning_rate=0.1,
            damping=0.01,
        )


def test_exact_state_sr_rejects_action_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="state-vector shape"):
        train_exact_state_sr(
            {"amplitudes": jnp.asarray([1.0, 1.0])},
            state_fn=lambda values: values["amplitudes"],
            hamiltonian_action=lambda state: state[:1],
            steps=1,
            learning_rate=0.1,
            damping=0.01,
        )


def test_exact_state_sr_is_invariant_to_constant_state_scale() -> None:
    params = {"amplitudes": jnp.asarray([0.8, 0.6])}
    kwargs = dict(
        hamiltonian=jnp.asarray([[0.0, -0.2], [-0.2, 1.0]]),
        steps=1,
        learning_rate=0.05,
        damping=1.0e-2,
        record_every=1,
    )
    first, _ = train_exact_state_sr(
        params, state_fn=lambda values: values["amplitudes"], **kwargs
    )
    second, _ = train_exact_state_sr(
        params, state_fn=lambda values: 7.0 * values["amplitudes"], **kwargs
    )
    np.testing.assert_allclose(first["amplitudes"], second["amplitudes"], atol=1e-11)


def test_exact_state_sr_rejects_complex_parameters() -> None:
    with pytest.raises(TypeError, match="real parameters"):
        train_exact_state_sr(
            {"x": jnp.asarray([1.0 + 0.0j])},
            state_fn=lambda values: values["x"],
            hamiltonian=jnp.eye(1),
            steps=1,
            learning_rate=0.1,
            damping=0.01,
        )


def test_trust_region_sr_is_energy_monotone_and_respects_fs_radius() -> None:
    trained, report = train_exact_state_sr_trust_region(
        {"amplitudes": jnp.asarray([1.0, 1.0])},
        state_fn=lambda values: values["amplitudes"],
        hamiltonian=jnp.diag(jnp.asarray([0.0, 1.0])),
        steps=12,
        initial_learning_rate=1.0,
        initial_damping=1.0e-5,
        maximum_fs_infidelity=1.0e-2,
        record_every=1,
    )
    energies = [row["energy"] for row in report["history"]]
    assert all(right <= left + 1.0e-14 for left, right in zip(energies, energies[1:]))
    assert all(
        row["fs_infidelity"] <= 1.0e-2 + 1.0e-14
        for row in report["history"]
    )
    assert report["final_energy"] < 1.0e-4
    normalized = np.array(trained["amplitudes"])
    normalized /= np.linalg.norm(normalized)
    assert abs(normalized[0]) > 0.9999


def test_product_fs_sr_lowers_each_ray_without_relative_phase_metric() -> None:
    params = {"angles": jnp.asarray([0.7, 0.6]), "phases": jnp.asarray([0.3, -0.8])}

    def states(values):
        angles = values["angles"]
        phases = values["phases"]
        rays = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
        return jnp.exp(1j * phases)[:, None] * rays

    diagonal = jnp.asarray([[0.0, 1.0], [0.0, 2.0]])
    trained, report = train_product_state_sr_trust_region(
        params,
        states_fn=states,
        hamiltonian_action=lambda rays: diagonal * rays,
        steps=16,
        initial_learning_rate=1.0,
        initial_damping=1.0e-4,
        maximum_mean_fs_infidelity=1.0e-2,
        maximum_state_fs_infidelity=2.0e-2,
    )
    assert max(report["final_state_energies"]) < 1.0e-7
    # Per-ray phases are pure gauges and must not be driven by the FS metric.
    np.testing.assert_allclose(trained["phases"], params["phases"], atol=1e-11)
    assert all(
        row["mean_fs_infidelity"] <= 1.0e-2 + 1.0e-14
        and row["maximum_state_fs_infidelity"] <= 2.0e-2 + 1.0e-14
        for row in report["history"]
    )


def test_product_fs_sr_can_fix_an_exact_uniform_parameter_gauge() -> None:
    params = {"weights": jnp.asarray([0.9, -0.1, 1.4])}

    def center(vector):
        return vector - jnp.mean(vector)

    def states(values):
        centered = center(values["weights"])
        return jnp.stack((centered[:2], centered[1:]), axis=0)

    flat_size = 3
    canonicalize = lambda vector: center(vector)
    trained, report = train_product_state_sr_trust_region(
        params,
        states_fn=states,
        hamiltonian_action=lambda rays: rays * jnp.asarray([[0.0, 1.0], [0.0, 2.0]]),
        steps=4,
        initial_learning_rate=0.2,
        initial_damping=1.0e-2,
        maximum_mean_fs_infidelity=5.0e-2,
        canonicalize_flat=canonicalize,
        project_flat_tangent=canonicalize,
    )
    assert flat_size == len(trained["weights"])
    assert abs(float(jnp.mean(trained["weights"]))) < 1.0e-13
    assert report["accepted_steps"] >= 1


def test_product_fs_dual_dense_matches_registered_linear_residual() -> None:
    params = {"matrix": jnp.asarray([[0.9, 0.2], [0.6, -0.4], [0.1, 0.7]])}

    def states(values):
        matrix = values["matrix"]
        return jnp.stack((matrix[0] + 0.2 * matrix[2], matrix[1] - 0.3 * matrix[2]))

    _, report = train_product_state_sr_trust_region(
        params,
        states_fn=states,
        hamiltonian_action=lambda rays: rays * jnp.asarray([[0.0, 1.0], [0.0, 1.5]]),
        steps=3,
        initial_learning_rate=0.2,
        initial_damping=1.0e-3,
        maximum_mean_fs_infidelity=2.0e-2,
        linear_solver="dual_dense",
        maximum_linear_relative_residual=1.0e-9,
    )
    assert report["linear_solver"] == "dual_dense"
    assert max(row["cg_relative_residual"] for row in report["history"]) < 1.0e-9
    assert report["final_mean_energy"] < np.mean(report["initial_state_energies"])
