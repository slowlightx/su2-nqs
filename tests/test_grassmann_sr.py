from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from su2_nqs.models.fusion_gru import FusionGRUConfig, init_fusion_gru, log_psi
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.grassmann_sr import (
    exact_grassmann_energy_sr_direction,
    exact_grassmann_sr_direction,
)
from su2_nqs.vmc.multistate import exhaustive_state_matrix


jax.config.update("jax_enable_x64", True)


def _mean_ky_fan(states: jax.Array, hamiltonian: jax.Array) -> jax.Array:
    gram_values, gram_vectors = jnp.linalg.eigh(states.conj().T @ states)
    whitening = (gram_vectors / jnp.sqrt(gram_values)[None, :]) @ gram_vectors.conj().T
    orthonormal = states @ whitening
    return jnp.real(jnp.trace(orthonormal.conj().T @ hamiltonian @ orthonormal)) / states.shape[1]


def _complex_rank_two_problem():
    key = jax.random.PRNGKey(8103)
    real, imaginary = jax.random.split(key)
    raw = jax.random.normal(real, (6, 2, 8), dtype=jnp.float64) + 1j * jax.random.normal(
        imaginary, (6, 2, 8), dtype=jnp.float64
    )
    raw = raw.at[:, :, 0].set(
        jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.2j, -0.1],
                [0.1, 0.3j],
                [-0.2, 0.1j],
                [0.05j, -0.15],
            ]
        )
    )
    parameters = jnp.asarray([1.0, 0.03, -0.04, 0.02, 0.01, -0.02, 0.04, -0.01])
    diagonal = jnp.asarray([-3.0, -2.0, -0.4, 0.7, 1.3, 2.1])
    coupling = jnp.asarray(
        [
            [0.0, 0.1j, 0.03, 0.0, 0.0, 0.0],
            [-0.1j, 0.0, 0.0, -0.04j, 0.0, 0.0],
            [0.03, 0.0, 0.0, 0.08, 0.0, 0.0],
            [0.0, 0.04j, 0.08, 0.0, -0.02j, 0.0],
            [0.0, 0.0, 0.0, 0.02j, 0.0, 0.05],
            [0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
        ],
        dtype=jnp.complex128,
    )
    hamiltonian = jnp.diag(diagonal).astype(jnp.complex128) + coupling
    return lambda values: jnp.einsum("dri,i->dr", raw, values), parameters, hamiltonian


def test_grassmann_sr_lowers_a_two_plane_energy() -> None:
    hamiltonian = jnp.diag(jnp.asarray([-3.0, -2.0, 1.0, 4.0]))

    def states(parameters):
        first = jnp.asarray([jnp.cos(parameters[0]), 0.0, jnp.sin(parameters[0]), 0.0])
        second = jnp.asarray([0.0, jnp.cos(parameters[1]), 0.0, jnp.sin(parameters[1])])
        return jnp.stack((first, second), axis=1)

    def loss(parameters):
        value = states(parameters)
        return jnp.real(jnp.trace(value.conj().T @ hamiltonian @ value))

    parameters = jnp.asarray([0.4, -0.3])
    result = exact_grassmann_sr_direction(
        parameters,
        state_fn=states,
        loss_fn=loss,
        damping=1.0e-3,
        cg_tolerance=1.0e-10,
        cg_max_iterations=20,
    )
    assert np.all(np.isfinite(np.asarray(result.direction)))
    assert float(loss(parameters + 0.05 * result.direction)) < float(loss(parameters))
    assert float(result.relative_linear_residual) < 1.0e-8


def test_grassmann_metric_projects_out_basis_gauge_motion() -> None:
    # theta[0] changes the represented line, theta[1] only rescales it.
    def states(parameters):
        direction = jnp.asarray([jnp.cos(parameters[0]), jnp.sin(parameters[0])])
        return (jnp.exp(parameters[1]) * direction)[:, None]

    target = jnp.asarray([1.0, 0.0])

    def loss(parameters):
        state = states(parameters)[:, 0]
        state = state / jnp.linalg.norm(state)
        return 1.0 - jnp.abs(jnp.vdot(target, state)) ** 2

    parameters = jnp.asarray([0.2, 0.7])
    result = exact_grassmann_sr_direction(
        parameters,
        state_fn=states,
        loss_fn=loss,
        damping=0.1,
        cg_tolerance=1.0e-10,
        cg_max_iterations=20,
    )
    # The scale coordinate is a vertical GL(1) gauge direction and has zero
    # loss gradient as well as zero projected tangent metric.
    assert abs(float(result.direction[1])) < 1.0e-12


def test_grassmann_sr_fails_closed_for_rank_deficient_reference() -> None:
    def states(parameters):
        first = jnp.asarray([parameters[0], 1.0, 0.0])
        return jnp.stack((first, first), axis=1)

    loss = lambda parameters: jnp.sum(parameters**2)
    result = exact_grassmann_sr_direction(
        jnp.asarray([0.2]), state_fn=states, loss_fn=loss
    )
    assert np.all(np.isnan(np.asarray(result.direction)))


def test_exact_energy_sr_matches_complex_ky_fan_gradient_and_finite_difference() -> None:
    states, parameters, hamiltonian = _complex_rank_two_problem()
    result = exact_grassmann_energy_sr_direction(
        parameters,
        state_fn=states,
        hamiltonian=hamiltonian,
        damping=1.0e-3,
    )
    reference_gradient = jax.grad(
        lambda values: _mean_ky_fan(states(values), hamiltonian)
    )(parameters)
    np.testing.assert_allclose(result.gradient, reference_gradient, rtol=2e-10, atol=2e-10)

    direction = jnp.asarray([0.2, -0.1, 0.3, -0.4, 0.15, 0.25, -0.2, 0.1])
    epsilon = 1.0e-6
    finite_difference = (
        _mean_ky_fan(states(parameters + epsilon * direction), hamiltonian)
        - _mean_ky_fan(states(parameters - epsilon * direction), hamiltonian)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        finite_difference,
        jnp.vdot(result.gradient, direction).real,
        rtol=2e-7,
        atol=2e-9,
    )
    assert float(result.relative_linear_residual) < 1.0e-9
    assert float(result.directional_derivative) < 0.0
    assert float(result.horizontality_error) < 1.0e-10


def test_exact_energy_sr_is_invariant_to_complex_head_mixing_shift_and_permutation() -> None:
    states, parameters, hamiltonian = _complex_rank_two_problem()
    mixing = jnp.asarray(
        [[1.2 + 0.1j, -0.3j], [0.25 - 0.05j, 0.8]], dtype=jnp.complex128
    )
    permutation = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    reference = exact_grassmann_energy_sr_direction(
        parameters, state_fn=states, hamiltonian=hamiltonian
    )
    mixed = exact_grassmann_energy_sr_direction(
        parameters,
        state_fn=lambda values: states(values) @ mixing,
        hamiltonian=hamiltonian,
    )
    permuted = exact_grassmann_energy_sr_direction(
        parameters,
        state_fn=lambda values: states(values) @ permutation,
        hamiltonian=hamiltonian,
    )
    shifted = exact_grassmann_energy_sr_direction(
        parameters,
        state_fn=states,
        hamiltonian=hamiltonian + 4.7 * jnp.eye(6),
    )
    for alternative in (mixed, permuted, shifted):
        np.testing.assert_allclose(alternative.direction, reference.direction, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(mixed.loss, reference.loss, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(permuted.loss, reference.loss, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(shifted.loss - reference.loss, 4.7, rtol=2e-12, atol=2e-12)


def test_canonical_metric_matches_projector_finite_difference() -> None:
    states, parameters, hamiltonian = _complex_rank_two_problem()
    result = exact_grassmann_energy_sr_direction(
        parameters, state_fn=states, hamiltonian=hamiltonian
    )

    def projector(values):
        vectors = states(values)
        return vectors @ jnp.linalg.solve(vectors.conj().T @ vectors, vectors.conj().T)

    epsilon = 2.0e-5
    difference = projector(parameters + epsilon * result.direction) - projector(parameters)
    projector_metric = jnp.linalg.norm(difference) ** 2 / (
        2.0 * epsilon**2 * states(parameters).shape[1]
    )
    np.testing.assert_allclose(
        projector_metric,
        result.canonical_direction_norm**2,
        rtol=3e-4,
        atol=3e-7,
    )


def test_exact_energy_sr_reports_zero_tangent_capacity_for_vertical_model() -> None:
    base = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=jnp.complex128,
    )

    def states(parameters):
        mixing = jnp.asarray(
            [[jnp.exp(parameters[0]), parameters[1]], [0.0, 1.0]],
            dtype=jnp.complex128,
        )
        return base @ mixing

    result = exact_grassmann_energy_sr_direction(
        jnp.asarray([0.2, -0.3]),
        state_fn=states,
        hamiltonian=jnp.diag(jnp.asarray([-3.0, -2.0, 1.0, 4.0])),
    )
    np.testing.assert_allclose(result.direction, 0.0, atol=1e-12)
    assert abs(float(result.canonical_direction_norm)) < 1.0e-12
    assert abs(float(result.damped_tangent_capture_fraction)) < 1.0e-12


def test_exact_energy_sr_fails_closed_for_duplicate_columns() -> None:
    def states(parameters):
        first = jnp.asarray([parameters[0], 1.0, 0.0], dtype=jnp.complex128)
        return jnp.stack((first, first), axis=1)

    result = exact_grassmann_energy_sr_direction(
        jnp.asarray([0.2]),
        state_fn=states,
        hamiltonian=jnp.diag(jnp.asarray([-2.0, -1.0, 3.0])),
    )
    assert np.all(np.isnan(np.asarray(result.direction)))


def test_exact_energy_sr_lowers_fusion_gru_mean_ky_fan_energy() -> None:
    n_sites = 6
    rank = 2
    config = FusionGRUConfig(n_sites, 0, embedding_dim=3, hidden_dim=5)
    parameters = tuple(
        init_fusion_gru(key, config)
        for key in jax.random.split(jax.random.PRNGKey(991), rank)
    )
    flat, unravel = ravel_pytree(parameters)
    paths = jnp.asarray(enumerate_paths(n_sites, 0), dtype=jnp.int32)
    hamiltonian = jnp.asarray(
        heisenberg_graph_fusion_matrix(
            n_sites,
            tuple((site, site + 1, 1.0) for site in range(n_sites - 1)),
            target_q=0,
        ).toarray()
    )

    def states(candidate):
        heads = unravel(candidate)
        return exhaustive_state_matrix(
            heads,
            paths,
            lambda values, path: jnp.stack(
                [log_psi(head, path, config) for head in values]
            ),
        )

    result = exact_grassmann_energy_sr_direction(
        flat, state_fn=states, hamiltonian=hamiltonian
    )
    nominal_step = 0.15 / n_sites
    trust_step = 0.05 / max(float(result.canonical_direction_norm), 1.0e-300)
    step = min(nominal_step, trust_step)
    assert float(
        _mean_ky_fan(states(flat + step * result.direction), hamiltonian)
    ) < float(_mean_ky_fan(states(flat), hamiltonian))


def test_exact_energy_sr_rejects_nonhermitian_hamiltonian_and_float32() -> None:
    states = lambda parameters: jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [parameters[0], 0.2j]],
        dtype=jnp.complex128,
    )
    nonhermitian = jnp.asarray(
        [[-2.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=jnp.complex128,
    )
    result = exact_grassmann_energy_sr_direction(
        jnp.asarray([0.1], dtype=jnp.float64),
        state_fn=states,
        hamiltonian=nonhermitian,
    )
    assert float(result.hamiltonian_hermiticity_error) == 1.0
    assert np.all(np.isnan(np.asarray(result.direction)))

    with np.testing.assert_raises(TypeError):
        exact_grassmann_energy_sr_direction(
            jnp.asarray([0.1], dtype=jnp.float32),
            state_fn=lambda parameters: states(parameters).astype(jnp.complex64),
            hamiltonian=jnp.eye(3, dtype=jnp.float32),
        )
