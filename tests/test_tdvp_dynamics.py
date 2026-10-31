import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.dynamics import exact_time_evolution, state_fidelities
from su2_nqs.vmc.dynamics import (
    evolve_exact_tdvp,
    exact_tdvp_direction,
    normalized_wavefunction,
    sampled_tdvp_direction,
)


def _complete_two_state_log_psi(params, state):
    index = state[0]
    log_probability = jax.nn.log_softmax(params["logits"])[index]
    return 0.5 * log_probability + 1j * params["phases"][index]


def test_complete_two_state_tdvp_tangent_matches_schrodinger_direction():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    hamiltonian = jnp.asarray([[0.3, -0.7], [-0.7, 1.1]], dtype=jnp.float64)
    params = {
        "logits": jnp.asarray([0.2, -0.4], dtype=jnp.float64),
        "phases": jnp.asarray([0.1, -0.2], dtype=jnp.float64),
    }
    direction, diagnostics = exact_tdvp_direction(
        params,
        basis,
        hamiltonian,
        _complete_two_state_log_psi,
        damping=0.0,
    )
    epsilon = 1.0e-6
    moved = jax.tree.map(lambda x, dx: x + epsilon * dx, params, direction)
    first = np.asarray(normalized_wavefunction(params, basis, _complete_two_state_log_psi))
    second = np.asarray(normalized_wavefunction(moved, basis, _complete_two_state_log_psi))
    numerical = (second - first) / epsilon
    numerical -= np.vdot(first, numerical) * first
    exact = -1j * np.asarray(hamiltonian) @ first
    exact -= np.vdot(first, exact) * first
    np.testing.assert_allclose(numerical, exact, atol=2e-6)
    assert diagnostics.residual_norm < 1e-12
    assert diagnostics.tangent_rank == 2


def test_rk4_exact_support_tdvp_reproduces_two_state_unitary_dynamics():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    hamiltonian = jnp.asarray([[0.3, -0.7], [-0.7, 1.1]], dtype=jnp.float64)
    params = {
        "logits": jnp.asarray([0.2, -0.4], dtype=jnp.float64),
        "phases": jnp.asarray([0.1, -0.2], dtype=jnp.float64),
    }
    initial = np.asarray(normalized_wavefunction(params, basis, _complete_two_state_log_psi))
    time_step = 0.02
    steps = 50
    _, tdvp, diagnostics = evolve_exact_tdvp(
        params,
        basis,
        hamiltonian,
        _complete_two_state_log_psi,
        time_step=time_step,
        steps=steps,
        damping=0.0,
    )
    exact = exact_time_evolution(
        np.asarray(hamiltonian), initial, np.arange(steps + 1) * time_step
    )
    assert np.max(1.0 - state_fidelities(exact, tdvp)) < 2e-12
    assert max(row.residual_norm for row in diagnostics) < 2e-12


def test_implicit_midpoint_tdvp_converges_for_complete_two_state_model():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    hamiltonian = jnp.asarray([[0.3, -0.7], [-0.7, 1.1]], dtype=jnp.float64)
    params = {
        "logits": jnp.asarray([0.2, -0.4], dtype=jnp.float64),
        "phases": jnp.asarray([0.1, -0.2], dtype=jnp.float64),
    }
    initial = np.asarray(normalized_wavefunction(params, basis, _complete_two_state_log_psi))
    time_step = 0.02
    steps = 50
    _, tdvp, diagnostics = evolve_exact_tdvp(
        params,
        basis,
        hamiltonian,
        _complete_two_state_log_psi,
        time_step=time_step,
        steps=steps,
        damping=0.0,
        integrator="implicit_midpoint",
        implicit_maximum_iterations=12,
        implicit_tolerance=1e-11,
    )
    exact = exact_time_evolution(
        np.asarray(hamiltonian), initial, np.arange(steps + 1) * time_step
    )
    assert np.max(1.0 - state_fidelities(exact, tdvp)) < 2e-8
    assert max(row.integrator_residual for row in diagnostics) < 1e-10
    assert max(row.integrator_iterations for row in diagnostics) <= 12


def test_sampled_tdvp_matches_exact_direction_for_exact_empirical_weights():
    basis = jnp.asarray([[0], [1]], dtype=jnp.int32)
    hamiltonian = np.asarray([[0.3, -0.7], [-0.7, 1.1]], dtype=np.float64)
    params = {
        "logits": jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        "phases": jnp.asarray([0.1, -0.2], dtype=jnp.float64),
    }

    def get_conn(state):
        column = int(state[0])
        return ((0,), (1,)), hamiltonian[:, column]

    exact_direction, _ = exact_tdvp_direction(
        params,
        basis,
        jnp.asarray(hamiltonian),
        _complete_two_state_log_psi,
        damping=1e-6,
    )
    sampled_direction, diagnostics = sampled_tdvp_direction(
        params,
        jnp.asarray([[0], [1], [0], [1]]),
        _complete_two_state_log_psi,
        get_conn,
        damping=1e-6,
    )
    for exact, sampled in zip(
        jax.tree.leaves(exact_direction), jax.tree.leaves(sampled_direction)
    ):
        np.testing.assert_allclose(sampled, exact, atol=2e-12)
    assert diagnostics.sample_count == 4
    assert diagnostics.unique_sample_count == 2

    importance_direction, importance_diagnostics = sampled_tdvp_direction(
        params,
        jnp.asarray([[0], [0], [0], [1]]),
        _complete_two_state_log_psi,
        get_conn,
        damping=1e-6,
        importance_weights=jnp.asarray([2 / 3, 2 / 3, 2 / 3, 2.0]),
    )
    for exact, sampled in zip(
        jax.tree.leaves(exact_direction), jax.tree.leaves(importance_direction)
    ):
        np.testing.assert_allclose(sampled, exact, atol=2e-12)
    assert importance_diagnostics.unique_sample_count == 2
