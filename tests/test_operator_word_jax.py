from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.operator_tangent_response import triangular_a1_word_actions
from su2_nqs.benchmarks.operator_word_jax import (
    build_triangular_a1_word_action_plan,
    evaluate_fixed_capacity_scalar_word_kernel,
    evaluate_scalar_word_action_plan_jax,
    evaluate_scalar_word_action_plan_numpy,
    make_fixed_capacity_scalar_word_kernel,
)
from su2_nqs.benchmarks.qsl_three_arm import triangular_pi_flux_operator_components
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import SMALL_CASES
from su2_nqs.operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS


jax.config.update("jax_enable_x64", True)


def test_batched_jax_scalar_word_plan_matches_reference_through_depth_three() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    states = data.configurations[[321, 4321]]
    weights_real = np.linspace(0.2, 1.1, states.shape[1])
    weights_imag = np.linspace(-0.3, 0.4, states.shape[1])

    def numpy_amplitude(configurations: np.ndarray) -> np.ndarray:
        values = np.asarray(configurations, dtype=np.float64)
        return values @ weights_real + 1j * (values @ weights_imag + 0.7)

    def jax_amplitude(configurations: jax.Array) -> jax.Array:
        values = jnp.asarray(configurations, dtype=jnp.float64)
        return values @ jnp.asarray(weights_real) + 1j * (
            values @ jnp.asarray(weights_imag) + 0.7
        )

    words = (
        (),
        (TRIANGULAR_A1_SCALAR_LABELS[0],),
        (TRIANGULAR_A1_SCALAR_LABELS[2],),
        (
            TRIANGULAR_A1_SCALAR_LABELS[0],
            TRIANGULAR_A1_SCALAR_LABELS[2],
        ),
        (
            TRIANGULAR_A1_SCALAR_LABELS[0],
            TRIANGULAR_A1_SCALAR_LABELS[2],
            TRIANGULAR_A1_SCALAR_LABELS[1],
        ),
    )
    reference, _ = triangular_a1_word_actions(
        numpy_amplitude,
        states,
        SMALL_CASES["square4"].supercell,
        words,
        amplitude_batch_size=31,
    )
    plan, ledger = build_triangular_a1_word_action_plan(
        states, SMALL_CASES["square4"].supercell, words
    )
    numpy_actions = evaluate_scalar_word_action_plan_numpy(
        plan, numpy_amplitude, amplitude_batch_size=29
    )
    jax_actions, jax_ledger = evaluate_scalar_word_action_plan_jax(
        plan, jax_amplitude, amplitude_batch_size=32
    )

    np.testing.assert_allclose(numpy_actions, reference, atol=3.0e-11)
    np.testing.assert_allclose(jax_actions, reference, atol=3.0e-11)
    assert ledger["sample_count"] == 2
    assert ledger["maximum_word_depth"] == 3
    assert ledger["unique_terminal_amplitude_evaluations"] < ledger[
        "raw_terminal_configuration_requests"
    ]
    assert jax_ledger["unique_terminal_count"] == len(
        plan.terminal_configurations
    )
    assert jax_ledger["amplitude_batch_count"] >= 1


def test_fixed_capacity_kernel_is_reused_across_distinct_plans() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    first_states = data.configurations[[100, 200]]
    second_states = data.configurations[[300, 400]]
    weights = jnp.linspace(-0.7, 0.9, first_states.shape[1])

    def amplitude(configurations: jax.Array) -> jax.Array:
        values = jnp.asarray(configurations, dtype=jnp.float64)
        return values @ weights + 0.25j

    words = ((), ("J1_shell",), ("J2_shell", "J1_shell"))
    first, _ = build_triangular_a1_word_action_plan(
        first_states, SMALL_CASES["square4"].supercell, words
    )
    second, _ = build_triangular_a1_word_action_plan(
        second_states, SMALL_CASES["square4"].supercell, words
    )
    terminal_capacity = 32 * max(
        (len(first.terminal_configurations) + 31) // 32,
        (len(second.terminal_configurations) + 31) // 32,
    )
    entry_capacity = max(len(first.entry_coefficients), len(second.entry_coefficients))
    kernel = make_fixed_capacity_scalar_word_kernel(
        amplitude,
        n_sites=first_states.shape[1],
        amplitude_batch_size=32,
        terminal_capacity=terminal_capacity,
        entry_capacity=entry_capacity,
        output_count=first.sample_count * first.word_count,
    )
    for plan in (first, second):
        observed, ledger = evaluate_fixed_capacity_scalar_word_kernel(kernel, plan)
        expected = evaluate_scalar_word_action_plan_numpy(
            plan, lambda x: np.asarray(amplitude(jnp.asarray(x))), amplitude_batch_size=19
        )
        np.testing.assert_allclose(observed, expected, atol=3.0e-11)
        assert ledger["terminal_capacity"] == terminal_capacity
        assert ledger["entry_capacity"] == entry_capacity
