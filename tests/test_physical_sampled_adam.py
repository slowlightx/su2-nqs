from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.accelerator_sampled_force import sampled_vmc_log_value_and_grad
from su2_nqs.vmc.physical_sampled_adam import (
    MAX_PHYSICAL_MICROBATCH_SIZE,
    SampledAdamState,
    initialize_sampled_adam,
    sampled_adam_update,
    sampled_vmc_value_and_grad,
)
from su2_nqs.vmc.physical_vmc import physical_hpsi_vmc_loss_with_aux
from su2_nqs.vmc.resumable_checkpoint import (
    load_resumable_checkpoint,
    save_resumable_checkpoint,
)


jax.config.update("jax_enable_x64", True)


def _toy_problem():
    features = jnp.asarray(
        [
            [1.0, -0.4, 0.1],
            [1.0, -0.2, -0.3],
            [1.0, 0.0, 0.7],
            [1.0, 0.2, -0.5],
            [1.0, 0.4, 0.2],
            [1.0, 0.6, -0.1],
            [1.0, 0.8, 0.4],
            [1.0, 1.0, -0.6],
            [1.0, 1.2, 0.3],
            [1.0, 1.4, -0.2],
            [1.0, 1.6, 0.6],
            [1.0, 1.8, -0.4],
            [1.0, 2.0, 0.5],
            [1.0, 2.2, -0.7],
            [1.0, 2.4, 0.8],
            [1.0, 2.6, -0.9],
        ],
        dtype=jnp.float64,
    )
    diagonal = jnp.linspace(-1.2, 0.9, features.shape[0])
    hamiltonian = jnp.diag(diagonal)
    hopping = jnp.asarray(
        [0.13 if abs(row - column) == 1 else 0.0
         for row in range(features.shape[0])
         for column in range(features.shape[0])],
        dtype=jnp.float64,
    ).reshape((features.shape[0], features.shape[0]))
    hamiltonian = hamiltonian + hopping + hopping.T
    params = {
        "linear": jnp.asarray([0.17, -0.11, 0.08], dtype=jnp.float64),
        "scale": jnp.asarray(0.93, dtype=jnp.float64),
    }

    def hpsi(candidate, indices):
        # A positive nonlinear amplitude avoids accidental zero centers while
        # retaining a nested parameter pytree and nontrivial log derivatives.
        full = candidate["scale"] * jnp.exp(features @ candidate["linear"])
        action = hamiltonian @ full
        selected = jnp.asarray(indices, dtype=jnp.int32)
        return full[selected], action[selected]

    return params, hpsi


def _assert_pytrees_close(first: object, second: object) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    for actual, expected in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


@pytest.mark.parametrize(
    "split",
    (
        (8, 8),
        (3, 5, 8),
    ),
)
def test_global_centered_microbatches_match_full_batch(split: tuple[int, ...]) -> None:
    params, hpsi = _toy_problem()
    full_indices = jnp.arange(sum(split), dtype=jnp.int32)
    batches = []
    start = 0
    for size in split:
        batches.append(full_indices[start : start + size])
        start += size

    result = sampled_vmc_value_and_grad(params, batches, hpsi_fn=hpsi)

    def full_loss(candidate):
        return physical_hpsi_vmc_loss_with_aux(
            candidate,
            hpsi_fn=lambda values: hpsi(values, full_indices),
        )

    (_, auxiliary), expected_gradient = jax.value_and_grad(
        full_loss, has_aux=True
    )(params)
    np.testing.assert_allclose(result.energy, auxiliary["energy"], atol=2.0e-13)
    np.testing.assert_allclose(result.variance, auxiliary["variance"], atol=2.0e-13)
    np.testing.assert_allclose(
        result.energy_imaginary_abs,
        auxiliary["energy_imaginary_abs"],
        atol=2.0e-13,
    )
    _assert_pytrees_close(result.gradient, expected_gradient)
    assert result.sample_count == sum(split)
    assert result.microbatch_count == len(split)
    assert result.maximum_microbatch_size == max(split)
    assert result.maximum_microbatch_size <= MAX_PHYSICAL_MICROBATCH_SIZE
    assert result.minimum_center_magnitude > 0.0
    assert result.nonfinite_local_count == 0
    assert np.isfinite(result.gradient_norm)


def test_sampled_vmc_rejects_oversize_empty_and_nonfinite_batches() -> None:
    params, hpsi = _toy_problem()
    with pytest.raises(ValueError, match="at least one"):
        sampled_vmc_value_and_grad(params, (), hpsi_fn=hpsi)
    with pytest.raises(ValueError, match="limited to 8"):
        sampled_vmc_value_and_grad(
            params,
            (jnp.arange(9, dtype=jnp.int32),),
            hpsi_fn=hpsi,
        )


def test_frozen_local_log_force_matches_two_pass_hpsi_force() -> None:
    params, hpsi = _toy_problem()
    indices = jnp.arange(8, dtype=jnp.int32)
    centers, actions = hpsi(params, indices)
    one_action = sampled_vmc_log_value_and_grad(
        params,
        indices,
        actions / centers,
        log_amplitude_fn=lambda candidate, batch: jnp.log(
            hpsi(candidate, batch)[0].astype(jnp.complex128)
        ),
    )
    reference = sampled_vmc_value_and_grad(
        params, (indices,), hpsi_fn=hpsi
    )
    np.testing.assert_allclose(one_action.energy, reference.energy, atol=2.0e-13)
    np.testing.assert_allclose(
        one_action.variance, reference.variance, atol=2.0e-13
    )
    _assert_pytrees_close(one_action.gradient, reference.gradient)
    assert one_action.sample_count == len(indices)
    assert one_action.nonfinite_local_count == 0

    def zero_center(candidate, indices):
        del candidate
        size = len(indices)
        return jnp.zeros(size), jnp.ones(size)

    with pytest.raises(FloatingPointError, match="non-finite centers or local"):
        sampled_vmc_value_and_grad(
            params,
            (jnp.arange(2, dtype=jnp.int32),),
            hpsi_fn=zero_center,
        )


def test_sampled_adam_matches_first_step_formula_and_is_checkpointable(
    tmp_path: Path,
) -> None:
    params = {
        "a": jnp.asarray([1.2, -0.7], dtype=jnp.float64),
        "b": (jnp.asarray(0.4, dtype=jnp.float64),),
    }
    gradient = {
        "a": jnp.asarray([0.3, -0.2], dtype=jnp.float64),
        "b": (jnp.asarray(0.5, dtype=jnp.float64),),
    }
    state = initialize_sampled_adam(params)
    assert isinstance(state, SampledAdamState)
    assert state.step == 0
    assert state._fields == ("step", "first_moment", "second_moment")

    learning_rate = 2.0e-3
    candidate, updated_state, diagnostics = sampled_adam_update(
        params,
        gradient,
        state,
        learning_rate=learning_rate,
        max_relative_update=0.1,
    )
    expected = jax.tree_util.tree_map(
        lambda value, grad: value
        - learning_rate * grad / (jnp.abs(grad) + 1.0e-8),
        params,
        gradient,
    )
    _assert_pytrees_close(candidate, expected)
    assert updated_state.step == 1
    assert diagnostics.step == 1
    assert diagnostics.all_finite
    assert 0.0 < diagnostics.relative_update_norm < 0.1

    destination = tmp_path / "checkpoint.pkl"
    identity = {"task": "sampled_adam_unit", "learning_rate": learning_rate}
    save_resumable_checkpoint(
        destination,
        candidate,
        optimizer_state=updated_state,
        completed_steps=1,
        next_sample_index=16,
        run_identity=identity,
        history=[{"step": 1}],
    )
    restored = load_resumable_checkpoint(
        destination, expected_run_identity=identity
    )
    assert isinstance(restored["optimizer_state"], SampledAdamState)
    restored_state = restored["optimizer_state"]
    assert restored_state.step == updated_state.step
    _assert_pytrees_close(restored_state.first_moment, updated_state.first_moment)
    _assert_pytrees_close(restored_state.second_moment, updated_state.second_moment)

    continuous, continuous_state, _ = sampled_adam_update(
        candidate,
        gradient,
        updated_state,
        learning_rate=learning_rate,
        max_relative_update=0.1,
    )
    resumed, resumed_state, _ = sampled_adam_update(
        restored["params"],
        gradient,
        restored_state,
        learning_rate=learning_rate,
        max_relative_update=0.1,
    )
    _assert_pytrees_close(continuous, resumed)
    _assert_pytrees_close(continuous_state.first_moment, resumed_state.first_moment)
    _assert_pytrees_close(continuous_state.second_moment, resumed_state.second_moment)


def test_sampled_adam_fails_closed_on_relative_or_nonfinite_update() -> None:
    params = {"x": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}
    gradient = {"x": jnp.asarray([0.4, -0.1], dtype=jnp.float64)}
    state = initialize_sampled_adam(params)
    with pytest.raises(FloatingPointError, match="relative update exceeds"):
        sampled_adam_update(
            params,
            gradient,
            state,
            learning_rate=0.2,
            max_relative_update=1.0e-4,
        )
    assert state.step == 0
    np.testing.assert_array_equal(state.first_moment["x"], 0.0)
    np.testing.assert_array_equal(state.second_moment["x"], 0.0)

    nonfinite = {"x": jnp.asarray([jnp.nan, 0.1], dtype=jnp.float64)}
    with pytest.raises(FloatingPointError, match="input contains non-finite"):
        sampled_adam_update(
            params,
            nonfinite,
            state,
            learning_rate=1.0e-3,
            max_relative_update=0.1,
        )


def test_sampled_adam_rejects_complex_or_mismatched_pytrees() -> None:
    complex_params = {"x": jnp.asarray([1.0 + 0.2j])}
    with pytest.raises(TypeError, match="real floating-point"):
        initialize_sampled_adam(complex_params)

    params = {"x": jnp.asarray([1.0])}
    state = initialize_sampled_adam(params)
    with pytest.raises(ValueError, match="share a pytree"):
        sampled_adam_update(
            params,
            {"y": jnp.asarray([0.1])},
            state,
            learning_rate=1.0e-3,
            max_relative_update=0.1,
        )


def test_global_centered_microbatches_match_complex_phase_full_batch() -> None:
    """Exercise the complex-amplitude convention used by physical NQSs."""

    features = jnp.asarray(
        [
            [1.0, -0.6, 0.2],
            [1.0, -0.3, -0.4],
            [1.0, 0.0, 0.7],
            [1.0, 0.2, -0.5],
            [1.0, 0.5, 0.1],
            [1.0, 0.8, -0.2],
            [1.0, 1.1, 0.6],
        ],
        dtype=jnp.float64,
    )
    diagonal = jnp.linspace(-0.9, 0.8, len(features), dtype=jnp.float64)
    hamiltonian = jnp.diag(diagonal.astype(jnp.complex128))
    for index in range(len(features) - 1):
        hopping = 0.07 + 0.04j * (-1.0) ** index
        hamiltonian = hamiltonian.at[index, index + 1].set(hopping)
        hamiltonian = hamiltonian.at[index + 1, index].set(jnp.conj(hopping))
    params = {
        "log_magnitude": jnp.asarray([0.11, -0.08, 0.04]),
        "phase": jnp.asarray([-0.07, 0.13, 0.05]),
    }

    def hpsi(candidate, indices):
        log_amplitude = features @ candidate["log_magnitude"]
        phase = features @ candidate["phase"]
        full = jnp.exp(log_amplitude + 1.0j * phase)
        action = hamiltonian @ full
        selected = jnp.asarray(indices, dtype=jnp.int32)
        return full[selected], action[selected]

    full_indices = jnp.arange(len(features), dtype=jnp.int32)
    result = sampled_vmc_value_and_grad(
        params,
        (full_indices[:2], full_indices[2:]),
        hpsi_fn=hpsi,
    )

    def full_loss(candidate):
        return physical_hpsi_vmc_loss_with_aux(
            candidate,
            hpsi_fn=lambda values: hpsi(values, full_indices),
        )

    (_, auxiliary), expected_gradient = jax.value_and_grad(
        full_loss, has_aux=True
    )(params)
    np.testing.assert_allclose(result.energy, auxiliary["energy"], atol=3.0e-13)
    np.testing.assert_allclose(result.variance, auxiliary["variance"], atol=3.0e-13)
    np.testing.assert_allclose(
        result.energy_imaginary_abs,
        auxiliary["energy_imaginary_abs"],
        atol=3.0e-13,
    )
    _assert_pytrees_close(result.gradient, expected_gradient)
