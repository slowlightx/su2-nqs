from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc import physical_sampled_adam as sealed_adam
from su2_nqs.vmc import physical_sampled_adam_center_only as center_only_adam
from su2_nqs.vmc.physical_vmc import physical_hpsi_vmc_loss_with_aux


jax.config.update("jax_enable_x64", True)


def _complex_problem():
    features = jnp.asarray(
        [
            [1.0, -0.8, 0.1, 0.3],
            [1.0, -0.6, -0.2, 0.5],
            [1.0, -0.4, 0.7, -0.1],
            [1.0, -0.2, -0.5, 0.4],
            [1.0, 0.0, 0.2, -0.6],
            [1.0, 0.2, -0.3, 0.8],
            [1.0, 0.4, 0.6, -0.7],
            [1.0, 0.6, -0.8, 0.2],
            [1.0, 0.8, 0.4, -0.5],
            [1.0, 1.0, -0.1, 0.7],
            [1.0, 1.2, 0.9, -0.4],
            [1.0, 1.4, -0.7, 0.1],
            [1.0, 1.6, 0.3, -0.8],
            [1.0, 1.8, -0.4, 0.6],
            [1.0, 2.0, 0.8, -0.2],
            [1.0, 2.2, -0.6, 0.9],
        ],
        dtype=jnp.float64,
    )
    diagonal = jnp.linspace(-1.1, 0.8, len(features), dtype=jnp.float64)
    hamiltonian = jnp.diag(diagonal.astype(jnp.complex128))
    for index in range(len(features) - 1):
        hopping = (0.09 + 0.013 * index) + 0.035j * (-1.0) ** index
        hamiltonian = hamiltonian.at[index, index + 1].set(hopping)
        hamiltonian = hamiltonian.at[index + 1, index].set(jnp.conj(hopping))
    params = {
        "log_magnitude": jnp.asarray([0.08, -0.06, 0.03, 0.02]),
        "phase": jnp.asarray([-0.05, 0.09, 0.04, -0.03]),
    }

    def full_amplitudes(candidate):
        log_magnitude = features @ candidate["log_magnitude"]
        phase = features @ candidate["phase"]
        return jnp.exp(log_magnitude + 1.0j * phase)

    def hpsi(candidate, indices):
        full = full_amplitudes(candidate)
        action = hamiltonian @ full
        selected = jnp.asarray(indices, dtype=jnp.int32)
        return full[selected], action[selected]

    def centers(candidate, indices):
        selected = jnp.asarray(indices, dtype=jnp.int32)
        return full_amplitudes(candidate)[selected]

    return params, hpsi, centers


def _batches(split: tuple[int, ...]) -> tuple[jax.Array, ...]:
    indices = jnp.arange(sum(split), dtype=jnp.int32)
    result = []
    start = 0
    for size in split:
        result.append(indices[start : start + size])
        start += size
    return tuple(result)


def _assert_pytrees_close(
    actual: object,
    expected: object,
    *,
    tolerance: float = 3.0e-12,
) -> None:
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        np.testing.assert_allclose(
            actual_leaf,
            expected_leaf,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("split", ((8, 8), (3, 5, 8)))
def test_center_only_matches_sealed_two_hpsi_and_full_batch_complex_gradient(
    split: tuple[int, ...],
) -> None:
    params, raw_hpsi, raw_centers = _complex_problem()
    batches = _batches(split)
    calls = {"hpsi": 0, "centers": 0, "sealed_hpsi": 0}

    def counted_hpsi(candidate, batch):
        calls["hpsi"] += 1
        return raw_hpsi(candidate, batch)

    def counted_centers(candidate, batch):
        calls["centers"] += 1
        return raw_centers(candidate, batch)

    def counted_sealed_hpsi(candidate, batch):
        calls["sealed_hpsi"] += 1
        return raw_hpsi(candidate, batch)

    optimized = center_only_adam.sampled_vmc_value_and_grad(
        params,
        batches,
        hpsi_fn=counted_hpsi,
        center_fn=counted_centers,
    )
    sealed = sealed_adam.sampled_vmc_value_and_grad(
        params,
        batches,
        hpsi_fn=counted_sealed_hpsi,
    )

    full_indices = jnp.arange(sum(split), dtype=jnp.int32)

    def full_loss(candidate):
        return physical_hpsi_vmc_loss_with_aux(
            candidate,
            hpsi_fn=lambda values: raw_hpsi(values, full_indices),
        )

    (_, auxiliary), full_gradient = jax.value_and_grad(
        full_loss, has_aux=True
    )(params)
    assert calls == {
        "hpsi": len(split),
        "centers": len(split),
        "sealed_hpsi": 2 * len(split),
    }
    np.testing.assert_allclose(optimized.energy, sealed.energy, atol=2.0e-13)
    np.testing.assert_allclose(optimized.variance, sealed.variance, atol=2.0e-13)
    np.testing.assert_allclose(
        optimized.energy_imaginary_abs,
        sealed.energy_imaginary_abs,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(optimized.energy, auxiliary["energy"], atol=2.0e-13)
    np.testing.assert_allclose(
        optimized.variance, auxiliary["variance"], atol=2.0e-13
    )
    _assert_pytrees_close(optimized.gradient, sealed.gradient)
    _assert_pytrees_close(optimized.gradient, full_gradient)
    assert optimized.sample_count == sum(split)
    assert optimized.microbatch_count == len(split)
    assert optimized.maximum_microbatch_size == max(split)


def test_center_only_fails_closed_if_center_kernel_disagrees_or_has_wrong_shape() -> None:
    params, hpsi, centers = _complex_problem()
    batches = _batches((3, 5, 8))

    def disagreeing(candidate, batch):
        return centers(candidate, batch) * (1.0 + 1.0e-7)

    with pytest.raises(ValueError, match="center_fn disagrees"):
        center_only_adam.sampled_vmc_value_and_grad(
            params,
            batches,
            hpsi_fn=hpsi,
            center_fn=disagreeing,
        )

    def wrong_shape(candidate, batch):
        return centers(candidate, batch)[:-1]

    with pytest.raises(ValueError, match="matching hpsi_fn"):
        center_only_adam.sampled_vmc_value_and_grad(
            params,
            batches,
            hpsi_fn=hpsi,
            center_fn=wrong_shape,
        )


def test_center_only_reuses_exact_checkpoint_state_and_adam_update_types() -> None:
    assert center_only_adam.SampledAdamState is sealed_adam.SampledAdamState
    assert center_only_adam.initialize_sampled_adam is sealed_adam.initialize_sampled_adam
    assert center_only_adam.sampled_adam_update is sealed_adam.sampled_adam_update

    params = {"x": jnp.asarray([0.7, -0.2], dtype=jnp.float64)}
    state = center_only_adam.initialize_sampled_adam(params)
    assert isinstance(state, sealed_adam.SampledAdamState)
    assert state._fields == ("step", "first_moment", "second_moment")


def test_center_only_projected_amplitude_work_proxy_for_sealed_6x6_shapes() -> None:
    """Document the ledger replacement without timing or efficiency claims."""

    samples = 640
    for padded_connected_width, sealed_total in (
        (145, 186_880),
        (217, 279_040),
    ):
        sealed_two_hpsi = 2 * samples * (1 + padded_connected_width)
        center_only = samples * (2 + padded_connected_width)
        assert sealed_two_hpsi == sealed_total
        assert center_only < sealed_two_hpsi
        # The removed work is exactly one connected-state amplitude block;
        # the second-pass sampled centers remain in the ledger.
        assert sealed_two_hpsi - center_only == samples * padded_connected_width
