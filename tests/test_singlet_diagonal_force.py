from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.vmc.singlet_diagonal_force import (
    sampled_singlet_diagonal_value_and_grad,
    three_hz_local_energies,
)


jax.config.update("jax_enable_x64", True)


def test_three_hz_matches_hand_computed_spin_bits() -> None:
    samples = np.asarray(((0, 1), (1, 1), (1, 0)), dtype=np.int32)
    actual = three_hz_local_energies(samples, ((0, 1, 2.0),))
    np.testing.assert_allclose(actual, (-1.5, 1.5, -1.5), atol=0.0, rtol=0.0)


def test_diagonal_force_uses_only_two_center_batches_and_matches_surrogate() -> None:
    samples = jnp.asarray(
        (
            (0, 0, 1, 1),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
        ),
        dtype=jnp.int32,
    )
    features = 2.0 * samples.astype(jnp.float64) - 1.0
    params = {"weights": jnp.asarray((0.13, -0.07, 0.04, 0.09))}
    calls = {"count": 0}

    def raw_centers(candidate, batch):
        indices = jnp.asarray(batch, dtype=jnp.int32)
        signs = 2.0 * indices.astype(jnp.float64) - 1.0
        return jnp.exp(signs @ candidate["weights"])

    def counted_centers(candidate, batch):
        calls["count"] += 1
        return raw_centers(candidate, batch)

    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.5, periodic=False)
    batches = (samples[:2], samples[2:])
    result = sampled_singlet_diagonal_value_and_grad(
        params, batches, center_fn=counted_centers, bonds=bonds
    )
    assert calls["count"] == 2 * len(batches)

    local = three_hz_local_energies(samples, bonds)

    def reference_loss(candidate):
        centers = raw_centers(candidate, samples)
        mean = jnp.mean(local)
        surrogate = 2.0 * jnp.mean(jnp.log(centers) * (local - mean))
        return jax.lax.stop_gradient(mean) + surrogate - jax.lax.stop_gradient(surrogate)

    expected_energy, expected_gradient = jax.value_and_grad(reference_loss)(params)
    np.testing.assert_allclose(result.energy, expected_energy, atol=2.0e-14)
    np.testing.assert_allclose(
        result.gradient["weights"], expected_gradient["weights"], atol=2.0e-14
    )
    np.testing.assert_allclose(result.variance, np.var(np.asarray(local)), atol=2.0e-14)


def test_diagonal_force_rejects_bad_bonds_and_center_shape() -> None:
    samples = jnp.asarray(((0, 1), (1, 0)), dtype=jnp.int32)
    with pytest.raises(ValueError, match="distinct valid"):
        three_hz_local_energies(samples, ((0, 0, 1.0),))

    def bad_center(_params, batch):
        return jnp.ones((len(batch) + 1,), dtype=jnp.float64)

    with pytest.raises(ValueError, match="one amplitude"):
        sampled_singlet_diagonal_value_and_grad(
            {}, (samples,), center_fn=bad_center, bonds=((0, 1, 1.0),)
        )
