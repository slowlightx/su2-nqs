from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.physical_trust_region import (
    bound_physical_state_update,
    sampled_physical_state_distance,
)


def test_sampled_physical_distance_is_zero_for_global_rescaling_and_phase() -> None:
    samples = jnp.arange(6, dtype=jnp.int32)[:, None]
    features = jnp.asarray([-1.5, -0.8, -0.2, 0.3, 0.9, 1.4])

    def log_amplitude(params, batch):
        x = features[batch[:, 0]]
        return params[0] * x + params[1] + 1j * params[2]

    old = jnp.asarray([0.4, -0.7, 0.2])
    gauge_changed = jnp.asarray([0.4, 2.1, -1.3])
    distance = sampled_physical_state_distance(
        old,
        gauge_changed,
        samples,
        log_amplitude_fn=log_amplitude,
    )
    assert distance.fubini_study_distance < 3.0e-8
    assert distance.symmetric_kl < 2.0e-15


def test_nonlinear_fs_kl_bound_shortens_one_common_direction() -> None:
    samples = jnp.arange(8, dtype=jnp.int32)[:, None]
    features = jnp.linspace(-1.5, 1.5, 8)

    def log_amplitude(params, batch):
        x = features[batch[:, 0]]
        return (params[0] * x + params[1] * x**2).astype(jnp.complex128)

    params = jnp.asarray([0.1, -0.2])
    direction = jnp.asarray([2.0, -1.0])
    result = bound_physical_state_update(
        params,
        direction,
        samples,
        log_amplitude_fn=log_amplitude,
        maximum_fubini_study_distance=0.04,
        maximum_symmetric_kl=0.004,
    )
    assert 0.0 < result.scale < 1.0
    assert result.distance.fubini_study_distance <= 0.04 * (1.0 + 1.0e-9)
    assert result.distance.symmetric_kl <= 0.004 * (1.0 + 1.0e-9)
    np.testing.assert_allclose(np.asarray(result.direction), result.scale * direction)


def test_center_and_log_callbacks_give_the_same_distance() -> None:
    samples = jnp.arange(5, dtype=jnp.int32)[:, None]
    features = jnp.asarray([-1.0, -0.4, 0.1, 0.7, 1.2])

    def logs(params, batch):
        x = features[batch[:, 0]]
        return params[0] * x + 1j * params[1] * x

    def centers(params, batch):
        return jnp.exp(logs(params, batch))

    old = jnp.asarray([0.2, -0.1])
    new = jnp.asarray([0.5, 0.3])
    from_log = sampled_physical_state_distance(
        old, new, samples, log_amplitude_fn=logs
    )
    from_center = sampled_physical_state_distance(
        old, new, samples, center_fn=centers
    )
    np.testing.assert_allclose(
        list(from_log.__dict__.values()),
        list(from_center.__dict__.values()),
        rtol=2.0e-7,
        atol=2.0e-7,
    )
