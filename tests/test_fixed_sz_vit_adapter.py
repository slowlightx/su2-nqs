from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.fixed_sz_vit_adapter import (
    FixedSzViTAdapterConfig,
    fixed_sz_vit_adapter_log_amplitude,
    fixed_sz_vit_adapter_parameter_count,
    init_fixed_sz_vit_adapter,
    seeded_fixed_sz_vit_adapter_directions,
)


jax.config.update("jax_enable_x64", True)


def _base() -> FixedSzSpatialViTConfig:
    return FixedSzSpatialViTConfig(
        geometry="triangular",
        rows=4,
        columns=4,
        patch_rows=2,
        patch_columns=2,
        embedding_dim=8,
        head_count=2,
        depth=1,
        mlp_ratio=2,
        target_m2=0,
    )


def _configuration() -> jax.Array:
    return jnp.asarray(([1, 0] * 8), dtype=jnp.int32)


def test_zero_adapter_exactly_recovers_the_frozen_vit() -> None:
    base = _base()
    frozen = init_fixed_sz_spatial_vit(jax.random.PRNGKey(4), base)
    config = FixedSzViTAdapterConfig(
        base=base,
        directions=seeded_fixed_sz_vit_adapter_directions(8, 6, seed=11),
        labels=tuple(f"d{index}" for index in range(6)),
    )
    adapter = init_fixed_sz_vit_adapter(config)
    actual = fixed_sz_vit_adapter_log_amplitude(
        adapter, frozen, _configuration(), config
    )
    expected = fixed_sz_spatial_vit_log_amplitude(
        frozen, _configuration(), base
    )
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
    assert fixed_sz_vit_adapter_parameter_count(adapter) == 6


def test_every_seeded_adapter_direction_has_a_finite_nonzero_tangent() -> None:
    base = _base()
    frozen = init_fixed_sz_spatial_vit(jax.random.PRNGKey(7), base)
    config = FixedSzViTAdapterConfig(
        base=base,
        directions=seeded_fixed_sz_vit_adapter_directions(8, 6, seed=13),
        labels=tuple(f"d{index}" for index in range(6)),
    )
    adapter = init_fixed_sz_vit_adapter(config)

    def value(candidate):
        return fixed_sz_vit_adapter_log_amplitude(
            candidate, frozen, _configuration(), config
        )

    tangents = []
    for index in range(6):
        direction = {"coefficients": jnp.eye(6)[index]}
        _, tangent = jax.jvp(value, (adapter,), (direction,))
        tangents.append(complex(np.asarray(tangent)))
    assert np.all(np.isfinite(tangents))
    assert min(abs(value) for value in tangents) > 1.0e-9


def test_adapter_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        FixedSzViTAdapterConfig(
            base=_base(), directions=(((1.0,), (1.0,)),), labels=("bad",)
        )


def test_real_only_adapter_directions_do_not_add_a_phase() -> None:
    values = np.asarray(
        seeded_fixed_sz_vit_adapter_directions(
            8, 6, seed=31, real_only=True
        )
    )
    assert values.shape == (6, 2, 8)
    assert np.linalg.matrix_rank(values.reshape(6, -1)) == 6
    assert np.array_equal(values[:, 1], np.zeros((6, 8)))
