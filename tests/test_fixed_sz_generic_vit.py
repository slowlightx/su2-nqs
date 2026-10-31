from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.fixed_sz_generic_vit import (
    FixedSzGenericViTConfig,
    fixed_sz_generic_vit_log_amplitude,
    init_fixed_sz_generic_vit,
)
from su2_nqs.models.fixed_sz_generic_vit_adapter import (
    FixedSzGenericViTAdapterConfig,
    fixed_sz_generic_vit_adapter_amplitude,
    init_fixed_sz_generic_vit_adapter,
    seeded_fixed_sz_generic_vit_adapter_directions,
)


jax.config.update("jax_enable_x64", True)


def test_generic_vit_is_fixed_sz_but_not_translation_invariant() -> None:
    config = FixedSzGenericViTConfig(n_sites=16, embedding_dim=8, head_count=2)
    params = init_fixed_sz_generic_vit(jax.random.PRNGKey(8), config)
    state = jnp.asarray(([1, 1, 0, 0] * 4), dtype=jnp.int32)
    translated = jnp.roll(state.reshape(4, 4), 1, axis=1).reshape(-1)
    first = fixed_sz_generic_vit_log_amplitude(params, state, config)
    second = fixed_sz_generic_vit_log_amplitude(params, translated, config)
    assert abs(complex(np.asarray(first - second))) > 1.0e-8
    invalid = state.at[2].set(1)
    assert np.isneginf(np.real(np.asarray(fixed_sz_generic_vit_log_amplitude(params, invalid, config))))


def test_generic_vit_zero_adapter_recovers_frozen_amplitude() -> None:
    base = FixedSzGenericViTConfig(n_sites=16, embedding_dim=8, head_count=2)
    frozen = init_fixed_sz_generic_vit(jax.random.PRNGKey(10), base)
    config = FixedSzGenericViTAdapterConfig(
        base=base,
        directions=seeded_fixed_sz_generic_vit_adapter_directions(base, 6, seed=21),
        labels=tuple(f"d{index}" for index in range(6)),
    )
    adapter = init_fixed_sz_generic_vit_adapter(config)
    state = jnp.asarray(([1, 0] * 8), dtype=jnp.int32)
    actual = fixed_sz_generic_vit_adapter_amplitude(adapter, frozen, state, config)
    expected = jnp.exp(fixed_sz_generic_vit_log_amplitude(frozen, state, base))
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
