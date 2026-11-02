"""Matched low-dimensional adapter for the conventional generic ViT."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_sz_generic_vit import (
    FixedSzGenericViTConfig,
    fixed_sz_generic_vit_log_amplitude,
)


@dataclass(frozen=True)
class FixedSzGenericViTAdapterConfig:
    base: FixedSzGenericViTConfig
    directions: tuple[tuple[tuple[float, ...], ...], ...]
    labels: tuple[str, ...]
    adapter_scale: float = 0.15

    def __post_init__(self) -> None:
        values = np.asarray(self.directions, dtype=np.float64)
        expected = (
            len(self.labels),
            self.base.n_sites + 1,
            self.base.embedding_dim,
        )
        if not self.labels or values.shape != expected:
            raise ValueError(f"adapter directions must have shape {expected}")
        if np.any(~np.isfinite(values)) or np.linalg.matrix_rank(
            values.reshape(len(values), -1)
        ) != len(values):
            raise ValueError("adapter directions must be finite and independent")
        if not np.isfinite(self.adapter_scale) or self.adapter_scale <= 0.0:
            raise ValueError("adapter_scale must be positive and finite")

    @property
    def n_sites(self) -> int:
        return self.base.n_sites

    @property
    def adapter_count(self) -> int:
        return len(self.labels)


def init_fixed_sz_generic_vit_adapter(
    config: FixedSzGenericViTAdapterConfig,
) -> dict[str, jax.Array]:
    return {"coefficients": jnp.zeros((config.adapter_count,), dtype=jnp.float64)}


def seeded_fixed_sz_generic_vit_adapter_directions(
    config: FixedSzGenericViTConfig,
    count: int,
    *,
    seed: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Return orthonormal absolute-position adapter directions."""

    size = (config.n_sites + 1) * config.embedding_dim
    if count < 1 or count > size:
        raise ValueError("generic adapter count is outside the position space")
    matrix = np.random.default_rng(int(seed)).normal(size=(size, int(count)))
    q, _ = np.linalg.qr(matrix)
    for column in range(count):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0.0:
            q[:, column] *= -1.0
    values = np.moveaxis(
        q[:, :count].reshape(config.n_sites + 1, config.embedding_dim, count),
        2,
        0,
    )
    return tuple(
        tuple(tuple(map(float, row)) for row in item) for item in values
    )


def _adapted_params(adapter_params, frozen_params, config):
    coefficients = jnp.asarray(adapter_params["coefficients"], dtype=jnp.float64)
    if coefficients.shape != (config.adapter_count,):
        raise ValueError("adapter coefficient vector has the wrong shape")
    delta = jnp.einsum(
        "k,kse->se",
        float(config.adapter_scale) * jnp.tanh(coefficients),
        jnp.asarray(config.directions, dtype=jnp.float64),
    )
    return {
        **frozen_params,
        "position_embedding": frozen_params["position_embedding"] + delta,
    }


def fixed_sz_generic_vit_adapter_amplitude(
    adapter_params,
    frozen_params,
    configuration,
    config: FixedSzGenericViTAdapterConfig,
):
    params = _adapted_params(adapter_params, frozen_params, config)
    value = fixed_sz_generic_vit_log_amplitude(params, configuration, config.base)
    return jnp.where(jnp.isfinite(jnp.real(value)), jnp.exp(value), 0.0 + 0.0j)


__all__ = [
    "FixedSzGenericViTAdapterConfig",
    "fixed_sz_generic_vit_adapter_amplitude",
    "init_fixed_sz_generic_vit_adapter",
    "seeded_fixed_sz_generic_vit_adapter_directions",
]
