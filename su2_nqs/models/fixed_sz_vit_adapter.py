"""Low-dimensional trainable adapters for frozen fixed-Sz spatial ViTs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FixedSzViTAdapterConfig:
    """Static frozen-ViT and log-cosh-bias adapter contract."""

    base: FixedSzSpatialViTConfig
    directions: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    labels: tuple[str, ...]
    adapter_scale: float = 0.15
    log_amplitude_shift: float = 0.0

    def __post_init__(self) -> None:
        if not self.directions:
            raise ValueError("at least one adapter direction is required")
        if len(self.labels) != len(self.directions):
            raise ValueError("one label is required per adapter direction")
        values = np.asarray(self.directions, dtype=np.float64)
        expected = (len(self.directions), 2, self.base.embedding_dim)
        if values.shape != expected:
            raise ValueError(f"adapter directions must have shape {expected}")
        if np.any(~np.isfinite(values)):
            raise ValueError("adapter directions must be finite")
        if np.linalg.matrix_rank(values.reshape(len(values), -1)) != len(values):
            raise ValueError("adapter directions must be linearly independent")
        if not np.isfinite(self.adapter_scale) or self.adapter_scale <= 0.0:
            raise ValueError("adapter_scale must be positive and finite")
        if not np.isfinite(self.log_amplitude_shift):
            raise ValueError("log_amplitude_shift must be finite")

    @property
    def n_sites(self) -> int:
        return self.base.n_sites

    @property
    def adapter_count(self) -> int:
        return len(self.directions)


def seeded_fixed_sz_vit_adapter_directions(
    embedding_dim: int,
    count: int,
    *,
    seed: int,
    real_only: bool = False,
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return deterministic orthonormal log-cosh bias directions.

    ``real_only`` is useful for real-Hamiltonian ground-state controls: the
    shared SU(2) base then supplies the sign structure while the ViT acts as a
    positive real correlator.  This avoids accidental phase singularities in
    a deliberately low-work comparison.
    """

    available = embedding_dim if real_only else 2 * embedding_dim
    if embedding_dim < 1 or count < 1 or count > available:
        raise ValueError("adapter count exceeds the available bias space")
    matrix = np.random.default_rng(int(seed)).normal(
        size=(int(available), int(count))
    )
    q, _ = np.linalg.qr(matrix)
    for column in range(count):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0.0:
            q[:, column] *= -1.0
    if real_only:
        real = np.moveaxis(q[:, :count].reshape(embedding_dim, count), 1, 0)
        values = np.stack((real, np.zeros_like(real)), axis=1)
    else:
        values = np.moveaxis(
            q[:, :count].reshape(2, embedding_dim, count), 2, 0
        )
    return tuple(
        (tuple(map(float, item[0])), tuple(map(float, item[1])))
        for item in values
    )


def init_fixed_sz_vit_adapter(
    config: FixedSzViTAdapterConfig,
) -> dict[str, jax.Array]:
    """Initialize at the exactly shared frozen ViT state."""

    return {
        "coefficients": jnp.zeros((config.adapter_count,), dtype=jnp.float64)
    }


def _adapted_params(
    adapter_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    config: FixedSzViTAdapterConfig,
) -> dict[str, object]:
    coefficients = jnp.asarray(
        adapter_params["coefficients"], dtype=jnp.float64
    )
    if coefficients.shape != (config.adapter_count,):
        raise ValueError("adapter coefficient vector has the wrong shape")
    directions = jnp.asarray(config.directions, dtype=jnp.float64)
    controls = float(config.adapter_scale) * jnp.tanh(coefficients)
    delta = jnp.einsum("k,kce->ce", controls, directions)
    head = frozen_params["log_cosh_head"]
    adapted_head = {
        **head,
        "bias_real": head["bias_real"] + delta[0],
        "bias_imag": head["bias_imag"] + delta[1],
    }
    return {**frozen_params, "log_cosh_head": adapted_head}


def fixed_sz_vit_adapter_log_amplitude(
    adapter_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzViTAdapterConfig,
) -> jax.Array:
    """Evaluate one frozen-backbone adapter log amplitude."""

    params = _adapted_params(adapter_params, frozen_params, config)
    return fixed_sz_spatial_vit_log_amplitude(params, configuration, config.base)


def fixed_sz_vit_adapter_amplitude(
    adapter_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    configuration: jax.Array,
    config: FixedSzViTAdapterConfig,
) -> jax.Array:
    """Exponentiate one amplitude with a fixed, ratio-neutral shift."""

    log_value = fixed_sz_vit_adapter_log_amplitude(
        adapter_params, frozen_params, configuration, config
    )
    finite = jnp.isfinite(jnp.real(log_value))
    value = jnp.exp(log_value - float(config.log_amplitude_shift))
    return jnp.where(finite, value, 0.0 + 0.0j)


def fixed_sz_vit_adapter_parameter_count(
    adapter_params: dict[str, jax.Array],
) -> int:
    """Count trainable adapter scalars, excluding the frozen ViT."""

    return int(
        sum(
            np.prod(leaf.shape)
            for leaf in jax.tree_util.tree_leaves(adapter_params)
        )
    )


__all__ = [
    "FixedSzViTAdapterConfig",
    "fixed_sz_vit_adapter_amplitude",
    "fixed_sz_vit_adapter_log_amplitude",
    "fixed_sz_vit_adapter_parameter_count",
    "init_fixed_sz_vit_adapter",
    "seeded_fixed_sz_vit_adapter_directions",
]
