"""Zero-extra-parameter coherent scalar ensemble over physical site orders.

Every cut feeds a differently ordered physical configuration through the same
tree and the same reduced tensors.  This is an inexpensive diagnostic for
complementary hierarchical locality, not a multi-cut fusion network: the
different trees exchange no reduced boundary information before the final
scalar sum.  Orders related only by a projected lattice symmetry are redundant
and should not be registered together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import SitePermutation, validate_site_permutation


class PhysicalAmplitude(Protocol):
    def __call__(self, params: object, state: jax.Array, config: object) -> jax.Array:
        ...


@dataclass(frozen=True)
class SharedCutConfig:
    """Static site orders sharing one physical-tree parameter pytree.

    The historical ``SharedCut`` name is retained for API stability, but this
    object represents a scalar-level shared-parameter ensemble only.
    """

    model_config: object
    site_orders: tuple[SitePermutation, ...]
    coefficients: tuple[complex, ...] | None = None

    def __post_init__(self) -> None:
        n_sites = getattr(self.model_config, "n_sites", None)
        if n_sites is None:
            raise TypeError("model_config must expose n_sites")
        orders = tuple(
            validate_site_permutation(order, n_sites=int(n_sites))
            for order in self.site_orders
        )
        if not orders or len(set(orders)) != len(orders):
            raise ValueError("site_orders must be nonempty and duplicate free")
        object.__setattr__(self, "site_orders", orders)
        if self.coefficients is None:
            coefficients = (1.0 + 0.0j,) * len(orders)
        else:
            coefficients = tuple(complex(value) for value in self.coefficients)
            if len(coefficients) != len(orders):
                raise ValueError("one coefficient is required per cut")
            if any(not np.isfinite(value.real) or not np.isfinite(value.imag) for value in coefficients):
                raise ValueError("cut coefficients must be finite")
            if not any(abs(value) > 0.0 for value in coefficients):
                raise ValueError("at least one cut coefficient must be nonzero")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_sites(self) -> int:
        return int(getattr(self.model_config, "n_sites"))

    @property
    def cut_count(self) -> int:
        return len(self.site_orders)


def shared_cut_amplitude(
    params: object,
    state: jax.Array,
    config: SharedCutConfig,
    *,
    amplitude_fn: PhysicalAmplitude,
) -> jax.Array:
    """Evaluate one coherent shared-parameter scalar ensemble."""

    array = jnp.asarray(state, dtype=jnp.int32)
    if array.shape != (config.n_sites,):
        raise ValueError(f"expected state shape ({config.n_sites},)")
    orders = jnp.asarray(config.site_orders, dtype=jnp.int32)
    cut_states = jnp.take(array, orders, axis=-1)
    values = jax.vmap(
        lambda row: amplitude_fn(params, row, config.model_config)
    )(cut_states)
    coefficients = jnp.asarray(config.coefficients, dtype=jnp.complex128)
    normalization = jnp.sqrt(jnp.sum(jnp.abs(coefficients) ** 2))
    return jnp.sum(coefficients * values) / normalization


def shared_cut_amplitudes(
    params: object,
    states: jax.Array,
    config: SharedCutConfig,
    *,
    amplitude_fn: PhysicalAmplitude,
) -> jax.Array:
    """Vectorize :func:`shared_cut_amplitude` over physical states."""

    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError("expected a rank-two physical-state batch")
    return jax.vmap(
        lambda row: shared_cut_amplitude(
            params, row, config, amplitude_fn=amplitude_fn
        )
    )(array)


__all__ = ["SharedCutConfig", "shared_cut_amplitude", "shared_cut_amplitudes"]
