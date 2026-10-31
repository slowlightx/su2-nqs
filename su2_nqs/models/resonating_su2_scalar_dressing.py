"""Configuration-dependent exact-SU(2) scalar carrier dressing.

The learned object in this module is a bank of symmetric singlet-pair
kernels.  A small spatial-attention block acts only on lattice coordinates
and produces low-rank, size-transferable corrections to a registered pair
kernel.  For a spin configuration ``sigma`` each kernel is evaluated as a
Gutzwiller-projected singlet determinant and subsequently Reynolds projected
to ``Gamma.A1`` by :mod:`su2_nqs.models.projected_singlet_agp`.

Consequently every column ``phi_r(sigma)`` is an exact spin singlet and an
exact spatial scalar.  The additive state

``psi(sigma) = c_0 psi_carrier(sigma) + sum_r c_r phi_r(sigma)``

is therefore exact in the same sector for arbitrary global coefficients.
The ratio to the carrier is genuinely configuration dependent, while the
explicit ``c_0`` channel guarantees that the original carrier is nested and
that its nodes are not imposed on the dressed state.

This is deliberately *not* an arbitrary neural multiplier in the magnetic
basis: such a multiplier would generally destroy exact SU(2).  Attention
only parameterises scalar pair kernels; all magnetic-index structure is fixed
analytically by the singlet determinant.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)


class SU2ScalarDressingBatch(NamedTuple):
    """Direct carrier, resonating columns, and their additive combination."""

    carrier_amplitudes: jax.Array
    resonating_amplitudes: jax.Array
    dressed_amplitudes: jax.Array


@dataclass(frozen=True)
class ResonatingSU2ScalarDressingConfig:
    """Static contract for a spatial-attention singlet-pair bank.

    ``component_count`` counts projected-AGP residual columns, excluding the
    immutable additive carrier.  ``embedding_dim`` must be divisible by
    ``head_count``.  The pair correction is Frobenius-normalised before the
    bounded ``tanh`` gate is applied, removing the homogeneous determinant
    scale gauge from optimisation.
    """

    rows: int
    columns: int
    embedding_dim: int = 12
    head_count: int = 3
    component_count: int = 2
    mlp_ratio: int = 2
    residual_scale: float = 0.35
    initialization_scale: float = 0.08

    def __post_init__(self) -> None:
        for name in (
            "rows",
            "columns",
            "embedding_dim",
            "head_count",
            "component_count",
            "mlp_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        for name in ("residual_scale", "initialization_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def n_sites(self) -> int:
        return self.rows * self.columns

    @property
    def coordinate_feature_count(self) -> int:
        # Constant plus three periodic reciprocal directions, with sine and
        # cosine retained separately.
        return 7


def periodic_coordinate_features(
    config: ResonatingSU2ScalarDressingConfig,
) -> jax.Array:
    """Return deterministic triangular-coordinate Fourier tokens."""

    sites = np.arange(config.n_sites, dtype=np.int64)
    x = sites % config.columns
    y = sites // config.columns
    first = 2.0 * np.pi * x / config.columns
    second = 2.0 * np.pi * y / config.rows
    third = first - second
    features = np.column_stack(
        (
            np.ones(config.n_sites),
            np.cos(first),
            np.sin(first),
            np.cos(second),
            np.sin(second),
            np.cos(third),
            np.sin(third),
        )
    )
    return jnp.asarray(features, dtype=jnp.float64)


def init_resonating_su2_scalar_dressing(
    key: jax.Array,
    config: ResonatingSU2ScalarDressingConfig,
) -> dict[str, jax.Array]:
    """Initialise the spatial-attention pair-kernel generator."""

    keys = iter(jax.random.split(key, 10))
    d = config.embedding_dim
    hidden = d * config.mlp_ratio
    scale = config.initialization_scale

    def normal(shape: tuple[int, ...], denominator: float = 1.0) -> jax.Array:
        return (scale / np.sqrt(denominator)) * jax.random.normal(
            next(keys), shape, dtype=jnp.float64
        )

    return {
        "input_kernel": normal((config.coordinate_feature_count, d), config.coordinate_feature_count),
        "input_bias": jnp.zeros((d,), dtype=jnp.float64),
        "query_kernel": normal((d, d), d),
        "key_kernel": normal((d, d), d),
        "value_kernel": normal((d, d), d),
        "output_kernel": normal((d, d), d),
        "mlp_in": normal((d, hidden), d),
        "mlp_out": normal((hidden, d), hidden),
        "pair_metrics": normal((config.component_count, d, d), d),
        # Small distinct nonzero gates avoid an initially rank-one bank while
        # remaining close to the registered carrier kernel.
        "pair_gates": jnp.linspace(
            0.08, 0.16, config.component_count, dtype=jnp.float64
        ),
    }


def _validate_pairing_inputs(
    base_pairing: jax.Array | np.ndarray,
    physical_directions: jax.Array | np.ndarray,
    config: ResonatingSU2ScalarDressingConfig,
) -> tuple[jax.Array, jax.Array]:
    base = jnp.asarray(base_pairing, dtype=jnp.float64)
    directions = jnp.asarray(physical_directions, dtype=jnp.float64)
    if base.shape != (config.n_sites, config.n_sites):
        raise ValueError("base_pairing has the wrong shape")
    if directions.ndim != 3 or directions.shape[1:] != base.shape:
        raise ValueError(
            "physical_directions must have shape (direction_count,n_sites,n_sites)"
        )
    return base, directions


def resonating_su2_scalar_pairing_bank(
    parameters: Mapping[str, jax.Array],
    base_pairing: jax.Array | np.ndarray,
    physical_directions: jax.Array | np.ndarray,
    config: ResonatingSU2ScalarDressingConfig,
) -> jax.Array:
    """Generate symmetric zero-onsite pair kernels with a fixed scale gauge.

    The supplied physical directions are used as fixed residual anchors.  A
    deterministic cyclic assignment gives every bank component a distinct
    local starting direction; attention can subsequently rotate away from
    those anchors.  This avoids reintroducing a separate trainable shell-only
    ansatz while retaining a stable physical initialization.
    """

    expected = {
        "input_kernel",
        "input_bias",
        "query_kernel",
        "key_kernel",
        "value_kernel",
        "output_kernel",
        "mlp_in",
        "mlp_out",
        "pair_metrics",
        "pair_gates",
    }
    if set(parameters) != expected:
        raise ValueError("dressing parameter keys do not match the model contract")
    base, directions = _validate_pairing_inputs(
        base_pairing, physical_directions, config
    )
    d = config.embedding_dim
    head_width = d // config.head_count
    features = periodic_coordinate_features(config)
    tokens = jnp.tanh(
        features @ jnp.asarray(parameters["input_kernel"], dtype=jnp.float64)
        + jnp.asarray(parameters["input_bias"], dtype=jnp.float64)
    )
    query = (tokens @ jnp.asarray(parameters["query_kernel"], dtype=jnp.float64)).reshape(
        config.n_sites, config.head_count, head_width
    )
    key = (tokens @ jnp.asarray(parameters["key_kernel"], dtype=jnp.float64)).reshape(
        config.n_sites, config.head_count, head_width
    )
    value = (tokens @ jnp.asarray(parameters["value_kernel"], dtype=jnp.float64)).reshape(
        config.n_sites, config.head_count, head_width
    )
    logits = jnp.einsum("ihd,jhd->hij", query, key) / np.sqrt(head_width)
    attention = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("hij,jhd->ihd", attention, value).reshape(
        config.n_sites, d
    )
    tokens = tokens + attended @ jnp.asarray(
        parameters["output_kernel"], dtype=jnp.float64
    )
    hidden = jax.nn.gelu(
        tokens @ jnp.asarray(parameters["mlp_in"], dtype=jnp.float64)
    )
    tokens = tokens + hidden @ jnp.asarray(
        parameters["mlp_out"], dtype=jnp.float64
    )

    metrics = jnp.asarray(parameters["pair_metrics"], dtype=jnp.float64)
    if metrics.shape != (config.component_count, d, d):
        raise ValueError("pair_metrics has the wrong shape")
    raw = jnp.einsum("id,rde,je->rij", tokens, metrics, tokens)
    raw = 0.5 * (raw + jnp.swapaxes(raw, 1, 2))
    raw = raw.at[:, jnp.arange(config.n_sites), jnp.arange(config.n_sites)].set(0.0)

    if directions.shape[0] > 0:
        anchor_indices = jnp.arange(config.component_count) % directions.shape[0]
        raw = raw + directions[anchor_indices]
    off_diagonal = 1.0 - jnp.eye(config.n_sites, dtype=jnp.float64)
    raw = raw * off_diagonal[None, :, :]
    base_norm = jnp.linalg.norm(base)
    residual_norm = jnp.linalg.norm(raw, axis=(1, 2))
    normalized = raw * (
        base_norm
        / jnp.maximum(residual_norm, jnp.asarray(1.0e-14, dtype=jnp.float64))
    )[:, None, None]
    gates = jnp.asarray(parameters["pair_gates"], dtype=jnp.float64)
    if gates.shape != (config.component_count,):
        raise ValueError("pair_gates has the wrong shape")
    bank = base[None, :, :] + (
        config.residual_scale * jnp.tanh(gates)
    )[:, None, None] * normalized
    # Pair-kernel norm is a homogeneous determinant gauge.  Fixing it prevents
    # meaningless radial drift and determinant overflow without restricting
    # the represented normalized states.
    bank_norm = jnp.linalg.norm(bank, axis=(1, 2))
    bank = bank * (base_norm / bank_norm)[:, None, None]
    return bank.at[:, jnp.arange(config.n_sites), jnp.arange(config.n_sites)].set(0.0)


def packed_resonating_pairing_bank(
    parameters: Mapping[str, jax.Array],
    base_pairing: jax.Array | np.ndarray,
    physical_directions: jax.Array | np.ndarray,
    config: ResonatingSU2ScalarDressingConfig,
) -> dict[str, jax.Array]:
    """Return a projected-AGP-compatible real parameter bank."""

    bank = resonating_su2_scalar_pairing_bank(
        parameters, base_pairing, physical_directions, config
    )
    first, second = jnp.triu_indices(config.n_sites, k=1)
    return {"pairing_real": bank[:, first, second]}


def projected_resonating_amplitudes(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray,
    base_pairing: jax.Array | np.ndarray,
    physical_directions: jax.Array | np.ndarray,
    dressing_config: ResonatingSU2ScalarDressingConfig,
    carrier_config: ProjectedSingletAGPConfig,
    *,
    microbatch_size: int | None = None,
) -> jax.Array:
    """Evaluate all exact-``S=0, Gamma.A1`` resonating columns."""

    if carrier_config.n_sites != dressing_config.n_sites:
        raise ValueError("carrier and dressing configurations disagree on n_sites")
    bank = packed_resonating_pairing_bank(
        parameters, base_pairing, physical_directions, dressing_config
    )["pairing_real"]
    columns = []
    for index in range(dressing_config.component_count):
        columns.append(
            projected_singlet_agp_amplitudes(
                {"pairing_real": bank[index]},
                configurations,
                carrier_config,
                microbatch_size=microbatch_size,
            ).amplitudes
        )
    return jnp.stack(columns, axis=1)


def additive_su2_scalar_dressing(
    carrier_amplitudes: jax.Array | np.ndarray,
    resonating_amplitudes: jax.Array | np.ndarray,
    coefficients: jax.Array | np.ndarray,
) -> SU2ScalarDressingBatch:
    """Combine the immutable carrier and projected resonating columns.

    ``coefficients[0]`` multiplies the immutable carrier.  The remaining
    coefficients multiply the resonating columns.  Keeping this operation
    direct-amplitude based makes exact nodes admissible.
    """

    carrier = jnp.asarray(carrier_amplitudes)
    resonating = jnp.asarray(resonating_amplitudes)
    values = jnp.asarray(coefficients)
    if carrier.ndim != 1:
        raise ValueError("carrier_amplitudes must be a vector")
    if resonating.ndim != 2 or resonating.shape[0] != carrier.shape[0]:
        raise ValueError("resonating_amplitudes must have shape (batch,components)")
    if values.shape != (resonating.shape[1] + 1,):
        raise ValueError("coefficients must contain carrier plus component weights")
    dressed = values[0] * carrier + resonating @ values[1:]
    return SU2ScalarDressingBatch(carrier, resonating, dressed)


__all__ = [
    "ResonatingSU2ScalarDressingConfig",
    "SU2ScalarDressingBatch",
    "additive_su2_scalar_dressing",
    "init_resonating_su2_scalar_dressing",
    "packed_resonating_pairing_bank",
    "periodic_coordinate_features",
    "projected_resonating_amplitudes",
    "resonating_su2_scalar_pairing_bank",
]
