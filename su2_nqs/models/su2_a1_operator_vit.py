"""Operator-token Transformer for exact-SU(2), ``Gamma-A1`` carrier spaces.

The network never acts on magnetic indices.  Its tokens label prevalidated
SU(2)-scalar, full-space-group operators and it produces the coefficients of
two residual operator layers,

``(I + sum_b c[1,b] O_b) (I + sum_a c[0,a] O_a) |carrier>``.

Consequently symmetry is structural: if the supplied component columns are
singlets in ``Gamma-A1``, every parameter value remains in that sector.  The
component expansion also makes the small-system implementation pure JAX and
Hamiltonian independent; scalable VMC may evaluate the same short operator
words through local connectivity instead of materialising the columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2A1OperatorViTConfig:
    """Static token metadata for a two-layer scalar Operator-ViT."""

    operator_labels: tuple[str, ...]
    operator_descriptors: tuple[tuple[float, ...], ...]
    embedding_dim: int = 12
    depth: int = 2
    coefficient_scale: float = 0.5
    initialization_scale: float = 0.05

    def __post_init__(self) -> None:
        if len(self.operator_labels) < 2:
            raise ValueError("at least two operator tokens are required")
        if len(set(self.operator_labels)) != len(self.operator_labels):
            raise ValueError("operator labels must be unique")
        descriptors = np.asarray(self.operator_descriptors, dtype=np.float64)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(
            self.operator_labels
        ):
            raise ValueError("one equal-length descriptor is required per token")
        if descriptors.shape[1] < 1 or np.any(~np.isfinite(descriptors)):
            raise ValueError("operator descriptors must be finite and nonempty")
        if (
            isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, Integral)
            or self.embedding_dim < 2
        ):
            raise ValueError("embedding_dim must be an integer of at least two")
        if self.depth != 2:
            raise ValueError("the validated component expansion requires depth=2")
        if not np.isfinite(self.coefficient_scale) or self.coefficient_scale <= 0:
            raise ValueError("coefficient_scale must be positive and finite")
        if (
            not np.isfinite(self.initialization_scale)
            or self.initialization_scale <= 0
        ):
            raise ValueError("initialization_scale must be positive and finite")

    @property
    def operator_count(self) -> int:
        return len(self.operator_labels)

    @property
    def descriptor_dim(self) -> int:
        return len(self.operator_descriptors[0])

    @property
    def component_count(self) -> int:
        return 1 + self.operator_count + self.operator_count**2


def _random_matrix(
    key: jax.Array,
    rows: int,
    columns: int,
    scale: float,
) -> jax.Array:
    return scale * np.sqrt(2.0 / float(rows + columns)) * jax.random.normal(
        key, (rows, columns), dtype=jnp.float64
    )


def init_su2_a1_operator_vit(
    key: jax.Array,
    config: SU2A1OperatorViTConfig,
) -> dict[str, object]:
    """Initialize a small attention network close to the bare carrier."""

    keys = iter(jax.random.split(key, 6 + 5 * config.depth))
    token_embedding = config.initialization_scale * jax.random.normal(
        next(keys),
        (config.operator_count, config.embedding_dim),
        dtype=jnp.float64,
    )
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "norm_scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                "norm_bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                "query": config.initialization_scale
                * jax.random.normal(
                    next(keys), (config.embedding_dim,), dtype=jnp.float64
                ),
                "key": _random_matrix(
                    next(keys),
                    config.embedding_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "value": _random_matrix(
                    next(keys),
                    config.embedding_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "output": _random_matrix(
                    next(keys),
                    config.embedding_dim,
                    config.embedding_dim,
                    config.initialization_scale,
                ),
                "readout": config.initialization_scale
                * jax.random.normal(
                    next(keys), (config.embedding_dim,), dtype=jnp.float64
                ),
            }
        )
    return {
        "descriptor_projection": _random_matrix(
            next(keys),
            config.descriptor_dim,
            config.embedding_dim,
            config.initialization_scale,
        ),
        "token_embedding": token_embedding,
        "layers": tuple(layers),
    }


def _layer_norm(values: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8) * scale + bias


def su2_a1_operator_vit_coefficients(
    params: dict[str, object],
    config: SU2A1OperatorViTConfig,
) -> jax.Array:
    """Return bounded coefficients with shape ``(2, operator_count)``."""

    descriptors = jnp.asarray(config.operator_descriptors, dtype=jnp.float64)
    hidden = (
        descriptors @ params["descriptor_projection"] + params["token_embedding"]
    )
    coefficients = []
    for layer in params["layers"]:
        normalized = _layer_norm(
            hidden, layer["norm_scale"], layer["norm_bias"]
        )
        keys = normalized @ layer["key"]
        values = normalized @ layer["value"]
        scores = keys @ layer["query"] / np.sqrt(config.embedding_dim)
        attention = jax.nn.softmax(scores)
        context = attention @ values
        hidden = hidden + jnp.tanh(context @ layer["output"])[None, :]
        raw = hidden @ layer["readout"] / np.sqrt(config.embedding_dim)
        coefficients.append(config.coefficient_scale * jnp.tanh(raw))
    result = jnp.stack(coefficients, axis=0)
    if result.shape != (config.depth, config.operator_count):
        raise RuntimeError("operator coefficient shape changed unexpectedly")
    return result


def su2_a1_operator_vit_component_weights(
    params: dict[str, object],
    config: SU2A1OperatorViTConfig,
) -> jax.Array:
    """Expand the two residual layers in the registered component order.

    Columns must be ordered as ``carrier``, ``O_a carrier``, followed by
    ``O_b O_a carrier`` with ``b`` the slow and ``a`` the fast index.
    """

    coefficients = su2_a1_operator_vit_coefficients(params, config)
    first, second = coefficients[0], coefficients[1]
    return jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.float64),
            first + second,
            jnp.outer(second, first).reshape(-1),
        )
    )


def su2_a1_operator_vit_amplitudes(
    params: dict[str, object],
    component_matrix: jax.Array,
    config: SU2A1OperatorViTConfig,
) -> jax.Array:
    """Combine prevalidated scalar-operator response columns coherently."""

    components = jnp.asarray(component_matrix)
    if components.ndim != 2 or components.shape[1] != config.component_count:
        raise ValueError(
            "component_matrix must have one carrier, first-word, and "
            "ordered second-word column"
        )
    return components @ su2_a1_operator_vit_component_weights(params, config)


def su2_a1_operator_vit_parameter_count(params: dict[str, object]) -> int:
    """Count real trainable scalars in the Operator-ViT pytree."""

    return int(
        sum(
            np.prod(np.asarray(leaf).shape)
            for leaf in jax.tree_util.tree_leaves(params)
        )
    )


__all__ = [
    "SU2A1OperatorViTConfig",
    "init_su2_a1_operator_vit",
    "su2_a1_operator_vit_amplitudes",
    "su2_a1_operator_vit_coefficients",
    "su2_a1_operator_vit_component_weights",
    "su2_a1_operator_vit_parameter_count",
]
