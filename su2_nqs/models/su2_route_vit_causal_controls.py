"""Causal controls for the exact-SU(2) route-token Transformer.

The production :mod:`su2_route_vit` model globally mixes physical route
tokens.  This module deliberately leaves that archived implementation
unchanged and supplies a parameter-count-identical local mixer.  Every
Q/K/V/O matrix remains active, but a route token can read only itself.  The
control therefore separates cross-route attention from reduced-tensor
conditioning without changing the SU(2) construction or parameter count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .spin_clebsch_tree import spin_clebsch_tree_amplitudes
from .su2_route_vit import SU2RouteViTConfig


jax.config.update("jax_enable_x64", True)


def _layer_norm(values: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8) * params[
        "scale"
    ] + params["bias"]


def su2_route_local_gate_tokens(
    params: dict[str, object],
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Return route-local contextual tokens with no cross-route read."""

    transformer = params["transformer"]
    features = jnp.asarray(config.spatial.route_features, dtype=jnp.float64)
    tokens = (
        features @ transformer["input_projection"]
        + transformer["input_bias"]
    )
    for layer in transformer["layers"]:
        normalized = _layer_norm(tokens, layer["norm_attention"])
        query = (normalized @ layer["query"]).reshape(
            config.n_routes, config.head_count, config.head_dim
        )
        key = (normalized @ layer["key"]).reshape(
            config.n_routes, config.head_count, config.head_dim
        )
        value = (normalized @ layer["value"]).reshape(
            config.n_routes, config.head_count, config.head_dim
        )
        # Q and K form a trainable route-local gate.  Multiplication by two
        # centers the residual scale at one, matching the global branch more
        # closely than a bare sigmoid while preserving locality.
        gate = 2.0 * jax.nn.sigmoid(
            jnp.einsum("rhd,rhd->rh", query, key) / np.sqrt(config.head_dim)
        )
        attended = (gate[..., None] * value).reshape(
            config.n_routes, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens, layer["norm_mlp"])
        hidden = jax.nn.gelu(normalized @ layer["mlp_in"])
        tokens = tokens + hidden @ layer["mlp_out"]
    return _layer_norm(tokens, transformer["final_norm"])


def su2_route_local_gate_controls(
    params: dict[str, object],
    config: SU2RouteViTConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return coherent coefficients and reduced controls for the local mixer."""

    tokens = su2_route_local_gate_tokens(params, config)
    transformer = params["transformer"]
    raw_coefficients = jnp.tanh(tokens @ transformer["coefficient_head"])
    residual = raw_coefficients / jnp.sqrt(
        jnp.asarray(config.n_routes, dtype=jnp.float64)
    )
    reference = int(config.spatial.reference_route)
    coefficients = residual.at[reference].add(1.0 - jnp.sum(residual))
    if config.modulation_rank:
        controls = jnp.tanh(tokens @ transformer["modulation_head"]) / jnp.sqrt(
            jnp.asarray(config.modulation_rank, dtype=jnp.float64)
        )
    else:
        controls = jnp.zeros((config.n_routes, 0), dtype=jnp.float64)
    return coefficients, controls


def _conditioned_backbone(
    params: dict[str, object],
    controls: jax.Array,
    config: SU2RouteViTConfig,
) -> object:
    return jax.tree_util.tree_map(
        lambda base, directions: base
        + float(config.modulation_scale)
        * jnp.einsum("k,k...->...", controls, directions),
        params["backbone"],
        params["directions"],
    )


def su2_route_local_gate_component_matrix(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Return exact-SU(2) route components for the local-mixer control."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    _coefficients, controls = su2_route_local_gate_controls(params, config)
    orders = jnp.asarray(config.spatial.route_orders, dtype=jnp.int32)

    def one_route(control: jax.Array, order: jax.Array) -> jax.Array:
        backbone = _conditioned_backbone(params, control, config)
        return spin_clebsch_tree_amplitudes(
            backbone, array[:, order], config.spatial.backbone
        )

    return jnp.swapaxes(jax.vmap(one_route)(controls, orders), 0, 1)


def su2_route_local_gate_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Evaluate the coherent route-local exact-SU(2) control state."""

    coefficients, _controls = su2_route_local_gate_controls(params, config)
    return su2_route_local_gate_component_matrix(params, spins, config) @ coefficients


__all__ = [
    "su2_route_local_gate_amplitudes",
    "su2_route_local_gate_component_matrix",
    "su2_route_local_gate_controls",
    "su2_route_local_gate_tokens",
]
