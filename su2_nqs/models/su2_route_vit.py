"""Exact-SU(2) route-token Transformer for resonating Clebsch trees.

This module is the minimal safe bridge between the established physical-spin
ClebschTree and a Transformer.  The Transformer never acts on a magnetic
index.  Instead, it reads immutable geometry descriptors of a bank of
complete-support tree routes and produces

* coherent coefficients for the route superposition; and
* low-rank modulations of the *reduced* ClebschTree tensors.

Every route component is therefore an exact ``(S,M)`` state for every value
of the neural parameters.  A coherent sum of those components remains in the
same irrep, as do all parameter tangents.  Lattice projection is intentionally
left to the existing finite-group projector.

The construction is a route-conditioned tensor hypernetwork, not a claim that
an ordinary configuration-space ViT can be inserted inside a CG contraction.
It is useful precisely where shared reduced tensors are too restrictive:
global attention can coordinate how the same low-rank reduced directions are
used by spatially distinct fusion organizations without learning magnetic
structure again.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .spin_clebsch_tree import (
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)
from .support_aware_spatial_cg import SupportAwareSpatialCGConfig


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2RouteViTConfig:
    """Static architecture contract for the route-token Transformer.

    ``spatial`` supplies the exact-spin backbone, complete site orders, and
    geometry descriptors.  ``modulation_rank=0`` is the coefficient-only
    control.  A positive rank additionally generates route-dependent reduced
    ClebschTree tensors from shared low-rank directions.
    """

    spatial: SupportAwareSpatialCGConfig
    embedding_dim: int = 12
    head_count: int = 3
    depth: int = 1
    mlp_ratio: int = 2
    modulation_rank: int = 2
    modulation_scale: float = 0.15
    initialization_scale: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("embedding_dim", self.embedding_dim),
            ("head_count", self.head_count),
            ("depth", self.depth),
            ("mlp_ratio", self.mlp_ratio),
            ("modulation_rank", self.modulation_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.embedding_dim < 1 or self.head_count < 1 or self.depth < 1:
            raise ValueError("embedding_dim, head_count, and depth must be positive")
        if self.embedding_dim % self.head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        if self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be positive")
        if self.modulation_rank < 0:
            raise ValueError("modulation_rank must be nonnegative")
        for name, value in (
            ("modulation_scale", self.modulation_scale),
            ("initialization_scale", self.initialization_scale),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def n_sites(self) -> int:
        return self.spatial.n_sites

    @property
    def n_routes(self) -> int:
        return self.spatial.n_routes

    @property
    def feature_count(self) -> int:
        return self.spatial.n_support_features

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.head_count

    @property
    def mlp_dim(self) -> int:
        return self.embedding_dim * self.mlp_ratio


def _matrix(
    key: jax.Array,
    rows: int,
    columns: int,
    scale: float,
) -> jax.Array:
    standard = float(scale) * np.sqrt(2.0 / float(rows + columns))
    return standard * jax.random.normal(
        key, (rows, columns), dtype=jnp.float64
    )


def _stacked_directions(
    key: jax.Array,
    backbone: object,
    rank: int,
    scale: float,
) -> object:
    """Return a backbone-shaped pytree with a leading low-rank axis."""

    leaves, structure = jax.tree_util.tree_flatten(backbone)
    keys = jax.random.split(key, len(leaves))
    directions = []
    for subkey, leaf in zip(keys, leaves, strict=True):
        fan_in = max(int(np.prod(leaf.shape)), 1)
        directions.append(
            (float(scale) / np.sqrt(fan_in))
            * jax.random.normal(
                subkey,
                (rank,) + tuple(leaf.shape),
                dtype=leaf.dtype,
            )
        )
    return jax.tree_util.tree_unflatten(structure, directions)


def init_su2_route_vit(
    key: jax.Array,
    config: SU2RouteViTConfig,
) -> dict[str, object]:
    """Initialize at the exactly nested reference-route ClebschTree.

    Both output heads start at zero.  The coefficient head has a nonzero first
    derivative and the reference route consequently has coefficient one while
    all other routes have coefficient zero.  Random reduced directions make
    the modulation head trainable on its first update without perturbing the
    nested initial state.
    """

    keys = iter(jax.random.split(key, 8 + 7 * config.depth))
    backbone = init_spin_clebsch_tree(next(keys), config.spatial.backbone)
    input_projection = _matrix(
        next(keys),
        config.feature_count,
        config.embedding_dim,
        config.initialization_scale,
    )
    input_bias = jnp.zeros((config.embedding_dim,), dtype=jnp.float64)
    layers = []
    for _ in range(config.depth):
        layers.append(
            {
                "norm_attention": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "query": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim,
                    config.initialization_scale,
                ),
                "key": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim,
                    config.initialization_scale,
                ),
                "value": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim,
                    config.initialization_scale,
                ),
                "output": _matrix(
                    next(keys), config.embedding_dim, config.embedding_dim,
                    config.initialization_scale,
                ),
                "norm_mlp": {
                    "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                    "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
                },
                "mlp_in": _matrix(
                    next(keys), config.embedding_dim, config.mlp_dim,
                    config.initialization_scale,
                ),
                "mlp_out": _matrix(
                    next(keys), config.mlp_dim, config.embedding_dim,
                    config.initialization_scale,
                ),
            }
        )
    directions = (
        _stacked_directions(
            next(keys),
            backbone,
            config.modulation_rank,
            config.initialization_scale,
        )
        if config.modulation_rank
        else jax.tree_util.tree_map(
            lambda leaf: jnp.zeros((0,) + leaf.shape, dtype=leaf.dtype),
            backbone,
        )
    )
    return {
        "backbone": backbone,
        "directions": directions,
        "transformer": {
            "input_projection": input_projection,
            "input_bias": input_bias,
            "layers": tuple(layers),
            "final_norm": {
                "scale": jnp.ones((config.embedding_dim,), dtype=jnp.float64),
                "bias": jnp.zeros((config.embedding_dim,), dtype=jnp.float64),
            },
            # Zero heads make the initial state exactly the registered
            # single-route baseline while retaining nonzero head gradients.
            "coefficient_head": jnp.zeros(
                (config.embedding_dim,), dtype=jnp.float64
            ),
            "modulation_head": jnp.zeros(
                (config.embedding_dim, config.modulation_rank), dtype=jnp.float64
            ),
        },
    }


def _layer_norm(values: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1.0e-8) * params[
        "scale"
    ] + params["bias"]


def su2_route_vit_tokens(
    params: dict[str, object],
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Return globally contextualized route tokens from static geometry."""

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
        logits = jnp.einsum("rhd,shd->hrs", query, key) / np.sqrt(
            config.head_dim
        )
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("hrs,shd->rhd", attention, value).reshape(
            config.n_routes, config.embedding_dim
        )
        tokens = tokens + attended @ layer["output"]
        normalized = _layer_norm(tokens, layer["norm_mlp"])
        hidden = jax.nn.gelu(normalized @ layer["mlp_in"])
        tokens = tokens + hidden @ layer["mlp_out"]
    return _layer_norm(tokens, transformer["final_norm"])


def su2_route_vit_controls(
    params: dict[str, object],
    config: SU2RouteViTConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return coherent route coefficients and reduced modulation controls."""

    tokens = su2_route_vit_tokens(params, config)
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
    if controls.shape != (config.modulation_rank,):
        raise ValueError("route modulation control has the wrong shape")
    return jax.tree_util.tree_map(
        lambda base, directions: base
        + float(config.modulation_scale)
        * jnp.einsum("k,k...->...", controls, directions),
        params["backbone"],
        params["directions"],
    )


def su2_route_vit_component_matrix(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Return ``(configuration, route)`` exact-SU(2) component amplitudes."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    _coefficients, controls = su2_route_vit_controls(params, config)
    orders = jnp.asarray(config.spatial.route_orders, dtype=jnp.int32)

    def one_route(control: jax.Array, order: jax.Array) -> jax.Array:
        backbone = _conditioned_backbone(params, control, config)
        return spin_clebsch_tree_amplitudes(
            backbone, array[:, order], config.spatial.backbone
        )

    # The explicit route batch is important beyond exact-small validation:
    # accelerator backends may fuse or shard this axis instead of compiling
    # one independent ClebschTree program for every spatial organization.
    by_route = jax.vmap(one_route)(controls, orders)
    return jnp.swapaxes(by_route, 0, 1)


def su2_route_vit_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteViTConfig,
) -> jax.Array:
    """Evaluate the coherent exact-SU(2) route superposition."""

    coefficients, _controls = su2_route_vit_controls(params, config)
    components = su2_route_vit_component_matrix(params, spins, config)
    return components @ coefficients


def su2_route_vit_parameter_count(params: dict[str, object]) -> int:
    """Count real scalar parameters, including low-rank directions."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


def su2_route_vit_backbone_parameter_count(
    params: dict[str, object],
) -> int:
    """Return the scalar count of the shared ClebschTree backbone only."""

    return spin_clebsch_tree_parameter_count(params["backbone"])


__all__ = [
    "SU2RouteViTConfig",
    "init_su2_route_vit",
    "su2_route_vit_amplitudes",
    "su2_route_vit_backbone_parameter_count",
    "su2_route_vit_component_matrix",
    "su2_route_vit_controls",
    "su2_route_vit_parameter_count",
    "su2_route_vit_tokens",
]
