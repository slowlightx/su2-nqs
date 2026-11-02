"""Minimal support-aware spatial Clebsch--Gordan spin ansatz.

The model coherently combines several complete, geometry-defined balanced
fusion routes.  Every route uses the *same* reduced ClebschTree parameters;
only a small vector of geometry-query coefficients is added.  Since route
features are static and every component is an exact ``(S,M)`` coefficient
tensor, the state and all parameter tangents remain in that irrep.

This is deliberately a small architecture test, not a Transformer.  It asks
whether resonating two-dimensional block-spin organizations improve a fixed
ClebschTree backbone before introducing configuration-dependent multiplicity-
space attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitude,
    spin_clebsch_tree_parameter_count,
)


@dataclass(frozen=True)
class SupportAwareSpatialCGConfig:
    """Static route and exact-spin contract for the minimal spatial model.

    ``route_orders[r]`` maps balanced-tree leaf positions to physical sites.
    Every order must cover all sites exactly once.  ``route_features`` are
    static standardized geometry descriptors, with one row per route.  The
    ``reference_route`` is the exactly nested single-tree baseline.
    """

    backbone: SpinClebschTreeConfig
    route_names: tuple[str, ...]
    route_orders: tuple[tuple[int, ...], ...]
    route_features: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    reference_route: int = 0

    def __post_init__(self) -> None:
        route_count = len(self.route_names)
        if route_count < 2:
            raise ValueError("support-aware Spatial-CG requires at least two routes")
        if len(set(self.route_names)) != route_count:
            raise ValueError("route_names must be unique")
        if len(self.route_orders) != route_count:
            raise ValueError("one route order is required per route name")
        expected = tuple(range(self.backbone.n_sites))
        if any(tuple(sorted(order)) != expected for order in self.route_orders):
            raise ValueError("every route must contain each physical site exactly once")
        if len(set(self.route_orders)) != route_count:
            raise ValueError("route_orders must be distinct")
        if len(self.route_features) != route_count:
            raise ValueError("one feature row is required per route")
        feature_count = len(self.feature_names)
        if feature_count < 1:
            raise ValueError("at least one support feature is required")
        if len(set(self.feature_names)) != feature_count:
            raise ValueError("feature_names must be unique")
        if any(len(row) != feature_count for row in self.route_features):
            raise ValueError("route feature rows have inconsistent width")
        features = np.asarray(self.route_features, dtype=np.float64)
        if np.any(~np.isfinite(features)):
            raise ValueError("route_features must be finite")
        if np.any(np.sqrt(np.mean(features**2, axis=0)) <= 1.0e-14):
            raise ValueError("every route feature must distinguish at least one route")
        if (
            isinstance(self.reference_route, bool)
            or not isinstance(self.reference_route, Integral)
            or not 0 <= int(self.reference_route) < route_count
        ):
            raise ValueError("reference_route is outside the route bank")

    @property
    def n_sites(self) -> int:
        return self.backbone.n_sites

    @property
    def n_routes(self) -> int:
        return len(self.route_names)

    @property
    def n_support_features(self) -> int:
        return len(self.feature_names)


def init_support_aware_spatial_cg(
    key: jax.Array,
    config: SupportAwareSpatialCGConfig,
) -> dict[str, object]:
    """Initialize at the exactly matched single-tree reference state.

    The geometry query starts at zero.  Its first derivative is nonzero
    because it directly multiplies route-state differences, so the optimizer
    can leave the nested baseline without a separate gate warm-up.
    """

    return {
        "backbone": init_spin_clebsch_tree(key, config.backbone),
        "geometry_query": jnp.zeros(
            (config.n_support_features,), dtype=jnp.float64
        ),
    }


def support_aware_spatial_cg_components(
    params: dict[str, object],
    spins: jax.Array,
    config: SupportAwareSpatialCGConfig,
) -> jax.Array:
    """Return all shared-backbone route amplitudes for one configuration."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.shape != (config.n_sites,):
        raise ValueError(f"expected spin shape ({config.n_sites},), got {array.shape}")
    orders = jnp.asarray(config.route_orders, dtype=jnp.int32)
    routed = array[orders]
    return jax.vmap(
        lambda row: spin_clebsch_tree_amplitude(
            params["backbone"], row, config.backbone
        )
    )(routed)


def support_aware_spatial_cg_route_coefficients(
    params: dict[str, object],
    config: SupportAwareSpatialCGConfig,
) -> jax.Array:
    """Return effective coherent coefficients of the route components.

    A bounded linearized attention ``tanh(F q)`` weights route residuals
    relative to the reference.  At ``q=0`` the coefficients are exactly
    one-hot on the reference route, while gradients with respect to ``q`` are
    generally nonzero.
    """

    query = jnp.asarray(params["geometry_query"], dtype=jnp.float64)
    if query.shape != (config.n_support_features,):
        raise ValueError("geometry_query shape does not match route features")
    features = jnp.asarray(config.route_features, dtype=jnp.float64)
    residual_weights = jnp.tanh(features @ query) / jnp.sqrt(
        jnp.asarray(config.n_routes, dtype=jnp.float64)
    )
    reference = int(config.reference_route)
    coefficients = residual_weights
    coefficients = coefficients.at[reference].add(1.0 - jnp.sum(residual_weights))
    return coefficients


def support_aware_spatial_cg_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: SupportAwareSpatialCGConfig,
) -> jax.Array:
    """Evaluate one exact-``(S,M)`` support-aware spatial amplitude."""

    components = support_aware_spatial_cg_components(params, spins, config)
    coefficients = support_aware_spatial_cg_route_coefficients(params, config)
    return jnp.dot(coefficients, components)


def support_aware_spatial_cg_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: SupportAwareSpatialCGConfig,
) -> jax.Array:
    """Vectorize the spatial-CG amplitude over physical configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: support_aware_spatial_cg_amplitude(params, row, config)
    )(array)


def support_aware_spatial_cg_component_matrix(
    params: dict[str, object],
    spins: jax.Array,
    config: SupportAwareSpatialCGConfig,
) -> jax.Array:
    """Return a ``(configuration, route)`` component-amplitude matrix."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: support_aware_spatial_cg_components(params, row, config)
    )(array)


def support_aware_spatial_cg_parameter_count(params: dict[str, object]) -> int:
    """Count backbone and support-query trainable scalars."""

    return spin_clebsch_tree_parameter_count(params["backbone"]) + int(
        np.prod(np.asarray(params["geometry_query"]).shape)
    )


__all__ = [
    "SupportAwareSpatialCGConfig",
    "init_support_aware_spatial_cg",
    "support_aware_spatial_cg_amplitude",
    "support_aware_spatial_cg_amplitudes",
    "support_aware_spatial_cg_component_matrix",
    "support_aware_spatial_cg_components",
    "support_aware_spatial_cg_parameter_count",
    "support_aware_spatial_cg_route_coefficients",
]
