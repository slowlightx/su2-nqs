"""Strict first-order route residual around a projected SU(2) tree.

The scalar interpolation used by :mod:`coherent_orbit_nested_dag` gives route
parameters zero gradient at ``alpha=0``.  This module removes that second-order
gate while preserving the exact tree submanifold.  For a deterministic
rank-one reference kernel ``K0`` it evaluates

``psi = P tree + P DAG(K0 + delta) - P DAG(K0)``.

At ``delta=0`` the two DAG terms cancel for *any* shared merge/readout
backbone.  Derivatives with respect to ``delta`` are nevertheless the ordinary
DAG kernel derivatives and are generically nonzero at the tree point.

Only identity-group internal catalogs are accepted.  A single physical group
label is applied to each complete scalar contraction and summed by the
terminal projector, exactly as in the coherent-orbit successor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .coherent_orbit_nested_dag import SpinCoherentOrbitNestedDAGConfig
from .persistent_route_pair_dag import (
    SpinPersistentRoutePairConfig,
    init_paired_rank_one_and_equivariant_route_pair_dag,
    init_spin_persistent_route_pair_dag,
    spin_persistent_route_pair_dag_amplitude,
    spin_route_marginal_orbit_layouts,
    spin_route_pair_orbit_layouts,
)
from .reduced_fusion_dag import (
    SpinReducedFusionDAGConfig,
    spin_reduced_fusion_dag_amplitude,
)
from .spatial_projector import PhysicalSpatialProjector, spin_pullback_orbit


@dataclass(frozen=True)
class SpinStrictResidualRouteDAGConfig:
    """Static contract for a first-order route residual.

    The resonance mode must be either equivariant factorized rank one or
    equivariant full rank.  Equivariance is trivial for the required identity
    internal group; the naming is retained because the public persistent-DAG
    layouts define the matched marginal/joint parameter counts.
    """

    baseline: SpinReducedFusionDAGConfig
    resonance: SpinPersistentRoutePairConfig
    projector: PhysicalSpatialProjector

    def __post_init__(self) -> None:
        # Reuse the strict identity-group, sector, width, cutoff, and shared
        # merge-layout validation of the coherent-orbit construction.
        SpinCoherentOrbitNestedDAGConfig(
            self.baseline, self.resonance, self.projector
        )
        if self.resonance.pair_mode not in {
            "equivariant_factorized_rank_one",
            "equivariant_full_rank",
        }:
            raise ValueError(
                "strict residual requires equivariant factorized or full mode"
            )

    @property
    def n_sites(self) -> int:
        return self.baseline.n_sites

    @property
    def baseline_is_seed_subgraph(self) -> bool:
        return set(self.baseline.catalog.routes) <= set(
            self.resonance.catalog.routes
        )


def _incoming_route_count(config: SpinPersistentRoutePairConfig, support) -> int:
    return max(
        sum(route.parent == support for route in config.catalog.routes), 1
    )


def spin_strict_residual_reference_route_parameters(
    config: SpinStrictResidualRouteDAGConfig,
) -> dict[str, tuple[jax.Array, ...]]:
    """Return the deterministic, non-trainable rank-one ``K0`` coordinates."""

    resonance = config.resonance
    if resonance.pair_mode == "equivariant_factorized_rank_one":
        values: dict[str, tuple[jax.Array, ...]] = {}
        for side in ("left", "right"):
            rows = []
            for layout in spin_route_marginal_orbit_layouts(resonance, side):
                parent = layout.representative.parent_route
                child = parent.left if side == "left" else parent.right
                rows.append(
                    jnp.full(
                        (layout.channel_count,),
                        1.0 / np.sqrt(_incoming_route_count(resonance, child)),
                        dtype=jnp.float64,
                    )
                )
            values[f"{side}_route_orbit_weights"] = tuple(rows)
        return values

    rows = []
    for layout in spin_route_pair_orbit_layouts(resonance):
        parent = layout.representative.parent_route
        left_count = _incoming_route_count(resonance, parent.left)
        right_count = _incoming_route_count(resonance, parent.right)
        rows.append(
            jnp.full(
                (layout.channel_count,),
                1.0 / np.sqrt(left_count * right_count),
                dtype=jnp.float64,
            )
        )
    return {"pair_orbit_weights": tuple(rows)}


def _delta_keys(config: SpinStrictResidualRouteDAGConfig) -> tuple[str, ...]:
    if config.resonance.pair_mode == "equivariant_factorized_rank_one":
        return ("left_route_deltas", "right_route_deltas")
    return ("pair_route_deltas",)


def _reference_key(delta_key: str) -> str:
    return {
        "left_route_deltas": "left_route_orbit_weights",
        "right_route_deltas": "right_route_orbit_weights",
        "pair_route_deltas": "pair_orbit_weights",
    }[delta_key]


def _validate_params(
    params: Mapping[str, object], config: SpinStrictResidualRouteDAGConfig
) -> None:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    expected = {"merges", "readout", *_delta_keys(config)}
    if set(params) != expected:
        raise ValueError(f"expected parameter keys {sorted(expected)}")
    reference = spin_strict_residual_reference_route_parameters(config)
    for delta_key in _delta_keys(config):
        values = params[delta_key]
        expected_values = reference[_reference_key(delta_key)]
        if not isinstance(values, (tuple, list)) or len(values) != len(
            expected_values
        ):
            raise ValueError(f"{delta_key} has the wrong number of arrays")
        for index, (value, expected_value) in enumerate(
            zip(values, expected_values, strict=True)
        ):
            if np.shape(value) != np.shape(expected_value):
                raise ValueError(
                    f"{delta_key}[{index}] has shape {np.shape(value)}, "
                    f"expected {np.shape(expected_value)}"
                )


def spin_strict_residual_inner_params(
    params: Mapping[str, object],
    config: SpinStrictResidualRouteDAGConfig,
    *,
    active: bool,
) -> dict[str, object]:
    """Return persistent-DAG parameters for ``K0+delta`` or frozen ``K0``."""

    _validate_params(params, config)
    reference = spin_strict_residual_reference_route_parameters(config)
    result: dict[str, object] = {
        "merges": params["merges"],
        "readout": params["readout"],
    }
    for delta_key in _delta_keys(config):
        inner_key = _reference_key(delta_key)
        deltas = params[delta_key]
        result[inner_key] = tuple(
            base + delta if active else base
            for base, delta in zip(reference[inner_key], deltas, strict=True)
        )
    return result


def _from_initialized_inner(
    inner: Mapping[str, object], config: SpinStrictResidualRouteDAGConfig
) -> dict[str, object]:
    reference = spin_strict_residual_reference_route_parameters(config)
    result: dict[str, object] = {
        "merges": inner["merges"],
        "readout": inner["readout"],
    }
    for delta_key in _delta_keys(config):
        result[delta_key] = tuple(
            jnp.zeros_like(value)
            for value in reference[_reference_key(delta_key)]
        )
    _validate_params(result, config)
    return result


def init_spin_strict_residual_route_dag(
    key: jax.Array, config: SpinStrictResidualRouteDAGConfig
) -> dict[str, object]:
    """Initialize a strict residual exactly on its projected-tree submanifold."""

    inner = init_spin_persistent_route_pair_dag(key, config.resonance)
    return _from_initialized_inner(inner, config)


def init_paired_spin_strict_residual_route_dag(
    key: jax.Array,
    joint_config: SpinStrictResidualRouteDAGConfig,
) -> tuple[
    SpinStrictResidualRouteDAGConfig,
    dict[str, object],
    dict[str, object],
]:
    """Return capacity-matched factorized/joint zero-residual parameters."""

    if joint_config.resonance.pair_mode != "equivariant_full_rank":
        raise ValueError("joint_config must use equivariant_full_rank")
    factorized_resonance, factorized_inner, joint_inner = (
        init_paired_rank_one_and_equivariant_route_pair_dag(
            key, joint_config.resonance
        )
    )
    factorized_config = SpinStrictResidualRouteDAGConfig(
        joint_config.baseline,
        factorized_resonance,
        joint_config.projector,
    )
    factorized = _from_initialized_inner(factorized_inner, factorized_config)
    joint = _from_initialized_inner(joint_inner, joint_config)
    return factorized_config, factorized, joint


def _character_project(
    values: jax.Array, projector: PhysicalSpatialProjector
) -> jax.Array:
    characters = jnp.asarray(projector.characters, dtype=jnp.complex128)
    return jnp.sum(jnp.conj(characters) * values) / projector.group_order


def spin_strict_residual_route_dag_components(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinStrictResidualRouteDAGConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return global-``h`` tree, active-DAG, and reference-DAG carriers."""

    _validate_params(params, config)
    state = jnp.asarray(spins, dtype=jnp.int32)
    if state.shape != (config.n_sites,):
        raise ValueError(f"expected spin shape ({config.n_sites},)")
    orbit = spin_pullback_orbit(state, config.projector)
    tree_params = {"merges": params["merges"], "readout": params["readout"]}
    active = spin_strict_residual_inner_params(params, config, active=True)
    reference = spin_strict_residual_inner_params(params, config, active=False)
    tree_values = jax.vmap(
        lambda row: spin_reduced_fusion_dag_amplitude(
            tree_params, row, config.baseline
        )
    )(orbit)
    active_values = jax.vmap(
        lambda row: spin_persistent_route_pair_dag_amplitude(
            active, row, config.resonance
        )
    )(orbit)
    reference_values = jax.vmap(
        lambda row: spin_persistent_route_pair_dag_amplitude(
            reference, row, config.resonance
        )
    )(orbit)
    return tree_values, active_values, reference_values


def spin_strict_residual_route_dag_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinStrictResidualRouteDAGConfig,
) -> jax.Array:
    """Evaluate one strictly projected first-order residual amplitude."""

    tree, active, reference = spin_strict_residual_route_dag_components(
        params, spins, config
    )
    return (
        _character_project(tree, config.projector)
        + _character_project(active, config.projector)
        - _character_project(reference, config.projector)
    )


def spin_strict_residual_route_dag_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinStrictResidualRouteDAGConfig,
) -> jax.Array:
    """Vectorize strict residual amplitudes over physical configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(
        lambda row: spin_strict_residual_route_dag_amplitude(
            params, row, config
        )
    )(array)


def spin_strict_residual_route_dag_parameter_count(
    params: Mapping[str, object],
) -> int:
    """Count trainable scalars; the deterministic ``K0`` is not counted."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "SpinStrictResidualRouteDAGConfig",
    "init_paired_spin_strict_residual_route_dag",
    "init_spin_strict_residual_route_dag",
    "spin_strict_residual_inner_params",
    "spin_strict_residual_reference_route_parameters",
    "spin_strict_residual_route_dag_amplitude",
    "spin_strict_residual_route_dag_amplitudes",
    "spin_strict_residual_route_dag_components",
    "spin_strict_residual_route_dag_parameter_count",
]
