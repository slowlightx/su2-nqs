"""Strictly nested spin DAG with one coherent global space-group label.

The group-closed reduced-fusion catalogs used in the first local-DAG test
sum over symmetry-related routes independently at every support.  Expanding
that contraction therefore also produces ``Frankenstein`` diagrams in which
different subtrees carry different group elements.  A terminal projector
cannot remove those terms: it projects the scalar produced *after* the
independent local sums.

This module implements the smallest correction that distinguishes the two
sums.  Both a single-tree backbone and an identity-group resonating seed DAG
are evaluated for every physical group element ``h`` while retaining the
same ``h`` for the complete contraction.  Only then is the global carrier
summed with the requested one-dimensional character.  In formulas,

``psi = P_G psi_tree + alpha (P_G psi_seed_DAG - P_G psi_tree)``.

The seed DAG may contain persistent child-route-pair kernels, so local route
resonance remains possible *within* a fixed global frame.  At ``alpha=0`` the
model is exactly the projected single-tree orbit for arbitrary DAG kernel
parameters.  Merge tensors and the readout are shared between the two arms;
the strict extension therefore adds only the route-pair parameters and one
scalar interpolation parameter, rather than a second neural backbone.

The identity-group restriction on both internal catalogs is intentional.
Passing an already group-closed catalog would reintroduce the independent
local group sums that this construction is designed to exclude.  Spatial
symmetry is supplied only by :class:`PhysicalSpatialProjector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .persistent_route_pair_dag import (
    SpinPersistentRoutePairConfig,
    init_spin_persistent_route_pair_dag,
    spin_persistent_route_pair_dag_amplitude,
)
from .reduced_fusion_dag import (
    SpinReducedFusionDAGConfig,
    spin_reduced_fusion_dag_amplitude,
    spin_reduced_fusion_dag_merge_layouts,
)
from .spatial_projector import PhysicalSpatialProjector, spin_pullback_orbit


@dataclass(frozen=True)
class SpinCoherentOrbitNestedDAGConfig:
    """Static contract for a strict projected-tree/DAG extension.

    ``baseline`` must describe one complete fusion tree.  ``resonance`` must
    use an identity-group seed catalog.  A geometry candidate normally
    contains all baseline routes; a capacity-matched scrambled control need
    not, because the explicit tree skip still guarantees strict nesting.  The
    two configurations must have identical doubled-spin sector, feature
    width, cutoff, and size-shared merge layout.  ``projector`` alone owns
    the physical space-group action and may carry any validated
    one-dimensional character.
    """

    baseline: SpinReducedFusionDAGConfig
    resonance: SpinPersistentRoutePairConfig
    projector: PhysicalSpatialProjector

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, SpinReducedFusionDAGConfig):
            raise TypeError("baseline must be a SpinReducedFusionDAGConfig")
        if not isinstance(self.resonance, SpinPersistentRoutePairConfig):
            raise TypeError("resonance must be a SpinPersistentRoutePairConfig")
        if not isinstance(self.projector, PhysicalSpatialProjector):
            raise TypeError("projector must be a PhysicalSpatialProjector")
        if self.baseline.route_mode != "single_route":
            raise ValueError("baseline must use route_mode='single_route'")
        if self.projector.fermionic:
            raise ValueError("the spin coherent-orbit model requires fermionic=False")
        identity = tuple(range(self.baseline.n_sites))
        if self.baseline.catalog.group != (identity,):
            raise ValueError("baseline catalog must have the identity group only")
        if self.resonance.catalog.group != (identity,):
            raise ValueError(
                "resonance catalog must be an unclosed identity-group seed; "
                "group closure would restore independent local group sums"
            )
        if self.projector.n_sites != self.baseline.n_sites:
            raise ValueError("projector and baseline site counts disagree")
        if self.resonance.n_sites != self.baseline.n_sites:
            raise ValueError("baseline and resonance site counts disagree")
        if (
            self.baseline.target_q,
            self.baseline.target_m2,
            self.baseline.feature_dim,
            self.baseline.maximum_intermediate_q,
        ) != (
            self.resonance.target_q,
            self.resonance.target_m2,
            self.resonance.feature_dim,
            self.resonance.maximum_intermediate_q,
        ):
            raise ValueError(
                "baseline and resonance must share target (q,m), width, and cutoff"
            )
        if _merge_parameter_signature(self.baseline) != _resonance_signature(
            self.resonance
        ):
            raise ValueError(
                "baseline and resonance do not share the same merge-parameter layout"
            )

    @property
    def n_sites(self) -> int:
        return self.baseline.n_sites

    @property
    def baseline_is_seed_subgraph(self) -> bool:
        """Whether the local seed literally contains the anchored tree routes."""

        return set(self.baseline.catalog.routes) <= set(
            self.resonance.catalog.routes
        )


def _merge_parameter_signature(
    config: SpinReducedFusionDAGConfig,
) -> tuple[tuple[int, int, bool, tuple[tuple[int, int, int], ...]], ...]:
    """Return semantic (not merely shape) labels for shared merge tensors."""

    root_sites = config.n_sites
    representatives: dict[
        int, tuple[int, int, bool, tuple[tuple[int, int, int], ...]]
    ] = {}
    for layout in spin_reduced_fusion_dag_merge_layouts(config):
        representatives.setdefault(
            layout.parameter_index,
            (
                layout.left_size,
                layout.right_size,
                len(layout.route.parent) == root_sites,
                layout.keys,
            ),
        )
    return tuple(representatives[index] for index in range(len(representatives)))


def _resonance_signature(
    config: SpinPersistentRoutePairConfig,
) -> tuple[tuple[int, int, bool, tuple[tuple[int, int, int], ...]], ...]:
    # Persistent-route DAGs use precisely the route-diagonal base compiler
    # for their merge tensors.  Reconstructing that public config here avoids
    # reaching into the implementation's private cache helpers.
    base = SpinReducedFusionDAGConfig(
        catalog=config.catalog,
        target_q=config.target_q,
        target_m2=config.target_m2,
        feature_dim=config.feature_dim,
        maximum_intermediate_q=config.maximum_intermediate_q,
        initialization_scale=config.initialization_scale,
        route_mode="route_diagonal",
    )
    return _merge_parameter_signature(base)


def _validate_nested_params(
    params: Mapping[str, object], config: SpinCoherentOrbitNestedDAGConfig
) -> None:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    if set(params) != {"resonance", "alpha"}:
        raise ValueError("expected parameter keys ['alpha', 'resonance']")
    if not isinstance(params["resonance"], Mapping):
        raise TypeError("params['resonance'] must be a mapping")
    if np.shape(params["alpha"]) != ():
        raise ValueError("alpha must be a scalar")


def wrap_spin_coherent_orbit_nested_dag_params(
    resonance_params: Mapping[str, object],
    config: SpinCoherentOrbitNestedDAGConfig,
    *,
    alpha: float = 0.0,
) -> dict[str, object]:
    """Wrap preinitialised DAG parameters around the strict tree anchor.

    This is the fair-family entry point for benchmarks that already construct
    capacity-matched factorized, joint, and scrambled parameter sets.  Their
    ``merges`` and ``readout`` arrays become the common tree backbone without
    copying or reinitialising them.
    """

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real scalar")
    if not np.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    result: dict[str, object] = {
        "resonance": resonance_params,
        "alpha": jnp.asarray(float(alpha), dtype=jnp.float64),
    }
    # Calling one amplitude would unnecessarily compile numerical work.  The
    # nested boundary validates the exact keys/scalar now; the existing
    # persistent evaluator validates every inner tensor before contraction.
    _validate_nested_params(result, config)
    return result


def init_spin_coherent_orbit_nested_dag(
    key: jax.Array,
    config: SpinCoherentOrbitNestedDAGConfig,
    *,
    alpha: float = 0.0,
) -> dict[str, object]:
    """Initialise a seed DAG and place it on the exact tree submanifold."""

    resonance = init_spin_persistent_route_pair_dag(key, config.resonance)
    return wrap_spin_coherent_orbit_nested_dag_params(
        resonance, config, alpha=alpha
    )


def _baseline_params(resonance_params: Mapping[str, object]) -> dict[str, object]:
    return {
        "merges": resonance_params["merges"],
        "readout": resonance_params["readout"],
    }


def spin_coherent_orbit_nested_dag_components(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinCoherentOrbitNestedDAGConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(tree_h, resonance_h)`` before the coherent global-``h`` sum.

    The returned arrays both have shape ``(group_order,)``.  No image is
    deduplicated when a configuration or support has a nontrivial stabilizer:
    the explicit group label is the correlation carrier that was absent from
    the independently group-closed local DAG.
    """

    _validate_nested_params(params, config)
    state = jnp.asarray(spins, dtype=jnp.int32)
    if state.shape != (config.n_sites,):
        raise ValueError(
            f"expected spin shape ({config.n_sites},), got {state.shape}"
        )
    orbit = spin_pullback_orbit(state, config.projector)
    resonance_params = params["resonance"]
    baseline_params = _baseline_params(resonance_params)  # type: ignore[arg-type]
    tree_values = jax.vmap(
        lambda row: spin_reduced_fusion_dag_amplitude(
            baseline_params, row, config.baseline
        )
    )(orbit)
    resonance_values = jax.vmap(
        lambda row: spin_persistent_route_pair_dag_amplitude(
            resonance_params, row, config.resonance  # type: ignore[arg-type]
        )
    )(orbit)
    return tree_values, resonance_values


def _character_project(values: jax.Array, projector: PhysicalSpatialProjector) -> jax.Array:
    characters = jnp.asarray(projector.characters, dtype=jnp.complex128)
    return jnp.sum(jnp.conj(characters) * values) / projector.group_order


def spin_coherent_orbit_nested_dag_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinCoherentOrbitNestedDAGConfig,
) -> jax.Array:
    """Evaluate one strictly nested, coherently projected spin amplitude."""

    tree_h, resonance_h = spin_coherent_orbit_nested_dag_components(
        params, spins, config
    )
    tree = _character_project(tree_h, config.projector)
    resonance = _character_project(resonance_h, config.projector)
    alpha = jnp.asarray(params["alpha"])
    return tree + alpha * (resonance - tree)


def spin_coherent_orbit_nested_dag_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    config: SpinCoherentOrbitNestedDAGConfig,
) -> jax.Array:
    """Vectorise the coherent-orbit nested amplitude over configurations."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected spin batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(
        lambda row: spin_coherent_orbit_nested_dag_amplitude(params, row, config)
    )(array)


def coherent_group_label_product(
    local_values: jax.Array, characters: jax.Array | None = None
) -> jax.Array:
    """Contract local factors while keeping one group label for all factors.

    ``local_values[h, block]`` is reduced as
    ``mean_h conj(chi_h) prod_block local_values[h, block]``.  It is
    deliberately *not* ``prod_block mean_h local_values[h, block]``: the
    latter contains cross terms with independently chosen group labels and is
    the algebraic source of the failed group-closed local construction.
    """

    values = jnp.asarray(local_values)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("local_values must have shape (group, nonempty_blocks)")
    if characters is None:
        weights = jnp.ones((values.shape[0],), dtype=jnp.complex128)
    else:
        weights = jnp.asarray(characters, dtype=jnp.complex128)
        if weights.shape != (values.shape[0],):
            raise ValueError("characters must have one entry per group label")
    return jnp.mean(jnp.conj(weights) * jnp.prod(values, axis=1))


def spin_coherent_orbit_nested_dag_parameter_count(
    params: Mapping[str, object],
) -> int:
    """Count scalar trainable parameters, including ``alpha`` once."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    )


__all__ = [
    "SpinCoherentOrbitNestedDAGConfig",
    "coherent_group_label_product",
    "init_spin_coherent_orbit_nested_dag",
    "spin_coherent_orbit_nested_dag_amplitude",
    "spin_coherent_orbit_nested_dag_amplitudes",
    "spin_coherent_orbit_nested_dag_components",
    "spin_coherent_orbit_nested_dag_parameter_count",
    "wrap_spin_coherent_orbit_nested_dag_params",
]
