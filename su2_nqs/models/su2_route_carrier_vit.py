"""Trainable symmetry-labelled heads on a frozen exact-SU(2) Route-ViT.

Each head direction controls a coherent route resonance and a reduced
virtual-spin fugacity across every non-root fusion scale.  The latter uses the
SU(2) Casimir of the intermediate channel, so it changes the balance of low-
and high-spin virtual fluctuations instead of merely rescaling every dimer.
Both operations only mix or reweight complete exact-SU(2) components, so the
wavefunction and every head tangent remain in the requested total-spin irrep.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .spin_clebsch_tree import (
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_layouts,
)
from .su2_route_vit import (
    SU2RouteViTConfig,
    _conditioned_backbone,
    su2_route_vit_controls,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SU2RouteCarrierViTConfig:
    """Frozen backbone plus ``(head, route, mechanism)`` carrier templates."""

    base: SU2RouteViTConfig
    templates: tuple[tuple[tuple[float, float], ...], ...]
    labels: tuple[str, ...]
    carrier_scale: float = 0.15

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("at least one carrier template is required")
        if len(self.labels) != len(self.templates):
            raise ValueError("one label is required per carrier template")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("carrier labels must be unique")
        values = np.asarray(self.templates, dtype=np.float64)
        if values.shape != (len(self.templates), self.base.n_routes, 2):
            raise ValueError(
                "carrier templates must have shape (head, route, 2)"
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("carrier templates must be finite")
        rank = np.linalg.matrix_rank(values.reshape(len(self.templates), -1))
        if rank != len(self.templates):
            raise ValueError("carrier templates must be linearly independent")
        if not np.isfinite(self.carrier_scale) or self.carrier_scale <= 0.0:
            raise ValueError("carrier_scale must be positive and finite")

    @property
    def n_sites(self) -> int:
        return self.base.n_sites

    @property
    def n_routes(self) -> int:
        return self.base.n_routes

    @property
    def head_count(self) -> int:
        return len(self.templates)


def init_su2_route_carrier_head(
    config: SU2RouteCarrierViTConfig,
) -> dict[str, jax.Array]:
    """Initialize the carrier at the exactly shared frozen backbone."""

    return {"coefficients": jnp.zeros((config.head_count,), dtype=jnp.float64)}


def su2_route_carrier_controls(
    head_params: dict[str, jax.Array],
    config: SU2RouteCarrierViTConfig,
) -> jax.Array:
    """Return bounded ``(route, resonance/reduced)`` carrier controls."""

    coefficients = jnp.asarray(head_params["coefficients"], dtype=jnp.float64)
    if coefficients.shape != (config.head_count,):
        raise ValueError("carrier coefficient vector has the wrong shape")
    templates = jnp.asarray(config.templates, dtype=jnp.float64)
    raw = jnp.einsum("k,krc->rc", coefficients, templates)
    return float(config.carrier_scale) * jnp.tanh(raw)


def _carrier_backbone(
    frozen_params: dict[str, object],
    base_control: jax.Array,
    reduced_control: jax.Array,
    config: SU2RouteCarrierViTConfig,
) -> object:
    backbone = _conditioned_backbone(frozen_params, base_control, config.base)
    merges = list(backbone["merges"])
    # A multiplicative fugacity cannot activate a channel that is tiny in a
    # near-dimer initialization.  Reuse the first Hamiltonian-blind Route-ViT
    # reduced direction as an additive, leaf-normalized virtual-fluctuation
    # direction.  It is common to every arm; only the preregistered route
    # templates differ between A1/E2 and generic controls.
    if config.base.modulation_rank:
        direction_merges = frozen_params["directions"]["merges"]
        for parameter_index, (merge, directions) in enumerate(
            zip(merges, direction_merges, strict=True)
        ):
            direction = directions[0]
            direction_rms = jnp.sqrt(jnp.mean(direction * direction))
            merge_rms = jnp.sqrt(jnp.mean(merge * merge))
            normalized = direction * merge_rms / jnp.maximum(
                direction_rms, jnp.asarray(1.0e-14, dtype=jnp.float64)
            )
            merges[parameter_index] = merge + reduced_control * normalized
    layouts = spin_clebsch_tree_layouts(config.base.spatial.backbone)
    representative = {}
    multiplicity = {}
    for layout in layouts:
        representative.setdefault(layout.parameter_index, layout)
        multiplicity[layout.parameter_index] = (
            multiplicity.get(layout.parameter_index, 0) + 1
        )
    for parameter_index, layout in representative.items():
        q_total = np.asarray([key[2] for key in layout.keys], dtype=np.float64)
        casimir = q_total * (q_total + 2.0) / 4.0
        centered = casimir - np.mean(casimir)
        rms = float(np.sqrt(np.mean(centered**2)))
        if rms <= 1.0e-14:
            continue
        score = jnp.asarray(centered / rms, dtype=jnp.float64)
        # A shared tensor can appear at several nodes.  The square-root count
        # keeps a single carrier coefficient comparable across tree depths.
        score = score / np.sqrt(float(multiplicity[parameter_index]))
        factors = jnp.exp(reduced_control * score)
        merges[parameter_index] = (
            merges[parameter_index] * factors[:, None, None, None]
        )
    return {**backbone, "merges": tuple(merges)}


def su2_route_carrier_component_matrix(
    head_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteCarrierViTConfig,
) -> jax.Array:
    """Return exact-SU(2) route components after carrier reweighting."""

    array = jnp.asarray(spins, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    _base_coefficients, base_controls = su2_route_vit_controls(
        frozen_params, config.base
    )
    carrier = su2_route_carrier_controls(head_params, config)
    orders = jnp.asarray(config.base.spatial.route_orders, dtype=jnp.int32)

    def one_route(
        base_control: jax.Array,
        reduced_control: jax.Array,
        order: jax.Array,
    ) -> jax.Array:
        backbone = _carrier_backbone(
            frozen_params, base_control, reduced_control, config
        )
        return spin_clebsch_tree_amplitudes(
            backbone, array[:, order], config.base.spatial.backbone
        )

    by_route = jax.vmap(one_route)(base_controls, carrier[:, 1], orders)
    return jnp.swapaxes(by_route, 0, 1)


def su2_route_carrier_coefficients(
    head_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    config: SU2RouteCarrierViTConfig,
) -> jax.Array:
    """Return base coefficients plus a norm-controlled coherent resonance."""

    base_coefficients, _ = su2_route_vit_controls(frozen_params, config.base)
    carrier = su2_route_carrier_controls(head_params, config)
    residual = carrier[:, 0] / jnp.sqrt(
        jnp.asarray(config.n_routes, dtype=jnp.float64)
    )
    reference = int(config.base.spatial.reference_route)
    residual = residual.at[reference].add(-jnp.sum(residual))
    return base_coefficients + residual


def su2_route_carrier_vit_amplitudes(
    head_params: dict[str, jax.Array],
    frozen_params: dict[str, object],
    spins: jax.Array,
    config: SU2RouteCarrierViTConfig,
) -> jax.Array:
    """Evaluate the carrier-augmented coherent exact-SU(2) state."""

    components = su2_route_carrier_component_matrix(
        head_params, frozen_params, spins, config
    )
    coefficients = su2_route_carrier_coefficients(
        head_params, frozen_params, config
    )
    return components @ coefficients


def su2_route_carrier_head_parameter_count(
    head_params: dict[str, jax.Array],
) -> int:
    """Count trainable head scalars, excluding the frozen common backbone."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(head_params))
    )


__all__ = [
    "SU2RouteCarrierViTConfig",
    "init_su2_route_carrier_head",
    "su2_route_carrier_coefficients",
    "su2_route_carrier_component_matrix",
    "su2_route_carrier_controls",
    "su2_route_carrier_head_parameter_count",
    "su2_route_carrier_vit_amplitudes",
]
