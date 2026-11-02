"""Minimal Hamiltonian-coupling input for the existing fixed-Sz Spatial-ViT.

Only the shared patch bias changes: ``b_eff = b + (j - center) * w_j``.
The backbone's width, depth, spatial mixing, complex head, and fixed doubled
magnetization constraint are unchanged. All backbone parameters remain
trainable. Coupling ``j`` is a dimensionless real scalar, not a spin label.

This wrapper is not an SU(2) decoder. An exact-small singlet/A1 projector may
be applied externally for a same-backbone variation-after-projection test.
It introduces no Hamiltonian or projection into the model and does not
claim scalable non-Abelian projection or phase-diagram generalization.
"""
from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_log_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)


def _real_scalar(value: jax.Array | float, name: str) -> jax.Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0 or jnp.iscomplexobj(scalar) or scalar.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite real scalar")
    if not isinstance(scalar, jax.core.Tracer) and not np.isfinite(np.asarray(scalar)):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar.astype(jnp.float64)


def _patch_bias(backbone: Mapping[str, object]) -> jax.Array:
    if not isinstance(backbone, Mapping) or "patch_embedding_bias" not in backbone:
        raise ValueError("backbone must contain patch_embedding_bias")
    bias = jnp.asarray(backbone["patch_embedding_bias"])
    if bias.ndim != 1 or bias.size < 1 or jnp.iscomplexobj(bias):
        raise ValueError("backbone patch_embedding_bias must be a nonempty real vector")
    if not isinstance(bias, jax.core.Tracer) and not np.isfinite(np.asarray(bias)).all():
        raise ValueError("backbone patch_embedding_bias must be finite")
    return bias


def wrap_spatial_vit_parameters(backbone: Mapping[str, object]) -> dict[str, object]:
    """Add a zero coupling vector to saved weights without modifying them.

    The immutable JAX leaves can be shared safely with the original pytree.
    No frozen reference, parameter detachment, or optimizer state is created.
    """
    bias = _patch_bias(backbone)
    return {"backbone": dict(backbone), "coupling_bias": jnp.zeros_like(bias)}


def init_coupling_conditioned_spatial_vit(
    key: jax.Array, config: FixedSzSpatialViTConfig,
) -> dict[str, object]:
    """Initialize the unchanged backbone and a zero coupling-input vector."""
    return wrap_spatial_vit_parameters(init_fixed_sz_spatial_vit(key, config))


def effective_parameters(
    params: Mapping[str, object], coupling: jax.Array | float, *,
    center: jax.Array | float = .5,
) -> dict[str, object]:
    """Return an old-style parameter dict for one coupling, fully differentiable.

    Concrete invalid inputs raise. Under JIT, dynamic nonfinite inputs produce
    a NaN bias rather than silently replacing the invalid coupling. The same
    bias is used for all patches, preserving the original coarse-translation
    symmetry. At ``coupling == center`` the extra direction is zero by design.
    """
    if not isinstance(params, Mapping) or set(params) != {"backbone", "coupling_bias"}:
        raise ValueError("params must contain exactly backbone and coupling_bias")
    backbone = params["backbone"]
    bias = _patch_bias(backbone)
    coupling_bias = jnp.asarray(params["coupling_bias"])
    if coupling_bias.shape != bias.shape or jnp.iscomplexobj(coupling_bias):
        raise ValueError("coupling_bias must be a real vector matching patch_embedding_bias")
    if not isinstance(coupling_bias, jax.core.Tracer) and not np.isfinite(np.asarray(coupling_bias)).all():
        raise ValueError("coupling_bias must be finite")
    j = _real_scalar(coupling, "coupling")
    origin = _real_scalar(center, "center")
    value = bias + (j - origin) * coupling_bias
    valid = (jnp.isfinite(j) & jnp.isfinite(origin)
             & jnp.all(jnp.isfinite(bias)) & jnp.all(jnp.isfinite(coupling_bias)))
    return {**backbone, "patch_embedding_bias": jnp.where(valid, value, jnp.nan)}


def coupling_conditioned_spatial_vit_log_amplitudes(
    params: Mapping[str, object], configurations: jax.Array,
    coupling: jax.Array | float, config: FixedSzSpatialViTConfig, *,
    center: jax.Array | float = .5,
) -> jax.Array:
    """Evaluate complex logs; computational bits and doubled Sz follow the base model."""
    return fixed_sz_spatial_vit_log_amplitudes(
        effective_parameters(params, coupling, center=center), configurations, config)


def coupling_conditioned_spatial_vit_amplitudes(
    params: Mapping[str, object], configurations: jax.Array,
    coupling: jax.Array | float, config: FixedSzSpatialViTConfig, *,
    center: jax.Array | float = .5,
) -> jax.Array:
    """Reuse the original batchwise stable amplitude gauge, not Born normalization.

    As in the original evaluator, the common log shift depends on the batch.
    Physical projection must reuse a consistent amplitude scale across images.
    """
    return fixed_sz_spatial_vit_amplitudes(
        effective_parameters(params, coupling, center=center), configurations, config)


def coupling_conditioned_spatial_vit_parameter_count(params: Mapping[str, object]) -> int:
    """Count all real backbone weights plus the additional width-d input vector."""
    effective_parameters(params, .5)
    return fixed_sz_spatial_vit_parameter_count(dict(params))


__all__ = ["effective_parameters", "wrap_spatial_vit_parameters",
           "init_coupling_conditioned_spatial_vit",
           "coupling_conditioned_spatial_vit_log_amplitudes",
           "coupling_conditioned_spatial_vit_amplitudes",
           "coupling_conditioned_spatial_vit_parameter_count"]
