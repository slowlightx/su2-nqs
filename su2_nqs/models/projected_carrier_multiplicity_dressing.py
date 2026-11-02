"""Exact-sector additive carrier plus SU(2)-multiplicity dressing.

This module closes a deliberately narrow symmetry boundary.  A fixed
spin-singlet carrier and an exact-SU(2) ClebschTree residual are evaluated in
the same physical spin basis.  Their coherent sum is then passed through the
same physical-space Reynolds projector on *every* amplitude call.  Training,
Hamiltonian ratios, and exchange sampling therefore all see only the target
spatial irrep; projection is never deferred to an analysis step.

The carrier callback must itself be a total-spin singlet.  The intended
choices are projected parton/AGP amplitudes.  The learned residual is the
configuration-dependent physical amplitude obtained by contracting trainable
reduced multiplicity maps with immutable Clebsch-Gordan tensors.  Neither the
carrier nor the learned network is allowed to act on magnetic indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from .su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
    init_su2_multiplicity_spatial_attention,
    su2_multiplicity_spatial_attention_amplitudes,
)


jax.config.update("jax_enable_x64", True)


class SingletCarrierAmplitude(Protocol):
    """One fixed physical-basis spin-singlet carrier amplitude."""

    def __call__(self, state: jax.Array) -> jax.Array:
        ...


@dataclass(frozen=True)
class ProjectedCarrierMultiplicityDressingConfig:
    """Static contract for an exact ``S=0, Gamma-A1`` dressed carrier.

    ``projector`` may technically carry any one-dimensional character, but
    production ground-state use must supply the trivial ``Gamma-A1``
    character.  The explicit ``spatial_sector`` label fails closed against an
    accidental translation-only projector being reported as full ``A1``.
    """

    multiplicity: SU2MultiplicitySpatialAttentionConfig
    projector: PhysicalSpatialProjector
    spatial_sector: str = "Gamma-A1"

    def __post_init__(self) -> None:
        if self.multiplicity.base.spatial.backbone.target_q != 0:
            raise ValueError("multiplicity residual must target exact q=0")
        if self.multiplicity.base.spatial.backbone.target_m2 != 0:
            raise ValueError("multiplicity residual must target m2=0")
        if self.projector.fermionic:
            raise ValueError("spin dressing requires a bosonic site projector")
        if self.projector.n_sites != self.multiplicity.n_sites:
            raise ValueError("projector and multiplicity residual site counts differ")
        if self.spatial_sector != "Gamma-A1":
            raise ValueError("ground-state dressing is restricted to Gamma-A1")
        characters = np.asarray(self.projector.characters, dtype=np.complex128)
        if np.max(np.abs(characters - 1.0)) > 1.0e-14:
            raise ValueError("Gamma-A1 requires the trivial spatial character")

    @property
    def n_sites(self) -> int:
        return self.multiplicity.n_sites


def init_projected_carrier_multiplicity_dressing(
    key: jax.Array,
    frozen_params: dict[str, object],
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    initial_residual_coefficient: float = 1.0e-3,
) -> dict[str, object]:
    """Initialize the reduced residual and its real additive coefficient."""

    if (
        isinstance(initial_residual_coefficient, bool)
        or not isinstance(initial_residual_coefficient, Real)
        or not np.isfinite(initial_residual_coefficient)
    ):
        raise ValueError("initial_residual_coefficient must be finite and real")
    return {
        "multiplicity": init_su2_multiplicity_spatial_attention(
            key, frozen_params, config.multiplicity
        ),
        "residual_coefficient": jnp.asarray(
            float(initial_residual_coefficient), dtype=jnp.float64
        ),
    }


def raw_carrier_multiplicity_amplitude(
    params: dict[str, object],
    frozen_params: dict[str, object],
    state: jax.Array,
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    carrier_amplitude_fn: SingletCarrierAmplitude,
) -> jax.Array:
    """Return the unprojected coherent carrier-plus-residual amplitude.

    This function is public for exact-small audits only.  VMC and sampling
    must use :func:`projected_carrier_multiplicity_dressing_amplitude`.
    """

    array = jnp.asarray(state, dtype=jnp.int32)
    if array.shape != (config.n_sites,):
        raise ValueError(f"expected one spin configuration of length {config.n_sites}")
    carrier = jnp.asarray(carrier_amplitude_fn(array))
    if carrier.ndim != 0:
        raise ValueError("carrier_amplitude_fn must return one scalar")
    residual = su2_multiplicity_spatial_attention_amplitudes(
        params["multiplicity"],
        frozen_params,
        array[None, :],
        config.multiplicity,
    )[0]
    coefficient = jnp.asarray(params["residual_coefficient"], dtype=jnp.float64)
    if coefficient.shape != ():
        raise ValueError("residual_coefficient must be scalar")
    return carrier + coefficient * residual


def projected_carrier_multiplicity_dressing_amplitude(
    params: dict[str, object],
    frozen_params: dict[str, object],
    state: jax.Array,
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    carrier_amplitude_fn: SingletCarrierAmplitude,
) -> jax.Array:
    """Evaluate one exact-``S=0, Gamma-A1`` dressed amplitude."""

    def raw(candidate, row, _model_config):
        return raw_carrier_multiplicity_amplitude(
            candidate,
            frozen_params,
            row,
            config,
            carrier_amplitude_fn=carrier_amplitude_fn,
        )

    return spatial_projected_amplitude(
        params,
        state,
        config,
        config.projector,
        amplitude_fn=raw,
    )


def projected_carrier_multiplicity_dressing_amplitudes(
    params: dict[str, object],
    frozen_params: dict[str, object],
    states: jax.Array,
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    carrier_amplitude_fn: SingletCarrierAmplitude,
) -> jax.Array:
    """Vectorize the exact-sector amplitude over physical configurations."""

    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(
        lambda row: projected_carrier_multiplicity_dressing_amplitude(
            params,
            frozen_params,
            row,
            config,
            carrier_amplitude_fn=carrier_amplitude_fn,
        )
    )(array)


def certified_carrier_projected_residual_amplitude(
    params: dict[str, object],
    frozen_params: dict[str, object],
    state: jax.Array,
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    carrier_amplitude_fn: SingletCarrierAmplitude,
) -> jax.Array:
    """Evaluate a certified projected carrier plus a projected residual.

    This is the scalable form used when the carrier already has an independent
    exact ``S=0, Gamma-A1`` certificate, for example the induced
    three-determinant triangular pi/0-flux carrier.  Only the learned
    multiplicity residual needs a Reynolds sum.  Linearity makes this exactly
    equal to projecting the coherent sum, while avoiding one carrier
    evaluation per group image.

    The function deliberately does not infer symmetry from numerical samples:
    callers must supply a carrier whose exact certificate is separately
    tested and provenance sealed.
    """

    array = jnp.asarray(state, dtype=jnp.int32)
    if array.shape != (config.n_sites,):
        raise ValueError(f"expected one spin configuration of length {config.n_sites}")
    carrier = jnp.asarray(carrier_amplitude_fn(array))
    if carrier.ndim != 0:
        raise ValueError("carrier_amplitude_fn must return one scalar")

    def raw_residual(candidate, row, _model_config):
        return su2_multiplicity_spatial_attention_amplitudes(
            candidate["multiplicity"],
            frozen_params,
            row[None, :],
            config.multiplicity,
        )[0]

    residual = spatial_projected_amplitude(
        params,
        array,
        config,
        config.projector,
        amplitude_fn=raw_residual,
    )
    coefficient = jnp.asarray(params["residual_coefficient"], dtype=jnp.float64)
    if coefficient.shape != ():
        raise ValueError("residual_coefficient must be scalar")
    return carrier + coefficient * residual


def certified_carrier_projected_residual_amplitudes(
    params: dict[str, object],
    frozen_params: dict[str, object],
    states: jax.Array,
    config: ProjectedCarrierMultiplicityDressingConfig,
    *,
    carrier_amplitude_fn: SingletCarrierAmplitude,
) -> jax.Array:
    """Batch :func:`certified_carrier_projected_residual_amplitude`."""

    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {config.n_sites})")
    return jax.vmap(
        lambda row: certified_carrier_projected_residual_amplitude(
            params,
            frozen_params,
            row,
            config,
            carrier_amplitude_fn=carrier_amplitude_fn,
        )
    )(array)


__all__ = [
    "ProjectedCarrierMultiplicityDressingConfig",
    "SingletCarrierAmplitude",
    "certified_carrier_projected_residual_amplitude",
    "certified_carrier_projected_residual_amplitudes",
    "init_projected_carrier_multiplicity_dressing",
    "projected_carrier_multiplicity_dressing_amplitude",
    "projected_carrier_multiplicity_dressing_amplitudes",
    "raw_carrier_multiplicity_amplitude",
]
