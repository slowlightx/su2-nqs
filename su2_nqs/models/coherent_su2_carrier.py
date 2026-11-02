"""Coherent nesting of a fixed physical carrier and an exact-SU(2) residual.

The operation in this module is deliberately additive rather than a
configuration-wise product.  If ``carrier`` and ``residual`` transform in the
same total-spin irrep, then every mixture and every parameter tangent stays in
that irrep.  This is the safe way to attach a reduced-multiplicity neural
correction to a projected parton state: multiplying the parton amplitude by a
generic fixed-``Sz`` correlator would not preserve total spin.

The zero head exactly returns the supplied carrier.  A caller may calibrate
``residual_scale`` without Hamiltonian information so that the two
unnormalised channels have comparable magnitudes on a reference sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CoherentSU2CarrierConfig:
    """Static contract for one fixed carrier plus one SU(2)-scalar channel."""

    n_sites: int
    residual_scale: float = 1.0
    complex_mixing: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or self.n_sites < 1
        ):
            raise ValueError("n_sites must be a positive integer")
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive and finite")
        if not isinstance(self.complex_mixing, bool):
            raise TypeError("complex_mixing must be boolean")

    @property
    def coordinate_count(self) -> int:
        return 2 if self.complex_mixing else 1


def init_coherent_su2_carrier_head(
    config: CoherentSU2CarrierConfig,
) -> dict[str, jax.Array]:
    """Return the exact-carrier nesting point ``lambda=0``."""

    return {
        "coordinates": jnp.zeros(
            (config.coordinate_count,), dtype=jnp.float64
        )
    }


def coherent_su2_carrier_coefficient(
    head_params: dict[str, jax.Array],
    config: CoherentSU2CarrierConfig,
) -> jax.Array:
    """Return a bounded real or complex coherent mixing coefficient."""

    coordinates = jnp.asarray(head_params["coordinates"], dtype=jnp.float64)
    if coordinates.shape != (config.coordinate_count,):
        raise ValueError("carrier head coordinate vector has the wrong shape")
    real = jnp.tanh(coordinates[0])
    if config.complex_mixing:
        coefficient = real + 1.0j * jnp.tanh(coordinates[1])
    else:
        coefficient = real
    return float(config.residual_scale) * coefficient


def coherent_su2_carrier_amplitudes(
    head_params: dict[str, jax.Array],
    carrier_amplitudes: jax.Array,
    residual_amplitudes: jax.Array,
    config: CoherentSU2CarrierConfig,
) -> jax.Array:
    """Combine two same-irrep amplitude batches coherently.

    The function does not infer symmetry from numerical values.  The caller is
    responsible for supplying a fixed carrier and a trainable residual in the
    same total-spin and lattice-symmetry sector.
    """

    carrier = jnp.asarray(carrier_amplitudes)
    residual = jnp.asarray(residual_amplitudes)
    if carrier.shape != residual.shape:
        raise ValueError("carrier and residual amplitude shapes must match")
    coefficient = coherent_su2_carrier_coefficient(head_params, config)
    return carrier + coefficient * residual


def coherent_su2_carrier_head_parameter_count(
    head_params: dict[str, jax.Array],
) -> int:
    """Count trainable real scalars in the coherent mixing head."""

    return int(
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(head_params))
    )


__all__ = [
    "CoherentSU2CarrierConfig",
    "coherent_su2_carrier_amplitudes",
    "coherent_su2_carrier_coefficient",
    "coherent_su2_carrier_head_parameter_count",
    "init_coherent_su2_carrier_head",
]
