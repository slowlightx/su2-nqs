"""Sample-fixed VMC surrogate for on-demand physical ``Hpsi`` kernels."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


PhysicalHpsi = Callable[[object], tuple[jax.Array, jax.Array]]


def physical_hpsi_energy_variance(
    params: object, *, hpsi_fn: PhysicalHpsi
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return sampled energy, variance and local energies."""

    centers, actions = hpsi_fn(params)
    if centers.ndim != 1 or actions.shape != centers.shape:
        raise ValueError("hpsi_fn must return two equal rank-one arrays")
    local = actions / centers
    energy = jnp.real(jnp.mean(local))
    variance = jnp.real(jnp.mean(jnp.abs(local - jnp.mean(local)) ** 2))
    return energy, variance, local


def physical_hpsi_vmc_loss(
    params: object, *, hpsi_fn: PhysicalHpsi
) -> jax.Array:
    """Energy-valued surrogate with the standard real-parameter VMC gradient."""

    loss, _ = physical_hpsi_vmc_loss_with_aux(params, hpsi_fn=hpsi_fn)
    return loss


def physical_hpsi_vmc_loss_with_aux(
    params: object, *, hpsi_fn: PhysicalHpsi
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the VMC surrogate and sample/zero-safety diagnostics."""

    centers, actions = hpsi_fn(params)
    if centers.ndim != 1 or actions.shape != centers.shape:
        raise ValueError("hpsi_fn must return two equal rank-one arrays")
    local = actions / centers
    mean_energy = jnp.mean(local)
    centered = jax.lax.stop_gradient(local - mean_energy)
    logs = jnp.log(centers.astype(jnp.complex128))
    surrogate = 2.0 * jnp.real(jnp.mean(jnp.conj(logs) * centered))
    loss = (
        jax.lax.stop_gradient(jnp.real(mean_energy))
        + surrogate
        - jax.lax.stop_gradient(surrogate)
    )
    auxiliary = {
        "energy": jnp.real(mean_energy),
        "energy_imaginary_abs": jnp.abs(jnp.imag(mean_energy)),
        "variance": jnp.real(
            jnp.mean(jnp.abs(local - jnp.mean(local)) ** 2)
        ),
        "minimum_center_magnitude": jnp.min(jnp.abs(centers)),
        "maximum_local_energy_magnitude": jnp.max(jnp.abs(local)),
        "nonfinite_local_count": jnp.sum(~jnp.isfinite(local)),
    }
    return loss, auxiliary


__all__ = [
    "PhysicalHpsi",
    "physical_hpsi_energy_variance",
    "physical_hpsi_vmc_loss",
    "physical_hpsi_vmc_loss_with_aux",
]
