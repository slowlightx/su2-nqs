"""Log-derivative VMC gradients for real-valued network parameters."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .local_energy import Connectivity, LogPsi, local_energies


def vmc_loss(
    params: object,
    samples: Sequence[Sequence[int]] | jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Connectivity,
) -> jax.Array:
    """Energy-valued surrogate with the unbiased VMC energy gradient.

    The forward value is the sampled energy.  Automatic differentiation sees
    only ``2 Re <O* (E_loc-E)>``; differentiating through the discrete sampler
    or Hamiltonian connectivity is neither needed nor attempted.
    """

    energies = local_energies(params, samples, log_psi_fn, get_conn_fn)
    mean_energy = jnp.mean(energies)
    centered = jax.lax.stop_gradient(energies - mean_energy)
    sample_array = jnp.asarray(samples, dtype=jnp.int32)
    logs = jax.vmap(lambda sample: log_psi_fn(params, sample))(sample_array)
    surrogate = 2.0 * jnp.real(jnp.mean(jnp.conj(logs) * centered))
    return (
        jax.lax.stop_gradient(jnp.real(mean_energy))
        + surrogate
        - jax.lax.stop_gradient(surrogate)
    )


def energy_gradient(
    params: object,
    samples: Sequence[Sequence[int]] | jax.Array,
    log_psi_fn: LogPsi,
    get_conn_fn: Connectivity,
) -> tuple[jax.Array, object]:
    """Return sampled energy and its pytree gradient."""

    loss = lambda candidate: vmc_loss(candidate, samples, log_psi_fn, get_conn_fn)
    return jax.value_and_grad(loss)(params)
