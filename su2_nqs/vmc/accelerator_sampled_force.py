"""One-H-action sampled VMC force for accelerator-native campaigns.

This module is intentionally separate from the sealed, two-pass physical
Adam primitive.  A caller computes complete-H local energies once, freezes
them, and differentiates only batched log amplitudes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


BatchLogAmplitude = Callable[[object, Any], jax.Array]


@dataclass(frozen=True)
class SampledLogVMCGradient:
    """One-H-action VMC force built from frozen sampled local energies."""

    energy: float
    variance: float
    energy_imaginary_abs: float
    sample_count: int
    maximum_local_energy_magnitude: float
    nonfinite_local_count: int
    gradient: object
    gradient_norm: float


def _finite_pytree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _real_pytree(tree: object, *, name: str) -> None:
    if any(
        not np.issubdtype(np.asarray(leaf).dtype, np.floating)
        for leaf in jax.tree_util.tree_leaves(tree)
    ):
        raise TypeError(f"{name} must contain real floating-point leaves")


def _pytree_l2_norm(tree: object) -> float:
    squared = sum(
        float(np.sum(np.abs(np.asarray(leaf)) ** 2, dtype=np.float64))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return float(math.sqrt(squared))


def sampled_vmc_log_value_and_grad(
    params: object,
    samples: Any,
    local_energies: jax.Array | np.ndarray,
    *,
    log_amplitude_fn: BatchLogAmplitude,
) -> SampledLogVMCGradient:
    """Differentiate a VMC force without recomputing ``H psi`` during AD."""

    if not _finite_pytree(params):
        raise FloatingPointError("sampled VMC parameters are non-finite")
    _real_pytree(params, name="params")
    local = jnp.asarray(local_energies, dtype=jnp.complex128)
    if local.ndim != 1 or local.shape[0] < 2:
        raise ValueError("local_energies must contain at least two samples")
    reference_logs = jnp.asarray(log_amplitude_fn(params, samples))
    if reference_logs.shape != local.shape:
        raise ValueError("log_amplitude_fn must return one value per sample")
    local_host = np.asarray(local)
    log_host = np.asarray(reference_logs)
    nonfinite_local_count = int(np.count_nonzero(~np.isfinite(local_host)))
    if nonfinite_local_count or not np.all(np.isfinite(log_host)):
        raise FloatingPointError(
            "sampled VMC encountered non-finite logs or local energies"
        )
    shared_mean = jax.lax.stop_gradient(jnp.mean(local))
    stopped_local = jax.lax.stop_gradient(local)

    def centered_surrogate(candidate: object) -> jax.Array:
        logs = jnp.asarray(log_amplitude_fn(candidate, samples))
        if logs.shape != local.shape:
            raise ValueError("log_amplitude_fn changed its sampled output shape")
        return 2.0 * jnp.real(
            jnp.mean(jnp.conj(logs) * (stopped_local - shared_mean))
        )

    gradient = jax.grad(centered_surrogate)(params)
    mean_host = complex(np.asarray(shared_mean))
    variance = float(np.mean(np.abs(local_host - mean_host) ** 2).real)
    gradient_norm = _pytree_l2_norm(gradient)
    maximum_local = float(np.max(np.abs(local_host)))
    scalars = (
        mean_host.real,
        abs(mean_host.imag),
        variance,
        maximum_local,
        gradient_norm,
    )
    if not all(math.isfinite(value) for value in scalars) or not _finite_pytree(
        gradient
    ):
        raise FloatingPointError("sampled VMC produced non-finite diagnostics")
    _real_pytree(gradient, name="gradient")
    return SampledLogVMCGradient(
        energy=float(mean_host.real),
        variance=variance,
        energy_imaginary_abs=float(abs(mean_host.imag)),
        sample_count=int(local.shape[0]),
        maximum_local_energy_magnitude=maximum_local,
        nonfinite_local_count=nonfinite_local_count,
        gradient=gradient,
        gradient_norm=gradient_norm,
    )


__all__ = [
    "BatchLogAmplitude",
    "SampledLogVMCGradient",
    "sampled_vmc_log_value_and_grad",
]
