"""Importance-corrected VMC gradients on an exactly shared sample set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CommonSampleGradient:
    """Gradient and overlap diagnostics for one common-sample VMC pass."""

    energy: float
    variance: float
    ess: float
    ess_fraction: float
    maximum_normalized_weight: float
    minimum_center_magnitude: float
    gradient: object
    gradient_norm: float


@dataclass(frozen=True)
class PairedEnergyDifference:
    """Ratio-estimator energy difference on exactly paired proposal samples."""

    energy_first: float
    energy_second: float
    difference: float
    standard_error: float
    z_score: float
    sample_count: int


def common_sample_vmc_value_and_grad(
    params: object,
    samples: object,
    reference_centers: jax.Array,
    *,
    hpsi_fn: Callable[[object, object], tuple[jax.Array, jax.Array]],
    center_fn: Callable[[object, object], jax.Array],
    gradient_microbatch_size: int | None = None,
) -> CommonSampleGradient:
    """Return the self-normalized importance VMC head gradient.

    ``samples`` are drawn once from a fixed reference wavefunction.  Every
    compared ansatz receives the identical configurations.  The current Born
    measure is recovered with ``|psi/reference|^2`` weights, while the local
    energies are stopped in the standard log-derivative estimator.  The
    returned ESS is mandatory evidence for interpreting the estimate.
    """

    centers, actions = hpsi_fn(params, samples)
    centers = jnp.asarray(centers, dtype=jnp.complex128)
    actions = jnp.asarray(actions, dtype=jnp.complex128)
    reference = jnp.asarray(reference_centers, dtype=jnp.complex128)
    if centers.ndim != 1 or actions.shape != centers.shape:
        raise ValueError("hpsi_fn must return matching rank-one arrays")
    if reference.shape != centers.shape:
        raise ValueError("reference_centers must match the shared sample count")
    concrete_centers = np.asarray(centers)
    concrete_reference = np.asarray(reference)
    if (
        np.any(~np.isfinite(concrete_centers))
        or np.any(~np.isfinite(concrete_reference))
        or np.any(np.abs(concrete_centers) == 0.0)
        or np.any(np.abs(concrete_reference) == 0.0)
    ):
        raise FloatingPointError("common-sample VMC encountered invalid centers")
    local = jax.lax.stop_gradient(actions / centers)
    raw_weights = jax.lax.stop_gradient(jnp.abs(centers / reference) ** 2)
    weights = raw_weights / jnp.sum(raw_weights)
    mean = jax.lax.stop_gradient(jnp.sum(weights * local))

    sample_count = len(concrete_centers)
    if gradient_microbatch_size is None:
        gradient_microbatch_size = sample_count
    gradient_microbatch_size = int(gradient_microbatch_size)
    if gradient_microbatch_size < 1:
        raise ValueError("gradient_microbatch_size must be positive")

    gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
    recomputed_chunks = []
    for start in range(0, sample_count, gradient_microbatch_size):
        stop = min(start + gradient_microbatch_size, sample_count)
        chunk_samples = samples[start:stop]
        chunk_weights = weights[start:stop]
        chunk_local = local[start:stop]

        def surrogate(candidate: object) -> tuple[jax.Array, jax.Array]:
            recomputed_chunk = jnp.asarray(
                center_fn(candidate, chunk_samples), dtype=jnp.complex128
            )
            if recomputed_chunk.shape != chunk_weights.shape:
                raise ValueError("center_fn returned the wrong microbatch shape")
            logs = jnp.log(recomputed_chunk)
            value = 2.0 * jnp.real(
                jnp.sum(
                    chunk_weights
                    * jnp.conj(logs)
                    * (chunk_local - mean)
                )
            )
            return value, recomputed_chunk

        (_value, recomputed_chunk), chunk_gradient = jax.value_and_grad(
            surrogate, has_aux=True
        )(params)
        recomputed_chunks.append(recomputed_chunk)
        gradient = jax.tree_util.tree_map(
            lambda total, update: total + update,
            gradient,
            chunk_gradient,
        )
    recomputed = jnp.concatenate(recomputed_chunks)
    if not np.allclose(
        np.asarray(recomputed), concrete_centers, atol=2.0e-14, rtol=2.0e-12
    ):
        raise ValueError("center_fn disagrees with hpsi_fn centers")
    weights_host = np.asarray(weights, dtype=np.float64)
    local_host = np.asarray(local, dtype=np.complex128)
    mean_host = complex(np.asarray(mean))
    ess = float(1.0 / np.sum(weights_host**2))
    gradient_norm = float(
        np.sqrt(
            sum(
                float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
                for leaf in jax.tree_util.tree_leaves(gradient)
            )
        )
    )
    diagnostics = (
        mean_host.real,
        ess,
        gradient_norm,
        float(np.min(np.abs(concrete_centers))),
    )
    if not all(math.isfinite(value) for value in diagnostics):
        raise FloatingPointError("common-sample VMC produced non-finite output")
    return CommonSampleGradient(
        energy=float(mean_host.real),
        variance=float(
            np.sum(weights_host * np.abs(local_host - mean_host) ** 2).real
        ),
        ess=ess,
        ess_fraction=ess / len(weights_host),
        maximum_normalized_weight=float(np.max(weights_host)),
        minimum_center_magnitude=float(np.min(np.abs(concrete_centers))),
        gradient=gradient,
        gradient_norm=gradient_norm,
    )


def paired_common_sample_energy_difference(
    first_centers: object,
    first_actions: object,
    second_centers: object,
    second_actions: object,
    reference_centers: object,
) -> PairedEnergyDifference:
    """Compare two variational energies with paired ratio-estimator errors.

    Samples are independent draws from the known proposal represented by
    ``reference_centers``.  For each state the self-normalized importance
    estimate is a ratio of sample means.  The reported uncertainty uses the
    difference of the two ratio-estimator influence functions, retaining the
    often-large covariance gained from evaluating both states on exactly the
    same configurations.
    """

    first = np.asarray(first_centers, dtype=np.complex128)
    first_hpsi = np.asarray(first_actions, dtype=np.complex128)
    second = np.asarray(second_centers, dtype=np.complex128)
    second_hpsi = np.asarray(second_actions, dtype=np.complex128)
    reference = np.asarray(reference_centers, dtype=np.complex128)
    if not (
        first.ndim == 1
        and first.shape == first_hpsi.shape == second.shape
        and first.shape == second_hpsi.shape == reference.shape
    ):
        raise ValueError("paired energy arrays must be matching rank-one vectors")
    if len(first) < 2:
        raise ValueError("paired uncertainty requires at least two samples")
    arrays = (first, first_hpsi, second, second_hpsi, reference)
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise FloatingPointError("paired energy arrays must be finite")
    if np.any(np.abs(first) == 0.0) or np.any(np.abs(second) == 0.0):
        raise FloatingPointError("paired energy centers must be nonzero")
    if np.any(np.abs(reference) == 0.0):
        raise FloatingPointError("paired proposal centers must be nonzero")

    def estimate(centers: np.ndarray, actions: np.ndarray):
        ratio = np.abs(centers / reference) ** 2
        local = np.real(actions / centers)
        ratio_mean = float(np.mean(ratio))
        if not math.isfinite(ratio_mean) or ratio_mean <= 0.0:
            raise FloatingPointError("paired importance ratio is invalid")
        energy = float(np.sum(ratio * local) / np.sum(ratio))
        influence = ratio * (local - energy) / ratio_mean
        return energy, influence

    first_energy, first_influence = estimate(first, first_hpsi)
    second_energy, second_influence = estimate(second, second_hpsi)
    difference = first_energy - second_energy
    paired_influence = first_influence - second_influence
    standard_error = float(
        np.std(paired_influence, ddof=1) / np.sqrt(len(paired_influence))
    )
    z_score = (
        float(difference / standard_error)
        if standard_error > 0.0
        else (0.0 if difference == 0.0 else math.copysign(math.inf, difference))
    )
    return PairedEnergyDifference(
        energy_first=first_energy,
        energy_second=second_energy,
        difference=difference,
        standard_error=standard_error,
        z_score=z_score,
        sample_count=len(first),
    )


__all__ = [
    "CommonSampleGradient",
    "PairedEnergyDifference",
    "common_sample_vmc_value_and_grad",
    "paired_common_sample_energy_difference",
]
