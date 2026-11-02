"""Versioned center-only sampled-VMC gradient primitives.

This module is a cost-correct successor to :mod:`physical_sampled_adam`.  It
is deliberately versioned instead of modifying that module because sealed
sampled-training artifacts include the old implementation in their
transitive source manifests.

For a fixed collection of samples, the Hamiltonian action is needed only to
form the stopped local energies.  The derivative itself needs
``d log(psi(center)) / d theta`` and therefore must not reconstruct the
connected-state Hamiltonian action.  Callers consequently provide two
callbacks:

``hpsi_fn``
    Evaluated once per microbatch to obtain centers and ``H|psi>``.

``center_fn``
    Evaluated once per microbatch under automatic differentiation and must
    compute only the sampled center amplitudes.

Both passes use one global local-energy center and each microbatch is divided
by the global sample count.  The result is algebraically identical to the
ordinary concatenated-batch VMC gradient, including unequal microbatches.

The Adam state and update functions are re-exported from the sealed module,
so existing optimizer checkpoints retain exactly the same ``NamedTuple``
type and field layout.  Existing workflows must opt in by importing this
module; the sealed workflow remains untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .physical_sampled_adam import (
    MAX_PHYSICAL_MICROBATCH_SIZE,
    PhysicalBatchHpsi,
    SampledAdamDiagnostics,
    SampledAdamState,
    SampledVMCGradient,
    _checked_hpsi,
    _finite_pytree,
    _pytree_l2_norm,
    _real_pytree,
    initialize_sampled_adam,
    sampled_adam_update,
)


PhysicalBatchCenter = Callable[[object, Any], jax.Array]

# A separately written center kernel should agree with the center output of
# Hpsi to near float64 roundoff.  Failing closed here prevents a cheap but
# mathematically inconsistent gradient from being accepted silently.
CENTER_CONSISTENCY_RTOL = 2.0e-12
CENTER_CONSISTENCY_ATOL = 2.0e-14


def _checked_centers(
    params: object,
    batch: Any,
    center_fn: PhysicalBatchCenter,
    *,
    expected_size: int,
) -> jax.Array:
    centers = jnp.asarray(center_fn(params, batch))
    if centers.ndim != 1 or centers.shape[0] != expected_size:
        raise ValueError(
            "center_fn must return a rank-one center array matching hpsi_fn"
        )
    if centers.shape[0] > MAX_PHYSICAL_MICROBATCH_SIZE:
        raise ValueError(
            "physical sampled Adam microbatches are limited to "
            f"{MAX_PHYSICAL_MICROBATCH_SIZE} samples"
        )
    return centers


def sampled_vmc_value_and_grad(
    params: object,
    batches: Sequence[Any],
    *,
    hpsi_fn: PhysicalBatchHpsi,
    center_fn: PhysicalBatchCenter,
) -> SampledVMCGradient:
    """Return a globally centered VMC gradient with one Hpsi pass.

    ``hpsi_fn`` is called exactly once per microbatch in the stopped forward
    pass.  ``center_fn`` is called exactly once per microbatch in the
    differentiated pass; it must not evaluate connected-state amplitudes.
    Recomputed centers are checked against the first-pass centers before the
    result is returned.

    The returned type, Adam state, and Adam update are the same objects used
    by :mod:`physical_sampled_adam`, preserving optimizer checkpoint
    compatibility.  Samples, chains, and amplitude caches are not retained.
    """

    concrete_batches = tuple(batches)
    if not concrete_batches:
        raise ValueError("batches must contain at least one microbatch")
    if not callable(hpsi_fn):
        raise TypeError("hpsi_fn must be callable")
    if not callable(center_fn):
        raise TypeError("center_fn must be callable")
    if not _finite_pytree(params):
        raise FloatingPointError("sampled VMC parameters are non-finite")
    _real_pytree(params, name="params")

    centers_by_batch: list[jax.Array] = []
    locals_by_batch: list[jax.Array] = []
    sizes: list[int] = []
    for batch in concrete_batches:
        centers, actions = _checked_hpsi(params, batch, hpsi_fn)
        centers_by_batch.append(centers)
        locals_by_batch.append(jax.lax.stop_gradient(actions / centers))
        sizes.append(int(centers.shape[0]))

    centers_all = jnp.concatenate(centers_by_batch)
    local_all = jnp.concatenate(locals_by_batch)
    local_host = np.asarray(local_all)
    center_host = np.asarray(centers_all)
    nonfinite_local_count = int(np.count_nonzero(~np.isfinite(local_host)))
    if not np.all(np.isfinite(center_host)) or nonfinite_local_count:
        raise FloatingPointError(
            "sampled VMC encountered non-finite centers or local energies"
        )
    center_magnitudes = np.abs(center_host)
    minimum_center_magnitude = float(np.min(center_magnitudes))
    if minimum_center_magnitude <= 0.0:
        raise FloatingPointError("sampled VMC encountered a zero center amplitude")

    shared_mean = jax.lax.stop_gradient(jnp.mean(local_all))
    sample_count = int(sum(sizes))
    gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
    surrogate_values: list[float] = []
    for batch_index, (batch, expected_centers, local, size) in enumerate(
        zip(
            concrete_batches,
            centers_by_batch,
            locals_by_batch,
            sizes,
            strict=True,
        )
    ):

        def centered_surrogate(candidate: object) -> tuple[jax.Array, jax.Array]:
            centers = _checked_centers(
                candidate,
                batch,
                center_fn,
                expected_size=size,
            )
            logs = jnp.log(centers.astype(jnp.complex128))
            value = 2.0 * jnp.real(
                jnp.sum(jnp.conj(logs) * (local - shared_mean)) / sample_count
            )
            return value, centers

        (surrogate, recomputed_centers), partial = jax.value_and_grad(
            centered_surrogate, has_aux=True
        )(params)
        surrogate_values.append(float(np.asarray(surrogate)))
        recomputed_host = np.asarray(recomputed_centers)
        if not np.all(np.isfinite(recomputed_host)):
            raise FloatingPointError(
                "center-only gradient encountered non-finite centers"
            )
        if not np.allclose(
            recomputed_host,
            np.asarray(expected_centers),
            rtol=CENTER_CONSISTENCY_RTOL,
            atol=CENTER_CONSISTENCY_ATOL,
        ):
            maximum_error = float(
                np.max(np.abs(recomputed_host - np.asarray(expected_centers)))
            )
            raise ValueError(
                "center_fn disagrees with hpsi_fn centers in microbatch "
                f"{batch_index}: maximum absolute error {maximum_error:.6g}"
            )
        gradient = jax.tree_util.tree_map(jnp.add, gradient, partial)

    mean_host = complex(np.asarray(shared_mean))
    variance = float(np.mean(np.abs(local_host - mean_host) ** 2).real)
    energy = float(mean_host.real)
    energy_imaginary_abs = float(abs(mean_host.imag))
    maximum_local_energy_magnitude = float(np.max(np.abs(local_host)))
    gradient_norm = _pytree_l2_norm(gradient)
    scalar_values = (
        energy,
        variance,
        energy_imaginary_abs,
        minimum_center_magnitude,
        maximum_local_energy_magnitude,
        gradient_norm,
        *surrogate_values,
    )
    if not all(math.isfinite(value) for value in scalar_values) or not _finite_pytree(
        gradient
    ):
        raise FloatingPointError("sampled VMC produced non-finite diagnostics")
    _real_pytree(gradient, name="gradient")

    return SampledVMCGradient(
        energy=energy,
        variance=variance,
        energy_imaginary_abs=energy_imaginary_abs,
        sample_count=sample_count,
        microbatch_count=len(concrete_batches),
        maximum_microbatch_size=max(sizes),
        minimum_center_magnitude=minimum_center_magnitude,
        maximum_local_energy_magnitude=maximum_local_energy_magnitude,
        nonfinite_local_count=nonfinite_local_count,
        gradient=gradient,
        gradient_norm=gradient_norm,
    )


__all__ = [
    "CENTER_CONSISTENCY_ATOL",
    "CENTER_CONSISTENCY_RTOL",
    "MAX_PHYSICAL_MICROBATCH_SIZE",
    "PhysicalBatchCenter",
    "PhysicalBatchHpsi",
    "SampledAdamDiagnostics",
    "SampledAdamState",
    "SampledVMCGradient",
    "initialize_sampled_adam",
    "sampled_adam_update",
    "sampled_vmc_value_and_grad",
]
