"""Memory-bounded sampled Adam primitives for physical-input wavefunctions.

The functions in this module deliberately own neither a sampler nor a model.
Callers provide already sampled physical configurations through arbitrary
microbatch payloads and an ``hpsi_fn(params, batch)`` callback returning the
wavefunction centers and Hamiltonian actions on that payload.  This keeps the
core usable for spin and Hubbard models, including spatially projected
amplitudes, without retaining Markov chains or amplitude caches across an
optimizer update.

The VMC gradient uses one global local-energy mean across all microbatches.
A first, non-differentiated pass constructs that shared mean; a second pass
accumulates weighted log-derivative gradients.  Consequently splitting a
fixed sample batch does not introduce the extra batch-local centering bias
that would result from averaging independent VMC losses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


PhysicalBatchHpsi = Callable[
    [object, Any], tuple[jax.Array, jax.Array]
]

# The first scalable physical-spin readiness gate was sealed at batch eight.
# Keep the training primitive inside that validated memory envelope until a
# larger static shape receives its own resource gate.
MAX_PHYSICAL_MICROBATCH_SIZE = 8


@dataclass(frozen=True)
class SampledVMCGradient:
    """One globally centered sampled-VMC gradient and its safety ledger."""

    energy: float
    variance: float
    energy_imaginary_abs: float
    sample_count: int
    microbatch_count: int
    maximum_microbatch_size: int
    minimum_center_magnitude: float
    maximum_local_energy_magnitude: float
    nonfinite_local_count: int
    gradient: object
    gradient_norm: float


class SampledAdamState(NamedTuple):
    """Minimal pickle/checkpoint-safe Adam state for a parameter pytree."""

    step: int
    first_moment: object
    second_moment: object


@dataclass(frozen=True)
class SampledAdamDiagnostics:
    """Finite and relative-size diagnostics for one accepted Adam update."""

    step: int
    gradient_norm: float
    parameter_norm_before: float
    update_norm: float
    relative_update_norm: float
    parameter_norm_after: float
    max_relative_update: float
    all_finite: bool


def _checked_hpsi(
    params: object,
    batch: Any,
    hpsi_fn: PhysicalBatchHpsi,
) -> tuple[jax.Array, jax.Array]:
    centers, actions = hpsi_fn(params, batch)
    centers = jnp.asarray(centers)
    actions = jnp.asarray(actions)
    if centers.ndim != 1 or centers.shape[0] < 1:
        raise ValueError("hpsi_fn must return a nonempty rank-one center array")
    if actions.shape != centers.shape:
        raise ValueError("hpsi_fn centers and actions must have equal shapes")
    if centers.shape[0] > MAX_PHYSICAL_MICROBATCH_SIZE:
        raise ValueError(
            "physical sampled Adam microbatches are limited to "
            f"{MAX_PHYSICAL_MICROBATCH_SIZE} samples"
        )
    return centers, actions


def _tree_structure_matches(*trees: object) -> bool:
    structures = [jax.tree_util.tree_structure(tree) for tree in trees]
    return all(structure == structures[0] for structure in structures[1:])


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


def sampled_vmc_value_and_grad(
    params: object,
    batches: Sequence[Any],
    *,
    hpsi_fn: PhysicalBatchHpsi,
) -> SampledVMCGradient:
    """Return a globally centered VMC gradient over physical microbatches.

    ``batches`` may contain any workflow-owned payload understood by
    ``hpsi_fn``.  Each callback result is limited to eight samples.  The
    function performs two Hpsi passes: the first obtains a shared stopped
    local-energy mean and the second differentiates each microbatch against
    that same mean.  Microbatch gradients are weighted by their sample counts,
    making the result algebraically equal to the ordinary full-batch VMC
    surrogate evaluated on the concatenated samples.

    Non-finite centers, local energies, summary scalars, or gradients fail
    closed.  The returned object contains no samples, Markov-chain state, or
    amplitude cache.
    """

    concrete_batches = tuple(batches)
    if not concrete_batches:
        raise ValueError("batches must contain at least one microbatch")
    if not _finite_pytree(params):
        raise FloatingPointError("sampled VMC parameters are non-finite")
    _real_pytree(params, name="params")

    centers_by_batch: list[jax.Array] = []
    locals_by_batch: list[jax.Array] = []
    sizes: list[int] = []
    for batch in concrete_batches:
        centers, actions = _checked_hpsi(params, batch, hpsi_fn)
        centers_by_batch.append(centers)
        locals_by_batch.append(actions / centers)
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
    zero_gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
    gradient = zero_gradient
    for batch in concrete_batches:

        def centered_surrogate(candidate: object) -> jax.Array:
            centers, actions = _checked_hpsi(candidate, batch, hpsi_fn)
            local = jax.lax.stop_gradient(actions / centers)
            logs = jnp.log(centers.astype(jnp.complex128))
            # Divide by the *global* sample count.  Summing the resulting
            # microbatch gradients is therefore exactly the concatenated-
            # batch surrogate, including unequal final microbatches.
            return 2.0 * jnp.real(
                jnp.sum(jnp.conj(logs) * (local - shared_mean)) / sample_count
            )

        partial = jax.grad(centered_surrogate)(params)
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


def initialize_sampled_adam(params: object) -> SampledAdamState:
    """Initialize a checkpointable Adam state matching ``params``."""

    if not _finite_pytree(params):
        raise FloatingPointError("cannot initialize Adam from non-finite parameters")
    _real_pytree(params, name="params")
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return SampledAdamState(step=0, first_moment=zeros, second_moment=zeros)


def sampled_adam_update(
    params: object,
    gradient: object,
    state: SampledAdamState,
    *,
    learning_rate: float,
    max_relative_update: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> tuple[object, SampledAdamState, SampledAdamDiagnostics]:
    """Apply one fail-closed Adam update to a real parameter pytree.

    ``max_relative_update`` bounds ``||delta theta||_2 / ||theta||_2``.
    Exceeding the bound raises before a candidate or optimizer state is
    returned, so a workflow cannot silently commit a rejected step.  Optimizer
    hyperparameters belong in the caller's checkpoint/run identity; this
    compact state stores only trajectory state.
    """

    if not isinstance(state, SampledAdamState):
        raise TypeError("state must be a SampledAdamState")
    if isinstance(state.step, bool) or not isinstance(state.step, int) or state.step < 0:
        raise ValueError("Adam state step must be a non-negative integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("max_relative_update", max_relative_update),
        ("epsilon", epsilon),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("beta1", beta1), ("beta2", beta2)):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
        ):
            raise ValueError(f"{name} must lie in [0, 1)")
    if not _tree_structure_matches(
        params, gradient, state.first_moment, state.second_moment
    ):
        raise ValueError("parameters, gradient, and Adam moments must share a pytree")
    _real_pytree(params, name="params")
    _real_pytree(gradient, name="gradient")
    if not _finite_pytree(
        (params, gradient, state.first_moment, state.second_moment)
    ):
        raise FloatingPointError("Adam input contains non-finite values")

    next_step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda moment, grad: beta1 * moment + (1.0 - beta1) * grad,
        state.first_moment,
        gradient,
    )
    second = jax.tree_util.tree_map(
        lambda moment, grad: beta2 * moment + (1.0 - beta2) * grad * grad,
        state.second_moment,
        gradient,
    )
    first_correction = 1.0 - beta1**next_step
    second_correction = 1.0 - beta2**next_step
    update = jax.tree_util.tree_map(
        lambda first_value, second_value: -learning_rate
        * (first_value / first_correction)
        / (jnp.sqrt(second_value / second_correction) + epsilon),
        first,
        second,
    )
    candidate = jax.tree_util.tree_map(jnp.add, params, update)

    gradient_norm = _pytree_l2_norm(gradient)
    parameter_norm_before = _pytree_l2_norm(params)
    update_norm = _pytree_l2_norm(update)
    denominator = max(parameter_norm_before, np.finfo(np.float64).tiny)
    relative_update_norm = update_norm / denominator
    parameter_norm_after = _pytree_l2_norm(candidate)
    diagnostics_values = (
        gradient_norm,
        parameter_norm_before,
        update_norm,
        relative_update_norm,
        parameter_norm_after,
    )
    if not all(math.isfinite(value) for value in diagnostics_values) or not _finite_pytree(
        (candidate, first, second)
    ):
        raise FloatingPointError("Adam update produced non-finite state")
    if relative_update_norm > max_relative_update:
        raise FloatingPointError(
            "Adam relative update exceeds the registered maximum: "
            f"{relative_update_norm:.6g} > {max_relative_update:.6g}"
        )

    next_state = SampledAdamState(
        step=next_step,
        first_moment=first,
        second_moment=second,
    )
    diagnostics = SampledAdamDiagnostics(
        step=next_step,
        gradient_norm=gradient_norm,
        parameter_norm_before=parameter_norm_before,
        update_norm=update_norm,
        relative_update_norm=relative_update_norm,
        parameter_norm_after=parameter_norm_after,
        max_relative_update=float(max_relative_update),
        all_finite=True,
    )
    return candidate, next_state, diagnostics


__all__ = [
    "MAX_PHYSICAL_MICROBATCH_SIZE",
    "PhysicalBatchHpsi",
    "SampledAdamDiagnostics",
    "SampledAdamState",
    "SampledVMCGradient",
    "initialize_sampled_adam",
    "sampled_adam_update",
    "sampled_vmc_value_and_grad",
]
