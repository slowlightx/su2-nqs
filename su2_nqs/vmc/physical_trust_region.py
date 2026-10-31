"""Sampled finite-distance trust regions for physical wavefunction updates.

Unlike a parameter-norm cap, these bounds compare the normalized old and new
physical states on configurations drawn from the old Born distribution.  The
Fubini--Study distance uses an importance estimate of the normalized overlap;
the forward and reverse KL estimates use the same amplitude ratios.  A scalar
bisection shortens, but never rotates, an already proposed update direction.

This module complements the local Fisher-radius bound in
``physical_sampled_sr``.  The Fisher metric controls the infinitesimal step;
the routines here verify the realized nonlinear state displacement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import jax
import numpy as np


PhysicalBatchCenter = Callable[[object, Any], jax.Array]
PhysicalBatchLogAmplitude = Callable[[object, Any], jax.Array]


@dataclass(frozen=True)
class SampledPhysicalStateDistance:
    """Importance-sampled distance between two normalized physical states."""

    sample_count: int
    normalized_overlap: float
    fubini_study_distance: float
    forward_kl: float
    reverse_kl: float
    symmetric_kl: float
    importance_effective_sample_fraction: float
    log_norm_ratio: float


@dataclass(frozen=True)
class BoundedPhysicalStateUpdate:
    """One proposed direction shortened to finite physical-state radii."""

    direction: object
    scale: float
    distance: SampledPhysicalStateDistance
    maximum_fubini_study_distance: float | None
    maximum_symmetric_kl: float | None
    bisection_iterations: int


def _optional_positive(value: Real | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    answer = float(value)
    if not math.isfinite(answer) or answer <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return answer


def _amplitude_values(
    params: object,
    samples: Any,
    *,
    center_fn: PhysicalBatchCenter | None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None,
) -> tuple[np.ndarray, bool]:
    if (center_fn is None) == (log_amplitude_fn is None):
        raise ValueError("supply exactly one of center_fn or log_amplitude_fn")
    if center_fn is not None:
        values = np.asarray(center_fn(params, samples), dtype=np.complex128)
        is_log = False
        if np.any(np.abs(values) == 0.0):
            raise FloatingPointError("physical trust region encountered a zero amplitude")
    else:
        values = np.asarray(log_amplitude_fn(params, samples), dtype=np.complex128)
        is_log = True
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("amplitude callback must return at least two scalar values")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("amplitude callback returned non-finite values")
    return values, is_log


def sampled_physical_state_distance(
    old_params: object,
    new_params: object,
    samples: Any,
    *,
    center_fn: PhysicalBatchCenter | None = None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
) -> SampledPhysicalStateDistance:
    """Estimate FS and two-sided KL distances on old-state Born samples."""

    if jax.tree_util.tree_structure(old_params) != jax.tree_util.tree_structure(
        new_params
    ):
        raise ValueError("old_params and new_params must share a pytree structure")
    old_values, old_is_log = _amplitude_values(
        old_params,
        samples,
        center_fn=center_fn,
        log_amplitude_fn=log_amplitude_fn,
    )
    new_values, new_is_log = _amplitude_values(
        new_params,
        samples,
        center_fn=center_fn,
        log_amplitude_fn=log_amplitude_fn,
    )
    if old_is_log != new_is_log or old_values.shape != new_values.shape:
        raise RuntimeError("amplitude callback changed representation or shape")
    if old_is_log:
        delta = new_values - old_values
    else:
        delta = np.log(new_values / old_values)
    maximum_real = float(np.max(delta.real))
    ratios = np.exp(delta - maximum_real)
    weights = np.abs(ratios) ** 2
    mean_weight = float(np.mean(weights))
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise FloatingPointError("physical trust-region importance weights vanished")
    normalized_overlap = float(
        np.clip(abs(np.mean(ratios)) / math.sqrt(mean_weight), 0.0, 1.0)
    )
    fs_distance = float(math.acos(normalized_overlap))
    log_weights = 2.0 * delta.real
    log_mean_weight = 2.0 * maximum_real + math.log(mean_weight)
    forward_kl = float(max(log_mean_weight - np.mean(log_weights), 0.0))
    normalized_weights = weights / mean_weight
    log_normalized_weights = log_weights - log_mean_weight
    reverse_kl = float(
        max(np.mean(normalized_weights * log_normalized_weights), 0.0)
    )
    symmetric_kl = 0.5 * (forward_kl + reverse_kl)
    effective_fraction = float(
        np.sum(weights) ** 2 / (len(weights) * np.sum(weights**2))
    )
    result = SampledPhysicalStateDistance(
        sample_count=int(len(weights)),
        normalized_overlap=normalized_overlap,
        fubini_study_distance=fs_distance,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
        symmetric_kl=symmetric_kl,
        importance_effective_sample_fraction=effective_fraction,
        log_norm_ratio=log_mean_weight,
    )
    if not all(math.isfinite(value) for value in result.__dict__.values()):
        raise FloatingPointError("physical trust-region distance is non-finite")
    return result


def bound_physical_state_update(
    params: object,
    direction: object,
    samples: Any,
    *,
    center_fn: PhysicalBatchCenter | None = None,
    log_amplitude_fn: PhysicalBatchLogAmplitude | None = None,
    maximum_fubini_study_distance: float | None = None,
    maximum_symmetric_kl: float | None = None,
    maximum_bisection_iterations: int = 40,
) -> BoundedPhysicalStateUpdate:
    """Shorten ``params - direction`` until sampled physical radii are met."""

    fs_limit = _optional_positive(
        maximum_fubini_study_distance, "maximum_fubini_study_distance"
    )
    kl_limit = _optional_positive(maximum_symmetric_kl, "maximum_symmetric_kl")
    if fs_limit is None and kl_limit is None:
        raise ValueError("at least one physical-state trust radius is required")
    if (
        isinstance(maximum_bisection_iterations, (bool, np.bool_))
        or not isinstance(maximum_bisection_iterations, Integral)
        or maximum_bisection_iterations < 1
    ):
        raise ValueError("maximum_bisection_iterations must be a positive integer")
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(direction):
        raise ValueError("params and direction must share a pytree structure")

    def candidate(scale: float) -> object:
        return jax.tree_util.tree_map(
            lambda value, step: value - scale * step, params, direction
        )

    def evaluate(scale: float) -> SampledPhysicalStateDistance:
        return sampled_physical_state_distance(
            params,
            candidate(scale),
            samples,
            center_fn=center_fn,
            log_amplitude_fn=log_amplitude_fn,
        )

    def passes(distance: SampledPhysicalStateDistance) -> bool:
        return bool(
            (fs_limit is None or distance.fubini_study_distance <= fs_limit)
            and (kl_limit is None or distance.symmetric_kl <= kl_limit)
        )

    full = evaluate(1.0)
    iterations = 0
    if passes(full):
        scale = 1.0
        accepted = full
    else:
        lower, upper = 0.0, 1.0
        accepted = evaluate(0.0)
        for iterations in range(1, int(maximum_bisection_iterations) + 1):
            midpoint = 0.5 * (lower + upper)
            distance = evaluate(midpoint)
            if passes(distance):
                lower, accepted = midpoint, distance
            else:
                upper = midpoint
        scale = lower
    bounded = jax.tree_util.tree_map(lambda step: scale * step, direction)
    return BoundedPhysicalStateUpdate(
        direction=bounded,
        scale=float(scale),
        distance=accepted,
        maximum_fubini_study_distance=fs_limit,
        maximum_symmetric_kl=kl_limit,
        bisection_iterations=iterations,
    )


__all__ = [
    "BoundedPhysicalStateUpdate",
    "SampledPhysicalStateDistance",
    "bound_physical_state_update",
    "sampled_physical_state_distance",
]
