"""Finite imaginary-time targets with exact physical-state trust radii.

The routines in this module do not optimize a neural state.  They construct
the ray

``|psi(tau)> = [1 - tau (H - <H>)] |psi>``

and choose the largest admitted ``tau`` from wavefunction distances.  This
separates the physical projection supplied by the Hamiltonian from the later
question of whether a normalized autoregressive model can compress that
projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class ExactStateDistance:
    """Exact distances between two normalized finite-support wavefunctions."""

    normalized_overlap: float
    fubini_study_distance: float
    forward_kl: float
    reverse_kl: float
    symmetric_kl: float
    importance_effective_sample_fraction: float


@dataclass(frozen=True)
class ImaginaryTimeEulerStep:
    """One exact Euler target selected by finite physical-state radii."""

    target: np.ndarray
    tau: float
    energy_shift: float
    distance: ExactStateDistance
    maximum_tau: float
    maximum_fubini_study_distance: float | None
    maximum_symmetric_kl: float | None
    minimum_importance_effective_sample_fraction: float | None
    bisection_iterations: int


def _positive_optional(value: Real | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    answer = float(value)
    if not math.isfinite(answer) or answer <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return answer


def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.complex128)
    if value.ndim != 1 or value.size < 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector with at least two entries")
    norm = float(np.vdot(value, value).real)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have positive finite norm")
    return value / math.sqrt(norm)


def exact_state_distance(
    old_state: np.ndarray,
    new_state: np.ndarray,
) -> ExactStateDistance:
    """Return exact FS, Born-KL, and importance-ESS distances."""

    old = _normalized(old_state, "old_state")
    new = _normalized(new_state, "new_state")
    if old.shape != new.shape:
        raise ValueError("old_state and new_state must have the same shape")
    overlap = float(np.clip(abs(np.vdot(old, new)), 0.0, 1.0))
    with np.errstate(under="ignore"):
        p = np.abs(old) ** 2
        q = np.abs(new) ** 2
    old_active = p > 0.0
    new_active = q > 0.0
    if np.any(old_active & ~new_active):
        forward = math.inf
    else:
        with np.errstate(under="ignore"):
            forward = float(
                np.sum(
                    p[old_active]
                    * (np.log(p[old_active]) - np.log(q[old_active]))
                )
            )
    if np.any(new_active & ~old_active):
        reverse = math.inf
        ess = 0.0
    else:
        with np.errstate(under="ignore"):
            reverse = float(
                np.sum(
                    q[new_active]
                    * (np.log(q[new_active]) - np.log(p[new_active]))
                )
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            denominator = float(np.sum(q[old_active] ** 2 / p[old_active]))
        ess = (
            0.0
            if denominator <= 0.0 or not math.isfinite(denominator)
            else float(1.0 / denominator)
        )
    symmetric = 0.5 * (forward + reverse)
    return ExactStateDistance(
        normalized_overlap=overlap,
        fubini_study_distance=float(math.acos(overlap)),
        forward_kl=max(forward, 0.0),
        reverse_kl=max(reverse, 0.0),
        symmetric_kl=max(symmetric, 0.0),
        importance_effective_sample_fraction=float(np.clip(ess, 0.0, 1.0)),
    )


def imaginary_time_euler_target(
    state: np.ndarray,
    hamiltonian_action: np.ndarray,
    tau: Real,
    *,
    energy_shift: Real | None = None,
) -> tuple[np.ndarray, float]:
    """Construct normalized ``[1-tau(H-E)]|psi>`` and return ``E``."""

    wavefunction = _normalized(state, "state")
    action = np.asarray(hamiltonian_action, dtype=np.complex128)
    if action.shape != wavefunction.shape or not np.all(np.isfinite(action)):
        raise ValueError("hamiltonian_action must be finite and match state")
    if isinstance(tau, (bool, np.bool_)):
        raise TypeError("tau must be a real scalar")
    step = float(tau)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("tau must be nonnegative and finite")
    expectation = float(np.vdot(wavefunction, action).real)
    shift = expectation if energy_shift is None else float(energy_shift)
    if not math.isfinite(shift):
        raise ValueError("energy_shift must be finite")
    target = wavefunction - step * (action - shift * wavefunction)
    return _normalized(target, "Euler target"), shift


def select_imaginary_time_euler_step(
    state: np.ndarray,
    hamiltonian_action: np.ndarray,
    *,
    maximum_tau: Real,
    maximum_fubini_study_distance: Real | None = None,
    maximum_symmetric_kl: Real | None = None,
    minimum_importance_effective_sample_fraction: Real | None = None,
    maximum_bisection_iterations: int = 60,
) -> ImaginaryTimeEulerStep:
    """Choose the largest Euler step admitted by exact state distances."""

    upper = _positive_optional(maximum_tau, "maximum_tau")
    assert upper is not None
    fs_limit = _positive_optional(
        maximum_fubini_study_distance, "maximum_fubini_study_distance"
    )
    kl_limit = _positive_optional(maximum_symmetric_kl, "maximum_symmetric_kl")
    ess_limit = _positive_optional(
        minimum_importance_effective_sample_fraction,
        "minimum_importance_effective_sample_fraction",
    )
    if ess_limit is not None and ess_limit > 1.0:
        raise ValueError("minimum ESS fraction cannot exceed one")
    if fs_limit is None and kl_limit is None and ess_limit is None:
        raise ValueError("at least one physical-state trust criterion is required")
    if (
        isinstance(maximum_bisection_iterations, (bool, np.bool_))
        or not isinstance(maximum_bisection_iterations, Integral)
        or maximum_bisection_iterations < 1
    ):
        raise ValueError("maximum_bisection_iterations must be a positive integer")
    old = _normalized(state, "state")
    action = np.asarray(hamiltonian_action, dtype=np.complex128)
    if action.shape != old.shape:
        raise ValueError("hamiltonian_action must match state")
    energy = float(np.vdot(old, action).real)

    def evaluate(tau: float) -> tuple[np.ndarray, ExactStateDistance]:
        target, _ = imaginary_time_euler_target(
            old, action, tau, energy_shift=energy
        )
        return target, exact_state_distance(old, target)

    def passes(distance: ExactStateDistance) -> bool:
        return bool(
            (fs_limit is None or distance.fubini_study_distance <= fs_limit)
            and (kl_limit is None or distance.symmetric_kl <= kl_limit)
            and (
                ess_limit is None
                or distance.importance_effective_sample_fraction >= ess_limit
            )
        )

    target, distance = evaluate(upper)
    iterations = 0
    accepted_tau = upper
    if not passes(distance):
        lower = 0.0
        upper_bound = upper
        target, distance = evaluate(0.0)
        for iterations in range(1, int(maximum_bisection_iterations) + 1):
            midpoint = 0.5 * (lower + upper_bound)
            candidate, candidate_distance = evaluate(midpoint)
            if passes(candidate_distance):
                lower = midpoint
                target = candidate
                distance = candidate_distance
            else:
                upper_bound = midpoint
        accepted_tau = lower
    return ImaginaryTimeEulerStep(
        target=target,
        tau=float(accepted_tau),
        energy_shift=energy,
        distance=distance,
        maximum_tau=float(upper),
        maximum_fubini_study_distance=fs_limit,
        maximum_symmetric_kl=kl_limit,
        minimum_importance_effective_sample_fraction=ess_limit,
        bisection_iterations=iterations,
    )


__all__ = [
    "ExactStateDistance",
    "ImaginaryTimeEulerStep",
    "exact_state_distance",
    "imaginary_time_euler_target",
    "select_imaginary_time_euler_step",
]
