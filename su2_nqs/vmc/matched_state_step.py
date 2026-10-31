"""Fixed-work finite physical-distance matching along one frozen direction.

This controller changes only the scalar step length, never its direction.
The positive fixed-Q quadrature uses physical amplitudes, including nodes.
It calls ``candidate`` exactly once at max_alpha plus bisection_steps times,
then returns the cached closest measured state whose FS angle does not exceed
the target. A failed bracket is reported, not relabeled a matched step.

FS need not be globally monotone in the step length. The bounded search finds
a sampled bracket when possible; it does NOT prove a target is unreachable
elsewhere on a nonmonotone path. No energy or optimization outcome is read.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .cutset_training import weighted_state_distance


@dataclass(frozen=True)
class MatchedStateStep:
    """A cached candidate and auditable fixed-work scalar-search result.

    ``amplitudes`` must be reused rather than calling candidate(alpha) again.
    ``matched`` uses an explicitly declared radius tolerance, not merely a
    successful cap. ``target_bracketed`` concerns evaluated points only.
    """

    alpha: float
    amplitudes: np.ndarray
    distance: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    matched: bool
    target_bracketed: bool
    status: str
    target_radius: float
    radius_shortfall: float
    matching_tolerance: float
    evaluation_count: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready audit metadata, deliberately excluding amplitudes."""
        return dict(alpha=self.alpha, distance=self.distance, records=list(self.records),
                    matched=self.matched, target_bracketed=self.target_bracketed,
                    status=self.status, target_radius=self.target_radius,
                    radius_shortfall=self.radius_shortfall,
                    matching_tolerance=self.matching_tolerance,
                    evaluation_count=self.evaluation_count)


def _real(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _stable_distance(old: np.ndarray, new: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    """Same weighted-Gram FS geometry, stable also for exact ray changes.

    atan2(perpendicular norm, parallel norm) avoids arccos roundoff of order
    sqrt(machine-epsilon) at an identical physical ray. Other diagnostics are
    inherited unchanged from the existing node-safe fixed-Q helper.
    """
    result = weighted_state_distance(old, new, q)
    before, after = old/np.sqrt(q), new/np.sqrt(q)
    norm = float(np.vdot(before, before).real)
    overlap = np.vdot(before, after)
    perpendicular = after-before*(overlap/norm)
    parallel_norm = abs(overlap)/np.sqrt(norm)
    result['fs'] = float(np.arctan2(np.linalg.norm(perpendicular), parallel_norm))
    result['fs_evaluation'] = 'weighted-perpendicular atan2; same Gram geometry'
    return result


def matched_state_step(
    candidate: Callable[[float], np.ndarray],
    old: np.ndarray,
    q: np.ndarray,
    target_radius: float,
    max_alpha: float,
    bisection_steps: int,
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-12,
    zero_distance_tolerance: float = 64*np.finfo(np.float64).eps,
) -> MatchedStateStep:
    """Match a fixed FS angle with exactly ``1+bisection_steps`` evaluations.

    The caller guarantees candidate(0) is the supplied parent; zero is not
    reevaluated. Candidate shapes must match old. Nonfinite/zero-norm returned
    candidates are logged as inadmissible and shorten the upper bracket;
    actual callback exceptions propagate. Null/gauge paths return alpha=0.

    Even when max_alpha is below target, perform the same fixed work. An
    interior probe can establish a bracket on a nonmonotone path. The result
    is selected across ALL evaluated points, never just the final midpoint.
    ``matched=False`` must not be presented as an equal-distance comparison.
    """
    target = _real(target_radius, 'target_radius', positive=True)
    maximum = _real(max_alpha, 'max_alpha', positive=True)
    relative = _real(relative_tolerance, 'relative_tolerance')
    absolute = _real(absolute_tolerance, 'absolute_tolerance')
    zero_tolerance = _real(zero_distance_tolerance, 'zero_distance_tolerance')
    tolerance = max(absolute, relative*target)
    if target >= np.pi/2 or target <= zero_tolerance or tolerance >= target:
        raise ValueError('need resolvable 0 < target < pi/2 and matching tolerance smaller than target')
    if isinstance(bisection_steps, bool) or not isinstance(bisection_steps, Integral) or bisection_steps < 0:
        raise ValueError('bisection_steps must be a nonnegative integer')
    if not callable(candidate):
        raise TypeError('candidate must be callable')
    parent = np.array(old, dtype=np.complex128, copy=True)
    proposal = np.array(q, dtype=np.float64, copy=True)
    parent_distance = _stable_distance(parent, parent, proposal)
    if not np.isfinite(parent_distance['fs']) or not np.isfinite(parent_distance['norm_ratio']):
        raise FloatingPointError('parent quadrature overflows or is not finite')
    parent_distance['fs'] = 0.
    best_alpha, best_values, best_distance = 0., parent.copy(), parent_distance
    records: list[dict[str, Any]] = []
    bracketed = False
    valid_steps = 0
    nonzero_steps = 0

    def evaluate(alpha: float) -> tuple[bool, float]:
        nonlocal best_alpha, best_values, best_distance, bracketed, valid_steps, nonzero_steps
        values = np.asarray(candidate(float(alpha)), dtype=np.complex128)
        if values.shape != parent.shape:
            raise ValueError('candidate changed physical-amplitude shape')
        if not np.isfinite(values).all():
            records.append(dict(alpha=float(alpha), valid=False, within_radius=False,
                                reason='nonfinite_candidate', distance=None))
            return False, float('inf')
        if not np.any(values != 0):
            records.append(dict(alpha=float(alpha), valid=False, within_radius=False,
                                reason='zero_candidate_norm', distance=None))
            return False, float('inf')
        distance = _stable_distance(parent, values, proposal)
        radius = float(distance['fs'])
        if not np.isfinite(radius) or not np.isfinite(distance['norm_ratio']):
            records.append(dict(alpha=float(alpha), valid=False, within_radius=False,
                                reason='nonfinite_distance', distance=None))
            return False, float('inf')
        valid_steps += 1
        nonzero_steps += int(radius > zero_tolerance)
        bracketed = bracketed or radius >= target
        admissible = radius <= target
        records.append(dict(alpha=float(alpha), valid=True, within_radius=bool(admissible),
                            reason='evaluated', distance=distance))
        if admissible and radius > zero_tolerance and radius > best_distance['fs']:
            best_alpha, best_values, best_distance = float(alpha), values.copy(), distance
        return True, radius

    # The same evaluation schedule is used even for exact matches, failed
    # brackets, pure gauge directions, and invalid max-alpha proposals.
    evaluate(maximum)
    lower, upper = 0., maximum
    for _ in range(int(bisection_steps)):
        midpoint = lower+(upper-lower)/2
        finite, radius = evaluate(midpoint)
        if finite and radius <= target:
            lower = midpoint
        else:
            upper = midpoint
    shortfall = max(0., target-float(best_distance['fs']))
    matched = bool(best_alpha > 0 and shortfall <= tolerance)
    if matched:
        status = 'matched'
    elif not valid_steps:
        status = 'no_finite_candidate_in_evaluations'
    elif not nonzero_steps:
        status = 'zero_or_gauge_on_evaluated_points'
    elif not bracketed:
        status = 'target_not_bracketed_by_evaluated_points'
    else:
        status = 'radius_below_target_after_fixed_budget'
    return MatchedStateStep(best_alpha, best_values, best_distance, tuple(records), matched,
                            bool(bracketed), status, target, shortfall, tolerance, len(records))


__all__ = ['MatchedStateStep', 'matched_state_step']
