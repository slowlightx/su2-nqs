"""Opt-in, bounded SR scheduling from independent training monitors.

This is a controller, not an energy estimator or a sampler. A monitor consumes
fresh full-H/JVP data for ONE frozen block/direction. Its data are training
feedback and can never be advertised as untouched terminal precision. Decisions
apply only to the NEXT block: no endpoint search, rollback, repeated line search,
automatic allocation, or change of Hilbert space is implemented here.

Two-sigma tests below are operational scheduling heuristics, not simultaneous
confidence bounds under repeated monitoring. A separate terminal audit remains
mandatory. A small energy change alone is not evidence of a noisy gradient or
of stationarity; sample growth also needs an unresolved independent slope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral
from typing import Any, Callable

import numpy as np


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class AdaptiveSRPolicy:
    """Predeclare finite control/sample ranges and a maximum number of looks.

    ``fs_radius`` bounds both the tangent proposal and the independent finite
    guard's nominal upper distance. Tangent control alone is not a certificate
    of realized nonlinear state distance. Apply ``select_controlled_finite_grid``
    before committing any proposal or optimizer counter.
    """
    fit_levels: tuple[int, ...]
    initial_fs_radius: float = .001
    minimum_fs_radius: float = .00025
    maximum_fs_radius: float = .003
    initial_damping: float = .05
    minimum_damping: float = .0125
    maximum_damping: float = .8
    maximum_monitor_looks: int = 8
    success_patience: int = 2
    noise_patience: int = 2
    confidence_multiplier: float = 2.
    radius_growth: float = 1.25
    radius_shrink: float = .5
    damping_growth: float = 2.
    damping_shrink: float = .8
    reliable_gain_ratio: float = .75

    def __post_init__(self) -> None:
        if not isinstance(self.fit_levels, tuple) or not self.fit_levels:
            raise ValueError('an explicit ordered tuple of training record levels is required')
        for n in self.fit_levels:
            _integer(n, 'fit_records', 2)
        if any(a >= b for a, b in zip(self.fit_levels, self.fit_levels[1:])):
            raise ValueError('fit levels must be strictly increasing')
        for name in ('maximum_monitor_looks', 'success_patience', 'noise_patience'):
            _integer(getattr(self, name), name)
        for name in ('initial_fs_radius', 'minimum_fs_radius', 'maximum_fs_radius',
                     'initial_damping', 'minimum_damping', 'maximum_damping',
                     'confidence_multiplier', 'radius_growth', 'radius_shrink',
                     'damping_growth', 'damping_shrink', 'reliable_gain_ratio'):
            _positive(getattr(self, name), name)
        if not (self.minimum_fs_radius <= self.initial_fs_radius <= self.maximum_fs_radius < 1.
                and self.minimum_damping <= self.initial_damping <= self.maximum_damping
                and self.radius_growth > 1 and self.damping_growth > 1
                and self.radius_shrink < 1 and self.damping_shrink < 1
                and self.reliable_gain_ratio <= 1):
            raise ValueError('invalid bounded SR schedule')


@dataclass(frozen=True)
class AdaptiveSRState:
    """Serializable controller state; checkpoint it with parameters and RNG."""
    fs_radius: float
    damping: float
    fit_records: int
    looks: int = 0
    good_streak: int = 0
    noise_streak: int = 0
    last_completed_step: int = 0
    last_endpoint_sha256: str = ''
    consumed_panel_ids: tuple[str, ...] = ()
    consumed_monitor_seeds: tuple[int, ...] = ()

    @classmethod
    def initial(cls, policy: AdaptiveSRPolicy) -> AdaptiveSRState:
        """Start from registered values, not the last terminal audit's optimum."""
        return cls(policy.initial_fs_radius, policy.initial_damping, policy.fit_levels[0])


@dataclass(frozen=True)
class SRMonitor:
    """Independent block endpoint differences and one frozen-direction slope.

    Energies, variance differences and their SEs must use consistent units.
    ``predicted_decrease`` is positive for predicted descent and must include
    the ACTUALLY accepted finite fractions, with the same per-site/total units
    as the energy differences. It is not the unscaled raw fit prediction. The optional
    slope/SE must come from independent full-H/JVP data, not fit gradients.
    Hashes identify the actual physical states; the caller owns raw-product,
    seed, support/coverage, full-H and finite-FS verification.
    """
    panel_id: str
    source_role: str
    sample_seed: int
    statistics_seed: int
    completed_step: int
    parent_sha256: str
    endpoint_sha256: str
    delta_energy: float
    energy_se: float
    delta_variance: float
    variance_se: float
    predicted_decrease: float
    geometry_ok: bool
    coverage_ok: bool
    directional_slope: float | None = None
    directional_slope_se: float | None = None


@dataclass(frozen=True)
class SRResourceEnvelope:
    """Remaining TRAINING budget after reserving monitoring and terminal audit.

    The memory expression is only a necessary Jacobian-storage check, not a
    measured peak guarantee. Record and wall budgets include all remaining
    updates; increased compilation cost belongs in ``other_seconds_per_step``.
    A refusal to increase samples never grants another GPU allocation.
    """
    parameter_count: int
    maximum_device_bytes: int
    remaining_updates: int
    remaining_fit_H_records: int
    remaining_training_seconds: float
    seconds_per_fit_record: float
    other_seconds_per_step: float
    reserved_device_bytes: int = 3 * 1024**3

    def permits(self, records: int) -> bool:
        for name in ('parameter_count', 'maximum_device_bytes', 'remaining_updates'):
            _integer(getattr(self, name), name)
        for name in ('remaining_fit_H_records', 'reserved_device_bytes'):
            _integer(getattr(self, name), name, 0)
        for name in ('remaining_training_seconds', 'seconds_per_fit_record', 'other_seconds_per_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError('finite nonnegative measured budget values required')
        _positive(self.seconds_per_fit_record, 'seconds_per_fit_record')
        return bool(records * self.parameter_count * 8 + self.reserved_device_bytes <= self.maximum_device_bytes
                    and records * self.remaining_updates <= self.remaining_fit_H_records
                    and (records * self.seconds_per_fit_record + self.other_seconds_per_step)
                    * self.remaining_updates <= self.remaining_training_seconds)


def validate_control_state(policy: AdaptiveSRPolicy, state: AdaptiveSRState) -> None:
    """Reject damaged/out-of-policy controls before constructing an SR solve."""
    _positive(state.fs_radius, 'fs_radius'); _positive(state.damping, 'damping')
    _integer(state.fit_records, 'fit_records', 2)
    if not (policy.minimum_fs_radius <= state.fs_radius <= policy.maximum_fs_radius
            and policy.minimum_damping <= state.damping <= policy.maximum_damping
            and state.fit_records in policy.fit_levels):
        raise ValueError('controller state outside the preregistered envelope')
    for name in ('looks', 'good_streak', 'noise_streak', 'last_completed_step'):
        _integer(getattr(state, name), name, 0)
    if (state.looks > policy.maximum_monitor_looks
            or len(state.consumed_panel_ids) != state.looks
            or len(set(state.consumed_panel_ids)) != state.looks
            or len(state.consumed_monitor_seeds) != 2 * state.looks
            or len(set(state.consumed_monitor_seeds)) != 2 * state.looks):
        raise ValueError('monitor history/counter mismatch')
    for seed in state.consumed_monitor_seeds:
        _integer(seed, 'consumed_monitor_seed', 0)
        if seed >= 2**32:
            raise ValueError('uint32 monitor history required')
    if state.looks and (not state.last_endpoint_sha256 or state.last_completed_step == 0):
        raise ValueError('monitored history needs its endpoint and actual step')


def advance_sr_controls(
    policy: AdaptiveSRPolicy, state: AdaptiveSRState, monitor: SRMonitor,
    resources: SRResourceEnvelope, *, reserved_seeds: tuple[int, ...],
) -> tuple[AdaptiveSRState, dict]:
    """Consume a new monitoring panel once, adjusting only future controls.

    ``reserved_seeds`` must include fit, finite-calibration and final-audit
    streams. Explicitly passing it is mandatory; no audit data are accepted.
    Unresolved changes are not labeled convergence or a variational win.
    """
    validate_control_state(policy, state)
    resources.permits(state.fit_records)  # Validate even when no growth is proposed.
    if monitor.source_role != 'independent_training_monitor':
        raise ValueError('fit, geometry, interim release and terminal audit data are not monitor panels')
    if state.looks >= policy.maximum_monitor_looks:
        raise ValueError('registered monitor-look budget exhausted')
    if not isinstance(monitor.panel_id, str) or not monitor.panel_id or monitor.panel_id in state.consumed_panel_ids:
        raise ValueError('new immutable panel identity required')
    for digest in (monitor.parent_sha256, monitor.endpoint_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('physical endpoint SHA256 identity required')
    if state.last_endpoint_sha256 and monitor.parent_sha256 != state.last_endpoint_sha256:
        raise ValueError('monitor block must start at the previous monitored endpoint')
    _integer(monitor.completed_step, 'completed_step')
    if monitor.completed_step <= state.last_completed_step:
        raise ValueError('do not reuse a completed block or search endpoints')
    seeds = (monitor.sample_seed, monitor.statistics_seed)
    for seed in seeds + reserved_seeds:
        _integer(seed, 'seed', 0)
        if seed >= 2**32:
            raise ValueError('uint32 stream identity required')
    if (len(set(seeds)) != 2 or set(seeds) & (set(reserved_seeds) | set(state.consumed_monitor_seeds))):
        raise ValueError('monitor streams must be fresh and disjoint from all fit/calibration/audit streams')
    values = (monitor.delta_energy, monitor.energy_se, monitor.delta_variance,
              monitor.variance_se, monitor.predicted_decrease)
    if not all(isfinite(v) and not isinstance(v, bool) for v in values) or min(values[1], values[3], values[4]) < 0:
        raise ValueError('finite moments, nonnegative SEs and predicted decrease required')
    if type(monitor.geometry_ok) is not bool or type(monitor.coverage_ok) is not bool:
        raise ValueError('explicit verified geometry/coverage flags required')
    if (monitor.directional_slope is None) != (monitor.directional_slope_se is None):
        raise ValueError('independent directional slope and SE are an inseparable pair')
    if monitor.directional_slope is not None and (not isfinite(monitor.directional_slope)
            or not isfinite(monitor.directional_slope_se) or monitor.directional_slope_se < 0):
        raise ValueError('finite independent slope/SE required')
    k = policy.confidence_multiplier
    harmful = monitor.delta_energy - k * monitor.energy_se > 0 or monitor.delta_variance - k * monitor.variance_se > 0
    resolved_descent = monitor.delta_energy + k * monitor.energy_se < 0
    prediction_resolved = monitor.predicted_decrease > k * monitor.energy_se
    reliable = resolved_descent and prediction_resolved and (
        -monitor.delta_energy - k * monitor.energy_se >= policy.reliable_gain_ratio * monitor.predicted_decrease)
    missed_prediction = prediction_resolved and not resolved_descent
    noisy = (0 < monitor.predicted_decrease <= k * monitor.energy_se
             and abs(monitor.delta_energy) <= k * monitor.energy_se
             and monitor.directional_slope_se is not None and monitor.directional_slope_se > 0
             and abs(monitor.directional_slope) <= k * monitor.directional_slope_se)
    new = replace(state, looks=state.looks + 1, last_completed_step=monitor.completed_step,
        last_endpoint_sha256=monitor.endpoint_sha256,
        consumed_panel_ids=state.consumed_panel_ids + (monitor.panel_id,),
        consumed_monitor_seeds=state.consumed_monitor_seeds + seeds)
    if not monitor.coverage_ok:
        # More of the same data is not an automatic cure for missing tails.
        reason = 'coverage_not_verified_hold_and_diagnose'
        new = replace(new, good_streak=0, noise_streak=0)
    elif not monitor.geometry_ok or harmful or missed_prediction:
        reason = 'geometry_failure' if not monitor.geometry_ok else ('resolved_harm' if harmful else 'missed_resolved_prediction')
        new = replace(new, fs_radius=max(policy.minimum_fs_radius, state.fs_radius * policy.radius_shrink),
            damping=min(policy.maximum_damping, state.damping * policy.damping_growth), good_streak=0, noise_streak=0)
    elif reliable:
        streak = state.good_streak + 1
        new = replace(new, good_streak=streak, noise_streak=0)
        reason = 'reliable_descent_hold'
        if streak >= policy.success_patience:
            new = replace(new, fs_radius=min(policy.maximum_fs_radius, state.fs_radius * policy.radius_growth),
                damping=max(policy.minimum_damping, state.damping * policy.damping_shrink), good_streak=0)
            reason = 'reliable_descent_relax_next_block'
    elif noisy:
        streak = state.noise_streak + 1
        new = replace(new, noise_streak=streak, good_streak=0)
        reason = 'unresolved_energy_and_slope_hold'
        if streak >= policy.noise_patience:
            index = policy.fit_levels.index(state.fit_records)
            if index + 1 == len(policy.fit_levels):
                reason = 'sample_ceiling_reached_no_extra_allocation'
            elif not resources.permits(policy.fit_levels[index + 1]):
                reason = 'sample_growth_blocked_by_reserved_memory_work_or_wall'
            else:
                new = replace(new, fit_records=policy.fit_levels[index + 1], noise_streak=0)
                reason = 'increase_next_block_training_records'
    else:
        new = replace(new, good_streak=0, noise_streak=0)
        reason = 'unresolved_or_moderate_change_hold_not_stationarity'
    validate_control_state(policy, new)
    return new, dict(reason=reason, previous_controls=asdict(state), next_controls=asdict(new),
        monitor=asdict(monitor), resources=asdict(resources), applies_to_future_blocks_only=True,
        can_train_next_block=monitor.coverage_ok and resources.permits(new.fit_records),
        monitoring_is_consumed_training_feedback=True, terminal_audit_used=False,
        convergence_claimed=False, repeated_testing_confidence_claimed=False)


def sr_control_checkpoint(policy: AdaptiveSRPolicy, state: AdaptiveSRState) -> dict:
    """Seal JSON-compatible metadata for the same atomic physical checkpoint."""
    validate_control_state(policy, state)
    payload = json.loads(json.dumps(dict(version=1, policy=asdict(policy), state=asdict(state)),
        sort_keys=True, allow_nan=False))
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(payload, sha256=digest)


def restore_sr_controls(policy: AdaptiveSRPolicy, payload: dict) -> AdaptiveSRState:
    """Reject changed schedule, damaged counters or reused monitor history."""
    body = {k: v for k, v in payload.items() if k != 'sha256'}
    expected = sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (set(payload) != {'version', 'policy', 'state', 'sha256'} or payload['sha256'] != expected
            or body['version'] != 1
            or body['policy'] != json.loads(json.dumps(asdict(policy)))):
        raise ValueError('adaptive checkpoint or registered policy identity changed')
    data = dict(body['state'])
    for name in ('consumed_panel_ids', 'consumed_monitor_seeds'):
        data[name] = tuple(data[name])
    state = AdaptiveSRState(**data)
    validate_control_state(policy, state)
    return state


def make_controlled_importance_sr(
    amplitude: Callable, template: object, policy: AdaptiveSRPolicy, **kwargs: Any,
) -> Callable:
    """Bind controls to the existing native-amplitude SR without changing it.

    Each new (ridge, tangent radius) pair may compile once and is cached. All
    compilation time must be charged. Sample count changes also recompile.
    The returned update remains a PROPOSAL: independent finite-FS acceptance
    and checkpoint transaction commit are still the caller's responsibility.
    """
    from .importance_amplitude_sr import make_importance_amplitude_sr
    if {'damping', 'sr_tangent_fs_radius'} & kwargs.keys():
        raise ValueError('damping and tangent radius are owned by the bounded controller')

    @lru_cache(maxsize=2 * policy.maximum_monitor_looks + 1)
    def solver(damping: float, radius: float) -> Callable:
        return make_importance_amplitude_sr(amplitude, template, damping=damping,
            sr_tangent_fs_radius=radius, **kwargs)

    def update(params: object, moments: object, rows: Any, hpsi: Any, density: Any,
               learning_rate: float, *, controls: AdaptiveSRState) -> tuple:
        validate_control_state(policy, controls)
        if len(rows) != controls.fit_records:
            raise ValueError('draw the registered next-block record count before solving SR')
        return solver(controls.damping, controls.fs_radius)(params, moments, rows, hpsi, density, learning_rate)

    return update


def select_controlled_finite_grid(
    policy: AdaptiveSRPolicy, controls: AdaptiveSRState, phi: np.ndarray,
    density: np.ndarray, fractions: np.ndarray,
    *, split: int, groups: int = 32,
) -> tuple[int, dict]:
    """Use ACTUAL coherent projected amplitudes and a separate geometry veto.

    The selection half gets 80% of the radius; the untouched guard half must
    satisfy the full nominal upper bound. Failure returns the parent, never
    another candidate selected on guard data. These amplitude-only geometry
    rows are separate from the full-H/JVP scheduling monitor AND final audit.
    """
    from .finite_candidate_calibration import select_finite_grid
    validate_control_state(policy, controls)
    chosen, report = select_finite_grid(phi, density, fractions, split=split, groups=groups,
        target=.8 * controls.fs_radius, maximum=controls.fs_radius)
    return chosen, dict(report, controlled_maximum_FS=controls.fs_radius,
        maximum_is_realized_guard_distance_not_parameter_norm=True)
