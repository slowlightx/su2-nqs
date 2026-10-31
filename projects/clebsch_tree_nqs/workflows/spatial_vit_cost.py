"""Fail-closed wall-time forecasts with explicit process compilation boundaries.

This module changes scheduling only. It never changes a model, samples,
optimizer, physical trust gate, or the registered cumulative training cap.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def forecast_training_cost(
    history: Sequence[Mapping[str, object]], *, process_first_steps: Sequence[int],
    target_step: int, active_training_seconds: float,
    maximum_training_seconds: float, remaining_processes: int,
    minimum_warm_updates: int = 3, warm_window: int = 8,
    safety_factor: float = 1.15,
) -> dict[str, object]:
    """Charge compilation once per future process, not once per future update.

    First updates of each independently started process include compilation.
    They are declared explicitly, never inferred from a timing outlier. Require
    at least three measured non-first updates in the LATEST process, then use
    the maximum of its latest warm observations. Count all historical time in
    ``active_training_seconds`` and add a conservative full first-update cost
    for each remaining process (even though its warm component is also counted).
    Insufficient observations or an over-budget forecast fail closed.
    """
    integers = (target_step, remaining_processes, minimum_warm_updates, warm_window)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in integers):
        raise ValueError('integer step/process/warm limits required')
    if (not history or target_step < 1 or remaining_processes < 0
            or minimum_warm_updates < 3 or warm_window < minimum_warm_updates):
        raise ValueError('history, target and at least three warm observations required')
    if (not math.isfinite(safety_factor) or safety_factor < 1
            or not math.isfinite(active_training_seconds) or active_training_seconds < 0
            or not math.isfinite(maximum_training_seconds) or maximum_training_seconds <= 0):
        raise ValueError('finite nonnegative spent time and positive fixed cap required')
    steps = [row['step'] for row in history]
    if steps != list(range(1, len(history)+1)) or target_step < len(history):
        raise ValueError('complete contiguous cumulative update history required')
    starts = list(process_first_steps)
    if (not starts or starts[0] != 1 or starts != sorted(set(starts))
            or any(isinstance(s, bool) or not isinstance(s, int)
                   or s < 1 or s > len(history) for s in starts)):
        raise ValueError('declare every completed process first step in order')
    times = [float(row['seconds_including_compile']) for row in history]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('finite positive measured update times required')
    if active_training_seconds+1e-6 < sum(times):
        raise ValueError('cumulative spent time omitted historical updates')
    latest = starts[-1]
    warm = [(step, times[step-1]) for step in steps if step > latest][-warm_window:]
    if len(warm) < minimum_warm_updates:
        raise ValueError('insufficient warm updates in latest process; keep checkpoint')
    warm_seconds = max(t for _, t in warm)
    first_seconds = max(times[s-1] for s in starts)
    update_seconds = (target_step-len(history))*warm_seconds
    startup_seconds = remaining_processes*first_seconds
    forecast = safety_factor*(update_seconds+startup_seconds)
    result = dict(
        completed_steps=len(history), target_steps=target_step,
        process_first_steps=starts, warm_steps=[s for s, _ in warm],
        warm_update_seconds=warm_seconds,
        conservative_first_update_seconds_per_process=first_seconds,
        remaining_processes=remaining_processes, safety_factor=safety_factor,
        already_spent_seconds=active_training_seconds,
        remaining_update_seconds_before_margin=update_seconds,
        remaining_startup_seconds_before_margin=startup_seconds,
        forecast_remaining_seconds=forecast,
        forecast_total_seconds=active_training_seconds+forecast,
        fixed_maximum_training_seconds=maximum_training_seconds,
        compilation_charged_per_process_not_per_update=True,
        physics_or_optimizer_changed=False,
    )
    if result['forecast_total_seconds'] > maximum_training_seconds:
        raise ValueError('measured warm cost exceeds fixed cumulative budget; keep checkpoint')
    return result
