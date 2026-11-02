"""Comparable observed energy-to-target records for benchmark trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def observed_energy_to_target(
    training: Mapping[str, object],
    *,
    final_energy: float,
    exact_energy: float,
    n_sites: int,
    total_steps: int,
    targets_per_site: Sequence[float],
) -> list[dict[str, object]]:
    """Return the first recorded checkpoint reaching each energy target.

    ``loss_before_update`` at loop step ``k`` is assigned to ``k-1`` completed
    updates; the separately evaluated final state is assigned to
    ``total_steps``.  Per-history timestamps are deliberately not attached to
    those pre-update states because they are recorded after update ``k``.
    Intermediate times therefore use a labelled linear estimate from aggregate
    training time and are not confirmatory timing.
    """

    if n_sites < 1 or total_steps < 1:
        raise ValueError("n_sites and total_steps must be positive")
    history = training.get("history")
    if not isinstance(history, list):
        raise TypeError("training history must be a list")
    compile_seconds = float(training.get("compile_seconds", 0.0))
    training_seconds = float(training.get("training_seconds", 0.0))
    points = []
    if "initial_loss" in training:
        points.append((0, float(training["initial_loss"]), 0.0))
    points.extend(
        (
            int(row["step"]) - 1,
            float(row["loss_before_update"]),
            # The timestamp is taken after update ``step`` whereas this loss
            # belongs to the state after ``step - 1`` completed updates.  Do
            # not silently attach that one-update-late time to the state.
            None,
        )
        for row in history
        if not ("initial_loss" in training and int(row["step"]) == 1)
    )
    if any(step < 0 or step > total_steps for step, _, _ in points):
        raise ValueError("history steps lie outside the optimization budget")
    if any(right[0] <= left[0] for left, right in zip(points, points[1:])):
        raise ValueError("history steps must be strictly increasing")
    points.append((total_steps, float(final_energy), training_seconds))
    if compile_seconds < 0.0 or training_seconds < 0.0:
        raise ValueError("recorded times must be non-negative")

    result = []
    for target in targets_per_site:
        threshold = float(target)
        if threshold < 0.0:
            raise ValueError("energy-error targets must be non-negative")
        reached = next(
            (
                (step, energy, checkpoint_seconds)
                for step, energy, checkpoint_seconds in points
                if (energy - exact_energy) / n_sites <= threshold
            ),
            None,
        )
        result.append(
            {
                "target_energy_error_per_site": threshold,
                "reached": reached is not None,
                "first_observed_step": None if reached is None else reached[0],
                "first_observed_completed_updates": None
                if reached is None
                else reached[0],
                "step_semantics": "number of completed optimizer updates",
                "observed_energy": None if reached is None else reached[1],
                "training_seconds_to_checkpoint": None
                if reached is None
                else (
                    reached[2]
                    if reached[2] is not None
                    else training_seconds * reached[0] / total_steps
                ),
                "cold_seconds_to_checkpoint": None
                if reached is None
                else compile_seconds
                + (
                    reached[2]
                    if reached[2] is not None
                    else training_seconds * reached[0] / total_steps
                ),
                "time_is_interpolated_from_aggregate": (
                    reached is not None and reached[2] is None
                ),
            }
        )
    return result


__all__ = ["observed_energy_to_target"]
