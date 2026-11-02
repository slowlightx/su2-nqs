"""Diagnostics for tangent quotients learned from incomplete fusion support."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def complex_centered_rank_ceiling(
    observed_state_count: int,
    *,
    full_state_count: int | None = None,
) -> int:
    """Maximum real rank of centered complex features on observed states."""

    if isinstance(observed_state_count, bool) or observed_state_count < 1:
        raise ValueError("observed_state_count must be a positive integer")
    if not isinstance(observed_state_count, int):
        raise TypeError("observed_state_count must be an integer")
    ceiling = 2 * (observed_state_count - 1)
    if full_state_count is not None:
        if isinstance(full_state_count, bool) or full_state_count < 1:
            raise ValueError("full_state_count must be a positive integer")
        if not isinstance(full_state_count, int):
            raise TypeError("full_state_count must be an integer")
        if observed_state_count > full_state_count:
            raise ValueError("observed support cannot exceed full support")
        ceiling = min(ceiling, 2 * (full_state_count - 1))
    return ceiling


@dataclass(frozen=True)
class InnovationRankStability:
    """Seed stability of normalized cumulative Doob innovation ranks."""

    normalized_cumulative_profiles: np.ndarray
    maximum_profile_range: float
    peak_decisions: tuple[int, ...]
    peak_decision_consistent: bool
    final_ranks: tuple[int, ...]


def innovation_rank_stability(
    innovation_rank_gains: Sequence[Sequence[int]],
) -> InnovationRankStability:
    """Compare filtration-rank allocation independently of total sample rank."""

    rows = tuple(tuple(row) for row in innovation_rank_gains)
    if not rows:
        raise ValueError("at least one innovation-rank sequence is required")
    length = len(rows[0])
    if length < 1 or any(len(row) != length for row in rows):
        raise ValueError("innovation-rank sequences must have one common length")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for row in rows
        for value in row
    ):
        raise TypeError("innovation ranks must be integers")
    values = np.asarray(rows, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("innovation ranks must be nonnegative")
    final = np.sum(values, axis=1)
    if np.any(final <= 0):
        raise ValueError("each innovation-rank sequence must have positive total")
    profiles = np.cumsum(values, axis=1) / final[:, None]
    maximum_range = float(np.max(np.ptp(profiles, axis=0), initial=0.0))
    peaks = tuple(int(np.argmax(row)) for row in values)
    return InnovationRankStability(
        normalized_cumulative_profiles=profiles,
        maximum_profile_range=maximum_range,
        peak_decisions=peaks,
        peak_decision_consistent=len(set(peaks)) == 1,
        final_ranks=tuple(int(value) for value in final),
    )


__all__ = [
    "InnovationRankStability",
    "complex_centered_rank_ceiling",
    "innovation_rank_stability",
]
