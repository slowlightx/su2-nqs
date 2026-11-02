"""Deterministic locality-token controls for route models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def maximally_mismatched_cyclic_permutation(
    features: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Return the nontrivial cyclic shift with largest feature mismatch.

    A cyclic control is deterministic, has no fixed route, and preserves the
    complete multiset (and hence every marginal statistic) of input tokens.
    Only the physical association between a token and its fusion route is
    destroyed.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must be a rank-two array with at least two routes")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    n_routes = values.shape[0]
    best_shift = max(
        range(1, n_routes),
        key=lambda shift: (
            float(np.sum((values - np.roll(values, shift, axis=0)) ** 2)),
            -shift,
        ),
    )
    return tuple((route - best_shift) % n_routes for route in range(n_routes))


def permute_route_features(
    features: Sequence[Sequence[float]],
    permutation: Sequence[int],
) -> tuple[tuple[float, ...], ...]:
    """Permute route-token rows while preserving their values exactly."""

    values = np.asarray(features, dtype=np.float64)
    order = tuple(int(index) for index in permutation)
    if sorted(order) != list(range(values.shape[0])):
        raise ValueError("permutation must contain each route index exactly once")
    return tuple(tuple(float(value) for value in values[index]) for index in order)


__all__ = [
    "maximally_mismatched_cyclic_permutation",
    "permute_route_features",
]
