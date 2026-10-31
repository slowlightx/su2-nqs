"""Leakage-safe weighted regression for autoregressive prefix diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WeightedGroupRidgeResult:
    """Out-of-fold predictions with complete prefixes kept in one fold."""

    predictions: np.ndarray
    fold_assignments: np.ndarray
    n_groups: int
    n_folds: int
    ridge: float


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    concrete = np.asarray(weights, dtype=np.float64)
    if concrete.ndim != 1 or np.any(~np.isfinite(concrete)) or np.any(concrete < 0.0):
        raise ValueError("weights must be a finite non-negative vector")
    total = float(np.sum(concrete))
    if total <= 0.0:
        raise ValueError("weights must have positive total")
    return concrete / total


def _group_keys(groups: np.ndarray) -> tuple[tuple[int, ...], ...]:
    concrete = np.asarray(groups)
    if concrete.ndim == 1:
        concrete = concrete[:, None]
    if concrete.ndim != 2:
        raise ValueError("groups must be a one- or two-dimensional array")
    return tuple(tuple(int(value) for value in row) for row in concrete)


def _balanced_group_folds(
    groups: tuple[tuple[int, ...], ...],
    weights: np.ndarray,
    maximum_folds: int,
) -> tuple[np.ndarray, int, int]:
    members: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(groups):
        members.setdefault(key, []).append(index)
    n_groups = len(members)
    if n_groups < 2:
        return np.full(len(groups), -1, dtype=np.int32), n_groups, 0
    n_folds = min(maximum_folds, n_groups)
    masses = {
        key: float(np.sum(weights[np.asarray(indices, dtype=np.int64)]))
        for key, indices in members.items()
    }
    # Largest groups are placed first into the currently lightest fold.  The
    # key tie-break makes the split independent of Python dictionary order.
    ordered = sorted(members, key=lambda key: (-masses[key], key))
    fold_mass = np.zeros(n_folds, dtype=np.float64)
    group_fold: dict[tuple[int, ...], int] = {}
    for key in ordered:
        fold = int(np.argmin(fold_mass))
        group_fold[key] = fold
        fold_mass[fold] += masses[key]
    assignment = np.asarray([group_fold[key] for key in groups], dtype=np.int32)
    return assignment, n_groups, n_folds


def _weighted_ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_weights: np.ndarray,
    test_features: np.ndarray,
    ridge: float,
) -> np.ndarray:
    weights = _normalized_weights(train_weights)
    feature_mean = np.sum(weights[:, None] * train_features, axis=0)
    target_mean = np.sum(weights * train_targets)
    centered = train_features - feature_mean
    scale = np.sqrt(np.sum(weights[:, None] * centered**2, axis=0))
    active = scale > 1.0e-13
    if not np.any(active):
        return np.full(len(test_features), target_mean, dtype=np.complex128)
    standardized = centered[:, active] / scale[active]
    centered_target = train_targets - target_mean
    gram = standardized.T @ (weights[:, None] * standardized)
    rhs = standardized.T @ (weights * centered_target)
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64), rhs
    )
    test_standardized = (test_features[:, active] - feature_mean[active]) / scale[active]
    return target_mean + test_standardized @ coefficients


def weighted_group_ridge_oof(
    features: np.ndarray,
    targets: Sequence[complex] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    groups: np.ndarray,
    *,
    ridge: float = 1.0e-3,
    maximum_folds: int = 5,
) -> WeightedGroupRidgeResult:
    """Fit complex targets with prefix-grouped out-of-fold weighted ridge.

    Rows sharing a group are never split between train and test.  Features are
    real; real and imaginary target components share the same ridge solution
    through a complex right-hand side.  Standardization and the intercept use
    training-fold weights only.
    """

    design = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.complex128)
    normalized = _normalized_weights(np.asarray(weights, dtype=np.float64))
    if design.ndim != 2 or design.shape[0] != len(target):
        raise ValueError("features and targets must have the same row count")
    if normalized.shape != target.shape:
        raise ValueError("weights and targets must have the same shape")
    if np.any(~np.isfinite(design)) or np.any(~np.isfinite(target)):
        raise ValueError("regression inputs must be finite")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if maximum_folds < 2:
        raise ValueError("maximum_folds must be at least two")
    keys = _group_keys(groups)
    if len(keys) != len(target):
        raise ValueError("groups must have one row per target")
    assignment, n_groups, n_folds = _balanced_group_folds(
        keys, normalized, maximum_folds
    )
    prediction = np.zeros_like(target)
    if n_folds == 0:
        return WeightedGroupRidgeResult(
            prediction, assignment, n_groups, n_folds, ridge
        )
    for fold in range(n_folds):
        test = assignment == fold
        train = ~test
        prediction[test] = _weighted_ridge_predict(
            design[train], target[train], normalized[train], design[test], ridge
        )
    return WeightedGroupRidgeResult(
        prediction, assignment, n_groups, n_folds, ridge
    )


def weighted_complex_prediction_diagnostics(
    targets: Sequence[complex] | np.ndarray,
    predictions: Sequence[complex] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    central_mass: float = 0.99,
) -> dict[str, float]:
    """Report zero-baseline explained norm and rare-target sensitivity."""

    target = np.asarray(targets, dtype=np.complex128)
    prediction = np.asarray(predictions, dtype=np.complex128)
    normalized = _normalized_weights(np.asarray(weights, dtype=np.float64))
    if target.shape != prediction.shape or target.shape != normalized.shape:
        raise ValueError("targets, predictions, and weights must share one shape")
    if not 0.0 < central_mass < 1.0:
        raise ValueError("central_mass must lie strictly between zero and one")
    squared = np.abs(target) ** 2
    error_squared = np.abs(target - prediction) ** 2
    norm = float(np.sum(normalized * squared))
    error = float(np.sum(normalized * error_squared))
    second = float(np.sum(normalized * squared**2))

    order = np.argsort(np.abs(target), kind="stable")
    cumulative = np.cumsum(normalized[order])
    cutoff_position = min(
        int(np.searchsorted(cumulative, central_mass, side="left")), len(order) - 1
    )
    threshold = float(np.abs(target[order[cutoff_position]]))
    central = np.abs(target) <= threshold
    tail = ~central

    def explained(mask: np.ndarray) -> float:
        signal = float(np.sum(normalized[mask] * squared[mask]))
        residual = float(np.sum(normalized[mask] * error_squared[mask]))
        if signal == 0.0:
            # A zero-signal subset has no internally normalized R2.  Returning
            # zero keeps JSON finite; the separately reported contribution to
            # the *total* norm remains the meaningful diagnostic.
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / signal

    central_signal = float(np.sum(normalized[central] * squared[central]))
    central_error = float(
        np.sum(normalized[central] * error_squared[central])
    )
    tail_signal = float(np.sum(normalized[tail] * squared[tail]))
    tail_error = float(np.sum(normalized[tail] * error_squared[tail]))

    return {
        "target_norm": norm,
        "prediction_error": error,
        "explained_norm_fraction": 1.0 - error / norm if norm > 0.0 else 1.0,
        "target_variance_ess_fraction": norm**2 / second if second > 0.0 else 1.0,
        "maximum_target_magnitude": float(np.max(np.abs(target), initial=0.0)),
        "maximum_prediction_magnitude": float(
            np.max(np.abs(prediction), initial=0.0)
        ),
        "central_mass_threshold": threshold,
        "central_actual_mass": float(np.sum(normalized[central])),
        "central_explained_norm_fraction": explained(central),
        "central_target_norm_fraction": central_signal / norm if norm > 0.0 else 0.0,
        "central_explained_total_norm_fraction": (
            (central_signal - central_error) / norm if norm > 0.0 else 0.0
        ),
        "tail_actual_mass": float(np.sum(normalized[tail])),
        "tail_target_norm_fraction": tail_signal / norm if norm > 0.0 else 0.0,
        "tail_explained_norm_fraction": explained(tail),
        "tail_explained_total_norm_fraction": (
            (tail_signal - tail_error) / norm if norm > 0.0 else 0.0
        ),
    }
