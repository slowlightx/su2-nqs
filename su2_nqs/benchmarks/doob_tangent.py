"""Exact matrix-valued Doob decomposition of model tangent features.

For complete small-system support, every score column can be decomposed into
orthogonal increments of the autoregressive filtration.  This exposes where
non-null tangent directions become visible without assigning a learned weight
to a suffix or Hamiltonian history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoobTangentDecomposition:
    """Conditional mean and decision increments for a feature matrix."""

    mean: np.ndarray
    increments: np.ndarray
    normalized_weights: np.ndarray
    reconstruction_maximum_error: float
    increment_mean_maximum_error: float
    cross_decision_gram_maximum_error: float
    decision_ranks: tuple[int, ...]
    decision_singular_values: tuple[np.ndarray, ...]
    cumulative_ranks: tuple[int, ...]
    innovation_rank_gains: tuple[int, ...]
    cumulative_singular_values: tuple[np.ndarray, ...]


def _validate(
    values: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(values)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2:
        raise ValueError("values must have shape (states, features)")
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.shape != (features.shape[0],):
        raise ValueError("probabilities must have one value per state")
    action_values = np.asarray(actions)
    if action_values.ndim != 2 or action_values.shape[0] != len(weights):
        raise ValueError("actions must have one row per state")
    if not np.issubdtype(action_values.dtype, np.integer):
        raise TypeError("actions must be integers")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("probabilities must have positive total weight")
    weights = weights / total
    active = weights > 0.0
    if np.any(~np.isfinite(features[active])):
        raise ValueError("features must be finite on positive-weight states")
    features = np.where(active[:, None], features, 0.0)
    return features, weights, action_values.astype(np.int32, copy=False)


def _conditional_matrix(
    values: np.ndarray,
    weights: np.ndarray,
    actions: np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    keys: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(actions):
        key = tuple(int(value) for value in row[:prefix_length])
        keys.setdefault(key, []).append(index)
    result = np.zeros_like(values)
    for indices in keys.values():
        selected = np.asarray(indices, dtype=np.int32)
        mass = float(np.sum(weights[selected]))
        if mass > 0.0:
            mean = np.sum(weights[selected, None] * values[selected], axis=0) / mass
            result[selected] = mean
    return result


def decompose_doob_tangent(
    values: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    *,
    rank_cutoff: float = 1.0e-10,
) -> DoobTangentDecomposition:
    """Decompose real or complex feature columns by revealed action prefix."""

    if not np.isfinite(rank_cutoff) or rank_cutoff <= 0.0:
        raise ValueError("rank_cutoff must be positive and finite")
    features, weights, action_values = _validate(values, probabilities, actions)
    mean = np.sum(weights[:, None] * features, axis=0)
    previous = np.broadcast_to(mean, features.shape).copy()
    increments = []
    spectra = []
    root = np.sqrt(2.0 * weights)
    for decision in range(action_values.shape[1]):
        current = _conditional_matrix(
            features, weights, action_values, decision + 1
        )
        increment = current - previous
        increments.append(increment)
        if np.iscomplexobj(increment):
            design = np.concatenate(
                (root[:, None] * increment.real, root[:, None] * increment.imag)
            )
        else:
            design = root[:, None] * increment
        singular = np.linalg.svd(design, compute_uv=False)
        spectra.append(np.asarray(singular, dtype=np.float64))
        previous = current
    increment_array = np.asarray(increments)
    centered = features - mean
    reconstructed = np.sum(increment_array, axis=0)
    reconstruction_error = float(
        np.max(np.abs(reconstructed - centered), initial=0.0)
    )
    mean_error = float(
        np.max(
            np.abs(np.einsum("n,dnk->dk", weights, increment_array)),
            initial=0.0,
        )
    )
    cross_error = 0.0
    for first in range(len(increment_array)):
        for second in range(first):
            block = np.einsum(
                "nk,n,nl->kl",
                increment_array[first].conj(),
                weights,
                increment_array[second],
            )
            cross_error = max(
                cross_error, float(np.max(np.abs(block), initial=0.0))
            )
    global_largest = max(
        (float(values[0]) for values in spectra if len(values)), default=0.0
    )
    ranks = tuple(
        int(np.count_nonzero(values > rank_cutoff * global_largest))
        if global_largest > 0.0
        else 0
        for values in spectra
    )
    cumulative_spectra = []
    cumulative_designs = []
    for increment in increment_array:
        if np.iscomplexobj(increment):
            design = np.concatenate(
                (root[:, None] * increment.real, root[:, None] * increment.imag)
            )
        else:
            design = root[:, None] * increment
        cumulative_designs.append(design)
        cumulative_spectra.append(
            np.asarray(
                np.linalg.svd(
                    np.concatenate(cumulative_designs, axis=0),
                    compute_uv=False,
                ),
                dtype=np.float64,
            )
        )
    cumulative_largest = max(
        (float(values[0]) for values in cumulative_spectra if len(values)),
        default=0.0,
    )
    cumulative_ranks = tuple(
        int(np.count_nonzero(values > rank_cutoff * cumulative_largest))
        if cumulative_largest > 0.0
        else 0
        for values in cumulative_spectra
    )
    previous_rank = 0
    gains = []
    for rank in cumulative_ranks:
        gains.append(rank - previous_rank)
        previous_rank = rank
    return DoobTangentDecomposition(
        mean=np.asarray(mean),
        increments=increment_array,
        normalized_weights=weights,
        reconstruction_maximum_error=reconstruction_error,
        increment_mean_maximum_error=mean_error,
        cross_decision_gram_maximum_error=cross_error,
        decision_ranks=ranks,
        decision_singular_values=tuple(spectra),
        cumulative_ranks=cumulative_ranks,
        innovation_rank_gains=tuple(gains),
        cumulative_singular_values=tuple(cumulative_spectra),
    )


__all__ = ["DoobTangentDecomposition", "decompose_doob_tangent"]
