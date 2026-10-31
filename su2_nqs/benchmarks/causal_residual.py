"""Exact autoregressive-filtration diagnostics for Hamiltonian residuals.

The central quantity is the local-energy residual ``g(x) = E_loc(x) - E``.
For an autoregressive action sequence, its conditional expectations form a
Doob martingale.  The squared increments are orthogonal, so their Born-weighted
variances add exactly to the energy variance.  This gives a physical
trainability diagnostic: it measures where the Hamiltonian becomes visible
along a fusion-tree sampling order, without introducing a model-dependent
loss or a geometric surrogate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..models.compact_tree_gru import split_tables
from ..models.tree_gru import tree_decisions
from ..su2.fusion_tree import FusionTree


def compact_tree_action_sequences(
    tree: FusionTree,
    target_q: int,
    states: np.ndarray,
) -> np.ndarray:
    """Return compact split-action indices for valid postorder tree labels."""

    labels = np.asarray(states)
    expected_shape = (tree.n_leaves - 1,)
    if labels.ndim != 2 or labels.shape[1:] != expected_shape:
        raise ValueError(
            f"expected states with trailing shape {expected_shape}, got {labels.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("fusion labels must be integer doubled quantum numbers")
    if np.any(labels[:, -1] != target_q):
        raise ValueError("every state must end in target_q")

    table_left, table_right, table_valid = split_tables(tree)
    answer = np.empty((len(labels), tree.n_leaves - 1), dtype=np.int32)
    for decision in tree_decisions(tree):
        for state_index, state in enumerate(labels):
            parent_q = int(state[decision.node_index])
            if not 0 <= parent_q <= tree.n_leaves:
                raise ValueError("a parent doubled spin lies outside the tree range")
            desired_left = (
                1 if decision.left_index is None else int(state[decision.left_index])
            )
            desired_right = (
                1 if decision.right_index is None else int(state[decision.right_index])
            )
            matches = np.flatnonzero(
                table_valid[decision.order, parent_q]
                & (table_left[decision.order, parent_q] == desired_left)
                & (table_right[decision.order, parent_q] == desired_right)
            )
            if len(matches) != 1:
                raise ValueError("state contains an invalid or ambiguous fusion split")
            answer[state_index, decision.order] = int(matches[0])
    return answer


def _conditional_means(
    probabilities: np.ndarray,
    values: np.ndarray,
    actions: np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    totals: dict[tuple[int, ...], list[complex | float]] = defaultdict(
        lambda: [0.0, 0.0j]
    )
    for probability, value, row in zip(probabilities, values, actions, strict=True):
        key = tuple(int(action) for action in row[:prefix_length])
        totals[key][0] += float(probability)
        totals[key][1] += probability * value
    means = {
        key: numerator / weight
        for key, (weight, numerator) in totals.items()
        if weight > 0.0
    }
    return np.asarray(
        [means[tuple(int(action) for action in row[:prefix_length])] for row in actions],
        dtype=np.complex128,
    )


def _conditional_action_statistics(
    probabilities: np.ndarray,
    actions: np.ndarray,
    decision: int,
) -> tuple[float, float]:
    groups: dict[tuple[int, ...], dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for probability, row in zip(probabilities, actions, strict=True):
        prefix = tuple(int(action) for action in row[:decision])
        groups[prefix][int(row[decision])] += float(probability)
    entropy = 0.0
    reachable = 0.0
    for weights in groups.values():
        total = float(sum(weights.values()))
        if total <= 0.0:
            continue
        local = np.asarray(tuple(weights.values()), dtype=np.float64) / total
        entropy -= total * float(np.sum(local * np.log2(local)))
        reachable += total * len(local)
    return entropy, reachable


def _increment_tail(
    probabilities: np.ndarray,
    squared_increment: np.ndarray,
) -> dict[str, float]:
    mean = float(np.sum(probabilities * squared_increment))
    if mean <= 0.0:
        return {
            "sampling_ess_fraction": 1.0,
            "maximum_normalized_squared_increment": 0.0,
            "born_probability_at_maximum": 0.0,
        }
    second = float(np.sum(probabilities * squared_increment**2))
    maximum_index = int(np.argmax(squared_increment))
    return {
        "sampling_ess_fraction": float(mean**2 / second) if second > 0.0 else 1.0,
        "maximum_normalized_squared_increment": float(
            squared_increment[maximum_index] / mean
        ),
        "born_probability_at_maximum": float(probabilities[maximum_index]),
    }


def martingale_residual_decomposition(
    probabilities: np.ndarray,
    values: np.ndarray,
    action_sequences: np.ndarray,
) -> dict[str, object]:
    """Decompose a complex residual along an autoregressive filtration.

    ``values`` may be a local-energy residual or any square-integrable complex
    observable.  Returned ``variance_contribution`` values sum to its exact
    Born variance when the complete action sequence uniquely identifies each
    state.
    """

    weights = np.asarray(probabilities, dtype=np.float64)
    observable = np.asarray(values, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if weights.ndim != 1 or observable.shape != weights.shape:
        raise ValueError("probabilities and values must be equal-length vectors")
    if actions.ndim != 2 or len(actions) != len(weights):
        raise ValueError("action_sequences must have one row per state")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    normalization = float(np.sum(weights))
    if normalization <= 0.0:
        raise ValueError("probabilities must have positive total weight")
    weights = weights / normalization
    active = weights > 0.0
    if np.any(~np.isfinite(observable[active])):
        raise ValueError("values must be finite on positive-probability states")
    observable = np.where(active, observable, 0.0j)

    mean = np.sum(weights * observable)
    variance = float(np.sum(weights * np.abs(observable - mean) ** 2))
    previous = np.full(len(weights), mean, dtype=np.complex128)
    rows: list[dict[str, float]] = []
    for decision in range(actions.shape[1]):
        current = _conditional_means(weights, observable, actions, decision + 1)
        squared_increment = np.abs(current - previous) ** 2
        contribution = float(np.sum(weights * squared_increment))
        entropy, reachable = _conditional_action_statistics(weights, actions, decision)
        rows.append(
            {
                "decision": float(decision),
                "conditional_entropy_bits": float(entropy),
                "mean_reachable_actions": float(reachable),
                "variance_contribution": contribution,
                "variance_fraction": contribution / variance if variance > 0.0 else 0.0,
                **_increment_tail(weights, squared_increment),
            }
        )
        previous = current

    explained = float(sum(row["variance_contribution"] for row in rows))
    cumulative = 0.0
    for row in rows:
        cumulative += row["variance_contribution"]
        row["cumulative_variance_fraction"] = (
            cumulative / variance if variance > 0.0 else 0.0
        )
    born_entropy = float(-np.sum(weights[active] * np.log2(weights[active])))
    return {
        "mean_real": float(np.real(mean)),
        "mean_imag": float(np.imag(mean)),
        "variance": variance,
        "explained_variance": explained,
        "orthogonality_error": float(abs(variance - explained)),
        "born_entropy_bits": born_entropy,
        "conditional_entropy_sum_bits": float(
            sum(row["conditional_entropy_bits"] for row in rows)
        ),
        "decisions": rows,
    }


def martingale_residual_increments(
    probabilities: np.ndarray,
    values: np.ndarray,
    action_sequences: np.ndarray,
) -> tuple[np.ndarray, complex]:
    """Return state-resolved Doob increments of an observable.

    Column ``k`` is the change in conditional expectation produced by action
    ``k``. Summing all columns reconstructs ``values - mean`` whenever the
    complete action sequence uniquely identifies each positive-weight state.
    """

    weights = np.asarray(probabilities, dtype=np.float64)
    observable = np.asarray(values, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if weights.ndim != 1 or observable.shape != weights.shape:
        raise ValueError("probabilities and values must be equal-length vectors")
    if actions.ndim != 2 or len(actions) != len(weights):
        raise ValueError("action_sequences must have one row per state")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    normalization = float(np.sum(weights))
    if normalization <= 0.0:
        raise ValueError("probabilities must have positive total weight")
    weights = weights / normalization
    active = weights > 0.0
    if np.any(~np.isfinite(observable[active])):
        raise ValueError("values must be finite on positive-probability states")
    observable = np.where(active, observable, 0.0j)
    mean = complex(np.sum(weights * observable))
    previous = np.full(len(weights), mean, dtype=np.complex128)
    columns = []
    for decision in range(actions.shape[1]):
        current = _conditional_means(weights, observable, actions, decision + 1)
        columns.append(current - previous)
        previous = current
    return np.stack(columns, axis=1), mean


def residual_concentration_summary(
    probabilities: np.ndarray,
    residual_values: np.ndarray,
    *,
    fractions: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict[str, object]:
    """Report Born mass needed to carry fixed fractions of residual variance."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    residual_values = np.asarray(residual_values, dtype=np.complex128)
    probabilities = probabilities / np.sum(probabilities)
    contributions = probabilities * np.abs(residual_values) ** 2
    total = float(np.sum(contributions))
    order = np.argsort(contributions)[::-1]
    cumulative = np.cumsum(contributions[order])
    rows = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("concentration fractions must lie in (0, 1]")
        count = int(np.searchsorted(cumulative, fraction * total, side="left") + 1)
        selected = order[:count]
        rows.append(
            {
                "variance_fraction": float(fraction),
                "state_count": count,
                "born_mass": float(np.sum(probabilities[selected])),
            }
        )
    squared = np.abs(residual_values) ** 2
    mean = float(np.sum(probabilities * squared))
    second = float(np.sum(probabilities * squared**2))
    maximum_index = int(np.argmax(squared))
    return {
        "variance": total,
        "variance_estimator_ess_fraction": float(mean**2 / second),
        "maximum_squared_residual_over_mean": float(squared[maximum_index] / mean),
        "born_probability_at_maximum_residual": float(probabilities[maximum_index]),
        "concentration": rows,
    }
