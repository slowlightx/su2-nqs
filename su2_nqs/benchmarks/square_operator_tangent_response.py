"""Square-lattice on-demand scalar actions and tangent diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np

from .operator_tangent_response import scalar_operator_tangent_statistics
from ..operators.square_a1_scalar import (
    SQUARE_A1_SCALAR_LABELS,
    square_a1_scalar_token_connectivity,
)


BatchAmplitude = Callable[[np.ndarray], np.ndarray]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def square_a1_operator_actions(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    linear_size: int,
    *,
    center_batch_size: int = 8,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate the five exact square scalar actions on supplied states."""

    center_batch_size = _positive_integer(center_batch_size, "center_batch_size")
    amplitude_batch_size = _positive_integer(
        amplitude_batch_size, "amplitude_batch_size"
    )
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    actions = np.zeros((len(states), 5), dtype=np.complex128)
    connected_requests = 0
    unique_evaluations = 0
    maximum_union_size = 0
    for start in range(0, len(states), center_batch_size):
        stop = min(len(states), start + center_batch_size)
        records = []
        union: dict[tuple[int, ...], int] = {}
        for state in states[start:stop]:
            tokens = square_a1_scalar_token_connectivity(state, linear_size)
            row = []
            for label in SQUARE_A1_SCALAR_LABELS:
                token_states, elements = tokens[label]
                row.append((token_states, elements))
                connected_requests += len(token_states)
                for connected in token_states:
                    if connected not in union:
                        union[connected] = len(union)
            records.append(row)
        ordered = np.asarray(tuple(union), dtype=np.int8)
        values = []
        for amplitude_start in range(0, len(ordered), amplitude_batch_size):
            batch = ordered[amplitude_start : amplitude_start + amplitude_batch_size]
            observed = np.asarray(amplitude_fn(batch), dtype=np.complex128)
            if observed.shape != (len(batch),) or np.any(~np.isfinite(observed)):
                raise FloatingPointError(
                    "amplitude_fn returned invalid connected amplitudes"
                )
            values.append(observed)
        connected_amplitudes = np.concatenate(values)
        unique_evaluations += len(ordered)
        maximum_union_size = max(maximum_union_size, len(ordered))
        for local_row, row in enumerate(records):
            for token_index, (token_states, elements) in enumerate(row):
                positions = np.fromiter(
                    (union[connected] for connected in token_states),
                    dtype=np.int64,
                    count=len(token_states),
                )
                actions[start + local_row, token_index] = np.dot(
                    elements, connected_amplitudes[positions]
                )
    if np.any(~np.isfinite(actions)):
        raise FloatingPointError("non-finite square scalar-token action")
    return actions, {
        "sample_count": len(states),
        "connected_configuration_requests": connected_requests,
        "unique_connected_amplitude_evaluations": unique_evaluations,
        "deduplication_fraction": 1.0
        - unique_evaluations / max(1, connected_requests),
        "maximum_center_batch_union_size": maximum_union_size,
    }


def square_a1_local_ratios(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    center_amplitudes: np.ndarray,
    linear_size: int,
    *,
    center_batch_size: int = 8,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate the five square scalar local ratios."""

    centers = np.asarray(center_amplitudes, dtype=np.complex128)
    if centers.shape != (len(configurations),):
        raise ValueError("center_amplitudes must contain one value per state")
    if np.any(~np.isfinite(centers)) or np.any(np.abs(centers) <= 0.0):
        raise FloatingPointError("sampled center amplitudes must be finite and nonzero")
    actions, ledger = square_a1_operator_actions(
        amplitude_fn,
        configurations,
        linear_size,
        center_batch_size=center_batch_size,
        amplitude_batch_size=amplitude_batch_size,
    )
    return actions / centers[:, None], ledger


def square_scalar_operator_tangent_statistics(
    local_ratios: np.ndarray, couplings: tuple[float, ...]
) -> dict[str, object]:
    """Reuse the common five-token statistics with square physical labels."""

    result = scalar_operator_tangent_statistics(local_ratios, couplings)
    result["token_labels"] = list(SQUARE_A1_SCALAR_LABELS)
    return result


__all__ = [
    "square_a1_local_ratios",
    "square_a1_operator_actions",
    "square_scalar_operator_tangent_statistics",
]
