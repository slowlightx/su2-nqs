"""Exact predictive ranks across autoregressive fusion-action cuts.

For an action sequence ``a=(a_0,...,a_{L-1})`` and a state-like vector
``v(a)``, a cut after ``d`` actions defines the zero-filled matrix

``M_d[prefix, suffix] = v(prefix + suffix)``.

Its singular spectrum measures the exact bond dimension required to predict
the unresolved action suffix from the revealed action prefix.  This is an
autoregressive/fusion-action complexity diagnostic.  It is not a spatial
Schmidt spectrum and must not be described as spatial entanglement.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np


DEFAULT_CAPTURE_FRACTIONS: tuple[float, float, float] = (0.9, 0.99, 0.999)


def _validated_actions(action_sequences: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    actions = np.asarray(action_sequences)
    if actions.ndim != 2:
        raise ValueError("action_sequences must be a rank-two array")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    keys = [tuple(int(value) for value in row) for row in actions]
    if len(set(keys)) != len(keys):
        raise ValueError("action_sequences must uniquely identify basis states")
    return np.asarray(actions, dtype=np.int32)


def _validated_values(values: np.ndarray | Sequence[complex], size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.shape != (size,):
        raise ValueError("values must contain one amplitude per action sequence")
    if np.any(~np.isfinite(vector)):
        raise ValueError("values must be finite")
    return vector


def _validated_prefix_length(prefix_length: int, action_count: int) -> int:
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, Integral):
        raise TypeError("prefix_length must be an integer")
    if not 0 <= prefix_length <= action_count:
        raise ValueError("prefix_length lies outside the action sequence")
    return int(prefix_length)


@dataclass(frozen=True)
class FusionActionCutMatrix:
    """Zero-filled amplitude matrix at one action prefix/suffix cut."""

    values: np.ndarray
    prefix_keys: tuple[tuple[int, ...], ...]
    suffix_keys: tuple[tuple[int, ...], ...]
    prefix_length: int
    valid_combination_count: int

    @property
    def prefix_count(self) -> int:
        return len(self.prefix_keys)

    @property
    def suffix_count(self) -> int:
        return len(self.suffix_keys)

    @property
    def possible_combination_count(self) -> int:
        return self.prefix_count * self.suffix_count

    @property
    def invalid_combination_count(self) -> int:
        return self.possible_combination_count - self.valid_combination_count

    @property
    def invalid_combination_fraction(self) -> float:
        total = self.possible_combination_count
        return self.invalid_combination_count / total if total else 0.0


def fusion_action_cut_matrix(
    values: np.ndarray | Sequence[complex],
    action_sequences: np.ndarray | Sequence[Sequence[int]],
    prefix_length: int,
) -> FusionActionCutMatrix:
    """Matricize a vector by action prefix/suffix, filling invalid pairs by zero."""

    actions = _validated_actions(action_sequences)
    vector = _validated_values(values, len(actions))
    prefix_length = _validated_prefix_length(prefix_length, actions.shape[1])
    prefixes = tuple(
        sorted({tuple(int(value) for value in row[:prefix_length]) for row in actions})
    )
    suffixes = tuple(
        sorted({tuple(int(value) for value in row[prefix_length:]) for row in actions})
    )
    prefix_position = {key: index for index, key in enumerate(prefixes)}
    suffix_position = {key: index for index, key in enumerate(suffixes)}
    matrix = np.zeros((len(prefixes), len(suffixes)), dtype=np.complex128)
    for row, value in zip(actions, vector, strict=True):
        prefix = tuple(int(item) for item in row[:prefix_length])
        suffix = tuple(int(item) for item in row[prefix_length:])
        matrix[prefix_position[prefix], suffix_position[suffix]] = value
    return FusionActionCutMatrix(
        values=matrix,
        prefix_keys=prefixes,
        suffix_keys=suffixes,
        prefix_length=prefix_length,
        valid_combination_count=len(actions),
    )


def _capture_rank(probabilities: np.ndarray, fraction: float) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("capture fractions must lie in (0, 1]")
    return int(
        np.searchsorted(np.cumsum(probabilities), fraction, side="left") + 1
    )


def _capture_rank_key(fraction: float) -> str:
    percentage = 100.0 * fraction
    rounded = round(percentage)
    if abs(percentage - rounded) < 1.0e-10:
        return f"r{int(rounded)}"
    compact = f"{percentage:.10f}".rstrip("0").rstrip(".").replace(".", "")
    return f"r{compact}"


def fusion_action_cut_spectrum(
    values: np.ndarray | Sequence[complex],
    action_sequences: np.ndarray | Sequence[Sequence[int]],
    prefix_length: int,
    *,
    rank_tolerance: float = 1.0e-12,
    capture_fractions: Sequence[float] = DEFAULT_CAPTURE_FRACTIONS,
) -> dict[str, object]:
    """Return normalized singular spectrum and predictive rank diagnostics.

    ``numeric_rank`` uses ``rank_tolerance * largest_singular_value``.  The
    vector is normalized before the SVD, so the spectrum and every reported
    rank are invariant under a nonzero global rescaling.
    """

    if not np.isfinite(rank_tolerance) or rank_tolerance < 0.0:
        raise ValueError("rank_tolerance must be finite and non-negative")
    fractions = tuple(float(value) for value in capture_fractions)
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("capture_fractions must be a nonempty subset of (0, 1]")
    if any(right <= left for left, right in zip(fractions, fractions[1:])):
        raise ValueError("capture_fractions must be strictly increasing")

    actions = _validated_actions(action_sequences)
    vector = _validated_values(values, len(actions))
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("values must have positive finite norm")
    cut = fusion_action_cut_matrix(vector / norm, actions, prefix_length)
    singular = np.linalg.svd(cut.values, compute_uv=False)
    squared = np.asarray(singular**2, dtype=np.float64)
    squared /= float(np.sum(squared))
    positive = squared > 0.0
    entropy_bits = float(-np.sum(squared[positive] * np.log2(squared[positive])))
    largest = float(singular[0])
    threshold = rank_tolerance * largest
    numeric_rank = int(np.sum(singular > threshold))
    capture_ranks = {
        _capture_rank_key(fraction): _capture_rank(squared, fraction)
        for fraction in fractions
    }
    effective_rank = float(2.0**entropy_bits)
    return {
        "prefix_length": cut.prefix_length,
        "prefix_count": cut.prefix_count,
        "suffix_count": cut.suffix_count,
        "valid_combination_count": cut.valid_combination_count,
        "possible_combination_count": cut.possible_combination_count,
        "invalid_combination_count": cut.invalid_combination_count,
        "invalid_combination_fraction": cut.invalid_combination_fraction,
        "input_norm": norm,
        "rank_tolerance": float(rank_tolerance),
        "numeric_rank_threshold": float(threshold),
        "numeric_rank": numeric_rank,
        "maximum_possible_rank": int(min(cut.values.shape)),
        "entropy_bits": entropy_bits,
        "effective_rank": effective_rank,
        "effective_schmidt_rank": effective_rank,
        "participation_rank": float(1.0 / np.sum(squared**2)),
        **capture_ranks,
        "normalized_singular_values": [float(value) for value in singular],
        "schmidt_probabilities": [float(value) for value in squared],
    }


def _relative_metrics(
    spectrum: Mapping[str, object], base: Mapping[str, object], rank_keys: Sequence[str]
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "numeric_rank_delta": float(spectrum["numeric_rank"] - base["numeric_rank"]),
        "numeric_rank_ratio": float(spectrum["numeric_rank"] / base["numeric_rank"]),
        "entropy_bits_delta": float(spectrum["entropy_bits"] - base["entropy_bits"]),
        "effective_rank_delta": float(spectrum["effective_rank"] - base["effective_rank"]),
        "effective_rank_ratio": float(spectrum["effective_rank"] / base["effective_rank"]),
        "participation_rank_ratio": float(
            spectrum["participation_rank"] / base["participation_rank"]
        ),
    }
    for key in rank_keys:
        metrics[f"{key}_delta"] = float(spectrum[key] - base[key])
        metrics[f"{key}_ratio"] = float(spectrum[key] / base[key])
    return metrics


def compare_fusion_action_cut_spectra(
    vectors: Mapping[str, np.ndarray | Sequence[complex]],
    action_sequences: np.ndarray | Sequence[Sequence[int]],
    *,
    base_name: str,
    rank_tolerance: float = 1.0e-12,
    capture_fractions: Sequence[float] = DEFAULT_CAPTURE_FRACTIONS,
) -> tuple[dict[str, object], ...]:
    """Evaluate every cut for several vectors and compare each with ``base``."""

    if not vectors:
        raise ValueError("vectors must not be empty")
    if base_name not in vectors:
        raise ValueError("base_name is absent from vectors")
    actions = _validated_actions(action_sequences)
    names = tuple(vectors)
    concrete = {
        name: _validated_values(values, len(actions))
        for name, values in vectors.items()
    }
    fractions = tuple(float(value) for value in capture_fractions)
    rank_keys = tuple(_capture_rank_key(fraction) for fraction in fractions)
    rows = []
    for prefix_length in range(actions.shape[1] + 1):
        spectra = {
            name: fusion_action_cut_spectrum(
                concrete[name],
                actions,
                prefix_length,
                rank_tolerance=rank_tolerance,
                capture_fractions=fractions,
            )
            for name in names
        }
        base = spectra[base_name]
        for name, spectrum in spectra.items():
            spectrum["relative_to_base"] = _relative_metrics(
                spectrum, base, rank_keys
            )
        rows.append(
            {
                "prefix_length": prefix_length,
                "prefix_count": int(base["prefix_count"]),
                "suffix_count": int(base["suffix_count"]),
                "invalid_combination_fraction": float(
                    base["invalid_combination_fraction"]
                ),
                "spectra": spectra,
            }
        )
    return tuple(rows)


__all__ = [
    "DEFAULT_CAPTURE_FRACTIONS",
    "FusionActionCutMatrix",
    "compare_fusion_action_cut_spectra",
    "fusion_action_cut_matrix",
    "fusion_action_cut_spectrum",
]
