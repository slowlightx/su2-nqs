"""Exact-small causal Hamiltonian-Doob features for frozen-base diagnosis.

This module is an oracle, not a scalable model: it contracts all suffixes to
ask whether exact operator-aligned conditional innovations would repair a
frozen autoregressive state.  The compiled table is nevertheless strictly a
function of the revealed prefix and candidate action, and it uses only the
base state and a supplied Hamiltonian, never a target eigenvector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoobCandidateFeatures:
    """Complex Doob innovations for every prefix and candidate action."""

    innovations: np.ndarray
    real_imag_features: np.ndarray
    mean: complex
    maximum_selected_reconstruction_error: float
    maximum_prefix_invariance_error: float
    target_amplitude_used: bool = False

    @property
    def feature_count(self) -> int:
        return int(self.real_imag_features.shape[-1])


def compile_doob_candidate_features(
    probabilities: np.ndarray,
    values: np.ndarray,
    action_sequences: np.ndarray,
    action_dim: int,
) -> DoobCandidateFeatures:
    """Compile conditional innovations with exact suffix contractions.

    ``values`` is commonly the frozen-base local energy ``(H psi)/psi``.
    Selected innovations telescope to ``values - E_p[values]``.  Candidate
    entries that never occur after a prefix remain zero and must be excluded
    by the model's exact reachability mask.
    """

    weights = np.asarray(probabilities, dtype=np.float64)
    observable = np.asarray(values, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if weights.ndim != 1 or observable.shape != weights.shape:
        raise ValueError("probabilities and values must be equal-length vectors")
    if actions.ndim != 2 or len(actions) != len(weights):
        raise ValueError("action_sequences must have one row per value")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    if action_dim < 1 or np.any(actions < 0) or np.any(actions >= action_dim):
        raise ValueError("action_dim does not contain every action")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("the exact-small compiler requires positive finite weights")
    if np.any(~np.isfinite(observable)):
        raise ValueError("values must be finite")
    weights = weights / np.sum(weights)
    mean = complex(np.sum(weights * observable))
    innovations = np.zeros(
        (len(weights), actions.shape[1], action_dim), dtype=np.complex128
    )
    prefix_error = 0.0

    for decision in range(actions.shape[1]):
        groups: dict[tuple[int, ...], list[int]] = {}
        for state, row in enumerate(actions):
            key = tuple(int(value) for value in row[:decision])
            groups.setdefault(key, []).append(state)
        for members in groups.values():
            indices = np.asarray(members, dtype=np.int64)
            parent_weights = weights[indices]
            parent_mean = np.sum(parent_weights * observable[indices]) / np.sum(
                parent_weights
            )
            by_action: dict[int, list[int]] = {}
            for state in members:
                by_action.setdefault(int(actions[state, decision]), []).append(state)
            candidate_values = np.zeros(action_dim, dtype=np.complex128)
            for action, children in by_action.items():
                child_indices = np.asarray(children, dtype=np.int64)
                child_weights = weights[child_indices]
                child_mean = np.sum(
                    child_weights * observable[child_indices]
                ) / np.sum(child_weights)
                candidate_values[action] = child_mean - parent_mean
            innovations[indices, decision, :] = candidate_values
            prefix_error = max(
                prefix_error,
                float(
                    np.max(
                        np.abs(
                            innovations[indices, decision]
                            - innovations[indices[0], decision]
                        ),
                        initial=0.0,
                    )
                ),
            )

    selected = np.take_along_axis(
        innovations, actions[..., None], axis=2
    )[..., 0]
    reconstructed = mean + np.sum(selected, axis=1)
    reconstruction_error = float(
        np.max(np.abs(reconstructed - observable), initial=0.0)
    )
    features = np.stack((np.real(innovations), np.imag(innovations)), axis=-1)
    return DoobCandidateFeatures(
        innovations=innovations,
        real_imag_features=np.asarray(features, dtype=np.float64),
        mean=mean,
        maximum_selected_reconstruction_error=reconstruction_error,
        maximum_prefix_invariance_error=prefix_error,
    )


__all__ = ["DoobCandidateFeatures", "compile_doob_candidate_features"]
