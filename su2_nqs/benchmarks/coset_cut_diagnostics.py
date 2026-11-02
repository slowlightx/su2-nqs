"""Exact conditional diagnostics for a symmetry-projected fusion-tree state."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .causal_residual import martingale_residual_decomposition
from .conditional_response import (
    complex_martingale_decomposition,
    conditional_kl_decomposition,
)


def _normalized_state(values: Sequence[complex] | np.ndarray, name: str) -> np.ndarray:
    state = np.asarray(values, dtype=np.complex128)
    if state.ndim != 1 or np.any(~np.isfinite(state)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError(f"{name} must have positive norm")
    return state / norm


def coset_cut_diagnostics(
    reference: Sequence[complex] | np.ndarray,
    target: Sequence[complex] | np.ndarray,
    hamiltonian: object,
    action_sequences: np.ndarray,
) -> dict[str, object]:
    """Resolve target mismatch and Hamiltonian residual over tree decisions.

    The inputs may already be projected into a momentum sector, but their
    coordinates must be the complete fusion-tree basis so that action strings
    define an exact filtration.  All conditional tables are exhaustive
    diagnostics; they are not exposed to a production sampler.
    """

    source = _normalized_state(reference, "reference")
    exact = _normalized_state(target, "target")
    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or actions.shape[0] != len(source):
        raise ValueError("action_sequences must have one row per state")
    if len(exact) != len(source):
        raise ValueError("reference and target dimensions disagree")
    if np.shape(hamiltonian) != (len(source), len(source)):
        raise ValueError("Hamiltonian dimension disagrees with the states")

    probabilities = np.abs(source) ** 2
    target_probabilities = np.abs(exact) ** 2
    applied = np.asarray(hamiltonian @ source, dtype=np.complex128)
    active = probabilities > 0.0
    hamiltonian_leakage = float(np.linalg.norm(applied[~active]))
    target_leakage = float(np.linalg.norm(exact[~active]))
    if target_leakage > 1.0e-12:
        raise ValueError(
            "target support is not contained in the reference support"
        )
    local_energy = np.divide(
        applied,
        source,
        out=np.zeros_like(applied),
        where=active,
    )
    kl = conditional_kl_decomposition(
        probabilities, target_probabilities, actions
    )
    doob = martingale_residual_decomposition(
        probabilities[active], local_energy[active], actions[active]
    )
    response = complex_martingale_decomposition(source, exact, actions)
    decisions = []
    for decision, (kl_row, doob_row, response_row) in enumerate(
        zip(kl["decisions"], doob["decisions"], response["rows"], strict=True)
    ):
        decisions.append(
            {
                "decision": decision,
                "conditional_kl_bits": kl_row["kl_contribution_bits"],
                "conditional_kl_fraction": kl_row["kl_fraction"],
                "hamiltonian_doob_variance": doob_row["variance_contribution"],
                "hamiltonian_doob_fraction": doob_row["variance_fraction"],
                "wavefunction_response_norm": response_row[
                    "wavefunction_response_norm"
                ],
                "wavefunction_response_fraction": response_row[
                    "wavefunction_response_fraction"
                ],
            }
        )
    return {
        "conditional_born_kl": kl,
        "hamiltonian_doob": doob,
        "complex_response": response,
        "active_support_size": int(np.sum(active)),
        "hamiltonian_support_leakage_norm": hamiltonian_leakage,
        "target_support_leakage_norm": target_leakage,
        "decisions": decisions,
    }


__all__ = ["coset_cut_diagnostics"]
