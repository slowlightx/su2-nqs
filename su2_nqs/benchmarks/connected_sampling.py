"""Exact diagnostics for Hamiltonian-connected sampling proposals.

These routines are deliberately independent of a neural model.  They operate
on a probability vector in the same basis as a Hamiltonian with connectivity
``H[new_state, old_state]``.  The connected distribution first draws an old
state from the base distribution and then an off-diagonal Hamiltonian edge
with probability proportional to its squared matrix element.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _normalized_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("probabilities must be a vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    normalization = float(np.sum(values))
    if normalization <= 0.0:
        raise ValueError("probabilities must have positive total weight")
    return values / normalization


def hamiltonian_connected_distribution(
    hamiltonian,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Push a base distribution through one off-diagonal ``|H|^2`` edge.

    Each Hamiltonian column defines a normalized conditional distribution over
    its off-diagonal neighbors.  A column without any off-diagonal entry uses
    a self-loop, so normalization is preserved even for isolated states.
    Repeated physical terms must already have been combined in ``hamiltonian``.
    """

    base = _normalized_probabilities(probabilities)
    if hamiltonian.shape != (len(base), len(base)):
        raise ValueError("Hamiltonian and probability dimensions do not match")

    if sparse.issparse(hamiltonian):
        weights = hamiltonian.tocsc(copy=True)
        weights.setdiag(0.0)
        weights.eliminate_zeros()
        weights.data = np.abs(weights.data) ** 2
        column_weights = np.asarray(weights.sum(axis=0)).reshape(-1).real
        scaled = np.divide(
            base,
            column_weights,
            out=np.zeros_like(base),
            where=column_weights > 0.0,
        )
        connected = np.asarray(weights @ scaled).reshape(-1).real
    else:
        matrix = np.asarray(hamiltonian)
        if matrix.ndim != 2:
            raise ValueError("Hamiltonian must be a matrix")
        weights = np.abs(matrix) ** 2
        weights = weights.copy()
        np.fill_diagonal(weights, 0.0)
        column_weights = np.sum(weights, axis=0)
        scaled = np.divide(
            base,
            column_weights,
            out=np.zeros_like(base),
            where=column_weights > 0.0,
        )
        connected = weights @ scaled

    isolated = column_weights <= 0.0
    connected[isolated] += base[isolated]
    connected = np.maximum(connected, 0.0)
    connected /= np.sum(connected)
    return connected


def connected_mixture_distribution(
    hamiltonian,
    probabilities: np.ndarray,
    *,
    connected_fraction: float,
) -> np.ndarray:
    """Mix Born sampling with one-hop Hamiltonian-connected sampling."""

    if not 0.0 <= connected_fraction <= 1.0:
        raise ValueError("connected_fraction must lie in [0, 1]")
    base = _normalized_probabilities(probabilities)
    connected = hamiltonian_connected_distribution(hamiltonian, base)
    proposal = (1.0 - connected_fraction) * base + connected_fraction * connected
    proposal /= np.sum(proposal)
    return proposal


def probability_importance_diagnostics(
    proposal_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    zero_tolerance: float = 0.0,
) -> dict[str, float | bool]:
    """Return exact target-over-proposal tail and ESS diagnostics.

    A positive ``target_weight_outside_proposal_support`` means ordinary
    importance sampling is invalid, irrespective of the conditional ESS on
    the covered support.  ESS is evaluated with log-domain second moments.
    """

    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    proposal = _normalized_probabilities(proposal_probabilities)
    target = _normalized_probabilities(target_probabilities)
    if proposal.shape != target.shape:
        raise ValueError("proposal and target dimensions do not match")
    support = proposal > zero_tolerance
    leakage = float(np.sum(target[~support]))
    covered_target = target[support]
    covered_proposal = proposal[support]
    covered_mass = float(np.sum(covered_target))
    active = covered_target > 0.0
    if not np.any(active):
        return {
            "effective_sample_size_fraction": 0.0,
            "covered_target_mass": 0.0,
            "target_weight_outside_proposal_support": leakage,
            "maximum_normalized_weight": 0.0,
            "weight_second_moment": 0.0,
            "importance_sampling_valid": False,
        }

    log_second_terms = (
        2.0 * np.log(covered_target[active])
        - np.log(covered_proposal[active])
    )
    maximum_log = float(np.max(log_second_terms))
    log_second = maximum_log + float(
        np.log(np.sum(np.exp(log_second_terms - maximum_log)))
    )
    log_ess = 2.0 * np.log(covered_mass) - log_second
    log_weights = (
        np.log(covered_target[active])
        - np.log(covered_proposal[active])
        - np.log(covered_mass)
    )
    return {
        "effective_sample_size_fraction": float(np.exp(log_ess)),
        "covered_target_mass": covered_mass,
        "target_weight_outside_proposal_support": leakage,
        "maximum_normalized_weight": float(np.exp(np.max(log_weights))),
        "weight_second_moment": float(np.exp(log_second)),
        "importance_sampling_valid": bool(leakage == 0.0),
    }
