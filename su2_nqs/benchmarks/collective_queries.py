"""Coherent low-rank physical tangent queries for frozen CT dictionaries.

This module never chooses a subspace from force labels.  The caller freezes
Q before requesting scores OQ.  Only the requested columns are accepted by
the SR fit; cached full dictionaries are useful for independent references,
but are not inputs to the restricted solver.  No finite-step energy, support
outside parent Born samples, or wall-clock saving is certified here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .projected_capacity import tangent_moments, heldout_direction_metrics


def collective_subspace(embedding: np.ndarray, rank: int) -> np.ndarray:
    """Return deterministic orthonormal collective columns, not token top-k.

    ``embedding`` has one row per already verified legal reduced-tensor
    direction.  All columns span complete SU(2)-scalar tangents; their linear
    combinations preserve the existing coherent spatial projector.  The SVD
    prioritizes embedding variance only, never target energy or force.
    """
    embedding = np.asarray(embedding, float)
    if (embedding.ndim != 2 or not np.isfinite(embedding).all()
            or isinstance(rank, bool) or not isinstance(rank, (int, np.integer))
            or rank < 1 or rank > min(embedding.shape)):
        raise ValueError('finite matrix and positive feasible collective rank required')
    left, values, _ = np.linalg.svd(embedding, full_matrices=False)
    if values[rank-1] <= max(values[0]*1e-12, 1e-20):
        raise ValueError('embedding lacks the requested numerical rank')
    basis = left[:, :rank].copy()
    for column in range(rank):
        pivot = int(np.argmax(abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis


@dataclass(frozen=True)
class CollectiveUpdate:
    """Restricted SR result; coefficients refer only to queried Q columns."""

    coefficients: np.ndarray
    gram: np.ndarray
    force: np.ndarray
    regularization_scale: float


def fit_collective_update(
    queried_scores: np.ndarray, local_energies: np.ndarray, *,
    ridge: float = .01, fs_radius: float = .001,
) -> CollectiveUpdate:
    """Fit a real physical update using r queries, not the unqueried force.

    Ridge is isotropic within an orthonormal coordinate subspace, with scale
    tr(S)/r.  It therefore respects Q -> Q R for orthogonal R; arbitrary
    nonorthogonal coordinate transformations are not claimed.  Constant log
    amplitude/phase gauges are removed by centered covariance.  Null physical
    modes are discarded.  Force is Re Cov(O,E), half the energy derivative.
    """
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(fs_radius) or fs_radius <= 0:
        raise ValueError('nonnegative finite ridge and positive FS radius required')
    queried_scores = np.asarray(queried_scores)
    if queried_scores.ndim != 2 or queried_scores.shape[1] < 1:
        raise ValueError('at least one physical query column required')
    gram, force = tangent_moments(queried_scores, local_energies)
    values, vectors = np.linalg.eigh(gram)
    scale = max(float(np.trace(gram)/len(force)), 0.)
    threshold = max(float(values.max(initial=0.))*1e-12, 1e-24)
    active = values > threshold
    inverse = np.zeros_like(values)
    inverse[active] = 1/(values[active]+ridge*scale)
    coefficients = -((vectors*inverse) @ (vectors.T @ force))
    norm = np.sqrt(max(float(coefficients @ gram @ coefficients), 0.))
    if norm > 1e-24:
        coefficients *= fs_radius/norm
    else:
        coefficients[:] = 0.
    return CollectiveUpdate(coefficients, gram, force, scale)


def audit_collective_update(
    coefficients: np.ndarray, queried_scores: np.ndarray,
    local_energies: np.ndarray, chain_labels: np.ndarray,
) -> dict:
    """Evaluate a frozen restricted update, with leave-whole-chain errors.

    Delete-chain values must be paired before comparing two controllers.
    They are not independent seeds and do not account for uncertainty in Q.
    """
    coefficients = np.asarray(coefficients, float)
    labels = np.asarray(chain_labels)
    if labels.shape != (len(queried_scores),) or len(np.unique(labels)) < 4:
        raise ValueError('at least four aligned independent chain groups required')
    gram, force = tangent_moments(queried_scores, local_energies)
    metrics = heldout_direction_metrics(coefficients, gram, force)
    chains = np.unique(labels)
    leave = []
    for chain in chains:
        keep = labels != chain
        subgram, subforce = tangent_moments(queried_scores[keep], local_energies[keep])
        leave.append(heldout_direction_metrics(coefficients, subgram, subforce))
    for name in ('energy_slope_total', 'energy_slope_per_fs_total', 'fs_tangent'):
        values = np.asarray([row[name] for row in leave])
        metrics[name+'_chain_jackknife'] = values.tolist()
        metrics[name+'_chain_SE'] = float(np.sqrt((len(chains)-1)/len(chains)
            *np.sum((values-values.mean())**2)))
    metrics['chains'] = len(chains)
    return metrics
