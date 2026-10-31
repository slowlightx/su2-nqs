"""Small physical tangent-subspace diagnostics, never a production optimizer.

Inputs are log-amplitude derivatives of a *complete physical amplitude*,
including any coherent spatial projection. The columns are a frozen panel
of parameter directions, not magnetic labels or new variational heads.
Independent chain folds estimate a common Fubini--Study metric. Its null
space removes constant-amplitude/global-phase gauges; disagreement is
penalized only after whitening by that physical metric.

No Hamiltonian, neural model, sample generation, or held-out-set selection
belongs here. A finite-sample metric is not a global Hilbert-space bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def tangent_metric(scores: np.ndarray) -> np.ndarray:
    """Real FS metric for real parameter directions and complex scores.

    Center over the complete supplied Born sample, NOT within each chain:
    doing the latter can erase between-basin physical fluctuations.
    """
    values = np.asarray(scores, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError('at least two rows and one frozen tangent required')
    if not np.isfinite(values).all():
        raise ValueError('finite complete-amplitude tangent scores required')
    centered = values-values.mean(axis=0, keepdims=True)
    return (centered.conj().T @ centered).real/len(values)


def chain_fold_metrics(scores: np.ndarray, chain_labels: np.ndarray,
                       folds: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Build equal-fold metrics with disjoint CHAIN identities.

    Every supplied chain must be assigned once. Fold sample counts must
    agree, avoiding an implicit dominance of a longer chain. All folds use
    the SAME pooled training-score mean: their mean metric then equals the
    pooled FS covariance exactly, preserving between-fold/basin structure.
    The caller reserves evaluation chains BEFORE calling this function.
    """
    values = np.asarray(scores)
    labels = np.asarray(chain_labels)
    if (values.ndim != 2 or labels.shape != (len(values),)
            or not np.issubdtype(labels.dtype, np.integer) or len(folds) < 2):
        raise ValueError('scores, integer chain labels, and at least two folds required')
    flat = tuple(item for fold in folds for item in fold)
    if (any(isinstance(item, bool) or not isinstance(item, Integral) for item in flat)
            or len(set(flat)) != len(flat) or set(flat) != set(labels.tolist())):
        raise ValueError('folds must partition all supplied chains exactly once')
    rows = [values[np.isin(labels, fold)] for fold in folds]
    if len(set(map(len, rows))) != 1 or min(map(len, rows)) < 2:
        raise ValueError('equal nonempty fold row counts required')
    if not np.isfinite(values).all():
        raise ValueError('finite complete-amplitude tangent scores required')
    center = values.mean(axis=0)
    return np.stack([((row-center).conj().T @ (row-center)).real/len(row) for row in rows])


@dataclass(frozen=True)
class ConsensusTangent:
    """A frozen subspace direction and physically meaningful fit diagnostics."""

    coefficients: np.ndarray
    common_metric: np.ndarray
    physical_rank: int
    null_force_norm: float
    predicted_common_fs: float
    predicted_fold_fs: np.ndarray
    directional_slope: float
    whitened_disagreement_eigenvalues: np.ndarray
    disagreement_strength: float


def consensus_tangent_direction(metrics: np.ndarray, force: np.ndarray, *,
                                disagreement_strength: float,
                                target_tangent_fs: float,
                                rank_relative_tolerance: float = 1e-12,
                                null_force_tolerance: float = 1e-9) -> ConsensusTangent:
    """Solve a gauge-quotient, cross-chain-regularized small natural gradient.

    Let G be the mean fold metric and W whiten its nonnull support. For
    B_f=W.T G_f W, use R=mean_f (B_f-I)^2, then
    u=-(I+lambda R)^-1 W.T force and c=W u. This is a *direction* correction,
    not scalar shrinking. Normalize once in the common physical metric.

    The chosen lambda and direction panel must be fixed without evaluation
    chains. Those chains can only report or veto, not tune this function.
    No coordinate-Euclidean ridge is introduced. In exact arithmetic, an
    invertible change of panel coordinates leaves the physical direction
    unchanged. A relative numerical rank cutoff is explicitly reported;
    arbitrary singular/ill-conditioned coordinate maps cannot be certified.
    """
    if ((np.iscomplexobj(metrics) and np.any(np.asarray(metrics).imag != 0))
            or (np.iscomplexobj(force) and np.any(np.asarray(force).imag != 0))):
        raise ValueError('real physical metrics and real-parameter force required')
    gs = np.asarray(metrics, dtype=np.float64)
    g = np.asarray(force, dtype=np.float64)
    scales = (disagreement_strength, target_tangent_fs,
              rank_relative_tolerance, null_force_tolerance)
    if (gs.ndim != 3 or gs.shape[0] < 2 or gs.shape[1] < 1 or gs.shape[1] != gs.shape[2]
            or g.shape != (gs.shape[1],) or not np.isfinite(gs).all()
            or not np.isfinite(g).all() or not np.isfinite(scales).all()
            or disagreement_strength < 0 or not 0 < target_tangent_fs < 1
            or not 0 < rank_relative_tolerance < 1 or null_force_tolerance < 0):
        raise ValueError('invalid fold metrics, force, or frozen regularization scales')
    if not np.allclose(gs, gs.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError('real symmetric physical metrics required')
    mean = gs.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(mean)
    scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    if np.linalg.eigvalsh(gs).min() < -rank_relative_tolerance*scale:
        raise ValueError('fold metric has a negative physical direction')
    retained = eigenvalues > rank_relative_tolerance*scale
    null_force = float(np.linalg.norm(eigenvectors[:, ~retained].T @ g))
    if not retained.any():
        raise ValueError('panel has no nonnull physical tangent')
    if null_force > null_force_tolerance*max(float(np.linalg.norm(g)), np.finfo(float).tiny):
        raise ValueError('force has unresolved support in a metric-null direction')
    whitening = eigenvectors[:, retained]/np.sqrt(eigenvalues[retained])
    blocks = np.einsum('ki,fkl,lj->fij', whitening, gs, whitening, optimize=True)
    centered = blocks-np.eye(len(eigenvalues[retained]))
    disagreement = np.einsum('fij,fjk->ik', centered, centered, optimize=True)/len(gs)
    reduced_force = whitening.T @ g
    reduced = -np.linalg.solve(np.eye(len(reduced_force))+
                              disagreement_strength*disagreement, reduced_force)
    norm = float(np.linalg.norm(reduced))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError('zero or nonfinite physical direction')
    coefficients = whitening @ (target_tangent_fs*reduced/norm)
    sizes = np.einsum('i,fij,j->f', coefficients, gs, coefficients, optimize=True)
    return ConsensusTangent(coefficients, mean, int(retained.sum()), null_force,
        float(np.sqrt(coefficients @ mean @ coefficients)),
        np.sqrt(np.maximum(sizes, 0.)), float(g @ coefficients),
        np.linalg.eigvalsh(disagreement), float(disagreement_strength))


def fixed_direction_report(scores: np.ndarray, coefficients: np.ndarray,
                           *, reference_fs: float) -> dict:
    """Evaluate one frozen direction without changing it or selecting lambda."""
    c = np.asarray(coefficients, dtype=np.float64)
    metric = tangent_metric(scores)
    if (c.shape != (metric.shape[0],) or not np.isfinite(c).all()
            or not np.isfinite(reference_fs) or reference_fs <= 0):
        raise ValueError('finite matching coefficients and positive reference distance required')
    predicted = float(np.sqrt(max(float(c @ metric @ c), 0.)))
    return dict(tangent_fs=predicted, relative_to_reference=predicted/reference_fs,
                statement='linearized held-out physical metric; not a finite-state distance or energy audit')
