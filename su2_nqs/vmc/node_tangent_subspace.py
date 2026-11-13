"""Couple precomputed real native-node directions in the full physical metric.

This module changes optimizer coordinates, not the wavefunction family. Each
column is the amplitude derivative of one complete native-node update. The
joint Gram retains all cross-node terms. No logarithmic derivative, H action,
new sampling, or Jacobian approximation is introduced.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np


def _array(value: Any, name: str, dimensions: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != dimensions or np.iscomplexobj(array)
            or array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
        raise ValueError(f"finite real {dimensions}-dimensional {name} required")
    return np.asarray(array, dtype=np.float64)


def stream_native_node_tangents(
    jacobian: np.ndarray, direction: Any, *,
    blocks: Sequence[tuple[str, int, int]], parameter_chunk: int = 16384,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> np.ndarray:
    """One exact cached-J pass, returning one amplitude tangent per node.

    ``blocks`` partitions every native column. All sample rows remain. Only
    one small parameter chunk is materialized; no records-by-parameters-by-
    nodes array or advanced-indexed copy of the full cache is constructed.
    """
    if (getattr(jacobian, "ndim", None) != 2 or jacobian.dtype != np.float64
            or not isinstance(parameter_chunk, Integral) or isinstance(parameter_chunk, bool)
            or parameter_chunk < 1):
        raise ValueError("real float64 matrix and positive integer parameter_chunk required")
    native = _array(direction, "native direction", 1)
    rows, parameters = jacobian.shape
    if native.shape != (parameters,) or rows < 2 or not blocks:
        raise ValueError("complete native direction and at least two rows required")
    cursor = 0
    labels: set[str] = set()
    for label, first, final in blocks:
        if (not isinstance(label, str) or not label or label in labels
                or type(first) is not int or type(final) is not int
                or first != cursor or not first < final <= parameters):
            raise ValueError("named contiguous nodes must partition every native column")
        labels.add(label)
        cursor = final
    if cursor != parameters:
        raise ValueError("a native column was omitted")
    result = np.zeros((rows, len(blocks)), dtype=np.float64)
    for index, (label, first, final) in enumerate(blocks):
        started = perf_counter()
        for start in range(first, final, int(parameter_chunk)):
            stop = min(start + int(parameter_chunk), final)
            raw = np.ascontiguousarray(jacobian[:, start:stop])
            if not np.isfinite(raw).all():
                raise FloatingPointError("nonfinite cached native derivative")
            result[:, index] += raw @ native[start:stop]
        if progress is not None:
            progress(dict(stage="cached_native_node_image", label=label,
                          completed_nodes=index + 1, seconds=perf_counter() - started))
    return result


def solve_physical_node_subspace(
    psi: Any, hpsi: Any, density: Any, node_tangents: Any, *,
    relative_damping: float = .5, n_sites: int = 144,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one coupled imaginary-time/SR step in existing node directions.

    The real normalization tangent is projected globally with known covering
    density Q, including parent nodes. After column physical-norm scaling,
    solve the COMPLETE small Gram with the fixed sample-eigenvalue ridge
    ``relative_damping*trace(Gram)/(records-1)``. Pure normalization-gauge
    columns have zero coefficient. Real amplitudes have no independent phase
    tangent. Returned coefficients act on the supplied raw native directions.

    This FIT-only projection is not a finite-step or H-delta certificate.
    Physical radius and energy require fresh, disjoint nonlinear evaluations.
    """
    p, hp, q = (_array(x, name, 1) for x, name in
                ((psi, "psi"), (hpsi, "Hpsi"), (density, "density")))
    images = _array(node_tangents, "node amplitude tangents", 2)
    count = len(p)
    if (count < 2 or hp.shape != p.shape or q.shape != p.shape
            or images.shape[0] != count or images.shape[1] < 1 or np.any(q <= 0)
            or isinstance(relative_damping, bool) or not np.isfinite(relative_damping)
            or relative_damping <= 0 or not isinstance(n_sites, Integral)
            or isinstance(n_sites, bool) or n_sites < 1):
        raise ValueError("matching complete rows, positive Q/ridge, and integer site count required")
    amplitude_scale = float(np.max(abs(p)))
    if amplitude_scale == 0:
        raise ValueError("positive observed parent norm required")
    # A common scale avoids under/overflow without changing any quotient.
    importance = np.sqrt(float(np.max(q)) / q) / amplitude_scale
    parent = p * importance
    norm = float(parent @ parent)
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError("nonfinite observed parent norm")
    importance /= np.sqrt(norm)
    unit = p * importance
    weighted = images * importance[:, None]
    physical = weighted - unit[:, None] * (unit @ weighted)[None, :]
    column_norms = np.linalg.norm(physical, axis=0)
    raw_norms = np.linalg.norm(weighted, axis=0)
    active = column_norms > 64 * np.finfo(float).eps * raw_norms
    if not np.any(active):
        raise ArithmeticError("all node directions are normalization gauge or zero")
    canonical = physical[:, active] / column_norms[active]
    gram = canonical.T @ canonical
    gram = (gram + gram.T) / 2
    energy = float(unit @ (hp * importance))
    force = 2 * (hp - energy * p) * importance
    force -= unit * float(unit @ force)
    right = canonical.T @ force
    ridge = float(relative_damping * np.trace(gram) / (count - 1))
    canonical_coefficients = -np.linalg.solve(gram + ridge * np.eye(len(gram)), right)
    coefficients = np.zeros(images.shape[1], dtype=np.float64)
    coefficients[active] = canonical_coefficients / column_norms[active]
    tangent = physical @ coefficients
    residual = (gram + ridge * np.eye(len(gram))) @ canonical_coefficients + right
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(right), np.finfo(float).tiny))
    eigenvalues = np.linalg.eigvalsh(gram)
    if (not np.isfinite(coefficients).all() or not np.isfinite(tangent).all()
            or not np.isfinite(relative_residual) or relative_residual >= 1e-9
            or eigenvalues[0] < -1e-10 * max(eigenvalues[-1], 1.)):
        raise ArithmeticError("coupled node physical Gram solve failed")
    info = dict(
        geometry="complete_cross_node_physical_Gram_in_existing_native_directions",
        records=count, node_directions=images.shape[1], active_columns=np.flatnonzero(active).tolist(),
        pure_normalization_gauge_columns=np.flatnonzero(~active).tolist(),
        global_normalization_projected=True, phase_projection="real amplitudes: phase tangent absent",
        column_physical_norms=column_norms.tolist(), relative_damping=float(relative_damping),
        ridge=ridge, ridge_scale="relative_damping*trace(normalized_node_Gram)/(records-1)",
        Gram=gram.tolist(), Gram_eigenvalues=eigenvalues.tolist(),
        effective_rank=float(np.trace(gram)**2 / np.sum(gram**2)),
        exact_coupled_normal_relative_residual=relative_residual,
        fitted_energy_per_site=energy / n_sites,
        fitted_tangent_FS=float(np.linalg.norm(tangent)),
        predicted_Hermitian_energy_slope_per_site=float(force @ tangent / n_sites),
        fitted_force_remaining_fraction=float(np.linalg.norm(force + tangent)
                                               / max(np.linalg.norm(force), np.finfo(float).tiny)),
        tangent_parent_overlap=float(unit @ tangent),
        new_H_queries=0, new_derivative_queries=0, no_new_trainable_parameters=True,
        no_FIT_claim_of_independent_descent=True, finite_nonlinear_FS_and_H_confirmation_required=True)
    return coefficients, tangent, info
