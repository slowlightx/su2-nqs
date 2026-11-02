"""Exact-small diagnostics for whether a dressing adds physical directions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _normalized_state(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite complex vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("state must have nonzero norm")
    return vector / norm


def projective_real_tangent_matrix(
    state: np.ndarray, tangents: np.ndarray
) -> np.ndarray:
    """Return real-coordinate tangents orthogonal to norm and global phase.

    Columns of ``tangents`` are derivatives with respect to real parameters.
    Subtracting the full complex overlap with the normalized state removes
    both the radial direction and the physically redundant global phase.
    """

    vector = _normalized_state(state)
    values = np.asarray(tangents, dtype=np.complex128)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or values.shape[0] != vector.size
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("tangents must be a finite (dimension, count) matrix")
    projected = values - vector[:, None] * (vector.conj() @ values)[None, :]
    return np.vstack((projected.real, projected.imag))


def _subspace(matrix: np.ndarray, relative_cutoff: float):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or np.any(~np.isfinite(values)):
        raise ValueError("tangent matrix must be finite and two-dimensional")
    if not 0.0 < relative_cutoff < 1.0:
        raise ValueError("relative_cutoff must lie between zero and one")
    if values.shape[1] == 0:
        return np.empty((values.shape[0], 0)), np.empty(0), 0.0
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    scale = float(singular[0]) if singular.size else 0.0
    retained = singular > relative_cutoff * scale if scale > 0.0 else np.zeros_like(singular, dtype=bool)
    return left[:, retained], singular, scale


def _capture_fraction(basis: np.ndarray, vector: np.ndarray) -> float:
    denominator = float(np.dot(vector, vector))
    if denominator <= 1.0e-30:
        return 0.0
    return float(np.dot(basis.T @ vector, basis.T @ vector) / denominator)


def tangent_extension_diagnostics(
    state: np.ndarray,
    base_tangents: np.ndarray,
    added_tangents: np.ndarray,
    *,
    residuals: Mapping[str, np.ndarray] | None = None,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Audit rank novelty and residual capture of added real parameters."""

    vector = _normalized_state(state)
    base = projective_real_tangent_matrix(vector, base_tangents)
    added = projective_real_tangent_matrix(vector, added_tangents)
    combined = np.column_stack((base, added))
    base_basis, base_singular, _ = _subspace(base, relative_cutoff)
    added_basis, added_singular, _ = _subspace(added, relative_cutoff)
    combined_basis, combined_singular, _ = _subspace(combined, relative_cutoff)
    perpendicular = added - base_basis @ (base_basis.T @ added)
    perpendicular_norm_squared = float(np.linalg.norm(perpendicular) ** 2)
    added_norm_squared = float(np.linalg.norm(added) ** 2)
    result: dict[str, object] = {
        "hilbert_complex_dimension": int(vector.size),
        "projective_real_dimension": int(2 * vector.size - 2),
        "base_parameter_count": int(base.shape[1]),
        "added_parameter_count": int(added.shape[1]),
        "relative_svd_cutoff": float(relative_cutoff),
        "base_rank": int(base_basis.shape[1]),
        "added_rank": int(added_basis.shape[1]),
        "combined_rank": int(combined_basis.shape[1]),
        "incremental_rank": int(combined_basis.shape[1] - base_basis.shape[1]),
        "added_perpendicular_fraction": (
            perpendicular_norm_squared / max(added_norm_squared, 1.0e-300)
        ),
        "base_singular_values": [float(value) for value in base_singular],
        "added_singular_values": [float(value) for value in added_singular],
        "combined_smallest_retained_singular_value": (
            float(combined_singular[combined_basis.shape[1] - 1])
            if combined_basis.shape[1]
            else 0.0
        ),
    }
    capture = {}
    for name, raw_residual in ({} if residuals is None else residuals).items():
        residual = np.asarray(raw_residual, dtype=np.complex128)
        if residual.shape != vector.shape or np.any(~np.isfinite(residual)):
            raise ValueError(f"residual {name!r} is incompatible with state")
        projective = residual - vector * np.vdot(vector, residual)
        real_residual = np.concatenate((projective.real, projective.imag))
        base_fraction = _capture_fraction(base_basis, real_residual)
        added_fraction = _capture_fraction(added_basis, real_residual)
        combined_fraction = _capture_fraction(combined_basis, real_residual)
        capture[str(name)] = {
            "base_fraction": base_fraction,
            "added_alone_fraction": added_fraction,
            "combined_fraction": combined_fraction,
            "incremental_fraction": max(0.0, combined_fraction - base_fraction),
        }
    result["residual_capture"] = capture
    return result


def reynolds_average_generators(
    generators: Sequence[np.ndarray], actions: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Average local generators under a finite unitary spatial action."""

    values = tuple(np.asarray(generator, dtype=np.complex128) for generator in generators)
    group = tuple(np.asarray(action, dtype=np.complex128) for action in actions)
    if not values or not group:
        raise ValueError("generators and actions must be nonempty")
    dimension = values[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in values):
        raise ValueError("all generators must be equal-size square matrices")
    identity = np.eye(dimension)
    if any(
        action.shape != (dimension, dimension)
        or np.max(np.abs(action.conj().T @ action - identity)) > 1.0e-12
        for action in group
    ):
        raise ValueError("actions must be equal-size unitary matrices")
    return tuple(
        sum(action @ generator @ action.conj().T for action in group) / len(group)
        for generator in values
    )


__all__ = [
    "projective_real_tangent_matrix",
    "reynolds_average_generators",
    "tangent_extension_diagnostics",
]
