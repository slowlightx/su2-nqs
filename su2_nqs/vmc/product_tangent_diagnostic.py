"""Full real-parameter tangents of a product of independent physical rays.

No optimizer or Hamiltonian construction lives here. Each ray has its own
phase/normalization gauge. Rank is computed from the un-squared Jacobian;
column equilibration is reported separately from the parameter-coordinate
spectrum that determines the effect of isotropic SR damping.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class ProductTangentDiagnostic:
    """Physical projection and reproducible product-FS design/target arrays."""
    horizontal_jacobians: np.ndarray
    design: np.ndarray
    target: np.ndarray
    projected_target: np.ndarray
    metrics: dict[str, object]


def diagnose_product_tangent(
    states: np.ndarray, actions: np.ndarray, jacobians: np.ndarray, *,
    n_sites: int, weights: np.ndarray | None = None,
    relative_cutoffs: tuple[float, ...] = (1e-6, 1e-8, 1e-10),
    damping: float = 1e-3,
) -> ProductTangentDiagnostic:
    """Project weighted energy residuals on a shared REAL parameter tangent.

    Shapes are ``states[K,D]``, ``actions[K,D]`` (TOTAL H units), and
    ``jacobians[K,D,P]``. Positive weights sum to one. The target is the
    concatenated residual ``sqrt(w_k)*(H_k-E_k)|psi_k>``; its norm squared is
    the mean individual variance, NOT the variance of a single Hamiltonian.
    This determines stationarity of the weighted mean objective; individual
    objectives need not be stationary at a shared optimum.

    The last supplied cutoff controls returned projection. Cutoff sensitivity,
    raw and column-equilibrated spectra must accompany any capacity claim.
    The damped response is ``D(D.T D + damping I)^-1 D.T r``; its captured
    norm is not an undamped orthogonal-projection fraction.
    """
    p, h, t = (np.asarray(x, dtype=np.complex128) for x in (states, actions, jacobians))
    if (p.ndim != 2 or not all(p.shape) or h.shape != p.shape
            or t.ndim != 3 or t.shape[:2] != p.shape or t.shape[2] < 1
            or not all(np.isfinite(x).all() for x in (p, h, t))):
        raise ValueError("finite matching states/actions and [K,D,P] Jacobians required")
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if np.max(abs(np.sum(abs(p)**2, axis=1) - 1)) > 1e-10:
        raise ValueError("every physical ray must be normalized")
    e = np.sum(p.conj()*h, axis=1)
    if np.max(abs(e.imag)) > 1e-10:
        raise ValueError("Hermitian physical-H expectation required for every ray")
    if (not relative_cutoffs or any(not np.isfinite(c) or not 0 < c < 1 for c in relative_cutoffs)
            or any(b >= a for a, b in zip(relative_cutoffs, relative_cutoffs[1:]))
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError("descending rank cutoffs and positive finite damping required")
    w = np.full(len(p), 1/len(p)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w <= 0) or abs(w.sum()-1) > 1e-12:
        raise ValueError("positive normalized ray weights required")
    residual = h-e.real[:, None]*p
    horizontal = t-p[:, :, None]*np.einsum('kd,kdp->kp', p.conj(), t)[:, None, :]
    packed_t = (np.sqrt(w)[:, None, None]*horizontal).reshape(-1, t.shape[2])
    packed_r = (np.sqrt(w)[:, None]*residual).reshape(-1)
    design = np.concatenate((packed_t.real, packed_t.imag), axis=0)
    target = np.concatenate((packed_r.real, packed_r.imag))
    variance = float(target@target)
    norms = np.linalg.norm(design, axis=0)
    kept = norms > 1e-14
    spectra: dict[str, object] = {}
    projection = np.zeros_like(target)
    for label, matrix in (("parameter_coordinates", design),
                          ("column_equilibrated", design[:, kept]/norms[kept])):
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        coordinates = u.T@target
        rows = {}
        for cutoff in relative_cutoffs:
            rank = int(np.sum(s > cutoff*s[0])) if s.size and s[0] > 0 else 0
            captured = float(coordinates[:rank]@coordinates[:rank])
            rows[str(cutoff)] = {
                "rank": rank, "projected_residual_fraction": captured/variance if variance > 1e-28 else None,
                "unit_product_FS_descent_slope_per_site": -2*np.sqrt(captured)/n_sites,
                "condition_number": float(s[0]/s[rank-1]) if rank else None,
            }
        entry: dict[str, object] = {"singular_values": s.tolist(), "cutoffs": rows}
        if label == "parameter_coordinates":
            response = s*s/(s*s+damping)
            damped = u@(response*coordinates)
            entry["damping"] = damping
            entry["damped_target_norm_fraction"] = float(damped@damped)/variance if variance > 1e-28 else None
            entry["damped_target_alignment_fraction"] = float(damped@target)/variance if variance > 1e-28 else None
            entry["gradient_norm_per_site"] = float(np.linalg.norm(2*design.T@target)/n_sites)
        else:
            rank = rows[str(relative_cutoffs[-1])]["rank"]
            projection = u[:, :rank]@coordinates[:rank]
        spectra[label] = entry
    metrics = {
        "ray_count": len(p), "complex_dimension_per_ray": p.shape[1], "parameter_count": t.shape[2],
        "maximum_projective_real_rank": len(p)*2*(p.shape[1]-1),
        "weights": w.tolist(), "energies_per_site": (e.real/n_sites).tolist(),
        "variances_per_site": (np.sum(abs(residual)**2, axis=1)/n_sites).tolist(),
        "mean_energy_per_site": float(w@e.real/n_sites), "mean_variance_per_site": variance/n_sites,
        "spectra": spectra, "null_or_tiny_parameter_columns": int(np.sum(~kept)),
        "maximum_horizontal_gauge_defect": float(np.max(abs(np.einsum('kd,kdp->kp', p.conj(), horizontal)))),
        "projection_orthogonality_defect": float(np.linalg.norm(design.T@(target-projection))),
        "projection_semantics": "real shared parameters, independent ray gauges; no shared eigenstate or held-response certificate",
    }
    return ProductTangentDiagnostic(horizontal, design, target, projection, metrics)


__all__ = ["ProductTangentDiagnostic", "diagnose_product_tangent"]
