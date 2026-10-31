"""Node-safe physical moments and finite-panel geometry for IID proposals."""
from __future__ import annotations

import numpy as np


def paired_moments(products: np.ndarray, density: np.ndarray, *, n_sites: int, groups: int) -> dict:
    """Paired endpoint energy/variance; target nodes are never discarded.

    ``products[N,2,2]`` holds two endpoints and ``(psi,Hpsi)`` at each.
    Group jackknife errors describe this IID panel, not unseen-tail accuracy.
    """
    values = np.asarray(products); q = np.asarray(density, float)
    if (values.ndim != 3 or values.shape[1:] != (2, 2) or q.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(q).all() or np.any(q <= 0)
            or groups < 4 or len(values) % groups or n_sites < 1):
        raise ValueError("finite paired products, positive Q, and equal independent groups required")
    psi, hp = values[:, :, 0], values[:, :, 1]
    def evaluate(ids):
        norm = np.mean(abs(psi[ids])**2/q[ids, None], axis=0)
        if np.any(norm <= 0): raise FloatingPointError("zero observed norm")
        energy = np.mean(psi[ids].conj()*hp[ids]/q[ids, None], axis=0)/norm
        variance = np.mean(abs(hp[ids]-energy[None]*psi[ids])**2/q[ids, None], axis=0)/norm
        return np.stack((energy.real/n_sites, energy.imag/n_sites, variance/n_sites, norm), axis=1)
    labels = np.repeat(np.arange(groups), len(values)//groups); ids = np.arange(len(values))
    point = evaluate(ids); leave = np.asarray([evaluate(ids[labels != g]) for g in range(groups)])
    errors = np.sqrt((groups-1)/groups*np.sum((leave-leave.mean(0))**2, axis=0))
    differences = leave[:, 1]-leave[:, 0]
    delta_se = np.sqrt((groups-1)/groups*np.sum((differences-differences.mean(0))**2, axis=0))
    norm_weights = abs(psi)**2/q[:, None]; h2_weights = abs(hp)**2/q[:, None]
    def concentration(weights):
        total = weights.sum(0); squares = np.sum(weights**2, axis=0)
        return dict(effective_contributions=np.divide(total**2, squares,
                out=np.zeros_like(total), where=squares > 0).tolist(),
            maximum_sample_fraction=np.divide(weights.max(0), total,
                out=np.zeros_like(total), where=total > 0).tolist(),
            concentration_defined=(total > 0).tolist())
    return dict(metric_order=["energy_per_site", "imaginary_energy_per_site", "variance_per_site", "norm"],
        endpoint_values=point.tolist(), group_jackknife_SE=errors.tolist(),
        candidate_minus_parent=(point[1]-point[0]).tolist(), paired_group_jackknife_SE=delta_se.tolist(),
        norm_concentration=concentration(norm_weights), H2_concentration=concentration(h2_weights),
        target_node_records=np.sum(psi == 0, axis=0).tolist(),
        target_node_H2_contributions=np.sum(np.where(psi == 0, h2_weights, 0), axis=0).tolist(),
        records=len(values), groups=groups, proposal_exact_known_full_support=True,
        finite_sample_ratio_bias_and_unseen_tails_not_certified=True)


def finite_FS(amplitudes: np.ndarray, density: np.ndarray) -> dict:
    """Fubini--Study angle from the actual finite parent/candidate amplitudes."""
    p = np.asarray(amplitudes); q = np.asarray(density)
    if (p.ndim != 2 or p.shape[1] != 2 or q.shape != (len(p),)
            or np.iscomplexobj(q) or not np.isfinite(q).all() or np.any(q <= 0)):
        raise ValueError("paired amplitudes and positive exact density required")
    norm = np.mean(abs(p)**2/q[:, None], axis=0)
    if not np.isfinite(p).all() or np.any(norm <= 0): raise FloatingPointError("invalid physical norm")
    overlap = np.mean(p[:, 0].conj()*p[:, 1]/q)
    fidelity = float(abs(overlap)**2/np.prod(norm))
    if fidelity > 1+1e-12: raise ArithmeticError("finite quadrature violates Cauchy--Schwarz")
    return dict(FS_angle=float(np.arccos(np.sqrt(np.clip(fidelity, 0., 1.)))),
        fidelity=fidelity, norm_estimates=norm.tolist(), records=len(p),
        finite_panel_not_population_geometry=True)
