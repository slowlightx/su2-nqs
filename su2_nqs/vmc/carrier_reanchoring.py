"""Polynomial physical carriers and carrier-relative tangent diagnostics.

Inputs are physical amplitudes and TOTAL H powers, never local-energy ratios.
No Hamiltonian, spin label, sampling or model logic is owned here. Empirical
span projections diagnose representation, not a variational-energy guarantee.
"""
from __future__ import annotations

import numpy as np

from .coherent_tangent import coherent_tangent_diagnostics


def polynomial_carrier_products(products: np.ndarray, coefficients: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Return P(H)C and H P(H)C from columns C,HC,H²C,... .

    A degree-d polynomial requires d+2 power columns. Nodes are retained;
    missing powers, nonfinite values and silent per-site rescalings are not
    repaired. The caller must provide products of the SAME total H and C.
    """
    powers, weights = np.asarray(products, complex), np.asarray(coefficients, complex)
    if (powers.ndim != 2 or not len(powers) or weights.ndim != 1 or not len(weights)
            or powers.shape[1] < len(weights)+1
            or not np.isfinite(powers).all() or not np.isfinite(weights).all()):
        raise ValueError('finite power columns and coefficients, including the next H power, required')
    if not np.any(weights):
        raise ValueError('nonzero carrier polynomial required')
    return powers[:, :len(weights)] @ weights, powers[:, 1:len(weights)+1] @ weights


def carrier_span_diagnostics(parent: np.ndarray, hparent: np.ndarray,
                             tangent: np.ndarray, htangent: np.ndarray,
                             basis: np.ndarray, hbasis: np.ndarray,
                             density: np.ndarray, n_sites: int,
                             *, relative_cutoff: float = 1e-10) -> dict:
    """Separate a tangent from a known complex physical carrier span on Q.

    Remove the parent's ray, then project on the centered carrier columns
    using an SVD. Apply the SAME fitted coefficients to H columns. This is
    readout only: no change to the actual model direction, state or gate.
    Includes both finite-Q coherent terms in outside-span slope diagnostics.
    """
    p,hp,d,hd,q = (np.asarray(v, t) for v,t in
        ((parent,complex),(hparent,complex),(tangent,complex),(htangent,complex),(density,float)))
    b,hb = np.asarray(basis,complex), np.asarray(hbasis,complex)
    if (p.ndim != 1 or not len(p) or any(v.shape != p.shape for v in (hp,d,hd,q))
            or b.ndim != 2 or b.shape[0] != len(p) or hb.shape != b.shape
            or not all(np.isfinite(v).all() for v in (p,hp,d,hd,b,hb,q))
            or np.any(q <= 0) or not 0 < relative_cutoff < 1):
        raise ValueError('matching finite products, positive Q and cutoff in (0,1) required')
    norm = float(np.sum(abs(p)**2/q))
    if norm <= 0:
        raise ValueError('positive parent norm required')
    gauge = np.sum(p.conj()*d/q)/norm
    d,hd = d-gauge*p, hd-gauge*hp
    overlap = np.sum(p.conj()[:,None]*b/q[:,None],axis=0)/norm
    b,hb = b-p[:,None]*overlap, hb-hp[:,None]*overlap
    weighted = b/np.sqrt(q[:,None])
    u,s,vh = np.linalg.svd(weighted,full_matrices=False)
    # Numerical parent-ray cancellation is not an independent direction.
    raw_scale = float(np.linalg.norm(basis/np.sqrt(q[:,None]),ord=2))
    keep = s > max(float(s[0]) if len(s) else 0.,raw_scale)*relative_cutoff
    target = d/np.sqrt(q)
    coefficients = (vh[keep].conj().T @ ((u[:,keep].conj().T @ target)/s[keep])
                    if keep.any() else np.zeros(b.shape[1],complex))
    outside, houtside = d-b@coefficients, hd-hb@coefficients
    total = float(np.sum(abs(d)**2/q))
    remaining = float(np.sum(abs(outside)**2/q))
    diagnostic = coherent_tangent_diagnostics(p,hp,outside,houtside,q,n_sites)
    return dict(centered_carrier_rank=int(keep.sum()),
        outside_carrier_span_fraction=remaining/total if total > 1e-28 else 0.,
        centered_tangent_norm_over_Q=total,
        outside_norm_over_Q=remaining,
        outside_tangent=diagnostic,
        empirical_projection_only=True,actual_direction_unchanged=True,
        no_energy_gain_guaranteed_by_linear_independence=True)


__all__ = ['polynomial_carrier_products','carrier_span_diagnostics']
