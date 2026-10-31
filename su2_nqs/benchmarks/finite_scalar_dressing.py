"""Finite-step energy, variance, and normalization with common full support."""

import numpy as np


def finite_step_moments(psi: np.ndarray, direction: np.ndarray, hpsi: np.ndarray,
                       hdirection: np.ndarray, proposal_squared: np.ndarray, epsilon: float) -> dict:
    """Evaluate a coherent finite state, not a first-order energy expansion.

    The normalization is relative to the carrier, since the proposal need
    not have known absolute normalization. Include support where psi=0.
    All estimates are self-normalized and can have finite-sample bias.
    """
    p,d,h,hd,q=map(np.asarray,(psi,direction,hpsi,hdirection,proposal_squared))
    if p.ndim!=1 or len(p)==0 or any(x.shape!=p.shape for x in (d,h,hd,q)) or np.any(q<=0):
        raise ValueError("matching vectors and positive full-support proposal required")
    if not np.isfinite(epsilon) or not all(np.isfinite(x).all() for x in (p,d,h,hd,q)):
        raise ValueError("finite inputs required")
    state=p+epsilon*d
    action=h+epsilon*hd
    old_weight=abs(p)**2/q;weight=abs(state)**2/q
    old_norm=float(old_weight.sum());norm=float(weight.sum())
    if min(norm,old_norm)<=0:
        raise ValueError("sampled state has zero norm")
    old_energy=float(np.sum(p.conj()*h/q).real/old_norm)
    old_variance=float(np.sum(abs(h-old_energy*p)**2/q)/old_norm)
    energy=float(np.sum(state.conj()*action/q).real/norm)
    variance=float(np.sum(abs(action-energy*state)**2/q)/norm)
    z1=float(np.sum(p.conj()*d/q).real/old_norm)
    z2=float(np.sum(abs(d)**2/q)/old_norm)
    return {"epsilon":float(epsilon),"norm_ratio":norm/old_norm,
            "normalization_quadratic_error":float(abs(norm/old_norm-(1+2*epsilon*z1+epsilon**2*z2))),
            "energy_total":energy,"variance_total":variance,
            "energy_change_total":energy-old_energy,"variance_change_total":variance-old_variance,
            "carrier_fidelity":float(abs(np.sum(p.conj()*state/q))**2/(norm*old_norm)),
            "importance_ess_fraction":float(norm**2/(len(q)*np.sum(weight**2)))}
