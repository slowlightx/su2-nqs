"""Complete real-energy cotangents on a fixed positive physical quadrature."""
from __future__ import annotations

import numpy as np

from .carrier_protection import physical_moments


def energy_moment_cotangents(
    psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray, n_sites: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return weights for d Re(E/N)=Re sum(wpsi*dpsi+wH*H dpsi).

Hpsi is TOTAL Hpsi. Both coherences and the norm derivative are present;
on incomplete Q, do not replace the reverse action by its Hermitian mean.
No amplitude division, conjugation of these weights, node clipping or
assumption that the empirical energy is real. Q is fixed, not learned.
"""
    moments=physical_moments(psi,hpsi,q,n_sites)
    p,h,density=np.asarray(psi,complex),np.asarray(hpsi,complex),np.asarray(q,float)
    norm=float(np.sum(abs(p)**2/density))
    e=np.sum(p.conj()*h/density)/norm
    denominator=n_sites*density*norm
    wp=(h.conj()-2*e.real*p.conj())/denominator
    wh=p.conj()/denominator
    return wp,wh,dict(energy_per_site=moments['energy_per_site'],
        imaginary_energy_per_site=moments['imaginary_energy_per_site'],
        complete_two_sided_derivative=True,exact_node_records=int(np.sum(p==0)))


__all__=['energy_moment_cotangents']
