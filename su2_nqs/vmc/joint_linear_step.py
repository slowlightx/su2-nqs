"""Joint E/variance step geometry in an existing full-H linear subspace.

No new wavefunction/head, H approximation, or audit fitting. Complex columns
are resolved into real/imaginary coefficient directions. Both finite-Q energy
coherences and Hdelta are retained; an energy-only Ritz direction need not
lower variance. These are known-quadrature controls, not population claims.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np

from .joint_moments import normalized_common_moment_direction, residual_moment_cotangents


def linear_moment_slopes(psi: np.ndarray, hpsi: np.ndarray, tangent: np.ndarray,
                         h_tangent: np.ndarray, q: np.ndarray, n_sites: int) -> dict:
    """Exact E/N and Var/N derivatives for REAL coefficients of each column.

    H products use total-H units. Includes Hpsi at psi=0 and the derivative
    of normalization; no presumed Hermitian identity on incomplete samples.
    """
    p, h, t, ht, density = map(np.asarray, (psi, hpsi, tangent, h_tangent, q))
    if (p.ndim != 1 or h.shape != p.shape or t.ndim != 2 or not t.shape[1]
            or t.shape[0] != len(p) or ht.shape != t.shape or density.shape != p.shape
            or any(not np.isfinite(x).all() for x in (p,h,t,ht,density))
            or np.iscomplexobj(density) or np.any(density <= 0)):
        raise ValueError('finite compatible amplitude/H columns and positive real Q required')
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError('positive integer site count required')
    norm = float(np.sum(abs(p)**2/density))
    if norm <= 0:
        raise ValueError('nonzero parent required')
    energy = np.sum(p.conj()*h/density)/norm
    derivative_norm = 2*np.sum(p.conj()[:,None]*t/density[:,None],axis=0).real/norm
    de = np.sum((t.conj()*h[:,None]+p.conj()[:,None]*ht)/density[:,None],axis=0)/norm-energy*derivative_norm
    wp, wh, _ = residual_moment_cotangents(p,h,density,int(n_sites))
    dr = np.sum(wp[:,None]*t+wh[:,None]*ht,axis=0).real
    return dict(energy=de.real/n_sites, imaginary_energy=de.imag/n_sites, variance=dr)


def joint_linear_coefficients(psi: np.ndarray, hpsi: np.ndarray, tangent: np.ndarray,
                               h_tangent: np.ndarray, q: np.ndarray,
                               blocks: Sequence[np.ndarray], *, n_sites: int,
                               damping: float = .1) -> tuple[np.ndarray, dict]:
    """Unit-FS common E/R descent in existing complex columns, or exact zero.

    Each predeclared block contributes E and variance objectives. There are
    2..6 objectives, reusing the existing normalized minimax/SR solver; no
    post-candidate/audit weight selection. Return actual UPDATE coefficients,
    not the positive gradient direction convention of that solver.
    """
    t, ht = np.asarray(tangent,complex), np.asarray(h_tangent,complex)
    linear_moment_slopes(psi,hpsi,t,ht,q,n_sites)
    if not 1 <= len(blocks) <= 3:
        raise ValueError('one to three declared control blocks required')
    real_t, real_ht = np.c_[t,1j*t], np.c_[ht,1j*ht]
    gradients, names = [], []
    for i, raw in enumerate(blocks):
        m = np.asarray(raw)
        if m.dtype != bool or m.shape != np.asarray(psi).shape or not m.any():
            raise ValueError('nonempty boolean control blocks required')
        row = linear_moment_slopes(np.asarray(psi)[m],np.asarray(hpsi)[m],real_t[m],real_ht[m],np.asarray(q)[m],n_sites)
        gradients.extend((row['energy'], row['variance']))
        names.extend((f'block{i}.energy',f'block{i}.variance'))
    d, info = normalized_common_moment_direction(psi,hpsi,real_t,q,np.asarray(gradients),names,damping=damping)
    k = t.shape[1]
    coef = -d[:k]-1j*d[k:]
    info.update(existing_complex_columns=k,real_coefficient_dimensions=2*k,
                no_new_H_calls=True, coefficients_real=coef.real.tolist(),
                coefficients_imag=coef.imag.tolist())
    return coef, info


def joint_step_gate(before: Sequence[dict], after: Sequence[dict], carrier: Sequence[dict], *,
                    energy_tolerance: float, variance_ratio: float,
                    roundoff_tolerance: float = 1e-12) -> dict:
    """Require energy descent and NONINCREASING variance on both known blocks.

    FIT energy must strictly decrease; control may use its preregistered
    tolerance. Variance must not increase versus current endpoint AND stay
    inside the unchanged carrier envelope. No statistical/audit claim.
    """
    if (len(before) != 2 or len(after) != 2 or len(carrier) != 2
            or not np.isfinite([energy_tolerance,variance_ratio,roundoff_tolerance]).all()
            or energy_tolerance < 0 or variance_ratio < 1 or roundoff_tolerance < 0):
        raise ValueError('two known blocks and finite valid quality limits required')
    de = [a['energy_per_site']-b['energy_per_site'] for a,b in zip(after,before)]
    dr = [a['variance_per_site']-b['variance_per_site'] for a,b in zip(after,before)]
    flags = dict(fit_energy_decreases=bool(de[0]<-roundoff_tolerance),
        control_energy_safe=bool(de[1]<=energy_tolerance),
        variance_nonincreasing_both=bool(max(dr)<=roundoff_tolerance),
        carrier_variance_safe=bool(all(a['variance_per_site']<=variance_ratio*c['variance_per_site']+roundoff_tolerance
                                      for a,c in zip(after,carrier))))
    return dict(accepted=all(flags.values()), checks=flags,
                delta_energy=de,delta_variance=dr,not_an_independent_audit=True)


__all__=['linear_moment_slopes','joint_linear_coefficients','joint_step_gate']
