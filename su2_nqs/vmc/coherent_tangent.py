"""Two-sided Rayleigh slopes and cross-chain tangent geometry on positive Q.

Inputs are physical amplitudes, TOTAL H products and their JVP tangents.
Incomplete quadrature does not enforce H Hermiticity. Keep both coherences
and remove the same complex parent-ray gauge from delta psi AND H delta psi.
These finite-cache screens are optimizer controls, not population guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from collections.abc import Sequence

import numpy as np


def coherent_tangent_diagnostics(psi: np.ndarray, hpsi: np.ndarray, delta: np.ndarray,
                                 hdelta: np.ndarray, q: np.ndarray, n_sites: int) -> dict:
    """Exact derivative of the finite-Q Rayleigh quotient and residual moment.

    Let t=delta-beta*psi, beta=<psi|delta>/<psi|psi>, and
    Ht=hdelta-beta*hpsi. Forward=<t|Hpsi>/norm and reverse=<psi|Ht>/norm.
    dE/ds=forward+reverse, NOT twice forward, at psi+s*delta. FS speed is
    ||t||/||psi||. For the variance use ||Hpsi||^2/norm-|E|^2: on finite Q
    its derivative must keep the complex energy, not force it real.
    """
    p,hp,d,hd,density=(np.asarray(psi,complex),np.asarray(hpsi,complex),
                      np.asarray(delta,complex),np.asarray(hdelta,complex),np.asarray(q,float))
    if p.ndim!=1 or not len(p) or any(x.shape!=p.shape for x in (hp,d,hd,density)):
        raise ValueError('matching nonempty physical products and tangents required')
    if not all(np.isfinite(x).all() for x in (p,hp,d,hd,density)) or np.any(density<=0):
        raise ValueError('finite inputs and strictly positive Q required')
    if isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or n_sites<1:
        raise ValueError('positive integer site count required')
    norm=float(np.sum(abs(p)**2/density))
    if not np.isfinite(norm) or norm<=0:raise ValueError('finite nonzero parent norm required')
    beta=np.sum(p.conj()*d/density)/norm
    t,ht=d-beta*p,hd-beta*hp
    energy=np.sum(p.conj()*hp/density)/norm
    forward=np.sum(t.conj()*hp/density)/norm
    reverse=np.sum(p.conj()*ht/density)/norm
    slope=forward+reverse
    speed=float(np.sqrt(np.sum(abs(t)**2/density)/norm))
    second_slope=2*float(np.sum(hp.conj()*ht/density).real/norm)
    variance_slope=(second_slope-2*(energy.conjugate()*slope).real)/n_sites
    return dict(energy_per_site=float(energy.real/n_sites),
        imaginary_energy_per_site=float(energy.imag/n_sites),
        forward_coherence_per_site=float(forward.real/n_sites),
        forward_imaginary_per_site=float(forward.imag/n_sites),
        reverse_coherence_per_site=float(reverse.real/n_sites),
        reverse_imaginary_per_site=float(reverse.imag/n_sites),
        energy_slope_per_site=float(slope.real/n_sites),
        imaginary_energy_slope_per_site=float(slope.imag/n_sites),
        one_sided_energy_slope_per_site=float(2*forward.real/n_sites),
        variance_slope_per_site=float(variance_slope),
        hermiticity_quadrature_defect_per_site=float(abs(reverse-forward.conjugate())/n_sites),
        tangent_fs_speed=speed,
        energy_slope_per_unit_fs=float(slope.real/n_sites/speed) if speed>0 else 0.,
        gauge_beta_real=float(beta.real),gauge_beta_imaginary=float(beta.imag),
        centered_orthogonality_scaled=float(abs(np.sum(p.conj()*t/density))/norm),
        exact_parent_node_records=int(np.sum(p==0)),
        tangent_at_parent_nodes_norm=float(np.sum(abs(t[p==0])**2/density[p==0])/norm),
        finite_cache_not_population_certificate=True)


@dataclass(frozen=True)
class CoherentTangentPolicy:
    """Conservative point-estimate screen, fixed before full-H JVP evaluation."""
    maximum_speed_ratio: float=2.
    minimum_nonzero_speed: float=1e-12

    def __post_init__(self) -> None:
        if (isinstance(self.maximum_speed_ratio,bool) or not np.isfinite(self.maximum_speed_ratio)
                or self.maximum_speed_ratio<1 or isinstance(self.minimum_nonzero_speed,bool)
                or not np.isfinite(self.minimum_nonzero_speed) or self.minimum_nonzero_speed<=0):
            raise ValueError('finite speed ratio >=1 and positive nonzero threshold required')


def cross_chain_tangent_gate(records: Sequence[dict], policy: CoherentTangentPolicy) -> dict:
    """Require favorable BOTH coherences on each declared independent chain split.

    Report disagreement without flipping/refitting a direction. This strict
    finite-control policy is not a fundamental physical admissibility rule;
    a rejected noisy direction is not proved globally uphill or unexpressive.
    """
    if len(records)<2:raise ValueError('at least two declared chain splits required')
    keys=('forward_coherence_per_site','reverse_coherence_per_site','energy_slope_per_site','tangent_fs_speed')
    values=np.asarray([[r[k] for k in keys] for r in records],float)
    if not np.isfinite(values).all() or np.any(values[:,3]<0):raise ValueError('finite valid tangent diagnostics required')
    speeds=values[:,3]; ratio=float(speeds.max()/speeds.min()) if speeds.min()>0 else None
    checks=dict(nonzero_tangent=bool(np.all(speeds>policy.minimum_nonzero_speed)),
        forward_descent_each_split=bool(np.all(values[:,0]<0)),
        reverse_descent_each_split=bool(np.all(values[:,1]<0)),
        total_descent_each_split=bool(np.all(values[:,2]<0)),
        cross_chain_geometry=bool(ratio is not None and ratio<=policy.maximum_speed_ratio))
    return dict(eligible=bool(all(checks.values())),checks=checks,
        rejected_by=[k for k,v in checks.items() if not v],speed_ratio=ratio,
        control_used_for_training=True,global_descent_certified=False,
        noisy_rejection_is_not_architecture_no_go=True)


__all__=['coherent_tangent_diagnostics','CoherentTangentPolicy','cross_chain_tangent_gate']
