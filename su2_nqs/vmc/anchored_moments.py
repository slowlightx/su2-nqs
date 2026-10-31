"""Fixed-coefficient Hermitian carrier control variates, readout only.

Node-safe physical products C,HC,H²C,psi,Hpsi; TOTAL H and positive density Q.
Never overwrite the original finite-Q moments or gates. Raw correction
numerators have zero population mean; self-normalized ratios remain biased.
No coefficient fit, sample filtering, training or acceptance logic here.
Finite-Q corrections depend on the candidate's phase relative to the anchor;
they are NOT a gauge-invariant empirical training objective. Keep the saved
phase fixed before reading audit data, never tune it to lower the estimate.
"""
from __future__ import annotations

import numpy as np
from .carrier_protection import physical_moments


def anchored_moments(carrier: np.ndarray,hcarrier: np.ndarray,h2carrier: np.ndarray,
                     psi: np.ndarray,hpsi: np.ndarray,q: np.ndarray,n_sites: int) -> dict:
    """Expand psi=C+D and move H across carrier cross terms in population.

    E numerator: <C|HC>+2Re<D|HC>+<D|HD>.
    H² numerator: ||HC||²+2Re<D|H²C>+||HD||².
    Compare both to the unchanged raw products on the SAME Q records.
    The empirical corrected second moment/variance can be negative: report
    it, never clip or treat it as an exact quantum variance.
    """
    c,hc,h2,p,hp,density=(np.asarray(v,dtype=t) for v,t in
        ((carrier,complex),(hcarrier,complex),(h2carrier,complex),(psi,complex),(hpsi,complex),(q,float)))
    raw=physical_moments(p,hp,density,n_sites)
    if any(v.shape!=p.shape or not np.isfinite(v).all() for v in (c,hc,h2)):
        raise ValueError('matching finite C,HC,H²C required')
    d,hd=p-c,hp-hc;norm=float(np.sum(abs(p)**2/density))
    numerator=np.sum(c.conj()*hc/density)+2*np.sum(d.conj()*hc/density).real+np.sum(d.conj()*hd/density)
    second=float(np.sum(abs(hc)**2/density)+2*np.sum(d.conj()*h2/density).real+np.sum(abs(hd)**2/density))
    energy=float(numerator.real/norm);variance=float((second/norm-energy**2)/n_sites)
    return dict(energy_per_site=energy/n_sites,variance_per_site=variance,
        second_moment_total=second/norm,imaginary_energy_numerator_per_norm=float(numerator.imag/norm),
        energy_correction_per_site=energy/n_sites-raw['energy_per_site'],
        second_moment_correction_total=second/norm-float(np.sum(abs(hp)**2/density)/norm),
        raw=raw,finite_variance_nonnegative=variance>=0,
        population_identity_not_finite_quadrature_identity=True,fixed_control_variate_coefficient=1.,
        no_clipping_or_gate_override=True,self_normalized_finite_sample_biased=True,
        relative_anchor_phase_fixed_before_data_required=True,
        empirical_candidate_ray_gauge_invariance_not_guaranteed=True)


__all__=['anchored_moments']
