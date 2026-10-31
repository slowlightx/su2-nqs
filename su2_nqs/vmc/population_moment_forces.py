"""Hermitian population energy/variance forces from C, HC and H²C.

Positive Q is a sampling density, not a doubled spin label. TOTAL Hamiltonian
products, real parameters and complex amplitudes; no division by a carrier
amplitude. These are population estimators, not derivatives of an incomplete
empirical ratio. Actual candidate validation must retain both coherences.
"""
from __future__ import annotations

import numpy as np
from .cross_sample_geometry import hermitian_energy_force


def population_moment_forces(psi: np.ndarray, hpsi: np.ndarray, h2psi: np.ndarray,
                             jacobian: np.ndarray, q: np.ndarray, n_sites: int
                             ) -> tuple[np.ndarray,np.ndarray,dict]:
    """Return gE,gR, where R=Var(H)/N for the Hermitian population.

    gR=2 Re<J_perp|(H²-2E H)psi>/(N Z); E is the real Hermitian energy
    estimate. Centering J removes the remaining scalar normalization term.
    The H²psi vs ||Hpsi||² finite-Q disagreement is reported, never forced
    to zero. A constant H shift cancels exactly in both forces.
    """
    p,hp,h2,density=(np.asarray(v,dtype=t) for v,t in
        ((psi,complex),(hpsi,complex),(h2psi,complex),(q,float)))
    ge=hermitian_energy_force(p,hp,jacobian,density,n_sites)
    if h2.shape!=p.shape or not np.isfinite(h2).all():
        raise ValueError('finite matching TOTAL H²psi vector required')
    norm=float(np.sum(abs(p)**2/density))
    energy=np.sum(p.conj()*hp/density)/norm
    h2_overlap=np.sum(p.conj()*h2/density)/norm
    second_norm=float(np.sum(abs(hp)**2/density)/norm)
    gr=hermitian_energy_force(p,h2-2*energy.real*hp,jacobian,density,n_sites)
    return ge,gr,dict(energy_per_site=float(energy.real/n_sites),
        imaginary_energy_per_site=float(energy.imag/n_sites),
        H2_overlap_real=float(h2_overlap.real),H2_overlap_imaginary=float(h2_overlap.imag),
        H_norm_second_moment=second_norm,
        second_moment_quadrature_defect=float(abs(h2_overlap-second_norm)),
        population_not_incomplete_quadrature_derivative=True,
        H2_products_required=True,carrier_nodes_retained=int(np.sum(p==0)))


def imaginary_time_energy_certificate(couplings: np.ndarray,anchor_energy: float) -> dict:
    """Bound a linear Heisenberg filter that cannot raise true energy.

    ||H||<=B=(3/4)sum|Jb|. For beta<=.5/(B-anchor), the spectral amplitude
    1+beta*(anchor-lambda) stays positive on [-B,B]; its squared weight is
    monotone decreasing in lambda. Reweighting a spectral probability measure
    by a decreasing positive function cannot increase its mean. This proves
    an energy statement, NOT variance descent or Monte Carlo convergence.
    """
    js=np.asarray(couplings,float)
    if js.ndim!=1 or not len(js) or not np.isfinite(js).all() or not np.isfinite(anchor_energy):
        raise ValueError('finite real Heisenberg couplings and anchor required')
    bound=float(.75*np.sum(abs(js)))
    if bound<=0:raise ValueError('nonzero Hamiltonian required')
    # For anchor>=B positivity holds for every positive beta; retaining this
    # finite conservative cap also handles arbitrary empirical anchors.
    cap=.5/max(bound-float(anchor_energy),bound)
    return dict(H_operator_norm_bound=bound,maximum_beta=cap,
        minimum_filter_amplitude_bound=1+cap*(float(anchor_energy)-bound),
        true_energy_nonincrease=True,variance_nonincrease_proved=False,
        strict_decrease_unless_spectral_eigenstate=True,
        incomplete_quadrature_energy_sign_not_certificate=True)


__all__=['population_moment_forces','imaginary_time_energy_certificate']
