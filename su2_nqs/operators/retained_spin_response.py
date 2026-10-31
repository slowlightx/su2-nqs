"""State-weighted local-spin response retained by an SU(2) block space.

This diagnoses a finite subspace, not a QSL order parameter or an energy
error bound. Every physical site is spin1/2; q=2j labels complete multiplets.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np

from .su2_sector_merge import ReducedBlockData, compact_reduced_block


def retained_local_spin_response(block: ReducedBlockData,
        multiplicity_density: Mapping[int, np.ndarray], *, tolerance: float = 2e-10) -> dict:
    """Return <sum_a (P S_i^a P)^2> and its missing onsite3/4 weight.

    ``multiplicity_density[q]`` is sigma_q in rho=direct_sum_q
    I_(q+1)/(q+1) tensor sigma_q, with sum_q Tr(sigma_q)=1. It is the actual
    state's density, not a uniform average over retained vectors. In this
    repository's WE convention S^z=CG(qi,m;2,0|qo,m) R[qo,qi], the weight is
    sum_qi,qo (qo+1)/(qi+1) Tr(R^dagger R sigma_qi).

    The missing weight equals <sum_a S_i^a(1-P)S_i^a>. It does not specify
    the missing charge, the energetic importance, or complete-H variance.
    """
    if not np.isfinite(tolerance) or not 0<tolerance<1e-6:
        raise ValueError('a positive small numerical tolerance is required')
    current=compact_reduced_block(block,tolerance=tolerance)
    if set(multiplicity_density)!=set(current.q_widths):
        raise ValueError('density must cover every retained doubled-q sector')
    density={}
    for q,width in current.q_widths.items():
        value=np.asarray(multiplicity_density[q],complex)
        if (value.shape!=(width,width) or not np.isfinite(value).all()
                or np.max(abs(value-value.conj().T))>tolerance
                or np.linalg.eigvalsh((value+value.conj().T)/2)[0]<-tolerance):
            raise ValueError('finite Hermitian positive multiplicity densities required')
        density[q]=value
    norm=sum(float(np.trace(value).real) for value in density.values())
    if abs(norm-1)>tolerance:raise ValueError('multiplicity density must have unit total trace')
    qmax=max(current.q_widths)
    # A rank-one spin vector has Delta q=0,+/-2. Only input qmax can
    # reach q>qmax; its complete spin1/2 Casimir bounds that contribution.
    high_q_bound=(.75*float(np.trace(density[qmax]).real)
                  if qmax<current.n_sites else 0.)
    sites=[]
    for site,operators in enumerate(current.site_reduced_spin):
        channels=[]
        for (qo,qi),reduced in sorted(operators.items()):
            value=(qo+1)/(qi+1)*np.trace(reduced.conj().T@reduced@density[qi])
            if abs(value.imag)>tolerance or value.real<-tolerance:
                raise ArithmeticError('projected local-spin response lost positivity/reality')
            channels.append(dict(q_in=qi,q_out=qo,weight=float(value.real)))
        kept=sum(row['weight'] for row in channels)
        if kept< -tolerance or kept>.75+tolerance:
            raise ArithmeticError('retained spin1/2 response exceeds exact onsite3/4')
        sites.append(dict(site=site,retained_weight=kept,missing_weight=.75-kept,
                          retained_fraction=kept/.75,channels=channels,
                          missing_above_retained_qmax_upper_bound=high_q_bound,
                          missing_not_above_retained_qmax_lower_bound=max(0.,.75-kept-high_q_bound)))
    return dict(sites=sites,onsite_total=.75,density_trace=norm,
        actual_state_weighted=True,all_magnetic_components=True,
        maximum_retained_q=qmax,missing_above_retained_qmax_upper_bound=high_q_bound,
        energy_error_bound=False,physical_H_variance=False,
        mean_retained_fraction=float(np.mean([x['retained_fraction'] for x in sites])))
