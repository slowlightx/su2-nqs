"""Real-parameter, finite-direction physical tangent stationarity diagnostics.

Project a Hermitian-H residual onto a DECLARED finite set of state tangents.
This gives a rigorous lower bound on full-tangent residual support, never a
certificate that the complete neural manifold is stationary. No optimizer,
sampler, neural evaluator or Hamiltonian constructor is called here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TangentResidualProjection:
    """Projection and unit-FS descent coefficients in the supplied directions."""
    descent_coefficients: np.ndarray
    projected_residual: np.ndarray
    horizontal_tangents: np.ndarray
    metrics: dict


def project_residual_on_tangents(
    state: np.ndarray, hstate: np.ndarray, tangents: np.ndarray, *,
    n_sites: int, relative_cutoff: float = 1e-10,
) -> TangentResidualProjection:
    """Solve a real least-squares problem using an un-squared SVD.

    State must be normalized, ``tangents[:, k]`` its derivative along a REAL
    parameter direction, and Hstate use TOTAL-H units. Gauge/ray components
    are removed. The largest possible first-order energy/site decrease in
    this real span at unit FS speed is ``2*||projected_residual||/N``.
    Individual nonzero columns are normalized before rank determination.
    """
    p,h=np.asarray(state,dtype=complex),np.asarray(hstate,dtype=complex)
    t=np.asarray(tangents,dtype=complex)
    if (p.ndim!=1 or not p.size or h.shape!=p.shape or t.ndim!=2 or t.shape[0]!=len(p)
            or not all(np.isfinite(a).all() for a in (p,h,t)) or n_sites<1
            or not np.isfinite(relative_cutoff) or not 0<relative_cutoff<1):
        raise ValueError('invalid normalized state/action/tangent inputs')
    if abs(np.vdot(p,p)-1)>1e-10:raise ValueError('state must be normalized')
    energy=np.vdot(p,h)
    if abs(energy.imag)>1e-10:raise ValueError('requires Hermitian physical-H expectation')
    residual=h-energy.real*p
    horizontal=t-p[:,None]*(p.conj()@t)[None,:]
    lengths=np.linalg.norm(horizontal,axis=0)
    kept=np.flatnonzero(lengths>1e-14)
    normalized=horizontal[:,kept]/lengths[kept] if kept.size else np.empty((len(p),0),complex)
    design=np.concatenate((normalized.real,normalized.imag),axis=0)
    target=np.concatenate((residual.real,residual.imag))
    u,s,vh=np.linalg.svd(design,full_matrices=False)
    rank=int(np.sum(s>relative_cutoff*s[0])) if s.size else 0
    coordinates=u[:,:rank].T@target
    coefficients=np.zeros(t.shape[1])
    if rank:coefficients[kept]=(vh[:rank].T@(coordinates/s[:rank]))/lengths[kept]
    projected=horizontal@coefficients
    length=float(np.linalg.norm(projected));variance=float(np.vdot(residual,residual).real)
    descent=-coefficients/length if length>1e-14 else np.zeros_like(coefficients)
    slope=float(2*np.real(np.vdot(horizontal@descent,residual))/n_sites)
    metrics=dict(direction_count=t.shape[1],rank=rank,singular_values=s.tolist(),
        retained_direction_indices=kept.tolist(),condition_number=float(s[0]/s[rank-1]) if rank else None,
        energy_per_site=float(energy.real/n_sites),variance_per_site=variance/n_sites,
        projected_residual_norm=length,projected_residual_fraction=length**2/variance if variance>1e-28 else None,
        unit_FS_descent_slope_per_site=slope,
        unit_FS_speed=float(np.linalg.norm(horizontal@descent)),
        orthogonality_defect=float(np.linalg.norm(np.real(horizontal.conj().T@(residual-projected)))),
        strict_lower_bound_on_full_tangent_support=True,full_tangent_stationarity_certified=False)
    return TangentResidualProjection(descent,projected,horizontal,metrics)


def gradient_conflict_matrix(gradients: np.ndarray) -> dict:
    """Euclidean parameter-gradient cosines; explicitly coordinate dependent."""
    gradients=np.asarray(gradients,dtype=float)
    if gradients.ndim!=2 or not np.isfinite(gradients).all():raise ValueError('finite gradient matrix required')
    norms=np.linalg.norm(gradients,axis=1);denominator=norms[:,None]*norms[None,:]
    cos=np.divide(gradients@gradients.T,denominator,out=np.zeros_like(denominator),where=denominator>0)
    return dict(norms=norms.tolist(),cosines=cos.tolist(),defined=(denominator>0).tolist(),
        coordinate_dependent=True,mean_gradient_norm=float(np.linalg.norm(gradients.mean(axis=0))),
        mean_individual_gradient_norm=float(norms.mean()))


def projector_secant_alignment(
    first_before: np.ndarray, first_after: np.ndarray,
    second_before: np.ndarray, second_after: np.ndarray, *, coupling_span: float,
) -> dict:
    """Phase-independent coarse projector-secant cosine and FS speed ratio.

    Tr[(Pa-Pb)(Pc-Pd)] is computed only from squared amplitude overlaps.
    These finite-span secants are not exact derivatives or susceptibilities.
    """
    vectors=[np.asarray(x,dtype=complex) for x in (first_before,first_after,second_before,second_after)]
    if (vectors[0].ndim!=1 or any(x.shape!=vectors[0].shape or not np.isfinite(x).all() for x in vectors)
            or not np.isfinite(coupling_span) or coupling_span<=0 or any(np.linalg.norm(x)==0 for x in vectors)):
        raise ValueError('matching nonzero vectors and positive coupling span required')
    a,b,c,d=[x/np.linalg.norm(x) for x in vectors]
    first=max(0.,2*(1-abs(np.vdot(a,b))**2))/coupling_span**2
    second=max(0.,2*(1-abs(np.vdot(c,d))**2))/coupling_span**2
    cross=(abs(np.vdot(b,d))**2-abs(np.vdot(b,c))**2-abs(np.vdot(a,d))**2+abs(np.vdot(a,c))**2)/coupling_span**2
    return dict(projector_secant_cosine=float(cross/np.sqrt(first*second)) if first*second>1e-28 else None,
        first_FS_secant_speed=float(np.sqrt(first/2)),second_FS_secant_speed=float(np.sqrt(second/2)),
        first_over_second_speed=float(np.sqrt(first/second)) if second>1e-28 else None,
        coupling_span=float(coupling_span),coarse_secants_not_exact_response=True)


__all__=['TangentResidualProjection','project_residual_on_tangents','gradient_conflict_matrix','projector_secant_alignment']
