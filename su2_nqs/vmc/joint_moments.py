"""Complete residual-moment cotangents and unit-invariant common descent.

Real parameters, complex physical amplitudes, TOTAL Hpsi, strictly positive
fixed Q. Residual moment is ||(H-E)psi||^2/(N||psi||^2), with complex finite-Q
E. It is not bare <H^2>. These controls are not population certificates.
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
import numpy as np

from .carrier_protection import physical_moments
from .full_rayleigh import full_rayleigh_sr_direction
from .importance_tangent import tangent_system


def residual_moment_cotangents(psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray,
                               n_sites: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return wpsi,wH for dR=Re sum(wpsi*dpsi+wH*H dpsi).

    r=Hpsi-Epsi is exactly Q-orthogonal to psi; hence the dE term cancels,
    even when E is complex on incomplete quadrature. Preserve the -E*dpsi
    and norm derivatives. Use these COMPLEX coefficients directly in the
    existing projected-product pullback, without another conjugation.
    No ratios by psi; nodes and unknown Q normalization are retained.
    """
    moments=physical_moments(psi,hpsi,q,n_sites)
    p,hp,density=np.asarray(psi,complex),np.asarray(hpsi,complex),np.asarray(q,float)
    norm=float(np.sum(abs(p)**2/density))
    energy=np.sum(p.conj()*hp/density)/norm
    residual=hp-energy*p
    total=float(np.sum(abs(residual)**2/density)/norm)
    weights=1/(density*norm*n_sites)
    wpsi=-2*(energy*residual.conj()+total*p.conj())*weights
    wh=2*residual.conj()*weights
    info=dict(energy_real=float(energy.real),energy_imaginary=float(energy.imag),
        residual_moment_per_site=total/n_sites,norm=norm,
        orthogonality_scaled=float(abs(np.sum(p.conj()*residual/density))/norm),
        exact_parent_node_records=int(np.sum(p==0)),
        residual_moment_matches_variance=abs(total/n_sites-moments['variance_per_site'])<1e-12)
    return wpsi,wh,info


def normalized_common_moment_direction(psi: np.ndarray, hpsi: np.ndarray,
                                       jacobian: np.ndarray, q: np.ndarray,
                                       gradients: np.ndarray, names: Sequence[str], *, damping: float=.1
                                       ) -> tuple[np.ndarray, dict]:
    """Common -d for 2..6 objectives in the SAME full-cache 2FS+damping metric.

    Normalize each gradient by its inverse-metric norm before the minimax
    simplex solve. This removes arbitrary objective units, never selects
    weights from candidate energies. Enumerate at most63 tiny active sets.
    The final physical direction is normalized to UNIT full-cache FS speed
    only when a nonzero common direction exists. Actual nonlinear worst-chain
    distance still determines step length; no promised fresh-chain descent.
    """
    gs=np.asarray(gradients,float)
    if (gs.ndim!=2 or not 2<=len(gs)<=6 or len(names)!=len(gs)
            or len(set(names))!=len(names) or not all(isinstance(n,str) for n in names)
            or not np.isfinite(gs).all()):
        raise ValueError('two to six finite real gradients with distinct objective names required')
    solved=[full_rayleigh_sr_direction(psi,hpsi,jacobian,q,g,damping=damping) for g in gs]
    directions=np.stack([d for d,_ in solved])
    gram=gs@directions.T;gram=(gram+gram.T)/2
    norms=np.sqrt(np.maximum(np.diag(gram),0))
    if not np.isfinite(gram).all() or not np.isfinite(norms).all():
        raise FloatingPointError('nonfinite inverse-metric gradient geometry')
    info=dict(objective_names=list(names),objective_inverse_metric_norms=norms.tolist(),
              individual_solves=[i for _,i in solved],weights_use_gradients_only=True,
              normalization='each objective to unit inverse-metric gradient norm; final direction to unit full-cache FS',
              finite_cache_not_population_certificate=True,damping=float(damping))
    if np.any(norms<=1e-14):
        info.update(common_descent=False,status='a declared objective is stationary at numeric resolution',
            objective_weights=[0.]*len(gs),predicted_region_slopes=[0.]*len(gs),
            original_tangent_fs_speed=0.,tangent_fs_speed=0.,simplex_kkt_scaled_violation=0.)
        return np.zeros(gs.shape[1]),info
    normalized=gram/(norms[:,None]*norms[None,:])
    options=[]
    for size in range(1,len(gs)+1):
        for active in combinations(range(len(gs)),size):
            sub=normalized[np.ix_(active,active)]
            kkt=np.block([[sub,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(size),1.]
            solution=np.linalg.lstsq(kkt,rhs,rcond=None)[0]
            if np.linalg.norm(kkt@solution-rhs)>1e-9 or np.min(solution[:size]) < -1e-10:
                continue
            weights=np.zeros(len(gs));weights[list(active)]=np.maximum(solution[:size],0)
            weights/=weights.sum()
            options.append((float(weights@normalized@weights),weights))
    _,weights=min(options,key=lambda row:row[0])
    raw_direction=(weights/norms)@directions
    normalized_gains=(gs@raw_direction)/norms
    size_squared=float(weights@normalized_gains)
    kkt_error=max(0.,size_squared-float(normalized_gains.min()))
    if kkt_error>1e-8:raise FloatingPointError('normalized common-moment simplex KKT failed')
    a,_,_,_=tangent_system(psi,hpsi,jacobian,q)
    speed=float(np.linalg.norm(a@raw_direction)/np.sqrt(2))
    common=bool(normalized_gains.min()>1e-12 and speed>1e-12)
    direction=raw_direction/speed if common else np.zeros(gs.shape[1])
    info.update(common_descent=common,status='common finite-cache descent' if common else 'no resolved common finite-cache direction',
        objective_weights=weights.tolist(),predicted_region_slopes=(-gs@direction).tolist(),
        normalized_raw_gains=normalized_gains.tolist(),metric_norm_squared=size_squared,
        normalized_metric_cosines=normalized.tolist(),simplex_kkt_scaled_violation=kkt_error,
        original_tangent_fs_speed=speed,tangent_fs_speed=1. if common else 0.,
        raw_direction_norm=float(np.linalg.norm(direction)))
    return direction,info


__all__=['residual_moment_cotangents','normalized_common_moment_direction']
