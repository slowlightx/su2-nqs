"""Residual-aware geometry on saved physical tangents, not a new ansatz.

All H amplitudes are TOTAL Hpsi. Real parameter directions, complex states,
strictly positive fixed Q. The positive response Gram is a Gauss--Newton
trust metric, NOT the complete Hessian or a population-variance certificate.
"""
from __future__ import annotations

from itertools import combinations
from collections.abc import Sequence
import numpy as np

from .carrier_protection import physical_moments


def residual_response_gram(psi: np.ndarray, hpsi: np.ndarray,
                           deltas: np.ndarray, hdeltas: np.ndarray,
                           q: np.ndarray, n_sites: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return FS Gram and positive centered-residual-response Gram per site.

    Tangent columns are physical delta psi,H delta psi, never tensor banks.
    Remove beta*psi from delta AND beta*Hpsi from Hdelta. With t,Ht so
    centered, retain complex finite-Q dE=<t|Hpsi>+<psi|Ht>, and
    k=(H-E)t-dE*psi. Gram(k)/N penalizes large residual response invisible
    to FS. Norm/energy second derivatives can have either sign, so this
    positive matrix must not be called the exact residual Hessian.
    """
    parent=physical_moments(psi,hpsi,q,n_sites)
    p,h,d,hd,density=(np.asarray(psi,complex),np.asarray(hpsi,complex),
                      np.asarray(deltas,complex),np.asarray(hdeltas,complex),np.asarray(q,float))
    if (d.ndim!=2 or not d.shape[1] or d.shape[0]!=len(p) or hd.shape!=d.shape
            or not np.isfinite(d).all() or not np.isfinite(hd).all()):
        raise ValueError('finite matching physical tangent columns required')
    z=float(np.sum(abs(p)**2/density));w=1/(density*z)
    e=np.sum(p.conj()*h*w)
    beta=np.sum(p.conj()[:,None]*d*w[:,None],axis=0)
    t,ht=d-p[:,None]*beta,hd-h[:,None]*beta
    de=np.sum((t.conj()*h[:,None]+p.conj()[:,None]*ht)*w[:,None],axis=0)
    k=ht-e*t-p[:,None]*de
    fs=(t.conj().T@(w[:,None]*t)).real
    response=(k.conj().T@(w[:,None]*k)).real/n_sites
    residual=h-e*p
    dr=2*np.sum(residual.conj()[:,None]*k*w[:,None],axis=0).real/n_sites
    return (fs+fs.T)/2,(response+response.T)/2,dict(
        parent_residual_per_site=parent['variance_per_site'],
        energy_slopes_per_site=(de.real/n_sites).tolist(),residual_slopes_per_site=dr.tolist(),
        residual_response_is_not_exact_Hessian=True,finite_cache_not_population_certificate=True)


def residual_curvature_direction(psi: np.ndarray, hpsi: np.ndarray,
                                  deltas: np.ndarray, hdeltas: np.ndarray,
                                  parameter_directions: np.ndarray, q: np.ndarray,
                                  region_masks: Sequence[np.ndarray], gradients: np.ndarray,
                                  objective_names: Sequence[str], n_sites: int, *,
                                  damping: float=.1, response_weight: float=1.,
                                  residual_floor: float=1e-12) -> tuple[np.ndarray, dict]:
    """Return coefficients c for actual update B@c, with common E/R descent.

    B contains SAVED real parameter-update columns. Base metric is EXACT
    B.T@(2FS+damping I)@B. Whiten its resolved span before adding the mean
    of each declared region's Gram(k)/(N*max(parent R,floor)). No change in
    state capacity; no selected radius, damping or objective-weight sweep.
    The final direction has unit FULL physical FS speed, not unit enlarged
    trust-metric speed. All coefficients use known controls only.
    """
    b,g=np.asarray(parameter_directions,float),np.asarray(gradients,float)
    d,hd=np.asarray(deltas,complex),np.asarray(hdeltas,complex)
    if (b.ndim!=2 or not np.isfinite(b).all() or g.ndim!=2 or not 2<=len(g)<=6
            or g.shape[1]!=b.shape[1] or d.ndim!=2 or d.shape[1]!=b.shape[1]
            or len(objective_names)!=len(g) or len(set(objective_names))!=len(g)
            or not all(isinstance(name,str) and name for name in objective_names)
            or not np.isfinite(g).all()):
        raise ValueError('finite parameter columns and two to six named projected gradients required')
    if any(isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in (damping,response_weight,residual_floor)):
        raise ValueError('positive finite trust parameters required')
    fs,_,_=residual_response_gram(psi,hpsi,d,hd,q,n_sites)
    base=2*fs+damping*(b.T@b)
    eig,vec=np.linalg.eigh((base+base.T)/2)
    keep=eig>max(float(eig.max()),1e-300)*1e-10
    if not keep.any():raise ValueError('saved span has no resolved positive metric direction')
    whitening=vec[:,keep]/np.sqrt(eig[keep])[None,:]
    normalized_response=np.zeros_like(base);region_info=[]
    if not region_masks:raise ValueError('declared full/half regions required')
    for mask in region_masks:
        m=np.asarray(mask)
        if m.dtype!=bool or m.shape!=np.asarray(q).shape or not m.any():
            raise ValueError('nonempty matching boolean region masks required')
        _,response,info=residual_response_gram(np.asarray(psi)[m],np.asarray(hpsi)[m],d[m],hd[m],np.asarray(q)[m],n_sites)
        normalized_response+=response/max(info['parent_residual_per_site'],residual_floor)/len(region_masks)
        region_info.append(info)
    metric=np.eye(int(keep.sum()))+response_weight*(whitening.T@normalized_response@whitening)
    gw=g@whitening
    solved=np.linalg.solve(metric,gw.T).T
    gram=gw@solved.T;gram=(gram+gram.T)/2
    norms=np.sqrt(np.maximum(np.diag(gram),0))
    info=dict(objective_names=list(objective_names),saved_span_rank=int(keep.sum()),
        base_metric_eigenvalues=eig.tolist(),residual_response_regions=region_info,
        damping=float(damping),response_weight=float(response_weight),
        residual_response_is_not_exact_Hessian=True,weights_use_gradients_only=True,
        finite_cache_not_population_certificate=True,architecture_unchanged=True)
    if np.any(norms<=1e-14):
        info.update(common_descent=False,status='a projected objective is unresolved stationary')
        return np.zeros(b.shape[1]),info
    normalized=gram/(norms[:,None]*norms[None,:]);options=[]
    for size in range(1,len(g)+1):
        for active in combinations(range(len(g)),size):
            sub=normalized[np.ix_(active,active)]
            kkt=np.block([[sub,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(size),1.];solution=np.linalg.lstsq(kkt,rhs,rcond=None)[0]
            if np.linalg.norm(kkt@solution-rhs)>1e-9 or np.min(solution[:size]) < -1e-10:continue
            weights=np.zeros(len(g));weights[list(active)]=np.maximum(solution[:size],0);weights/=weights.sum()
            options.append((float(weights@normalized@weights),weights))
    _,weights=min(options,key=lambda row:row[0])
    # Minus sign: B columns are actual UPDATES, unlike saved SR d with update -d.
    raw=-(whitening@((weights/norms)@solved))
    gains=-(g@raw)/norms;squared=float(weights@gains)
    violation=max(0.,squared-float(gains.min()))
    if violation>1e-8:raise FloatingPointError('residual-trust simplex KKT fails')
    speed=float(np.sqrt(max(float(raw@fs@raw),0)))
    common=bool(gains.min()>1e-12 and speed>1e-12)
    c=raw/speed if common else np.zeros_like(raw)
    linear_residual=float(np.linalg.norm(solved@metric-gw)/max(np.linalg.norm(gw),1e-30))
    if linear_residual>1e-7:raise FloatingPointError('residual-trust solve fails independent residual')
    info.update(common_descent=common,status='common saved-span descent' if common else 'no resolved common saved-span descent',
        objective_weights=weights.tolist(),predicted_region_slopes=(g@c).tolist(),
        simplex_kkt_scaled_violation=violation,relative_linear_residual=linear_residual,
        tangent_fs_speed=1. if common else 0.,
        normalized_residual_response=float(c@normalized_response@c),
        normalized_metric_cosines=normalized.tolist(),coefficients=c.tolist())
    return c,info


__all__=['residual_response_gram','residual_curvature_direction']
