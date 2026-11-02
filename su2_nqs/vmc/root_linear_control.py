"""One development residual direction in an existing raw-isometric root.

This is a restricted original-W control, not a full-network tangent fit or a
claim of population stationarity. Finite line diagnostics use both H sides.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


def native_root_matrix(canonical: np.ndarray, left_frame: np.ndarray,
                       right_frame: np.ndarray) -> np.ndarray:
    """Invert ket-basis transport Ccanonical=A_left Cnative A_right.T.

    Frames are cross overlaps <canonical child|native child> in one doubled-q
    sector. The right operation is an ordinary transpose, including complex
    gauges. Full-rank frames are required; no pseudoinverse drops directions.
    """
    c,l,r=map(np.asarray,(canonical,left_frame,right_frame))
    if (c.ndim!=2 or l.shape!=(c.shape[0],)*2 or r.shape!=(c.shape[1],)*2
            or not all(np.isfinite(a).all() for a in (c,l,r))):
        raise ValueError('finite square full-rank child frames matching the root matrix required')
    return np.linalg.solve(r,np.linalg.solve(l,c).T).T


def projected_root_residual_direction(phi: np.ndarray, parent: np.ndarray,
    hparent: np.ndarray, proposal: np.ndarray, *, damping: float = .1,
    metric_rcond: float = 1e-10, maximum_rank: int = 8,
    hfeatures: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Freeze a raw-normalized root direction from a regularized score metric.

    phi columns must be physical amplitudes of an orthonormal RAW root basis.
    The projected metric is centered and represented by a thin real/imag SVD.
    By default only H(parent) is supplied, so the residual force is ONE-SIDED,
    not the derivative of a finite-sample Rayleigh quotient. Optional complete
    hfeatures supplies H of every column and gives the exact bilateral gradient
    of the REAL finite-sample quotient, even when its sampled H is non-Hermitian.
    A later two-column H evaluation measures the finite slope and curvature.
    Old/incomplete-support proposals are permitted for DEVELOPMENT only.
    """
    f, c, hp, q = map(np.asarray, (phi, parent, hparent, proposal))
    if (f.ndim != 2 or c.shape != (f.shape[1],) or hp.shape != (len(f),)
            or q.shape != hp.shape or np.iscomplexobj(c) or np.iscomplexobj(q)
            or not all(np.isfinite(x).all() for x in (f, c, hp, q)) or np.any(q <= 0)
            or abs(np.vdot(c,c).real-1) > 2e-9):
        raise ValueError('finite matching features, raw-unit real parent, and positive development Q required')
    if (not np.isfinite([damping,metric_rcond]).all() or damping <= 0 or not 0 < metric_rcond < 1
            or isinstance(maximum_rank,bool) or not isinstance(maximum_rank,Integral) or maximum_rank < 1):
        raise ValueError('positive damping, relative cutoff, and integer rank cap required')
    hf = None if hfeatures is None else np.asarray(hfeatures)
    if hf is not None:
        if hf.shape != f.shape or not np.isfinite(hf).all():
            raise ValueError('finite Hfeatures matching every amplitude column required')
        if np.max(abs(hf@c-hp)) > 3e-10*max(float(np.max(abs(hp))),np.finfo(float).tiny):
            raise ValueError('Hfeatures applied to parent must replay supplied Hparent')
    weight = np.sqrt(np.max(q)/q)
    scale = max(float(np.max(abs(f))), np.finfo(float).tiny)
    x, y = f/scale*weight[:,None], hp/scale*weight
    p = x @ c; norm = float(np.vdot(p,p).real)
    if norm <= 0: raise ValueError('parent has zero observed norm')
    energy = np.vdot(p,y)/norm
    center = (x-p[:,None]*(p.conj()@x/norm)[None,:])/np.sqrt(norm)
    real_center = np.concatenate((center.real,center.imag),axis=0)
    _, singular, vh = np.linalg.svd(real_center,full_matrices=False)
    eigen = singular**2; largest = float(eigen[0])
    cutoff = max(largest*metric_rcond,32*np.finfo(float).eps*f.shape[1]*float(np.max(np.sum(abs(x)**2,axis=0)))/norm)
    kept = np.flatnonzero(eigen > cutoff)[:maximum_rank]
    if not len(kept): raise ArithmeticError('no resolved nonzero projected root metric')
    one_sided_force = 2*np.real(x.conj().T@(y-energy.real*p))/norm
    force = one_sided_force
    if hf is not None:
        hx = hf/scale*weight[:,None]
        force = np.real(x.conj().T@y+hx.conj().T@p-2*energy.real*(x.conj().T@p))/norm
    frame = vh[kept].T
    raw_direction = -frame @ ((frame.T@force)/(eigen[kept]+damping*largest))
    raw_parallel = float(c@raw_direction)
    direction = raw_direction-c*raw_parallel
    size = float(np.linalg.norm(direction))
    if not np.isfinite(size) or size <= 1e-14: raise ArithmeticError('root direction vanished after raw projective centering')
    direction /= size
    projected = frame@(frame.T@force)
    return direction, dict(raw_parent_norm=float(c@c), raw_direction_norm=float(direction@direction),
        raw_parent_direction_overlap=float(c@direction), raw_removed_parent_component=raw_parallel,
        raw_direction_pre_normalization_norm=size, metric_eigenvalues=eigen.tolist(),
        metric_retained_rank=len(kept), metric_rank_cap=maximum_rank, metric_rcond=metric_rcond,
        relative_ridge=damping, ridge=damping*largest, metric_cutoff=cutoff,
        one_sided_force_norm=float(np.linalg.norm(one_sided_force)),
        one_sided_force_dot_direction=float(one_sided_force@direction),
        selected_force_norm=float(np.linalg.norm(force)), selected_force_dot_direction=float(force@direction),
        force_mode='one_sided' if hf is None else 'complete_bilateral_finite_sample',
        force_omitted_fraction=float(np.linalg.norm(force-projected)/max(np.linalg.norm(force),1e-300)),
        development_parent_energy_real=float(energy.real), development_parent_energy_imag=float(energy.imag),
        one_sided_force_is_not_finite_sample_energy_derivative=True,
        selected_force_is_exact_finite_sample_derivative=hf is not None,
        development_only_no_proposal_support_or_unbiasedness_claim=True,
        root_only_not_all_native_W=True, raw_isometric_coordinates_required=True)


def root_line_diagnostics(phi: np.ndarray, hphi: np.ndarray, proposal: np.ndarray,
                          t: float) -> dict:
    """Exact finite-sample moments/derivatives of Phi0+t Phi1, including H².

    Both numerator coherences are retained even for a non-Hermitian sampled
    cross matrix. FS speed is G=det(projected Gram)/Z(t)^2. Intrinsic curvature
    is E''/G-E'G'/(2G²), not the coordinate-dependent parameter curvature.
    These are diagnostics of the supplied rows, not independent evidence.
    """
    f,h,q = map(np.asarray,(phi,hphi,proposal))
    if (f.ndim != 2 or f.shape[1] != 2 or h.shape != f.shape or q.shape != (len(f),)
            or np.iscomplexobj(q) or not all(np.isfinite(a).all() for a in (f,h,q))
            or np.any(q <= 0) or not np.isfinite(t)):
        raise ValueError('finite two-column amplitudes/actions and positive Q required')
    scale=max(float(np.max(abs(f))),np.finfo(float).tiny)
    weights=np.sqrt(np.max(q)/q)
    f=f/scale*weights[:,None];h=h/scale*weights[:,None]
    p=f[:,0]+t*f[:,1];hp=h[:,0]+t*h[:,1];d,hd=f[:,1],h[:,1]
    z=float(np.vdot(p,p).real)
    if z<=0:raise ValueError('line state has zero observed norm')
    z1=2*float(np.vdot(p,d).real);z2=2*float(np.vdot(d,d).real)
    n=np.vdot(p,hp);n1=np.vdot(d,hp)+np.vdot(p,hd);n2=2*np.vdot(d,hd)
    e=n/z;first=n1/z-n*z1/z**2
    second=n2/z-2*n1*z1/z**2-n*z2/z**2+2*n*z1*z1/z**3
    gram=f.conj().T@f
    det=float((gram[0,0]*gram[1,1]-abs(gram[0,1])**2).real)
    if det < -1e-10*max(float(np.prod(np.diag(gram).real)),1e-300):
        raise ArithmeticError('projected two-column Gram is not positive')
    metric=max(0.,det)/z**2;metric_first=-2*metric*z1/z
    overlap=abs(np.vdot(f[:,0],p))/np.sqrt(float(gram[0,0].real)*z)
    if overlap>1+2e-10:raise ArithmeticError('invalid projective overlap')
    return dict(energy_real=float(e.real),energy_imaginary=float(e.imag),
        energy_first=float(first.real),energy_second=float(second.real),
        residual_variance=float(np.vdot(hp-e*p,hp-e*p).real/z),
        FS_radians=float(np.arccos(min(1.,overlap))),FS_metric=metric,FS_metric_first=metric_first,
        arclength_slope=float(first.real/np.sqrt(metric)) if metric>0 else None,
        arclength_curvature=float(second.real/metric-first.real*metric_first/(2*metric**2)) if metric>0 else None,
        norm_ratio=float(z/gram[0,0].real),full_bilateral_energy_derivative=True,
        H2_from_complete_Hpsi_not_projected_H_squared=True)


def match_root_line_radius(phi: np.ndarray, proposal: np.ndarray, radius: float,
    *, sign: int = 1, bracket: float = .5, iterations: int = 16,
) -> tuple[float, dict]:
    """Fixed-bracket amplitude-only bisection; no energy or audit input.

    Choose one signed parent-connected branch, refuse an unbracketed target,
    and return its midpoint after exactly iterations steps. The radius is an
    empirical FS distance, not a confidence-bound or exact-population match.
    """
    f,q=map(np.asarray,(phi,proposal))
    if (f.ndim!=2 or f.shape[1]!=2 or q.shape!=(len(f),) or np.iscomplexobj(q)
            or not np.isfinite(f).all() or not np.isfinite(q).all() or np.any(q<=0)
            or sign not in (-1,1) or isinstance(sign,bool) or not np.isfinite([radius,bracket]).all()
            or not 0<radius<np.pi/2 or bracket<=0 or isinstance(iterations,bool)
            or not isinstance(iterations,Integral) or iterations<1):
        raise ValueError('finite two-column calibration, positive bracket/radius, and fixed iteration count required')
    x=f/max(float(np.max(abs(f))),1e-300)*np.sqrt(np.max(q)/q)[:,None]
    gram=x.conj().T@x
    def distance(value):
        c=np.array([1.,value]);norm=float(np.vdot(c,gram@c).real)
        overlap=abs(gram[0]@c)/np.sqrt(float(gram[0,0].real)*norm)
        return float(np.arccos(np.clip(overlap,0,1)))
    # The parent-connected interval must not cross zero real overlap.
    if (gram[0,0]+sign*bracket*gram[0,1]).real<=0 or distance(sign*bracket)<radius:
        raise ValueError('fixed signed interval does not bracket the requested parent-connected radius')
    low,high=0.,bracket
    for _ in range(iterations):
        mid=(low+high)/2
        if distance(sign*mid)<radius:low=mid
        else:high=mid
    t=sign*(low+high)/2
    return t,dict(target_FS_radians=radius,matched_FS_radians=distance(t),signed_t=t,
        absolute_FS_mismatch=abs(distance(t)-radius),fixed_bracket=bracket,bisection_iterations=iterations,
        amplitude_only=True,empirical_same_panel_radius_not_population_equality=True)
