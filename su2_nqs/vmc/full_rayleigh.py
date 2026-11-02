"""Full two-sided finite-Q energy gradient and amplitude-metric SR.

Real model parameters, complex amplitudes, TOTAL H products, positive Q.
The reverse gradient must come from the actual full H action, not a presumed
Hermitian identity on incomplete quadrature. These are optimizer controls,
not population-energy guarantees. Carrier nodes are retained.
"""
from __future__ import annotations

import math
from itertools import combinations
import numpy as np

from .importance_tangent import tangent_system


def reverse_rayleigh_weights(psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Complex coefficients of Re<C|H delta>/norm; Q normalization cancels."""
    p, density = np.asarray(psi, complex), np.asarray(q, float)
    if (p.ndim != 1 or not len(p) or density.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(density).all() or np.any(density <= 0)):
        raise ValueError('matching finite amplitudes and strictly positive Q required')
    norm = float(np.sum(abs(p)**2/density))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('finite positive parent norm required')
    return p.conj()/(density*norm)


def full_rayleigh_gradient(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                           q: np.ndarray, reverse_gradient: np.ndarray) -> dict[str, np.ndarray | float]:
    """Exact Re-Rayleigh derivative of supplied finite quadrature.

    reverse_gradient[j]=Re<C|H J_j>/norm BEFORE parent-ray gauge removal.
    Both centered coherences are returned separately; only their SUM is the
    energy derivative. Individual signs are not a physical descent criterion.
    """
    _, _, tangent, _ = tangent_system(psi, hpsi, jacobian, q)
    p, hp, j, density = (np.asarray(psi, complex), np.asarray(hpsi, complex),
                         np.asarray(jacobian, complex), np.asarray(q, float))
    reverse = np.asarray(reverse_gradient, float)
    if reverse.shape != (j.shape[1],) or not np.isfinite(reverse).all():
        raise ValueError('finite real reverse gradient matching parameter dimension required')
    norm = float(np.sum(abs(p)**2/density))
    energy = np.sum(p.conj()*hp/density)/norm
    beta = np.sum(p.conj()[:, None]*j/density[:, None], axis=0)/norm
    forward = np.sum(tangent.conj()*hp[:, None]/np.sqrt(density*norm)[:, None], axis=0).real
    backward = reverse-(beta*energy).real
    gradient = forward+backward
    return dict(gradient=gradient, forward=forward, reverse=backward,
        gauge_beta=beta, energy_real=float(energy.real), energy_imaginary=float(energy.imag),
        norm=float(norm), quadrature_hermiticity_defect_norm=float(np.linalg.norm(backward-forward)))


def full_rayleigh_sr_direction(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                               q: np.ndarray, gradient: np.ndarray, *, damping: float = .1
                               ) -> tuple[np.ndarray, dict]:
    """Solve (2*FS metric + damping I)d=FULL gradient; actual update is -d.

    Woodbury handles gradients outside the sampled Jacobian row space; never
    silently project away reverse-action components. Report that fraction:
    incomplete finite-Q gradients can exploit off-cache amplitudes, which is
    why cross-chain actual-distance, full-H quality and fresh audit are needed.
    """
    if isinstance(damping, bool) or not math.isfinite(damping) or damping <= 0:
        raise ValueError('positive finite damping required')
    a, _, _, _ = tangent_system(psi, hpsi, jacobian, q)
    g = np.asarray(gradient, float)
    if g.shape != (a.shape[1],) or not np.isfinite(g).all():
        raise ValueError('finite full real gradient of correct dimension required')
    d = (g-a.T@np.linalg.solve(a@a.T+damping*np.eye(a.shape[0]), a@g))/damping
    error = a.T@(a@d)+damping*d-g
    residual = float(np.linalg.norm(error)/max(np.linalg.norm(g), 1e-30))
    if not np.isfinite(d).all() or residual > 1e-7:
        raise FloatingPointError('full-Rayleigh SR solve failed independent residual')
    _, s, vh = np.linalg.svd(a, full_matrices=False)
    retained = s > (max(a.shape)*np.finfo(float).eps*s[0] if len(s) else 0)
    row = vh[retained].T@(vh[retained]@g)
    info = dict(relative_linear_residual=residual, gradient_norm=float(np.linalg.norm(g)),
        raw_direction_norm=float(np.linalg.norm(d)), predicted_total_energy_slope=float(-g@d),
        tangent_fs_speed=float(np.linalg.norm(a@d)/np.sqrt(2)), damping=float(damping),
        gradient_outside_sampled_tangent_fraction=float(np.linalg.norm(g-row)/max(np.linalg.norm(g), 1e-30)),
        sample_matrix_dimension=a.shape[0], finite_cache_not_population_certificate=True)
    return d, info


def common_rayleigh_sr_direction(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                                 q: np.ndarray, gradients: np.ndarray, *, damping: float = .1
                                 ) -> tuple[np.ndarray, dict]:
    """Minimax infinitesimal descent for up to three declared region objectives.

    With M=2*FS+damping I, minimize ||sum_i weight_i*g_i||^2_{M^-1}
    on the probability simplex, then d=M^-1 sum_i weight_i*g_i. KKT gives
    g_i.d >= d.M.d for EVERY objective. An origin in the convex hull signals
    no common first-order direction for these finite-cache objectives; it is
    not an architecture no-go. Weights use gradients only, never candidate
    energies, variance or audits. The physical parameter direction is -d.
    """
    gs = np.asarray(gradients, float)
    if gs.ndim != 2 or not 2 <= gs.shape[0] <= 3 or not np.isfinite(gs).all():
        raise ValueError('two or three finite real declared objective gradients required')
    solves = [full_rayleigh_sr_direction(psi,hpsi,jacobian,q,g,damping=damping) for g in gs]
    directions = np.stack([d for d,_ in solves])
    gram = gs@directions.T; gram=(gram+gram.T)/2
    scale=max(float(np.max(abs(gram))),1e-300); scaled=gram/scale
    options=[]
    for size in range(1,len(gs)+1):
        for active in combinations(range(len(gs)),size):
            sub=scaled[np.ix_(active,active)]
            kkt=np.block([[sub,np.ones((size,1))],[np.ones((1,size)),np.zeros((1,1))]])
            rhs=np.r_[np.zeros(size),1.]
            solution=np.linalg.lstsq(kkt,rhs,rcond=None)[0]
            if np.linalg.norm(kkt@solution-rhs)>1e-9 or np.min(solution[:size]) < -1e-10:
                continue
            weights=np.zeros(len(gs));weights[list(active)]=np.maximum(solution[:size],0)
            weights/=weights.sum()
            options.append((float(weights@scaled@weights),weights))
    _,weights=min(options,key=lambda row:row[0])
    direction=weights@directions; gains=gs@direction
    norm_squared=float(weights@gains)
    kkt_error=max(0.,norm_squared-float(gains.min()))/scale
    if kkt_error>1e-8:raise FloatingPointError('common-gradient simplex KKT condition failed')
    a,_,_,_=tangent_system(psi,hpsi,jacobian,q)
    result=dict(objective_weights=weights.tolist(),predicted_region_total_slopes=(-gains).tolist(),
        metric_norm_squared=norm_squared,simplex_kkt_scaled_violation=kkt_error,
        tangent_fs_speed=float(np.linalg.norm(a@direction)/np.sqrt(2)),
        raw_direction_norm=float(np.linalg.norm(direction)),damping=float(damping),
        individual_solves=[info for _,info in solves],weights_use_gradients_only=True,
        common_descent_at_fixed_controls=bool(gains.min()>1e-12*scale),
        finite_cache_not_population_certificate=True)
    return direction,result


__all__ = ['reverse_rayleigh_weights', 'full_rayleigh_gradient', 'full_rayleigh_sr_direction',
           'common_rayleigh_sr_direction']
