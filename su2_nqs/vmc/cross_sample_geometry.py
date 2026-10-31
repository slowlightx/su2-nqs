"""Node-safe forces and independent empirical geometry for real-parameter SR.

No model, Hamiltonian, sampling or Hilbert-label logic. Hpsi inputs are TOTAL
products. Q is a positive sampling density, not a doubled spin label. A
resolved tangent range is an explicit OPTIMIZER regularizer, never a new head
or a change in the wavefunction manifold. Empirical ranks are not guarantees
about the population metric or full-manifold stationarity.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
import numpy as np

from .importance_tangent import tangent_system
from .joint_moments import normalized_common_moment_direction


def hermitian_energy_force(psi: np.ndarray, hpsi: np.ndarray, jacobian: np.ndarray,
                           q: np.ndarray, n_sites: int) -> np.ndarray:
    """Return 2 Re<J_perp|(H-E)psi>/N in normalized importance units.

    This estimates the Hermitian POPULATION energy gradient. On incomplete
    finite Q it is NOT the derivative of the empirical Rayleigh ratio; that
    derivative retains both coherent terms and must still be used in FD and
    actual-candidate validation. Complete Hermitian support makes them agree.
    No division by psi, clipping, or presumed finite-Q reverse identity.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 1:
        raise ValueError('positive integer site count required')
    a,b,_,_=tangent_system(psi,hpsi,jacobian,q)
    return a.T@b/n_sites


def equal_mass_geometry(parents: Sequence[np.ndarray], jacobians: Sequence[np.ndarray],
                        densities: Sequence[np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Combine empirical blocks with equal parent-norm mass, then global gauge.

    Each Q block is multiplied by k*sum(|psi|^2/Q). The union parent norm is
    one; independent Q normalizers cancel. The later union gauge projection
    retains BETWEEN-block relative amplitude/phase changes, unlike averaging
    separately centered covariances. No energy/H products or outcomes enter.
    This is a declared empirical mixed geometry, not a globally known norm.
    """
    if not parents or len(parents)!=len(jacobians) or len(parents)!=len(densities):
        raise ValueError('matching nonempty geometry block sequences required')
    columns=None;ps=[];js=[];qs=[]
    for p,j,q in zip(parents,jacobians,densities,strict=True):
        p,j,q=np.asarray(p,complex),np.asarray(j,complex),np.asarray(q,float)
        tangent_system(p,np.zeros_like(p),j,q)
        if columns is not None and j.shape[1]!=columns:
            raise ValueError('all geometry blocks use the same real parameter columns')
        columns=j.shape[1];norm=float(np.sum(abs(p)**2/q))
        ps.append(p);js.append(j);qs.append(q*(len(parents)*norm))
    return np.concatenate(ps),np.concatenate(js),np.concatenate(qs)


def observable_common_direction(psi: np.ndarray, jacobian: np.ndarray, q: np.ndarray,
                                gradients: np.ndarray, names: Sequence[str], *,
                                damping: float=.1, relative_cutoff: float=1e-6
                                ) -> tuple[np.ndarray,dict]:
    """Minimax SR in eigenmodes of 2FS with lambda > cutoff*lambda_max.

    Build the small sample Gram, then orthonormal REAL parameter modes V.
    Reuse the existing normalized common-gradient solver on J@V,g@V and
    lift the update to ALL original parameters. Report every gradient's
    discarded fraction explicitly: unlike unfiltered ridge, unmeasured
    directions do not receive g_null/damping updates. Mode selection uses
    ONLY the empirical geometry, not H, gradients, energy or audit outcomes.
    Unit-speed normalization refers to this supplied geometry; callers must
    separately measure fit/fresh and nonlinear physical distances.
    """
    if (isinstance(relative_cutoff,bool) or not np.isfinite(relative_cutoff)
            or not 0<relative_cutoff<1 or isinstance(damping,bool)
            or not np.isfinite(damping) or damping<=0):
        raise ValueError('positive damping and relative cutoff in (0,1) required')
    p,j,density=np.asarray(psi,complex),np.asarray(jacobian,complex),np.asarray(q,float)
    a,_,_,_=tangent_system(p,np.zeros_like(p),j,density)
    gs=np.asarray(gradients,float)
    if (gs.ndim!=2 or not 2<=len(gs)<=6 or gs.shape[1]!=j.shape[1]
            or len(names)!=len(gs) or len(set(names))!=len(names)
            or not all(isinstance(name,str) and name for name in names)
            or not np.isfinite(gs).all()):
        raise ValueError('two to six matching finite named gradients required')
    sample_gram=a@a.T
    eig,u=np.linalg.eigh((sample_gram+sample_gram.T)/2)
    threshold=max(float(eig[-1]),0.)*relative_cutoff
    keep=(eig>threshold)&(eig>0)
    info=dict(optimizer_range_only=True,state_manifold_unchanged=True,
        mode_selection_uses_geometry_only=True,relative_cutoff=float(relative_cutoff),
        eigenvalue_threshold=threshold,resolved_rank=int(keep.sum()),
        parameter_count=j.shape[1],sample_gram_dimension=len(a),
        retained_metric_eigenvalues=eig[keep].tolist(),
        empirical_rank_not_population_or_stationarity_certificate=True)
    if not keep.any():
        info.update(common_descent=False,status='no resolved physical tangent',
            gradient_discarded_fractions=[1.]*len(gs),tangent_fs_speed=0.,mode_orthogonality_error=0.)
        return np.zeros(j.shape[1]),info
    modes=a.T@u[:,keep]/np.sqrt(eig[keep])[None,:]
    orth=float(np.linalg.norm(modes.T@modes-np.eye(keep.sum()),ord=2))
    if orth>1e-7:raise FloatingPointError('resolved real parameter modes not orthonormal')
    reduced=gs@modes
    discarded=np.linalg.norm(gs-reduced@modes.T,axis=1)/np.maximum(np.linalg.norm(gs,axis=1),1e-30)
    small,solve=normalized_common_moment_direction(p,np.zeros_like(p),j@modes,density,reduced,names,damping=damping)
    direction=modes@small
    info.update(solve)
    info.update(mode_orthogonality_error=orth,gradient_discarded_fractions=discarded.tolist(),
        original_parameter_direction_norm=float(np.linalg.norm(direction)),
        original_predicted_region_slopes=(-gs@direction).tolist())
    return direction,info


__all__=['hermitian_energy_force','equal_mass_geometry','observable_common_direction']
