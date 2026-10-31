"""Connected spin structure factors with explicit elastic and onsite terms.

The one-point moments are physical <S_i^alpha>, not spin labels. Full scalar
S(k) includes the spin-half self term 3/4. Connected subtraction removes only
<S_i>.<S_j>; the remaining onsite quantum variance is retained. An additional
offsite report can remove it explicitly, and need not be nonnegative.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


def _inputs(magnetizations: np.ndarray, coordinates: np.ndarray,
            momenta: Mapping[str, Sequence[float]]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    m=np.asarray(magnetizations);r=np.asarray(coordinates,float)
    k=np.asarray(list(momenta.values()),float)
    if (m.ndim!=2 or m.shape[1]!=3 or len(m)<1 or not np.isfinite(m).all()
            or np.iscomplexobj(m) or r.shape!=(len(m),2) or not np.isfinite(r).all()
            or k.ndim!=2 or k.shape[1]!=2 or len(k)<1 or not np.isfinite(k).all()
            or np.max(np.linalg.norm(m,axis=1))>.5+1e-12):
        raise ValueError('physical real spin-half moments (N,3), coordinates (N,2), and momenta required')
    return m.astype(float),r,k


def connected_spin_structure_factors(raw_scalar: np.ndarray, magnetizations: np.ndarray,
                                    coordinates: np.ndarray,
                                    momenta: Mapping[str, Sequence[float]]) -> dict[str,np.ndarray | float]:
    """Subtract elastic scalar Fourier weight; preserve onsite variance.

    raw_scalar[...,k] contains full <S_i.S_j>/N and includes i=j. Moment
    columns follow insertion order. This is an algebraic transformation;
    statistical error for estimated one-point inputs requires a paired
    nonlinear jackknife or equivalent propagation by the caller.
    """
    m,r,k=_inputs(magnetizations,coordinates,momenta)
    raw=np.asarray(raw_scalar)
    if raw.ndim<1 or raw.shape[-1]!=len(k) or not np.isfinite(raw).all():
        raise ValueError('one finite raw structure value per momentum required')
    phase=np.exp(2j*np.pi*r@k.T)
    elastic=np.sum(abs(phase.T@m)**2,axis=1)/len(m)
    onsite=float(np.mean(.75-np.sum(m*m,axis=1)))
    connected=raw-elastic
    return dict(raw=raw,elastic=elastic,connected=connected,
                onsite_quantum_variance=onsite,connected_offsite=connected-onsite)


def connected_longitudinal_structure_factors(raw_zz: np.ndarray,
                                             magnetizations: np.ndarray,
                                             coordinates: np.ndarray,
                                             momenta: Mapping[str, Sequence[float]]) -> dict[str,np.ndarray | float]:
    """SzSz covariance, retaining 1/4-<Sz_i>² on each diagonal."""
    m,r,k=_inputs(magnetizations,coordinates,momenta)
    raw=np.asarray(raw_zz)
    if raw.ndim<1 or raw.shape[-1]!=len(k) or not np.isfinite(raw).all():
        raise ValueError('one finite longitudinal structure value per momentum required')
    phase=np.exp(2j*np.pi*r@k.T)
    elastic=abs(phase.T@m[:,2])**2/len(m)
    onsite=float(np.mean(.25-m[:,2]**2));connected=raw-elastic
    return dict(raw=raw,elastic=elastic,connected=connected,
                onsite_quantum_variance=onsite,connected_offsite=connected-onsite)
