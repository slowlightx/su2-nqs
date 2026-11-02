"""Actual spatial transformation of a few physical Schmidt vectors.

Only vectors (2**|A| times a few columns), NEVER an |A|-site RDM, are
enumerated. No state projection, canonicalization or new variational state.
Spatial labels refer to the strip stabilizer, not the full 2D space group.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any
import numpy as np


def reconstruct_prefix_vectors(psi: Any, cut: int, coefficients: np.ndarray) -> np.ndarray:
    """Return A_prefix @ coefficients, including the ACTUAL cut SR factor.

    Columns are the inverse-Gram-normalized coefficients from
    ``gram_leading_spin_content``. Bit0 is down, bit1 up, site0 is MSB.
    TeNPy's physical-leg order is inspected, never assumed to match this.
    This bounded exact vector readout supports 1<=cut<=20, not a large-RDM
    algorithm. The supplied MPS is never mutated.
    """
    if (isinstance(cut,bool) or not isinstance(cut,Integral) or not 1<=cut<=20
            or cut>=psi.L or psi.bc!='finite'):
        raise ValueError('finite MPS and bounded internal prefix cut required')
    c=np.asarray(coefficients,dtype=np.complex128)
    if c.ndim!=2 or c.shape[0]!=psi.chi[cut-1] or not 1<=c.shape[1]<=16 or not np.isfinite(c).all():
        raise ValueError('finite physical Schmidt coefficients with at most16 columns required')
    s=psi.get_SR(cut-1)
    if hasattr(s,'to_ndarray'):
        right=np.asarray(s.to_ndarray()) @ c
    else:
        right=np.asarray(s)[:,None]*c
    right=right[:,None,:]
    for i in range(cut-1,-1,-1):
        b=psi.get_B(i,form='A').transpose(['vL','p','vR']).to_ndarray()
        sz=np.diag(psi.sites[i].get_op('Sz').to_ndarray()).real
        order=[int(np.argmin(abs(sz+.5))),int(np.argmin(abs(sz-.5)))]
        if len(sz)!=2 or not np.allclose(sz[order],[-.5,.5],atol=1e-14,rtol=0):
            raise ValueError('physical spin-half legs required')
        # Reverse contraction keeps only a few vectors, not all cut-bond
        # columns. Do not form the enormous A_prefix matrix first.
        right=np.einsum('apr,rsv->apsv',b[:,order,:],right,optimize=True)
        right=right.reshape(right.shape[0],-1,c.shape[1])
    if right.shape[0]!=1:
        raise ValueError('trivial finite left MPS boundary required')
    return right[0]


def strip_spatial_content(vectors: np.ndarray, columns: int) -> dict:
    """C_columns momentum weights and Gamma-parallel mirror parity.

    U(g) v(bits)=v(bits[gather]); T gathers c-1, R gathers r-c modulo L.
    R T R=T^-1. Momentum k and mirror parity cannot both label a generic
    nonzero-k vector; only at k=0 (and k=pi for even L) is this possible.
    Near-degenerate real K-pair vectors are reported by projector weights
    and compressed family matrices, never forced into separate real momenta.
    """
    v=np.asarray(vectors,dtype=np.complex128)
    if (v.ndim!=2 or len(v)<2 or len(v)&(len(v)-1) or not np.isfinite(v).all()
            or isinstance(columns,bool) or not isinstance(columns,Integral) or columns<2):
        raise ValueError('finite power-of-two physical vectors and integer strip columns required')
    n=int(np.log2(len(v)))
    if n>20 or n%columns:
        raise ValueError('bounded complete-row strip required')
    error=float(np.linalg.norm(v.conj().T @ v-np.eye(v.shape[1])))
    if error>1e-8:
        raise ValueError('physical Schmidt vectors must already be orthonormal; no hidden repair')
    grid=np.arange(n).reshape(-1,columns)
    mirror=np.array([r*columns+(r-c)%columns for r in range(n//columns) for c in range(columns)])
    codes=np.arange(len(v),dtype=np.uint32)
    def matrix(gather: np.ndarray) -> np.ndarray:
        target=np.zeros(len(v),dtype=np.uint32)
        for site,old in enumerate(gather):
            target |= ((codes >> (n-1-int(old))) & 1) << (n-1-site)
        return v.conj().T @ v[target]
    translations=[matrix(np.roll(grid,t,axis=1).ravel()) for t in range(columns)]
    reflected=[matrix(mirror[np.roll(grid,t,axis=1).ravel()]) for t in range(columns)]
    projectors=[sum(np.exp(-2j*np.pi*k*t/columns)*translations[t] for t in range(columns))/columns
                for k in range(columns)]
    gamma=projectors[0]; rgamma=sum(reflected)/columns
    weights=np.stack([np.diag(p).real for p in projectors],axis=1)
    def split(a: np.ndarray) -> dict:
        return dict(real=a.real.tolist(),imag=a.imag.tolist())
    return dict(n_region=n,columns=int(columns),physical_orthonormality_error=error,
        momentum_weights=weights.tolist(),momentum_labels=list(range(columns)),
        momentum_unit='2*pi/columns; winding direction only, NOT full2D momentum',
        gamma_parallel_even_weights=np.diag((gamma+rgamma)/2).real.tolist(),
        gamma_parallel_odd_weights=np.diag((gamma-rgamma)/2).real.tolist(),
        momentum_weight_sum_error=float(np.max(abs(weights.sum(axis=1)-1))),
        translation_matrix=split(translations[1]),mirror_matrix=split(reflected[0]),
        translation_family_leakage=(1-np.sum(abs(translations[1])**2,axis=0)).tolist(),
        mirror_family_leakage=(1-np.sum(abs(reflected[0])**2,axis=0)).tolist(),
        permutation_convention='U(g)v(bits)=v(bits[gather]); T:c->c-1, R:c->r-c',
        projected_or_repaired=False,full_2D_momentum_assigned=False)
