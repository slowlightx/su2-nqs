"""Small physical SU(2) gates, before any retained-space compression.

Computational rows use bit0=down and site0 as the most significant bit.
The real generator K=P_ij P_jk-P_jk P_ij is anti-Hermitian and commutes
with every total-spin generator. It is not an exponential of projected
single-spin matrices, which would discard physical response directions.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np


def cyclic_three_spin_action(vectors: np.ndarray, bits: np.ndarray, *,
        n_sites: int, sites: Sequence[int], theta: float) -> np.ndarray:
    """Apply exp(theta K) in a complete, permutation-closed physical row set.

    ``bits`` are ordered integer computational states, e.g. one complete
    doubled-m sector. A tall matrix applies the SAME invariant gate to all
    multiplicity columns. Missing permuted rows are rejected, not repaired.
    K^3=-3K gives an exact three-permutation formula, with no Trotter error.
    """
    b=np.asarray(bits);v=np.asarray(vectors)
    if (isinstance(n_sites,bool) or not isinstance(n_sites,Integral) or not 3<=n_sites<=30
            or len(sites)!=3 or len(set(sites))!=3
            or any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<n_sites for i in sites)):
        raise ValueError('three distinct physical sites in a3..30 spin block required')
    if (b.ndim!=1 or not np.issubdtype(b.dtype,np.integer) or not len(b)
            or np.any(b<0) or np.any(b>=1<<n_sites) or np.any(b[1:]<=b[:-1])
            or v.ndim not in (1,2) or len(v)!=len(b) or not np.isfinite(v).all()
            or isinstance(theta,(bool,complex)) or not np.isfinite(theta)):
        raise ValueError('finite vectors and sorted unique computational rows required')
    def swap(x, i,j):
        mi=1<<(n_sites-1-i);mj=1<<(n_sites-1-j)
        different=((x&mi)!=0)!=((x&mj)!=0)
        return x^np.where(different,mi|mj,0)
    i,j,k=map(int,sites)
    forward=swap(swap(b,i,j),j,k);reverse=swap(swap(b,j,k),i,j)
    fi=np.searchsorted(b,forward);ri=np.searchsorted(b,reverse)
    if (np.any(fi==len(b)) or np.any(ri==len(b))
            or not np.array_equal(b[fi],forward) or not np.array_equal(b[ri],reverse)):
        raise ValueError('physical row set is not closed under both gate permutations')
    a=np.sin(np.sqrt(3)*float(theta))/np.sqrt(3)
    c=(1-np.cos(np.sqrt(3)*float(theta)))/3
    return (1-2*c)*v+(a+c)*v[fi]+(c-a)*v[ri]
