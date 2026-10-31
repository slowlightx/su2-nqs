"""Explicitly bounded full-space diagnostics through L16.

This independent audit does not relax the original n<=14 core basis/ED
guards. It applies Pauli operators to numeric vectors, never in a model.
Labels use bit zero = Z=-1 and site zero as the most significant bit.
"""
from __future__ import annotations

import numpy as np
from fnqs.basis import validate_size


def finite_bits(n: int) -> np.ndarray:
    """Enumerate all bit labels for an even 4<=n<=16 diagnostic."""
    validate_size(n)
    if n > 16:
        raise ValueError('full-space diagnostic requires n <= 16')
    labels = np.arange(1 << n, dtype=np.uint32)
    return ((labels[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int32)


def vector_action(vector: np.ndarray, n: int, couplings: np.ndarray) -> np.ndarray:
    """Compute H@vector for Pauli PBC coupling order (j1,j2,jx,h)."""
    bits = finite_bits(n)
    vector = np.asarray(vector)
    g = np.asarray(couplings, dtype=np.float64)
    if vector.shape != (1 << n,) or g.shape != (4,) or not np.isfinite(vector).all() or not np.isfinite(g).all():
        raise ValueError('finite vector and four couplings required')
    z = 2*bits-1
    diagonal = -g[0]*np.sum(z*np.roll(z,-1,axis=1),axis=1)-g[1]*np.sum(z*np.roll(z,-2,axis=1),axis=1)
    result = diagonal*vector
    labels = np.arange(1 << n)
    for site in range(n):
        mask = 1 << (n-1-site)
        other = 1 << (n-1-(site+1)%n)
        result = result-g[3]*vector[labels^mask]-g[2]*vector[labels^mask^other]
    return result


def full_correlations(psi: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Return translation-averaged C(r), including C(0)=1, for all r."""
    bits = np.asarray(bits)
    psi = np.asarray(psi)
    if bits.ndim != 2 or psi.ndim != 1 or len(psi) != len(bits) or not np.isfinite(psi).all() or not np.all((bits==0)|(bits==1)):
        raise ValueError('state and binary configuration rows must agree')
    norm = float(np.sum(abs(psi)**2))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('nonzero finite state required')
    z = 2*bits.astype(np.int32)-1
    p = abs(psi)**2/norm
    return np.asarray([p@np.mean(z*np.roll(z,-r,axis=1),axis=1) for r in range(bits.shape[1])])
