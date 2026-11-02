"""Small complete triangle/6/9-spin tables and real C3 intertwiners.

All CG work is offline NumPy. Physical runtime tables fix doubled m from
the bits, and retain every multiplicity before the declared compression.
No global fusion basis or symbolic JAX runtime is introduced.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from .clebsch_gordan import clebsch_gordan
from .quantum_numbers import coupling_range
from .triangle_c3 import C3Multiplicity, ternary_c3_representation, triangle_chiral_schur


def _freeze(x: np.ndarray) -> np.ndarray:
    result = np.array(x, copy=True)
    result.setflags(write=False)
    return result


def real_multiplicity_frame(time_reversal: np.ndarray) -> np.ndarray:
    """S with T S* = S for a real involutive partner permutation T.

    Partner pairs become (|+>+|->)/sqrt2 and (-i|+>+i|->)/sqrt2.
    Magnetic time reversal remains in the fixed CG structure.
    """
    t = np.asarray(time_reversal, complex)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError('square multiplicity time-reversal matrix required')
    n = len(t)
    if (np.max(abs(t.imag)) > 1e-12 or not np.allclose(t @ t, np.eye(n), atol=1e-12)
            or not np.allclose(t.T @ t, np.eye(n), atol=1e-12)):
        raise ValueError('real involutive partner permutation required')
    columns = []
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        ids = np.flatnonzero(abs(t[:,i]) > 1e-12)
        if len(ids) != 1 or abs(t[ids[0],i]-1) > 1e-12:
            raise ValueError('only positive fixed/partner permutations are supported')
        j = int(ids[0]); visited.update((i,j))
        e = np.eye(n, dtype=complex)
        if i == j:
            columns.append(e[:,i])
        else:
            columns.extend(((e[:,i]+e[:,j])/np.sqrt(2),
                            (-1j*e[:,i]+1j*e[:,j])/np.sqrt(2)))
    s = np.column_stack(columns)
    if not np.allclose(t @ s.conj(), s, atol=1e-12):
        raise ArithmeticError('multiplicity realification failed')
    return _freeze(s)


def _projector_columns(projector: np.ndarray) -> np.ndarray:
    """Deterministic twice-reorthogonalized columns; no degenerate eigenbasis."""
    columns = []
    for i in range(len(projector)):
        v = projector[:,i].copy()
        for _ in range(2):
            for old in columns:
                v -= old*np.vdot(old,v)
        norm = np.linalg.norm(v)
        if norm <= 1e-10:
            continue
        v /= norm
        first = np.flatnonzero(abs(v) > 1e-10)[0]
        v *= np.conj(v[first])/abs(v[first])
        columns.append(v)
    return np.column_stack(columns) if columns else np.zeros((len(projector),0),complex)


def real_c3_axes(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real trivial axes and paired real E axes of a real orthogonal C3 map."""
    r = np.asarray(rotation, complex)
    n = len(r)
    if (r.shape != (n,n) or np.max(abs(r.imag)) > 1e-11
            or not np.allclose(r.T @ r,np.eye(n),atol=1e-11)
            or not np.allclose(r @ r @ r,np.eye(n),atol=1e-11)):
        raise ValueError('real orthogonal order-three representation required')
    omega = np.exp(2j*np.pi/3)
    a = _projector_columns((np.eye(n)+r+r@r)/3).real
    z = _projector_columns((np.eye(n)+omega.conjugate()*r+omega.conjugate()**2*(r@r))/3)
    e = np.column_stack([part for i in range(z.shape[1])
                         for part in (np.sqrt(2)*z[:,i].real,np.sqrt(2)*z[:,i].imag)]) if z.size else np.zeros((n,0))
    u = np.column_stack((a,e))
    if u.shape != (n,n) or not np.allclose(u.T @ u,np.eye(n),atol=2e-11):
        raise ArithmeticError('C3 real irrep axes failed orthonormality')
    return _freeze(a), _freeze(e)


def real_c3_synthesis_basis(rotation: np.ndarray, target: C3Multiplicity) -> np.ndarray:
    """Independent REAL C3+T maps in real input/output feature coordinates.

    No image-SVD over K*d raw parameters: decompose only the K<=48 local
    source rotation. Each source/target E pair has two real commuting maps.
    Reflection is not imposed. Return [independent_parameter,input,output].
    """
    trivial, pairs = real_c3_axes(rotation)
    n = len(rotation); d = target.width
    zero_out = target.a1+target.a2
    maps = []
    for i in range(trivial.shape[1]):
        for j in range(zero_out):
            w = np.zeros((n,d)); w[:,j] = trivial[:,i]; maps.append(w)
    for i in range(pairs.shape[1]//2):
        u,v = pairs[:,2*i:2*i+2].T
        for j in range(target.e_pairs):
            start = zero_out+2*j
            w = np.zeros((n,d)); w[:,start] = u/np.sqrt(2); w[:,start+1] = v/np.sqrt(2)
            maps.append(w)
            w = np.zeros((n,d)); w[:,start] = -v/np.sqrt(2); w[:,start+1] = u/np.sqrt(2)
            maps.append(w)
    if not maps:
        raise ValueError('no compatible source/output C3 channels')
    result = np.stack(maps)
    s = real_multiplicity_frame(target.time_reversal_multiplicity())
    out_rotation = (s.conj().T @ target.rotation() @ s).real
    error = np.max(abs(np.einsum('ij,pjk->pik',rotation,result)
                      -np.einsum('pij,jk->pik',result,out_rotation)))
    if error > 2e-11:
        raise ArithmeticError('C3 synthesis orientation mismatch')
    return _freeze(result)


def _cg(q0: int,m0: int,q1: int,m1: int,q: int) -> float:
    if any(abs(m)>charge or (charge-m)%2 for charge,m in ((q0,m0),(q1,m1),(q,m0+m1))):
        return 0.
    return clebsch_gordan(q0,m0,q1,m1,q,m0+m1)


@lru_cache(maxsize=1)
def triangle_block_tables() -> dict:
    """Complete physical tables; all q,m and feature labels use fixed orders.

    Each table's physical row already fixes m. Orthonormality is checked
    within one fixed m; summing over every m would multiply the Gram by q+1.
    """
    chiral, labels = triangle_chiral_schur()
    positions = {label:i for i,label in enumerate(labels)}
    bits3 = ((np.arange(8)[:,None] >> np.arange(2,-1,-1)) & 1)
    mags3 = (2*bits3-1).sum(1)
    local_c = {}
    local_r = {}
    for q in (1,3):
        width = 2 if q == 1 else 1
        u = np.zeros((8,width),complex)
        for packed,m in enumerate(mags3):
            if abs(m) <= q:
                u[packed] = [chiral[packed,positions[q,int(m),ell]] for ell in ((1,-1) if q==1 else (0,))]
        t = C3Multiplicity(q,e_pairs=1).time_reversal_multiplicity() if q==1 else np.eye(1)
        local_c[q] = _freeze(u)
        local_r[q] = _freeze((u @ real_multiplicity_frame(t)).real)
    six = {}; six_channels = {}
    for q in (0,2,4,6):
        channels = tuple((ql,qr,a,b) for ql,qr in product((1,3),repeat=2)
            if q in coupling_range(ql,qr)
            for a,b in product(range(local_r[ql].shape[1]),range(local_r[qr].shape[1])))
        u = np.zeros((64,len(channels)))
        for packed in range(64):
            left,right = packed>>3,packed&7
            ml,mr = int(mags3[left]),int(mags3[right])
            for k,(ql,qr,a,b) in enumerate(channels):
                u[packed,k] = _cg(ql,ml,qr,mr,q)*local_r[ql][left,a]*local_r[qr][right,b]
        six[q] = _freeze(u); six_channels[q] = channels
    nine = {}; source_rotations = {}; overlaps = {}; binary_channels = {}
    children = (C3Multiplicity(1,e_pairs=1),C3Multiplicity(3,a1=1))
    for q in (1,3,5,7,9):
        rep = ternary_c3_representation(children,q)
        s = real_multiplicity_frame(rep.time_reversal_multiplicity)
        u = np.zeros((512,len(rep.channels)),complex)
        binary = tuple((ql,qr,a,b) for ql in six for qr in local_r
            if q in coupling_range(ql,qr)
            for a,b in product(range(six[ql].shape[1]),range(local_r[qr].shape[1])))
        ub = np.zeros((512,len(binary)))
        for packed in range(512):
            p0,p1,p2 = packed>>6,(packed>>3)&7,packed&7
            m0,m1,m2 = (int(mags3[p]) for p in (p0,p1,p2))
            for k,(q0,q1,q2,q12,a0,a1,a2) in enumerate(rep.channels):
                u[packed,k] = (_cg(q0,m0,q1,m1,q12)*_cg(q12,m0+m1,q2,m2,q)
                    *local_c[q0][p0,a0]*local_c[q1][p1,a1]*local_c[q2][p2,a2])
            for k,(ql,qr,a,b) in enumerate(binary):
                ub[packed,k] = _cg(ql,m0+m1,qr,m2,q)*six[ql][packed>>3,a]*local_r[qr][p2,b]
        ur = u @ s
        if np.max(abs(ur.imag)) > 1e-12:
            raise ArithmeticError('real Schur table acquired an imaginary component')
        ur = ur.real
        high = np.asarray([2*int(i).bit_count()-9==q for i in range(512)])
        overlap = ub[high].T @ ur[high]
        if not np.allclose(overlap.T @ overlap,np.eye(len(rep.channels)),atol=1e-12):
            raise ArithmeticError('binary/ternary local basis is not complete')
        nine[q] = _freeze(ur)
        source_rotations[q] = _freeze((s.conj().T @ rep.rotation @ s).real)
        overlaps[q] = _freeze(overlap)
        binary_channels[q] = binary
    return dict(three=local_r,six=six,six_channels=six_channels,nine=nine,
                rotations=source_rotations,binary_overlaps=overlaps,
                binary_channels=binary_channels)


@lru_cache(maxsize=None)
def nine_site_output(q: int, multiplicity_cap: int) -> C3Multiplicity:
    """Retain complete real C3 irreps, including E whenever capacity permits."""
    if q not in (1,3,5,7,9) or isinstance(multiplicity_cap,bool) or not isinstance(multiplicity_cap,int) or multiplicity_cap<2:
        raise ValueError('odd doubled q<=9 and integer multiplicity cap>=2 required')
    rotation = triangle_block_tables()['rotations'][q]
    a,e = real_c3_axes(rotation)
    n0,ne = a.shape[1],e.shape[1]//2
    width = min(multiplicity_cap,len(rotation))
    allocations = [(z,(width-z)//2) for z in range(min(n0,width)+1)
                   if (width-z)%2==0 and (width-z)//2<=ne]
    if not allocations:
        raise ValueError('requested width does not retain complete C3 irreps')
    # Keep a trivial channel when available and at least one E pair when
    # possible; then favor pairs. At full width the allocation is unique.
    z,pairs = min(allocations,key=lambda x:(ne>0 and width>=2 and x[1]==0,
                     n0>0 and width>=3 and x[0]==0,-x[1],x[0]))
    return C3Multiplicity(q,a1=z,e_pairs=pairs)
