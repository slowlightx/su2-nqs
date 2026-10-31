"""Exact three-sublattice collective-spin singlet as a magnetic control.

This compressed Dicke-basis calculation is a reference for a coplanar
magnetic rotor, not an optimized J1-J2 state. Each sublattice is in its
maximum-spin irrep. Three equal integer spins couple uniquely to a singlet.
Actual reduced-density eigenvalues and rotation covariance are checked.
"""
from __future__ import annotations

from itertools import product
from numbers import Integral
from typing import Any

import numpy as np

from su2_nqs.su2.clebsch_gordan import clebsch_gordan


def coplanar_collective_singlet_entanglement(
    sites_per_sublattice: int, region_sites_per_sublattice: int,
) -> dict[str, Any]:
    """Physical spectrum for equal A/B/C populations on either side of a cut.

    All labels are doubled integers. ``q_global=n_per_sublattice`` and the
    two sublattice pieces have ``q_A=n_region`` and ``q_B=n_global-n_region``.
    Permutation symmetry permits an exact compressed orthonormal basis of
    only ``(n_region+1)**3`` states, including the N=36 / 18-spin partition.
    q-resolved output is legitimate for this reference because the physical
    raising commutator is explicitly checked, not inferred from degeneracy.
    """
    n, a = sites_per_sublattice, region_sites_per_sublattice
    if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n,a))
            or not 2 <= n <= 16 or n % 2 or not 1 <= a < n):
        raise ValueError('even n in 2..16 and integer 1<=region<n required')
    n, a = int(n), int(a)
    b = n-a
    left = np.asarray(list(product(range(-a,a+1,2), repeat=3)), dtype=np.int64)
    right = np.asarray(list(product(range(-b,b+1,2), repeat=3)), dtype=np.int64)
    amplitudes = np.zeros((len(left), len(right)), dtype=np.float64)
    for i, ma in enumerate(left):
        for j, mb in enumerate(right):
            mg = ma+mb
            if mg.sum() == 0:
                global_coupling = (clebsch_gordan(n,int(mg[0]),n,int(mg[1]),n,int(-mg[2]))
                                   *clebsch_gordan(n,int(-mg[2]),n,int(mg[2]),0,0))
                splitting = np.prod([clebsch_gordan(a,int(x),b,int(y),n,int(x+y))
                                     for x,y in zip(ma,mb)])
                amplitudes[i,j] = global_coupling*splitting
    norm_error = float(abs(np.sum(amplitudes**2)-1.))
    if norm_error > 1e-11:
        raise RuntimeError('exact magnetic control failed normalization')
    rho = amplitudes @ amplitudes.T
    columns = [(q12,q,m) for q12 in range(0,2*a+1,2)
               for q in range(abs(q12-a),q12+a+1,2) for m in range(-q,q+1,2)]
    transform = np.zeros((len(left),len(columns)), dtype=np.float64)
    for i,(m1,m2,m3) in enumerate(left):
        for j,(q12,q,m) in enumerate(columns):
            if m1+m2+m3 == m and abs(m1+m2) <= q12:
                transform[i,j] = (clebsch_gordan(a,int(m1),a,int(m2),q12,int(m1+m2))
                                 *clebsch_gordan(q12,int(m1+m2),a,int(m3),q,m))
    unitary_error = float(np.max(abs(transform.T @ transform-np.eye(len(columns)))))
    raising = np.zeros_like(rho)
    positions = {tuple(ms):i for i,ms in enumerate(left)}
    for i,ms in enumerate(left):
        for sublattice in range(3):
            m = int(ms[sublattice])
            if m < a:
                changed = ms.copy(); changed[sublattice] += 2
                raising[positions[tuple(changed)],i] += .5*np.sqrt((a-m)*(a+m+2))
    raising_relative = float(np.linalg.norm(rho @ raising-raising @ rho)/np.linalg.norm(rho))
    if max(unitary_error,raising_relative) > 1e-10:
        raise RuntimeError('control failed CG orthonormality or physical SU(2) covariance')
    coupled = transform.T @ rho @ transform
    q_labels = np.asarray([q for _,q,_ in columns])
    m_labels = np.asarray([m for _,_,m in columns])
    sectors = {}
    spectrum = []
    for q in sorted(set(q_labels.tolist())):
        representative_m = q % 2
        indices = np.flatnonzero((q_labels == q) & (m_labels == representative_m))
        values = np.linalg.eigvalsh(coupled[np.ix_(indices,indices)])[::-1]
        spectrum.extend(np.repeat(values, q+1))
        sectors[str(q)] = dict(q=q, multiplicity_dimension=len(indices),
            one_m_eigenvalues=values.tolist(), magnetic_degeneracy=q+1,
            entanglement_energies=[float(-np.log(v)) if v>0 else None for v in values],
            weight=float((q+1)*values.sum()))
    spectrum = np.asarray(spectrum)
    positive = spectrum > 0
    return dict(n_sites=3*n, n_region=3*a, sites_per_sublattice=n,
        region_sites_per_sublattice=a, compressed_region_dimension=len(left),
        compressed_complement_dimension=len(right), normalization_error=norm_error,
        cg_orthonormality_max_error=unitary_error,
        physical_raising_commutator_relative_norm=raising_relative,
        strict_su2_verified=True, q_sectors=sectors,
        von_neumann_entropy_nats=float(-spectrum[positive] @ np.log(spectrum[positive])),
        renyi2_entropy_nats=float(-np.log(spectrum @ spectrum)),
        amplitudes=amplitudes, region_rho=rho, region_magnetic_labels=left.sum(axis=1),
        warning='A collective-spin magnetic singlet reference, not a variational J1-J2 ground state or a finite-size phase classifier.')
