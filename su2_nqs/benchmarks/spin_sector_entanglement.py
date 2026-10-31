"""Spin screening and multiplicity entanglement of an actual six-spin RDM.

Offline CG readout, not a variational ansatz. Labels are doubled q=2S.
No rotational twirl, eigenvalue repair, or strict SU(2) spectrum is inferred.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns
from .spin_partition_readout import _joint_projectors


def _cut(left_sites: Sequence[int]) -> tuple[int, ...]:
    left = tuple(left_sites)
    if (len(left) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in left)
            or len(set(left)) != 3 or min(left) < 0 or max(left) >= 6):
        raise ValueError('left_sites must contain three distinct sites in 0..5')
    return tuple(int(i) for i in left)


def _physical(rho: np.ndarray, tolerance: float) -> np.ndarray:
    matrix = np.asarray(rho, np.complex128)
    if matrix.shape != (64, 64) or not np.isfinite(matrix).all():
        raise ValueError('finite six-site physical RDM required')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive')
    if (abs(np.trace(matrix)-1) > tolerance or
            np.linalg.norm(matrix-matrix.conj().T) > tolerance or
            np.linalg.eigvalsh(matrix).min() < -tolerance):
        raise ValueError('RDM fails normalization, Hermiticity or positivity; no repair allowed')
    return matrix


def bipartite_negativity(rho: np.ndarray, dimensions: tuple[int, int]) -> float:
    """Return sum of negative partial-transpose eigenvalue magnitudes.

Input is a normalized physical density matrix in left/right product order.
Zero negativity does NOT prove separability in general dimensions.
"""
    if (len(dimensions) != 2 or any(isinstance(d, bool) or not isinstance(d, Integral)
                                 or d < 1 for d in dimensions)):
        raise ValueError('two positive integer subsystem dimensions required')
    dl, dr = map(int, dimensions)
    matrix = np.asarray(rho, np.complex128)
    if (matrix.shape != (dl*dr, dl*dr) or not np.isfinite(matrix).all() or
            abs(np.trace(matrix)-1) > 1e-9 or
            np.linalg.norm(matrix-matrix.conj().T) > 1e-9 or
            np.linalg.eigvalsh(matrix).min() < -1e-9):
        raise ValueError('normalized physical bipartite RDM required')
    transpose = matrix.reshape(dl, dr, dl, dr).transpose(0, 3, 2, 1)
    values = np.linalg.eigvalsh(transpose.reshape(dl*dr, dl*dr))
    return float(-np.minimum(values, 0).sum())


@lru_cache(maxsize=12)
def spin_screening_witness(q: int, left_sites: tuple[int, ...]) -> np.ndarray:
    """W=P(q,q,total=0)-P(q,q)/(q+1), with q in {1,3}.

For every state separable across the specified 3+3 cut, <W> <= 0.
The spin-j singlet has Schmidt rank q+1; its maximal product overlap is
1/(q+1). Local q projection and magnetic-index tracing preserve separability.
W>0 is sufficient for block entanglement, not a spin-liquid witness.
"""
    if isinstance(q, bool) or not isinstance(q, Integral) or q not in (1, 3):
        raise ValueError('q must be doubled spin 1 or 3')
    left = _cut(left_sites)
    projectors = _joint_projectors(6, left)
    pair = sum(p for (ql, qr, _), p in projectors if ql == qr == q)
    singlet = next(p for labels, p in projectors if labels == (q, q, 0))
    witness = singlet-pair/(q+1)
    witness.setflags(write=False)
    return witness


@lru_cache(maxsize=12)
def _local_schur_product(left: tuple[int, ...]) -> tuple[np.ndarray, tuple]:
    u, labels = regional_schur_columns(3)
    right = tuple(i for i in range(6) if i not in left)
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    grouped = bits[:, left+right] @ (2**np.arange(5, -1, -1))
    columns = np.kron(u, u)[grouped]
    columns.setflags(write=False)
    return columns, labels


def spin_sector_entanglement(rho: np.ndarray, left_sites: Sequence[int],
                             *, tolerance: float = 1e-9) -> dict:
    """Resolve screening versus internal multiplicity entanglement, without twirl.

Condition on LOCAL block charges (qL,qR), not the nonlocal total singlet.
Trace magnetic spaces out of each conditional local-Schur RDM and measure
negativity between its multiplicity subsystems. This is a legal invariant
subsystem readout even when the actual rho is not rotationally invariant.
Local charge measurement discards inter-charge coherence by an explicit LOCC
measurement; it is NOT a repaired replacement for the input RDM.
"""
    matrix, left = _physical(rho, tolerance), _cut(left_sites)
    columns, labels = _local_schur_product(left)
    transformed = columns.conj().T @ matrix @ columns
    raw = bipartite_negativity(transformed, (8, 8))
    projectors = _joint_projectors(6, left)
    sectors, weighted_sector, weighted_mult = {}, 0., 0.
    for ql in (1, 3):
        for qr in (1, 3):
            li = [i for i, l in enumerate(labels) if l[0] == ql]
            ri = [i for i, l in enumerate(labels) if l[0] == qr]
            ids = [8*i+j for i in li for j in ri]
            block = transformed[np.ix_(ids, ids)]
            p = float(np.trace(block).real)
            row = dict(q_left=ql, q_right=qr, probability=p,
                       conditional_negativity=None, multiplicity_negativity=None)
            if p > 1e-14:
                conditional = block/p
                al, ar = len(li)//(ql+1), len(ri)//(qr+1)
                tensors = conditional.reshape(ql+1, al, qr+1, ar,
                                                ql+1, al, qr+1, ar)
                multiplicity = np.einsum('iajbicjd->abcd', tensors).reshape(al*ar, al*ar)
                sector_n = bipartite_negativity(conditional, (len(li), len(ri)))
                mult_n = bipartite_negativity(multiplicity, (al, ar))
                row.update(conditional_negativity=sector_n, multiplicity_negativity=mult_n)
                weighted_sector += p*sector_n
                weighted_mult += p*mult_n
            if ql == qr:
                p0 = float(np.trace(matrix @ next(v for l, v in projectors
                                                   if l == (ql, qr, 0))).real)
                row.update(singlet_probability=p0,
                           screening_witness=p0-p/(ql+1),
                           conditional_spin_singlet_fraction=p0/p if p > 1e-14 else None)
            sectors[f'{ql},{qr}'] = row
    closure = abs(sum(row['probability'] for row in sectors.values())-1)
    violations = max(0., weighted_sector-raw, weighted_mult-weighted_sector)
    if max(closure, violations) > tolerance:
        raise RuntimeError('local charge/partial-trace entanglement monotonicity failed')
    return dict(left_sites=list(left), right_sites=[i for i in range(6) if i not in left],
                raw_negativity=raw, weighted_local_charge_negativity=float(weighted_sector),
                weighted_multiplicity_negativity=float(weighted_mult), sectors=sectors,
                diagnostics=dict(probability_closure_error=closure,
                                 negativity_monotonicity_violation=violations),
                input_repaired=False, strict_su2_spectrum_reported=False,
                warning='Block entanglement is not QSL/topological evidence. Raw/weighted '
                        'plug-in negativity can have positive finite-sampling bias; conditional '
                        'normalization adds ratio bias with no general fixed sign. Jackknife '
                        'SE does not remove these biases. '
                        'Conditioning on a nonlocal total-singlet sector would NOT certify '
                        'multiplicity entanglement of the original blocks and is not used.')


__all__ = ['bipartite_negativity', 'spin_screening_witness', 'spin_sector_entanglement']
