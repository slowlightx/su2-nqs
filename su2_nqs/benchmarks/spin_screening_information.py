"""Screening versus internal resonance information in an actual six-spin RDM.

This is an offline physical readout, not a state decoder or an SU(2) repair.
Angular momenta q=2S and m=2M are doubled integers; bits are down/up, MSB first.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from .regional_su2_readout import regional_schur_columns
from .spin_operator_information import _low_order_span, spin_dot_matrix


def _left(left_sites: Sequence[int]) -> tuple[int, ...]:
    left = tuple(left_sites)
    if (len(left) != 3 or len(set(left)) != 3 or
            any(isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x < 6 for x in left)):
        raise ValueError('three distinct integer sites in 0..5 required')
    return tuple(map(int, left))


def _rho(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, np.complex128)
    if (matrix.shape != (64, 64) or not np.isfinite(matrix).all() or
            abs(np.trace(matrix)-1) > 1e-9 or
            np.linalg.norm(matrix-matrix.conj().T) > 1e-9 or
            np.linalg.eigvalsh(matrix).min() < -1e-9):
        raise ValueError('positive normalized Hermitian actual six-spin RDM required; no repair')
    return matrix


def _entropy(matrix: np.ndarray) -> float:
    """Entropy in nats, with 0 log0=0 and no support floor or ridge."""
    values = np.linalg.eigvalsh(matrix)
    if values.min() < -1e-9:
        raise ValueError('unphysical conditional state')
    support = values[values > 0]
    return float(-support @ np.log(support))


@lru_cache(maxsize=12)
def screening_columns(left: tuple[int, ...]) -> tuple[tuple[np.ndarray, ...], ...]:
    """CG columns |qL=1,qR=1;qT,mT,aL,aR>, qT in (0,2).

    Each 64x4 matrix has fixed doubled mT and ordered multiplicities (aL,aR).
    Nonlocal total-q conditioning is a measurement, not an LOCC entanglement witness.
    """
    left = _left(left)
    right = tuple(x for x in range(6) if x not in left)
    u, labels = regional_schur_columns(3)
    lookup = {label: i for i, label in enumerate(labels)}
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    grouped = bits[:, left+right] @ (2**np.arange(5, -1, -1))
    result = []
    for qt in (0, 2):
        magnetic = []
        for mt in range(-qt, qt+1, 2):
            vectors = []
            for al in (0, 1):
                for ar in (0, 1):
                    vector = np.zeros(64, complex)
                    for ml in (-1, 1):
                        for mr in (-1, 1):
                            vector += clebsch_gordan(1, ml, 1, mr, qt, mt)*np.kron(
                                u[:, lookup[(1, ml, al)]], u[:, lookup[(1, mr, ar)]])
                    vectors.append(vector[grouped])
            columns = np.column_stack(vectors)
            columns.setflags(write=False)
            magnetic.append(columns)
        result.append(tuple(magnetic))
    bank = np.column_stack([v for sector in result for v in sector])
    if np.linalg.norm(bank.conj().T@bank-np.eye(16)) > 1e-11:
        raise RuntimeError('screening CG orthonormality failed')
    return tuple(result)


def screening_multiplicity_information(rho: np.ndarray, left_sites: Sequence[int]) -> dict:
    """Holevo I(T:mu|qL=qR=1), and its probability-weighted contribution.

    T is the binary singlet/triplet screening outcome (doubled qT=0,2).
    mu denotes BOTH internal multiplicity qubits, not a magnetic index.
    All data come from a legal invariant channel on the actual rho, even
    when rho is not invariant. This is not a strict SU(2) entanglement spectrum.
    """
    matrix, left = _rho(rho), _left(left_sites)
    reduced = [sum(v.conj().T@matrix@v for v in sector)
               for sector in screening_columns(left)]
    weights = np.array([np.trace(r).real for r in reduced])
    p11 = float(weights.sum())
    if p11 <= 1e-14:
        return dict(p11=p11, screening_probabilities=None, holevo_nats=None,
                    weighted_holevo_nats=0., conditional_trace_distance=None,
                    reduced_multiplicity=reduced, strict_su2_spectrum=False)
    probabilities = weights/p11
    conditional = [r/p if p > 1e-14 else None for r, p in zip(reduced, weights)]
    holevo = _entropy(sum(reduced)/p11)-sum(
        probabilities[i]*_entropy(r) for i, r in enumerate(conditional) if r is not None)
    binary = -sum(p*np.log(p) for p in probabilities if p > 0)
    if holevo < -1e-9 or holevo > binary+1e-9:
        raise RuntimeError('Holevo positivity/binary bound failed')
    distance = None
    if all(r is not None for r in conditional):
        distance = float(np.abs(np.linalg.eigvalsh(conditional[0]-conditional[1])).sum()/2)
    return dict(p11=p11, screening_probabilities=probabilities.tolist(),
                holevo_nats=float(holevo), weighted_holevo_nats=float(p11*holevo),
                conditional_trace_distance=distance, reduced_multiplicity=reduced,
                strict_su2_spectrum=False, original_block_entanglement_certificate=False,
                warning='Information about total-spin screening and internal resonance, not QSL, topology, or an LOCC entanglement witness. Nonlinear plug-in/ratio bias remains.')


@lru_cache(maxsize=12)
def pseudoscalar_pair_split(left: tuple[int, ...]) -> dict:
    """Split chi_L chi_R against the DECLARED low-order+p(total q) linear span.

    chi=S_i.(S_j x S_k) follows ordered triples, which need not be elementary
    geometric triangles. Raw product is six-body, TR-even and SU(2)-scalar.
    O_high is HS-orthogonal to I, all pairs, symmetrized pair products and all
    regional total-spin projectors. Independence of this LINEAR span does not
    imply failure of a nonlinear maximum-entropy reconstruction.
    """
    left = _left(left)
    right = tuple(x for x in range(6) if x not in left)
    def chi(sites: tuple[int, ...]) -> np.ndarray:
        a = spin_dot_matrix(6, sites[0], sites[1])
        b = spin_dot_matrix(6, sites[1], sites[2])
        return 1j*(a@b-b@a)
    a, b = chi(left), chi(right)
    raw = a@b
    _, low_bank, _ = _low_order_span(6)
    u, labels = regional_schur_columns(6)
    projectors = [u[:, [i for i, row in enumerate(labels) if row[0] == q]] for q in (0, 2, 4, 6)]
    bank = np.column_stack([low_bank, *[(v@v.conj().T).reshape(-1) for v in projectors]])
    columns, singular, _ = np.linalg.svd(bank, full_matrices=False)
    cutoff = 1e-11*singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    basis = columns[:, :rank]
    low = (basis@(basis.conj().T@raw.reshape(-1))).reshape(64, 64)
    high = raw-low
    diagnostics = dict(span_rank=rank, relative_singular_cutoff=1e-11,
        raw_vs_pair_product_span_max=float(np.max(abs(low_bank.conj().T@raw.reshape(-1)))),
        high_vs_declared_span_max=float(np.max(abs(bank.conj().T@high.reshape(-1)))),
        high_relative_frobenius=float(np.linalg.norm(high)/np.linalg.norm(raw)),
        linear_span_only=True)
    for value in (a, b, raw, low, high):
        value.setflags(write=False)
    return dict(left=a, right=b, raw=raw, low=low, high=high, diagnostics=diagnostics)


def screening_information_readout(rho: np.ndarray, left_sites: Sequence[int]) -> dict:
    """Combined fixed readout bank, retaining raw and high scalar correlations."""
    matrix = _rho(rho)
    split = pseudoscalar_pair_split(_left(left_sites))
    report = screening_multiplicity_information(matrix, left_sites)
    means = {key: float(np.trace(matrix@split[key]).real) for key in ('left', 'right', 'raw', 'low', 'high')}
    report.update(pseudoscalar_product=means['raw'], pseudoscalar_high=means['high'],
        pseudoscalar_low=means['low'], pseudoscalar_connected=means['raw']-means['left']*means['right'],
        pseudoscalar_left=means['left'], pseudoscalar_right=means['right'],
        operator_diagnostics=split['diagnostics'])
    return report


__all__ = ['screening_columns', 'screening_multiplicity_information',
           'pseudoscalar_pair_split', 'screening_information_readout']
