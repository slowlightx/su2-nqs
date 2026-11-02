"""Joint block-spin measurements of existing physical regional density matrices.

q_L, q_R and q_T are doubled angular momenta. These commuting-projector
probabilities do not require an invariant state and are NOT entanglement
eigenvalues, phase labels, or a new variational fusion-path model.
"""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np

from .regional_su2_readout import regional_schur_columns
from .tensor_network_readout import physical_spin_block_readout


@lru_cache(maxsize=8)
def _spin_projectors(n: int) -> tuple[tuple[int, np.ndarray], ...]:
    columns, labels = regional_schur_columns(n)
    result = []
    for q in range(n % 2, n + 1, 2):
        ids = [i for i, label in enumerate(labels) if label[0] == q]
        p = columns[:, ids] @ columns[:, ids].conj().T
        p.setflags(write=False)
        result.append((q, p))
    return tuple(result)


@lru_cache(maxsize=8)
def _joint_projectors(
    n: int, left: tuple[int, ...],
) -> tuple[tuple[tuple[int, int, int], np.ndarray], ...]:
    right = tuple(i for i in range(n) if i not in left)
    order = left + right
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    grouped = bits[:, order] @ (2**np.arange(n - 1, -1, -1))
    result = []
    for ql, pl in _spin_projectors(len(left)):
        for qr, pr in _spin_projectors(len(right)):
            lr = np.kron(pl, pr)[np.ix_(grouped, grouped)]
            for qt, pt in _spin_projectors(n):
                if abs(ql - qr) <= qt <= ql + qr and (ql + qr - qt) % 2 == 0:
                    p = lr @ pt
                    if max(np.linalg.norm(p-p.conj().T), np.linalg.norm(p@p-p)) > 1e-11:
                        raise RuntimeError('joint Casimir projectors do not commute or are not idempotent')
                    p.setflags(write=False)
                    result.append(((ql, qr, qt), p))
    if np.linalg.norm(sum(p for _, p in result)-np.eye(2**n)) > 1e-11:
        raise RuntimeError('joint block-spin projectors are incomplete')
    return tuple(result)


def spin_partition_readout(
    rho: np.ndarray, left_sites: Sequence[int], *, tolerance: float = 1e-9,
) -> dict:
    """Measure p(q_L,q_R,q_T) for a nonempty bipartition of 2..8 spins.

    Sites refer to the supplied rho's physical down/up, MSB-first order.
    The right block is the complement. Return cross-block <S_L.S_R> and
    the classical mutual information of MEASURED block spins, not quantum
    entanglement or a multiplicity spectrum. No twirl or clipping is used.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    block = physical_spin_block_readout(rho, tolerance=tolerance)
    n = rho.shape[0].bit_length()-1
    left = tuple(left_sites)
    if (n < 2 or n > 8 or not left or len(left) >= n or
            any(isinstance(i, bool) or not isinstance(i, Integral) for i in left) or
            len(set(left)) != len(left) or min(left) < 0 or max(left) >= n):
        raise ValueError('left_sites must be a proper nonempty subset of a 2..8-spin region')
    left = tuple(int(i) for i in left)
    probabilities = [(labels, np.trace(rho @ p)) for labels, p in _joint_projectors(n, left)]
    maximum_imaginary = max(abs(p.imag) for _, p in probabilities)
    minimum_probability = min(p.real for _, p in probabilities)
    weights = [(labels, float(p.real)) for labels, p in probabilities]
    total = {str(q): sum(p for (_, _, qt), p in weights if qt == q)
             for q in range(n % 2, n + 1, 2)}
    closure = max(abs(total[q]-block['p_q'][q]) for q in total)
    if max(maximum_imaginary, -minimum_probability, closure, abs(sum(total.values())-1)) > tolerance:
        raise RuntimeError('joint spin probabilities fail positivity, reality or marginal closure')
    lr = {}
    for (ql, qr, _), p in weights:
        lr[ql, qr] = lr.get((ql, qr), 0.) + p
    pl = {ql: sum(p for (a, _), p in lr.items() if a == ql) for ql, _ in lr}
    pr = {qr: sum(p for (_, b), p in lr.items() if b == qr) for _, qr in lr}
    information = sum(p*np.log(p/(pl[ql]*pr[qr])) for (ql, qr), p in lr.items() if p > 1e-14)
    casimir = lambda q: q*(q+2)/4
    cross = sum(p*(casimir(qt)-casimir(ql)-casimir(qr))/2 for (ql, qr, qt), p in weights)
    return dict(
        region_size=n, left_sites=list(left), right_sites=[i for i in range(n) if i not in left],
        weights=[dict(q_left=ql, q_right=qr, q_total=qt, probability=p)
                 for (ql, qr, qt), p in weights],
        p_total=total, singlet_pair_channels={str(ql): p for (ql, qr, qt), p in weights if qt == 0},
        cross_block_spin_dot=float(cross), measured_spin_mutual_information_nats=float(information),
        strict_su2_eligible=block['strict_su2_eligible'],
        diagnostics=dict(total_spin_marginal_error=float(closure),
                         probability_closure_error=float(abs(sum(total.values())-1)),
                         maximum_imaginary_probability=float(maximum_imaginary),
                         minimum_probability=float(minimum_probability)),
        warning='Commuting Casimir probabilities for the actual rho, not entanglement eigenvalues or a QSL criterion.',
    )


def spin_partition_distance(a: dict, b: dict) -> dict:
    """Joint and marginal total-variation distances for the SAME physical cut."""
    if any(a[key] != b[key] for key in ('region_size', 'left_sites', 'right_sites')):
        raise ValueError('spin partitions differ')
    encode = lambda r: {(x['q_left'], x['q_right'], x['q_total']): x['probability'] for x in r['weights']}
    pa, pb = encode(a), encode(b)
    if pa.keys() != pb.keys():
        raise ValueError('joint sector labels differ')
    joint = .5*sum(abs(pa[k]-pb[k]) for k in pa)
    marginal = .5*sum(abs(a['p_total'][q]-b['p_total'][q]) for q in a['p_total'])
    return dict(joint_total_variation=float(joint), total_spin_total_variation=float(marginal),
                marginal_contractivity_error=float(max(0., marginal-joint)))


@lru_cache(maxsize=12)
def _three_plus_three_singlet_columns(left: tuple[int, ...]) -> np.ndarray:
    """Five regional singlets: four (q_L,q_R)=(1,1), then one (3,3)."""
    from ..su2.clebsch_gordan import clebsch_gordan
    u, labels = regional_schur_columns(3)
    lookup = {label: u[:, i] for i, label in enumerate(labels)}
    right = tuple(i for i in range(6) if i not in left)
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    grouped = bits[:, left+right] @ (2**np.arange(5, -1, -1))
    columns = []
    for q, multiplicity in ((1, 2), (3, 1)):
        for a in range(multiplicity):
            for b in range(multiplicity):
                v = sum(clebsch_gordan(q, m, q, -m, 0, 0) *
                        np.kron(lookup[q, m, a], lookup[q, -m, b])
                        for m in range(-q, q+1, 2))
                columns.append(v[grouped])
    result = np.column_stack(columns)
    if np.linalg.norm(result.conj().T@result-np.eye(5)) > 1e-11:
        raise RuntimeError('three-plus-three singlet columns are not orthonormal')
    result.setflags(write=False)
    return result


def singlet_resonance_readout(rho: np.ndarray, left_sites: Sequence[int]) -> tuple[dict, np.ndarray]:
    """Read six-spin singlet resonance in an explicit three-plus-three cut.

    q=2S. The four doublet-pair multiplicities and the single quartet-pair
    singlet are an exact five-dimensional REGIONAL total-q=0 block. It can
    be measured for any physical rho, even without global SU(2). Its spectra
    are conditional operator spectra, NOT the full RDM entanglement spectrum.
    Cross-channel coherence is reported separately from channel occupation.
    Its norm is basis invariant under rotations within each channel, but a
    noisy plug-in norm has positive bias; jackknife SE does not remove it.
    """
    matrix = np.asarray(rho, np.complex128)
    if matrix.shape != (64, 64):
        raise ValueError('six-site physical RDM required')
    partition = spin_partition_readout(matrix, left_sites)
    left = tuple(int(i) for i in left_sites)
    if len(left) != 3:
        raise ValueError('exactly three left sites required')
    columns = _three_plus_three_singlet_columns(left)
    block = columns.conj().T@matrix@columns
    p11, p33 = float(np.trace(block[:4, :4]).real), float(block[4, 4].real)
    closure = max(abs(p11-partition['singlet_pair_channels']['1']),
                  abs(p33-partition['singlet_pair_channels']['3']))
    if closure > 1e-11:
        raise RuntimeError('singlet resonance and joint Casimir weights disagree')
    norm = float(np.linalg.norm(block[:4, 4]))
    conditional = block[:4, :4]/p11 if p11 > 1e-12 else None
    report = dict(left_sites=list(left), right_sites=[i for i in range(6) if i not in left],
        basis_labels=[[1, a, b] for a in range(2) for b in range(2)]+[[3, 0, 0]],
        p11_singlet=p11, p33_singlet=p33, p_total_singlet=p11+p33,
        doublet_conditional_operator_purity=(float(np.trace(conditional@conditional).real)
                                             if conditional is not None else None),
        doublet_conditional_operator_spectrum=(np.linalg.eigvalsh(conditional)[::-1].tolist()
                                               if conditional is not None else None),
        cross_q1_q3_coherence_norm=norm,
        cross_q1_q3_normalized_coherence=(norm/np.sqrt(p11*p33) if p11*p33 > 1e-24 else None),
        probability_closure_error=closure, full_entanglement_spectrum=False,
        noisy_norm_bias_removed=False,
        convention='Condon-Shortley; local balanced Schur labels (q,m,a); left followed by ordered complement; down/up MSB',
        interpretation='Regional singlet-channel operator data, not a QSL or entanglement certificate.')
    return report, block


__all__ = ['spin_partition_readout', 'spin_partition_distance', 'singlet_resonance_readout']
