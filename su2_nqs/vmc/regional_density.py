"""Coherence-aware regional RDMs from a fixed positive proposal, without training.

Rows are complete physical bit configurations, site zero MSB. Enumeration
preserves total Sz, NOT individual regional spins. No Born-ratio division,
positivity clipping, SU(2) twirl or spectral interpretation is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegionalQueryPlan:
    """All compatible regional inputs in a deterministic physical bit order."""
    configurations: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    center_local_indices: np.ndarray
    sites: tuple[int, ...]


def regional_query_plan(states: np.ndarray, sites: Sequence[int]) -> RegionalQueryPlan:
    """Deduplicate coherent regional queries for a fixed-total-Sz sample set."""
    x = np.asarray(states); region = tuple(sites)
    if (x.ndim != 2 or not len(x) or x.dtype.kind not in 'biu' or np.any((x != 0)&(x != 1)) or
            not region or len(region) > 8 or len(set(region)) != len(region) or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < x.shape[1] for i in region)):
        raise ValueError('nonempty physical bit matrix and 1..8 distinct region sites required')
    if np.any(x.sum(axis=1) != x[0].sum()):
        raise ValueError('sample set must have fixed total up-spin count')
    m = len(region); labels = np.arange(2**m)
    local_bits = ((labels[:, None] >> np.arange(m-1, -1, -1)) & 1).astype(np.int8)
    valid = x[:, region].sum(axis=1)[:, None] == local_bits.sum(axis=1)[None, :]
    requests = np.broadcast_to(x[:, None, :], (len(x), 2**m, x.shape[1])).copy()
    requests[:, :, region] = local_bits
    unique, inverse = np.unique(requests[valid], axis=0, return_inverse=True)
    indices = np.zeros(valid.shape, np.int64); indices[valid] = inverse
    local = x[:, region]@(1 << np.arange(m-1, -1, -1))
    for array in (unique, indices, valid, local): array.setflags(write=False)
    return RegionalQueryPlan(unique, indices, valid, local, tuple(map(int, region)))


def conditional_regional_matrices(plan: RegionalQueryPlan, amplitudes: np.ndarray,
                                 proposal_density: np.ndarray) -> np.ndarray:
    """Rao--Blackwellize over local spins: psi_A psi_A† / sum_A Q(A,env).

    Samples must be from the SAME (possibly unnormalized) positive global Q.
    Its entire compatible regional conditional, not just Q at the center,
    must be supplied. Averaging and trace normalization is self-normalized
    importance estimation with O(1/sample-count) ratio bias. Every returned
    matrix is PSD, but is NOT silently made SU(2) invariant.
    """
    p, q = np.asarray(amplitudes, complex), np.asarray(proposal_density, float)
    if (p.shape != (len(plan.configurations),) or q.shape != p.shape or
            not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(q <= 0)):
        raise ValueError('one finite amplitude and strictly positive Q per unique input required')
    v = np.where(plan.valid, p[plan.indices], 0)
    marginal = np.sum(np.where(plan.valid, q[plan.indices], 0), axis=1)
    return v[:, :, None]*v[:, None, :].conj()/marginal[:, None, None]


def regional_density_summary(matrices: np.ndarray, groups: np.ndarray) -> dict:
    """Normalize raw RDM and delete-WHOLE-chain replicates, preserving coherences."""
    a, g = np.asarray(matrices, complex), np.asarray(groups)
    if (a.ndim != 3 or a.shape[1] != a.shape[2] or g.shape != (len(a),) or
            not np.isfinite(a).all() or len(np.unique(g)) < 2):
        raise ValueError('finite square per-record matrices and at least two chains required')
    def normalized(rows):
        rho = rows.sum(axis=0); norm = np.trace(rho)
        if abs(norm.imag) > 1e-10*max(1., abs(norm)) or norm.real <= 0:
            raise ValueError('nonpositive/nonreal estimated state norm')
        return rho/norm.real
    rho = normalized(a)
    jack = np.stack([normalized(a[g != c]) for c in np.unique(g)])
    w = np.trace(a, axis1=1, axis2=2).real
    return dict(rho=rho, jackknife_rhos=jack, independent_chains=len(jack),
        records=len(a), weight_ess_fraction=float(w.sum()**2/(np.sum(w*w)*len(w))),
        maximum_weight_fraction=float(w.max()/w.sum()),
        hermiticity_error=float(np.linalg.norm(rho-rho.conj().T)),
        minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min()))


def regional_observable_summary(summary: dict, operator: np.ndarray) -> dict:
    """Signed observable and whole-chain jackknife SE, without physical relabeling."""
    op = np.asarray(operator, complex)
    if op.shape != summary['rho'].shape or np.linalg.norm(op-op.conj().T) > 1e-10:
        raise ValueError('Hermitian regional observable with matching shape required')
    value = np.trace(summary['rho']@op)
    jack = np.einsum('cab,ba->c', summary['jackknife_rhos'], op)
    se = np.sqrt((len(jack)-1)*np.mean(abs(jack-jack.mean())**2))
    return dict(value=float(value.real), imaginary=float(value.imag),
                chain_jackknife_se=float(se))


__all__ = ['RegionalQueryPlan', 'regional_query_plan', 'conditional_regional_matrices',
           'regional_density_summary', 'regional_observable_summary']


@dataclass(frozen=True)
class DisjointRegionalPlan:
    """Born quadrature conditioned on each disjoint block's doubled Sz.

    This extra conditioning is exact only for operators conserving each
    block's Sz. It avoids enumerating irrelevant magnetization transfers.
    Site order is the supplied order, with the first site the MSB.
    """
    configurations: np.ndarray
    query_indices: np.ndarray
    record_indices: np.ndarray
    local_indices: np.ndarray
    regions: tuple[tuple[int, ...], ...]
    records: int


def disjoint_regional_query_plan(states: np.ndarray,
                                regions: Sequence[Sequence[int]]) -> DisjointRegionalPlan:
    """Enumerate exact conditional amplitudes for one or two 1..6-spin blocks."""
    from itertools import product
    x = np.asarray(states)
    cuts = tuple(tuple(r) for r in regions)
    flat = tuple(i for r in cuts for i in r)
    if (x.ndim != 2 or not len(x) or x.dtype.kind not in 'biu' or
            not np.isin(x, (0, 1)).all() or len(cuts) not in (1, 2) or
            any(not 1 <= len(r) <= 6 for r in cuts) or len(set(flat)) != len(flat) or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or
                not 0 <= i < x.shape[1] for i in flat) or
            np.any(x.sum(1) != x[0].sum())):
        raise ValueError('fixed-Sz bits and one/two disjoint ordered 1..6-spin regions required')
    labels = [np.arange(1 << len(r)) for r in cuts]
    bits = [((v[:, None] >> np.arange(len(r)-1, -1, -1)) & 1).astype(np.int8)
            for r, v in zip(cuts, labels)]
    requests, records, local = [], [], []
    for k, row in enumerate(x):
        allowed = [v[b.sum(1) == row[list(r)].sum()] for v, b, r in zip(labels, bits, cuts)]
        for ids in product(*allowed):
            new = row.copy()
            for r, b, i in zip(cuts, bits, ids):
                new[list(r)] = b[i]
            requests.append(new); records.append(k); local.append(ids)
    unique, inverse = np.unique(np.asarray(requests), axis=0, return_inverse=True)
    return DisjointRegionalPlan(unique, inverse, np.asarray(records), np.asarray(local), cuts, len(x))


def disjoint_born_operator_moments(plan: DisjointRegionalPlan, log_amplitudes: np.ndarray,
                                  operators: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact conditional first/product moments of frozen Hermitian operators.

    Return one real bounded estimate per original Born sample. Products act
    on disjoint blocks and are genuine joint observables, not products of
    separately estimated means. No RDM positivity or SU2 repair is applied.
    Identical block sizes are required for the common operator dictionary.
    """
    sizes = [len(r) for r in plan.regions]
    if len(set(sizes)) != 1 or not operators:
        raise ValueError('common operators require equal nonempty block sizes')
    d = 1 << sizes[0]
    m = np.array([i.bit_count() for i in range(d)])
    for op in operators.values():
        op = np.asarray(op)
        if (op.shape != (d, d) or not np.isfinite(op).all() or
                np.linalg.norm(op-op.conj().T) > 1e-10 or
                np.linalg.norm((m[:, None]-m[None, :])*op) > 1e-10):
            raise ValueError('Hermitian, block-Sz-conserving operators required')
    logs = np.asarray(log_amplitudes, complex)
    if logs.shape != (len(plan.configurations),) or np.isnan(logs).any() or np.isposinf(logs.real).any():
        raise ValueError('finite phases and finite or negative-infinite log magnitudes required')
    pulled = logs[plan.query_indices]
    maxima = np.full(plan.records, -np.inf)
    np.maximum.at(maxima, plan.record_indices, pulled.real)
    if not np.isfinite(maxima).all():
        raise ValueError('Born conditional has zero support')
    amplitudes = np.exp(pulled-maxima[plan.record_indices])
    shape = (plan.records,)+(d,)*len(sizes)
    vectors = np.zeros(shape, complex)
    vectors[(plan.record_indices, *plan.local_indices.T)] = amplitudes
    norm = np.sum(abs(vectors.reshape(plan.records, -1))**2, axis=1)
    output = {}
    for name, operator in operators.items():
        op = np.asarray(operator)
        if len(sizes) == 1:
            output[name+'__A'] = np.einsum('ni,ij,nj->n', vectors.conj(), op, vectors).real/norm
        else:
            left = np.einsum('ij,njk->nik', op, vectors, optimize=True)
            right = np.einsum('njk,ik->nji', vectors, op, optimize=True)
            both = np.einsum('njk,ik->nji', left, op, optimize=True)
            for suffix, applied in [('A', left), ('B', right), ('AB', both)]:
                output[name+'__'+suffix] = np.einsum('nij,nij->n', vectors.conj(), applied).real/norm
    return output
