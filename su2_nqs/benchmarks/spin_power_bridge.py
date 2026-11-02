"""Bounded physical-amplitude bridge for low-degree Heisenberg carriers.

This is host-side preparation, NOT an accelerator-native training H kernel.
It reuses the scalar-word executor; no ED, carrier division, magnetic-label
learning or coefficient clipping. Site zero is the most significant bit.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from time import perf_counter

import numpy as np

from ..operators.spin_config import SpinBond
from .operator_word_jax import ScalarWordActionPlan


def standardized_krylov_polynomial(
    coefficients: np.ndarray, means: np.ndarray, scales: np.ndarray,
    hamiltonian_weights: np.ndarray, *, tolerance: float = 2e-10,
) -> tuple[np.ndarray, dict]:
    """Algebraically convert I,Z_i,Z_i Z_j coefficients to I,H,H².

Z_i=(O_i-mean_i I)/scale_i. Check each ORDERED quadratic coefficient:
noncommuting products are not symmetrized. Reject a non-Krylov state instead
of fitting/projecting it onto a polynomial. H is the total, not per-site H.
"""
    c, m, s, w = (np.asarray(x, dtype=float) for x in
                  (coefficients, means, scales, hamiltonian_weights))
    n = len(m) if m.ndim == 1 else 0
    if (n < 1 or s.shape != (n,) or w.shape != (n,)
            or c.shape != (1+n+n*n,) or not all(np.isfinite(x).all() for x in (c,m,s,w))
            or np.any(s <= 0) or not np.any(w) or not 0 < tolerance < 1):
        raise ValueError('finite standardized coefficients, positive scales and nonzero H required')
    quadratic = c[1+n:].reshape(n,n)/np.outer(s,s)
    linear = c[1:1+n]/s - quadratic@m - quadratic.T@m
    constant = c[0] - np.dot(c[1:1+n]/s,m) + m@quadratic@m
    pivot = int(np.argmax(abs(w)))
    polynomial = np.array([constant, linear[pivot]/w[pivot],
                           quadratic[pivot,pivot]/w[pivot]**2])
    defect = max(float(np.max(abs(linear-polynomial[1]*w))),
                 float(np.max(abs(quadratic-polynomial[2]*np.outer(w,w)))))
    if defect > tolerance*max(1.,float(np.max(abs(c)))):
        raise ValueError('state is not exactly a polynomial of the specified total H')
    return polynomial, dict(ordered_operator_coefficient_defect=defect,
                            algebraic_conversion_not_variational_refit=True)


def build_bounded_spin_power_plan(
    configurations: np.ndarray, bonds: Sequence[SpinBond], *, maximum_power: int,
    maximum_expansion_entries: int = 2_000_000,
    maximum_terminal_queries: int = 524_288,
) -> tuple[ScalarWordActionPlan, dict]:
    """Build C,...,H^p C, aggregating repeated bonds/paths before queries.

For 1<=N<=63, integer bit keys permit vectorized local exchanges. Explicit
allocation bounds fail BEFORE each expansion. Dropping exact zero sums is
allowed; arbitrarily small nonzero coefficients and wavefunction nodes stay.
Connectivity is H[new,old]; real symmetric H permits terminal-row action.
"""
    start = perf_counter()
    x = np.asarray(configurations)
    if (x.ndim != 2 or not len(x) or not 1 <= x.shape[1] <= 63
            or not np.issubdtype(x.dtype,np.integer) or np.any((x!=0)&(x!=1))):
        raise ValueError('nonempty integer bit matrix with 1..63 sites required')
    for value in (maximum_power, maximum_expansion_entries, maximum_terminal_queries):
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise ValueError('integer powers and resource bounds required')
    if not 0 <= maximum_power <= 3 or min(maximum_expansion_entries,maximum_terminal_queries) < 1:
        raise ValueError('bounded power 0..3 and positive resource bounds required')
    if not bonds:
        raise ValueError('nonempty real bonds required')
    n = x.shape[1]
    combined = {}
    for bond in bonds:
        if (len(bond) not in (2,3) or any(isinstance(v,bool) or not isinstance(v,Integral) for v in bond[:2])
                or not 0 <= bond[0] < n or not 0 <= bond[1] < n or bond[0] == bond[1]):
            raise ValueError('distinct integer bond endpoints in range required')
        coupling = float(bond[2]) if len(bond)==3 else 1.
        if not np.isfinite(coupling):
            raise ValueError('finite real coupling required')
        edge = tuple(sorted(bond[:2]))
        combined[edge] = combined.get(edge,0.) + coupling
    checked = [(i,j,v) for (i,j),v in sorted(combined.items()) if v != 0]
    shifts = np.arange(n-1,-1,-1,dtype=np.uint64)
    centers = np.sum(x.astype(np.uint64) << shifts,axis=1,dtype=np.uint64)
    keys, weights, output_indices = [], [], []
    requests = 0
    peak_expansion = 0
    per_center = []
    for row, center in enumerate(centers):
        frontier, coefficients = np.array([center],np.uint64), np.ones(1)
        counts = []
        for power in range(maximum_power+1):
            counts.append(len(frontier)); requests += len(frontier)
            if requests > maximum_expansion_entries:
                raise MemoryError('accumulated sparse-entry bound exceeded')
            keys.append(frontier); weights.append(coefficients)
            output_indices.append(np.full(len(frontier),row*(maximum_power+1)+power,np.int32))
            if power == maximum_power:
                break
            allocation = len(frontier)*(len(checked)+1)
            if allocation > maximum_expansion_entries:
                raise MemoryError(f'power{power+1} expansion {allocation} exceeds {maximum_expansion_entries}')
            peak_expansion = max(peak_expansion,allocation)
            diagonal = np.zeros(len(frontier))
            following, values = [], []
            for i,j,coupling in checked:
                mi, mj = np.uint64(1) << np.uint64(n-1-i), np.uint64(1) << np.uint64(n-1-j)
                opposite = ((frontier&mi)!=0) != ((frontier&mj)!=0)
                diagonal += coupling*(.25-.5*opposite)
                following.append(frontier[opposite] ^ (mi|mj))
                values.append(.5*coupling*coefficients[opposite])
            following.append(frontier); values.append(coefficients*diagonal)
            targets = np.concatenate(following); entries = np.concatenate(values)
            frontier, inverse = np.unique(targets,return_inverse=True)
            coefficients = np.bincount(inverse,weights=entries,minlength=len(frontier))
            nonzero = coefficients != 0
            frontier, coefficients = frontier[nonzero], coefficients[nonzero]
            if len(frontier) > maximum_terminal_queries:
                raise MemoryError('per-center terminal-query bound exceeded')
        per_center.append(counts)
    all_keys = np.concatenate(keys)
    union, inverse = np.unique(all_keys,return_inverse=True)
    if len(union) > maximum_terminal_queries:
        raise MemoryError('union terminal-query bound exceeded')
    terminals = ((union[:,None] >> shifts)&1).astype(np.int8)
    plan = ScalarWordActionPlan(terminals, inverse.astype(np.int32),
        np.concatenate(output_indices), np.concatenate(weights), len(x),
        maximum_power+1, maximum_power, requests)
    work = dict(centers=len(x), powers=list(range(maximum_power+1)),
        terminals_per_center_and_power=per_center, unique_terminal_queries=len(union),
        sparse_entries=len(all_keys), peak_expansion_entry_bound=peak_expansion,
        combined_bonds=len(checked), host_plan_seconds=perf_counter()-start,
        full_H_applications_per_center=maximum_power, small_coefficient_clipping=False,
        host_plan_not_accelerator_native=True, ED_performed=False)
    return plan, work


__all__ = ['standardized_krylov_polynomial','build_bounded_spin_power_plan']
