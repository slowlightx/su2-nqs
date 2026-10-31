"""State-independent separation of two/four-spin and higher spin information.

The metric is the Hilbert--Schmidt inner product on the FULL regional Hilbert
space, not a fit to a state ensemble. A nonzero remainder proves linear
operator nonredundancy, not a phase, an entanglement witness, or nonredundancy
on a restricted low-energy manifold. Bits are down/up, site zero MSB.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
from numbers import Integral
from typing import Sequence

import numpy as np

from .spin_partition_readout import _joint_projectors


def spin_dot_matrix(n: int, i: int, j: int) -> np.ndarray:
    """Return S_i.S_j on 2..6 physical spins in the down/up MSB convention."""
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in (n, i, j)):
        raise ValueError('integer site counts and endpoints required')
    if not 2 <= n <= 6 or not 0 <= i < n or not 0 <= j < n or i == j:
        raise ValueError('distinct endpoints on 2..6 spins required')
    rows = np.arange(2**n)
    bits = (rows[:, None] >> np.arange(n-1, -1, -1)) & 1
    op = np.diag((bits[:, i]-.5)*(bits[:, j]-.5))
    unlike = rows[bits[:, i] != bits[:, j]]
    op[unlike ^ ((1 << (n-1-i)) | (1 << (n-1-j))), unlike] = .5
    return op


@lru_cache(maxsize=5)
def _low_order_span(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    pairs = [spin_dot_matrix(n, i, j) for i, j in combinations(range(n), 2)]
    columns = [np.eye(2**n), *pairs]
    columns += [(a@b+b@a)/2 for a, b in combinations_with_replacement(pairs, 2)]
    bank = np.stack([a.reshape(-1) for a in columns], axis=1)
    u, s, _ = np.linalg.svd(bank, full_matrices=False)
    rank = int(np.count_nonzero(s > 1e-11*s[0]))
    basis = u[:, :rank]
    basis.setflags(write=False); bank.setflags(write=False)
    return basis, bank, rank


@dataclass(frozen=True)
class SpinOperatorSplit:
    """Exact full = low + high; q labels are DOUBLED total spins."""
    labels: tuple[int, int, int]
    full: np.ndarray
    low: np.ndarray
    high: np.ndarray
    diagnostics: dict

    def measure(self, rho: np.ndarray) -> dict[str, float]:
        """Measure actual rho; high is a signed observable, not a probability."""
        rho = np.asarray(rho, complex)
        if rho.shape != self.full.shape or not np.isfinite(rho).all():
            raise ValueError('finite regional matrix with matching dimension required')
        values = [np.trace(rho@op) for op in (self.full, self.low, self.high)]
        if max(abs(v.imag) for v in values) > 1e-10:
            raise ValueError('observable expectations must be real')
        return dict(zip(('full', 'low', 'high'), (float(v.real) for v in values), strict=True))


def joint_spin_operator_split(n: int, left_sites: Sequence[int],
                             labels: Sequence[int]) -> SpinOperatorSplit:
    """Split P(q_L,q_R,q_total) against I, ALL pairs and symmetrized pair products.

    These time-even scalar products include all disjoint four-spin bond
    correlations, not just Hamiltonian bonds. Shared-endpoint products reduce
    to lower order for spin half. No state quality or expectation enters the
    projection. All matrices are immutable complex128/float64 references.
    """
    left, q = tuple(left_sites), tuple(labels)
    if (isinstance(n, bool) or not isinstance(n, Integral) or not 2 <= n <= 6 or
            not left or len(left) >= n or len(set(left)) != len(left) or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < n for i in left) or
            len(q) != 3 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in q)):
        raise ValueError('valid regional partition and three doubled nonnegative integer q labels required')
    projectors = dict(_joint_projectors(int(n), tuple(map(int, left))))
    if q not in projectors:
        raise ValueError('joint sector is not reachable')
    full = projectors[q]
    basis, columns, rank = _low_order_span(int(n))
    low = (basis@(basis.conj().T@full.reshape(-1))).reshape(full.shape)
    high = full-low
    diagnostics = dict(low_order_columns=columns.shape[1], low_order_rank=rank,
        orthogonality_max=float(np.max(abs(columns.conj().T@high.reshape(-1)))),
        decomposition_error=float(np.linalg.norm(full-low-high)),
        high_hermiticity_error=float(np.linalg.norm(high-high.conj().T)),
        relative_operator_residual=float(np.linalg.norm(high)/np.linalg.norm(full)),
        metric='unweighted full-region Hilbert-Schmidt, cutoff 1e-11 times largest singular value')
    for a in (low, high): a.setflags(write=False)
    return SpinOperatorSplit(tuple(map(int, q)), full, low, high, diagnostics)


def three_bond_readout(rho: np.ndarray, left_sites: Sequence[int]) -> dict:
    """Read six-spin bond products AND connected third cumulants.

    The fifteen perfect matchings contain disjoint bonds, so all three bond
    operators in a matching commute. ``X`` has one internal bond in each
    three-spin block and one bridge. ``Y`` has three bridges. Their raw sums
    obey P_perp(1,1,0)=-4X/9 and P_perp(3,3,0)=X/9-Y/3, using DOUBLED spins.
    Connected sums instead subtract lower-order expectation products and are
    state-dependent nonlinear diagnostics. They are not projector weights.
    No SU(2) twirl or symmetry assumption is applied to the input RDM.
    """
    left = tuple(left_sites)
    matrix = np.asarray(rho, dtype=np.complex128)
    if (len(left) != 3 or len(set(left)) != 3 or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                or not 0 <= i < 6 for i in left)):
        raise ValueError('three distinct integer sites in a six-spin region required')
    if (matrix.shape != (64, 64) or not np.isfinite(matrix).all() or
            np.linalg.norm(matrix-matrix.conj().T) > 1e-10 or
            abs(np.trace(matrix)-1) > 1e-10 or
            np.linalg.eigvalsh(matrix).min() < -1e-10):
        raise ValueError('normalized positive Hermitian six-spin RDM required')

    def matchings(sites: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
        if not sites:
            return [()]
        first = sites[0]
        return [((first, other),)+tail for other in sites[1:]
                for tail in matchings(tuple(i for i in sites[1:] if i != other))]

    dots = {bond: spin_dot_matrix(6, *bond) for bond in combinations(range(6), 2)}
    groups = {name: dict(raw=0., connected=0., count=0)
              for name in ('X', 'Y')}
    rows = []
    for cover in matchings(tuple(range(6))):
        ops = [dots[bond] for bond in cover]
        means = np.array([np.trace(matrix@op).real for op in ops])
        second = np.array([np.trace(matrix@ops[i]@ops[j]).real
                           for i, j in ((0, 1), (0, 2), (1, 2))])
        raw = float(np.trace(matrix@ops[0]@ops[1]@ops[2]).real)
        cumulant = float(raw-second[0]*means[2]-second[1]*means[1]
                         -second[2]*means[0]+2*np.prod(means))
        bridges = sum((a in left) != (b in left) for a, b in cover)
        name = 'X' if bridges == 1 else 'Y'
        groups[name]['raw'] += raw
        groups[name]['connected'] += cumulant
        groups[name]['count'] += 1
        rows.append(dict(bonds=cover, orbit=name, first=means.tolist(),
                         second=second.tolist(), raw=raw, connected=cumulant))
    return dict(orbits=groups, matchings=rows,
                high_q1=-4*groups['X']['raw']/9,
                high_q3=groups['X']['raw']/9-groups['Y']['raw']/3,
                raw_and_connected_are_distinct=True)


__all__ = ['spin_dot_matrix', 'SpinOperatorSplit', 'joint_spin_operator_split',
           'three_bond_readout']
