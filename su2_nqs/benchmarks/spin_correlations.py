"""Node-safe sparse physical spin/dimer correlations, without dense Hilbert spaces.

Return O[row,column] connectivity. Bit 0 is down, site zero is MSB. These
physical operators require neither SU(2) covariance nor a particular ansatz.
"""
from __future__ import annotations

from numbers import Integral
import numpy as np


def spin_dot_connections(
    states: np.ndarray, bond: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of S_i.S_j, including the diagonal and opposite-spin exchange."""
    x = np.asarray(states)
    if (x.ndim != 2 or not np.isin(x, (0, 1)).all() or len(bond) != 2
            or bond[0] == bond[1] or any(isinstance(i, bool) or not isinstance(i, Integral)
                or not 0 <= i < x.shape[1] for i in bond)):
        raise ValueError('binary physical rows and two distinct legal sites required')
    i, j = bond
    ids = np.flatnonzero(x[:, i] != x[:, j])
    flipped = x[ids].copy()
    flipped[:, i], flipped[:, j] = x[ids, j], x[ids, i]
    return (np.concatenate((x, flipped)), np.concatenate((np.arange(len(x)), ids)),
            np.concatenate(((x[:, i]-.5)*(x[:, j]-.5), np.full(len(ids), .5))))


def dimer_product_connections(
    states: np.ndarray, left: tuple[int, int], right: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of {S_i.S_j,S_k.S_l}/2, retaining intermediate wavefunction nodes.

    The symmetric definition also applies when bonds overlap. Duplicate paths
    are coalesced, never divided by an intermediate wavefunction amplitude.
    """
    x = np.asarray(states)
    targets, sources, weights = [], [], []
    orders = ((left, right, 1.),) if left == right else (
        (left, right, .5), (right, left, .5))
    for first, second, factor in orders:
        middle, old, a = spin_dot_connections(x, first)
        target, source, b = spin_dot_connections(middle, second)
        targets.append(target); sources.append(old[source]); weights.append(factor*a[source]*b)
    t, s, c = map(np.concatenate, (targets, sources, weights))
    if not len(s):
        return t, s, c
    keys, inverse = np.unique(np.column_stack((s, t)), axis=0, return_inverse=True)
    summed = np.zeros(len(keys), complex)
    np.add.at(summed, inverse, c)
    keep = summed != 0
    return keys[keep, 1:].astype(x.dtype), keys[keep, 0], summed[keep]


def chirality_dot_connections(
    states: np.ndarray, left: tuple[int, int], right: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of (S_i cross S_j).(S_k cross S_l) for FOUR distinct sites.

    The exact identity is (S_i.S_k)(S_j.S_l)-(S_i.S_l)(S_j.S_k).
    It requires disjoint bonds: overlapping quantum-spin products have extra
    commutator terms and are explicitly rejected. No isotropy/factor3 is used.
    Orientation reversal on either bond changes the sign. Intermediate
    wavefunction nodes are retained by the composed dimer connectors.
    """
    left, right = tuple(left), tuple(right)
    if len(left) != 2 or len(right) != 2 or len(set(left+right)) != 4:
        raise ValueError('chirality dot identity requires two disjoint oriented bonds')
    i, j = left; k, l = right
    a = dimer_product_connections(states, (i, k), (j, l))
    b = dimer_product_connections(states, (i, l), (j, k))
    targets = np.concatenate((a[0], b[0])); sources = np.concatenate((a[1], b[1]))
    coefficients = np.concatenate((a[2], -b[2]))
    if not len(sources):
        return targets, sources, coefficients
    keys, inverse = np.unique(np.column_stack((sources, targets)), axis=0, return_inverse=True)
    summed = np.zeros(len(keys), complex)
    np.add.at(summed, inverse, coefficients)
    keep = summed != 0
    return keys[keep, 1:].astype(np.asarray(states).dtype), keys[keep, 0], summed[keep]
