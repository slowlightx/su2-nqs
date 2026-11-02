"""Node-safe ordinary chirality correlations, independent of any ansatz.

Use rows of the composed operator, not products of single-action ratios:
the latter can omit intermediate computational-basis nodes of the state.
All sites follow bit 0 down, site zero MSB. No SU(2) assumption is made.
"""
from __future__ import annotations

import numpy as np

from .replica_spin_response import vector_chirality_z_connections

PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
MIRROR_BASIS = np.array([[1., 0., 1.], [0., np.sqrt(2.), 0.],
                         [1., 0., -1.]]) / np.sqrt(2.)


def chirality_product_connections(
    states: np.ndarray, left: tuple[tuple[int, int], ...],
    right: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of {K_left,K_right}/2, combining identical target configurations.

    Intermediate rows can have zero wavefunction amplitude. They are NEVER
    removed or used as denominators. Returns target, source and O[row,col].
    """
    x = np.asarray(states)
    all_targets, all_sources, all_weights = [], [], []
    orders = ((left, right, 1.),) if left == right else (
        (left, right, .5), (right, left, .5))
    for first, second, factor in orders:
        middle, source, a = vector_chirality_z_connections(x, first)
        target, mid_source, b = vector_chirality_z_connections(middle, second)
        all_targets.append(target)
        all_sources.append(source[mid_source])
        all_weights.append(factor * a[mid_source] * b)
    targets, sources, weights = map(np.concatenate,
                                    (all_targets, all_sources, all_weights))
    if not len(sources):
        return targets, sources, weights
    # A physical target may be reached by several two-bond sequences.
    keys, inverse = np.unique(np.column_stack((sources, targets)), axis=0,
                              return_inverse=True)
    summed = np.zeros(len(keys), complex)
    np.add.at(summed, inverse, weights)
    keep = summed != 0
    return keys[keep, 1:].astype(x.dtype), keys[keep, 0], summed[keep]


def symmetric_matrix(entries: np.ndarray) -> np.ndarray:
    """Map three diagonal then (01,02,12) entries to a symmetric matrix."""
    v = np.asarray(entries)
    if v.shape[-1:] != (6,):
        raise ValueError('six independent symmetric entries required')
    matrix = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    for k, (a, b) in enumerate(PAIRS):
        matrix[..., a, b] = matrix[..., b, a] = v[..., k]
    return matrix


def normalized_coordinates(matrix: np.ndarray) -> np.ndarray:
    """Interior share, boundary-odd share, and 2*M_even,interior / trace.

    This normalization removes total response strength, not cutoff or finite
    size effects. A nonpositive estimated trace is unresolved, never clipped.
    """
    m = np.asarray(matrix, float)
    trace = np.trace(m, axis1=-2, axis2=-1)
    if np.any(trace <= 0):
        raise ValueError('positive trace required for normalized coordinates')
    rotated = np.einsum('ia,...ij,jb->...ab', MIRROR_BASIS, m, MIRROR_BASIS)
    return np.stack((rotated[..., 1, 1], rotated[..., 2, 2],
                     2 * rotated[..., 0, 1]), axis=-1) / trace[..., None]


def cluster_jackknife(records: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and leave-one-independent-pair-out means of equal clusters."""
    x, labels = np.asarray(records), np.asarray(labels)
    if x.ndim < 1 or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('finite records and matched cluster labels required')
    groups, counts = np.unique(labels, return_counts=True)
    if len(groups) < 4 or not np.all(counts == counts[0]):
        raise ValueError('at least four equally weighted independent clusters required')
    grouped = np.array([x[labels == group].mean(axis=0) for group in groups])
    return grouped.mean(axis=0), (grouped.sum(axis=0) - grouped) / (len(groups) - 1)


def jackknife_error(leave: np.ndarray) -> np.ndarray:
    """Real cluster-jackknife SE; retain complex means separately."""
    a = np.asarray(leave, float)
    return np.sqrt((len(a)-1)/len(a)*np.sum((a-a.mean(axis=0))**2, axis=0))


def compare_chirality_matrices(
    ordinary_products: np.ndarray, replica_numerators: np.ndarray,
    swap: np.ndarray, labels: np.ndarray,
) -> dict:
    """Compare C and M using exactly the same chain-pair jackknife deletions.

    Ordinary products use PAIRS order. Replica probes use the three rows,
    followed by the three row sums squared. No SU2 or mirror/PSD repair.
    """
    c, lc = cluster_jackknife(ordinary_products, labels)
    n, ln = cluster_jackknife(replica_numerators, labels)
    w, lw = cluster_jackknife(swap, labels)
    cm, lcm = symmetric_matrix(c.real), symmetric_matrix(lc.real)
    if w.real <= 0 or np.any(lw.real <= 0):
        raise ValueError('replica purity unresolved')
    def replica_matrix(values: np.ndarray) -> np.ndarray:
        entries = values.copy()
        entries[..., 3:] = (values[..., 3:]
            - values[..., [0, 0, 1]] - values[..., [1, 2, 2]]) / 2
        return symmetric_matrix(entries)
    mm = replica_matrix(n.real / w.real)
    lmm = replica_matrix(ln.real / lw.real[:, None])
    cc, lcc = normalized_coordinates(cm), normalized_coordinates(lcm)
    mc, lmc = normalized_coordinates(mm), normalized_coordinates(lmm)
    return dict(ordinary_matrix=cm.tolist(), ordinary_matrix_SE=jackknife_error(lcm).tolist(),
        ordinary_imaginary_entries=c.imag.tolist(),
        ordinary_imaginary_entries_SE=jackknife_error(lc.imag).tolist(),
        replica_matrix=mm.tolist(), replica_matrix_SE=jackknife_error(lmm).tolist(),
        coordinate_order=['interior_share', 'boundary_odd_share', 'even_interior_coherence'],
        ordinary_coordinates=cc.tolist(), ordinary_coordinates_SE=jackknife_error(lcc).tolist(),
        replica_coordinates=mc.tolist(), replica_coordinates_SE=jackknife_error(lmc).tolist(),
        replica_minus_ordinary_coordinates=(mc-cc).tolist(),
        replica_minus_ordinary_coordinates_SE=jackknife_error(lmc-lcc).tolist(),
        independent_chain_pairs=len(lc),
        ordinary_definition='Re <{kappa_i,kappa_j}/2>; exact spin-flip-even one-point means vanish',
        replica_definition='Re Tr(rho kappa_i rho kappa_j)/Tr(rho^2)',
        PSD_clipping=False, mirror_repair=False, strict_SU2_assumed=False)
