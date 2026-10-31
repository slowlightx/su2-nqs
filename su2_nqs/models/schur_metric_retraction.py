"""Whole-irrep two-site truncation in the UNPROJECTED physical tree norm.

For each doubled shared spin q, nonorthogonal subtree/environment bases give
||W||^2 = ||sqrt(R) W sqrt(C).T||_F^2. Truncate this whitened matrix, not W.
Magnetic Schmidt eigenvalues repeat q+1 times; only multiplicities are split.
This exact tree metric is NOT the metric after spatial projection or after
adding a carrier. Those physical amplitudes must be checked separately.

The coordinate-SVD module is immutable historical experiment provenance.
This companion reuses its layout/joining conventions without rewriting it.
"""
from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import Leaf
from ..su2.local_schur import cell_schur_spec
from .cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from .finite_schur_decoder import FiniteSchurDecoderConfig
from .schur_two_site import TwoSiteLayout


def schur_norm_environments(bank: dict, config: FiniteSchurDecoderConfig) -> tuple[dict, dict]:
    """Return exact fixed-m subtree Grams and summed-m norm environments.

    Dictionary q keys are doubled integers. E has no extra q+1 factor:
    E/(q+1) is the environment Gram at one fixed magnetic index. The magnetic
    sum cancels this factor when contracting with the subtree Gram.
    No computational-space enumeration, Hamiltonian, or symbolic runtime.
    """
    tree, chi = config.tree.cell_tree, config.tree.feature_dim
    layouts = cell_multiplicity_layouts(config.tree)
    if len(bank['nodes']) != len(layouts):
        raise ValueError('bank and tree node counts differ')
    for a, row in zip(bank['nodes'], layouts, strict=True):
        if (np.shape(a) != (len(row.keys), chi, max(row.left_widths), max(row.right_widths))
                or not np.isfinite(np.asarray(a)).all()):
            raise ValueError('finite reduced bank with configured padded shapes required')
    local = cell_schur_spec(config.tree.cell_kind)
    grams = {leaf: {q: np.eye(local.multiplicity(q), dtype=complex)
                    for q in local.q_values} for leaf in tree.leaves}
    for i, (node, row) in enumerate(zip(tree.internal_nodes, layouts, strict=True)):
        out = {}
        for c, (ql, qr, qt) in enumerate(row.keys):
            na, nb = row.left_widths[c], row.right_widths[c]
            t = np.asarray(bank['nodes'][i][c, :, :na, :nb])
            value = np.einsum('kab,lcd,ac,bd->kl', t.conj(), t,
                              grams[node.left][ql], grams[node.right][qr])
            out[qt] = out.get(qt, np.zeros((chi, chi), complex)) + value
        grams[node] = out
    env = {tree.root: {0: np.ones((chi, chi), complex)/chi}}
    for i in range(len(tree.internal_nodes)-1, -1, -1):
        node, row = tree.internal_nodes[i], layouts[i]
        for left, child, sibling in ((True, node.left, node.right), (False, node.right, node.left)):
            if isinstance(child, Leaf):
                continue
            out = {}
            for c, (ql, qr, qt) in enumerate(row.keys):
                t = np.asarray(bank['nodes'][i][c, :, :row.left_widths[c], :row.right_widths[c]])
                if left:
                    q = ql
                    value = np.einsum('kl,kab,lcd,bd->ac', env[node][qt], t.conj(), t, grams[sibling][qr])
                else:
                    q = qr
                    value = np.einsum('kl,kab,lcd,ac->bd', env[node][qt], t.conj(), t, grams[sibling][ql])
                out[q] = out.get(q, np.zeros_like(value)) + value
            env[child] = out
    return grams, env


def _positive_factors(gram: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray, dict]:
    gram = np.asarray(gram, dtype=complex)
    values, vectors = np.linalg.eigh((gram+gram.conj().T)/2)
    maximum = max(float(values[-1]), 1e-300)
    if float(values[0]) < -2e-11*maximum:
        raise FloatingPointError('physical Gram is not positive semidefinite')
    positive = np.maximum(values, 0.)
    active = positive > cutoff*maximum
    root = (vectors*np.sqrt(positive))@vectors.conj().T
    inverse_values = np.zeros_like(values)
    inverse_values[active] = 1/np.sqrt(positive[active])
    inverse = (vectors*inverse_values)@vectors.conj().T
    return root, inverse, dict(resolved_rank=int(active.sum()), dimension=len(values),
                               eigenvalues=values.tolist(), relative_cutoff=cutoff)


def _group_indices(layout: TwoSiteLayout, q: int) -> tuple[list, dict, dict, int, int]:
    group = [b for b in layout.blocks if b.q == q]
    columns, rows, nc, nr = {}, {}, 0, 0
    for b in group:
        if b.child_channel not in columns:
            columns[b.child_channel] = (nc, b.shape[1:3])
            nc += b.shape[1]*b.shape[2]
        if b.parent_channel not in rows:
            rows[b.parent_channel] = (nr, (b.shape[0], b.shape[3]))
            nr += b.shape[0]*b.shape[3]
    return group, columns, rows, nc, nr


def split_physical_reduced_pair(entries: np.ndarray, bank: dict, layout: TwoSiteLayout, *,
                                rank: int | None = None, metric_cutoff: float = 1e-12,
                                comparison_entries: np.ndarray | None = None) -> tuple[dict, dict]:
    """Minimize resolved unprojected physical norm error separately at each q.

    Bases are whitened with their exact physical Grams. Pseudoinverses remove
    unobservable coordinate gauges; no ridge is added to the physical norm.
    A declared cutoff bounds ill-conditioned inversion, and its effect is
    included in the measured error. This is not projected-state/FS optimal.
    """
    z = np.asarray(entries, dtype=np.complex128)
    chi = layout.config.tree.feature_dim
    rank = chi if rank is None else rank
    if isinstance(rank, bool) or not isinstance(rank, Integral) or not 1 <= rank <= chi:
        raise ValueError('rank must be an integer in 1..chi')
    if z.shape != (layout.size,) or not np.isfinite(z).all():
        raise ValueError('finite correctly shaped joined entries required')
    if isinstance(metric_cutoff, bool) or not np.isfinite(metric_cutoff) or not 0 < metric_cutoff < 1:
        raise ValueError('Gram inversion cutoff must be finite in (0,1)')
    comparison = None if comparison_entries is None else np.asarray(comparison_entries, complex)
    if comparison is not None and (comparison.shape != z.shape or not np.isfinite(comparison).all()):
        raise ValueError('comparison must be finite joined entries with matching shape')
    grams, env = schur_norm_environments(bank, layout.config)
    tree = layout.config.tree.cell_tree
    child, parent = tree.internal_nodes[layout.child], tree.internal_nodes[layout.parent]
    sibling = parent.right if layout.child_is_left else parent.left
    layouts = cell_multiplicity_layouts(layout.config.tree)
    nodes = [np.asarray(a, dtype=complex).copy() for a in bank['nodes']]
    nodes[layout.child].fill(0); nodes[layout.parent].fill(0)
    records = {}; total_norm = total_error = comparison_error = 0.
    for q in sorted({b.q for b in layout.blocks}):
        group, columns, rows, nc, nr = _group_indices(layout, q)
        matrix = np.zeros((nr, nc), complex)
        column_gram = np.zeros((nc, nc), complex)
        row_gram = np.zeros((nr, nr), complex)
        for ic, (c, (na, nb)) in columns.items():
            ql, qr, _ = layouts[layout.child].keys[ic]
            column_gram[c:c+na*nb, c:c+na*nb] = np.kron(grams[child.left][ql], grams[child.right][qr])
        for ip, (r, (nk, ns)) in rows.items():
            key = layouts[layout.parent].keys[ip]
            qs, qt = key[1 if layout.child_is_left else 0], key[2]
            row_gram[r:r+nk*ns, r:r+nk*ns] = np.kron(env[parent][qt], grams[sibling][qs])
        for b in group:
            r, (nk, ns) = rows[b.parent_channel]; c, (na, nb) = columns[b.child_channel]
            matrix[r:r+nk*ns, c:c+na*nb] = z[b.start:b.stop].reshape(b.shape).transpose(0, 3, 1, 2).reshape(nk*ns, na*nb)
        left, left_inverse, li = _positive_factors(row_gram, metric_cutoff)
        right, right_inverse, ri = _positive_factors(column_gram, metric_cutoff)
        physical = left@matrix@right.T
        u, s, vh = np.linalg.svd(physical, full_matrices=False)
        keep = min(rank, len(s))
        p = left_inverse@(u[:, :keep]*np.sqrt(s[:keep]))
        a = (np.sqrt(s[:keep])[:, None]*vh[:keep])@right_inverse.T
        error = float(np.linalg.norm(left@(matrix-p@a)@right.T)**2)
        norm = float(np.sum(s*s)); total_norm += norm; total_error += error
        if comparison is not None:
            other = np.zeros_like(matrix)
            for b in group:
                r, (nk, ns) = rows[b.parent_channel]; c, (na, nb) = columns[b.child_channel]
                other[r:r+nk*ns, c:c+na*nb] = comparison[b.start:b.stop].reshape(b.shape).transpose(0,3,1,2).reshape(nk*ns,na*nb)
            comparison_error += float(np.linalg.norm(left@(matrix-other)@right.T)**2)
        for ic, (c, (na, nb)) in columns.items():
            nodes[layout.child][ic, :keep, :na, :nb] = a[:, c:c+na*nb].reshape(keep, na, nb)
        for ip, (r, (nk, ns)) in rows.items():
            value = p[r:r+nk*ns].reshape(nk, ns, keep)
            if layout.child_is_left:
                nodes[layout.parent][ip, :, :keep, :ns] = value.transpose(0, 2, 1)
            else:
                nodes[layout.parent][ip, :, :ns, :keep] = value
        records[str(q)] = dict(singular_values=s.tolist(), kept_rank=keep,
            physical_norm_squared=norm, discarded_physical_norm_squared=error,
            ideal_svd_discarded_norm_squared=float(np.sum(s[keep:]**2)),
            row_metric=li, column_metric=ri, magnetic_degeneracy=q+1)
    return {'nodes': tuple(jnp.asarray(a) for a in nodes)}, dict(q_blocks=records,
        physical_Schmidt_spectrum=True, physical_norm_squared=total_norm,
        discarded_physical_norm_squared=total_error,
        relative_physical_error=total_error/total_norm if total_norm else 0.,
        comparison_discarded_physical_norm_squared=comparison_error if comparison is not None else None,
        metric_scope='EXACT UNPROJECTED decoder tree norm; NOT full A1/additive carrier FS',
        full_projected_state_still_requires_actual_amplitude_check=True,
        truncation_is_coordinate_not_physical_metric=False)


__all__ = ['schur_norm_environments', 'split_physical_reduced_pair']
