"""Two-node reduced-tensor updates, without changing magnetic CG tensors.

The joined tensor is linear in the physical amplitude. A separate SVD for
each doubled intermediate spin q splits it back into whole SU(2) multiplets.
The singular values here are *tensor-coordinate* values, not physical Schmidt
values unless both surrounding bases are orthonormal. No Hamiltonian logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import Leaf
from ..su2.local_schur import cell_schur_tensor
from .cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from .finite_schur_decoder import FiniteSchurDecoderConfig
from .schur_cutset_compact import packed_cg_table


def embed_reduced_bank(bank: dict, source: FiniteSchurDecoderConfig,
                       target: FiniteSchurDecoderConfig) -> dict:
    """Zero-pad multiplicity copies; preserve EVERY physical amplitude.

    q, cell partition and tree are unchanged. The root receives sqrt(chi_new/
    chi_old) to cancel the decoder's root readout normalization.
    """
    if (source.tree.cells != target.tree.cells or source.tree.maximum_intermediate_q
            != target.tree.maximum_intermediate_q or source.tree.target_q != target.tree.target_q
            or source.tree.merge_mode != target.tree.merge_mode):
        raise ValueError('embedding must keep cells, doubled q and merge convention')
    old, new = source.tree.feature_dim, target.tree.feature_dim
    if new < old:
        raise ValueError('multiplicity embedding cannot shrink')
    result = []
    for i, (a, b) in enumerate(zip(cell_multiplicity_layouts(source.tree),
                                 cell_multiplicity_layouts(target.tree), strict=True)):
        if a.keys != b.keys:
            raise ValueError('embedding cannot add or remove q channels')
        values = jnp.asarray(bank['nodes'][i])
        padded = jnp.pad(values, ((0, 0), (0, new-old),
            (0, max(b.left_widths)-values.shape[2]),
            (0, max(b.right_widths)-values.shape[3])))
        result.append(padded * (np.sqrt(new/old) if i == len(bank['nodes'])-1 else 1.))
    return {'nodes': tuple(result)}


@dataclass(frozen=True)
class JoinedBlock:
    """One allowed pair of child/parent CG channels; all spin labels doubled."""
    child_channel: int
    parent_channel: int
    q: int
    start: int
    stop: int
    shape: tuple[int, int, int, int]  # parent-out, grandchild-L/R, sibling


@dataclass(frozen=True)
class TwoSiteLayout:
    config: FiniteSchurDecoderConfig
    parent: int
    child: int
    child_is_left: bool
    blocks: tuple[JoinedBlock, ...]

    @property
    def size(self) -> int:
        return self.blocks[-1].stop


def internal_edges(config: FiniteSchurDecoderConfig) -> tuple[tuple[int, int], ...]:
    """Deterministic parent/child edges between trainable internal nodes."""
    nodes = config.tree.cell_tree.internal_nodes
    indices = {node: i for i, node in enumerate(nodes)}
    return tuple((i, indices[child]) for i, node in enumerate(nodes)
                 for child in (node.left, node.right) if not isinstance(child, Leaf))


@lru_cache(maxsize=None)
def two_site_layout(config: FiniteSchurDecoderConfig, parent: int,
                    child: int) -> TwoSiteLayout:
    """Validate adjacency and enumerate joined ACTIVE reduced entries."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (parent, child)):
        raise TypeError('node indices must be integers')
    if (parent, child) not in internal_edges(config):
        raise ValueError('expected an adjacent internal parent/child pair')
    nodes = config.tree.cell_tree.internal_nodes
    left = nodes[parent].left == nodes[child]
    layouts = cell_multiplicity_layouts(config.tree)
    c, p, width = layouts[child], layouts[parent], config.tree.feature_dim
    blocks, offset = [], 0
    for ic, (ql, qr, q) in enumerate(c.keys):
        for ip, key in enumerate(p.keys):
            if key[0 if left else 1] != q:
                continue
            sibling_width = (p.right_widths if left else p.left_widths)[ip]
            shape = (width, c.left_widths[ic], c.right_widths[ic], sibling_width)
            count = int(np.prod(shape))
            blocks.append(JoinedBlock(ic, ip, q, offset, offset+count, shape))
            offset += count
    return TwoSiteLayout(config, int(parent), int(child), left, tuple(blocks))


def join_reduced_pair(bank: dict, layout: TwoSiteLayout) -> np.ndarray:
    """Contract only the shared multiplicity index; keep its doubled q."""
    child, parent = (np.asarray(bank['nodes'][i]) for i in (layout.child, layout.parent))
    result = []
    for b in layout.blocks:
        _, na, nb, ns = b.shape
        a = child[b.child_channel, :, :na, :nb]
        p = parent[b.parent_channel]
        result.append((np.einsum('kvs,vab->kabs', p[:, :, :ns], a) if layout.child_is_left
            else np.einsum('ksv,vab->kabs', p[:, :ns, :], a)).reshape(-1))
    return np.concatenate(result)


def split_reduced_pair(entries: np.ndarray, bank: dict, layout: TwoSiteLayout,
                       *, rank: int | None = None) -> tuple[dict, dict]:
    """Block-q SVD retraction; no magnetic truncation or sample repair.

    Every parent/child block belonging to the same shared q is assembled into
    ONE matrix before SVD, not decomposed independently per CG triple.
    """
    z = np.asarray(entries, dtype=np.complex128)
    chi = layout.config.tree.feature_dim
    rank = chi if rank is None else rank
    if isinstance(rank, bool) or not isinstance(rank, Integral) or not 1 <= rank <= chi:
        raise ValueError('rank must be an integer between one and configured chi')
    if z.shape != (layout.size,) or not np.isfinite(z).all():
        raise ValueError('joined active entries must be finite and correctly shaped')
    nodes = [np.asarray(v, dtype=np.complex128).copy() for v in bank['nodes']]
    nodes[layout.child].fill(0); nodes[layout.parent].fill(0)
    record = {}
    for q in sorted({b.q for b in layout.blocks}):
        group = [b for b in layout.blocks if b.q == q]
        columns, rows = {}, {}
        nc = nr = 0
        for b in group:
            if b.child_channel not in columns:
                columns[b.child_channel] = (nc, b.shape[1:3]); nc += b.shape[1]*b.shape[2]
            if b.parent_channel not in rows:
                rows[b.parent_channel] = (nr, (b.shape[0], b.shape[3])); nr += b.shape[0]*b.shape[3]
        matrix = np.zeros((nr, nc), dtype=np.complex128)
        for b in group:
            r, (nk, ns) = rows[b.parent_channel]; c, (na, nb) = columns[b.child_channel]
            matrix[r:r+nk*ns, c:c+na*nb] = z[b.start:b.stop].reshape(b.shape).transpose(0, 3, 1, 2).reshape(nk*ns, na*nb)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = min(rank, len(s))
        # Balanced factors control coordinate gauges, not physical FS distance.
        a, p = np.sqrt(s[:keep])[:, None]*vh[:keep], u[:, :keep]*np.sqrt(s[:keep])[None]
        for ic, (c, (na, nb)) in columns.items():
            nodes[layout.child][ic, :keep, :na, :nb] = a[:, c:c+na*nb].reshape(keep, na, nb)
        for ip, (r, (nk, ns)) in rows.items():
            value = p[r:r+nk*ns].reshape(nk, ns, keep)
            if layout.child_is_left:
                nodes[layout.parent][ip, :, :keep, :ns] = value.transpose(0, 2, 1)
            else:
                nodes[layout.parent][ip, :, :ns, :keep] = value
        total = float(np.sum(s*s))
        record[str(q)] = dict(singular_values=s.tolist(), kept_rank=keep,
            discarded_coordinate_norm_squared=float(np.sum(s[keep:]**2)),
            discarded_coordinate_fraction=float(np.sum(s[keep:]**2)/total) if total else 0.)
    return {'nodes': tuple(jnp.asarray(v) for v in nodes)}, dict(q_blocks=record,
        physical_Schmidt_spectrum=False, truncation_is_coordinate_not_physical_metric=True)


def make_two_site_features(layout: TwoSiteLayout) -> Callable:
    """Return F(bank,bits), with amplitude = F @ joined_entries exactly.

    Fixed local Schur and CG tensors determine the magnetic structure. All
    feature columns and all subsequent complex linear combinations are S=0.
    """
    config, tree = layout.config, layout.config.tree.cell_tree
    nodes = tree.internal_nodes; layouts = cell_multiplicity_layouts(config.tree)
    spec = config.tree
    cap, chi = spec.maximum_intermediate_q, spec.feature_dim
    schur = jnp.asarray(cell_schur_tensor('square_plaquette'))
    tables = tuple(jnp.asarray(packed_cg_table(row.left_size, row.right_size, row.keys)) for row in layouts)
    keys = tuple(jnp.asarray(row.keys, dtype=jnp.int32) for row in layouts)
    labels = []
    for b in layout.blocks:
        for k, a, c, s in np.ndindex(b.shape):
            labels.append((b.child_channel, b.parent_channel, k, a, c, s))
    lab = jnp.asarray(labels, dtype=jnp.int32)

    def features(bank: dict, bits: jax.Array) -> jax.Array:
        bits = jnp.asarray(bits)
        if bits.shape != (spec.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('expected one integer physical bit row')
        valid = jnp.all((bits == 0) | (bits == 1)); bits = jnp.clip(bits, 0, 1)
        values, magnetic, cgs = {}, {}, []
        for leaf in tree.leaves:
            local = bits[jnp.asarray(spec.cells[leaf.index])]; m = jnp.sum(2*local-1)
            magnetic[leaf] = m
            values[leaf] = schur[jnp.dot(local, jnp.asarray([8, 4, 2, 1])), :, m+4, :]
        for i, (node, row) in enumerate(zip(nodes, layouts, strict=True)):
            ml, mr = magnetic[node.left], magnetic[node.right]
            magnetic[node] = ml+mr
            ql, qr, qt = keys[i].T
            cg = tables[i][:, ml+row.left_size, mr+row.right_size]; cgs.append(cg)
            lv, rv = values[node.left], values[node.right]
            value = jnp.einsum('cl,cr,cklr->ck', lv[ql, :max(row.left_widths)],
                               rv[qr, :max(row.right_widths)], bank['nodes'][i])
            values[node] = jnp.zeros((cap+1, chi), dtype=jnp.complex128).at[qt].add(cg[:, None]*value)
        environments = {tree.root: jnp.zeros((cap+1, chi), dtype=jnp.complex128).at[0].set(1/np.sqrt(chi))}
        for i in range(len(nodes)-1, -1, -1):
            node, row = nodes[i], layouts[i]
            if node not in environments or i == layout.parent:
                continue
            ql, qr, qt = keys[i].T
            weights = environments[node][qt]*cgs[i][:, None]
            if not isinstance(node.left, Leaf):
                value = jnp.einsum('ck,cr,cklr->cl', weights,
                    values[node.right][qr, :max(row.right_widths)], bank['nodes'][i])
                environments[node.left] = jnp.zeros((cap+1, chi), dtype=jnp.complex128).at[ql].add(value)
            if not isinstance(node.right, Leaf):
                value = jnp.einsum('ck,cl,cklr->cr', weights,
                    values[node.left][ql, :max(row.left_widths)], bank['nodes'][i])
                environments[node.right] = jnp.zeros((cap+1, chi), dtype=jnp.complex128).at[qr].add(value)
        child, parent = nodes[layout.child], nodes[layout.parent]
        ic, ip, k, a, b, s = lab.T
        ql, qr, _ = keys[layout.child][ic].T
        pkeys = keys[layout.parent][ip]
        sibling = parent.right if layout.child_is_left else parent.left
        qs = pkeys[:, 1 if layout.child_is_left else 0]; qt = pkeys[:, 2]
        result = (environments[parent][qt, k]*values[child.left][ql, a]
                  *values[child.right][qr, b]*values[sibling][qs, s]
                  *cgs[layout.child][ic]*cgs[layout.parent][ip])
        return jnp.where(valid & (magnetic[tree.root] == 0), result, 0j)
    return features


__all__ = ['embed_reduced_bank', 'internal_edges', 'two_site_layout',
           'join_reduced_pair', 'split_reduced_pair', 'make_two_site_features']
