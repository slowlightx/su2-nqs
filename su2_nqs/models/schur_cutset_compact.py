"""Equivalent packed-CG evaluation of the finite Schur-cutset model.

This changes contraction/storage only, not the state manifold, six branches,
allowed doubled-q labels, or full spatial projector. It accepts the exact
per-node branch bank produced by ``schur_cutset_attention``. Unlike the old
dense table, a CG lookup stores only the channel triples the model uses.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.local_schur import cell_schur_tensor
from ..su2.quantum_numbers import coupling_range, magnetic_numbers
from .cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from .schur_cutset_attention import CutsetAttentionConfig, SCHUR_LABELS
from .spatial_projector import PhysicalSpatialProjector


@lru_cache(maxsize=None)
def packed_cg_table(
    left_size: int, right_size: int, keys: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Readonly ``CG[channel,m_left+L,m_right+R]`` in doubled labels.

    Channel selection is inherited from the complete model layout: no q cutoff
    is introduced. Invalid magnetic projections remain exact zeros. The root
    singlet consequently needs only its allowed ``(q,q,0)`` channels instead
    of allocating every unused total-spin block.
    """
    for name, size in (('left_size', left_size), ('right_size', right_size)):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('nonempty distinct channel triples are required')
    for key in keys:
        if len(key) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in key):
            raise TypeError('channel triples must contain doubled integer labels')
        left, right, total = key
        if (not 0 <= left <= left_size or (left_size-left) % 2
                or not 0 <= right <= right_size or (right_size-right) % 2
                or total not in coupling_range(left, right)):
            raise ValueError('channel is not reachable for the physical subtree sizes')
    table = np.zeros((len(keys), 2*left_size+1, 2*right_size+1), dtype=np.float64)
    for index, (left, right, total) in enumerate(keys):
        for ml in magnetic_numbers(left):
            for mr in magnetic_numbers(right):
                if abs(ml+mr) <= total:
                    table[index, ml+left_size, mr+right_size] = clebsch_gordan(
                        left, ml, right, mr, total, ml+mr)
    table.setflags(write=False)
    return table


def compact_cg_storage(config: CutsetAttentionConfig) -> dict:
    """Static allocation counts; does not allocate the legacy dense tables."""
    unique = {}
    for layout in cell_multiplicity_layouts(config.tree):
        key = (layout.left_size, layout.right_size, layout.keys)
        if key in unique:
            continue
        left, right = layout.left_size, layout.right_size
        unique[key] = dict(left_size=left, right_size=right, channels=len(layout.keys),
            packed_bytes=len(layout.keys)*(2*left+1)*(2*right+1)*8,
            legacy_dense_bytes=(left+1)*(2*left+1)*(right+1)*(2*right+1)*(left+right+1)*8)
    rows = list(unique.values())
    return dict(tables=rows, packed_bytes=sum(row['packed_bytes'] for row in rows),
                legacy_dense_bytes=sum(row['legacy_dense_bytes'] for row in rows),
                allowed_channels_truncated=False)


def make_compact_cutset_amplitude(config: CutsetAttentionConfig) -> Callable:
    """Build ``raw(bank,bits)`` with six branches and vectorized CG channels.

    Reduced-axis zero padding is a numerical packing operation, not a new
    feature dimension or a truncation. Numeric CG tables are prepared outside
    JAX tracing; runtime never performs factorial or symbolic operations.
    """
    if not isinstance(config, CutsetAttentionConfig):
        raise TypeError('config must be CutsetAttentionConfig')
    tree_config = config.tree; tree = tree_config.cell_tree
    layouts = cell_multiplicity_layouts(tree_config)
    tables = tuple(jnp.asarray(packed_cg_table(row.left_size, row.right_size, row.keys))
                   for row in layouts)
    labels = jnp.asarray(SCHUR_LABELS, dtype=jnp.int32)
    schur = jnp.asarray(cell_schur_tensor(tree_config.cell_kind))
    powers = jnp.asarray([8, 4, 2, 1], dtype=jnp.int32)
    channels = tuple(jnp.asarray(row.keys, dtype=jnp.int32) for row in layouts)
    pads = tuple((max(row.left_widths), max(row.right_widths)) for row in layouts)

    def raw(bank: dict, spins: jax.Array) -> jax.Array:
        bits = jnp.asarray(spins, dtype=jnp.int32)
        if bits.shape != (tree_config.n_sites,):
            raise ValueError('wrong physical spin shape')
        valid = jnp.all((bits == 0) | (bits == 1)); bits = jnp.clip(bits, 0, 1)
        leaves = {}
        for leaf in tree.leaves:
            local_bits = bits[jnp.asarray(tree_config.cells[leaf.index])]
            m = jnp.sum(2*local_bits-1)
            local = jnp.broadcast_to(schur[jnp.dot(local_bits, powers), :, m+4, :], (6, 5, 3))
            if leaf.index == config.cutset_cell:
                local = local*((jnp.arange(5)[None, :, None] == labels[:, 0, None, None])
                              & (jnp.arange(3)[None, None, :] == labels[:, 1, None, None]))
            leaves[leaf] = (m, local)
        output = []
        for part in ('real', 'imag'):
            values = dict(leaves)
            for index, (node, layout) in enumerate(zip(tree.internal_nodes, layouts, strict=True)):
                ml, left = values[node.left]; mr, right = values[node.right]
                ql, qr, qt = channels[index].T
                lw, rw = pads[index]
                kernels = jnp.stack([
                    jnp.pad(kernel, ((0, 0), (0, 0), (0, lw-a), (0, rw-b)))
                    for kernel, a, b in zip(bank[part]['merges'][index]['joint'],
                                            layout.left_widths, layout.right_widths, strict=True)], axis=1)
                reduced = jnp.einsum('zcl,zcr,zcklr->zck',
                    left[:, ql, :lw], right[:, qr, :rw], kernels)
                cg = tables[index][:, ml+layout.left_size, mr+layout.right_size]
                result = jnp.zeros((6, layout.size+1, tree_config.feature_dim), dtype=reduced.dtype)
                result = result.at[:, qt, :].add(cg[None, :, None]*reduced)
                values[node] = (ml+mr, result)
            m, root = values[tree.root]
            output.append(jnp.sum(root[:, 0, :]*bank[part]['readout'], axis=-1))
        amplitude = jnp.sum(output[0]+1j*output[1])
        return jnp.where(valid & (m == 0), amplitude, 0j)
    return raw


def make_full_a1_batch(
    raw_amplitude: Callable, projector: PhysicalSpatialProjector, *, chunk_size: int = 16,
) -> Callable:
    """Key-free full-A1 scan/rematerialization, including N>=64 physical sites.

    No bit packing, identity deduplication, empirical symmetry quotient, or
    omitted group images. This is the same coherent mean as the legacy
    chunked evaluator, whose shared uint64 guard unnecessarily rejects N64.
    """
    if not isinstance(projector, PhysicalSpatialProjector) or projector.fermionic:
        raise ValueError('need a bosonic physical projector')
    if any(character != 1 for character in projector.characters):
        raise ValueError('this wrapper requires the trivial full-A1 character')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    images = jnp.asarray(projector.site_permutations, dtype=jnp.int32)
    group = projector.group_order
    evaluate = jax.checkpoint(jax.vmap(raw_amplitude, in_axes=(None, 0)))

    def amplitudes(bank: dict, states: jax.Array) -> jax.Array:
        states = jnp.asarray(states, dtype=jnp.int32)
        if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != projector.n_sites:
            raise ValueError('expected nonempty (batch,n_sites) physical states')
        count = states.shape[0]*group; chunks = (count+chunk_size-1)//chunk_size
        def step(_, index):
            locations = index*chunk_size+jnp.arange(chunk_size)
            valid = locations < count; safe = jnp.minimum(locations, count-1)
            bits = states[safe//group]
            pulled = jnp.take_along_axis(bits, images[safe % group], axis=1)
            return None, jnp.where(valid, evaluate(bank, pulled), 0)
        _, values = jax.lax.scan(step, None, jnp.arange(chunks))
        return values.reshape(-1)[:count].reshape(-1, group).mean(axis=1)
    return amplitudes


__all__ = ['packed_cg_table', 'compact_cg_storage', 'make_compact_cutset_amplitude', 'make_full_a1_batch']
