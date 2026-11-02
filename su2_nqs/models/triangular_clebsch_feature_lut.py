"""Optional exact nine-site feature cache for the physical ClebschTree.

This is an execution-only representation of an existing reduced-tensor bank,
not a new ansatz. Rebuild the512-row tables after each parameter update. Keep
preparation inside AD when differentiating with respect to the original bank.
The sealed legacy factories remain independent numerical references.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.triangle_blocks import triangle_block_tables
from .triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude, mixed_clebsch_layout,
)
from .triangular_clebsch_active_layout import mixed_active_layout


def prepare_feature_lut_bank(bank: dict, config: MixedClebschConfig) -> dict:
    """Precontract U_q[512,multiplicity] @ W_q once per immutable bank.

    q is doubled angular momentum. Every legal local magnetic component and
    all retained reduced copies remain; no new parameter or truncation appears.
    Preparation is pure JAX and differentiable, without a hidden persistent cache.
    """
    tree = triangular_schur_hierarchy(config.length)
    tables = triangle_block_tables()['nine']
    if len(bank['local']) != len(tree.blocks) or len(bank['top']) != len(tree.binary_merges):
        raise ValueError('bank does not match the declared hierarchy')
    local = []
    for node in bank['local']:
        if len(node) != len(config.local_q):
            raise ValueError('local bank must retain every declared doubled-q block')
        values = []
        for q, weights in zip(config.local_q, node, strict=True):
            if weights.ndim != 2 or weights.shape[0] != tables[q].shape[1]:
                raise ValueError('local reduced multiplicity shape mismatch')
            values.append(jnp.asarray(tables[q]) @ weights)
        local.append(tuple(values))
    return dict(local_features=tuple(local), top=bank['top'])


def make_feature_lut_amplitude(config: MixedClebschConfig, *,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> Callable:
    """Return raw(prepared_bank,bits), with exactly the old root q=m=0.

    The upper contraction below deliberately mirrors the independent legacy
    factory. Only its closed-over local U[bits]@W operation becomes a gather.
    ``active_top_widths`` is an optional fixed-bank audit layout. Use
    ``repack_active_clebsch_bank`` first to prove exact parent closure; this
    reduced execution does not preserve native channel-opening gradients.
    """
    tree = triangular_schur_hierarchy(config.length)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    layout = (mixed_clebsch_layout(config) if active_top_widths is None
              else mixed_active_layout(config, active_top_widths))
    powers = jnp.asarray(1 << np.arange(8, -1, -1))

    def amplitude(bank: dict, bits: jax.Array) -> jax.Array:
        bits = jnp.asarray(bits)
        if bits.shape != (tree.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('one physical integer-bit configuration required')
        valid = jnp.all((bits == 0) | (bits == 1)); bits = jnp.clip(bits, 0, 1)
        values = []; magnetic = []
        for i, ids in enumerate(sites):
            local = bits[ids]; packed = jnp.dot(local, powers)
            values.append({q: table[packed] for q, table in zip(
                config.local_q, bank['local_features'][i], strict=True)})
            magnetic.append(jnp.sum(2*local-1))
        # Identical charge buckets, W, CG orientation and magnetic selection.
        for index, row in enumerate(layout):
            left, right = values[row['left']], values[row['right']]
            ml, mr = magnetic[row['left']], magnetic[row['right']]
            out = {q: jnp.zeros((width,), jnp.float64) for q, width in row['widths'].items()}
            for group, weights in zip(row['groups'], bank['top'][index], strict=True):
                l = jnp.stack([left[ql] for ql, _, _ in group['keys']])
                r = jnp.stack([right[qr] for _, qr, _ in group['keys']])
                sl, sr = group['left_span'], group['right_span']
                cg = jnp.asarray(group['cg'])[:, jnp.clip(ml, -sl, sl)+sl, jnp.clip(mr, -sr, sr)+sr]
                cg = jnp.where((abs(ml) <= sl) & (abs(mr) <= sr), cg, 0.)
                terms = jnp.einsum('ca,cb,ckab->ck', l, r, weights)*cg[:, None]
                for q, ids in group['output_groups']:
                    out[q] = out[q]+jnp.sum(terms[jnp.asarray(ids)], axis=0)
            values.append(out); magnetic.append(ml+mr)
        return jnp.where(valid & (magnetic[-1] == 0), values[-1][0][0], 0.)

    return amplitude


def make_feature_lut_projected_amplitude(
    config: MixedClebschConfig, images: np.ndarray, *, chunk: int = 4,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> Callable:
    """Apply the same complete coherent A1 mean, without changing image weights.

    Supply the same verified full lattice group as the legacy projector. No
    geometry is inferred from local C3 labels or from coincident LUT entries.
    Optional active widths have the same fixed-bank-only meaning as the raw
    feature factory; no spatial images are removed by that optimization.
    """
    # Reuse its API validation without changing the sealed implementation.
    make_mixed_projected_amplitude(config, images, chunk=chunk)
    raw = make_feature_lut_amplitude(config, active_top_widths=active_top_widths)
    maps = np.asarray(images)
    groups = jnp.asarray(maps.reshape(-1, chunk, config.length**2))

    def amplitude(bank: dict, bits: jax.Array) -> jax.Array:
        def partial(group):
            return jnp.sum(jax.vmap(raw, in_axes=(None, 0))(bank, bits[group]))
        return jnp.sum(jax.lax.map(jax.checkpoint(partial), groups))/len(maps)

    return amplitude
