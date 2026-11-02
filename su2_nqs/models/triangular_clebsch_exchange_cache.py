"""Exact unchanged-subtree reuse for physical triangular ClebschTree queries.

This execution-only factory consumes the differentiable feature-LUT bank.
It prepares all subtree messages once for one configuration, then recomputes
only the union of ancestors of the swapped sites. No Hamiltonian, division,
sampling rule, truncation, parameter, or persistent mutable cache appears here.
Every new parameter bank must rebuild its feature LUT inside AD.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from .triangular_clebsch_active_layout import mixed_active_layout


def exchange_query_plan(config: MixedClebschConfig, pairs: Sequence) -> tuple[dict, ...]:
    """Group integer site swaps by affected leaves and exact ancestor union.

    Charges and magnetic labels in all messages remain doubled integers. Site
    pairs are ordered exactly as supplied, including repeated queries.
    """
    tree = triangular_schur_hierarchy(config.length)
    edges = np.asarray(pairs)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= tree.n_sites)
            or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('nonempty integer site pairs without self swaps required')
    owner = np.empty(tree.n_sites, np.int32)
    for index, block in enumerate(tree.blocks):
        owner[list(block.sites)] = index
    groups = {}
    for index, (a, b) in enumerate(edges):
        leaves = tuple(sorted({int(owner[a]), int(owner[b])}))
        groups.setdefault(leaves, []).append(index)
    output = []
    for leaves, indices in sorted(groups.items()):
        affected = set(leaves)
        ancestors = []
        for index, row in enumerate(mixed_clebsch_layout(config)):
            if row['left'] in affected or row['right'] in affected:
                ancestors.append(index)
                affected.add(row['parent'])
        output.append(dict(leaves=leaves, ancestors=tuple(ancestors),
                           indices=tuple(indices), pairs=edges[indices].astype(np.int32)))
    return tuple(output)


def make_feature_lut_exchange_queries(config: MixedClebschConfig, pairs: Sequence, *,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> Callable:
    """Return query(prepared_bank,bits)->(raw_psi,raw_swapped_psi_vector).

    All swaps, including equal-spin null swaps, are evaluated in original order.
    Cached messages include both reduced vectors and their actual magnetic
    labels; changed CG coefficients are always recomputed. Exact image nodes
    are legal and never used as a denominator.
    Active upper widths require a bank proved/repacked by
    ``repack_active_clebsch_bank`` and are for fixed-state audits only.
    """
    tree = triangular_schur_hierarchy(config.length)
    layout = (mixed_clebsch_layout(config) if active_top_widths is None
              else mixed_active_layout(config, active_top_widths))
    groups = exchange_query_plan(config, pairs)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers = jnp.asarray(1 << np.arange(8, -1, -1))
    pair_count = len(pairs)

    def local_message(bank, bits, index):
        local = bits[sites[index]]
        packed = jnp.dot(local, powers)
        return ({q: table[packed] for q, table in zip(
            config.local_q, bank['local_features'][index], strict=True)},
            jnp.sum(2*local-1))

    def merge(bank, values, magnetic, index):
        row = layout[index]
        left, right = values[row['left']], values[row['right']]
        ml, mr = magnetic[row['left']], magnetic[row['right']]
        out = {q: jnp.zeros((width,), jnp.float64) for q, width in row['widths'].items()}
        for group, weights in zip(row['groups'], bank['top'][index], strict=True):
            l = jnp.stack([left[ql] for ql, _, _ in group['keys']])
            r = jnp.stack([right[qr] for _, qr, _ in group['keys']])
            sl, sr = group['left_span'], group['right_span']
            cg = jnp.asarray(group['cg'])[:, jnp.clip(ml, -sl, sl)+sl,
                                            jnp.clip(mr, -sr, sr)+sr]
            cg = jnp.where((abs(ml) <= sl) & (abs(mr) <= sr), cg, 0.)
            terms = jnp.einsum('ca,cb,ckab->ck', l, r, weights)*cg[:, None]
            for q, ids in group['output_groups']:
                out[q] = out[q]+jnp.sum(terms[jnp.asarray(ids)], axis=0)
        return out, ml+mr

    def query(bank: dict, bits: jax.Array) -> tuple[jax.Array, jax.Array]:
        bits = jnp.asarray(bits)
        if bits.shape != (tree.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('one physical integer-bit configuration required')
        valid = jnp.all((bits == 0) | (bits == 1))
        bits = jnp.clip(bits, 0, 1)
        values, magnetic = [], []
        for index in range(len(sites)):
            v, m = local_message(bank, bits, index)
            values.append(v); magnetic.append(m)
        for index in range(len(layout)):
            v, m = merge(bank, values, magnetic, index)
            values.append(v); magnetic.append(m)
        legal = valid & (magnetic[-1] == 0)
        raw = jnp.where(legal, values[-1][0][0], 0.)
        result = jnp.zeros((pair_count,), dtype=raw.dtype)
        for group in groups:
            def swapped(pair):
                a, b = pair
                changed = bits.at[a].set(bits[b]).at[b].set(bits[a])
                updated, updated_m = list(values), list(magnetic)
                for index in group['leaves']:
                    updated[index], updated_m[index] = local_message(bank, changed, index)
                for index in group['ancestors']:
                    row = layout[index]
                    updated[row['parent']], updated_m[row['parent']] = merge(
                        bank, updated, updated_m, index)
                return jnp.where(legal, updated[-1][0][0], 0.)
            computed = jax.vmap(swapped)(jnp.asarray(group['pairs']))
            result = result.at[jnp.asarray(group['indices'])].set(computed)
        return raw, result

    return query
