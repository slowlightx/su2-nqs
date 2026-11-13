"""Exact exchange queries that stop at the lowest common ancestor (LCA).

Drop-in alternative to ``make_feature_lut_exchange_queries`` with the SAME
signature and output: ``query(prepared_bank,bits)->(raw_psi,swapped_vector)``.

The raw tree amplitude is multilinear in the node messages once all doubled
magnetic labels are fixed. Swapping two sites whose lowest common tree
ancestor is ``c`` leaves the magnetic label of ``c`` and of every node above
``c`` unchanged, so

    psi(x^b) = < env_c(x) , message_c(x^b) >,

where ``env_c = d psi / d message_c`` is obtained for every node from ONE
reverse pass per configuration.  Hence an intra-leaf swap costs one local
feature gather plus a dot product (no merges), and an inter-leaf swap only
recomputes merges from its two leaves up to ``c`` instead of up to the root.
No CG table, doubled-q/m label, truncation or parameter is changed; every
query is exact (not sampled).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def lca_exchange_plan(config: MixedClebschConfig, pairs: Sequence) -> tuple[dict, ...]:
    """Group site swaps by touched leaves; record merges up to the LCA only."""
    tree = triangular_schur_hierarchy(config.length)
    layout = mixed_clebsch_layout(config)
    edges = np.asarray(pairs)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= tree.n_sites)
            or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('nonempty integer site pairs without self swaps required')
    blocks = len(tree.blocks)
    owner = np.empty(tree.n_sites, np.int32)
    for index, block in enumerate(tree.blocks):
        owner[list(block.sites)] = index
    parent_row = {}
    for index, row in enumerate(layout):
        parent_row[row['left']] = index
        parent_row[row['right']] = index

    def chain(node):
        nodes = [node]
        while node in parent_row:
            node = layout[parent_row[node]]['parent']
            nodes.append(node)
        return nodes

    groups: dict[tuple[int, ...], list[int]] = {}
    for index, (a, b) in enumerate(edges):
        groups.setdefault(tuple(sorted({int(owner[a]), int(owner[b])})), []).append(index)
    output = []
    for leaves, indices in sorted(groups.items()):
        if len(leaves) == 1:
            target, rows = leaves[0], ()
        else:
            first, second = chain(leaves[0]), chain(leaves[1])
            target = next(node for node in first if node in second)
            needed = set(first[:first.index(target)]) | set(second[:second.index(target)])
            rows = tuple(sorted(parent_row[node] for node in needed))
            if layout[rows[-1]]['parent'] != target:
                raise RuntimeError('LCA merge ordering is inconsistent')
        output.append(dict(leaves=leaves, rows=rows, target=target, indices=tuple(indices),
                           pairs=edges[indices].astype(np.int32)))
    if blocks + len(layout) - 1 != layout[-1]['parent']:
        raise RuntimeError('noncanonical node numbering')
    return tuple(output)


def make_lca_exchange_queries(config: MixedClebschConfig, pairs: Sequence) -> Callable:
    """Return query(prepared_bank,bits)->(raw_psi,raw_swapped_psi_vector).

    Output order equals ``pairs``; equal-spin (null) swaps are evaluated too,
    exactly as in the legacy factory.  Only the default (non-active) layout is
    supported.
    """
    tree = triangular_schur_hierarchy(config.length)
    layout = mixed_clebsch_layout(config)
    groups = lca_exchange_plan(config, pairs)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers = jnp.asarray(1 << np.arange(8, -1, -1))
    pair_count = len(pairs)

    def local_message(bank, bits, index):
        local = bits[sites[index]]
        packed = jnp.dot(local, powers)
        return ({q: table[packed] for q, table in zip(
            config.local_q, bank['local_features'][index], strict=True)},
            jnp.sum(2*local-1))

    def merge_values(bank, index, left, right, ml, mr):
        row = layout[index]
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
        return out

    def contract(env, message):
        return sum(jnp.dot(env[q], message[q]) for q in env)

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
        for index, row in enumerate(layout):
            values.append(merge_values(bank, index, values[row['left']], values[row['right']],
                                       magnetic[row['left']], magnetic[row['right']]))
            magnetic.append(magnetic[row['left']]+magnetic[row['right']])
        legal = valid & (magnetic[-1] == 0)
        raw = jnp.where(legal, values[-1][0][0], 0.)
        # One reverse pass: environments of every node at fixed magnetic labels.
        envs: list = [None]*len(values)
        # Seed dtype must follow the bank (real or complex). JAX's VJP is the
        # plain transpose of the linearization, so <env, message> needs no conj.
        envs[-1] = {0: jnp.ones((1,), values[-1][0].dtype)}
        for index in reversed(range(len(layout))):
            row = layout[index]
            ml, mr = magnetic[row['left']], magnetic[row['right']]
            _, pullback = jax.vjp(lambda l, r, i=index, a=ml, b=mr: merge_values(bank, i, l, r, a, b),
                                  values[row['left']], values[row['right']])
            envs[row['left']], envs[row['right']] = pullback(envs[row['parent']])
        result = jnp.zeros((pair_count,), dtype=raw.dtype)
        for group in groups:
            def swapped(pair, group=group):
                a, b = pair
                changed = bits.at[a].set(bits[b]).at[b].set(bits[a])
                updated, updated_m = {}, {}
                for leaf in group['leaves']:
                    updated[leaf], updated_m[leaf] = local_message(bank, changed, leaf)
                for index in group['rows']:
                    row = layout[index]
                    left = updated.get(row['left'], values[row['left']])
                    right = updated.get(row['right'], values[row['right']])
                    ml = updated_m.get(row['left'], magnetic[row['left']])
                    mr = updated_m.get(row['right'], magnetic[row['right']])
                    updated[row['parent']] = merge_values(bank, index, left, right, ml, mr)
                    updated_m[row['parent']] = ml+mr
                return jnp.where(legal, contract(envs[group['target']], updated[group['target']]), 0.)
            computed = jax.vmap(swapped)(jnp.asarray(group['pairs']))
            result = result.at[jnp.asarray(group['indices'])].set(computed)
        return raw, result

    return query

