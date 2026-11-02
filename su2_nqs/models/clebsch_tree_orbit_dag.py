"""Exact ClebschTree orbit-DAG execution backend.

The parameterization, doubled labels, CG phases and coherent A1 sum are those
of ``literature_clebsch_tree``. Only identical *ordered physical subtrees* in
different space-group images are shared. No subtree permutation is treated as
an identity. This optional execution backend does not change the physical ansatz.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, clebsch_layers,
)
from su2_nqs.models.clebsch_tree_projected import root_swap_bank, make_projected_clebsch
from su2_nqs.lattices.spatial_symmetry import validated_permutation_group
from su2_nqs.su2.fusion_tree import Leaf, TreeNode


def _leaves(node: TreeNode) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (node.index,)
    return _leaves(node.left) + _leaves(node.right)


def make_orbit_dag_clebsch(
    config: LiteratureClebschTreeConfig,
    physical_order: Sequence[int],
    group_images: Sequence[Sequence[int]],
    *,
    fold_root: bool = True,
) -> tuple[Callable, dict]:
    """Return an exact projected batch amplitude and deterministic DAG sizes.

    This first prototype requires balanced, equal-child layers, as used by
    the square ``L=2**k`` reproduction. The learned output labels remain
    doubled ``q=2j``; only active feature widths are padded within a layer.
    """
    n = config.tree.n_leaves
    order = np.asarray(physical_order, dtype=np.int32)
    if order.shape != (n,) or sorted(order.tolist()) != list(range(n)):
        raise ValueError("physical_order must be a permutation")
    group = np.asarray(validated_permutation_group(
        tuple(tuple(int(x) for x in row) for row in group_images)), dtype=np.int32)
    if group.shape[1] != n:
        raise ValueError("group/site mismatch")
    representatives = group
    use_fold = False
    root = config.tree.root
    left, right = _leaves(root.left), _leaves(root.right)
    if fold_root and config.target_q == 0 and len(left) == len(right):
        swap = np.arange(n, dtype=np.int32)
        swap[order[np.asarray(left)]] = order[np.asarray(right)]
        swap[order[np.asarray(right)]] = order[np.asarray(left)]
        if tuple(swap) in set(map(tuple, group)):
            keys = [min(tuple(g), tuple(g[swap])) for g in group]
            representatives, counts = np.unique(keys, axis=0, return_counts=True)
            if not np.all(counts == 2):
                raise ArithmeticError("root-subgroup cosets incomplete")
            use_fold = True
    layers = clebsch_layers(config)
    previous = {(i,): i for i in range(n)}
    previous_width = 1
    plans = []
    sizes = []
    for index, layer in enumerate(layers):
        patterns = {}
        for image in representatives:
            for node in layer.nodes:
                ll, rr = _leaves(node.left), _leaves(node.right)
                if len(ll) != len(rr) or len(ll) != 2**index:
                    raise ValueError("prototype requires balanced equal-child layers")
                pattern = tuple(int(x) for x in image[order[np.asarray(ll + rr)]])
                patterns[pattern] = (pattern[:len(ll)], pattern[len(ll):])
        ordered = tuple(sorted(patterns))
        current = {pattern: i for i, pattern in enumerate(ordered)}
        left_ids = np.asarray([previous[patterns[p][0]] for p in ordered], dtype=np.int32)
        right_ids = np.asarray([previous[patterns[p][1]] for p in ordered], dtype=np.int32)
        output_width = max(layer.widths)
        indices = np.full((len(layer.keys), output_width,
                           previous_width, previous_width), -1, dtype=np.int32)
        cursor = 0
        for k, shape in enumerate(layer.shapes):
            count = int(np.prod(shape))
            indices[k, :shape[0], :shape[1], :shape[2]] = np.arange(
                cursor, cursor + count).reshape(shape)
            cursor += count
        if cursor != layer.parameter_size:
            raise ArithmeticError("parameter layout changed")
        plans.append((jnp.asarray(left_ids), jnp.asarray(right_ids),
                      jnp.asarray(indices), jnp.asarray(layer.keys),
                      jnp.asarray(layer.cg), output_width))
        sizes.append(dict(layer=index, original_nodes_per_image=len(layer.nodes),
                          image_count=len(representatives), unique_ordered_nodes=len(ordered),
                          original_padded_scalars=int(np.prod(layer.padded_indices.shape)),
                          active_padded_scalars=int(np.prod(indices.shape))))
        previous, previous_width = current, output_width
    root_ids = jnp.asarray([previous[tuple(int(x) for x in image[order])]
                            for image in representatives], dtype=jnp.int32)
    bound = config.maximum_q

    def amplitudes(params: dict, bits: jax.Array) -> jax.Array:
        rows = jnp.asarray(bits, dtype=jnp.int32)
        if rows.ndim != 2 or rows.shape[1] != n:
            raise ValueError("expected [batch,n_sites] binary rows")
        valid = jnp.all((rows == 0) | (rows == 1), axis=1)
        safe = jnp.clip(rows, 0, 1)
        magnetic = 2 * safe - 1
        feature = jnp.zeros((len(rows), n, bound + 1, 1), dtype=params['readout'].dtype)
        feature = feature.at[:, :, 1, 0].set(1.)
        banks = root_swap_bank(params, config) if use_fold else params
        for bank, plan in zip(banks['merges'], plans, strict=True):
            li, ri, indices, channels, cg_table, width = plan
            lm, rm = magnetic[:, li], magnetic[:, ri]
            lf, rf = feature[:, li], feature[:, ri]
            padded = jnp.concatenate((bank, jnp.zeros((1,), dtype=bank.dtype)))[indices]
            reduced = jnp.einsum('bnca,bncd,coad->bnco',
                                lf[:, :, channels[:, 0]], rf[:, :, channels[:, 1]], padded)
            cg = cg_table[jnp.arange(len(channels))[None, None, :],
                          jnp.clip(lm + bound, 0, 2 * bound)[:, :, None],
                          jnp.clip(rm + bound, 0, 2 * bound)[:, :, None]]
            cg = jnp.where(((abs(lm) <= bound) & (abs(rm) <= bound))[:, :, None], cg, 0.)
            feature = jnp.zeros((len(rows), len(li), bound + 1, width), dtype=bank.dtype)
            feature = feature.at[:, :, channels[:, 2], :].add(reduced * cg[:, :, :, None])
            magnetic = lm + rm
        root = feature[:, root_ids, config.target_q, :len(params['readout'])]
        value = jnp.mean(root @ params['readout'], axis=1)
        return jnp.where(valid & (jnp.sum(2 * safe - 1, axis=1) == config.target_m2), value, 0.)

    return amplitudes, dict(full_group_order=len(group), evaluation_images=len(representatives),
                            root_fold=use_fold, layers=sizes,
                            parameters_unchanged=True, physical_state_unchanged=True,
                            ordered_subtrees_only=True)


def make_clebsch_execution(
    config: LiteratureClebschTreeConfig,
    physical_order: Sequence[int],
    group_images: Sequence[Sequence[int]],
    *,
    kernel: str = 'legacy',
    orbit_chunk: int = 8,
) -> tuple[Callable, dict]:
    """Select an explicitly registered evaluator; default remains unchanged.

    Both backends represent the same layer-shared physical-spin ClebschTree,
    with fixed doubled ``q=2j`` labels and the complete coherent A1 projector.
    ``orbit_chunk`` remains the legacy batching setting, not a truncated orbit.
    """
    if kernel == 'legacy':
        evaluate, report = make_projected_clebsch(
            config, physical_order, group_images, orbit_chunk=orbit_chunk)
    elif kernel == 'orbit_dag':
        evaluate, report = make_orbit_dag_clebsch(config, physical_order, group_images)
    else:
        raise ValueError('execution kernel must be legacy or orbit_dag')
    return evaluate, dict(report, execution_kernel=kernel)
