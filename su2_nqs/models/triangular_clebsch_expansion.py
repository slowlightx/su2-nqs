"""Exact, first-order-live width inclusion of the physical triangular tree.

Only retained multiplicity widths change.  The triangle cover, doubled charge
cutoff, fixed CG tables, optional six-site binary rank and complete A1 projector
are unchanged.  New child features are nonzero, but all inherited parent outputs
initially annihilate them.  Consequently the wider model starts at precisely
the old physical state, while new downstream couplings have a live tangent.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .triangular_arity_clebsch import ArityMode, binary_intermediate_widths
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.triangle_blocks import nine_site_output, triangle_block_tables


def _top_blocks(params: Mapping[str, Any], config: MixedClebschConfig) -> tuple[dict, ...]:
    layout = mixed_clebsch_layout(config)
    if len(params['top']) != len(layout):
        raise ValueError('top parameters disagree with the declared hierarchy')
    result = []
    for banks, row in zip(params['top'], layout, strict=True):
        if len(banks) != len(row['groups']):
            raise ValueError('top charge buckets disagree with the declared hierarchy')
        unpacked = {}
        for weights, group in zip(banks, row['groups'], strict=True):
            if weights.shape != group['shape']:
                raise ValueError('top reduced tensor has an invalid shape')
            unpacked.update(zip(group['keys'], weights, strict=True))
        result.append(unpacked)
    return tuple(result)


def initialize_live_triangular_expansion(
    source_config: MixedClebschConfig,
    source: Mapping[str, Any],
    target_config: MixedClebschConfig,
    key: jax.Array,
    *, mode: ArityMode, binary_rank: int = 8, new_feature_scale: float = 0.2,
) -> tuple[dict, dict]:
    """Widen a native binary/joint tree, preserving every physical amplitude.

    All spin labels are doubled integers.  Binary ``six`` maps and their rank
    are retained verbatim; the genuine joint three-input tensor is not
    refactorized.  Old top tensors are copied by ``(q_left,q_right,q_out)``,
    independently of width-dependent bucket grouping.  New local output
    columns and new top output rows are random; inherited top output rows
    have zero coefficients on every new child input.

    A fresh optimizer registration is required in the larger parameter space.
    This function neither transfers optimizer moments nor promises that every
    added coordinate has nonzero gradient immediately.  At least the new
    downstream gates can be first-order live, which callers must verify on
    the actual projected parent before training.
    """
    if mode not in ('binary_truncated', 'ternary_joint'):
        raise ValueError('native binary_truncated or ternary_joint mode required')
    if (source_config.length != target_config.length
            or source_config.maximum_q != target_config.maximum_q):
        raise ValueError('live width expansion requires the same cover and doubled cutoff')
    if target_config.multiplicity_cap <= source_config.multiplicity_cap:
        raise ValueError('live width expansion requires a strictly larger multiplicity cap')
    if not np.isfinite(new_feature_scale) or new_feature_scale <= 0:
        raise ValueError('new_feature_scale must be positive and finite')
    six_widths = binary_intermediate_widths(binary_rank)
    tree = triangular_schur_hierarchy(source_config.length)
    if len(source['local']) != len(tree.blocks):
        raise ValueError('local parameters disagree with the declared cover')
    if set(source) != ({'local', 'top', 'six'} if mode == 'binary_truncated' else {'local', 'top'}):
        raise ValueError('native parameter keys disagree with the declared arity')
    if mode == 'binary_truncated':
        if len(source['six']) != len(tree.blocks):
            raise ValueError('six-site maps disagree with the declared cover')
        tables = triangle_block_tables()
        for node in source['six']:
            if len(node) != 4 or any(w.shape != (tables['six'][q].shape[1], six_widths[q])
                                      for q, w in zip((0, 2, 4, 6), node, strict=True)):
                raise ValueError('six-site maps disagree with the retained binary rank')
    old_top = _top_blocks(source, source_config)
    new_layout = mixed_clebsch_layout(target_config)
    if len(old_top) != len(new_layout):
        raise ValueError('live expansion cannot change the tree topology')
    local_keys = len(tree.blocks)*len(source_config.local_q)
    top_keys = sum(len(g['keys']) for row in new_layout for g in row['groups'])
    keys = iter(jax.random.split(key, local_keys+top_keys))
    local, local_rows = [], []
    for node in source['local']:
        if len(node) != len(source_config.local_q):
            raise ValueError('local doubled-charge blocks disagree with the config')
        values = []
        for q, old in zip(source_config.local_q, node, strict=True):
            old_width = nine_site_output(q, int(source_config.multiplicity_cap)).width
            new_width = nine_site_output(q, int(target_config.multiplicity_cap)).width
            if old.ndim != 2 or old.shape[1] != old_width:
                raise ValueError('local reduced tensor has an invalid output width')
            if mode == 'ternary_joint':
                input_size = triangle_block_tables()['nine'][q].shape[1]
            else:
                input_size = sum(six_widths[q6]*triangle_block_tables()['three'][q3].shape[1]
                                 for q6 in (0, 2, 4, 6) for q3 in (1, 3)
                                 if abs(q6-q3) <= q <= q6+q3)
            if old.shape[0] != input_size:
                raise ValueError('local reduced tensor has an invalid input width')
            fresh = jax.random.normal(next(keys), (input_size, new_width), dtype=old.dtype)
            fresh = fresh*(new_feature_scale/np.sqrt(input_size))
            values.append(fresh.at[:, :old_width].set(old))
            local_rows.append(dict(q=q, old_width=old_width, new_width=new_width))
        local.append(tuple(values))
    top, top_rows = [], []
    for old, row in zip(old_top, new_layout, strict=True):
        if set(old) != {key for group in row['groups'] for key in group['keys']}:
            raise ValueError('live expansion cannot add or remove doubled-charge blocks')
        groups = []
        for group in row['groups']:
            pieces = []
            for qkey in group['keys']:
                previous = old[qkey]
                shape = group['shape'][1:]
                if any(a > b for a, b in zip(previous.shape, shape, strict=True)):
                    raise ValueError('live expansion cannot truncate a reduced tensor')
                fresh = jax.random.normal(next(keys), shape, dtype=previous.dtype)
                fresh = fresh*(new_feature_scale/np.sqrt(shape[1]*shape[2]))
                # Every old parent row excludes every new child feature.
                block = jnp.zeros(shape, dtype=previous.dtype)
                block = block.at[tuple(slice(s) for s in previous.shape)].set(previous)
                block = block.at[previous.shape[0]:].set(fresh[previous.shape[0]:])
                pieces.append(block)
                top_rows.append(dict(doubled_channels=list(qkey), old_shape=list(previous.shape),
                                     new_shape=list(shape)))
            groups.append(jnp.stack(pieces))
        top.append(tuple(groups))
    result = dict(local=tuple(local), top=tuple(top))
    if mode == 'binary_truncated':
        result['six'] = tuple(tuple(w for w in node) for node in source['six'])
    return result, dict(
        exact_same_physical_state=True, same_cover=True, same_doubled_cutoff=True,
        complete_A1_preserved_by_linearity=True, mode=mode,
        binary_rank=binary_rank if mode == 'binary_truncated' else None,
        source_width=int(source_config.multiplicity_cap), target_width=int(target_config.multiplicity_cap),
        source_parameters=sum(int(v.size) for v in jax.tree.leaves(source)),
        target_parameters=sum(int(v.size) for v in jax.tree.leaves(result)),
        new_feature_scale=float(new_feature_scale), new_optimizer_registration_required=True,
        tangent_gain_to_be_measured_not_assumed=True, local=local_rows, top=top_rows,
    )
