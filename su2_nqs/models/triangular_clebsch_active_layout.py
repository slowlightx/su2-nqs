"""Exact fixed-bank execution layout for inactive upper multiplicity padding.

The local nine-site blocks retain their ORIGINAL widths. Upper edges can be
executed at smaller declared widths only when every retained parent output
has exactly zero coefficients on discarded child coordinates. This is an
audit optimization, not a variational truncation. Gradients opening discarded
native channels are NOT preserved by this representation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.quantum_numbers import coupling_range
from ..su2.triangle_blocks import nine_site_output
from .triangular_mixed_clebsch import MixedClebschConfig, _packed_tree_cg, mixed_clebsch_layout


def mixed_active_layout(config: MixedClebschConfig,
                        active_top_widths: Sequence[Mapping[int, int]]) -> tuple[dict, ...]:
    """Build exact charge buckets for fixed-bank upper widths (doubled q keys)."""
    original = mixed_clebsch_layout(config)
    if len(active_top_widths) != len(original):
        raise ValueError('one active-width mapping per upper merge is required')
    tree = triangular_schur_hierarchy(config.length)
    widths = [{q: nine_site_output(q, int(config.multiplicity_cap)).width for q in config.local_q}
              for _ in tree.blocks]
    result = []
    for index, (row, selected) in enumerate(zip(original, active_top_widths, strict=True)):
        if not selected:
            raise ValueError('each upper edge must keep a nonempty set of doubled-q sectors')
        out = {}
        for q, d in sorted(selected.items()):
            if (isinstance(q, bool) or not isinstance(q, Integral) or q not in row['widths']
                    or isinstance(d, bool) or not isinstance(d, Integral) or not 1 <= d <= row['widths'][q]):
                raise ValueError('active doubled-q widths must be positive native-width subsets')
            out[int(q)] = int(d)
        if index == len(original)-1 and out != {0: 1}:
            raise ValueError('fixed singlet root must retain exactly q=0 width one')
        left, right = row['left'], row['right']
        buckets = {}
        for q, d in out.items():
            pairs = [(ql, qr) for ql in widths[left] for qr in widths[right]
                     if q in coupling_range(ql, qr)]
            capacity = sum(widths[left][ql]*widths[right][qr] for ql, qr in pairs)
            if d > capacity:
                raise ValueError('active child product cannot support requested output width')
            for ql, qr in pairs:
                shape = (d, widths[left][ql], widths[right][qr])
                buckets.setdefault(shape, []).append((ql, qr, q))
        groups = []
        for shape, triples in sorted(buckets.items()):
            keys = tuple(triples)
            # The existing CG cache requires positive table spans. It still
            # contains the exact q=0 row, so span one safely supports a bucket
            # containing only singlet inputs without introducing any channel.
            sl = max(1, max(q[0] for q in keys)); sr = max(1, max(q[1] for q in keys))
            groups.append(dict(shape=(len(keys), *shape), keys=keys, left_span=sl, right_span=sr,
                cg=_packed_tree_cg(sl, sr, keys),
                output_groups=tuple((q, tuple(i for i, key in enumerate(keys) if key[2] == q))
                                    for q in out if any(key[2] == q for key in keys))))
        result.append(dict(left=left, right=right, parent=row['parent'], widths=out, groups=tuple(groups)))
        widths.append(out)
    return tuple(result)


def repack_active_clebsch_bank(bank: dict, config: MixedClebschConfig,
                                active_top_widths: Sequence[Mapping[int, int]]) -> dict:
    """Prove exact inactive-parent closure and repack an immutable native bank.

    No magnitude tolerance is used: even tiny nonzero discarded couplings are
    rejected. Nonzero INACTIVE output columns are allowed because all active
    ancestor maps, including the root, are checked to ignore them exactly.
    The returned bank is only for fixed-state amplitudes, swaps, and H audits.
    """
    original = mixed_clebsch_layout(config)
    active = mixed_active_layout(config, active_top_widths)
    tree = triangular_schur_hierarchy(config.length)
    if len(bank['local']) != len(tree.blocks) or len(bank['top']) != len(original):
        raise ValueError('native bank does not match the declared hierarchy')
    widths = [{q: nine_site_output(q, int(config.multiplicity_cap)).width for q in config.local_q}
              for _ in tree.blocks]
    for node in bank['local']:
        if len(node) != len(config.local_q):
            raise ValueError('local charge blocks must remain unchanged')
        for q, value in zip(config.local_q, node, strict=True):
            if np.asarray(value).ndim != 2 or np.asarray(value).shape[1] != widths[0][q]:
                raise ValueError('local multiplicity widths must remain unchanged')
    top = []
    for old, new, values in zip(original, active, bank['top'], strict=True):
        if len(values) != len(old['groups']):
            raise ValueError('native upper group count mismatch')
        lookup = {}
        for group, weights in zip(old['groups'], values, strict=True):
            weights = np.asarray(weights)
            if weights.shape != group['shape'] or not np.all(np.isfinite(weights)):
                raise ValueError('native upper weights must have the declared finite shape')
            for key, value in zip(group['keys'], weights, strict=True):
                lookup[key] = value
                ql, qr, qo = key
                do = new['widths'].get(qo, 0)
                dl = widths[new['left']].get(ql, 0); dr = widths[new['right']].get(qr, 0)
                if do and (np.any(value[:do, dl:, :] != 0.) or np.any(value[:do, :, dr:] != 0.)):
                    raise ValueError('active parent output has nonzero discarded-child paths')
        groups = []
        for group in new['groups']:
            groups.append(np.stack([lookup[ql, qr, qo][:new['widths'][qo],
                :widths[new['left']][ql], :widths[new['right']][qr]] for ql, qr, qo in group['keys']]))
        top.append(tuple(groups)); widths.append(new['widths'])
    return dict(local=bank['local'], top=tuple(top))
