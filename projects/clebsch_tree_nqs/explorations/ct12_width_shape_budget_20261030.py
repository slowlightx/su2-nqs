"""Allocation-free doubled-q CT12 width alternatives; no state truncation.

Only profile shapes and real float64 storage are counted. A hypothetical
local20/upper28 layout is distinguished from executable existing-API profiles.
No checkpoint, model API, optimizer, Hamiltonian or training state is changed.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy


def width_shape_plan(local_cap: int, profile: tuple[dict[int, int], ...], *,
                     runtime_cap: int) -> dict:
    """Count legal q=2j blocks without allocating a tensor or CG table."""
    if local_cap < 9 or runtime_cap < local_cap:
        raise ValueError('legal local cap and at-least-as-wide runtime required')
    tree = triangular_schur_hierarchy(12)
    if len(profile) != len(tree.binary_merges):
        raise ValueError('one explicit width map per upper node required')
    physical = {q: comb(9, (9-q)//2)- (comb(9, (9-q)//2-1) if q < 9 else 0)
                for q in range(1, 10, 2)}
    local = {q: min(local_cap, d) for q, d in physical.items()}
    runtime_local = {q: min(runtime_cap, d) for q, d in physical.items()}
    widths = [local.copy() for _ in tree.blocks]
    sizes = [9]*len(tree.blocks)
    rows = []
    total = len(tree.blocks)*sum(physical[q]*d for q, d in local.items())
    local_parameters = total
    by_size: dict[int, int] = {}
    for index, ((left, right, parent), requested) in enumerate(zip(tree.binary_merges, profile, strict=True)):
        size = sizes[left]+sizes[right]
        if index == len(profile)-1 and requested != {0: 1}:
            raise ValueError('the singlet root must be q=0 width one')
        count = 0
        for q, d in requested.items():
            if not (q in range(0, 11, 2) and 1 <= d <= runtime_cap):
                raise ValueError('explicit retained doubled-q width outside runtime cap')
            capacity = sum(dl*dr for ql, dl in widths[left].items()
                           for qr, dr in widths[right].items()
                           if abs(ql-qr) <= q <= ql+qr and (ql+qr-q) % 2 == 0)
            if d > capacity:
                raise ValueError('requested output exceeds physical child-product capacity')
            count += d*capacity
        rows.append(dict(index=index, subtree_sites=size, left=left, right=right, parent=parent,
                         widths=requested, parameters=count))
        widths.append(requested.copy()); sizes.append(size)
        by_size[size] = by_size.get(size, 0)+count
        total += count
    executable = local == runtime_local
    return dict(local_cap=local_cap, local_widths=local, local_parameters=local_parameters,
                runtime_cap=runtime_cap, existing_API_executable=executable,
                missing_interface=None if executable else 'separate local cap from native upper cap; do not pass this shape to current decoder',
                active_profile=list(profile), upper_rows=rows, upper_parameters_by_subtree_size=by_size,
                parameters=total, parameter_GiB=total*8/1024**3,
                parameters_plus_two_moments_GiB=total*24/1024**3,
                real_J_1024_GiB=total*8*1024/1024**3,
                real_J_2048_GiB=total*8*2048/1024**3,
                six_parameter_vectors_GiB=total*8*6/1024**3,
                no_JVP_or_runtime_memory_measurement=True)


def make_width_profiles(old: tuple[dict[int, int], ...]) -> dict[str, tuple[int, int, tuple[dict[int, int], ...]]]:
    """Return explicit immutable alternatives; original profile is never replaced."""
    tree = triangular_schur_hierarchy(12)
    sizes = [9]*len(tree.blocks); subtree = []
    for left, right, _ in tree.binary_merges:
        sizes.append(sizes[left]+sizes[right]); subtree.append(sizes[-1])
    def uniform(width: int) -> tuple[dict[int, int], ...]:
        return tuple({0: 1} if n == 144 else {q: width for q in range(0, 11, 2)} for n in subtree)
    layered = tuple({0: 1} if n == 144 else {q: (20 if n <= 36 else 28) for q in range(0, 11, 2)}
                    for n in subtree)
    old_lower_wide_72 = tuple({q: 28 for q in range(0, 11, 2)} if n == 72 else dict(old[i])
                             for i, n in enumerate(subtree))
    return {'parent_global45_original_profile': (20, 20, old),
            'local20_upper8_shape_only_not_parent_preserving': (20, 20, uniform(8)),
            'local20_upper28_hypothetical_interface_gap': (20, 28, uniform(28)),
            'local20_all_upper20': (20, 20, uniform(20)),
            'local28_original_lower_profile_upper72_at28': (28, 28, old_lower_wide_72),
            'local28_upper18_36_at20_upper72_at28': (28, 28, layered),
            'local28_all_upper20': (28, 28, uniform(20)),
            'local28_all_upper28': (28, 28, uniform(28))}


def run(registration: Path, output: Path) -> dict:
    """Save allocation-free plans, explicitly flagging truncation/interface gaps."""
    if output.exists():
        raise FileExistsError('preserve any previous width budget')
    plan = json.loads(registration.read_text())['plan']
    old = tuple({int(q): int(d) for q, d in row.items()} for row in plan['active_profile'])
    result = dict(task='CT12_allocation_free_shape_alternatives', maximum_q=10, root={0: 1},
                  source_registration=str(registration), no_model_API_changes=True,
                  plans={})
    for name, (localcap, runtime, profile) in make_width_profiles(old).items():
        value = width_shape_plan(localcap, profile, runtime_cap=runtime)
        value['same_parent_state_inclusion_possible'] = all(
            profile[i].get(q, 0) >= d for i, row in enumerate(old) for q, d in row.items())
        if not value['same_parent_state_inclusion_possible']:
            value['warning'] = 'would discard inherited active axes; not a same-state restart'
        result['plans'][name] = value
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); run(args.registration, args.output)
