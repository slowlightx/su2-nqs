"""Explicit host budgets for exact physical ClebschTree orbit sampling.

This is an opt-in adapter.  The existing 768-MiB default and sealed consumers
are unchanged.  Every doubled-m block and every q/multiplicity interference
is kept by :class:`RawTreeBornSampler`; this module only budgets its resident
tensors and reuses frozen samplers across independent caller-owned streams.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from ..models.triangular_clebsch_active_layout import mixed_active_layout
from ..models.triangular_mixed_clebsch import MixedClebschConfig
from ..su2.triangle_blocks import nine_site_output
from .full_support_proposal import draw_full_support_sz0
from .triangular_raw_born import make_triangular_raw_born_sampler


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name+" must be a positive integer")


def magnetic_cache_storage_plan(config: MixedClebschConfig,
        profile: Sequence[Mapping[int, int]], *, itemsize: int = 8,
        bank_count: int = 1) -> dict[str, Any]:
    """Count dense fixed-m magnetic tensors, without allocating them.

    The count is an upper bound without within-bank identical-tensor sharing.
    It excludes physical-leaf tables, QR copies and transient density work.
    ``itemsize`` must follow actual canonical tensor dtype, not an assumed
    real-valued physical state.
    """
    _positive(bank_count, 'bank_count')
    if itemsize not in (8, 16):
        raise ValueError('float64 or complex128 tensor storage required')
    layout = mixed_active_layout(config, profile)
    leaf_count = len(layout)+1
    widths = [{q:nine_site_output(q, config.multiplicity_cap).width
               for q in config.local_q} for _ in range(leaf_count)]
    def positions(blocks):
        return {m:sum(d for q,d in blocks.items() if abs(m) <= q)
                for m in sorted({m for q in blocks for m in range(-q, q+1, 2)})}
    nodes = []
    for index, row in enumerate(layout):
        left, right, parent = (positions(widths[row['left']]),
                               positions(widths[row['right']]), positions(row['widths']))
        entries = sum(dl*dr*parent[ml+mr] for ml,dl in left.items()
                      for mr,dr in right.items() if ml+mr in parent)
        nodes.append(dict(node=index, magnetic_entries=entries, bytes=entries*itemsize))
        widths.append(row['widths'])
    total = sum(row['bytes'] for row in nodes)
    return dict(per_bank_bytes=total, all_bank_bytes=bank_count*total,
        bank_count=int(bank_count), itemsize=itemsize, nodes=nodes,
        no_group_or_physical_channel_truncation=True,
        bound_without_identical_tensor_sharing=True,
        excludes_leaf_QR_and_transient_workspaces=True)


class BudgetedTriangularOrbitDrawer:
    """Frozen exact samplers, explicit per-bank/aggregate/host array budgets.

    Independent ``draw`` calls use separate seeds but reuse the same isometric
    banks and tensor caches.  No RNG is retained on this object.  In particular,
    selection and guard cannot accidentally continue one common stream.
    """
    def __init__(self, banks: Sequence[dict], runtime: dict, *,
                 per_bank_bytes: int, aggregate_bytes: int, host_array_bytes: int,
                 other_live_host_bytes: int = 0,
                 sampler_factory: Callable = make_triangular_raw_born_sampler):
        for name, value in (('per_bank_bytes', per_bank_bytes),
                            ('aggregate_bytes', aggregate_bytes),
                            ('host_array_bytes', host_array_bytes)):
            _positive(value, name)
        if (isinstance(other_live_host_bytes, bool)
                or not isinstance(other_live_host_bytes, Integral) or other_live_host_bytes < 0):
            raise ValueError('nonnegative other_live_host_bytes required')
        if not banks:
            raise ValueError('at least one frozen canonical bank required')
        itemsize = max(np.result_type(*[np.asarray(x).dtype for node in b['top']
                                        for x in node], np.float64).itemsize for b in banks)
        plan = magnetic_cache_storage_plan(runtime['cfg'], runtime['profile'],
                                           itemsize=itemsize, bank_count=len(banks))
        if plan['per_bank_bytes'] > per_bank_bytes:
            raise MemoryError('shape-planned single-bank magnetic cache exceeds explicit host budget')
        if plan['all_bank_bytes'] > aggregate_bytes:
            raise MemoryError('shape-planned aggregate magnetic caches exceed explicit host budget')
        if plan['all_bank_bytes']+other_live_host_bytes > host_array_bytes:
            raise MemoryError('magnetic caches plus other live host arrays exceed explicit budget')
        tick = perf_counter()
        self.samplers = []
        measured = 0
        for bank in banks:
            sampler = sampler_factory(bank, runtime['cfg'], active_top_widths=runtime['profile'],
                                       memory_limit_bytes=int(per_bank_bytes))
            measured += int(sampler.diagnostics['unique_magnetic_tensor_bytes'])
            if measured > aggregate_bytes or measured+other_live_host_bytes > host_array_bytes:
                raise MemoryError('actual aggregate magnetic cache exceeds explicit host budget')
            self.samplers.append(sampler)
        self.runtime = runtime
        self.diagnostics = dict(storage_plan=plan, measured_all_bank_tensor_bytes=measured,
            per_bank_limit_bytes=int(per_bank_bytes), aggregate_limit_bytes=int(aggregate_bytes),
            host_array_limit_bytes=int(host_array_bytes),
            other_live_host_array_bytes=int(other_live_host_bytes),
            constructor_seconds=perf_counter()-tick, frozen_sampler_reuse_across_separate_seeds=True,
            original_default_768MiB_unchanged=True,
            magnetic_labels_remain_coherent=True,
            diagnostics_are_array_budgets_not_total_RSS_certificates=True)

    def draw(self, count: int, seed: int, *, bridge: bool) -> tuple[np.ndarray, dict]:
        """Draw full physical IID orbit rows, optionally H-push/uniform mixed."""
        _positive(count, 'count')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError('explicit uint32 stream seed required')
        rt = self.runtime
        def base(number, base_seed):
            rng = np.random.default_rng(base_seed)
            before = deepcopy(rng.bit_generator.state)
            rows, latents, errors = [], [], []
            for _ in range(number):
                if 'remaining' in rt and rt['remaining']() < rt.get('sampling_reserve_seconds', 120.):
                    raise TimeoutError('complete physical IID panel cannot fit remaining reserved time')
                index = int(rng.integers(len(self.samplers)))
                sample = self.samplers[index].draw(rng)
                image = int(rng.integers(len(rt['images'])))
                row = sample.bits[rt['inverse'][image]].copy()
                pushed = bool(bridge and rng.random() < .5)
                bond = int(rng.choice(len(rt['edges']), p=rt['probability']))
                if pushed:
                    a,b = rt['edges'][bond]
                    row[a],row[b] = row[b],row[a]
                rows.append(row); latents.append((index, image, int(pushed), bond))
                errors.append(sample.log_probability_replay_error)
            return np.asarray(rows, np.int32), dict(seed=int(base_seed),
                rng_before=before, rng_after=rng.bit_generator.state, latents=latents,
                maximum_Born_replay_error=max(errors, default=0.))
        tick = perf_counter()
        if bridge:
            rows, report = draw_full_support_sz0(base, int(count), n_sites=self.runtime['cfg'].length**2,
                seed=int(seed), epsilon=rt.get('epsilon', .125))
        else:
            rows, report = base(int(count), int(seed))
        report.update(sampler_storage=self.diagnostics, draw_seconds=perf_counter()-tick,
                      independent_seed=int(seed), exact_physical_rows_not_hidden_paths=True)
        return rows, report


__all__ = ['magnetic_cache_storage_plan', 'BudgetedTriangularOrbitDrawer']
