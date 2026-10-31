"""Exact A1-carrier pullbacks; retain the complete dressed-state projection.

For an invariant carrier C and image g, C(localchange(x[g])) equals C of
the same local changes placed on g[support] in the ORIGINAL x. This shares
carrier inputs across images and H connections without equating the different
operator weights. It is invalid for a non-A1 carrier or individual pi-flux
boundary components. No claim that the dressed state has a three-term orbit.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from numbers import Integral

import numpy as np

from ..lattices.spatial_symmetry import validated_permutation_group
from .overlapping_carrier_queries import OverlappingCarrierQueryPlan, local_overlap_connections


@lru_cache(maxsize=1)
def _local_tables() -> tuple[np.ndarray, np.ndarray]:
    columns = np.stack([local_overlap_connections(row)[0] for row in range(4096)])
    valid = np.stack([local_overlap_connections(row)[1] for row in range(4096)])
    columns.setflags(write=False); valid.setflags(write=False)
    return columns, valid


def compile_a1_overlapping_carrier_queries(
    configurations: np.ndarray, group_images: Sequence[Sequence[int]], *,
    cells: Sequence[Sequence[int]], carrier_sector: str,
    max_unique_queries: int = 4_000_000,
    max_terminal_requests: int = 25_000_000,
    max_plan_bytes: int = 768*2**20,
) -> OverlappingCarrierQueryPlan:
    """Compile all images, pulling only C's arguments back to physical centers.

    Explicit ``carrier_sector='Gamma-A1'`` acknowledges a mathematical
    precondition, not a numerical symmetry certificate supplied by this
    compiler. The caller must establish it independently. N<=64 is an exact
    packed-key implementation limit, not a Hilbert-space truncation.
    ``max_plan_bytes`` bounds retained arrays; temporary sorting buffers and
    JAX arrays require a separate process-memory budget.
    """
    if carrier_sector != 'Gamma-A1':
        raise ValueError('pullback reuse requires an independently established exact Gamma-A1 carrier')
    for value in (max_unique_queries, max_terminal_requests, max_plan_bytes):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError('resource caps must be positive integers')
    states = np.asarray(configurations)
    if (states.ndim != 2 or not len(states) or not 12 <= states.shape[1] <= 64
        or states.dtype.kind not in 'biu' or np.any((states != 0)&(states != 1))):
        raise ValueError('nonempty physical bit matrix with 12<=N<=64 required')
    states = states.astype(np.int8, copy=True)
    count, n = states.shape
    checked = tuple(tuple(cell) for cell in cells)
    if (len(checked) != 3 or any(len(cell) != 4 for cell in checked) or
        any(isinstance(i, (bool,np.bool_)) or not isinstance(i, Integral) or not 0<=i<n
            for cell in checked for i in cell) or len(set(sum(checked, ()))) != 12):
        raise ValueError('three disjoint ordered four-site cells required')
    checked = tuple(tuple(int(i) for i in cell) for cell in checked)
    group = np.asarray(validated_permutation_group(group_images), dtype=np.int32)
    if group.shape[1] != n:
        raise ValueError('space group/site count mismatch')
    # Each final index occupies four bytes. Preflight before the large allocation.
    shape = (count, len(group), 216)
    base_bytes = int(np.prod(shape))*4 + count*len(group)*12 + states.nbytes + group.nbytes
    if base_bytes > max_plan_bytes:
        raise MemoryError('numeric query-plan budget exhausted before allocation')
    support = np.asarray(sum(checked, ()), dtype=np.int32)
    images = group[:, support]
    powers = np.left_shift(np.uint64(1), np.arange(n-1, -1, -1, dtype=np.uint64))
    center_keys = states.astype(np.uint64) @ powers
    image_powers = powers[images]
    support_masks = np.sum(image_powers, axis=1, dtype=np.uint64)
    rows = (states[:, images] @ (1 << np.arange(11, -1, -1))).astype(np.int32)
    columns_table, masks_table = _local_tables()
    masks = masks_table[rows]
    total = int(masks.sum())
    if total > max_terminal_requests:
        raise MemoryError('terminal carrier-request budget exhausted')
    # Vectorize the 12 local bit writes; no output*group*216*N bit tensor.
    targets = np.broadcast_to(center_keys[:,None,None] & ~support_masks[None,:,None], shape).copy()
    columns = columns_table[rows]
    for k in range(12):
        targets |= (((columns >> (11-k)) & 1).astype(np.uint64) * image_powers[None,:,k,None])
    flat = np.concatenate((center_keys, targets[masks]))
    del targets, columns
    keys, inverse = np.unique(flat, return_inverse=True)
    del flat
    if len(keys) > max_unique_queries:
        raise MemoryError('unique carrier-query budget exhausted')
    # Byte unpacking avoids an additional Q x N array of eight-byte integers.
    packed = keys.astype('>u8').view(np.uint8).reshape(-1,8)
    unique = np.unpackbits(packed,axis=1,bitorder='big')[:,-n:].astype(np.int8)
    connections = np.zeros(shape, dtype=np.int32)
    connections[masks] = inverse[count:].astype(np.int32)
    carrier_indices = inverse[:count].astype(np.int32)
    # For an A1 carrier its original and group-projected center are identical.
    projected = np.broadcast_to(carrier_indices[:,None], (count,len(group))).copy()
    unique_rows, row_inverse = np.unique(rows, return_inverse=True)
    row_inverse = row_inverse.reshape(rows.shape).astype(np.int32)
    arrays = (unique, states, group, unique_rows, row_inverse, connections, carrier_indices, projected)
    numeric_bytes = sum(a.nbytes for a in arrays)
    if numeric_bytes > max_plan_bytes:
        raise MemoryError('retained numeric plan exceeds registered cap')
    diagnostics = dict(output_count=count,n_sites=n,group_order=len(group),
        padded_terminal_requests=int(np.prod(shape)), valid_terminal_requests=total,
        raw_total_requests=total+count, unique_carrier_queries=len(unique),
        duplicate_requests_removed=total+count-len(unique), maximum_local_connections=int(masks.sum(axis=-1).max()),
        unique_local_rows=len(unique_rows),numeric_plan_bytes=numeric_bytes,
        max_unique_queries=int(max_unique_queries),max_terminal_requests=int(max_terminal_requests),
        max_plan_bytes=int(max_plan_bytes), complete_group=True,projector_only_after_both_gates=True,
        intermediate_shared_cell_summed_coherently=True,magnetic_channels_truncated=False,
        carrier_symmetry_used_for_deduplication=True,carrier_sector_precondition='Gamma-A1',new_H_calls=0,
        interpretation='All group weights retained; only invariant carrier arguments are pulled back.')
    for array in arrays:
        array.setflags(write=False)
    return OverlappingCarrierQueryPlan(unique,states,checked,group,unique_rows,row_inverse,connections,
                                      carrier_indices,projected,diagnostics)


__all__ = ['compile_a1_overlapping_carrier_queries']
