"""Exact two-filter physical-carrier contraction with one final Reynolds sum.

Three DISJOINT ordered four-site cells A/B/C are used. K1 controls the active
B multiplicity from spectator A; K2 controls active C from spectator B. The
physical product is K2 K1. The intermediate B configuration is coherently
summed, not read from the original output configuration. Each cell's magnetic
projection is preserved, giving at most6^3=216 final inputs, not36^2 paths.

Both60-coordinate filters remain dynamic JAX arguments. Carrier queries may
be frozen for a fixed carrier, but no3600-column response bank or full ViT is
claimed. Only P_A1 K2 K1 C and the distinct no-cascade control
P_A1(K1+K2-I)C are implemented; a projector is NEVER inserted between gates.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import validated_permutation_group
from ..models.conditional_carrier_resonance import conditional_filter_matrix


@dataclass(frozen=True)
class OverlappingCarrierQueryPlan:
    """Unique fixed-carrier inputs and exact final-group reconstruction maps."""

    configurations: np.ndarray
    output_configurations: np.ndarray
    cells: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    group_images: np.ndarray
    unique_local_rows: np.ndarray
    local_row_indices: np.ndarray
    connection_indices: np.ndarray
    carrier_indices: np.ndarray
    projected_carrier_indices: np.ndarray
    diagnostics: dict[str, int | bool | str]


@lru_cache(maxsize=1)
def _cell_alternatives() -> tuple[np.ndarray, np.ndarray]:
    alternatives = np.zeros((16, 6), dtype=np.int32)
    valid = np.zeros((16, 6), dtype=bool)
    for row in range(16):
        columns = [col for col in range(16) if col.bit_count() == row.bit_count()]
        alternatives[row, :len(columns)] = columns; valid[row, :len(columns)] = True
    alternatives.setflags(write=False); valid.setflags(write=False)
    return alternatives, valid


@lru_cache(maxsize=4096)
def local_overlap_connections(row: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return216 packed local inputs, validity mask, and identity slot."""
    if isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral) or not 0 <= row < 4096:
        raise ValueError('row must be a12-bit integer configuration')
    alternatives, valid = _cell_alternatives(); a, b, c = row//256, (row//16)%16, row%16
    columns = (256*alternatives[a, :, None, None]+16*alternatives[b, None, :, None]+
               alternatives[c, None, None, :]).reshape(216)
    mask = (valid[a, :, None, None]&valid[b, None, :, None]&valid[c, None, None, :]).reshape(216)
    identity = int(np.flatnonzero((columns == row)&mask)[0])
    columns.setflags(write=False); mask.setflags(write=False)
    return columns, mask, identity


def overlap_filter_rows(
    first_parameters: jax.Array, second_parameters: jax.Array,
    local_rows: jax.Array, *, mode: str = 'cascade',
) -> jax.Array:
    """Return dynamic216-entry rows of K2 K1 or K1+K2-I.

    Local output index is256*a+16*b+c. Intermediate b_mid is summed in
    K2[(c,b),(c_in,b_mid)] K1[(b_mid,a),(b_in,a_in)]. Every G/exponential and
    both factors participate in autodiff, including at identity.
    """
    if mode not in ('cascade', 'no_cascade'): raise ValueError('unknown composition mode')
    rows = jnp.asarray(local_rows)
    if rows.ndim != 1 or rows.size < 1 or not jnp.issubdtype(rows.dtype, jnp.integer):
        raise ValueError('local_rows must be a nonempty integer vector')
    if not isinstance(rows, jax.core.Tracer) and (np.any(np.asarray(rows) < 0) or np.any(np.asarray(rows) >= 4096)):
        raise ValueError('local_rows must be12-bit configurations')
    alternatives, valid = map(jnp.asarray, _cell_alternatives())
    a, b, c = rows//256, (rows//16)%16, rows%16
    av, bv, cv = alternatives[a], alternatives[b], alternatives[c]
    ma, mb, mc = valid[a], valid[b], valid[c]
    k1 = conditional_filter_matrix(first_parameters)
    k2 = conditional_filter_matrix(second_parameters)
    if mode == 'cascade':
        first = k1[16*bv[:, :, None, None]+a[:, None, None, None],
                   16*bv[:, None, :, None]+av[:, None, None, :]]
        second = k2[(16*c+b)[:, None, None], 16*cv[:, :, None]+bv[:, None, :]]
        second = second*mb[:, None, :]
        result = jnp.einsum('nct,ntba->nabc', second, first)
    else:
        first = k1[(16*b+a)[:, None, None], 16*bv[:, :, None]+av[:, None, :]]
        second = k2[(16*c+b)[:, None, None], 16*cv[:, :, None]+bv[:, None, :]]
        da, db, dc = av == a[:, None], bv == b[:, None], cv == c[:, None]
        result = (jnp.swapaxes(first, 1, 2)[:, :, :, None]*dc[:, None, None, :]
                  +da[:, :, None, None]*jnp.swapaxes(second, 1, 2)[:, None, :, :]
                  -da[:, :, None, None]*db[:, None, :, None]*dc[:, None, None, :])
    mask = ma[:, :, None, None]&mb[:, None, :, None]&mc[:, None, None, :]
    return jnp.where(mask, result, 0.).reshape(len(rows), 216)


def _positive_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def compile_overlapping_carrier_queries(
    configurations: np.ndarray, group_images: Sequence[Sequence[int]], *,
    cells: Sequence[Sequence[int]] = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)),
    max_unique_queries: int = 500_000,
    max_terminal_requests: int = 10_000_000,
    max_plan_bytes: int = 512*2**20,
) -> OverlappingCarrierQueryPlan:
    """Stream valid local inputs and deduplicate across ALL outputs/images.

    Pass the union of all desired H-connected output centers in ONE call to
    share carrier inputs across Hamiltonian bonds. The compiler itself does
    not construct H or treat a center-H stencil as an amplitude query. Caps
    are checked before carrier evaluation; exceeding one fails closed.
    ``max_plan_bytes`` limits retained numeric arrays, not Python-map RSS.
    A separate process RSS cap is still required for a physical cost probe.
    """
    max_unique_queries = _positive_limit(max_unique_queries, 'max_unique_queries')
    max_terminal_requests = _positive_limit(max_terminal_requests, 'max_terminal_requests')
    max_plan_bytes = _positive_limit(max_plan_bytes, 'max_plan_bytes')
    states = np.asarray(configurations)
    if (states.ndim != 2 or len(states) < 1 or states.shape[1] < 12 or
        states.dtype.kind not in 'biu' or np.any((states != 0)&(states != 1))):
        raise ValueError('expected a nonempty integer bit matrix with N>=12')
    states = states.astype(np.int8, copy=True); count, n = states.shape
    checked = tuple(tuple(cell) for cell in cells)
    if (len(checked) != 3 or any(len(cell) != 4 for cell in checked) or
        any(isinstance(site, (bool, np.bool_)) or not isinstance(site, Integral) or not 0 <= site < n
            for cell in checked for site in cell) or len(set(site for cell in checked for site in cell)) != 12):
        raise ValueError('three disjoint ordered4-site cells are required')
    checked = tuple(tuple(int(i) for i in cell) for cell in checked)
    group = np.asarray(validated_permutation_group(group_images), dtype=np.int32)
    if group.shape[1] != n: raise ValueError('group/site count mismatch')
    # Never allocate(output,group,216,N) or raw36² path arrays.
    base_bytes = count*len(group)*216*4+count*len(group)*4*3+states.nbytes+group.nbytes
    if base_bytes > max_plan_bytes: raise MemoryError('numeric query-plan budget exhausted before allocation')
    connections = np.zeros((count, len(group), 216), dtype=np.int32)
    local_rows = np.empty((count, len(group)), dtype=np.int32)
    projected = np.empty((count, len(group)), dtype=np.int32)
    carrier_indices = np.empty(count, dtype=np.int32)
    support = np.asarray(sum(checked, ()), dtype=np.int32)
    powers = 1 << np.arange(11, -1, -1)
    mapping: dict[bytes, int] = {}; keys: list[bytes] = []
    raw = 0; maximum_connections = 0
    def identify(packed: bytes) -> int:
        old = mapping.get(packed)
        if old is not None: return old
        index = len(keys)
        if index >= max_unique_queries: raise MemoryError('unique carrier-query budget exhausted')
        if base_bytes+(index+1)*n > max_plan_bytes: raise MemoryError('retained numeric query-plan budget exhausted')
        mapping[packed] = index; keys.append(packed); return index
    for sample, state in enumerate(states):
        carrier_indices[sample] = identify(np.packbits(state, bitorder='big').tobytes())
        for image_index, image in enumerate(group):
            target = state[image]
            row = int(target[support]@powers); local_rows[sample, image_index] = row
            columns, valid, identity = local_overlap_connections(row)
            active_slots = np.flatnonzero(valid); raw += len(active_slots)
            maximum_connections = max(maximum_connections, len(active_slots))
            if raw > max_terminal_requests: raise MemoryError('terminal carrier-request budget exhausted')
            inputs = np.broadcast_to(target, (len(active_slots), n)).copy()
            inputs[:, support] = ((columns[valid, None] >> np.arange(11, -1, -1)) & 1).astype(np.int8)
            encoded = np.packbits(inputs, axis=1, bitorder='big')
            for slot, packed in zip(active_slots, encoded, strict=True):
                connections[sample, image_index, slot] = identify(packed.tobytes())
            projected[sample, image_index] = connections[sample, image_index, identity]
    width = (n+7)//8
    unique = np.unpackbits(np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(-1, width),
                          axis=1, count=n, bitorder='big').astype(np.int8)
    unique_rows, row_inverse = np.unique(local_rows, return_inverse=True)
    row_inverse = row_inverse.reshape(local_rows.shape).astype(np.int32)
    arrays = (unique, states, group, unique_rows, row_inverse, connections, carrier_indices, projected)
    numeric_bytes = sum(array.nbytes for array in arrays)
    if numeric_bytes > max_plan_bytes: raise MemoryError('retained numeric plan exceeds registered cap')
    diagnostics = dict(output_count=count, n_sites=n, group_order=len(group),
        padded_terminal_requests=count*len(group)*216, valid_terminal_requests=raw,
        raw_total_requests=raw+count, unique_carrier_queries=len(unique),
        duplicate_requests_removed=raw+count-len(unique),
        maximum_local_connections=maximum_connections, unique_local_rows=len(unique_rows),
        numeric_plan_bytes=numeric_bytes, max_unique_queries=max_unique_queries,
        max_terminal_requests=max_terminal_requests, max_plan_bytes=max_plan_bytes,
        complete_group=True, projector_only_after_both_gates=True,
        intermediate_shared_cell_summed_coherently=True,
        magnetic_channels_truncated=False, new_H_calls=0,
        carrier_symmetry_used_for_deduplication=False,
        interpretation='Dynamic two-filter composition, not a frozen3600-response bank or a complete ViT.')
    for array in arrays: array.setflags(write=False)
    return OverlappingCarrierQueryPlan(unique, states, checked, group, unique_rows, row_inverse,
                                       connections, carrier_indices, projected, diagnostics)


def evaluate_overlapping_carrier_queries(
    plan: OverlappingCarrierQueryPlan, carrier_fn: Callable[[np.ndarray], np.ndarray], *, batch_size: int = 256,
) -> np.ndarray:
    """Evaluate each unique fixed-carrier configuration once, preserving nodes."""
    batch_size = _positive_limit(batch_size, 'batch_size'); parts = []
    for start in range(0, len(plan.configurations), batch_size):
        x = plan.configurations[start:start+batch_size]
        amplitudes = np.asarray(carrier_fn(x), dtype=np.complex128)
        if amplitudes.shape != (len(x),) or not np.isfinite(amplitudes).all():
            raise FloatingPointError('invalid carrier callback output')
        parts.append(amplitudes)
    values = np.concatenate(parts); values.setflags(write=False)
    return values


def overlapping_carrier_amplitudes(
    first_parameters: jax.Array, second_parameters: jax.Array,
    plan: OverlappingCarrierQueryPlan, unique_carrier_amplitudes: jax.Array,
    *, mode: str = 'cascade',
) -> jax.Array:
    """Pure-JAX full-group amplitudes; both local filters remain trainable."""
    amplitudes = jnp.asarray(unique_carrier_amplitudes)
    if amplitudes.shape != (len(plan.configurations),): raise ValueError('wrong unique-carrier amplitude count')
    weights = overlap_filter_rows(first_parameters, second_parameters, jnp.asarray(plan.unique_local_rows), mode=mode)
    values = amplitudes[jnp.asarray(plan.connection_indices)]
    selected = weights[jnp.asarray(plan.local_row_indices)]
    return jnp.mean(jnp.sum(selected*values, axis=-1), axis=1)


__all__ = ['OverlappingCarrierQueryPlan', 'local_overlap_connections', 'overlap_filter_rows',
    'compile_overlapping_carrier_queries', 'evaluate_overlapping_carrier_queries', 'overlapping_carrier_amplitudes']
