"""Deduplicated physical-carrier queries for a local controlled Schur filter.

Compile every image in a validated finite spatial group, then every allowed
local eight-site input.  Only identical complete spin configurations are
deduplicated: no group images, q sectors, amplitudes or small coefficients
are discarded.  Carrier nodes remain exact zeros.  This is a host reference
compiler for bounded cost probes, not a GPU-native sampler or H evaluator.

For each output x, the response bank is R_r(x)=<x|P_A1 B_r|C>.  Contracting
``C(x)+sum_r u_r(theta) R_r(x)`` gives C+P_A1(K-I)C.  This is an A1 state
only when C is already A1; projected-carrier values are returned separately
so a caller can check that assumption on its fixed output probes.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import validated_permutation_group
from ..models.conditional_carrier_resonance import (
    conditional_filter_coefficients, packed_scalar_rows,
)


@dataclass(frozen=True)
class ConditionalCarrierQueryPlan:
    """Immutable query/reconstruction maps, with no N<=64 integer-key limit."""

    configurations: np.ndarray
    output_configurations: np.ndarray
    group_images: np.ndarray
    active_sites: tuple[int, int, int, int]
    spectator_sites: tuple[int, int, int, int]
    local_rows: np.ndarray
    connection_indices: np.ndarray
    carrier_indices: np.ndarray
    projected_carrier_indices: np.ndarray
    diagnostics: dict[str, int | bool | float]


@dataclass(frozen=True)
class ConditionalCarrierResponses:
    """Carrier and60 full-group scalar responses at the original outputs."""

    carrier_amplitudes: np.ndarray
    projected_carrier_amplitudes: np.ndarray
    responses: np.ndarray
    diagnostics: dict[str, int | bool | float]


def _sites(raw: Sequence[int], n_sites: int, name: str) -> tuple[int, int, int, int]:
    values = tuple(raw)
    if (len(values) != 4 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                               or not 0 <= i < n_sites for i in values) or len(set(values)) != 4):
        raise ValueError(f'{name} must be four distinct integer site indices')
    return tuple(int(i) for i in values)


def compile_conditional_carrier_queries(
    configurations: np.ndarray,
    group_images: Sequence[Sequence[int]],
    *,
    active_sites: Sequence[int] = (0, 1, 2, 3),
    spectator_sites: Sequence[int] = (4, 5, 6, 7),
) -> ConditionalCarrierQueryPlan:
    """Compile P_A1 B_r C inputs using canonical pullbacks ``x[group]``.

    Inputs have shape(output_count,N), N>=8. The eight support positions are
    ordered: four active followed by four spectator. Deduplication spans ALL
    outputs, group images and valid36-slot local rows before any carrier is
    evaluated. No carrier invariance is used to identify different inputs.
    """
    states = np.asarray(configurations)
    if (states.ndim != 2 or len(states) < 1 or states.shape[1] < 8 or
        states.dtype.kind not in 'biu' or np.any((states != 0) & (states != 1))):
        raise ValueError('configurations must be a nonempty integer bit matrix with N>=8')
    states = states.astype(np.int8, copy=True); count, n_sites = states.shape
    active = _sites(active_sites, n_sites, 'active_sites')
    spectator = _sites(spectator_sites, n_sites, 'spectator_sites')
    if set(active) & set(spectator): raise ValueError('active and spectator supports must be disjoint')
    group = np.asarray(validated_permutation_group(group_images), dtype=np.int32)
    if group.shape[1] != n_sites: raise ValueError('group and configurations have different site counts')
    support = np.asarray(active+spectator, dtype=np.int32)
    orbit = states[:, group]
    powers = 1 << np.arange(7, -1, -1)
    local_rows = np.sum(orbit[:, :, support]*powers, axis=-1).astype(np.int32)
    packed = packed_scalar_rows()
    local_columns = packed.columns[local_rows]
    valid = packed.mask[local_rows]
    local_bits = ((local_columns[:, :, :, None] >> np.arange(7, -1, -1)) & 1).astype(np.int8)
    inputs = np.broadcast_to(orbit[:, :, None, :], (count, len(group), 36, n_sites)).copy()
    inputs[:, :, :, support] = local_bits
    flattened_valid = inputs[valid]
    merged = np.concatenate((flattened_valid, states), axis=0)
    unique, inverse = np.unique(merged, axis=0, return_inverse=True)
    connection_indices = np.zeros(valid.shape, dtype=np.int32)
    connection_indices[valid] = inverse[:len(flattened_valid)]
    carrier_indices = inverse[len(flattened_valid):].astype(np.int32)
    positions = packed.identity_positions[local_rows]
    projected_indices = np.take_along_axis(connection_indices, positions[:, :, None], axis=2)[:, :, 0]
    diagnostics = dict(output_count=count, n_sites=n_sites, group_order=len(group),
        padded_local_queries=int(count*len(group)*36), valid_local_queries=int(valid.sum()),
        carrier_center_requests=count, raw_total_requests=int(valid.sum()+count),
        unique_carrier_queries=len(unique), maximum_local_connections=int(valid.sum(axis=-1).max()),
        duplicate_requests_removed=int(len(merged)-len(unique)),
        complete_group=True, magnetic_channels_truncated=False,
        carrier_symmetry_used_for_deduplication=False, normalized=False)
    for array in (unique, states, group, local_rows, connection_indices, carrier_indices, projected_indices):
        array.setflags(write=False)
    return ConditionalCarrierQueryPlan(unique, states, group, active, spectator,
        local_rows, connection_indices, carrier_indices, projected_indices, diagnostics)


def contract_conditional_carrier_queries(
    plan: ConditionalCarrierQueryPlan, unique_amplitudes: np.ndarray,
) -> ConditionalCarrierResponses:
    """Reuse one value per unique carrier input to form all60 responses."""
    amplitudes = np.asarray(unique_amplitudes, dtype=np.complex128)
    if amplitudes.shape != (len(plan.configurations),) or not np.isfinite(amplitudes).all():
        raise ValueError('one finite carrier amplitude is required per unique input')
    packed = packed_scalar_rows()
    response = np.empty((len(plan.output_configurations), 60), dtype=np.complex128)
    for sample, rows in enumerate(plan.local_rows):
        source = amplitudes[plan.connection_indices[sample]]
        response[sample] = np.einsum('rgk,gk->r', packed.coefficients[:, rows, :], source)/len(plan.group_images)
    carrier = amplitudes[plan.carrier_indices]
    projected = np.mean(amplitudes[plan.projected_carrier_indices], axis=1)
    scale = max(float(np.linalg.norm(carrier)), np.finfo(float).tiny)
    diagnostics = dict(plan.diagnostics,
        carrier_A1_probe_relative_residual=float(np.linalg.norm(projected-carrier)/scale),
        carrier_A1_probe_max_absolute_residual=float(np.max(abs(projected-carrier))),
        unique_exact_carrier_nodes=int(np.count_nonzero(amplitudes == 0)),
        probe_agreement_is_not_global_symmetry_proof=True,
        new_H_calls=0, trained=False)
    for array in (carrier, projected, response): array.setflags(write=False)
    return ConditionalCarrierResponses(carrier, projected, response, diagnostics)


def evaluate_conditional_carrier_queries(
    plan: ConditionalCarrierQueryPlan,
    carrier_fn: Callable[[np.ndarray], np.ndarray],
    *,
    batch_size: int = 256,
) -> ConditionalCarrierResponses:
    """Evaluate the deduplicated inputs once, then reuse their amplitudes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    parts = []
    for start in range(0, len(plan.configurations), batch_size):
        inputs = plan.configurations[start:start+batch_size]
        values = np.asarray(carrier_fn(inputs))
        if values.shape != (len(inputs),) or not np.isfinite(values).all():
            raise FloatingPointError('carrier callback returned invalid amplitudes')
        parts.append(values)
    return contract_conditional_carrier_queries(plan, np.concatenate(parts))


def conditional_carrier_amplitudes(
    parameters: jax.Array, carrier_amplitudes: jax.Array, responses: jax.Array,
) -> jax.Array:
    """Pure-JAX C+P_A1(K-I)C; gradients reuse the frozen carrier responses.

    This assumes a fixed carrier.  Jointly training AGP pairings would require
    differentiating the carrier queries, not reusing a frozen response bank.
    """
    carrier = jnp.asarray(carrier_amplitudes)
    bank = jnp.asarray(responses)
    if carrier.ndim != 1 or bank.shape != (len(carrier), 60):
        raise ValueError('carrier[B] and responses[B,60] shapes required')
    return carrier+bank@conditional_filter_coefficients(parameters)


__all__ = ['ConditionalCarrierQueryPlan', 'ConditionalCarrierResponses',
    'compile_conditional_carrier_queries', 'contract_conditional_carrier_queries',
    'evaluate_conditional_carrier_queries', 'conditional_carrier_amplitudes']
