"""Exact 1176-feature compression of two overlapping scalar carrier gates.

For K1: (A,C)->B followed by K2: (B,A)->C, the spectator cell A is
diagonal in its six complete Schur projectors. B and C each live in the
four-spin scalar commutant of dimension 2**2+3**2+1**2=14. Thus every
K2 K1 is in the *exact*, untruncated span P_a^A tensor E_u^B tensor E_v^C.
E^(q)_(a,b)=sum_m |q,a,m><q,b,m| is NOT Frobenius normalized: its squared
norm is q+1. Its reduced matrix coefficient has no extra dimension factor.

Full spatial projection and a fixed physical carrier can be contracted once
into these1176 features per requested configuration. Hamiltonian rows may
then act linearly on the features. Training must still differentiate both
psi and Hpsi for a finite fixed-Q full-H objective. The nonlinear generator
exponentials and neural coefficients remain dynamic; no state truncation,
new ansatz, or claim of long-range ViT expressivity follows from compression.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..models.conditional_carrier_resonance import conditional_filter_coefficients, local_scalar_components
from ..models.multicontext_resonance_attention import free_multicontext_generators
from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from .overlapping_carrier_queries import OverlappingCarrierQueryPlan, local_overlap_connections


@dataclass(frozen=True)
class CommutantComponents:
    """Canonical q/a matrix units in increasing (q,a_out,a_in) order."""

    spectator_labels: tuple[tuple[int, int], ...]
    unit_labels: tuple[tuple[int, int, int], ...]
    projectors: np.ndarray
    matrix_units: np.ndarray
    identity_coefficients: np.ndarray


@dataclass(frozen=True)
class MulticontextFeatureBatch:
    """Immutable fixed-carrier features including the complete final A1 sum."""

    features: np.ndarray
    projected_carrier: np.ndarray
    diagnostics: dict[str, int | float | bool | str]


@lru_cache(maxsize=1)
def commutant_components() -> CommutantComponents:
    """Use the existing square-cell Schur convention, keeping all magnetic m."""
    spec = cell_schur_spec('square_plaquette')
    tensor = cell_schur_tensor('square_plaquette')
    labels = tuple((q, a, b) for q, d in ((0, 2), (2, 3), (4, 1))
                   for a in range(d) for b in range(d))
    units = np.stack([sum(np.outer(tensor[:, q, m+spec.cell_size, a],
                                   tensor[:, q, m+spec.cell_size, b])
                          for m in range(-q, q+1, 2)) for q, a, b in labels])
    diagonal = np.asarray([a == b for _, a, b in labels], dtype=np.float64)
    identity = np.broadcast_to(diagonal[None, :, None]*diagonal[None, None, :], (6, 14, 14)).reshape(1176).copy()
    units.setflags(write=False); identity.setflags(write=False)
    parts = local_scalar_components()
    return CommutantComponents(parts.channels, labels, parts.spectator_projectors, units, identity)


def _unit_deltas(tables: jax.Array) -> jax.Array:
    """Return reduced exp(G)-I entries [env1,env2,unit], not physical traces."""
    generators = free_multicontext_generators(tables)
    coefficients = jax.vmap(conditional_filter_coefficients)(generators).reshape(6, 6, 10)
    bases = local_scalar_components().reduced_basis
    return jnp.stack([jnp.einsum('...r,r->...', coefficients, jnp.asarray(bases[q][:, a, b]))
                      for q, a, b in commutant_components().unit_labels], axis=-1)


def multicontext_feature_coefficients(
    tables1: jax.Array, tables2: jax.Array, *, mode: str = 'cascade', centered: bool = True,
) -> jax.Array:
    """Dynamic1176 coefficients of K2 K1-I, or K1+K2-2I without the cascade.

    A is labeled a. For B unit(qB,bo,bi), C unit(qC,co,ci), the full
    coefficient is U1[a,(qC,ci)]_(bo,bi) U2[(qB,bo),a]_(co,ci).
    The outgoing B and incoming C labels arise from coherent operator
    multiplication, NOT from a classical environment measurement.
    ``centered=False`` adds the exact identity coefficient vector.
    """
    if mode not in ('cascade', 'no_cascade'):
        raise ValueError('unknown composition mode')
    parts = commutant_components()
    lookup = {label:i for i, label in enumerate(parts.spectator_labels)}
    b_label = jnp.asarray([lookup[(q, a)] for q, a, _ in parts.unit_labels])
    c_label = jnp.asarray([lookup[(q, b)] for q, _, b in parts.unit_labels])
    a = jnp.arange(6)[:, None, None]
    unit_b = jnp.arange(14)[None, :, None]
    unit_c = jnp.arange(14)[None, None, :]
    first = _unit_deltas(tables1)[a, c_label[None, None, :], unit_b]
    second = _unit_deltas(tables2)[b_label[None, :, None], a, unit_c]
    diagonal = jnp.asarray([x == y for _, x, y in parts.unit_labels])
    coefficients = first*diagonal[None, None, :]+diagonal[None, :, None]*second
    if mode == 'cascade':
        coefficients = coefficients+first*second
    coefficients = coefficients.reshape(1176)
    if not centered:
        coefficients = coefficients+jnp.asarray(parts.identity_coefficients)
    return coefficients


@lru_cache(maxsize=1)
def _cell_alternatives() -> tuple[np.ndarray, np.ndarray]:
    choices = np.zeros((16, 6), dtype=np.int32)
    valid = np.zeros((16, 6), dtype=bool)
    for row in range(16):
        columns, mask, _ = local_overlap_connections(256*row)
        selected = columns[mask]//256
        choices[row, :len(selected)] = selected
        valid[row, :len(selected)] = True
    choices.setflags(write=False); valid.setflags(write=False)
    return choices, valid


def features_from_rows(local_rows: jax.Array, terminal_values: jax.Array) -> jax.Array:
    """Pure-JAX unprojected features for M rows and their216 carrier inputs.

    Shape is (M,1176). Staged tensor contractions avoid allocating the full
    (M,1176,216) product basis. The largest output scales only as1176*M.
    Padding in the public216-input row map is masked before contraction.
    """
    rows, terminal = jnp.asarray(local_rows), jnp.asarray(terminal_values)
    if rows.ndim != 1 or len(rows) < 1 or not jnp.issubdtype(rows.dtype, jnp.integer):
        raise ValueError('local_rows must be a nonempty integer vector')
    if terminal.shape != (len(rows), 216):
        raise ValueError('terminal_values must have shape (rows,216)')
    if not isinstance(rows, jax.core.Tracer) and (np.any(np.asarray(rows) < 0) or np.any(np.asarray(rows) >= 4096)):
        raise ValueError('local_rows must be12-bit configurations')
    choices, valid = map(jnp.asarray, _cell_alternatives())
    a, b, c = rows//256, (rows//16) % 16, rows % 16
    av, bv, cv = choices[a], choices[b], choices[c]
    mask = valid[a, :, None, None] & valid[b, None, :, None] & valid[c, None, None, :]
    source = jnp.where(mask, terminal.reshape(-1, 6, 6, 6), 0.)
    parts = commutant_components()
    projectors, units = jnp.asarray(parts.projectors), jnp.asarray(parts.matrix_units)
    pa = jnp.moveaxis(projectors[:, a[:, None], av], 0, 1)
    eb = jnp.moveaxis(units[:, b[:, None], bv], 0, 1)
    ec = jnp.moveaxis(units[:, c[:, None], cv], 0, 1)
    first = jnp.einsum('nabc,nza->nzbc', source, pa)
    second = jnp.einsum('nzbc,nub->nzuc', first, eb)
    return jnp.einsum('nzuc,nvc->nzuv', second, ec).reshape(len(rows), 1176)


_compiled_features_from_rows = jax.jit(features_from_rows)


def features_from_plan(
    plan: OverlappingCarrierQueryPlan, unique_carrier_amplitudes: np.ndarray,
    *, image_batch_size: int = 128,
) -> MulticontextFeatureBatch:
    """Stream all complete group images into immutable1176-column features.

    The carrier cache must correspond exactly to this plan's query order.
    No carrier calls are made. Spatial images are averaged once, after both
    gates. A fixed-shape padded JAX contraction is reused across image batches;
    only O(image_batch_size*1176) intermediate feature memory is retained.
    The cache remains valid only while carrier, cells, and group are fixed.
    """
    values = np.asarray(unique_carrier_amplitudes)
    if values.shape != (len(plan.configurations),) or not np.isfinite(values).all():
        raise ValueError('one finite amplitude per exact carrier query is required')
    if isinstance(image_batch_size, bool) or not isinstance(image_batch_size, Integral) or image_batch_size < 1:
        raise ValueError('image_batch_size must be a positive integer')
    rows = plan.unique_local_rows[plan.local_row_indices].reshape(-1)
    columns = plan.connection_indices.reshape(-1, 216)
    group_order = len(plan.group_images)
    count = len(plan.output_configurations)
    dtype = np.result_type(values.dtype, np.float64)
    features = np.zeros((count, 1176), dtype=dtype)
    for start in range(0, len(rows), image_batch_size):
        size = min(image_batch_size, len(rows)-start)
        padded_rows = np.zeros(image_batch_size, dtype=np.int32)
        terminals = np.zeros((image_batch_size, 216), dtype=dtype)
        padded_rows[:size] = rows[start:start+size]
        terminals[:size] = values[columns[start:start+size]]
        contribution = np.asarray(_compiled_features_from_rows(jnp.asarray(padded_rows), jnp.asarray(terminals)))[:size]
        output_indices = np.arange(start, start+size)//group_order
        np.add.at(features, output_indices, contribution/group_order)
    projected = np.mean(values[plan.projected_carrier_indices], axis=1)
    identity_error = float(np.max(np.abs(features@commutant_components().identity_coefficients-projected)))
    features.setflags(write=False); projected.setflags(write=False)
    return MulticontextFeatureBatch(features, projected, dict(
        feature_count=1176, output_count=count, group_order=group_order,
        image_batch_size=int(image_batch_size), valid_images=len(rows),
        padded_images=((len(rows)+image_batch_size-1)//image_batch_size)*image_batch_size,
        retained_feature_bytes=features.nbytes+projected.nbytes,
        identity_absolute_error=identity_error, new_carrier_calls=0, new_H_calls=0,
        complete_group=True, magnetic_channels_truncated=False,
        cache_scope='fixed carrier, ordered three cells, complete final spatial group; generator parameters remain dynamic'))


def amplitudes_from_features(
    tables1: jax.Array, tables2: jax.Array, features: jax.Array,
    projected_carrier: jax.Array, *, mode: str = 'cascade',
) -> jax.Array:
    """Centered exact reconstruction; zero generators replay P_A1 C exactly.

    The same function evaluates Hpsi when passed linearly H-transformed
    features and H(P_A1 C). No carrier or candidate division occurs at nodes.
    """
    matrix, carrier = jnp.asarray(features), jnp.asarray(projected_carrier)
    if matrix.ndim != 2 or matrix.shape[1] != 1176 or carrier.shape != (len(matrix),):
        raise ValueError('need matching (configurations,1176) features and carrier vector')
    return carrier+matrix@multicontext_feature_coefficients(tables1, tables2, mode=mode)


__all__ = ['CommutantComponents', 'MulticontextFeatureBatch', 'commutant_components',
           'multicontext_feature_coefficients', 'features_from_rows', 'features_from_plan',
           'amplitudes_from_features']
