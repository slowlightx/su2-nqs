"""Local reduced-label-controlled scalar filters on a physical carrier.

Four active spin-half sites admit q=0,2,4 multiplicities 2,3,1.  A disjoint
four-site spectator is resolved into its six complete (q,a) projectors, each
summed over ALL magnetic components.  A real symmetric active multiplicity
generator G_q(z) is exponentiated separately in each spectator channel z.

The resulting eight-site operator is an exact SU(2) scalar.  It acts directly
on physical carrier amplitudes; no global carrier-to-fusion conversion or
independent residual wavefunction is involved.  Spatial A1 projection remains
the caller's responsibility.  This is a local mechanism/capacity oracle, NOT
a ViT or a new claim of scalable configuration-dependent attention.  A single
filter lies in the explicitly exposed 60-dimensional local operator span.

All angular momenta are doubled integers.  Local bit order is active0..3,
spectator0..3, with bit0=down and the first site most significant.  Operator
rows are output configurations, columns input configurations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from ..su2.local_schur_fusion import local_schur_channels


Q_VALUES = (0, 2, 4)
MULTIPLICITIES = (2, 3, 1)
ACTIVE_DIAGONAL_COORDINATES = (0, 1, 3, 4, 5, 9)


@dataclass(frozen=True)
class LocalScalarComponents:
    """Immutable offline Schur data; no trainable magnetic-index entries.

    ``reduced_basis[q]`` has shape (10,d_q,d_q), with zeros for generators
    belonging to other q blocks.  Diagonal matrix units precede normalized
    symmetric off-diagonal units in each q block.  The q0 basis is the
    repository's square-cell A1/B1 convention, not an unrotated dimer gauge.
    """

    channels: tuple[tuple[int, int], ...]
    generator_labels: tuple[tuple[int, int, int], ...]
    reduced_basis: dict[int, np.ndarray]
    active_basis: np.ndarray
    spectator_projectors: np.ndarray


@dataclass(frozen=True)
class PackedScalarRows:
    """Complete fixed-m sparse connectivity of the local eight-site filter.

    ``columns[row,k]`` contains local input indices. ``coefficients[r,row,k]``
    is the r-th operator basis element.  At most36 inputs occur because each
    four-site scalar independently preserves its magnetic projection.
    Invalid padding has zero coefficients and ``mask=False``.  The identity
    position is always valid, including rows with only one input.
    """

    columns: np.ndarray
    mask: np.ndarray
    coefficients: np.ndarray
    identity_positions: np.ndarray


def conditional_generator_count() -> int:
    """Number of real symmetric generator coordinates:6*(3+6+1)."""
    return 60


def factorized_generator_count() -> int:
    """Shared active10 plus spectator6 log weights, with one scale gauge."""
    return 16


def diagonal_conditioned_generator_count() -> int:
    """Shared active10 plus6 spectator-by6 active-diagonal offsets."""
    return 46


@lru_cache(maxsize=1)
def local_scalar_components() -> LocalScalarComponents:
    """Construct complete fixed numeric tables using the canonical CG data."""
    spec = cell_schur_spec('square_plaquette')
    tensor = cell_schur_tensor('square_plaquette')
    labels = []
    for q, size in zip(Q_VALUES, MULTIPLICITIES, strict=True):
        labels.extend((q, a, a) for a in range(size))
        labels.extend((q, a, b) for a in range(size) for b in range(a+1, size))
    reduced = {q:np.zeros((10, size, size), dtype=np.float64)
               for q, size in zip(Q_VALUES, MULTIPLICITIES, strict=True)}
    for index, (q, a, b) in enumerate(labels):
        if a == b: reduced[q][index, a, b] = 1.
        else:
            reduced[q][index, a, b] = 1/np.sqrt(2.)
            reduced[q][index, b, a] = 1/np.sqrt(2.)
    active = np.zeros((10, 16, 16), dtype=np.float64)
    for q, size in zip(Q_VALUES, MULTIPLICITIES, strict=True):
        for m in range(-q, q+1, 2):
            u = tensor[:, q, m+spec.cell_size, :size]
            active += np.einsum('ia,rab,jb->rij', u, reduced[q], u)
    channels = local_schur_channels('square_plaquette')
    projectors = []
    for q, a in channels:
        matrix = np.zeros((16, 16), dtype=np.float64)
        for m in range(-q, q+1, 2):
            u = tensor[:, q, m+spec.cell_size, a]
            matrix += np.outer(u, u)
        projectors.append(matrix)
    spectator = np.stack(projectors)
    if np.max(abs(spectator.sum(axis=0)-np.eye(16))) > 2e-12:
        raise RuntimeError('spectator Schur projectors are incomplete')
    if np.max(abs(active[list(ACTIVE_DIAGONAL_COORDINATES)].sum(axis=0)-np.eye(16))) > 2e-12:
        raise RuntimeError('active reduced matrix units are incomplete')
    for value in (*reduced.values(), active, spectator): value.setflags(write=False)
    return LocalScalarComponents(channels, tuple(labels), reduced, active, spectator)


def _real_parameters(parameters: jax.Array, count: int, *, name: str) -> jax.Array:
    array = jnp.asarray(parameters)
    if jnp.iscomplexobj(array): raise ValueError(f'{name} must contain real generator coordinates')
    if array.shape != (count,): raise ValueError(f'{name} must have shape({count},)')
    array = array.astype(jnp.float64)
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError(f'{name} must be finite')
    return array


def _conditional_parameters(parameters: jax.Array) -> jax.Array:
    array = jnp.asarray(parameters)
    if array.shape == (6, 10): array = array.reshape(60)
    return _real_parameters(array, 60, name='conditional parameters').reshape(6, 10)


def factorized_to_conditional(parameters: jax.Array) -> jax.Array:
    """Embed a shared active filter times arbitrary scalar spectator weights.

    The first10 entries parameterize one active generator.  The last6 are
    spectator log weights lambda_z, so G_q(z)=G_q+lambda_z I and the physical
    operator factors as exp(G_active) tensor sum_z exp(lambda_z) P_z.
    """
    values = _real_parameters(parameters, 16, name='factorized parameters')
    identity = jnp.zeros(10, dtype=jnp.float64).at[jnp.asarray(ACTIVE_DIAGONAL_COORDINATES)].set(1.)
    return values[None, :10]+values[10:, None]*identity[None, :]


def diagonal_conditioned_to_conditional(parameters: jax.Array) -> jax.Array:
    """Embed the shared off-diagonal/null with conditional diagonal offsets.

    The first10 entries parameterize a shared active generator.  Remaining
    entries form (spectator6, active(q,a)6) generator-diagonal offsets.  This
    is an exact nested restriction of the full60 generator coordinates, not
    a claim that exponentiation leaves off-diagonal FILTER entries fixed.
    Shared diagonal coordinates are redundant but kept explicit.
    """
    values = _real_parameters(parameters, 46, name='diagonal-conditioned parameters')
    result = jnp.broadcast_to(values[:10], (6, 10))
    return result.at[:, jnp.asarray(ACTIVE_DIAGONAL_COORDINATES)].add(values[10:].reshape(6, 6))


def conditional_reduced_generators(parameters: jax.Array) -> dict[int, jax.Array]:
    """Return q→G_q[z,a_out,a_in], with no magnetic-index dependence."""
    values = _conditional_parameters(parameters)
    return {q:jnp.einsum('zr,rab->zab', values, jnp.asarray(basis))
            for q, basis in local_scalar_components().reduced_basis.items()}


def conditional_filter_coefficients(parameters: jax.Array) -> jax.Array:
    """Return60 coefficients of K-I in the exposed fixed local basis.

    Matrix exponentials use scaling-and-squaring rather than differentiating
    eigenvectors; identity/repeated-eigenvalue initialization has finite AD.
    Subtracting each reduced identity makes parameters=0 exactly K=I, rather
    than reassembling a numerically approximate identity from CG columns.
    """
    generators = conditional_reduced_generators(parameters)
    coefficients = jnp.zeros((6, 10), dtype=jnp.float64)
    for q, basis in local_scalar_components().reduced_basis.items():
        exponentials = jax.vmap(jsl.expm)(generators[q])
        exponentials = (exponentials+jnp.swapaxes(exponentials, -1, -2))/2
        delta = exponentials-jnp.eye(basis.shape[-1], dtype=jnp.float64)[None]
        coefficients += jnp.einsum('rab,zab->zr', jnp.asarray(basis), delta)
    return coefficients.reshape(60)


@lru_cache(maxsize=1)
def packed_scalar_rows() -> PackedScalarRows:
    """Pack all allowed same-active-m/same-spectator-m local transitions."""
    parts = local_scalar_components()
    columns = np.zeros((256, 36), dtype=np.int32)
    mask = np.zeros((256, 36), dtype=bool)
    coefficients = np.zeros((60, 256, 36), dtype=np.float64)
    identity_positions = np.empty(256, dtype=np.int32)
    by_count = {n:tuple(i for i in range(16) if i.bit_count() == n) for n in range(5)}
    for row in range(256):
        a, s = divmod(row, 16)
        inputs = [(b, t) for b in by_count[a.bit_count()] for t in by_count[s.bit_count()]]
        ac = np.asarray([b for b, _ in inputs]); sc = np.asarray([t for _, t in inputs])
        count = len(inputs); columns[row, :count] = 16*ac+sc; mask[row, :count] = True
        active = parts.active_basis[:, a, ac]
        spectator = parts.spectator_projectors[:, s, sc]
        coefficients[:, row, :count] = (spectator[:, None, :]*active[None, :, :]).reshape(60, count)
        identity_positions[row] = inputs.index((a, s))
    for array in (columns, mask, coefficients, identity_positions): array.setflags(write=False)
    return PackedScalarRows(columns, mask, coefficients, identity_positions)


def local_filter_rows(parameters: jax.Array, *, delta_only: bool = False) -> jax.Array:
    """Differentiable [256,36] physical coefficients for K or K-I."""
    packed = packed_scalar_rows()
    result = jnp.einsum('r,rsk->sk', conditional_filter_coefficients(parameters), jnp.asarray(packed.coefficients))
    if not delta_only:
        result = result+jax.nn.one_hot(jnp.asarray(packed.identity_positions), 36, dtype=jnp.float64)
    return jnp.where(jnp.asarray(packed.mask), result, 0.)


def conditional_filter_matrix(parameters: jax.Array, *, delta_only: bool = False) -> jax.Array:
    """Small physical K[out,in] oracle, also exact at zero parameters."""
    packed = packed_scalar_rows(); rows = jnp.arange(256)[:, None]
    return jnp.zeros((256, 256), dtype=jnp.float64).at[rows, jnp.asarray(packed.columns)].add(
        local_filter_rows(parameters, delta_only=delta_only))


@lru_cache(maxsize=1)
def local_scalar_basis() -> np.ndarray:
    """Return60 fixed physical basis matrices; allocate only when requested.

    About31.5 MB, intended for exact-small precontraction.  Sampled callers
    should use the approximately4.4 MB packed connectivity instead.
    """
    parts = local_scalar_components()
    result = np.einsum('zij,rab->zraibj', parts.spectator_projectors,
                        parts.active_basis).reshape(60, 256, 256)
    result.setflags(write=False)
    return result


__all__ = [
    'LocalScalarComponents', 'PackedScalarRows', 'Q_VALUES', 'MULTIPLICITIES',
    'ACTIVE_DIAGONAL_COORDINATES', 'conditional_generator_count',
    'factorized_generator_count', 'diagonal_conditioned_generator_count',
    'local_scalar_components', 'conditional_reduced_generators',
    'conditional_filter_coefficients', 'factorized_to_conditional',
    'diagonal_conditioned_to_conditional', 'packed_scalar_rows',
    'local_filter_rows', 'conditional_filter_matrix', 'local_scalar_basis',
]
