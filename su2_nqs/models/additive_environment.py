"""Nonredundant additive coordinates for two complete four-spin environments.

The six environment labels are the complete doubled-spin/multiplicity pairs
``(q,a) = (0,0),(0,1),(2,0),(2,1),(2,2),(4,0)``.  Magnetic projections are
not arguments: a physical gate must retain each label's entire multiplet.
For ten real active reduced-generator coordinates, additive tables have
dimension ``(6 + 6 - 1) * 10 = 110``, not120.  The basis used here is
orthonormal in the unweighted Frobenius table metric, so coordinates are
unique and have exactly the same Euclidean norm as the generated table.

This module only parameterizes generators. Exponentiating an additive
generator can still yield a nonadditive filter; no linear state-manifold or
physical-entanglement interpretation is implied.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


ADDITIVE_ENVIRONMENT_PARAMETER_COUNT = 110


@lru_cache(maxsize=1)
def additive_environment_basis() -> np.ndarray:
    """Return immutable orthonormal tables ``basis[11,6,6]``.

    Coordinate0 is the constant table1/6. Coordinates1..5 are zero-mean
    Helmert contrasts of the first environment;6..10 contrast the second.
    This deterministic label order is the complete local Schur order above,
    not an ordering by magnetic projection.
    """
    contrasts = np.zeros((6, 5), dtype=np.float64)
    for column in range(5):
        length = column + 1
        denominator = np.sqrt(length * (length + 1))
        contrasts[:length, column] = 1. / denominator
        contrasts[length, column] = -length / denominator
    basis = np.empty((11, 6, 6), dtype=np.float64)
    basis[0] = 1. / 6
    for column in range(5):
        basis[column + 1] = contrasts[:, column, None] / np.sqrt(6.)
        basis[column + 6] = contrasts[None, :, column] / np.sqrt(6.)
    basis.setflags(write=False)
    return basis


def _real_array(value: jax.Array, shape: tuple[int, ...], name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.shape != shape or jnp.iscomplexobj(array):
        raise ValueError(f'{name} must be real with shape{shape}')
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError(f'{name} must be finite')
    return array.astype(jnp.float64)


def additive_environment_generators(parameters: jax.Array) -> jax.Array:
    """Map110 free real coordinates to the complete table ``[6,6,10]``.

    Accept flat110 or shaped[11,10] arrays. Every additive table is generated
    exactly once; zero is the identity-generator initialization. Traced
    nonfinite inputs return NaNs so compiled callers can fail closed.
    """
    values = jnp.asarray(parameters)
    if values.shape == (ADDITIVE_ENVIRONMENT_PARAMETER_COUNT,):
        values = values.reshape(11, 10)
    values = _real_array(values, (11, 10), 'parameters')
    table = jnp.einsum('kzw,kd->zwd', jnp.asarray(additive_environment_basis()), values)
    return jnp.where(jnp.all(jnp.isfinite(values)), table, jnp.nan)


def additive_environment_coordinates(generators: jax.Array) -> jax.Array:
    """Unique least-squares additive coordinates ``[11,10]`` of any table.

    On an additive table this is the exact inverse. Otherwise it returns
    coordinates of the Frobenius orthogonal projection
    ``row_mean + column_mean - global_mean``; the discarded interaction must
    not be mistaken for a coordinate redundancy. The function is JAX
    differentiable and accepts only real complete tables[6,6,10].
    """
    values = _real_array(generators, (6, 6, 10), 'generators')
    coordinates = jnp.einsum('kzw,zwd->kd', jnp.asarray(additive_environment_basis()), values)
    return jnp.where(jnp.all(jnp.isfinite(values)), coordinates, jnp.nan)


__all__ = ['ADDITIVE_ENVIRONMENT_PARAMETER_COUNT', 'additive_environment_basis',
           'additive_environment_generators', 'additive_environment_coordinates']
