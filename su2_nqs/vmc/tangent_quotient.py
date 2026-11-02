"""Exact-support non-null tangent coordinates for autoregressive models.

This module is deliberately an exact small-system oracle.  It forms the
Born-weighted centered complex-log Jacobian, removes its parameter null space,
and rescales the retained right-singular directions so their output tangents
are orthonormal.  The construction uses no teacher values, but its dense SVD
is not a production N=16 algorithm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np


@dataclass(frozen=True)
class ExactTangentQuotient:
    """Whitened non-null coordinates around one parameter origin."""

    origin_flat: jax.Array
    directions: jax.Array
    singular_values: np.ndarray
    output_basis: np.ndarray
    unravel: Callable[[jax.Array], Any]
    relative_cutoff: float
    parameter_count: int
    quotient_rank: int
    nullity: int
    whitening_maximum_error: float
    stored_matrix_elements: int

    def parameters(self, coordinates: jax.Array) -> Any:
        """Map quotient coordinates back to the original parameter pytree."""

        values = jnp.asarray(coordinates, dtype=self.origin_flat.dtype)
        if values.shape != (self.quotient_rank,):
            raise ValueError(
                f"expected quotient coordinate shape ({self.quotient_rank},)"
            )
        return self.unravel(self.origin_flat + self.directions @ values)


@dataclass(frozen=True)
class MatrixFreeTangentQuotient:
    """Randomized non-null coordinates built from normal-operator products."""

    origin_flat: jax.Array
    directions: jax.Array
    approximate_singular_values: np.ndarray
    unravel: Callable[[jax.Array], Any]
    requested_rank: int
    quotient_rank: int
    sketch_width: int
    subspace_iterations: int
    sample_count: int
    sampled_whitening_maximum_error: float
    stored_matrix_elements: int

    def parameters(self, coordinates: jax.Array) -> Any:
        """Map randomized quotient coordinates to the original pytree."""

        values = jnp.asarray(coordinates, dtype=self.origin_flat.dtype)
        if values.shape != (self.quotient_rank,):
            raise ValueError(
                f"expected quotient coordinate shape ({self.quotient_rank},)"
            )
        return self.unravel(self.origin_flat + self.directions @ values)


def _normalized_weights(sample_weights: jax.Array, size: int, dtype) -> jax.Array:
    weights = jnp.asarray(sample_weights, dtype=dtype)
    if weights.shape != (size,):
        raise ValueError("sample_weights must have one entry per sample")
    if bool(np.any(~np.isfinite(np.asarray(weights)))) or bool(
        np.any(np.asarray(weights) < 0.0)
    ):
        raise ValueError("sample_weights must be finite and non-negative")
    total = float(np.sum(np.asarray(weights)))
    if total <= 0.0:
        raise ValueError("sample_weights must have positive total weight")
    return weights / jnp.sum(weights)


def build_exact_tangent_quotient(
    params: Any,
    samples: jax.Array,
    sample_weights: jax.Array,
    log_psi_fn: Callable[[Any, jax.Array], jax.Array],
    *,
    relative_cutoff: float = 1.0e-10,
) -> ExactTangentQuotient:
    """Construct a dense teacher-independent centered-log tangent quotient.

    If ``D`` is the weighted realified score Jacobian, returned directions
    satisfy ``D @ directions = U`` for the retained left singular vectors.
    Thus the quotient Fisher matrix is the identity at the construction
    point.  Only model scores and sampling weights enter; no target residual is
    inspected.
    """

    if not np.isfinite(relative_cutoff) or relative_cutoff <= 0.0:
        raise ValueError("relative_cutoff must be positive and finite")
    flat, unravel = ravel_pytree(params)
    if flat.ndim != 1 or not jnp.issubdtype(flat.dtype, jnp.floating):
        raise TypeError("tangent quotient requires flat real parameters")
    sample_values = jnp.asarray(samples)
    if sample_values.ndim < 1 or len(sample_values) < 1:
        raise ValueError("samples must be nonempty")
    weights = _normalized_weights(
        sample_weights, len(sample_values), flat.dtype
    )
    root_weights = jnp.sqrt(2.0 * weights)

    def centered_features(values):
        candidate = unravel(values)
        logs = jax.vmap(lambda sample: log_psi_fn(candidate, sample))(
            sample_values
        )
        centered = logs - jnp.sum(weights * logs)
        return jnp.concatenate(
            (root_weights * jnp.real(centered), root_weights * jnp.imag(centered))
        )

    design = np.asarray(jax.jacrev(centered_features)(flat), dtype=np.float64)
    left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    rank = int(np.count_nonzero(singular_values > relative_cutoff * largest))
    if rank < 1:
        raise ValueError("centered complex-log tangent is numerically null")
    directions = right[:rank].T / singular_values[:rank][None, :]
    output_basis = left[:, :rank]
    whitening_error = float(
        np.max(np.abs(design @ directions - output_basis), initial=0.0)
    )
    return ExactTangentQuotient(
        origin_flat=flat,
        directions=jnp.asarray(directions, dtype=flat.dtype),
        singular_values=np.asarray(singular_values[:rank], dtype=np.float64),
        output_basis=np.asarray(output_basis, dtype=np.float64),
        unravel=unravel,
        relative_cutoff=float(relative_cutoff),
        parameter_count=int(flat.size),
        quotient_rank=rank,
        nullity=int(flat.size - rank),
        whitening_maximum_error=whitening_error,
        stored_matrix_elements=int(directions.size + output_basis.size),
    )


def build_matrix_free_tangent_quotient(
    params: Any,
    samples: jax.Array,
    sample_weights: jax.Array,
    log_psi_fn: Callable[[Any, jax.Array], jax.Array],
    key: jax.Array,
    *,
    rank: int,
    oversampling: int = 8,
    subspace_iterations: int = 2,
    relative_cutoff: float = 1.0e-8,
) -> MatrixFreeTangentQuotient:
    """Build a randomized quotient using only score JVP/VJP products.

    The algorithm applies the empirical Fisher normal operator to a Gaussian
    parameter sketch, performs fixed subspace iteration, and diagonalizes only
    the resulting small Rayleigh matrix.  It never materializes the
    ``(2 * samples) x parameters`` score matrix.  Sampling weights and model
    scores enter, but no teacher residual does.
    """

    if rank < 1 or oversampling < 0 or subspace_iterations < 0:
        raise ValueError("rank must be positive and sketch controls non-negative")
    if not np.isfinite(relative_cutoff) or relative_cutoff <= 0.0:
        raise ValueError("relative_cutoff must be positive and finite")
    flat, unravel = ravel_pytree(params)
    if flat.ndim != 1 or not jnp.issubdtype(flat.dtype, jnp.floating):
        raise TypeError("matrix-free quotient requires flat real parameters")
    if rank > flat.size:
        raise ValueError("rank may not exceed the raw parameter count")
    sample_values = jnp.asarray(samples)
    if sample_values.ndim < 1 or len(sample_values) < 1:
        raise ValueError("samples must be nonempty")
    weights = _normalized_weights(sample_weights, len(sample_values), flat.dtype)
    root_weights = jnp.sqrt(2.0 * weights)

    def centered_features(values):
        candidate = unravel(values)
        logs = jax.vmap(lambda sample: log_psi_fn(candidate, sample))(
            sample_values
        )
        centered = logs - jnp.sum(weights * logs)
        return jnp.concatenate(
            (root_weights * jnp.real(centered), root_weights * jnp.imag(centered))
        )

    _, pushforward = jax.linearize(centered_features, flat)
    _, pullback = jax.vjp(centered_features, flat)

    def normal_product(vector):
        return pullback(pushforward(vector))[0]

    def apply_normal(matrix):
        return jax.vmap(normal_product, in_axes=1, out_axes=1)(matrix)

    sketch_width = min(int(flat.size), rank + oversampling)
    sketch = jax.random.normal(
        key, (flat.size, sketch_width), dtype=flat.dtype
    ) / jnp.sqrt(jnp.asarray(flat.size, dtype=flat.dtype))
    image = apply_normal(sketch)
    for _ in range(subspace_iterations):
        basis, _ = jnp.linalg.qr(image, mode="reduced")
        image = apply_normal(basis)
    basis, _ = jnp.linalg.qr(image, mode="reduced")
    normal_basis = apply_normal(basis)
    rayleigh = np.asarray(
        0.5 * (basis.T @ normal_basis + normal_basis.T @ basis),
        dtype=np.float64,
    )
    eigenvalues, rotations = np.linalg.eigh(rayleigh)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    rotations = rotations[:, order]
    largest = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    retained = min(
        rank,
        int(np.count_nonzero(eigenvalues > relative_cutoff**2 * largest)),
    )
    if retained < 1:
        raise ValueError("randomized empirical tangent is numerically null")
    eigenvalues = eigenvalues[:retained]
    rotations = rotations[:, :retained]
    raw_directions = basis @ jnp.asarray(rotations, dtype=flat.dtype)
    directions = raw_directions / jnp.sqrt(
        jnp.asarray(eigenvalues, dtype=flat.dtype)
    )[None, :]
    output_tangents = jax.vmap(pushforward, in_axes=1, out_axes=1)(directions)
    gram = np.asarray(output_tangents.T @ output_tangents, dtype=np.float64)
    whitening_error = float(
        np.max(np.abs(gram - np.eye(retained)), initial=0.0)
    )
    return MatrixFreeTangentQuotient(
        origin_flat=flat,
        directions=directions,
        approximate_singular_values=np.sqrt(eigenvalues),
        unravel=unravel,
        requested_rank=rank,
        quotient_rank=retained,
        sketch_width=sketch_width,
        subspace_iterations=subspace_iterations,
        sample_count=len(sample_values),
        sampled_whitening_maximum_error=whitening_error,
        stored_matrix_elements=int(directions.size + rayleigh.size),
    )


__all__ = [
    "ExactTangentQuotient",
    "MatrixFreeTangentQuotient",
    "build_exact_tangent_quotient",
    "build_matrix_free_tangent_quotient",
]
