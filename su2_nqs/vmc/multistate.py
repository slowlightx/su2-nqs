"""Basis-invariant exact-small utilities for multistate NQS optimization."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def exhaustive_state_matrix(
    params: object,
    paths: jax.Array,
    log_psi_heads_fn: Callable[[object, jax.Array], jax.Array],
) -> jax.Array:
    """Evaluate normalized autoregressive heads as full-space columns."""

    logs = jax.vmap(lambda path: log_psi_heads_fn(params, path))(paths)
    if logs.ndim != 2:
        raise ValueError("log_psi_heads_fn must return one vector per path")
    return jnp.exp(logs)


def ky_fan_subspace_energy(
    states: jax.Array,
    hamiltonian: jax.Array,
    *,
    relative_rank_tolerance: float = 1.0e-10,
) -> jax.Array:
    """Return ``Tr(Q^dagger H Q)`` for the span of full-space columns.

    The thin QR form makes the objective invariant under any nonsingular
    mixing of the represented heads.  It is an exact-small training oracle,
    not a sampled production estimator.
    """

    vectors = jnp.asarray(states)
    operator = jnp.asarray(hamiltonian)
    if vectors.ndim != 2:
        raise ValueError("states must be a matrix of column vectors")
    if operator.shape != (vectors.shape[0], vectors.shape[0]):
        raise ValueError("hamiltonian shape does not match states")
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")
    gram_values = jnp.linalg.eigvalsh(vectors.conj().T @ vectors)
    full_rank = gram_values[0] > relative_rank_tolerance * gram_values[-1]
    orthonormal, _ = jnp.linalg.qr(vectors, mode="reduced")
    value = jnp.real(
        jnp.trace(orthonormal.conj().T @ operator @ orthonormal)
    )
    # LAPACK QR completes a basis even for duplicate columns.  That completion
    # is not part of the represented neural span, so fail closed rather than
    # silently award the missing Ritz direction.
    return jnp.where(full_rank, value, jnp.asarray(jnp.inf, dtype=value.dtype))


def exact_ky_fan_loss(
    params: object,
    paths: jax.Array,
    hamiltonian: jax.Array,
    log_psi_heads_fn: Callable[[object, jax.Array], jax.Array],
) -> jax.Array:
    """Evaluate the exhaustive basis-invariant multistate training loss."""

    states = exhaustive_state_matrix(params, paths, log_psi_heads_fn)
    return ky_fan_subspace_energy(states, hamiltonian)


def exact_projector_infidelity_loss(
    params: object,
    paths: jax.Array,
    target_subspace: jax.Array,
    log_psi_heads_fn: Callable[[object, jax.Array], jax.Array],
) -> jax.Array:
    """Oracle loss measuring missed weight of a fixed exact subspace.

    This target-dependent exhaustive loss is only a capacity/optimization
    diagnostic.  It must never be presented as a variational training method.
    """

    states = exhaustive_state_matrix(params, paths, log_psi_heads_fn)
    target = jnp.asarray(target_subspace)
    if target.ndim != 2 or target.shape != states.shape:
        raise ValueError("target_subspace shape must match the represented states")
    gram_values = jnp.linalg.eigvalsh(states.conj().T @ states)
    full_rank = gram_values[0] > 1.0e-10 * gram_values[-1]
    represented, _ = jnp.linalg.qr(states, mode="reduced")
    captured = jnp.sum(jnp.abs(target.conj().T @ represented) ** 2)
    value = jnp.real(states.shape[1] - captured)
    return jnp.where(full_rank, value, jnp.asarray(jnp.inf, dtype=value.dtype))


def generalized_ritz_analysis(
    states: np.ndarray,
    hamiltonian: object,
    *,
    relative_rank_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Resolve Gram rank, Ritz roots, residuals and an orthonormal span."""

    vectors = np.asarray(states, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError("states must be a nonempty matrix of columns")
    if getattr(hamiltonian, "shape", None) != (vectors.shape[0], vectors.shape[0]):
        raise ValueError("hamiltonian shape does not match states")
    if not np.isfinite(relative_rank_tolerance) or not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")
    gram = vectors.conj().T @ vectors
    gram = 0.5 * (gram + gram.conj().T)
    gram_values, gram_vectors = np.linalg.eigh(gram)
    largest = float(np.max(gram_values))
    keep = gram_values > relative_rank_tolerance * largest
    rank = int(np.count_nonzero(keep))
    if rank < 1:
        raise ValueError("state span has zero numerical rank")
    whitening = gram_vectors[:, keep] / np.sqrt(gram_values[keep])[None, :]
    orthonormal = vectors @ whitening
    projected = orthonormal.conj().T @ (hamiltonian @ orthonormal)
    projected = 0.5 * (projected + projected.conj().T)
    roots, coefficients = np.linalg.eigh(projected)
    ritz_states = orthonormal @ coefficients
    residuals = [
        float(np.linalg.norm(hamiltonian @ ritz_states[:, index] - roots[index] * ritz_states[:, index]))
        for index in range(rank)
    ]
    positive = gram_values[keep]
    return {
        "rank": rank,
        "gram_eigenvalues": [float(value) for value in gram_values],
        "gram_condition_number": float(np.max(positive) / np.min(positive)),
        "orthonormality_error": float(
            np.max(
                np.abs(orthonormal.conj().T @ orthonormal - np.eye(rank)),
                initial=0.0,
            )
        ),
        "ritz_values": [float(value) for value in roots],
        "ritz_residuals": residuals,
        "orthonormal_basis": orthonormal,
        "ritz_states": ritz_states,
    }


def subspace_projector_overlap(
    first: np.ndarray, second: np.ndarray, *, tolerance: float = 1.0e-10
) -> float:
    """Return the mean principal-angle fidelity between equal-rank spans."""

    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("subspaces must have the same nonempty matrix shape")
    rank = left.shape[1]
    if rank < 1:
        raise ValueError("subspaces must be nonempty")
    for name, vectors in (("first", left), ("second", right)):
        error = float(
            np.max(
                np.abs(vectors.conj().T @ vectors - np.eye(rank)), initial=0.0
            )
        )
        if error > tolerance:
            raise ValueError(f"{name} subspace is not orthonormal")
    return float(np.linalg.norm(left.conj().T @ right, ord="fro") ** 2 / rank)


def subspace_principal_fidelities(
    first: np.ndarray, second: np.ndarray, *, tolerance: float = 1.0e-10
) -> np.ndarray:
    """Return squared cosines of principal angles between equal-rank spans."""

    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("subspaces must have the same nonempty matrix shape")
    rank = left.shape[1]
    if rank < 1:
        raise ValueError("subspaces must be nonempty")
    for name, vectors in (("first", left), ("second", right)):
        error = float(
            np.max(
                np.abs(vectors.conj().T @ vectors - np.eye(rank)), initial=0.0
            )
        )
        if error > tolerance:
            raise ValueError(f"{name} subspace is not orthonormal")
    singular_values = np.linalg.svd(left.conj().T @ right, compute_uv=False)
    return np.sort(np.asarray(singular_values**2, dtype=np.float64))


__all__ = [
    "exact_ky_fan_loss",
    "exact_projector_infidelity_loss",
    "exhaustive_state_matrix",
    "generalized_ritz_analysis",
    "ky_fan_subspace_energy",
    "subspace_projector_overlap",
    "subspace_principal_fidelities",
]
