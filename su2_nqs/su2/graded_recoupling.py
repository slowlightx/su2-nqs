"""Fermion-graded ordering primitives for geometry-native fusion catalogs."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import jax
import jax.numpy as jnp

from .hubbard_cg_irrep import DOUBLON, DOWN, EMPTY, UP, validate_hubbard_token


def _checked_sites(name: str, sites: Sequence[int], n_sites: int) -> tuple[int, ...]:
    raw = tuple(sites)
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
        raise TypeError(f"{name} entries must be integers")
    checked = tuple(int(site) for site in raw)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} cannot contain repeated sites")
    if any(site < 0 or site >= n_sites for site in checked):
        raise ValueError(f"{name} contains a site outside the occupation array")
    if checked != tuple(sorted(checked)):
        raise ValueError(f"{name} must use canonical increasing site order")
    return checked


def validated_child_merge_sites(
    left_sites: Sequence[int],
    right_sites: Sequence[int],
    n_sites: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return host-validated, canonical supports for one oriented merge.

    The returned tuples are suitable static inputs for the pure-JAX kernels
    below.  Runtime kernels deliberately do not repeat value-dependent Python
    validation, which would make them incompatible with ``jax.jit``.
    """

    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or n_sites < 1
    ):
        raise ValueError("n_sites must be a positive integer")
    checked_n_sites = int(n_sites)
    left = _checked_sites("left_sites", left_sites, checked_n_sites)
    right = _checked_sites("right_sites", right_sites, checked_n_sites)
    if set(left) & set(right):
        raise ValueError("child supports must be disjoint")
    return left, right


def canonical_child_merge_sign_jax(
    left_sites: Sequence[int] | jax.Array,
    right_sites: Sequence[int] | jax.Array,
    occupation_counts: jax.Array,
) -> jax.Array:
    """Pure-JAX Koszul sign for validated static child supports.

    ``occupation_counts`` may have shape ``[sites]`` or ``[..., sites]`` and
    must contain integer onsite particle counts in ``{0,1,2}``.  Validate the
    static supports once with :func:`validated_child_merge_sites` before
    closing them over ``jax.jit``; valid runtime occupations are an amplitude-
    input invariant and are not checked with Python conditionals here.
    """

    counts = jnp.asarray(occupation_counts)
    if counts.ndim < 1:
        raise ValueError("occupation_counts must have a site axis")
    if not jnp.issubdtype(counts.dtype, jnp.integer):
        raise TypeError("occupation_counts must have an integer dtype")
    left = jnp.asarray(left_sites, dtype=jnp.int32)
    right = jnp.asarray(right_sites, dtype=jnp.int32)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("child supports must be rank-one index arrays")
    left_counts = jnp.take(counts, left, axis=-1)
    right_counts = jnp.take(counts, right, axis=-1)
    crossings = left[:, None] > right[None, :]
    parity = jnp.sum(
        left_counts[..., :, None]
        * right_counts[..., None, :]
        * crossings,
        axis=(-2, -1),
        dtype=jnp.int32,
    )
    return jnp.where(parity % 2, -1, 1)


def canonical_child_merge_sign(
    left_sites: Sequence[int],
    right_sites: Sequence[int],
    occupation_counts: Sequence[int],
) -> int:
    """Host-validated Koszul sign from onsite particle counts.

    ``occupation_counts[i]`` is the onsite Hubbard occupation in ``{0,1,2}``.
    Modes use the repository's canonical order ``(site 0 up, site 0 down,
    site 1 up, ...)``.  Reordering two complete site blocks contributes
    ``(-1)**(n_i*n_j)``.  Child supports must be disjoint and internally use
    increasing canonical site order.

    Physical Hubbard configurations use tokens ``EMPTY, UP, DOWN, DOUBLON``;
    pass those to :func:`canonical_hubbard_child_merge_sign` instead.  In
    particular, the token ``DOWN == 2`` is not an occupation count of two.
    """

    raw_counts = tuple(occupation_counts)
    if not raw_counts:
        raise ValueError("occupation_counts must be nonempty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_counts
    ):
        raise TypeError("occupation_counts must contain integers")
    checked_counts = tuple(int(value) for value in raw_counts)
    if any(value < 0 or value > 2 for value in checked_counts):
        raise ValueError("onsite Hubbard occupations must lie in {0,1,2}")
    left, right = validated_child_merge_sites(
        left_sites, right_sites, len(checked_counts)
    )
    parity = sum(
        checked_counts[left_site] * checked_counts[right_site]
        for left_site in left
        for right_site in right
        if left_site > right_site
    )
    return -1 if parity % 2 else 1


def canonical_hubbard_child_merge_sign_jax(
    left_sites: Sequence[int] | jax.Array,
    right_sites: Sequence[int] | jax.Array,
    tokens: jax.Array,
) -> jax.Array:
    """Pure-JAX merge sign from valid physical Hubbard tokens.

    The explicit token-to-count map prevents the numerical value
    ``DOWN == 2`` from being confused with a doublon.  As with
    :func:`canonical_child_merge_sign_jax`, supports must first be host
    validated and token validity is an upstream runtime invariant.
    """

    token_array = jnp.asarray(tokens)
    if token_array.ndim < 1:
        raise ValueError("tokens must have a site axis")
    if not jnp.issubdtype(token_array.dtype, jnp.integer):
        raise TypeError("tokens must have an integer dtype")
    counts = (
        (token_array == UP).astype(jnp.int32)
        + (token_array == DOWN).astype(jnp.int32)
        + 2 * (token_array == DOUBLON).astype(jnp.int32)
    )
    return canonical_child_merge_sign_jax(left_sites, right_sites, counts)


def canonical_hubbard_child_merge_sign(
    left_sites: Sequence[int],
    right_sites: Sequence[int],
    tokens: Sequence[int],
) -> int:
    """Host-validated merge sign from physical Hubbard tokens."""

    checked_tokens = tuple(validate_hubbard_token(token) for token in tokens)
    if not checked_tokens:
        raise ValueError("tokens must be nonempty")
    occupation_counts = tuple(
        0 if token == EMPTY else 2 if token == DOUBLON else 1
        for token in checked_tokens
    )
    return canonical_child_merge_sign(
        left_sites, right_sites, occupation_counts
    )


__all__ = [
    "canonical_child_merge_sign",
    "canonical_child_merge_sign_jax",
    "canonical_hubbard_child_merge_sign",
    "canonical_hubbard_child_merge_sign_jax",
    "validated_child_merge_sites",
]
