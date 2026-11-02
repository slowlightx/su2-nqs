"""Exact local row replacements of closed-shell Slater amplitudes.

The host plan records only discrete row orderings.  Evaluation is pure JAX and
differentiable in an occupied-orbital matrix ``V[N, N/2]``.  Each nonsingular
reference needs two factorizations; each requested spin configuration then
needs two determinants of at most ``max_rank`` rows.  Singular or poorly
conditioned references use the original, dense determinant, *not* an inverse
regularization or an amplitude floor.

This module does not assert spatial invariance of an individual determinant.
Any coherent carrier sum or spatial projection remains the caller's job.
Bits are zero for down, one for up; sites are ordered increasingly in each
Slater matrix.  For symmetric pairing ``F = V @ V.T`` (no conjugation), the
physical singlet amplitude is ``(-1)**sum(up_sites) det(V_up) det(V_down)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SlaterMinorPlan:
    """Static, immutable replacement data, closed over when JIT compiling.

    Spin axis zero means up, axis one means down.  ``replacement_positions``
    refer to the *reference Slater row order*, not physical site addresses.
    ``sorting_signs`` restore sorted target row order after the replacements.
    The same plan is reusable for every occupied-orbital component.
    """

    reference_bits: np.ndarray
    target_bits: np.ndarray
    reference_indices: np.ndarray
    reference_rows: np.ndarray
    added_sites: np.ndarray
    replacement_positions: np.ndarray
    active: np.ndarray
    sorting_signs: np.ndarray
    ranks: np.ndarray
    physical_signs: np.ndarray
    max_rank: int

    @property
    def n_sites(self) -> int:
        return int(self.reference_bits.shape[1])

    @property
    def numeric_plan_bytes(self) -> int:
        """Retained arrays only, excluding Python/JAX runtime memory."""
        return sum(value.nbytes for value in vars(self).values()
                   if isinstance(value, np.ndarray))


class SlaterMinorResult(NamedTuple):
    """Unnormalised, signed-row-order determinants and routing diagnostics.

    ``determinants[query, spin]`` are *not ratios*.  ``dense_fallback`` has the
    same first two axes and is componentwise, independent of coherent-carrier
    cancellation.  A condition limit only chooses an exact evaluation route.
    """

    determinants: jax.Array
    dense_fallback: jax.Array
    reference_conditions: jax.Array
    ranks: jax.Array


def _validated_bits(value: np.ndarray, *, name: str) -> np.ndarray:
    bits = np.asarray(value)
    if (bits.ndim != 2 or len(bits) == 0 or bits.shape[1] < 2
            or bits.shape[1] % 2 or np.any((bits != 0) & (bits != 1))
            or np.any(bits.sum(axis=1) != bits.shape[1] // 2)):
        raise ValueError(f"{name} must be a nonempty half-filled binary matrix")
    return np.asarray(bits, dtype=np.int8).copy()


def compile_slater_minor_plan(
    reference_bits: np.ndarray,
    target_bits: np.ndarray,
    reference_indices: np.ndarray | None = None,
    *,
    max_rank: int = 6,
) -> SlaterMinorPlan:
    """Compile exact spin-sector row replacements without orbital evaluation.

    Each target may select any reference via ``reference_indices``.  Omitting
    those indices is allowed only for a single reference.  Rank exceeds the
    declared bound => error; queries are never truncated or discarded.  Rank
    six covers rearrangements in twelve sites, and seven can include an extra
    Hamiltonian exchange.  No physical site count or integer-bitword limit is
    imposed.  For large query corpora, compile bounded batches.
    """
    refs = _validated_bits(reference_bits, name="reference_bits")
    targets = _validated_bits(target_bits, name="target_bits")
    if refs.shape[1] != targets.shape[1]:
        raise ValueError("reference and target site counts differ")
    half = refs.shape[1] // 2
    if (isinstance(max_rank, bool) or not isinstance(max_rank, Integral)
            or max_rank < 0 or max_rank > half):
        raise ValueError("max_rank must be an integer between zero and N/2")
    if reference_indices is None:
        if len(refs) != 1:
            raise ValueError("multiple references require reference_indices")
        indices = np.zeros(len(targets), dtype=np.int32)
    else:
        indices = np.asarray(reference_indices)
        if (indices.shape != (len(targets),)
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(refs))):
            raise ValueError("invalid target-to-reference indices")
        indices = np.asarray(indices, dtype=np.int32).copy()
    reference_rows = np.stack([
        np.stack([np.flatnonzero(row), np.flatnonzero(1-row)])
        for row in refs
    ]).astype(np.int32)
    # A one-dimensional identity also represents a declared rank-zero plan.
    width = max(1, int(max_rank))
    added = np.zeros((len(targets), 2, width), dtype=np.int32)
    positions = np.zeros_like(added)
    active = np.zeros_like(added, dtype=bool)
    signs = np.ones((len(targets), 2), dtype=np.int8)
    ranks = np.zeros((len(targets), 2), dtype=np.int32)
    # The largest temporary is chunk x two spins x half x half for sorting
    # parity.  This avoids both one Python loop per carrier query and an
    # unbounded query-count-squared-row temporary for large projection plans.
    spin_values = np.asarray([1, 0], dtype=np.int8)[None, :, None]
    site_order = np.arange(refs.shape[1], dtype=np.int32)
    row_order = np.arange(half, dtype=np.int32)
    for start in range(0, len(targets), 4096):
        stop = min(start+4096, len(targets))
        selected = indices[start:stop]
        old = reference_rows[selected]
        bits = targets[start:stop]
        query_axis = np.arange(stop-start)[:, None, None]
        removed = bits[query_axis, old] != spin_values
        changed_rank = removed.sum(axis=-1)
        if np.any(changed_rank > max_rank):
            local_query, spin = np.argwhere(changed_rank > max_rank)[0]
            raise ValueError(f"query {start+local_query} replacement rank "
                             f"{changed_rank[local_query, spin]} exceeds {max_rank}")
        insert_mask = ((bits[:, None, :] == spin_values)
                       & (refs[selected, None, :] != spin_values))
        candidate_positions = np.sort(np.where(removed, row_order, half), axis=-1)[..., :width]
        candidate_added = np.sort(np.where(insert_mask, site_order, refs.shape[1]), axis=-1)[..., :width]
        enabled = np.arange(width)[None, None, :] < changed_rank[..., None]
        # A padding index is zero only in masked-out positions.
        positions[start:stop] = np.where(enabled, candidate_positions, 0)
        added[start:stop] = np.where(enabled, candidate_added, 0)
        active[start:stop] = enabled
        ranks[start:stop] = changed_rank
        patched = old.copy()
        q, s, k = np.nonzero(enabled)
        patched[q, s, candidate_positions[q, s, k]] = candidate_added[q, s, k]
        inversions = np.count_nonzero(np.triu(patched[..., :, None] > patched[..., None, :], 1), axis=(-2, -1))
        signs[start:stop] = np.where(inversions % 2, -1, 1)
    physical_signs = np.where((targets @ np.arange(refs.shape[1])) % 2, -1, 1).astype(np.int8)
    arrays = (refs, targets, indices, reference_rows, added, positions, active,
              signs, ranks, physical_signs)
    for array in arrays:
        array.setflags(write=False)
    return SlaterMinorPlan(*arrays, int(max_rank))


@jax.custom_jvp
def _node_safe_determinant(matrix: jax.Array) -> jax.Array:
    """Determinant with a cofactor JVP at a zero, including rank n-1 nodes.

    Some JAX versions return a zero tangent at singular matrices of size > 3.
    The determinant polynomial instead has a possibly nonzero cofactor
    derivative.  The uncommon singular branch evaluates those cofactors
    explicitly and never solves against the singular matrix.
    """
    return jnp.linalg.det(matrix)


@_node_safe_determinant.defjvp
def _determinant_jvp(primals, tangents):
    (matrix,), (direction,) = primals, tangents
    size = matrix.shape[-1]
    value = _node_safe_determinant(matrix)
    if size == 1:
        return value, direction[0, 0]

    def regular(_):
        # Under vmap, cond may become a select.  Even an unselected branch
        # must then avoid solving a singular matrix (otherwise reverse-mode
        # can receive NaNs through a masked intermediate).
        safe = jnp.where(value != 0, matrix, jnp.eye(size, dtype=matrix.dtype))
        return value * jnp.linalg.inv(safe).T

    def singular(_):
        keep = np.asarray([[j for j in range(size) if j != i]
                           for i in range(size)], dtype=np.int32)
        rows = jnp.asarray(keep)[:, None, :, None]
        columns = jnp.asarray(keep)[None, :, None, :]
        minors = matrix[rows, columns]
        cofactors = jnp.linalg.det(minors)
        parity = jnp.asarray(np.where((np.arange(size)[:, None]
                                       + np.arange(size)[None, :]) % 2, -1, 1))
        return cofactors * parity

    cofactors = jax.lax.cond(value != 0, regular, singular, operand=None)
    # Keeping the tangent-linear contraction outside cond also makes its
    # transpose explicit; no branch depends on a tangent value.
    return value, jnp.sum(cofactors * direction)


def slater_minor_updates(
    orbitals: jax.Array,
    plan: SlaterMinorPlan,
    *,
    condition_limit: float = 1.0e9,
) -> SlaterMinorResult:
    """Evaluate all target up/down Slaters using exact bounded-rank minors.

    This function may be closed over a plan and passed to ``jax.jit``/``jvp``.
    It supports real or complex orbitals; matrix products do not conjugate.
    Routing conditions are stop-gradient diagnostics, not a modification to
    the physical polynomial.  Unsafe references are replaced by identity
    *only in unused inverse data*, and every affected target determinant is
    evaluated directly.  A global fast branch avoids all dense target work
    when every used reference component is well-conditioned.
    """
    v = jnp.asarray(orbitals)
    n = plan.n_sites
    if v.shape != (n, n // 2) or not jnp.issubdtype(v.dtype, jnp.inexact):
        raise ValueError("orbitals must be a real/complex floating [N,N/2] matrix")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    reference_matrices = v[jnp.asarray(plan.reference_rows)]
    singular_values = jnp.linalg.svd(jax.lax.stop_gradient(reference_matrices), compute_uv=False)
    conditions = jnp.where(singular_values[..., -1] > 0,
                           singular_values[..., 0] / singular_values[..., -1], jnp.inf)
    regular = jnp.isfinite(conditions) & (conditions <= condition_limit)
    safe_matrices = jnp.where(regular[..., None, None], reference_matrices,
                              jnp.eye(n // 2, dtype=v.dtype))
    inverses = jnp.linalg.inv(safe_matrices)
    ratios = jnp.einsum("nm,bsmp->bsnp", v, inverses)
    determinants = jax.vmap(jax.vmap(_node_safe_determinant))(safe_matrices)
    ref = jnp.asarray(plan.reference_indices)
    spin = jnp.arange(2)[None, :, None, None]
    minor = ratios[ref[:, None, None, None], spin,
                   jnp.asarray(plan.added_sites)[..., :, None],
                   jnp.asarray(plan.replacement_positions)[..., None, :]]
    active = jnp.asarray(plan.active)
    keep = active[..., :, None] & active[..., None, :]
    minor = jnp.where(keep, minor, jnp.eye(active.shape[-1], dtype=v.dtype))
    minor_determinants = jax.vmap(jax.vmap(_node_safe_determinant))(minor)
    fast = (minor_determinants * determinants[ref]
            * jnp.asarray(plan.sorting_signs))
    fallback = ~regular[ref]

    def with_fallback(_):
        # lax.map, not vmap(cond): the latter can evaluate dense determinants
        # for every query even when only one component needs the fallback.
        def one(query):
            index, target = query
            def spin_value(s):
                def dense(_):
                    rows = jnp.argsort(jnp.where(target == 1-s, 0, 1), stable=True)[:n // 2]
                    return _node_safe_determinant(v[rows])
                return jax.lax.cond(fallback[index, s], dense,
                                    lambda _: fast[index, s], operand=None)
            return jax.lax.map(spin_value, jnp.arange(2))
        return jax.lax.map(one, (jnp.arange(len(plan.target_bits)), jnp.asarray(plan.target_bits)))

    values = jax.lax.cond(jnp.any(fallback), with_fallback, lambda _: fast, operand=None)
    return SlaterMinorResult(values, fallback, conditions, jnp.asarray(plan.ranks))


def closed_shell_amplitudes(result: SlaterMinorResult, plan: SlaterMinorPlan) -> jax.Array:
    """Combine up/down determinants with physical singlet fermionic parity.

    No coherent-carrier division, clipping, floor, or normalization is used.
    The caller must combine independently evaluated orbital components using
    their actual complex coefficients; individual components need not be A1.
    """
    if result.determinants.shape != (len(plan.target_bits), 2):
        raise ValueError("result and plan query dimensions differ")
    return jnp.prod(result.determinants, axis=1) * jnp.asarray(plan.physical_signs)
