"""Point amplitudes of a few physical SU(2) four-spin multiplicity gates.

This isolated adapter acts BEFORE any spatial projection. It reuses the
validated SO(2) x SO(3) generators, never learns magnetic-index entries, and
never constructs a full-system state. Physical matrices use U[new, old].
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .multiplicity_gate import plaquette_generators


@lru_cache(maxsize=1)
def _physical_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Static numerical CG data; no symbolic work occurs in JAX runtime."""
    generators = np.asarray(plaquette_generators())
    if np.max(abs(generators.imag)) > 1e-13:
        raise ArithmeticError('the validated real gate family acquired a complex generator')
    generators = generators.real
    powers = generators @ generators
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int32)
    old = np.empty((16, 6), np.int32)
    valid = np.zeros((16, 6), bool)
    for new in range(16):
        allowed = np.flatnonzero(bits.sum(1) == bits[new].sum())
        old[new] = new  # padded terms remain valid physical configurations
        old[new, :len(allowed)] = allowed
        valid[new, :len(allowed)] = True
    return generators, powers, bits, old, valid


def jax_plaquette_unitary(angles: jax.Array) -> jax.Array:
    """Return the real16x16 U[new,old] for four finite real angles.

    Angles are (q0_01,q2_01,q2_02,q2_12), with doubled q=0,2,4. The
    q=2 product is R12 @ R02 @ R01, exactly as in multiplicity_gate.py.
    I+sin(theta)G+(1-cos(theta))G² makes zero angles EXACTLY identity while
    retaining the nonzero identity JVP. Float64 is used when JAX x64 is on.
    """
    values = jnp.asarray(angles)
    if values.shape != (4,) or jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise ValueError('four real angles are required')
    if not isinstance(values, jax.core.Tracer) and not np.isfinite(np.asarray(values)).all():
        raise ValueError('four finite real angles are required')
    values = values.astype(jnp.result_type(values, jnp.float64))
    generators, squares, _, _, _ = _physical_tables()
    generators, squares = jnp.asarray(generators, values.dtype), jnp.asarray(squares, values.dtype)
    identity = jnp.eye(16, dtype=values.dtype)
    result = identity
    for index in range(4):
        rotation = identity+jnp.sin(values[index])*generators[index]+(1-jnp.cos(values[index]))*squares[index]
        result = rotation @ result
    return result


def make_four_spin_gate_expansion(
    n_sites: int,
    ordered_site_groups: Sequence[Sequence[int]],
    *,
    max_expansion_terms: int = 216,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Create expand(angles[g,4],bits[N]) -> (old_rows[6**g,N],weights).

    Gates act on the ket in the supplied order, so expansion traverses them
    last-to-first. Ordered site zero is the MOST significant local bit.
    Equal local doubled-m support gives at most six predecessors per gate;
    padding has exactly zero weight. Overlapping gates are allowed, but cost
    is exponential in the number of gates. This is for a few gates, not a
    scalable circuit contraction. No spatial projection is performed here.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 4:
        raise ValueError('n_sites must be an integer at least four')
    if (isinstance(max_expansion_terms, bool) or not isinstance(max_expansion_terms, Integral)
            or max_expansion_terms < 1):
        raise ValueError('max_expansion_terms must be a positive integer')
    groups = tuple(tuple(group) for group in ordered_site_groups)
    for sites in groups:
        if (len(sites) != 4 or len(set(sites)) != 4
                or any(isinstance(site, bool) or not isinstance(site, Integral)
                       or not 0 <= site < n_sites for site in sites)):
            raise ValueError('each gate needs four distinct ordered integer sites in range')
    if 6**len(groups) > max_expansion_terms:
        raise ValueError('gate expansion exceeds the declared term budget')
    _, _, bit_table, old_table, valid_table = _physical_tables()
    bit_table, old_table, valid_table = map(jnp.asarray, (bit_table, old_table, valid_table))
    local_weights = jnp.asarray([8, 4, 2, 1], jnp.int32)

    def expand(angles: jax.Array, configuration: jax.Array) -> tuple[jax.Array, jax.Array]:
        values, row = jnp.asarray(angles), jnp.asarray(configuration)
        if values.shape != (len(groups), 4) or jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise ValueError('angles must have shape (number_of_gates,4) and be real')
        if row.shape != (n_sites,) or not jnp.issubdtype(row.dtype, jnp.integer):
            raise ValueError('configuration must be an integer bit vector of length n_sites')
        if not isinstance(row, jax.core.Tracer) and not np.isin(np.asarray(row), (0, 1)).all():
            raise ValueError('configuration entries must be zero or one')
        matrices = jax.vmap(jax_plaquette_unitary)(values)
        rows = row[None, :]
        weights = jnp.ones(1, dtype=matrices.dtype)
        for index in range(len(groups)-1, -1, -1):
            sites = jnp.asarray(groups[index], jnp.int32)
            new = rows[:, sites] @ local_weights
            old = old_table[new]
            expanded = jnp.broadcast_to(rows[:, None, :], (rows.shape[0], 6, n_sites))
            expanded = expanded.at[:, :, sites].set(bit_table[old])
            factors = matrices[index][new[:, None], old]*valid_table[new]
            rows = expanded.reshape(-1, n_sites)
            weights = (weights[:, None]*factors).reshape(-1)
        return rows, weights

    return expand


def make_four_spin_gated_amplitude(
    raw_amplitude: Callable[[Any, jax.Array], jax.Array],
    n_sites: int,
    ordered_site_groups: Sequence[Sequence[int]],
    *,
    max_expansion_terms: int = 216,
) -> Callable[[Any, jax.Array, jax.Array], jax.Array]:
    """Wrap raw(params,bits) as <bits|U_last...U_first|raw(params)>.

    Call the complete spatial projector OUTSIDE this function to obtain
    P_A1 U|raw>; applying U to an already projected parent is a different state.
    Both gate angles and raw parameters remain JAX differentiable.
    """
    expand = make_four_spin_gate_expansion(n_sites, ordered_site_groups,
                                          max_expansion_terms=max_expansion_terms)

    def amplitude(params: Any, angles: jax.Array, configuration: jax.Array) -> jax.Array:
        rows, weights = expand(angles, configuration)
        values = jax.vmap(lambda row: raw_amplitude(params, row))(rows)
        return jnp.sum(weights*values)

    return amplitude
