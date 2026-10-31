"""Two-environment scalar gates on the existing three-cell query support.

On disjoint four-spin cells A/B/C, K1 conditions active B on (A,C), and K2
conditions active C on (B,A). Both gates read complete(q,a) projectors. The
product is evaluated coherently, not by assigning a measured environment.

    K2 K1 = sum_(a,b,c) P_a^A tensor [P_b U_B(a,c)]
                                  tensor [U_C(b,a) P_c].

Each cell's magnetic projection is preserved, so the original216 terminal
inputs and their exact carrier-query cache remain valid. There is no new
global-fusion transform or enlarged carrier-query support. Spatial Reynolds
projection occurs once, after both gates, in the amplitude wrapper.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from ..models.conditional_carrier_resonance import local_scalar_components
from ..models.multicontext_resonance_attention import multicontext_active_filters
from .overlapping_carrier_queries import OverlappingCarrierQueryPlan, local_overlap_connections


@lru_cache(maxsize=1)
def _cell_alternatives() -> tuple[np.ndarray, np.ndarray]:
    """Extract the canonical cell ordering from the existing public row map."""
    choices = np.zeros((16, 6), dtype=np.int32)
    valid = np.zeros((16, 6), dtype=bool)
    for row in range(16):
        columns, mask, _ = local_overlap_connections(256 * row)
        alternatives = columns[mask] // 256  # B=C=0000 have exactly one input.
        choices[row, :len(alternatives)] = alternatives
        valid[row, :len(alternatives)] = True
    choices.setflags(write=False)
    valid.setflags(write=False)
    return choices, valid


def multicontext_overlap_filter_rows(
    tables1: jax.Array, tables2: jax.Array, local_rows: jax.Array,
    *, mode: str = "cascade",
) -> jax.Array:
    """Dynamic ``[row,216]`` entries of K2 K1 or K1+K2-I.

    Tables contain real reduced GENERATORS, shape(6,6,10), ordered
    ``tables1[a,c]`` and ``tables2[b,a]``. Local row index is256*A+16*B+C;
    packed columns exactly match ``local_overlap_connections``. The centered
    I+Delta1+Delta2(+Delta2 Delta1) form gives bitwise identity at zero tables.
    This is a local contraction, not a Hamiltonian or sampling routine.
    """
    if mode not in ("cascade", "no_cascade"):
        raise ValueError("unknown multicontext composition mode")
    rows = jnp.asarray(local_rows)
    if rows.ndim != 1 or rows.size < 1 or not jnp.issubdtype(rows.dtype, jnp.integer):
        raise ValueError("local_rows must be a nonempty integer vector")
    if not isinstance(rows, jax.core.Tracer) and (np.any(np.asarray(rows) < 0) or np.any(np.asarray(rows) >= 4096)):
        raise ValueError("local_rows must contain12-bit configurations")
    alternatives, valid = map(jnp.asarray, _cell_alternatives())
    a, b, c = rows // 256, (rows // 16) % 16, rows % 16
    av, bv, cv = alternatives[a], alternatives[b], alternatives[c]
    ma, mb, mc = valid[a], valid[b], valid[c]
    projectors = jnp.asarray(local_scalar_components().spectator_projectors)
    delta1 = multicontext_active_filters(tables1) - jnp.eye(16, dtype=jnp.float64)
    delta2 = multicontext_active_filters(tables2) - jnp.eye(16, dtype=jnp.float64)
    identity = ((av == a[:, None])[:, :, None, None]
                & (bv == b[:, None])[:, None, :, None]
                & (cv == c[:, None])[:, None, None, :])
    result = identity.astype(jnp.float64)
    pb = projectors[:, b[:, None], bv]
    pc = projectors[:, c[:, None], cv]
    for label_a in range(6):
        pa = projectors[label_a, a[:, None], av]
        # Single-gate terms use complete coherent sums over the other label.
        ub = delta1[label_a][:, b[:, None], bv]
        uc = delta2[:, label_a][:, c[:, None], cv]
        local_bc = jnp.einsum("unb,unc->nbc", ub, pc) + jnp.einsum("unb,unc->nbc", pb, uc)
        if mode == "cascade":
            # Precontract16x16 products before gathering packed row entries.
            left = jnp.einsum("uij,vjk->uvik", projectors, delta1[label_a])
            right = jnp.einsum("uij,vjk->uvik", delta2[:, label_a], projectors)
            left_rows = left[:, :, b[:, None], bv]
            right_rows = right[:, :, c[:, None], cv]
            local_bc = local_bc + jnp.einsum("uvnb,uvnc->nbc", left_rows, right_rows)
        result = result + pa[:, :, None, None] * local_bc[:, None, :, :]
    mask = ma[:, :, None, None] & mb[:, None, :, None] & mc[:, None, None, :]
    return jnp.where(mask, result, 0.).reshape(len(rows), 216)


def multicontext_carrier_amplitudes(
    tables1: jax.Array, tables2: jax.Array, plan: OverlappingCarrierQueryPlan,
    unique_carrier_amplitudes: jax.Array, *, mode: str = "cascade",
) -> jax.Array:
    """Reuse the old exact query plan and apply the full group only once.

    The carrier cache is reusable only for a fixed carrier. At zero tables
    this returns P_A1 C; equality to C requires the carrier to already be A1.
    No intermediate spatial projector or context measurement is inserted.
    """
    amplitudes = jnp.asarray(unique_carrier_amplitudes)
    if amplitudes.shape != (len(plan.configurations),):
        raise ValueError("one amplitude per unique carrier query is required")
    weights = multicontext_overlap_filter_rows(tables1, tables2,
                                               jnp.asarray(plan.unique_local_rows), mode=mode)
    source = amplitudes[jnp.asarray(plan.connection_indices)]
    selected = weights[jnp.asarray(plan.local_row_indices)]
    return jnp.mean(jnp.sum(selected * source, axis=-1), axis=1)


__all__ = ["multicontext_overlap_filter_rows", "multicontext_carrier_amplitudes"]
