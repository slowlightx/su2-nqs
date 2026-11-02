"""Exact uncompressed scalar-width carrier for sparse causal SU(2) plans.

The sparse neural recurrence deliberately compresses a rank-one boundary
operator into learned hidden vectors.  This module removes that learned
compression and propagates the numerical Wigner--Eckart carrier itself.  For
one ket fusion path, endpoint entries collect all compatible bra-prefix
matrix elements with a fixed boundary ``delta_q``.  A dying endpoint is
contracted into the revealed scalar sector.

The carrier is not a wavefunction model: it supplies the Hamiltonian side of
the physically useful factorisation ``exact local carrier x learned causal
environment``.  In particular, its final scalar is a linear functional of a
Hamiltonian column, not a variational amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..su2.fusion_path import is_valid_path
from .sparse_causal_recurrence import SparseCausalRecurrencePlan


@dataclass(frozen=True)
class ExactCausalCarrierTrace:
    """Carrier values committed after every autoregressive decision."""

    endpoint: np.ndarray
    closed_scalar: np.ndarray


def _decision_q_values(
    path: tuple[int, ...], decision: int, n_sites: int
) -> tuple[int, tuple[int, ...]]:
    """Return doubled ``q_before`` and per-microstep ``q_after`` values."""

    q_before = int(path[n_sites - 1 - decision])
    if decision < n_sites - 2:
        return q_before, (int(path[n_sites - 2 - decision]),)
    # The final compact-tree decision resolves the two remaining spins.  The
    # intermediate one-spin block has q=1 and the complete singlet has q=0.
    return q_before, (1, 0)


def propagate_exact_causal_carrier(
    plan: SparseCausalRecurrencePlan,
    path: tuple[int, ...],
) -> ExactCausalCarrierTrace:
    """Propagate the exact local carrier along one left-associated singlet path.

    All angular momenta are doubled integers.  ``path`` has the repository's
    standard length-``N`` convention.  No fusion paths are enumerated and no
    learned parameter enters this calculation.
    """

    if bool(getattr(plan, "uses_hidden_suffix", False)):
        raise ValueError("an exact causal carrier plan may not use a hidden suffix")
    if bool(getattr(plan, "enumerates_revealed_prefixes", False)):
        raise ValueError("an exact causal carrier plan may not enumerate prefixes")

    path = tuple(int(q) for q in path)
    if len(path) != plan.n_sites:
        raise ValueError(f"expected a length-{plan.n_sites} fusion path")
    if not is_valid_path(path, target_q=plan.target_q):
        raise ValueError("path is not a valid spin-half fusion path")

    endpoint = np.zeros(
        (plan.endpoint_slot_count, len(plan.delta_q)), dtype=np.float64
    )
    closed = 0.0
    endpoint_trace = []
    closed_trace = []

    for decision in range(plan.n_decisions):
        q_before, q_after_values = _decision_q_values(
            path, decision, plan.n_sites
        )
        valid_microsteps = np.flatnonzero(plan.event_valid[decision])
        if len(valid_microsteps) != len(q_after_values):
            raise RuntimeError("recurrence event count disagrees with tree decision")
        current_q = q_before
        for microstep, q_after in zip(
            valid_microsteps, q_after_values, strict=True
        ):
            microstep = int(microstep)
            if not plan.q_transition_valid[
                decision, microstep, current_q, q_after
            ]:
                raise RuntimeError("fusion path requests an invalid recurrence transition")

            old_endpoint = endpoint
            old_slots = plan.persist_old_slot_for_new[decision, microstep]
            safe_slots = np.clip(old_slots, 0, plan.endpoint_slot_count - 1)
            persistent = old_slots >= 0
            gathered = old_endpoint[safe_slots] * persistent[:, None]
            transport = plan.old_block_transport[
                decision, microstep, current_q, q_after
            ]
            endpoint = np.einsum("ab,wb->wa", transport, gathered)
            endpoint += (
                plan.source_coupling[decision, microstep, :, None]
                * plan.endpoint_source[
                    decision, microstep, current_q, q_after
                ][None, :]
            )
            endpoint *= plan.new_endpoint_mask[
                decision, microstep, :, None
            ]

            if plan.death_mask[decision, microstep]:
                dying_slot = int(plan.dying_old_slot[decision, microstep])
                closed += float(
                    np.dot(
                        old_endpoint[dying_slot],
                        plan.dying_scalar[
                            decision, microstep, current_q, q_after
                        ],
                    )
                )
            current_q = q_after

        endpoint_trace.append(endpoint.copy())
        closed_trace.append(closed)

    return ExactCausalCarrierTrace(
        endpoint=np.asarray(endpoint_trace, dtype=np.float64),
        closed_scalar=np.asarray(closed_trace, dtype=np.float64),
    )


__all__ = ["ExactCausalCarrierTrace", "propagate_exact_causal_carrier"]
