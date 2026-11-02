"""Independent NumPy propagation of the fixed K1/K2 causal carrier.

This is the reference implementation for the JAX autoregressive decoder.  It
propagates the first-order endpoint vectors and the unordered second-order
``q_op=0,2,4`` endpoint pairs using only the compiled numerical tables.  It
does not call the neural-model implementation and contains no trainable
quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..su2.fusion_path import is_valid_path
from .second_order_causal_plan import SecondOrderCausalPairPlan
from .sparse_causal_recurrence import SparseCausalRecurrencePlan


@dataclass(frozen=True)
class ExactSecondOrderCarrierTrace:
    """K1 endpoints, closed scalar, and K2 pairs after each decision."""

    endpoint: np.ndarray
    closed_scalar: np.ndarray
    pair: np.ndarray


def _decision_q_values(
    path: tuple[int, ...], decision: int, n_sites: int
) -> tuple[int, tuple[int, ...]]:
    q_before = int(path[n_sites - 1 - decision])
    if decision < n_sites - 2:
        return q_before, (int(path[n_sites - 2 - decision]),)
    return q_before, (1, 0)


def propagate_exact_second_order_carrier(
    first_order: SparseCausalRecurrencePlan,
    second_order: SecondOrderCausalPairPlan,
    path: tuple[int, ...],
    *,
    coefficient_scale: int = 1,
) -> ExactSecondOrderCarrierTrace:
    """Propagate one valid left-associated singlet path with NumPy."""

    if first_order.n_sites != second_order.n_sites:
        raise ValueError("first- and second-order plan sizes disagree")
    if first_order.target_q != second_order.target_q:
        raise ValueError("first- and second-order target q disagree")
    if coefficient_scale not in (0, 1) or isinstance(coefficient_scale, bool):
        raise ValueError("coefficient_scale must be the integer 0 or 1")
    if bool(getattr(second_order, "trainable_pair_weights", False)):
        raise ValueError("the exact K2 reference cannot use trainable pair weights")
    if not bool(getattr(second_order, "unordered_endpoint_pairs", False)):
        raise ValueError("the exact K2 reference requires unordered pairs")

    path = tuple(int(q) for q in path)
    if len(path) != first_order.n_sites:
        raise ValueError(f"expected a length-{first_order.n_sites} fusion path")
    if not is_valid_path(path, target_q=first_order.target_q):
        raise ValueError("path is not a valid spin-half fusion path")

    endpoint = np.zeros(
        (first_order.endpoint_slot_count, len(first_order.delta_q)),
        dtype=np.float64,
    )
    pair = np.zeros(
        (
            second_order.pair_slot_count,
            len(second_order.rank_q),
            len(second_order.delta_q),
        ),
        dtype=np.float64,
    )
    closed = 0.0
    endpoint_trace = []
    pair_trace = []
    closed_trace = []
    scale = float(coefficient_scale)

    for decision in range(first_order.n_decisions):
        q_before, q_after_values = _decision_q_values(
            path, decision, first_order.n_sites
        )
        microsteps = np.flatnonzero(first_order.event_valid[decision])
        if len(microsteps) != len(q_after_values):
            raise RuntimeError("recurrence event count disagrees with tree decision")
        current_q = q_before
        for microstep, q_after in zip(microsteps, q_after_values, strict=True):
            microstep = int(microstep)
            if not first_order.q_transition_valid[
                decision, microstep, current_q, q_after
            ]:
                raise RuntimeError("fusion path requests an invalid transition")

            old_endpoint = endpoint
            old_pair = pair

            old_pair_slots = second_order.persist_old_pair_slot_for_new[
                decision, microstep
            ]
            safe_pair_slots = np.clip(
                old_pair_slots, 0, second_order.pair_slot_count - 1
            )
            gathered_pair = old_pair[safe_pair_slots] * (
                old_pair_slots >= 0
            )[:, None, None]
            tensor_transport = second_order.tensor_transport[
                decision, microstep, current_q, q_after
            ]
            pair = np.einsum(
                "rab,wrb->wra", tensor_transport, gathered_pair
            )

            first_slots = second_order.first_old_endpoint_slot_for_new[
                decision, microstep
            ]
            second_slots = second_order.second_old_endpoint_slot_for_new[
                decision, microstep
            ]
            safe_first = np.clip(
                first_slots, 0, first_order.endpoint_slot_count - 1
            )
            safe_second = np.clip(
                second_slots, 0, first_order.endpoint_slot_count - 1
            )
            first_vector = old_endpoint[safe_first] * (
                first_slots >= 0
            )[:, None]
            second_vector = old_endpoint[safe_second] * (
                second_slots >= 0
            )[:, None]
            left_source = second_order.left_vector_source[
                decision, microstep, current_q, q_after
            ]
            right_source = second_order.right_vector_source[
                decision, microstep, current_q, q_after
            ]
            pair += scale * second_order.second_source_coupling[
                decision, microstep, :, None, None
            ] * np.einsum("rav,wv->wra", left_source, first_vector)
            pair += scale * second_order.first_source_coupling[
                decision, microstep, :, None, None
            ] * np.einsum("rav,wv->wra", right_source, second_vector)

            diagonal = second_order.pair_diagonal_mask[
                decision, microstep
            ]
            local_factor = np.where(
                diagonal[:, None],
                1.0,
                second_order.offdiagonal_same_spin_factor[None, :],
            )
            local_source = second_order.appended_spin_ordered_source[
                decision, microstep, current_q, q_after
            ]
            pair += scale**2 * second_order.source_coupling_product[
                decision, microstep, :, None, None
            ] * local_factor[:, :, None] * local_source[None, :, :]
            pair *= second_order.new_pair_mask[
                decision, microstep, :, None, None
            ]
            pair *= second_order.physical_rank_mask[
                decision, microstep, :, :, None
            ]
            pair *= second_order.rank_delta_mask[None, :, :]

            old_slots = first_order.persist_old_slot_for_new[
                decision, microstep
            ]
            safe_slots = np.clip(
                old_slots, 0, first_order.endpoint_slot_count - 1
            )
            gathered = old_endpoint[safe_slots] * (old_slots >= 0)[:, None]
            transport = first_order.old_block_transport[
                decision, microstep, current_q, q_after
            ]
            endpoint = np.einsum("ab,wb->wa", transport, gathered)
            endpoint += scale * first_order.source_coupling[
                decision, microstep, :, None
            ] * first_order.endpoint_source[
                decision, microstep, current_q, q_after
            ][None, :]
            endpoint *= first_order.new_endpoint_mask[
                decision, microstep, :, None
            ]

            if first_order.death_mask[decision, microstep]:
                dying_slot = int(first_order.dying_old_slot[decision, microstep])
                closed += scale * float(
                    np.dot(
                        old_endpoint[dying_slot],
                        first_order.dying_scalar[
                            decision, microstep, current_q, q_after
                        ],
                    )
                )
            current_q = q_after

        endpoint_trace.append(endpoint.copy())
        pair_trace.append(pair.copy())
        closed_trace.append(closed)

    return ExactSecondOrderCarrierTrace(
        endpoint=np.asarray(endpoint_trace, dtype=np.float64),
        closed_scalar=np.asarray(closed_trace, dtype=np.float64),
        pair=np.asarray(pair_trace, dtype=np.float64),
    )


__all__ = [
    "ExactSecondOrderCarrierTrace",
    "propagate_exact_second_order_carrier",
]
