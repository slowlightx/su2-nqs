"""Fixed-shape stable endpoint-pair plan for the SU(2) K2 boundary layer.

For active first-order endpoint vectors ``B_i``, diagonal pair slots carry
the ordered tensor ``[B_i x B_i]^Q``.  An off-diagonal slot ``i < k`` carries
the Jordan tensor

``([B_i x B_k]^Q + (-1)^(Q/2) [B_k x B_i]^Q) / 2``.

This halves the ordered endpoint-pair burden without approximation because
the hidden spins on distinct endpoints commute.  Every source and transport
coefficient below is fixed by the graph coupling and the same finite
Condon--Shortley CG sums as the exact reduced-product recurrence.  The plan
contains no fusion path, hidden suffix, or trainable bond-history weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..su2.reduced_tensor_product import (
    VECTOR_PRODUCT_RANKS,
    appended_spin_product_source_coefficient,
    old_block_tensor_advance_coefficient,
    old_vector_appended_spin_product_coefficient,
    tensor_delta_q,
)
from ..su2.reduced_vector import VECTOR_DELTA_Q
from .sparse_causal_recurrence import SparseCausalRecurrencePlan


K2_DELTA_Q: tuple[int, int, int, int, int] = (-4, -2, 0, 2, 4)


def _reachable_singlet_cut(q: int, block_sites: int, hidden_sites: int) -> bool:
    return (
        0 <= q <= block_sites
        and (block_sites - q) % 2 == 0
        and q <= hidden_sites
        and (hidden_sites - q) % 2 == 0
    )


@dataclass(frozen=True)
class SecondOrderCausalPairPlan:
    """Plain-array plan for unordered stable endpoint-pair K2 channels."""

    n_sites: int
    target_q: int
    rank_q: tuple[int, int, int]
    delta_q: tuple[int, int, int, int, int]
    vector_delta_q: tuple[int, int, int]
    maximum_active_endpoint_pairs: int
    pair_slot_count: int
    event_valid: np.ndarray
    q_transition_valid: np.ndarray
    rank_delta_mask: np.ndarray
    old_pair_endpoint_sites: np.ndarray
    old_pair_endpoint_slots: np.ndarray
    old_pair_mask: np.ndarray
    new_pair_endpoint_sites: np.ndarray
    new_pair_endpoint_slots: np.ndarray
    new_pair_mask: np.ndarray
    persist_old_pair_slot_for_new: np.ndarray
    first_old_endpoint_slot_for_new: np.ndarray
    second_old_endpoint_slot_for_new: np.ndarray
    pair_birth_mask: np.ndarray
    pair_diagonal_mask: np.ndarray
    pair_multiplicity: np.ndarray
    first_source_coupling: np.ndarray
    second_source_coupling: np.ndarray
    source_coupling_product: np.ndarray
    physical_rank_mask: np.ndarray
    tensor_transport: np.ndarray
    left_vector_source: np.ndarray
    right_vector_source: np.ndarray
    appended_spin_ordered_source: np.ndarray
    offdiagonal_same_spin_factor: np.ndarray
    uses_hidden_suffix: bool = False
    enumerates_revealed_prefixes: bool = False
    trainable_pair_weights: bool = False
    unordered_endpoint_pairs: bool = True

    @property
    def q_size(self) -> int:
        return self.n_sites + 1

    @property
    def rank_count(self) -> int:
        return len(self.rank_q)

    def numeric_payload(self) -> dict[str, np.ndarray]:
        """Return only arrays suitable for immutable JAX loading."""

        names = (
            "event_valid",
            "q_transition_valid",
            "rank_delta_mask",
            "old_pair_endpoint_sites",
            "old_pair_endpoint_slots",
            "old_pair_mask",
            "new_pair_endpoint_sites",
            "new_pair_endpoint_slots",
            "new_pair_mask",
            "persist_old_pair_slot_for_new",
            "first_old_endpoint_slot_for_new",
            "second_old_endpoint_slot_for_new",
            "pair_birth_mask",
            "pair_diagonal_mask",
            "pair_multiplicity",
            "first_source_coupling",
            "second_source_coupling",
            "source_coupling_product",
            "physical_rank_mask",
            "tensor_transport",
            "left_vector_source",
            "right_vector_source",
            "appended_spin_ordered_source",
            "offdiagonal_same_spin_factor",
        )
        payload = {name: np.asarray(getattr(self, name)) for name in names}
        payload.update(
            {
                "n_sites": np.asarray(self.n_sites, dtype=np.int32),
                "target_q": np.asarray(self.target_q, dtype=np.int32),
                "rank_q": np.asarray(self.rank_q, dtype=np.int32),
                "delta_q": np.asarray(self.delta_q, dtype=np.int32),
                "vector_delta_q": np.asarray(
                    self.vector_delta_q, dtype=np.int32
                ),
                "maximum_active_endpoint_pairs": np.asarray(
                    self.maximum_active_endpoint_pairs, dtype=np.int32
                ),
                "pair_slot_count": np.asarray(
                    self.pair_slot_count, dtype=np.int32
                ),
            }
        )
        return payload

    def payload_nbytes(self) -> int:
        return int(sum(value.nbytes for value in self.numeric_payload().values()))


def _endpoint_pairs(
    sites: np.ndarray, mask: np.ndarray
) -> tuple[tuple[int, int, int, int], ...]:
    active_slots = tuple(int(slot) for slot in np.flatnonzero(mask))
    return tuple(
        (
            first_slot,
            second_slot,
            int(sites[first_slot]),
            int(sites[second_slot]),
        )
        for first_position, first_slot in enumerate(active_slots)
        for second_slot in active_slots[first_position:]
    )


def compile_second_order_causal_pair_plan(
    first_order: SparseCausalRecurrencePlan,
) -> SecondOrderCausalPairPlan:
    """Compile unordered K2 pair slots from an exact K1 endpoint plan."""

    if bool(getattr(first_order, "uses_hidden_suffix", False)):
        raise ValueError("K2 pair plans may not use a hidden suffix")
    if bool(getattr(first_order, "enumerates_revealed_prefixes", False)):
        raise ValueError("K2 pair plans may not enumerate revealed prefixes")
    if tuple(first_order.delta_q) != VECTOR_DELTA_Q:
        raise ValueError("first-order plan must carry q_op=2 endpoint vectors")

    event_shape = first_order.event_valid.shape
    maximum_pairs = first_order.maximum_active_endpoints * (
        first_order.maximum_active_endpoints + 1
    ) // 2
    pair_slots = max(1, maximum_pairs)
    pair_shape = event_shape + (pair_slots,)
    pair_site_shape = pair_shape + (2,)
    old_pair_endpoint_sites = np.full(pair_site_shape, -1, dtype=np.int32)
    old_pair_endpoint_slots = np.full(pair_site_shape, -1, dtype=np.int32)
    old_pair_mask = np.zeros(pair_shape, dtype=bool)
    new_pair_endpoint_sites = np.full(pair_site_shape, -1, dtype=np.int32)
    new_pair_endpoint_slots = np.full(pair_site_shape, -1, dtype=np.int32)
    new_pair_mask = np.zeros(pair_shape, dtype=bool)
    persist_old_pair_slot_for_new = np.full(pair_shape, -1, dtype=np.int32)
    first_old_endpoint_slot_for_new = np.full(pair_shape, -1, dtype=np.int32)
    second_old_endpoint_slot_for_new = np.full(pair_shape, -1, dtype=np.int32)
    pair_birth_mask = np.zeros(pair_shape, dtype=bool)
    pair_diagonal_mask = np.zeros(pair_shape, dtype=bool)
    pair_multiplicity = np.zeros(pair_shape, dtype=np.float64)
    first_source_coupling = np.zeros(pair_shape, dtype=np.float64)
    second_source_coupling = np.zeros(pair_shape, dtype=np.float64)
    physical_rank_mask = np.zeros(pair_shape + (3,), dtype=bool)

    for decision, microstep in np.ndindex(event_shape):
        if not first_order.event_valid[decision, microstep]:
            continue
        old_pairs = _endpoint_pairs(
            first_order.old_endpoint_sites[decision, microstep],
            first_order.old_endpoint_mask[decision, microstep],
        )
        new_pairs = _endpoint_pairs(
            first_order.new_endpoint_sites[decision, microstep],
            first_order.new_endpoint_mask[decision, microstep],
        )
        old_position = {
            (first_site, second_site): slot
            for slot, (_, _, first_site, second_site) in enumerate(old_pairs)
        }
        for slot, (first_slot, second_slot, first_site, second_site) in enumerate(
            old_pairs
        ):
            old_pair_endpoint_slots[decision, microstep, slot] = (
                first_slot,
                second_slot,
            )
            old_pair_endpoint_sites[decision, microstep, slot] = (
                first_site,
                second_site,
            )
            old_pair_mask[decision, microstep, slot] = True
        endpoint_old_slots = first_order.persist_old_slot_for_new[
            decision, microstep
        ]
        endpoint_sources = first_order.source_coupling[decision, microstep]
        for slot, (first_slot, second_slot, first_site, second_site) in enumerate(
            new_pairs
        ):
            first_old = int(endpoint_old_slots[first_slot])
            second_old = int(endpoint_old_slots[second_slot])
            new_pair_endpoint_slots[decision, microstep, slot] = (
                first_slot,
                second_slot,
            )
            new_pair_endpoint_sites[decision, microstep, slot] = (
                first_site,
                second_site,
            )
            new_pair_mask[decision, microstep, slot] = True
            first_old_endpoint_slot_for_new[
                decision, microstep, slot
            ] = first_old
            second_old_endpoint_slot_for_new[
                decision, microstep, slot
            ] = second_old
            if first_old >= 0 and second_old >= 0:
                old_first_site = int(
                    first_order.old_endpoint_sites[
                        decision, microstep, first_old
                    ]
                )
                old_second_site = int(
                    first_order.old_endpoint_sites[
                        decision, microstep, second_old
                    ]
                )
                persist_old_pair_slot_for_new[
                    decision, microstep, slot
                ] = old_position[(old_first_site, old_second_site)]
            pair_birth_mask[decision, microstep, slot] = (
                persist_old_pair_slot_for_new[decision, microstep, slot] < 0
            )
            diagonal = first_site == second_site
            pair_diagonal_mask[decision, microstep, slot] = diagonal
            pair_multiplicity[decision, microstep, slot] = (
                1.0 if diagonal else 2.0
            )
            first_source_coupling[decision, microstep, slot] = endpoint_sources[
                first_slot
            ]
            second_source_coupling[decision, microstep, slot] = endpoint_sources[
                second_slot
            ]
            physical_rank_mask[decision, microstep, slot] = (
                True,
                True,
                not diagonal,
            )

    source_coupling_product = first_source_coupling * second_source_coupling
    rank_delta_mask = np.zeros((3, len(K2_DELTA_Q)), dtype=bool)
    for rank_slot, q_op in enumerate(VECTOR_PRODUCT_RANKS):
        for delta in tensor_delta_q(q_op):
            rank_delta_mask[rank_slot, K2_DELTA_Q.index(delta)] = True

    q_size = first_order.n_sites + 1
    local_shape = event_shape + (
        q_size,
        q_size,
        len(VECTOR_PRODUCT_RANKS),
    )
    tensor_transport = np.zeros(
        local_shape + (len(K2_DELTA_Q), len(K2_DELTA_Q)),
        dtype=np.float64,
    )
    left_vector_source = np.zeros(
        local_shape + (len(K2_DELTA_Q), len(VECTOR_DELTA_Q)),
        dtype=np.float64,
    )
    right_vector_source = np.zeros_like(left_vector_source)
    appended_spin_ordered_source = np.zeros(
        local_shape + (len(K2_DELTA_Q),), dtype=np.float64
    )

    for decision, microstep in np.ndindex(event_shape):
        if not first_order.event_valid[decision, microstep]:
            continue
        old_count = int(
            first_order.revealed_count_before[decision, microstep]
        )
        new_count = int(first_order.revealed_count_after[decision, microstep])
        hidden_before = first_order.n_sites - old_count
        hidden_after = first_order.n_sites - new_count
        for q_before, q_after in np.argwhere(
            first_order.q_transition_valid[decision, microstep]
        ):
            q_before = int(q_before)
            q_after = int(q_after)
            for rank_slot, q_op in enumerate(VECTOR_PRODUCT_RANKS):
                exchange_phase = -1.0 if (q_op // 2) % 2 else 1.0
                for new_delta_slot, delta_new in enumerate(K2_DELTA_Q):
                    if not rank_delta_mask[rank_slot, new_delta_slot]:
                        continue
                    q_bra_after = q_after + delta_new
                    if not _reachable_singlet_cut(
                        q_bra_after, new_count, hidden_after
                    ):
                        continue
                    appended_spin_ordered_source[
                        decision,
                        microstep,
                        q_before,
                        q_after,
                        rank_slot,
                        new_delta_slot,
                    ] = appended_spin_product_source_coefficient(
                        q_before, q_bra_after, q_after, q_op
                    )
                    if old_count == 0:
                        continue
                    for old_delta_slot, delta_old in enumerate(K2_DELTA_Q):
                        if not rank_delta_mask[rank_slot, old_delta_slot]:
                            continue
                        q_bra_before = q_before + delta_old
                        if _reachable_singlet_cut(
                            q_bra_before, old_count, hidden_before
                        ):
                            tensor_transport[
                                decision,
                                microstep,
                                q_before,
                                q_after,
                                rank_slot,
                                new_delta_slot,
                                old_delta_slot,
                            ] = old_block_tensor_advance_coefficient(
                                q_bra_before,
                                q_before,
                                q_bra_after,
                                q_after,
                                q_op,
                            )
                    for vector_slot, vector_delta in enumerate(VECTOR_DELTA_Q):
                        q_vector_bra_before = q_before + vector_delta
                        if not _reachable_singlet_cut(
                            q_vector_bra_before, old_count, hidden_before
                        ):
                            continue
                        coefficient = old_vector_appended_spin_product_coefficient(
                            q_vector_bra_before,
                            q_before,
                            q_bra_after,
                            q_after,
                            q_op,
                        )
                        left_vector_source[
                            decision,
                            microstep,
                            q_before,
                            q_after,
                            rank_slot,
                            new_delta_slot,
                            vector_slot,
                        ] = coefficient
                        right_vector_source[
                            decision,
                            microstep,
                            q_before,
                            q_after,
                            rank_slot,
                            new_delta_slot,
                            vector_slot,
                        ] = exchange_phase * coefficient

    offdiagonal_same_spin_factor = np.asarray(
        [1.0 if (q_op // 2) % 2 == 0 else 0.0 for q_op in VECTOR_PRODUCT_RANKS],
        dtype=np.float64,
    )
    return SecondOrderCausalPairPlan(
        n_sites=first_order.n_sites,
        target_q=first_order.target_q,
        rank_q=VECTOR_PRODUCT_RANKS,
        delta_q=K2_DELTA_Q,
        vector_delta_q=VECTOR_DELTA_Q,
        maximum_active_endpoint_pairs=maximum_pairs,
        pair_slot_count=pair_slots,
        event_valid=np.asarray(first_order.event_valid),
        q_transition_valid=np.asarray(first_order.q_transition_valid),
        rank_delta_mask=rank_delta_mask,
        old_pair_endpoint_sites=old_pair_endpoint_sites,
        old_pair_endpoint_slots=old_pair_endpoint_slots,
        old_pair_mask=old_pair_mask,
        new_pair_endpoint_sites=new_pair_endpoint_sites,
        new_pair_endpoint_slots=new_pair_endpoint_slots,
        new_pair_mask=new_pair_mask,
        persist_old_pair_slot_for_new=persist_old_pair_slot_for_new,
        first_old_endpoint_slot_for_new=first_old_endpoint_slot_for_new,
        second_old_endpoint_slot_for_new=second_old_endpoint_slot_for_new,
        pair_birth_mask=pair_birth_mask,
        pair_diagonal_mask=pair_diagonal_mask,
        pair_multiplicity=pair_multiplicity,
        first_source_coupling=first_source_coupling,
        second_source_coupling=second_source_coupling,
        source_coupling_product=source_coupling_product,
        physical_rank_mask=physical_rank_mask,
        tensor_transport=tensor_transport,
        left_vector_source=left_vector_source,
        right_vector_source=right_vector_source,
        appended_spin_ordered_source=appended_spin_ordered_source,
        offdiagonal_same_spin_factor=offdiagonal_same_spin_factor,
    )


__all__ = [
    "K2_DELTA_Q",
    "SecondOrderCausalPairPlan",
    "compile_second_order_causal_pair_plan",
]
