"""Normalized TreeGRU decoder with an immutable SU(2) K1/K2 pair carrier.

The carrier propagates first-order endpoint vectors and unordered second-order
``q_op=0,2,4`` endpoint pairs.  The learned decoder sees only 39 fixed
physical feature groups at ``N=10``: three aggregated K1 channels, one closed
scalar per decision, and the nine allowed K2 rank/delta channels resolved by
three predetermined spatial moments.  No endpoint pair or bond history gets
an independent trainable weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import FusionTree
from ._core import embed, gru_step, heads
from .compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    init_compact_tree_gru,
    split_tables,
)
from .tree_gru import tree_decisions


K1_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)
K2_RANK_Q: tuple[int, int, int] = (0, 2, 4)
K2_DELTA_Q: tuple[int, int, int, int, int] = (-4, -2, 0, 2, 4)
K2_CHANNELS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
)
SPATIAL_MOMENTS: tuple[str, str, str] = (
    "uniform",
    "centered_pair_midpoint",
    "pair_separation",
)
DECODER_HEAD_NAMES: tuple[str, str] = (
    "probability_physical_head",
    "phase_physical_head",
)


def _plan_value(plan: object, name: str) -> Any:
    if not hasattr(plan, name):
        raise AttributeError(f"K2 carrier plan is missing {name}")
    return getattr(plan, name)


@dataclass(frozen=True)
class K2PairCarrierTreeGRUConfig:
    """Configuration for the fixed-rank, fixed-moment K2 decoder."""

    base: CompactTreeGRUConfig
    recurrence_plan: object
    pair_plan: object
    coefficient_scale: int = 1
    pair_form_factors: object | None = None
    spatial_moment_names: tuple[str, str, str] = SPATIAL_MOMENTS
    feature_layout: str = "balanced_moments"

    def __post_init__(self) -> None:
        if self.base.tree != FusionTree.left_associated(self.base.n):
            raise ValueError("K2 pair carrier requires a left-associated tree")
        if self.base.target_q != 0:
            raise ValueError("K2 pair carrier currently requires target_q=0")
        if int(_plan_value(self.recurrence_plan, "n_sites")) != self.base.n:
            raise ValueError("recurrence plan and base tree sizes disagree")
        if int(_plan_value(self.pair_plan, "n_sites")) != self.base.n:
            raise ValueError("pair plan and base tree sizes disagree")
        if int(getattr(self.recurrence_plan, "target_q", 0)) != self.target_q:
            raise ValueError("recurrence plan and base target q disagree")
        if int(getattr(self.pair_plan, "target_q", 0)) != self.target_q:
            raise ValueError("pair plan and base target q disagree")
        if tuple(int(q) for q in _plan_value(self.recurrence_plan, "delta_q")) != K1_DELTA_Q:
            raise ValueError("K2 pair carrier requires K1 delta_q=(-2,0,+2)")
        if tuple(int(q) for q in _plan_value(self.pair_plan, "rank_q")) != K2_RANK_Q:
            raise ValueError("K2 pair carrier requires q_op=(0,2,4)")
        if tuple(int(q) for q in _plan_value(self.pair_plan, "delta_q")) != K2_DELTA_Q:
            raise ValueError("K2 pair carrier requires delta_q=(-4,-2,0,+2,+4)")
        first_events = np.asarray(_plan_value(self.recurrence_plan, "event_valid"))
        pair_events = np.asarray(_plan_value(self.pair_plan, "event_valid"))
        if not np.array_equal(first_events, pair_events):
            raise ValueError("first- and second-order recurrence events disagree")
        for plan in (self.recurrence_plan, self.pair_plan):
            if bool(getattr(plan, "uses_hidden_suffix", False)):
                raise ValueError("K2 pair carrier may not use a hidden suffix")
            if bool(getattr(plan, "enumerates_revealed_prefixes", False)):
                raise ValueError("K2 pair carrier may not enumerate prefixes")
        if bool(getattr(self.pair_plan, "trainable_pair_weights", False)):
            raise ValueError("K2 pair plan may not contain trainable pair weights")
        if not bool(getattr(self.pair_plan, "unordered_endpoint_pairs", False)):
            raise ValueError("K2 pair plan must use unordered endpoint pairs")
        rank_delta = np.asarray(_plan_value(self.pair_plan, "rank_delta_mask"))
        expected = np.zeros((3, 5), dtype=bool)
        for rank, delta in K2_CHANNELS:
            expected[rank, delta] = True
        if not np.array_equal(rank_delta, expected):
            raise ValueError("pair plan has an incompatible rank/delta mask")
        if isinstance(self.coefficient_scale, bool) or self.coefficient_scale not in (0, 1):
            raise ValueError("coefficient_scale must be the integer 0 or 1")
        if self.feature_layout not in (
            "balanced_moments",
            "endpoint_k1_uniform_k2",
        ):
            raise ValueError(f"unknown K2 feature layout: {self.feature_layout}")
        if len(self.spatial_moment_names) != len(SPATIAL_MOMENTS):
            raise ValueError("exactly three spatial moment names are required")
        if self.pair_form_factors is not None:
            form_factors = np.asarray(self.pair_form_factors, dtype=np.float64)
            if form_factors.shape != (len(SPATIAL_MOMENTS), self.n, self.n):
                raise ValueError(
                    "pair_form_factors must have shape (3, n_sites, n_sites)"
                )
            if not np.all(np.isfinite(form_factors)):
                raise ValueError("pair_form_factors must be finite")
            if not np.allclose(
                form_factors, np.swapaxes(form_factors, 1, 2), atol=1.0e-14
            ):
                raise ValueError("pair_form_factors must be symmetric")

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def tree(self) -> FusionTree:
        return self.base.tree

    @property
    def target_q(self) -> int:
        return self.base.target_q

    @property
    def hidden_dim(self) -> int:
        return self.base.hidden_dim

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def endpoint_slots(self) -> int:
        return int(_plan_value(self.recurrence_plan, "endpoint_slot_count"))

    @property
    def pair_slots(self) -> int:
        return int(_plan_value(self.pair_plan, "pair_slot_count"))

    @property
    def feature_group_count(self) -> int:
        if self.feature_layout == "endpoint_k1_uniform_k2":
            return 3 * self.n + len(K2_CHANNELS)
        return 3 + (self.n - 1) + len(SPATIAL_MOMENTS) * len(K2_CHANNELS)

    @property
    def context_width(self) -> int:
        return self.hidden_dim + 1

    @property
    def parameter_count(self) -> int:
        return 2 * self.feature_group_count * self.context_width


def init_k2_pair_carrier_tree_gru(
    key: jax.Array, config: K2PairCarrierTreeGRUConfig
) -> dict[str, object]:
    """Initialize the decoder exactly at its compact TreeGRU base."""

    base_key, _ = jax.random.split(key)
    shape = (config.feature_group_count, config.context_width)
    decoder = {
        "probability_physical_head": jnp.zeros(shape, dtype=jnp.float64),
        "phase_physical_head": jnp.zeros(shape, dtype=jnp.float64),
    }
    return {"base": init_compact_tree_gru(base_key, config.base), "decoder": decoder}


def condition_existing_compact_tree_gru_k2_pair_carrier(
    base_params: dict[str, object],
    key: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> dict[str, object]:
    """Attach zero K2 physical heads to an existing compact TreeGRU."""

    initialized = init_k2_pair_carrier_tree_gru(key, config)
    return {"base": base_params, "decoder": initialized["decoder"]}


def _zero_carrier(
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((config.endpoint_slots, 3), dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.zeros((config.pair_slots, 3, 5), dtype=jnp.float64),
    )


def _microstep_new_q(
    decision: int,
    substep: int,
    candidate_q: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> jax.Array:
    if decision < config.n - 2:
        return candidate_q
    return jnp.asarray(1 if substep == 0 else 0, dtype=jnp.int32)


def _advance_event(
    endpoint: jax.Array,
    closed: jax.Array,
    pair: jax.Array,
    decision: int,
    substep: int,
    q_before: jax.Array,
    q_after: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    first = config.recurrence_plan
    second = config.pair_plan
    q_before = jnp.clip(q_before, 0, config.n)
    q_after = jnp.clip(q_after, 0, config.n)
    scale = jnp.asarray(config.coefficient_scale, dtype=jnp.float64)

    old_pair_slots = jnp.asarray(_plan_value(second, "persist_old_pair_slot_for_new"))[
        decision, substep
    ]
    safe_pair_slots = jnp.clip(old_pair_slots, 0, config.pair_slots - 1)
    gathered_pair = pair[safe_pair_slots] * (old_pair_slots >= 0)[:, None, None]
    tensor_transport = jnp.asarray(_plan_value(second, "tensor_transport"))[
        decision, substep, q_before, q_after
    ]
    new_pair = jnp.einsum("rab,wrb->wra", tensor_transport, gathered_pair)

    first_slots = jnp.asarray(_plan_value(second, "first_old_endpoint_slot_for_new"))[
        decision, substep
    ]
    second_slots = jnp.asarray(_plan_value(second, "second_old_endpoint_slot_for_new"))[
        decision, substep
    ]
    safe_first = jnp.clip(first_slots, 0, config.endpoint_slots - 1)
    safe_second = jnp.clip(second_slots, 0, config.endpoint_slots - 1)
    first_vector = endpoint[safe_first] * (first_slots >= 0)[:, None]
    second_vector = endpoint[safe_second] * (second_slots >= 0)[:, None]
    left_source = jnp.asarray(_plan_value(second, "left_vector_source"))[
        decision, substep, q_before, q_after
    ]
    right_source = jnp.asarray(_plan_value(second, "right_vector_source"))[
        decision, substep, q_before, q_after
    ]
    new_pair += scale * jnp.asarray(_plan_value(second, "second_source_coupling"))[
        decision, substep, :, None, None
    ] * jnp.einsum("rav,wv->wra", left_source, first_vector)
    new_pair += scale * jnp.asarray(_plan_value(second, "first_source_coupling"))[
        decision, substep, :, None, None
    ] * jnp.einsum("rav,wv->wra", right_source, second_vector)

    diagonal = jnp.asarray(_plan_value(second, "pair_diagonal_mask"))[
        decision, substep
    ]
    offdiagonal_factor = jnp.asarray(
        _plan_value(second, "offdiagonal_same_spin_factor"), dtype=jnp.float64
    )
    local_factor = jnp.where(diagonal[:, None], 1.0, offdiagonal_factor[None, :])
    local_source = jnp.asarray(_plan_value(second, "appended_spin_ordered_source"))[
        decision, substep, q_before, q_after
    ]
    new_pair += scale**2 * jnp.asarray(_plan_value(second, "source_coupling_product"))[
        decision, substep, :, None, None
    ] * local_factor[:, :, None] * local_source[None, :, :]
    new_pair *= jnp.asarray(_plan_value(second, "new_pair_mask"))[
        decision, substep, :, None, None
    ]
    new_pair *= jnp.asarray(_plan_value(second, "physical_rank_mask"))[
        decision, substep, :, :, None
    ]
    new_pair *= jnp.asarray(_plan_value(second, "rank_delta_mask"))[None, :, :]

    old_slots = jnp.asarray(_plan_value(first, "persist_old_slot_for_new"))[
        decision, substep
    ]
    safe_slots = jnp.clip(old_slots, 0, config.endpoint_slots - 1)
    gathered = endpoint[safe_slots] * (old_slots >= 0)[:, None]
    transport = jnp.asarray(_plan_value(first, "old_block_transport"))[
        decision, substep, q_before, q_after
    ]
    new_endpoint = jnp.einsum("ab,wb->wa", transport, gathered)
    source = jnp.asarray(_plan_value(first, "source_coupling"))[
        decision, substep, :, None
    ] * jnp.asarray(_plan_value(first, "endpoint_source"))[
        decision, substep, q_before, q_after
    ][None, :]
    new_endpoint += scale * source
    new_endpoint *= jnp.asarray(_plan_value(first, "new_endpoint_mask"))[
        decision, substep, :, None
    ]

    dying_slot = jnp.asarray(_plan_value(first, "dying_old_slot"))[
        decision, substep
    ]
    safe_dying = jnp.clip(dying_slot, 0, config.endpoint_slots - 1)
    death = jnp.asarray(_plan_value(first, "death_mask"), dtype=jnp.float64)[
        decision, substep
    ] * jnp.sum(
        endpoint[safe_dying]
        * jnp.asarray(_plan_value(first, "dying_scalar"))[
            decision, substep, q_before, q_after
        ]
    )
    return new_endpoint, closed + scale * death, new_pair


def _advance_candidate(
    endpoint: jax.Array,
    closed: jax.Array,
    pair: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    state = (endpoint, closed, pair)
    current_q = q_before
    valid = np.asarray(_plan_value(config.recurrence_plan, "event_valid"), dtype=bool)
    for substep in range(valid.shape[1]):
        if not valid[decision, substep]:
            continue
        q_after = _microstep_new_q(decision, substep, candidate_q, config)
        state = _advance_event(
            *state, decision, substep, current_q, q_after, config
        )
        current_q = q_after
    return state


def _physical_features(
    state: tuple[jax.Array, jax.Array, jax.Array],
    decision: int,
    config: K2PairCarrierTreeGRUConfig,
) -> jax.Array:
    endpoint, closed, pair = state
    first = config.recurrence_plan
    second = config.pair_plan
    valid = np.asarray(_plan_value(first, "event_valid"), dtype=bool)
    last = int(np.flatnonzero(valid[decision])[-1])

    endpoint_active = jnp.asarray(
        _plan_value(first, "new_endpoint_mask"), dtype=jnp.float64
    )[decision, last]
    endpoint_sites = jnp.asarray(_plan_value(first, "new_endpoint_sites"))[
        decision, last
    ]

    pair_sites = jnp.asarray(_plan_value(second, "new_pair_endpoint_sites"))[
        decision, last
    ]
    pair_active = jnp.asarray(
        _plan_value(second, "new_pair_mask"), dtype=jnp.float64
    )[decision, last]
    multiplicity = jnp.asarray(_plan_value(second, "pair_multiplicity"))[
        decision, last
    ]
    safe_sites = jnp.clip(pair_sites, 0, config.n - 1).astype(jnp.float64)
    denominator = float(max(config.n - 1, 1))
    if config.pair_form_factors is None:
        moments = jnp.stack(
            (
                jnp.ones((config.pair_slots,), dtype=jnp.float64),
                (safe_sites[:, 0] + safe_sites[:, 1]) / denominator - 1.0,
                jnp.abs(safe_sites[:, 1] - safe_sites[:, 0]) / denominator,
            ),
            axis=1,
        )
    else:
        form_factors = jnp.asarray(config.pair_form_factors, dtype=jnp.float64)
        moments = jnp.transpose(
            form_factors[
                :, safe_sites[:, 0].astype(jnp.int32), safe_sites[:, 1].astype(jnp.int32)
            ]
        )
    weights = pair_active[:, None] * multiplicity[:, None] * moments
    weights /= jnp.sqrt(
        jnp.maximum(1.0, jnp.sum(jnp.square(weights), axis=0, keepdims=True))
    )
    resolved = jnp.einsum("wrd,wm->mrd", pair, weights)
    if config.feature_layout == "endpoint_k1_uniform_k2":
        safe_endpoint_sites = jnp.clip(endpoint_sites, 0, config.n - 1)
        site_resolved = jnp.zeros((config.n, 3), dtype=jnp.float64)
        site_resolved = site_resolved.at[safe_endpoint_sites].add(
            endpoint * endpoint_active[:, None]
        )
        k2 = jnp.stack(
            tuple(resolved[0, rank, delta] for rank, delta in K2_CHANNELS)
        )
        return jnp.concatenate((jnp.ravel(site_resolved), k2))

    endpoint_norm = jnp.sqrt(jnp.maximum(1.0, jnp.sum(endpoint_active)))
    k1 = jnp.sum(endpoint * endpoint_active[:, None], axis=0) / endpoint_norm
    closed_by_decision = jnp.zeros((config.n - 1,), dtype=jnp.float64)
    closed_by_decision = closed_by_decision.at[decision].set(closed)
    k2 = jnp.stack(
        tuple(
            resolved[moment, rank, delta]
            for moment in range(len(SPATIAL_MOMENTS))
            for rank, delta in K2_CHANNELS
        )
    )
    return jnp.concatenate((k1, closed_by_decision, k2))


def _score_candidate(
    decoder: dict[str, jax.Array],
    state: tuple[jax.Array, jax.Array, jax.Array],
    hidden: jax.Array,
    decision: int,
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    features = _physical_features(state, decision, config)
    context = jnp.concatenate((hidden, jnp.ones((1,), dtype=jnp.float64)))

    def score(head: jax.Array) -> jax.Array:
        return features @ (head @ context)

    return (
        score(decoder["probability_physical_head"]),
        score(decoder["phase_physical_head"]),
    )


def _candidate_states_and_scores(
    decoder: dict[str, jax.Array],
    endpoint: jax.Array,
    closed: jax.Array,
    pair: jax.Array,
    hidden: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    candidate_mask: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
):
    def one(q_after, is_valid):
        def valid_branch(_):
            state = _advance_candidate(
                endpoint, closed, pair, decision, q_before, q_after, config
            )
            probability, phase = _score_candidate(
                decoder, state, hidden, decision, config
            )
            return state, probability, phase

        def invalid_branch(_):
            zero = jnp.asarray(0.0, dtype=jnp.float64)
            return (endpoint, closed, pair), zero, zero

        return jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)

    return jax.vmap(one)(candidate_q, candidate_mask)


def k2_pair_carrier_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return exactly normalized conditional and cumulative path logs."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},)")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    endpoint, closed, pair = _zero_carrier(config)
    conditionals = []
    valid_state = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        in_range = (parent_q >= 0) & (parent_q <= config.n)
        safe_parent = jnp.clip(parent_q, 0, config.n)
        desired_left = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.left_index is None
            else labels[decision.left_index]
        )
        desired_right = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.right_index is None
            else labels[decision.right_index]
        )
        candidates = left[decision.order, safe_parent]
        candidate_right = right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = mask & (candidates == desired_left) & (candidate_right == desired_right)
        action = jnp.argmax(matches).astype(jnp.int32)
        inputs = embed(
            base,
            (
                previous,
                safe_parent,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(base["gru"], hidden, inputs)
        base_logits, base_phases = heads(base, hidden)
        candidate_states, local_logits, local_phases = _candidate_states_and_scores(
            decoder,
            endpoint,
            closed,
            pair,
            hidden,
            decision.order,
            safe_parent,
            candidates,
            mask,
            config,
        )
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        conditionals.append(
            0.5 * jax.nn.log_softmax(logits)[action]
            + 1j * (base_phases[action] + local_phases[action])
        )
        endpoint = candidate_states[0][action]
        closed = candidate_states[1][action]
        pair = candidate_states[2][action]
        valid_state = valid_state & in_range & jnp.any(matches)
        previous = action
    values = jnp.stack(conditionals)
    invalid = jnp.full_like(values, -jnp.inf + 0.0j)
    values = jnp.where(valid_state, values, invalid)
    return values, jnp.cumsum(values)


def k2_pair_carrier_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> jax.Array:
    """Return the normalized K2 physical-carrier amplitude."""

    conditionals, _ = k2_pair_carrier_tree_prefix_logs(params, labels, config)
    value = jnp.sum(conditionals)
    base_value = compact_tree_log_psi(params["base"], labels, config.base)
    return jnp.where(jnp.isfinite(jnp.real(value)), value, base_value)


def k2_pair_carrier_tree_state_trace(
    labels: jax.Array,
    config: K2PairCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return endpoint, closed-scalar, and pair traces for a valid path."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    endpoint, closed, pair = _zero_carrier(config)
    endpoints = []
    scalars = []
    pairs = []
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        desired_left = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.left_index is None
            else labels[decision.left_index]
        )
        endpoint, closed, pair = _advance_candidate(
            endpoint,
            closed,
            pair,
            decision.order,
            parent_q,
            desired_left,
            config,
        )
        endpoints.append(endpoint)
        scalars.append(closed)
        pairs.append(pair)
    return jnp.stack(endpoints), jnp.stack(scalars), jnp.stack(pairs)


def sample_k2_pair_carrier_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: K2PairCarrierTreeGRUConfig,
) -> jax.Array:
    """Draw valid paths from the same K2 candidate recurrence."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(base["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    zero_endpoint, zero_closed, zero_pair = _zero_carrier(config)
    endpoint = jnp.broadcast_to(zero_endpoint, (n_samples,) + zero_endpoint.shape)
    closed = jnp.broadcast_to(zero_closed, (n_samples,))
    pair = jnp.broadcast_to(zero_pair, (n_samples,) + zero_pair.shape)
    batch = jnp.arange(n_samples, dtype=jnp.int32)
    for decision in tree_decisions(config.tree):
        key, subkey = jax.random.split(key)
        parent_q = labels[:, decision.node_index]
        candidates = left[decision.order, parent_q]
        candidate_right = right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
        inputs = embed(
            base,
            (
                previous,
                parent_q,
                jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                jnp.full((n_samples,), decision.order, dtype=jnp.int32),
            ),
        )
        hidden = gru_step(base["gru"], hidden, inputs)
        base_logits, _ = heads(base, hidden)

        def one(z, c, p, h, q, candidate, candidate_valid):
            return _candidate_states_and_scores(
                decoder,
                z,
                c,
                p,
                h,
                decision.order,
                q,
                candidate,
                candidate_valid,
                config,
            )

        candidate_states, local_logits, _ = jax.vmap(one)(
            endpoint, closed, pair, hidden, parent_q, candidates, mask
        )
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        endpoint = candidate_states[0][batch, action]
        closed = candidate_states[1][batch, action]
        pair = candidate_states[2][batch, action]
        chosen_left = jnp.take_along_axis(candidates, action[:, None], axis=1)[:, 0]
        chosen_right = jnp.take_along_axis(
            candidate_right, action[:, None], axis=1
        )[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        previous = action
    return labels


__all__ = [
    "DECODER_HEAD_NAMES",
    "K1_DELTA_Q",
    "K2_CHANNELS",
    "K2_DELTA_Q",
    "K2_RANK_Q",
    "SPATIAL_MOMENTS",
    "K2PairCarrierTreeGRUConfig",
    "condition_existing_compact_tree_gru_k2_pair_carrier",
    "init_k2_pair_carrier_tree_gru",
    "k2_pair_carrier_tree_log_psi",
    "k2_pair_carrier_tree_prefix_logs",
    "k2_pair_carrier_tree_state_trace",
    "sample_k2_pair_carrier_tree",
]
