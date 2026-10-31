"""Autoregressive TreeGRU with an immutable exact Hamiltonian carrier.

The carrier follows the sparse ``q_op=2`` birth/transport/death tables
without learned recurrent matrices or nonlinearities.  Candidate scores are
bilinear between that immutable carrier and the already revealed TreeGRU
hidden state.  This separates the local Hamiltonian action from its learned
causal environment while preserving exact autoregressive normalization.
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


EXACT_CARRIER_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)
DECODER_HEAD_NAMES: tuple[str, ...] = (
    "probability_endpoint_head",
    "phase_endpoint_head",
    "probability_closed_head",
    "phase_closed_head",
)


def _plan_value(plan: object, name: str) -> Any:
    if not hasattr(plan, name):
        raise AttributeError(f"exact carrier plan is missing {name}")
    return getattr(plan, name)


@dataclass(frozen=True)
class ExactCarrierTreeGRUConfig:
    """Configuration of the immutable-carrier causal decoder."""

    base: CompactTreeGRUConfig
    recurrence_plan: object
    coefficient_scale: int = 1

    def __post_init__(self) -> None:
        if self.base.tree != FusionTree.left_associated(self.base.n):
            raise ValueError("exact carrier decoder requires a left-associated tree")
        if self.base.target_q != 0:
            raise ValueError("exact carrier decoder currently requires target_q=0")
        if int(_plan_value(self.recurrence_plan, "n_sites")) != self.base.n:
            raise ValueError("recurrence plan and base tree sizes disagree")
        if int(getattr(self.recurrence_plan, "target_q", 0)) != self.base.target_q:
            raise ValueError("recurrence plan and base target q disagree")
        delta_q = tuple(int(q) for q in _plan_value(self.recurrence_plan, "delta_q"))
        if delta_q != EXACT_CARRIER_DELTA_Q:
            raise ValueError("exact carrier requires delta_q=(-2,0,+2)")
        if bool(getattr(self.recurrence_plan, "uses_hidden_suffix", False)):
            raise ValueError("exact carrier plan may not use a hidden suffix")
        if bool(getattr(self.recurrence_plan, "enumerates_revealed_prefixes", False)):
            raise ValueError("exact carrier plan may not enumerate prefixes")
        if isinstance(self.coefficient_scale, bool) or self.coefficient_scale not in (0, 1):
            raise ValueError("coefficient_scale must be the integer 0 or 1")

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


def init_exact_carrier_tree_gru(
    key: jax.Array, config: ExactCarrierTreeGRUConfig
) -> dict[str, object]:
    """Initialize exactly at the base TreeGRU by zeroing bilinear heads."""

    base_key, _ = jax.random.split(key)
    context_width = config.hidden_dim + 1
    decoder = {
        "probability_endpoint_head": jnp.zeros(
            (config.n, 3, context_width), dtype=jnp.float64
        ),
        "phase_endpoint_head": jnp.zeros(
            (config.n, 3, context_width), dtype=jnp.float64
        ),
        "probability_closed_head": jnp.zeros(
            (config.n - 1, context_width), dtype=jnp.float64
        ),
        "phase_closed_head": jnp.zeros(
            (config.n - 1, context_width), dtype=jnp.float64
        ),
    }
    return {"base": init_compact_tree_gru(base_key, config.base), "decoder": decoder}


def condition_existing_compact_tree_gru_exact_carrier(
    base_params: dict[str, object],
    key: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> dict[str, object]:
    """Attach zero decoder heads to an existing base checkpoint."""

    initialized = init_exact_carrier_tree_gru(key, config)
    return {"base": base_params, "decoder": initialized["decoder"]}


def _zero_carrier(config: ExactCarrierTreeGRUConfig) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.zeros((config.endpoint_slots, 3), dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
    )


def _microstep_new_q(
    decision: int,
    substep: int,
    candidate_q: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> jax.Array:
    if decision < config.n - 2:
        return candidate_q
    return jnp.asarray(1 if substep == 0 else 0, dtype=jnp.int32)


def _advance_event(
    endpoint: jax.Array,
    closed: jax.Array,
    decision: int,
    substep: int,
    q_before: jax.Array,
    q_after: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    plan = config.recurrence_plan
    q_before = jnp.clip(q_before, 0, config.n)
    q_after = jnp.clip(q_after, 0, config.n)
    scale = jnp.asarray(config.coefficient_scale, dtype=jnp.float64)
    old_slots = jnp.asarray(_plan_value(plan, "persist_old_slot_for_new"))[
        decision, substep
    ]
    safe_slots = jnp.clip(old_slots, 0, config.endpoint_slots - 1)
    persistent = old_slots >= 0
    gathered = endpoint[safe_slots] * persistent[:, None]
    transport = jnp.asarray(_plan_value(plan, "old_block_transport"))[
        decision, substep, q_before, q_after
    ]
    advanced = jnp.einsum("ab,wb->wa", transport, gathered)
    source = (
        jnp.asarray(_plan_value(plan, "source_coupling"))[decision, substep, :, None]
        * jnp.asarray(_plan_value(plan, "endpoint_source"))[
            decision, substep, q_before, q_after
        ][None, :]
    )
    new_endpoint = advanced + scale * source
    new_endpoint *= jnp.asarray(_plan_value(plan, "new_endpoint_mask"))[
        decision, substep, :, None
    ]

    dying_slot = jnp.asarray(_plan_value(plan, "dying_old_slot"))[
        decision, substep
    ]
    safe_dying = jnp.clip(dying_slot, 0, config.endpoint_slots - 1)
    death = jnp.asarray(_plan_value(plan, "death_mask"), dtype=jnp.float64)[
        decision, substep
    ] * jnp.sum(
        endpoint[safe_dying]
        * jnp.asarray(_plan_value(plan, "dying_scalar"))[
            decision, substep, q_before, q_after
        ]
    )
    return new_endpoint, closed + scale * death


def _advance_candidate(
    endpoint: jax.Array,
    closed: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    state = (endpoint, closed)
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


def _score_candidate(
    decoder: dict[str, jax.Array],
    state: tuple[jax.Array, jax.Array],
    base_hidden: jax.Array,
    decision: int,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    endpoint, closed = state
    plan = config.recurrence_plan
    valid = np.asarray(_plan_value(plan, "event_valid"), dtype=bool)
    last = int(np.flatnonzero(valid[decision])[-1])
    sites = jnp.asarray(_plan_value(plan, "new_endpoint_sites"), dtype=jnp.int32)[
        decision, last
    ]
    active = jnp.asarray(
        _plan_value(plan, "new_endpoint_mask"), dtype=jnp.float64
    )[decision, last]
    safe_sites = jnp.clip(sites, 0, config.n - 1)
    context = jnp.concatenate((base_hidden, jnp.ones((1,), dtype=jnp.float64)))

    def score(endpoint_head: jax.Array, closed_head: jax.Array) -> jax.Array:
        endpoint_weights = jnp.einsum(
            "wdh, h->wd", endpoint_head[safe_sites], context
        )
        endpoint_score = jnp.sum(endpoint * endpoint_weights * active[:, None])
        endpoint_score /= jnp.sqrt(jnp.maximum(1.0, 3.0 * jnp.sum(active)))
        return endpoint_score + closed * (closed_head[decision] @ context)

    return (
        score(
            decoder["probability_endpoint_head"],
            decoder["probability_closed_head"],
        ),
        score(
            decoder["phase_endpoint_head"],
            decoder["phase_closed_head"],
        ),
    )


def _candidate_states_and_scores(
    decoder: dict[str, jax.Array],
    endpoint: jax.Array,
    closed: jax.Array,
    base_hidden: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    candidate_mask: jax.Array,
    config: ExactCarrierTreeGRUConfig,
):
    def one(q_after, is_valid):
        def valid_branch(_):
            state = _advance_candidate(
                endpoint, closed, decision, q_before, q_after, config
            )
            probability, phase = _score_candidate(
                decoder, state, base_hidden, decision, config
            )
            return state, probability, phase

        def invalid_branch(_):
            zero = jnp.asarray(0.0, dtype=jnp.float64)
            return (endpoint, closed), zero, zero

        return jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)

    return jax.vmap(one)(candidate_q, candidate_mask)


def exact_carrier_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return conditional and cumulative logs from the carrier decoder."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},)")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    endpoint, closed = _zero_carrier(config)
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
        valid_state = valid_state & in_range & jnp.any(matches)
        previous = action
    values = jnp.stack(conditionals)
    invalid = jnp.full_like(values, -jnp.inf + 0.0j)
    values = jnp.where(valid_state, values, invalid)
    return values, jnp.cumsum(values)


def exact_carrier_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> jax.Array:
    """Exactly normalized carrier-environment autoregressive amplitude."""

    conditionals, _ = exact_carrier_tree_prefix_logs(params, labels, config)
    value = jnp.sum(conditionals)
    # Preserve the base evaluator's invalid-state semantics.
    base_value = compact_tree_log_psi(params["base"], labels, config.base)
    return jnp.where(jnp.isfinite(jnp.real(value)), value, base_value)


def exact_carrier_tree_state_trace(
    labels: jax.Array, config: ExactCarrierTreeGRUConfig
) -> tuple[jax.Array, jax.Array]:
    """Return the immutable endpoint/scalar trace for a complete valid path."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    endpoint, closed = _zero_carrier(config)
    endpoints = []
    scalars = []
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        desired_left = (
            jnp.asarray(1, dtype=jnp.int32)
            if decision.left_index is None
            else labels[decision.left_index]
        )
        endpoint, closed = _advance_candidate(
            endpoint,
            closed,
            decision.order,
            parent_q,
            desired_left,
            config,
        )
        endpoints.append(endpoint)
        scalars.append(closed)
    return jnp.stack(endpoints), jnp.stack(scalars)


def sample_exact_carrier_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: ExactCarrierTreeGRUConfig,
) -> jax.Array:
    """Draw fusion labels with the same exact candidate carrier as evaluation."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(base["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    zero_endpoint, zero_closed = _zero_carrier(config)
    endpoint = jnp.broadcast_to(
        zero_endpoint, (n_samples,) + zero_endpoint.shape
    )
    closed = jnp.broadcast_to(zero_closed, (n_samples,))
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

        def one(z, c, h, q, candidate, candidate_valid):
            return _candidate_states_and_scores(
                decoder,
                z,
                c,
                h,
                decision.order,
                q,
                candidate,
                candidate_valid,
                config,
            )

        candidate_states, local_logits, _ = jax.vmap(one)(
            endpoint, closed, hidden, parent_q, candidates, mask
        )
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        batch = jnp.arange(n_samples, dtype=jnp.int32)
        endpoint = candidate_states[0][batch, action]
        closed = candidate_states[1][batch, action]
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
    "EXACT_CARRIER_DELTA_Q",
    "ExactCarrierTreeGRUConfig",
    "condition_existing_compact_tree_gru_exact_carrier",
    "exact_carrier_tree_log_psi",
    "exact_carrier_tree_prefix_logs",
    "exact_carrier_tree_state_trace",
    "init_exact_carrier_tree_gru",
    "sample_exact_carrier_tree",
]
