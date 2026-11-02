"""Matched free-history K2 control for the exact SU(2) carrier decoder.

The control is deliberately graph blind.  It augments a frozen compact
left-path TreeGRU with one independent causal feature for every ordered pair
of autoregressive decisions.  The pair feature is the product of the two
fusion steps, which are doubled-label changes ``+/-1``.  Remaining feature
groups are generic candidate-odd K1 monomials.  No lattice edge, coupling, or
recoupling coefficient enters the forward pass.

For ``N <= 10`` the number of feature groups is fixed to ``4*N - 1``.  With a
context of width ``hidden_dim + 1``, separate probability and phase heads
therefore contain exactly the same number of real parameters as the exact
carrier decoder: ``2 * (4*N - 1) * (hidden_dim + 1)``.  At ``N=10`` all 36
causal decision pairs are independent and the remaining three groups are K1
features, giving the registered 702-parameter control at hidden dimension 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=None)
def _single_feature_exponents(count: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic ``parent^a * decision^b`` monomial powers."""

    if count < 1:
        raise ValueError("the free-K2 control needs at least one K1 feature")
    answer: list[tuple[int, int]] = []
    degree = 0
    while len(answer) < count:
        for parent_power in range(degree + 1):
            decision_power = degree - parent_power
            answer.append((parent_power, decision_power))
            if len(answer) == count:
                break
        degree += 1
    return tuple(answer)


@dataclass(frozen=True)
class FreeK2TreeGRUConfig:
    """Configuration of the graph-blind, parameter-matched K2 control."""

    base: CompactTreeGRUConfig

    def __post_init__(self) -> None:
        if self.base.tree != FusionTree.left_associated(self.base.n):
            raise ValueError("free-K2 control requires a left-associated tree")
        if self.base.target_q != 0:
            raise ValueError("free-K2 control currently requires target_q=0")
        if self.base.n > 10:
            raise ValueError("complete free decision-pair control is limited to N<=10")
        if self.single_feature_count < 1:
            raise ValueError("free-K2 feature budget cannot hold all decision pairs")

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
    def pair_feature_count(self) -> int:
        decisions = self.n - 1
        return decisions * (decisions - 1) // 2

    @property
    def feature_group_count(self) -> int:
        return 4 * self.n - 1

    @property
    def single_feature_count(self) -> int:
        return self.feature_group_count - self.pair_feature_count

    @property
    def context_width(self) -> int:
        return self.hidden_dim + 1

    @property
    def parameter_count(self) -> int:
        return 2 * self.feature_group_count * self.context_width


def init_free_k2_tree_gru(
    key: jax.Array, config: FreeK2TreeGRUConfig
) -> dict[str, object]:
    """Initialize the matched control exactly at its compact base state."""

    base_key, _ = jax.random.split(key)
    shape = (config.feature_group_count, config.context_width)
    decoder = {
        "probability_history_head": jnp.zeros(shape, dtype=jnp.float64),
        "phase_history_head": jnp.zeros(shape, dtype=jnp.float64),
    }
    return {"base": init_compact_tree_gru(base_key, config.base), "decoder": decoder}


def condition_existing_compact_tree_gru_free_k2(
    base_params: dict[str, object],
    key: jax.Array,
    config: FreeK2TreeGRUConfig,
) -> dict[str, object]:
    """Attach zero free-history heads to an existing compact TreeGRU."""

    initialized = init_free_k2_tree_gru(key, config)
    return {"base": base_params, "decoder": initialized["decoder"]}


def _candidate_history_scores(
    decoder: dict[str, jax.Array],
    history: jax.Array,
    hidden: jax.Array,
    decision: int,
    parent_q: jax.Array,
    candidate_q: jax.Array,
    config: FreeK2TreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Score one candidate using only its revealed free K1/K2 history."""

    delta = candidate_q - parent_q
    context = jnp.concatenate((hidden, jnp.ones((1,), dtype=jnp.float64)))
    pair_offset = decision * (decision - 1) // 2
    pair_slice = slice(pair_offset, pair_offset + decision)
    pair_features = history[:decision] * delta
    pair_scale = jnp.sqrt(jnp.asarray(max(decision, 1), dtype=jnp.float64))
    parent_coordinate = parent_q.astype(jnp.float64) / float(config.n)
    decision_coordinate = jnp.asarray(
        (decision + 1) / (config.n - 1), dtype=jnp.float64
    )
    single_features = jnp.asarray(
        [
            delta.astype(jnp.float64)
            * parent_coordinate**parent_power
            * decision_coordinate**decision_power
            for parent_power, decision_power in _single_feature_exponents(
                config.single_feature_count
            )
        ]
    )
    single_start = config.pair_feature_count

    def score(head: jax.Array) -> jax.Array:
        coefficients = head @ context
        pair = jnp.sum(coefficients[pair_slice] * pair_features) / pair_scale
        single = jnp.sum(coefficients[single_start:] * single_features)
        single /= jnp.sqrt(float(config.single_feature_count))
        return pair + single

    return (
        score(decoder["probability_history_head"]),
        score(decoder["phase_history_head"]),
    )


def free_k2_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: FreeK2TreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return normalized conditional logs for the graph-blind K2 control."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},)")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    history = jnp.zeros((config.n - 1,), dtype=jnp.int32)
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

        def one(candidate_q):
            return _candidate_history_scores(
                decoder,
                history,
                hidden,
                decision.order,
                safe_parent,
                candidate_q,
                config,
            )

        local_logits, local_phases = jax.vmap(one)(candidates)
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        conditionals.append(
            0.5 * jax.nn.log_softmax(logits)[action]
            + 1j * (base_phases[action] + local_phases[action])
        )
        chosen_delta = desired_left - safe_parent
        history = history.at[decision.order].set(chosen_delta)
        valid_state = (
            valid_state
            & in_range
            & jnp.any(matches)
            & ((chosen_delta == -1) | (chosen_delta == 1))
        )
        previous = action
    values = jnp.stack(conditionals)
    invalid = jnp.full_like(values, -jnp.inf + 0.0j)
    values = jnp.where(valid_state, values, invalid)
    return values, jnp.cumsum(values)


def free_k2_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: FreeK2TreeGRUConfig,
) -> jax.Array:
    """Return an exactly normalized graph-blind free-K2 amplitude."""

    conditionals, _ = free_k2_tree_prefix_logs(params, labels, config)
    value = jnp.sum(conditionals)
    base_value = compact_tree_log_psi(params["base"], labels, config.base)
    return jnp.where(jnp.isfinite(jnp.real(value)), value, base_value)


def sample_free_k2_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: FreeK2TreeGRUConfig,
) -> jax.Array:
    """Sample valid paths from the matched free-history control."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    base = params["base"]
    decoder = params["decoder"]
    left, right, table_valid = (jnp.asarray(value) for value in split_tables(config.tree))
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(base["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    history = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
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

        def one(sample_history, sample_hidden, sample_parent, sample_candidates):
            return jax.vmap(
                lambda candidate: _candidate_history_scores(
                    decoder,
                    sample_history,
                    sample_hidden,
                    decision.order,
                    sample_parent,
                    candidate,
                    config,
                )
            )(sample_candidates)

        local_logits, _ = jax.vmap(one)(history, hidden, parent_q, candidates)
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        chosen_left = candidates[batch, action]
        chosen_right = candidate_right[batch, action]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        history = history.at[:, decision.order].set(chosen_left - parent_q)
        previous = action
    return labels


__all__ = [
    "FreeK2TreeGRUConfig",
    "condition_existing_compact_tree_gru_free_k2",
    "free_k2_tree_log_psi",
    "free_k2_tree_prefix_logs",
    "init_free_k2_tree_gru",
    "sample_free_k2_tree",
]
