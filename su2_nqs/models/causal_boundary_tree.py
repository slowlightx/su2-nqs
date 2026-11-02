"""Causal SU(2) boundary-message residual for a left TreeGRU.

The model consumes a pure-numeric :class:`CausalBoundaryMessagePlan`.  At an
autoregressive cut, every alternative appearing in a vector or scalar message
is another *revealed prefix*.  No unrevealed fusion label, full wavefunction,
or conditional-suffix oracle enters the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

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


BOUNDARY_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)


@dataclass(frozen=True)
class CausalBoundaryMessagePlan:
    """Ragged-by-decision numeric tables for causal boundary messages.

    ``vector_operators[d][k,s,new,old]`` is a revealed-side reduced vector
    matrix element for physical channel ``k`` and doubled-spin change slot
    ``s``.  ``closed_scalar_operators[d][new,old]`` contains bonds whose two
    endpoints have both been revealed.  Matrices use ``operator[new,old]``.
    ``action_sequences[d]`` maps each revealed path to its root-first compact
    TreeGRU action prefix.
    """

    n_sites: int
    target_q: int
    maximum_channels: int
    revealed_paths: tuple[np.ndarray, ...]
    action_sequences: tuple[np.ndarray, ...]
    parent_q_sequences: tuple[np.ndarray, ...]
    vector_operators: tuple[np.ndarray, ...]
    vector_transition_valid: tuple[np.ndarray, ...]
    channel_active: tuple[np.ndarray, ...]
    closed_scalar_operators: tuple[np.ndarray, ...]
    revealed_site_order: tuple[np.ndarray, ...]
    vector_endpoint_weights: tuple[np.ndarray, ...]
    delta_q: tuple[int, int, int] = BOUNDARY_DELTA_Q
    uses_hidden_suffix: bool = False

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("a boundary-message plan requires at least two sites")
        if self.target_q != 0:
            raise ValueError("the current reversed-prefix plan requires target_q=0")
        if self.maximum_channels < 0:
            raise ValueError("maximum_channels must be non-negative")
        if self.delta_q != BOUNDARY_DELTA_Q:
            raise ValueError("vector transitions must use doubled delta q=-2,0,+2")
        if self.uses_hidden_suffix:
            raise ValueError("a production boundary plan may not use a hidden suffix")
        collections = (
            self.revealed_paths,
            self.action_sequences,
            self.parent_q_sequences,
            self.vector_operators,
            self.vector_transition_valid,
            self.channel_active,
            self.closed_scalar_operators,
            self.revealed_site_order,
            self.vector_endpoint_weights,
        )
        if any(len(values) != self.n_sites - 1 for values in collections):
            raise ValueError("every plan table must have one entry per AR decision")
        for decision in range(self.n_sites - 1):
            paths = np.asarray(self.revealed_paths[decision])
            actions = np.asarray(self.action_sequences[decision])
            parents = np.asarray(self.parent_q_sequences[decision])
            operators = np.asarray(self.vector_operators[decision])
            transition_valid = np.asarray(
                self.vector_transition_valid[decision]
            )
            channel_active = np.asarray(self.channel_active[decision])
            scalar = np.asarray(self.closed_scalar_operators[decision])
            sites = np.asarray(self.revealed_site_order[decision])
            weights = np.asarray(self.vector_endpoint_weights[decision])
            path_count = len(paths)
            expected_steps = decision + 1
            if paths.ndim != 2 or len(sites) != paths.shape[1]:
                raise ValueError("revealed paths and site order disagree")
            if actions.shape != (path_count, expected_steps):
                raise ValueError("action prefix table has the wrong shape")
            if parents.shape != actions.shape:
                raise ValueError("parent-q and action prefix tables disagree")
            expected_operator = (
                self.maximum_channels,
                len(self.delta_q),
                path_count,
                path_count,
            )
            if operators.shape != expected_operator:
                raise ValueError("vector operator table has the wrong shape")
            if transition_valid.shape != (path_count, len(self.delta_q)):
                raise ValueError("vector transition mask has the wrong shape")
            if channel_active.shape != (self.maximum_channels,):
                raise ValueError("channel mask has the wrong shape")
            if scalar.shape != (path_count, path_count):
                raise ValueError("closed scalar table has the wrong shape")
            if weights.shape != (self.maximum_channels, len(sites)):
                raise ValueError("endpoint-weight table has the wrong shape")
            if np.any(~np.isfinite(operators)) or np.any(~np.isfinite(scalar)):
                raise ValueError("operator messages must be finite")
            if np.any(~np.isfinite(weights)):
                raise ValueError("endpoint weights must be finite")
            path_q = paths[:, -1]
            for slot, delta in enumerate(self.delta_q):
                forbidden = path_q[:, None] - path_q[None, :] != delta
                if np.max(
                    np.abs(operators[:, slot] * forbidden[None, :, :]),
                    initial=0.0,
                ) > 2.0e-12:
                    raise ValueError("a vector table contains a forbidden q transition")

    def numeric_payload(self) -> dict[str, np.ndarray]:
        """Return a flat numeric payload suitable for an NPZ checkpoint."""

        payload: dict[str, np.ndarray] = {
            "n_sites": np.asarray(self.n_sites, dtype=np.int32),
            "target_q": np.asarray(self.target_q, dtype=np.int32),
            "maximum_channels": np.asarray(
                self.maximum_channels, dtype=np.int32
            ),
            "delta_q": np.asarray(self.delta_q, dtype=np.int32),
            "uses_hidden_suffix": np.asarray(False),
        }
        for decision in range(self.n_sites - 1):
            payload[f"revealed_paths_{decision}"] = np.asarray(
                self.revealed_paths[decision], dtype=np.int32
            )
            payload[f"action_sequences_{decision}"] = np.asarray(
                self.action_sequences[decision], dtype=np.int32
            )
            payload[f"parent_q_sequences_{decision}"] = np.asarray(
                self.parent_q_sequences[decision], dtype=np.int32
            )
            payload[f"vector_operators_{decision}"] = np.asarray(
                self.vector_operators[decision], dtype=np.complex128
            )
            payload[f"vector_transition_valid_{decision}"] = np.asarray(
                self.vector_transition_valid[decision], dtype=bool
            )
            payload[f"channel_active_{decision}"] = np.asarray(
                self.channel_active[decision], dtype=bool
            )
            payload[f"closed_scalar_operators_{decision}"] = np.asarray(
                self.closed_scalar_operators[decision], dtype=np.complex128
            )
            payload[f"revealed_site_order_{decision}"] = np.asarray(
                self.revealed_site_order[decision], dtype=np.int32
            )
            payload[f"vector_endpoint_weights_{decision}"] = np.asarray(
                self.vector_endpoint_weights[decision], dtype=np.float64
            )
        return payload


@dataclass(frozen=True)
class CausalBoundaryTreeGRUConfig:
    """Configuration for a zero-initialized boundary-message correction."""

    base: CompactTreeGRUConfig
    message_plan: CausalBoundaryMessagePlan
    message_hidden_dim: int = 32
    train_message_encoder: bool = False

    def __post_init__(self) -> None:
        if self.base.tree != FusionTree.left_associated(self.base.n):
            raise ValueError("causal boundary messages require a left-associated tree")
        if self.base.target_q != 0:
            raise ValueError("the current boundary-message model requires a singlet")
        if self.message_plan.n_sites != self.base.n:
            raise ValueError("message plan and base tree sizes disagree")
        if self.message_plan.target_q != self.base.target_q:
            raise ValueError("message plan and base target q disagree")
        if self.message_plan.uses_hidden_suffix:
            raise ValueError("suffix-oracle features are forbidden in this model")
        if self.message_hidden_dim < 1:
            raise ValueError("message_hidden_dim must be positive")

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
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def hidden_dim(self) -> int:
        return self.base.hidden_dim

    @property
    def message_feature_dim(self) -> int:
        channels = self.message_plan.maximum_channels
        return 2 * channels * len(BOUNDARY_DELTA_Q) + 2 + len(
            BOUNDARY_DELTA_Q
        ) + channels


def _matrix(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    scale = (2.0 / (shape[0] + shape[1])) ** 0.5
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_causal_boundary_tree_gru(
    key: jax.Array, config: CausalBoundaryTreeGRUConfig
) -> dict[str, object]:
    """Initialize with exactly zero boundary correction at every conditional."""

    base_key, projection_key, context_key = jax.random.split(key, 3)
    local = {
        "message_projection": _matrix(
            projection_key,
            (config.message_feature_dim, config.message_hidden_dim),
        ),
        "context_projection": _matrix(
            context_key, (config.hidden_dim, config.message_hidden_dim)
        ),
        "bias": jnp.zeros((config.message_hidden_dim,), dtype=jnp.float64),
        "probability_head": jnp.zeros(
            (config.message_hidden_dim,), dtype=jnp.float64
        ),
        "phase_head": jnp.zeros(
            (config.message_hidden_dim,), dtype=jnp.float64
        ),
    }
    return {"base": init_compact_tree_gru(base_key, config.base), "message": local}


def condition_existing_compact_tree_gru(
    base_params: dict[str, object],
    key: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> dict[str, object]:
    """Attach the zero-output causal layer to an existing TreeGRU checkpoint."""

    initialized = init_causal_boundary_tree_gru(key, config)
    return {"base": base_params, "message": initialized["message"]}


def _base_prefix_logs(
    base: dict[str, object],
    decision_index: int,
    config: CausalBoundaryTreeGRUConfig,
) -> jax.Array:
    """Evaluate base amplitudes for every revealed prefix at one cut."""

    action_sequences = jnp.asarray(
        config.message_plan.action_sequences[decision_index]
    )
    parent_sequences = jnp.asarray(
        config.message_plan.parent_q_sequences[decision_index]
    )
    table_valid = jnp.asarray(split_tables(config.tree)[2])
    decisions = tree_decisions(config.tree)

    def one_prefix(actions: jax.Array, parents: jax.Array) -> jax.Array:
        hidden = base["h0"]
        previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
        total = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
        for step in range(decision_index + 1):
            decision = decisions[step]
            parent_q = parents[step]
            inputs = embed(
                base,
                (
                    previous,
                    parent_q,
                    decision.left_leaves,
                    decision.right_leaves,
                    decision.depth,
                    decision.order,
                ),
            )
            hidden = gru_step(base["gru"], hidden, inputs)
            logits, phase_values = heads(base, hidden)
            logits = jnp.where(
                table_valid[step, parent_q], logits, -1.0e30
            )
            action = actions[step]
            total += 0.5 * jax.nn.log_softmax(logits)[action]
            total += 1j * phase_values[action]
            previous = action
        return total

    values = jax.vmap(one_prefix)(action_sequences, parent_sequences)
    if not config.train_message_encoder:
        values = jax.lax.stop_gradient(values)
    return values


def causal_boundary_message_table(
    params: dict[str, object],
    decision_index: int,
    config: CausalBoundaryTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return vector and closed-scalar messages for every revealed prefix.

    The vector result has shape ``(prefix, channel, delta_q)``.  It contracts
    only alternative revealed prefixes using the base TreeGRU prefix
    amplitudes.  The scalar result has one value per prefix.
    """

    if not 0 <= decision_index < config.n - 1:
        raise ValueError("decision_index lies outside the autoregressive sweep")
    prefix_logs = _base_prefix_logs(params["base"], decision_index, config)
    centered_logs = prefix_logs - jnp.max(jnp.real(prefix_logs))
    amplitudes = jnp.exp(centered_logs)
    vector_operators = jnp.asarray(
        config.message_plan.vector_operators[decision_index]
    )
    scalar_operator = jnp.asarray(
        config.message_plan.closed_scalar_operators[decision_index]
    )
    vector_numerator = jnp.einsum(
        "kdno,n->kdo", vector_operators, amplitudes
    )
    vector = jnp.transpose(vector_numerator, (2, 0, 1)) / amplitudes[:, None, None]
    scalar = (scalar_operator.T @ amplitudes) / amplitudes
    return vector, scalar


def _bounded_components(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Signed-log compress complex messages without discarding their phase."""

    real = jnp.real(values)
    imaginary = jnp.imag(values)
    return (
        jnp.sign(real) * jnp.log1p(jnp.abs(real)),
        jnp.sign(imaginary) * jnp.log1p(jnp.abs(imaginary)),
    )


def _decision_feature_table(
    params: dict[str, object],
    decision_index: int,
    config: CausalBoundaryTreeGRUConfig,
) -> jax.Array:
    vector, scalar = causal_boundary_message_table(
        params, decision_index, config
    )
    vector_real, vector_imaginary = _bounded_components(vector)
    scalar_real, scalar_imaginary = _bounded_components(scalar)
    path_count = vector.shape[0]
    transition_valid = jnp.asarray(
        config.message_plan.vector_transition_valid[decision_index],
        dtype=jnp.float64,
    )
    channel_active = jnp.broadcast_to(
        jnp.asarray(
            config.message_plan.channel_active[decision_index],
            dtype=jnp.float64,
        )[None, :],
        (path_count, config.message_plan.maximum_channels),
    )
    return jnp.concatenate(
        (
            vector_real.reshape(path_count, -1),
            vector_imaginary.reshape(path_count, -1),
            scalar_real[:, None],
            scalar_imaginary[:, None],
            transition_valid,
            channel_active,
        ),
        axis=1,
    )


def _candidate_message_scores(
    local: dict[str, jax.Array],
    hidden: jax.Array,
    action_history: jax.Array,
    decision_index: int,
    feature_table: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Look up every candidate using only the already revealed action prefix."""

    single = action_history.ndim == 1
    if single:
        action_history = action_history[None, :]
        hidden = hidden[None, :]
    plan_actions = jnp.asarray(
        config.message_plan.action_sequences[decision_index]
    )
    if decision_index:
        prefix_matches = jnp.all(
            action_history[:, None, :]
            == plan_actions[None, :, :decision_index],
            axis=2,
        )
    else:
        prefix_matches = jnp.ones(
            (action_history.shape[0], len(plan_actions)), dtype=bool
        )
    candidate_slots = jnp.arange(config.action_dim, dtype=jnp.int32)
    matches = prefix_matches[:, :, None] & (
        plan_actions[None, :, decision_index, None]
        == candidate_slots[None, None, :]
    )
    indices = jnp.argmax(matches, axis=1)
    exists = jnp.any(matches, axis=1)
    candidate_features = feature_table[indices]
    candidate_features = jnp.where(
        exists[:, :, None], candidate_features, 0.0
    )

    def parameter(name: str) -> jax.Array:
        value = local[name]
        return value if config.train_message_encoder else jax.lax.stop_gradient(value)

    encoded = jnp.tanh(
        candidate_features @ parameter("message_projection")
        + (hidden @ parameter("context_projection"))[:, None, :]
        + parameter("bias")
    )
    probability = encoded @ local["probability_head"]
    phase = encoded @ local["phase_head"]
    if single:
        return probability[0], phase[0]
    return probability, phase


def _causal_boundary_tree_conditionals(
    params: dict[str, object],
    labels: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conditional log amplitudes from the strictly causal forward pass."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},), got {labels.shape}")
    base = params["base"]
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    history: list[jax.Array] = []
    conditionals: list[jax.Array] = []
    base_conditionals: list[jax.Array] = []
    valid_state = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        parent_in_range = (parent_q >= 0) & (parent_q <= config.n)
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
        candidates_left = table_left[decision.order, safe_parent]
        candidates_right = table_right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = (
            mask
            & (candidates_left == desired_left)
            & (candidates_right == desired_right)
        )
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
        logits, phase_values = heads(base, hidden)
        base_logits = jnp.where(mask, logits, -1.0e30)
        feature_table = _decision_feature_table(params, decision.order, config)
        local_logits, local_phases = _candidate_message_scores(
            params["message"],
            hidden,
            jnp.stack(history) if history else jnp.zeros((0,), dtype=jnp.int32),
            decision.order,
            feature_table,
            config,
        )
        logits = jnp.where(mask, logits + local_logits, -1.0e30)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        base_conditionals.append(
            0.5 * jax.nn.log_softmax(base_logits)[action]
            + 1j * phase_values[action]
        )
        conditionals.append(
            0.5 * jax.nn.log_softmax(logits)[action]
            + 1j * (phase_values[action] + local_phases[action])
        )
        history.append(action)
        previous = action
    return jnp.stack(conditionals), jnp.stack(base_conditionals), valid_state


def causal_boundary_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return conditional/cumulative logs without reading future actions."""

    conditionals, _, valid_state = _causal_boundary_tree_conditionals(
        params, labels, config
    )
    invalid = jnp.full_like(conditionals, -jnp.inf + 0j)
    conditionals = jnp.where(valid_state, conditionals, invalid)
    return conditionals, jnp.cumsum(conditionals)


def causal_boundary_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> jax.Array:
    """Exactly normalized log amplitude with causal physical messages."""

    conditionals, base_conditionals, valid_state = (
        _causal_boundary_tree_conditionals(params, labels, config)
    )
    base_value = compact_tree_log_psi(params["base"], labels, config.base)
    correction = jnp.sum(conditionals - base_conditionals)
    return jnp.where(valid_state, base_value + correction, base_value)


def causal_boundary_tree_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: CausalBoundaryTreeGRUConfig,
) -> jax.Array:
    logs = jax.vmap(
        lambda state: causal_boundary_tree_log_psi(params, state, config)
    )(states)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_causal_boundary_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: CausalBoundaryTreeGRUConfig,
) -> jax.Array:
    """Sample the causal-message model with the exact compact split mask."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    base = params["base"]
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(base["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    history: list[jax.Array] = []
    for decision in tree_decisions(config.tree):
        key, subkey = jax.random.split(key)
        parent_q = labels[:, decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
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
        logits, _ = heads(base, hidden)
        feature_table = _decision_feature_table(params, decision.order, config)
        action_history = (
            jnp.stack(history, axis=1)
            if history
            else jnp.zeros((n_samples, 0), dtype=jnp.int32)
        )
        local_logits, _ = _candidate_message_scores(
            params["message"],
            hidden,
            action_history,
            decision.order,
            feature_table,
            config,
        )
        logits = jnp.where(mask, logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        chosen_left = jnp.take_along_axis(
            candidates_left, action[:, None], axis=1
        )[:, 0]
        chosen_right = jnp.take_along_axis(
            candidates_right, action[:, None], axis=1
        )[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        history.append(action)
        previous = action
    return labels


__all__ = [
    "BOUNDARY_DELTA_Q",
    "CausalBoundaryMessagePlan",
    "CausalBoundaryTreeGRUConfig",
    "causal_boundary_message_table",
    "causal_boundary_tree_log_psi",
    "causal_boundary_tree_prefix_logs",
    "causal_boundary_tree_probabilities",
    "condition_existing_compact_tree_gru",
    "init_causal_boundary_tree_gru",
    "sample_causal_boundary_tree",
]
