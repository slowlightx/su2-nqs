"""Scalable causal SU(2) endpoint-message correction for a left TreeGRU.

Unlike :mod:`causal_boundary_tree`, this module never enumerates revealed
fusion prefixes and never stores an operator in multiplicity space.  It
consumes a fixed-shape, pure-numeric event plan and carries only

``endpoint x delta_q x message_hidden``

dynamic values, together with one closed/revealed scalar state and one
learned hidden-scalar proxy.  Every candidate action is advanced
hypothetically from the same committed history; only the sampled/evaluated
candidate is committed.  Consequently no suffix label is read by a score.

The graph partition and the local Condon--Shortley recoupling coefficients in
the plan are exact.  The finite-dimensional recurrent states are deliberately
a learned compression, not an exact contraction with an unknown suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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


SPARSE_CAUSAL_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)

_MESSAGE_HEAD_NAMES: tuple[str, ...] = (
    "probability_endpoint_head",
    "phase_endpoint_head",
    "probability_closed_head",
    "phase_closed_head",
    "probability_hidden_head",
    "phase_hidden_head",
)


def _plan_value(plan: object, *names: str) -> Any:
    for name in names:
        if hasattr(plan, name):
            return getattr(plan, name)
    joined = ", ".join(names)
    raise AttributeError(f"sparse recurrence plan is missing one of: {joined}")


def _maximum_endpoints(plan: object) -> int:
    return int(
        _plan_value(
            plan,
            "endpoint_slot_count",
            "maximum_active_endpoints",
            "maximum_endpoints",
            "maximum_endpoint_channels",
            "maximum_endpoint_count",
        )
    )


def _shell_count(plan: object) -> int:
    if hasattr(plan, "n_shells"):
        return max(int(getattr(plan, "n_shells")), 1)
    if hasattr(plan, "shell_count"):
        return max(int(getattr(plan, "shell_count")), 1)
    slots = np.asarray(
        _plan_value(plan, "closing_shell_slot", "closing_partner_shell_slot")
    )
    mask = np.asarray(
        _plan_value(plan, "closing_mask", "closing_partner_mask"), dtype=bool
    )
    return max(int(np.max(slots[mask], initial=-1)) + 1, 1)


@dataclass(frozen=True)
class SparseCausalTreeGRUConfig:
    """Configuration of the stable-endpoint causal recurrent layer.

    ``coefficient_scale=0`` is the parameter-matched K0 control: topology and
    every trainable array are retained, while all physical vector/scalar
    drives are zero.  Component and decision masks are diagnostic ablations;
    they alter no parameter shapes.
    """

    base: CompactTreeGRUConfig
    recurrence_plan: object
    message_hidden_dim: int = 24
    train_message_encoder: bool = False
    anchored_residual: bool = False
    coefficient_scale: int = 1
    enable_vector: bool = True
    enable_closed_scalar: bool = True
    enable_hidden_scalar: bool = True
    endpoint_slot_limit: int | None = None
    decision_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if self.base.tree != FusionTree.left_associated(self.base.n):
            raise ValueError("sparse causal recurrence requires a left-associated tree")
        if self.base.target_q != 0:
            raise ValueError("the current sparse recurrence requires target_q=0")
        if int(_plan_value(self.recurrence_plan, "n_sites")) != self.base.n:
            raise ValueError("recurrence plan and base tree sizes disagree")
        plan_target = int(getattr(self.recurrence_plan, "target_q", 0))
        if plan_target != self.base.target_q:
            raise ValueError("recurrence plan and target q disagree")
        if bool(getattr(self.recurrence_plan, "uses_hidden_suffix", False)):
            raise ValueError("a causal recurrence plan may not use a hidden suffix")
        if bool(getattr(self.recurrence_plan, "enumerates_revealed_prefixes", False)):
            raise ValueError("a production recurrence may not enumerate prefixes")
        delta_q = tuple(
            int(value)
            for value in getattr(
                self.recurrence_plan, "delta_q", SPARSE_CAUSAL_DELTA_Q
            )
        )
        if delta_q != SPARSE_CAUSAL_DELTA_Q:
            raise ValueError("q_op=2 messages require delta_q=(-2,0,+2)")
        if self.message_hidden_dim < 1:
            raise ValueError("message_hidden_dim must be positive")
        if self.anchored_residual and not self.train_message_encoder:
            raise ValueError(
                "anchored_residual requires train_message_encoder=True"
            )
        if isinstance(self.coefficient_scale, bool) or self.coefficient_scale not in (0, 1):
            raise ValueError("coefficient_scale must be the integer 0 or 1")
        if self.decision_mask is not None and len(self.decision_mask) != self.n - 1:
            raise ValueError("decision_mask must contain one entry per AR decision")
        if self.endpoint_slot_limit is not None and not (
            1 <= self.endpoint_slot_limit <= self.maximum_endpoints
        ):
            raise ValueError("endpoint_slot_limit lies outside the padded endpoint width")

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
    def maximum_endpoints(self) -> int:
        return _maximum_endpoints(self.recurrence_plan)

    @property
    def n_shells(self) -> int:
        return _shell_count(self.recurrence_plan)


def _matrix(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    scale = (2.0 / (shape[0] + shape[1])) ** 0.5
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_sparse_causal_tree_gru(
    key: jax.Array, config: SparseCausalTreeGRUConfig
) -> dict[str, object]:
    """Initialize at the base TreeGRU pointwise by zeroing complex heads."""

    keys = iter(jax.random.split(key, 24))
    base_key = next(keys)
    width = config.message_hidden_dim
    local = {
        "endpoint_embedding": 0.05
        * jax.random.normal(next(keys), (config.n, width), dtype=jnp.float64),
        "delta_embedding": 0.05
        * jax.random.normal(next(keys), (3, width), dtype=jnp.float64),
        "q_embedding": 0.05
        * jax.random.normal(next(keys), (config.n + 1, width), dtype=jnp.float64),
        "source_embedding": _matrix(next(keys), (3, width)),
        "endpoint_recurrent": _matrix(next(keys), (width, width)),
        "endpoint_context": _matrix(next(keys), (config.hidden_dim, width)),
        "endpoint_bias": jnp.zeros((width,), dtype=jnp.float64),
        "closed_recurrent": _matrix(next(keys), (width, width)),
        "closed_death": _matrix(next(keys), (width, width)),
        "closed_shell": _matrix(next(keys), (config.n_shells, width)),
        "closed_context": _matrix(next(keys), (config.hidden_dim, width)),
        "closed_bias": jnp.zeros((width,), dtype=jnp.float64),
        "hidden_recurrent": _matrix(next(keys), (width, width)),
        "hidden_shell": _matrix(next(keys), (config.n_shells, width)),
        "hidden_context": _matrix(next(keys), (config.hidden_dim, width)),
        "hidden_bias": jnp.zeros((width,), dtype=jnp.float64),
        # Site-resolved heads retain the stable endpoint gauge.  K0 has the
        # identical arrays and parameter count; only its numeric drives vanish.
        "probability_endpoint_head": jnp.zeros(
            (config.n, 3, width), dtype=jnp.float64
        ),
        "phase_endpoint_head": jnp.zeros(
            (config.n, 3, width), dtype=jnp.float64
        ),
        "probability_closed_head": jnp.zeros((width,), dtype=jnp.float64),
        "phase_closed_head": jnp.zeros((width,), dtype=jnp.float64),
        "probability_hidden_head": jnp.zeros((width,), dtype=jnp.float64),
        "phase_hidden_head": jnp.zeros((width,), dtype=jnp.float64),
    }
    result = {
        "base": init_compact_tree_gru(base_key, config.base),
        "message": local,
    }
    if config.anchored_residual:
        # The trainable heads above are *delta* heads.  A fixed nonzero head
        # makes encoder derivatives visible at initialization, while a frozen
        # encoder copy supplies the pointwise anchor subtraction.
        anchor = {
            name: jnp.array(value) for name, value in local.items()
        }
        anchor_scale = 0.12 / math.sqrt(width)
        anchor["probability_endpoint_head"] = anchor_scale * jax.random.normal(
            next(keys), (config.n, 3, width), dtype=jnp.float64
        )
        anchor["phase_endpoint_head"] = anchor_scale * jax.random.normal(
            next(keys), (config.n, 3, width), dtype=jnp.float64
        )
        anchor["probability_closed_head"] = anchor_scale * jax.random.normal(
            next(keys), (width,), dtype=jnp.float64
        )
        anchor["phase_closed_head"] = anchor_scale * jax.random.normal(
            next(keys), (width,), dtype=jnp.float64
        )
        anchor["probability_hidden_head"] = anchor_scale * jax.random.normal(
            next(keys), (width,), dtype=jnp.float64
        )
        anchor["phase_hidden_head"] = anchor_scale * jax.random.normal(
            next(keys), (width,), dtype=jnp.float64
        )
        result["anchor"] = {
            "base": jax.tree_util.tree_map(
                jax.lax.stop_gradient, result["base"]
            ),
            "message": jax.tree_util.tree_map(jax.lax.stop_gradient, anchor),
        }
    return result


def condition_existing_compact_tree_gru_sparse(
    base_params: dict[str, object],
    key: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> dict[str, object]:
    """Attach a zero-output sparse causal layer to a trained checkpoint."""

    initialized = init_sparse_causal_tree_gru(key, config)
    result = {"base": base_params, "message": initialized["message"]}
    if config.anchored_residual:
        result["anchor"] = {
            "base": jax.tree_util.tree_map(jax.lax.stop_gradient, base_params),
            "message": initialized["anchor"]["message"],
        }
    return result


def partition_sparse_causal_parameters(
    params: dict[str, object], config: SparseCausalTreeGRUConfig
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Split optimizer-visible parameters from immutable encoder arrays.

    With the default ``train_message_encoder=False``, the base TreeGRU and
    all six complex-residual heads form the trainable pytree.  Recurrent
    encoder arrays are returned separately and should be closed over by an SR
    or optimizer.  This avoids counting zero-derivative storage as trainable
    parameters and gives coefficient-zero/full models exactly the same
    trainable-pytree structure and count.  This does not claim equal active
    Jacobian rank: coefficient-zero message heads are deliberately null.
    """

    if config.train_message_encoder:
        trainable = {"base": params["base"], "message": params["message"]}
        frozen = (
            {"anchor": params["anchor"]}
            if config.anchored_residual
            else None
        )
        return trainable, frozen
    message = params["message"]
    trainable_message = {name: message[name] for name in _MESSAGE_HEAD_NAMES}
    frozen_message = {
        name: value
        for name, value in message.items()
        if name not in _MESSAGE_HEAD_NAMES
    }
    frozen: dict[str, object] = {"message": frozen_message}
    if config.anchored_residual:
        frozen["anchor"] = params["anchor"]
    return (
        {"base": params["base"], "message": trainable_message},
        frozen,
    )


def merge_sparse_causal_parameters(
    trainable: dict[str, object], frozen: dict[str, object] | None
) -> dict[str, object]:
    """Inverse of :func:`partition_sparse_causal_parameters`."""

    if frozen is None:
        return trainable
    if "message" in frozen:
        message = dict(frozen["message"])
        message.update(trainable["message"])
    else:
        message = trainable["message"]
    result = {"base": trainable["base"], "message": message}
    if "anchor" in frozen:
        result["anchor"] = frozen["anchor"]
    return result


def _encoder_parameter(
    local: dict[str, jax.Array], name: str, config: SparseCausalTreeGRUConfig
) -> jax.Array:
    value = local[name]
    return value if config.train_message_encoder else jax.lax.stop_gradient(value)


def _safe_exact_zero_norm(
    values: jax.Array,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> jax.Array:
    """Euclidean norm with an exact zero value and finite derivative at zero."""

    squared = jnp.sum(values * values, axis=axis, keepdims=keepdims)
    # Differentiating sqrt at an exact zero produces 0/0 before a later mask
    # can remove it.  The max branch has zero derivative below ``tiny`` and the
    # final where restores an exactly represented zero state.
    safe = jnp.sqrt(jnp.maximum(squared, jnp.finfo(values.dtype).tiny))
    return jnp.where(squared > 0.0, safe, 0.0)


def _zero_state(config: SparseCausalTreeGRUConfig) -> tuple[jax.Array, ...]:
    width = config.message_hidden_dim
    endpoints = jnp.zeros(
        (config.maximum_endpoints, 3, width), dtype=jnp.float64
    )
    scalar = jnp.zeros((width,), dtype=jnp.float64)
    hidden = jnp.zeros((width,), dtype=jnp.float64)
    return endpoints, scalar, hidden


def _event_shell_values(
    plan: object, decision: int, substep: int, n_shells: int
) -> jax.Array:
    couplings = jnp.asarray(
        _plan_value(plan, "closing_couplings", "closing_partner_couplings")
    )[decision, substep]
    slots = jnp.asarray(
        _plan_value(plan, "closing_shell_slot", "closing_partner_shell_slot"),
        dtype=jnp.int32,
    )[decision, substep]
    mask = jnp.asarray(
        _plan_value(plan, "closing_mask", "closing_partner_mask"),
        dtype=jnp.float64,
    )[decision, substep]
    safe_slots = jnp.clip(slots, 0, max(n_shells - 1, 0))
    return jnp.sum(
        jax.nn.one_hot(safe_slots, n_shells, dtype=jnp.float64)
        * (couplings * mask)[:, None],
        axis=0,
    )


def _hidden_shell_values(
    plan: object, decision: int, substep: int, n_shells: int
) -> jax.Array:
    for name in (
        "hidden_shell_after",
        "hidden_shell_sum_after",
        "hidden_shell_couplings_after",
        "hidden_scalar_shell_after",
    ):
        if hasattr(plan, name):
            values = jnp.asarray(getattr(plan, name))[decision, substep]
            return values[:n_shells]
    # A plan without a hidden-shell summary still has a valid vector/closed
    # recurrence.  The hidden environment then remains the honest zero proxy.
    return jnp.zeros((n_shells,), dtype=jnp.float64)


def _microstep_new_q(
    decision: int,
    substep: int,
    candidate_q: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> jax.Array:
    if decision < config.n - 2:
        return candidate_q
    # The last left-tree decision reveals sites 1 then 0.  Coupling the first
    # of those to the existing revealed block has q=1, after which the fixed
    # singlet target has q=0.
    return jnp.asarray(1 if substep == 0 else config.target_q, dtype=jnp.int32)


def _advance_event(
    local: dict[str, jax.Array],
    endpoint_state: jax.Array,
    closed_state: jax.Array,
    hidden_state: jax.Array,
    base_hidden: jax.Array,
    decision: int,
    substep: int,
    q_before: jax.Array,
    q_after: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance one reveal event using only the committed causal state."""

    plan = config.recurrence_plan
    q_before = jnp.clip(q_before, 0, config.n)
    q_after = jnp.clip(q_after, 0, config.n)
    scale = jnp.asarray(config.coefficient_scale, dtype=jnp.float64)

    old_slot = jnp.asarray(
        _plan_value(plan, "persist_old_slot_for_new"), dtype=jnp.int32
    )[decision, substep]
    safe_old_slot = jnp.clip(old_slot, 0, max(config.maximum_endpoints - 1, 0))
    persistent = old_slot >= 0
    gathered = endpoint_state[safe_old_slot]
    gathered = gathered * persistent[:, None, None]

    transport = jnp.asarray(_plan_value(plan, "old_block_transport"))[
        decision, substep, q_before, q_after
    ]
    transported = jnp.einsum("ab,wbh->wah", transport, gathered)
    source_coefficient = jnp.asarray(_plan_value(plan, "endpoint_source"))[
        decision, substep, q_before, q_after
    ]
    source_coupling = jnp.asarray(_plan_value(plan, "source_coupling"))[
        decision, substep
    ]
    source = scale * source_coupling[:, None] * source_coefficient[None, :]
    drive = (
        transported @ _encoder_parameter(local, "endpoint_recurrent", config)
        + source[:, :, None]
        * _encoder_parameter(local, "source_embedding", config)[None, :, :]
    )

    new_sites = jnp.asarray(
        _plan_value(plan, "new_endpoint_sites"), dtype=jnp.int32
    )[decision, substep]
    new_mask = jnp.asarray(
        _plan_value(plan, "new_endpoint_mask", "new_endpoint_active"),
        dtype=jnp.float64,
    )[decision, substep]
    if config.endpoint_slot_limit is not None:
        new_mask = new_mask * (
            jnp.arange(config.maximum_endpoints) < config.endpoint_slot_limit
        )
    safe_sites = jnp.clip(new_sites, 0, config.n - 1)
    context = (
        _encoder_parameter(local, "endpoint_embedding", config)[safe_sites, None, :]
        + _encoder_parameter(local, "delta_embedding", config)[None, :, :]
        + _encoder_parameter(local, "q_embedding", config)[q_after]
        + base_hidden
        @ _encoder_parameter(local, "endpoint_context", config)
        + _encoder_parameter(local, "endpoint_bias", config)
    )
    # Context may reshape an existing physical message, but may not create a
    # channel from zero.  This also makes coefficient_scale=0 an exact K0.
    magnitude = _safe_exact_zero_norm(drive, axis=-1, keepdims=True)
    new_endpoint_state = jnp.tanh(drive + magnitude * jnp.tanh(context))
    new_endpoint_state = new_endpoint_state * new_mask[:, None, None]

    death_mask = jnp.asarray(
        _plan_value(plan, "death_mask"), dtype=jnp.float64
    )[decision, substep]
    dying_slot = jnp.asarray(
        _plan_value(plan, "dying_old_slot"), dtype=jnp.int32
    )[decision, substep]
    safe_dying = jnp.clip(dying_slot, 0, max(config.maximum_endpoints - 1, 0))
    death_coefficients = jnp.asarray(_plan_value(plan, "dying_scalar"))[
        decision, substep, q_before, q_after
    ]
    death_message = death_mask * jnp.sum(
        endpoint_state[safe_dying]
        * (scale * death_coefficients)[:, None],
        axis=0,
    )
    closed_shell = scale * _event_shell_values(
        plan, decision, substep, config.n_shells
    )
    closed_drive = (
        closed_state @ _encoder_parameter(local, "closed_recurrent", config)
        + death_message @ _encoder_parameter(local, "closed_death", config)
        + closed_shell @ _encoder_parameter(local, "closed_shell", config)
    )
    closed_magnitude = _safe_exact_zero_norm(closed_drive)
    closed_context = (
        base_hidden @ _encoder_parameter(local, "closed_context", config)
        + _encoder_parameter(local, "q_embedding", config)[q_after]
        + _encoder_parameter(local, "closed_bias", config)
    )
    new_closed_state = jnp.tanh(
        closed_drive + closed_magnitude * jnp.tanh(closed_context)
    )

    hidden_shell = scale * _hidden_shell_values(
        plan, decision, substep, config.n_shells
    )
    hidden_drive = (
        hidden_state @ _encoder_parameter(local, "hidden_recurrent", config)
        + hidden_shell @ _encoder_parameter(local, "hidden_shell", config)
    )
    hidden_magnitude = _safe_exact_zero_norm(hidden_drive)
    hidden_context = (
        base_hidden @ _encoder_parameter(local, "hidden_context", config)
        + _encoder_parameter(local, "q_embedding", config)[q_after]
        + _encoder_parameter(local, "hidden_bias", config)
    )
    new_hidden_state = jnp.tanh(
        hidden_drive + hidden_magnitude * jnp.tanh(hidden_context)
    )
    return new_endpoint_state, new_closed_state, new_hidden_state


def _advance_candidate(
    local: dict[str, jax.Array],
    endpoint_state: jax.Array,
    closed_state: jax.Array,
    hidden_state: jax.Array,
    base_hidden: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    plan = config.recurrence_plan
    current_q = q_before
    state = (endpoint_state, closed_state, hidden_state)
    event_valid = np.asarray(_plan_value(plan, "event_valid"), dtype=bool)
    for substep in range(event_valid.shape[1]):
        if not event_valid[decision, substep]:
            continue
        next_q = _microstep_new_q(decision, substep, candidate_q, config)
        state = _advance_event(
            local,
            *state,
            base_hidden,
            decision,
            substep,
            current_q,
            next_q,
            config,
        )
        current_q = next_q
    return state


def _score_state(
    local: dict[str, jax.Array],
    state: tuple[jax.Array, jax.Array, jax.Array],
    decision: int,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    endpoint_state, closed_state, hidden_state = state
    plan = config.recurrence_plan
    event_valid = np.asarray(_plan_value(plan, "event_valid"), dtype=bool)
    last_substep = int(np.flatnonzero(event_valid[decision])[-1])
    sites = jnp.asarray(
        _plan_value(plan, "new_endpoint_sites"), dtype=jnp.int32
    )[decision, last_substep]
    active = jnp.asarray(
        _plan_value(plan, "new_endpoint_mask", "new_endpoint_active"),
        dtype=jnp.float64,
    )[decision, last_substep]
    if config.endpoint_slot_limit is not None:
        active = active * (
            jnp.arange(config.maximum_endpoints) < config.endpoint_slot_limit
        )
    safe_sites = jnp.clip(sites, 0, config.n - 1)
    probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    if config.enable_vector:
        probability += jnp.sum(
            endpoint_state
            * local["probability_endpoint_head"][safe_sites]
            * active[:, None, None]
        ) / jnp.sqrt(jnp.maximum(1.0, 3.0 * jnp.sum(active)))
        phase += jnp.sum(
            endpoint_state
            * local["phase_endpoint_head"][safe_sites]
            * active[:, None, None]
        ) / jnp.sqrt(jnp.maximum(1.0, 3.0 * jnp.sum(active)))
    if config.enable_closed_scalar:
        probability += closed_state @ local["probability_closed_head"]
        phase += closed_state @ local["phase_closed_head"]
    if config.enable_hidden_scalar:
        probability += hidden_state @ local["probability_hidden_head"]
        phase += hidden_state @ local["phase_hidden_head"]
    enabled = (
        True if config.decision_mask is None else config.decision_mask[decision]
    )
    factor = jnp.asarray(enabled, dtype=jnp.float64)
    return factor * probability, factor * phase


def _candidate_states_and_scores(
    local: dict[str, jax.Array],
    endpoint_state: jax.Array,
    closed_state: jax.Array,
    hidden_state: jax.Array,
    base_hidden: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    candidate_mask: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array, jax.Array]:
    """Advance and score valid candidates only; padded actions stay inert."""

    def one(q_after: jax.Array, is_valid: jax.Array):
        def score_valid(_):
            state = _advance_candidate(
                local,
                endpoint_state,
                closed_state,
                hidden_state,
                base_hidden,
                decision,
                q_before,
                q_after,
                config,
            )
            probability, phase = _score_state(local, state, decision, config)
            return state, probability, phase

        def keep_committed(_):
            zero = jnp.asarray(0.0, dtype=jnp.float64)
            return (endpoint_state, closed_state, hidden_state), zero, zero

        return jax.lax.cond(is_valid, score_valid, keep_committed, operand=None)

    return jax.vmap(one)(candidate_q, candidate_mask)


def _anchored_message_pair(
    params: dict[str, object], config: SparseCausalTreeGRUConfig
) -> tuple[dict[str, jax.Array], dict[str, jax.Array] | None]:
    """Return the effective trainable message and immutable anchor copy."""

    local = params["message"]
    if not config.anchored_residual:
        return local, None
    anchor = jax.tree_util.tree_map(
        jax.lax.stop_gradient, params["anchor"]["message"]
    )
    effective = dict(local)
    for name in _MESSAGE_HEAD_NAMES:
        # ``local[name]`` is a zero-initialized delta head.  The nonzero
        # immutable anchor head exposes encoder derivatives immediately.
        effective[name] = local[name] + anchor[name]
    return effective, anchor


def _sparse_causal_conditionals(
    params: dict[str, object],
    labels: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},), got {labels.shape}")
    base = params["base"]
    local, anchor_local = _anchored_message_pair(params, config)
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    base_hidden = base["h0"]
    if anchor_local is not None:
        anchor_base = jax.tree_util.tree_map(
            jax.lax.stop_gradient, params["anchor"]["base"]
        )
        anchor_base_hidden = anchor_base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    endpoint_state, closed_state, hidden_state = _zero_state(config)
    if anchor_local is not None:
        anchor_endpoint_state, anchor_closed_state, anchor_hidden_state = _zero_state(
            config
        )
    conditionals = []
    base_conditionals = []
    endpoint_trace = []
    closed_trace = []
    hidden_trace = []
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
        base_hidden = gru_step(base["gru"], base_hidden, inputs)
        if anchor_local is not None:
            anchor_inputs = embed(
                anchor_base,
                (
                    previous,
                    safe_parent,
                    decision.left_leaves,
                    decision.right_leaves,
                    decision.depth,
                    decision.order,
                ),
            )
            anchor_base_hidden = jax.lax.stop_gradient(
                gru_step(
                    anchor_base["gru"], anchor_base_hidden, anchor_inputs
                )
            )
        base_logits, base_phases = heads(base, base_hidden)
        candidate_states, local_logits, local_phases = _candidate_states_and_scores(
            local,
            endpoint_state,
            closed_state,
            hidden_state,
            base_hidden,
            decision.order,
            safe_parent,
            candidates_left,
            mask,
            config,
        )
        if anchor_local is not None:
            anchor_candidate_states, anchor_logits, anchor_phases = (
                _candidate_states_and_scores(
                    anchor_local,
                    anchor_endpoint_state,
                    anchor_closed_state,
                    anchor_hidden_state,
                    anchor_base_hidden,
                    decision.order,
                    safe_parent,
                    candidates_left,
                    mask,
                    config,
                )
            )
            anchor_candidate_states = jax.tree_util.tree_map(
                jax.lax.stop_gradient, anchor_candidate_states
            )
            local_logits = local_logits - jax.lax.stop_gradient(anchor_logits)
            local_phases = local_phases - jax.lax.stop_gradient(anchor_phases)
        masked_base_logits = jnp.where(mask, base_logits, -1.0e30)
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        base_conditionals.append(
            0.5 * jax.nn.log_softmax(masked_base_logits)[action]
            + 1j * base_phases[action]
        )
        conditionals.append(
            0.5 * jax.nn.log_softmax(logits)[action]
            + 1j * (base_phases[action] + local_phases[action])
        )
        endpoint_state = candidate_states[0][action]
        closed_state = candidate_states[1][action]
        hidden_state = candidate_states[2][action]
        if anchor_local is not None:
            anchor_endpoint_state = anchor_candidate_states[0][action]
            anchor_closed_state = anchor_candidate_states[1][action]
            anchor_hidden_state = anchor_candidate_states[2][action]
        endpoint_trace.append(endpoint_state)
        closed_trace.append(closed_state)
        hidden_trace.append(hidden_state)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        previous = action
    return (
        jnp.stack(conditionals),
        jnp.stack(base_conditionals),
        valid_state,
        (
            jnp.stack(endpoint_trace),
            jnp.stack(closed_trace),
            jnp.stack(hidden_trace),
        ),
    )


def sparse_causal_tree_prefix_logs(
    params: dict[str, object],
    labels: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return conditionals/cumulative logs from the prefix-only recurrence."""

    conditionals, _, valid, _ = _sparse_causal_conditionals(params, labels, config)
    invalid = jnp.full_like(conditionals, -jnp.inf + 0.0j)
    conditionals = jnp.where(valid, conditionals, invalid)
    return conditionals, jnp.cumsum(conditionals)


def sparse_causal_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> jax.Array:
    """Exactly normalized complex log amplitude with sparse causal messages."""

    conditionals, base_conditionals, valid, _ = _sparse_causal_conditionals(
        params, labels, config
    )
    base_value = compact_tree_log_psi(params["base"], labels, config.base)
    correction = jnp.sum(conditionals - base_conditionals)
    return jnp.where(valid, base_value + correction, base_value)


def sparse_causal_tree_message_states(
    params: dict[str, object],
    labels: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return committed endpoint/closed/hidden states after every decision.

    This is a causal diagnostic, not an additional model input.  It is useful
    for proving that the coefficient-zero K0 has identically zero dynamic
    state rather than merely a zero output head.
    """

    _, _, valid, trace = _sparse_causal_conditionals(params, labels, config)
    zeros = tuple(jnp.zeros_like(value) for value in trace)
    return tuple(jnp.where(valid, value, zero) for value, zero in zip(trace, zeros))


def sparse_causal_tree_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: SparseCausalTreeGRUConfig,
) -> jax.Array:
    logs = jax.vmap(
        lambda state: sparse_causal_tree_log_psi(params, state, config)
    )(states)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_sparse_causal_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: SparseCausalTreeGRUConfig,
) -> jax.Array:
    """Sample with the exact compact mask and candidate-commit recurrence."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    base = params["base"]
    local, anchor_local = _anchored_message_pair(params, config)
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    base_hidden = jnp.broadcast_to(base["h0"], (n_samples, config.hidden_dim))
    if anchor_local is not None:
        anchor_base = jax.tree_util.tree_map(
            jax.lax.stop_gradient, params["anchor"]["base"]
        )
        anchor_base_hidden = jnp.broadcast_to(
            anchor_base["h0"], (n_samples, config.hidden_dim)
        )
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    zero_endpoint, zero_closed, zero_hidden = _zero_state(config)
    endpoint_state = jnp.broadcast_to(
        zero_endpoint, (n_samples,) + zero_endpoint.shape
    )
    closed_state = jnp.broadcast_to(zero_closed, (n_samples,) + zero_closed.shape)
    hidden_state = jnp.broadcast_to(zero_hidden, (n_samples,) + zero_hidden.shape)
    if anchor_local is not None:
        anchor_endpoint_state = endpoint_state
        anchor_closed_state = closed_state
        anchor_hidden_state = hidden_state

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
        base_hidden = gru_step(base["gru"], base_hidden, inputs)
        if anchor_local is not None:
            anchor_inputs = embed(
                anchor_base,
                (
                    previous,
                    parent_q,
                    jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                    jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                    jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                    jnp.full((n_samples,), decision.order, dtype=jnp.int32),
                ),
            )
            anchor_base_hidden = jax.lax.stop_gradient(
                gru_step(
                    anchor_base["gru"], anchor_base_hidden, anchor_inputs
                )
            )
        base_logits, _ = heads(base, base_hidden)

        def one_sample(z, c, u, h, q, candidate, candidate_valid):
            return _candidate_states_and_scores(
                local,
                z,
                c,
                u,
                h,
                decision.order,
                q,
                candidate,
                candidate_valid,
                config,
            )

        candidate_states, local_logits, _ = jax.vmap(one_sample)(
            endpoint_state,
            closed_state,
            hidden_state,
            base_hidden,
            parent_q,
            candidates_left,
            mask,
        )
        if anchor_local is not None:
            def one_anchor_sample(z, c, u, h, q, candidate, candidate_valid):
                return _candidate_states_and_scores(
                    anchor_local,
                    z,
                    c,
                    u,
                    h,
                    decision.order,
                    q,
                    candidate,
                    candidate_valid,
                    config,
                )

            anchor_candidate_states, anchor_logits, _ = jax.vmap(
                one_anchor_sample
            )(
                anchor_endpoint_state,
                anchor_closed_state,
                anchor_hidden_state,
                anchor_base_hidden,
                parent_q,
                candidates_left,
                mask,
            )
            anchor_candidate_states = jax.tree_util.tree_map(
                jax.lax.stop_gradient, anchor_candidate_states
            )
            local_logits = local_logits - jax.lax.stop_gradient(anchor_logits)
        logits = jnp.where(mask, base_logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        batch = jnp.arange(n_samples, dtype=jnp.int32)
        endpoint_state = candidate_states[0][batch, action]
        closed_state = candidate_states[1][batch, action]
        hidden_state = candidate_states[2][batch, action]
        if anchor_local is not None:
            anchor_endpoint_state = anchor_candidate_states[0][batch, action]
            anchor_closed_state = anchor_candidate_states[1][batch, action]
            anchor_hidden_state = anchor_candidate_states[2][batch, action]
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
        previous = action
    return labels


__all__ = [
    "SPARSE_CAUSAL_DELTA_Q",
    "SparseCausalTreeGRUConfig",
    "condition_existing_compact_tree_gru_sparse",
    "init_sparse_causal_tree_gru",
    "merge_sparse_causal_parameters",
    "partition_sparse_causal_parameters",
    "sample_sparse_causal_tree",
    "sparse_causal_tree_log_psi",
    "sparse_causal_tree_message_states",
    "sparse_causal_tree_prefix_logs",
    "sparse_causal_tree_probabilities",
]
