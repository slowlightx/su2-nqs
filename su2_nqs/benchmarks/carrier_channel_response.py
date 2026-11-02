"""Channel-resolved linear response of the immutable exact carrier.

This module resolves a *directional* zero-head exact-carrier tangent into a
canonical causal ledger.  A bond-shell carrier is read after every
autoregressive decision and split into

``birth``
    sources created while revealing the current site(s),
``transport``
    pre-existing open endpoints transported through the current decision,
    together with scalar closures remembered from earlier decisions, and
``death``
    endpoints contracted to scalars during the current decision.

Every entry is additionally resolved by the doubled boundary-spin change
``delta_q=(-2, 0, +2)``.  The pieces add exactly to the ordinary analytic
zero-head tangent.  Keeping the decomposition directional avoids allocating
an enormous ``state x channel x parameter`` tensor at N=16.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from ..models._core import embed, gru_step, heads
from ..models.compact_tree_gru import split_tables
from ..models.exact_carrier_tree import (
    DECODER_HEAD_NAMES,
    ExactCarrierTreeGRUConfig,
    _plan_value,
)
from ..models.tree_gru import tree_decisions
from .sparse_causal_recurrence import SparseCausalRecurrencePlan


CARRIER_EVENT_NAMES: tuple[str, str, str] = ("birth", "transport", "death")
BIRTH_EVENT = 0
TRANSPORT_EVENT = 1
DEATH_EVENT = 2


@dataclass(frozen=True)
class WeightedProjection:
    """Weighted orthogonal projection onto a centered real feature span."""

    direction: np.ndarray
    fitted: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    target_norm_squared: float
    projected_norm_squared: float
    residual_norm_squared: float


def split_plan_sources_by_shell(
    plan: SparseCausalRecurrencePlan,
) -> dict[int, SparseCausalRecurrencePlan]:
    """Return shell-source carriers while retaining the full readout topology.

    Compiling each shell as a separate graph changes the number of active
    endpoints and hence the decoder's endpoint normalization.  That would not
    be an additive decomposition of the original model.  Here every returned
    plan retains the combined graph's endpoint slots, masks, transports and
    death tensors, but its ``source_coupling`` contains only one physical bond
    shell.  Consequently the shell carriers sum pointwise to the combined
    carrier in the *same* decoder gauge.
    """

    shell_ids = tuple(int(value) for value in np.asarray(plan.shell_ids))
    if not shell_ids:
        raise ValueError("plan contains no physical bond shells")
    matrices = {
        shell: np.zeros((plan.n_sites, plan.n_sites), dtype=np.float64)
        for shell in shell_ids
    }
    for first, second, coupling, shell_slot in zip(
        np.asarray(plan.bond_first),
        np.asarray(plan.bond_second),
        np.asarray(plan.bond_coupling),
        np.asarray(plan.bond_shell_slot),
        strict=True,
    ):
        shell = shell_ids[int(shell_slot)]
        matrices[shell][int(first), int(second)] += float(coupling)
        matrices[shell][int(second), int(first)] += float(coupling)

    result = {}
    for shell in shell_ids:
        source = np.zeros_like(plan.source_coupling)
        for decision in range(plan.n_decisions):
            for substep in range(plan.event_valid.shape[1]):
                if not plan.event_valid[decision, substep]:
                    continue
                revealed = int(plan.revealed_site[decision, substep])
                for slot in np.flatnonzero(
                    plan.new_endpoint_mask[decision, substep]
                ):
                    endpoint = int(
                        plan.new_endpoint_sites[decision, substep, int(slot)]
                    )
                    source[decision, substep, int(slot)] = matrices[shell][
                        endpoint, revealed
                    ]
        result[shell] = replace(plan, source_coupling=source)
    reconstructed = sum(
        (item.source_coupling for item in result.values()),
        start=np.zeros_like(plan.source_coupling),
    )
    if not np.allclose(reconstructed, plan.source_coupling, atol=1.0e-14, rtol=0.0):
        raise RuntimeError("shell source decomposition does not reconstruct the plan")
    return result


def weighted_centered_projection(
    features: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    *,
    relative_cutoff: float = 1.0e-8,
) -> WeightedProjection:
    """Project a centered real target in the exact weighted function metric.

    ``direction`` is the truncated Moore--Penrose solution in the supplied
    feature gauge.  ``fitted`` and all squared norms include the repository's
    realified VMC factor of two.  Constant target/score gauges are projected
    out explicitly before factorization.
    """

    return weighted_centered_projection_family(
        features,
        weights,
        target,
        relative_cutoffs=(relative_cutoff,),
    )[float(relative_cutoff)]


def weighted_centered_projection_family(
    features: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    *,
    relative_cutoffs: tuple[float, ...],
) -> dict[float, WeightedProjection]:
    """Factor once and return projections for several relative SVD cutoffs."""

    values = np.asarray(features, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64)
    response = np.asarray(target, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be a rank-two array")
    if probability.shape != (values.shape[0],):
        raise ValueError("weights must have one entry per feature row")
    if response.shape != (values.shape[0],):
        raise ValueError("target must have one entry per feature row")
    cutoffs = tuple(float(value) for value in relative_cutoffs)
    if not cutoffs:
        raise ValueError("relative_cutoffs must not be empty")
    if any(not np.isfinite(value) or value <= 0.0 for value in cutoffs):
        raise ValueError("relative_cutoff must be positive and finite")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("relative_cutoffs must be unique")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(response)):
        raise ValueError("features and target must be finite")
    if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(probability))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    probability = probability / total
    centered_values = values - probability @ values
    centered_response = response - probability @ response
    root_weight = np.sqrt(2.0 * probability)
    design = root_weight[:, None] * centered_values
    weighted_target = root_weight * centered_response
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    largest = float(singular[0]) if len(singular) else 0.0
    result = {}
    for cutoff in cutoffs:
        rank = int(np.count_nonzero(singular > cutoff * largest))
        if rank:
            coefficients = (left[:, :rank].T @ weighted_target) / singular[:rank]
            direction = right[:rank].T @ coefficients
            fitted = design @ direction
        else:
            direction = np.zeros((values.shape[1],), dtype=np.float64)
            fitted = np.zeros_like(weighted_target)
        residual = weighted_target - fitted
        result[cutoff] = WeightedProjection(
            direction=direction,
            fitted=fitted,
            singular_values=singular,
            numerical_rank=rank,
            target_norm_squared=float(np.dot(weighted_target, weighted_target)),
            projected_norm_squared=float(np.dot(fitted, fitted)),
            residual_norm_squared=float(np.dot(residual, residual)),
        )
    return result


def _split_direction(
    direction: jax.Array, config: ExactCarrierTreeGRUConfig
) -> tuple[jax.Array, jax.Array]:
    context_width = config.hidden_dim + 1
    endpoint_size = config.n * 3 * context_width
    role_size = endpoint_size + (config.n - 1) * context_width
    values = jnp.asarray(direction, dtype=jnp.float64)
    if values.shape != (role_size,):
        raise ValueError(f"expected a length-{role_size} decoder-role direction")
    return (
        values[:endpoint_size].reshape(config.n, 3, context_width),
        values[endpoint_size:].reshape(config.n - 1, context_width),
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


def _advance_component_candidate(
    endpoint: jax.Array,
    closed_by_delta: jax.Array,
    decision: int,
    q_before: jax.Array,
    candidate_q: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Advance one candidate and return its event ledger plus total state."""

    plan = config.recurrence_plan
    valid = np.asarray(_plan_value(plan, "event_valid"), dtype=bool)
    transported = endpoint
    born = jnp.zeros_like(endpoint)
    remembered = closed_by_delta
    died = jnp.zeros_like(closed_by_delta)
    current_q = q_before
    for substep in range(valid.shape[1]):
        if not valid[decision, substep]:
            continue
        q_after = _microstep_new_q(
            decision, substep, candidate_q, config
        )
        safe_before = jnp.clip(current_q, 0, config.n)
        safe_after = jnp.clip(q_after, 0, config.n)
        dying_slot = jnp.asarray(_plan_value(plan, "dying_old_slot"))[
            decision, substep
        ]
        safe_dying = jnp.clip(dying_slot, 0, config.endpoint_slots - 1)
        death_mask = jnp.asarray(
            _plan_value(plan, "death_mask"), dtype=jnp.float64
        )[decision, substep]
        contraction = jnp.asarray(_plan_value(plan, "dying_scalar"))[
            decision, substep, safe_before, safe_after
        ]
        died = died + death_mask * (
            transported[safe_dying] + born[safe_dying]
        ) * contraction

        old_slots = jnp.asarray(_plan_value(plan, "persist_old_slot_for_new"))[
            decision, substep
        ]
        safe_slots = jnp.clip(old_slots, 0, config.endpoint_slots - 1)
        persistent = old_slots >= 0
        transport_matrix = jnp.asarray(
            _plan_value(plan, "old_block_transport")
        )[decision, substep, safe_before, safe_after]
        active = jnp.asarray(
            _plan_value(plan, "new_endpoint_mask"), dtype=jnp.float64
        )[decision, substep]

        def move(values: jax.Array) -> jax.Array:
            gathered = values[safe_slots] * persistent[:, None]
            advanced = jnp.einsum("ab,wb->wa", transport_matrix, gathered)
            return advanced * active[:, None]

        transported = move(transported)
        born = move(born)
        born = born + (
            jnp.asarray(_plan_value(plan, "source_coupling"))[
                decision, substep, :, None
            ]
            * jnp.asarray(_plan_value(plan, "endpoint_source"))[
                decision, substep, safe_before, safe_after
            ][None, :]
            * active[:, None]
        )
        current_q = q_after

    total_endpoint = transported + born
    total_closed_by_delta = remembered + died
    return transported, born, remembered, died, total_endpoint, total_closed_by_delta


def _component_directional_scores(
    transported: jax.Array,
    born: jax.Array,
    remembered: jax.Array,
    died: jax.Array,
    hidden: jax.Array,
    decision: int,
    endpoint_direction: jax.Array,
    closed_direction: jax.Array,
    config: ExactCarrierTreeGRUConfig,
) -> jax.Array:
    """Return ``[event, delta_q]`` scalar scores for one candidate."""

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
    context = jnp.concatenate((hidden, jnp.ones((1,), dtype=jnp.float64)))
    weights = jnp.einsum("wdh,h->wd", endpoint_direction[safe_sites], context)
    normalization = jnp.sqrt(jnp.maximum(1.0, 3.0 * jnp.sum(active)))
    transport_endpoint = jnp.sum(
        transported * weights * active[:, None], axis=0
    ) / normalization
    birth_endpoint = jnp.sum(born * weights * active[:, None], axis=0) / normalization
    closed_weight = closed_direction[decision] @ context
    zero = jnp.zeros((3,), dtype=jnp.float64)
    return jnp.stack(
        (
            birth_endpoint,
            transport_endpoint + remembered * closed_weight,
            zero + died * closed_weight,
        ),
        axis=0,
    )


def exact_carrier_channel_directional_derivatives(
    params: dict[str, object],
    labels: jax.Array,
    config: ExactCarrierTreeGRUConfig,
    probability_direction: jax.Array,
    phase_direction: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Resolve two decoder-role directions by cut, event, and ``delta_q``.

    The decoder heads must be zero, so conditional probabilities are exactly
    those of the frozen base TreeGRU.  The return arrays both have shape
    ``(n_sites - 1, 3, 3)`` ordered as ``[cut, event, delta_q]``.  Probability
    entries include the exact autoregressive ``1/2`` and conditional-centering
    terms; phase entries are the chosen-action directional derivatives.
    """

    if any(
        bool(np.any(np.asarray(params["decoder"][name]) != 0.0))
        for name in DECODER_HEAD_NAMES
    ):
        raise ValueError("channel resolution requires exactly zero decoder heads")
    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},)")
    probability_endpoint, probability_closed = _split_direction(
        probability_direction, config
    )
    phase_endpoint, phase_closed = _split_direction(phase_direction, config)
    base = params["base"]
    left, right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    endpoint = jnp.zeros((config.endpoint_slots, 3), dtype=jnp.float64)
    closed_by_delta = jnp.zeros((3,), dtype=jnp.float64)
    probability_rows = []
    phase_rows = []

    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
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
        base_logits, _ = heads(base, hidden)

        def candidate_rows(q_after: jax.Array, is_valid: jax.Array):
            def valid_branch(_):
                components = _advance_component_candidate(
                    endpoint,
                    closed_by_delta,
                    decision.order,
                    safe_parent,
                    q_after,
                    config,
                )
                transported, born, remembered, died, total_endpoint, total_closed = (
                    components
                )
                probability_score = _component_directional_scores(
                    transported,
                    born,
                    remembered,
                    died,
                    hidden,
                    decision.order,
                    probability_endpoint,
                    probability_closed,
                    config,
                )
                phase_score = _component_directional_scores(
                    transported,
                    born,
                    remembered,
                    died,
                    hidden,
                    decision.order,
                    phase_endpoint,
                    phase_closed,
                    config,
                )
                return total_endpoint, total_closed, probability_score, phase_score

            def invalid_branch(_):
                zeros = jnp.zeros((3, 3), dtype=jnp.float64)
                return endpoint, closed_by_delta, zeros, zeros

            return jax.lax.cond(is_valid, valid_branch, invalid_branch, operand=None)

        candidate_endpoint, candidate_closed, probability_score, phase_score = (
            jax.vmap(candidate_rows)(candidates, mask)
        )
        masked_logits = jnp.where(mask, base_logits, -1.0e30)
        conditional_probability = jax.nn.softmax(masked_logits)
        probability_rows.append(
            0.5
            * (
                probability_score[action]
                - jnp.sum(
                    conditional_probability[:, None, None] * probability_score,
                    axis=0,
                )
            )
        )
        phase_rows.append(phase_score[action])
        endpoint = candidate_endpoint[action]
        closed_by_delta = candidate_closed[action]
        previous = action

    return jnp.stack(probability_rows), jnp.stack(phase_rows)


__all__ = [
    "CARRIER_EVENT_NAMES",
    "WeightedProjection",
    "exact_carrier_channel_directional_derivatives",
    "split_plan_sources_by_shell",
    "weighted_centered_projection",
    "weighted_centered_projection_family",
]
