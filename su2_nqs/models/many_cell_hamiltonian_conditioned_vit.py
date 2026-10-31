"""Bond-causal SU(2)-scalar head for the many-cell Schur Spatial-ViT.

The base autoregressive ViT is left unchanged.  A zero-output residual head
sees only doubled-spin labels on a physical bond's fusion-tree path, and only
after every label on that path is revealed.  The construction therefore adds
configuration-dependent Hamiltonian geometry without violating exact
reachability masks or autoregressive normalization.

This module intentionally provides the exact-small log-amplitude gate first.
Sampling is promoted only if the added tangent beats a bond-scrambled control.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .compact_tree_gru import split_tables
from .hamiltonian_conditioned_tree import (
    HamiltonianBond,
    bond_causal_cone_tables,
)
from .many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    _candidate_features,
    _candidate_scores,
    _context_tokens,
    _global_configuration_phase,
    _selected_tree_actions,
)
from .tree_gru import tree_decisions


@dataclass(frozen=True)
class ManyCellHamiltonianConditionedViTConfig:
    """A many-cell Spatial-ViT plus shared bond-path feature channels."""

    base: ManyCellSchurViTConfig
    bonds: tuple[HamiltonianBond, ...]
    n_channels: int
    local_embedding_dim: int = 8
    local_hidden_dim: int = 32
    train_local_encoder: bool = False

    def __post_init__(self) -> None:
        for name in ("n_channels", "local_embedding_dim", "local_hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.train_local_encoder, bool):
            raise TypeError("train_local_encoder must be boolean")
        for bond in self.bonds:
            if not 0 <= bond.first < self.n or not 0 <= bond.second < self.n:
                raise ValueError("Hamiltonian bond endpoint lies outside the tree")
            if bond.channel >= self.n_channels:
                raise ValueError("Hamiltonian bond channel lies outside n_channels")

    @property
    def n(self) -> int:
        return self.base.n


def _matrix(key: jax.Array, rows: int, columns: int) -> jax.Array:
    scale = np.sqrt(2.0 / float(rows + columns))
    return scale * jax.random.normal(key, (rows, columns), dtype=jnp.float64)


def init_many_cell_hamiltonian_conditioning_head(
    key: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> dict[str, jax.Array]:
    """Initialize a zero-output head with an active first-order tangent."""

    keys = iter(jax.random.split(key, 11))
    embedding = config.local_embedding_dim
    hidden = config.local_hidden_dim
    n = config.n
    return {
        "q_embedding": 0.05
        * jax.random.normal(next(keys), (n + 1, embedding), dtype=jnp.float64),
        "role_embedding": 0.05
        * jax.random.normal(next(keys), (3, embedding), dtype=jnp.float64),
        "distance_embedding": 0.05
        * jax.random.normal(next(keys), (n + 1, embedding), dtype=jnp.float64),
        "size_embedding": 0.05
        * jax.random.normal(next(keys), (n + 1, embedding), dtype=jnp.float64),
        "pair_embedding": 0.05
        * jax.random.normal(
            next(keys), ((n + 1) * (n + 1), embedding), dtype=jnp.float64
        ),
        "length_embedding": 0.05
        * jax.random.normal(next(keys), (n + 1, embedding), dtype=jnp.float64),
        "channel_embedding": 0.05
        * jax.random.normal(
            next(keys), (config.n_channels, embedding), dtype=jnp.float64
        ),
        "node_projection": _matrix(next(keys), 4 * embedding, hidden),
        "pair_projection": _matrix(next(keys), embedding, hidden),
        "context_projection": _matrix(
            next(keys), config.base.embedding_dim, hidden
        ),
        "global_projection": _matrix(next(keys), 2 * embedding, hidden),
        "bias": jnp.zeros((hidden,), dtype=jnp.float64),
        "probability_head": jnp.zeros(
            (config.n_channels, hidden), dtype=jnp.float64
        ),
        "phase_head": jnp.zeros(
            (config.n_channels, hidden), dtype=jnp.float64
        ),
    }


def condition_many_cell_schur_vit(
    base_params: dict[str, object],
    key: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> dict[str, object]:
    """Attach the zero-output bond head without changing the wavefunction."""

    return {
        "base": base_params,
        "local": init_many_cell_hamiltonian_conditioning_head(key, config),
    }


def _local_candidate_features(
    local: dict[str, jax.Array],
    labels: jax.Array,
    contexts: jax.Array,
    decision,
    candidates_left: jax.Array,
    candidates_right: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> jax.Array:
    """Return ``(batch, actions, channels, hidden)`` causal bond features."""

    def parameter(name: str) -> jax.Array:
        value = local[name]
        return value if config.train_local_encoder else jax.lax.stop_gradient(value)

    tables = tuple(
        jnp.asarray(value)
        for value in bond_causal_cone_tables(config.base.layout.tree, config.bonds)
    )
    (
        all_indices,
        all_path_masks,
        all_roles,
        all_distances,
        all_sizes,
        all_channels,
        all_couplings,
        all_bond_masks,
        all_lengths,
    ) = tables
    indices = all_indices[decision.order]
    path_mask = all_path_masks[decision.order]
    roles = all_roles[decision.order]
    distances = all_distances[decision.order]
    sizes = all_sizes[decision.order]
    channels = all_channels[decision.order]
    couplings = all_couplings[decision.order]
    active_bonds = all_bond_masks[decision.order]
    lengths = all_lengths[decision.order]

    batch, action_count = candidates_left.shape
    candidate_labels = jnp.broadcast_to(
        labels[:, None, :], (batch, action_count, config.n - 1)
    )
    if decision.left_index is not None:
        candidate_labels = candidate_labels.at[:, :, decision.left_index].set(
            candidates_left
        )
    if decision.right_index is not None:
        candidate_labels = candidate_labels.at[:, :, decision.right_index].set(
            candidates_right
        )
    path_q = candidate_labels[:, :, indices]
    path_q = jnp.where(path_mask[None, None, :, :], path_q, 0)

    node_inputs = jnp.concatenate(
        (
            parameter("q_embedding")[path_q],
            jnp.broadcast_to(
                parameter("role_embedding")[roles][None, None, ...],
                path_q.shape + (config.local_embedding_dim,),
            ),
            jnp.broadcast_to(
                parameter("distance_embedding")[distances][None, None, ...],
                path_q.shape + (config.local_embedding_dim,),
            ),
            jnp.broadcast_to(
                parameter("size_embedding")[sizes][None, None, ...],
                path_q.shape + (config.local_embedding_dim,),
            ),
        ),
        axis=-1,
    )
    node = jnp.tanh(node_inputs @ parameter("node_projection"))
    node = jnp.sum(
        node * path_mask[None, None, :, :, None], axis=3
    ) / jnp.sqrt(lengths)[None, None, :, None]

    pair_mask = path_mask[:, :-1] & path_mask[:, 1:]
    pair_index = path_q[:, :, :, :-1] * (config.n + 1) + path_q[:, :, :, 1:]
    pair = parameter("pair_embedding")[pair_index] @ parameter("pair_projection")
    pair = jnp.sum(
        pair * pair_mask[None, None, :, :, None], axis=3
    ) / jnp.sqrt(jnp.maximum(lengths - 1, 1))[None, None, :, None]

    global_input = jnp.concatenate(
        (
            parameter("channel_embedding")[channels],
            parameter("length_embedding")[lengths],
        ),
        axis=-1,
    )
    global_feature = global_input @ parameter("global_projection")
    context = contexts @ parameter("context_projection")
    bond = jnp.tanh(
        node
        + pair
        + global_feature[None, None, :, :]
        + context[:, None, None, :]
        + parameter("bias")
    )
    bond = bond * (couplings * active_bonds)[None, None, :, None]
    channel_feature = jnp.zeros(
        (batch, action_count, config.n_channels, config.local_hidden_dim),
        dtype=jnp.float64,
    )
    return channel_feature.at[:, :, channels, :].add(bond)


def many_cell_hamiltonian_conditioned_log_amplitudes(
    params: dict[str, object],
    labels: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> jax.Array:
    """Evaluate normalized exact-SU(2) amplitudes with the causal bond head."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    base = params["base"]
    local = params["local"]
    actions, masks, chosen_features, valid = _selected_tree_actions(
        values, config.base
    )
    tokens = _context_tokens(base, chosen_features, config.base)
    left, right, _ = (
        jnp.asarray(value) for value in split_tables(config.base.layout.tree)
    )
    candidate_features = jnp.asarray(
        _candidate_features(config.base), dtype=jnp.float64
    )
    parent = values[:, [
        decision.node_index for decision in tree_decisions(config.base.layout.tree)
    ]]
    safe_parent = jnp.clip(parent, 0, config.n)
    order = jnp.arange(config.base.decision_count)[None, :]
    all_candidates = candidate_features[order, safe_parent]
    logits = _candidate_scores(base, tokens, all_candidates, config.base)
    local_phases = []
    for decision in tree_decisions(config.base.layout.tree):
        candidate_left = left[decision.order, safe_parent[:, decision.order]]
        candidate_right = right[decision.order, safe_parent[:, decision.order]]
        features = _local_candidate_features(
            local,
            values,
            tokens[:, decision.order],
            decision,
            candidate_left,
            candidate_right,
            config,
        )
        logits = logits.at[:, decision.order].add(
            jnp.einsum("bach,ch->ba", features, local["probability_head"])
        )
        phases = jnp.einsum("bach,ch->ba", features, local["phase_head"])
        local_phases.append(
            jnp.take_along_axis(
                phases, actions[:, decision.order, None], axis=1
            )[:, 0]
        )
    log_probabilities = jax.nn.log_softmax(
        jnp.where(masks, logits, -jnp.inf), axis=-1
    )
    selected_log_p = jnp.take_along_axis(
        log_probabilities, actions[..., None], axis=-1
    )[..., 0]
    phase = _global_configuration_phase(base, chosen_features, config.base)
    phase += jnp.sum(jnp.stack(local_phases, axis=1), axis=1)
    result = 0.5 * jnp.sum(selected_log_p, axis=1) + 1.0j * phase
    return jnp.where(valid, result, -jnp.inf + 0.0j)


def many_cell_hamiltonian_conditioned_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> jax.Array:
    """Single-state counterpart of the batched evaluator."""

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.shape != (config.n - 1,):
        raise ValueError(f"labels must have shape ({config.n - 1},)")
    return many_cell_hamiltonian_conditioned_log_amplitudes(
        params, values[None, :], config
    )[0]


def prepare_many_cell_hamiltonian_conditioned_head_design(
    base_params: dict[str, object],
    local_encoder: dict[str, jax.Array],
    labels: jax.Array,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> dict[str, jax.Array]:
    """Cache the exact dependence on the two active output heads.

    The local encoder and base Spatial-ViT are held fixed.  The returned design
    is therefore sufficient to train ``probability_head`` and ``phase_head``
    without retracing the frozen network.  It is an exact algebraic
    factorization, not a tangent approximation.
    """

    values = jnp.asarray(labels, dtype=jnp.int32)
    if values.ndim != 2 or values.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    actions, masks, chosen_features, valid = _selected_tree_actions(
        values, config.base
    )
    tokens = _context_tokens(base_params, chosen_features, config.base)
    left, right, _ = (
        jnp.asarray(value) for value in split_tables(config.base.layout.tree)
    )
    candidate_features = jnp.asarray(
        _candidate_features(config.base), dtype=jnp.float64
    )
    decisions = tree_decisions(config.base.layout.tree)
    parent = values[:, [decision.node_index for decision in decisions]]
    safe_parent = jnp.clip(parent, 0, config.n)
    order = jnp.arange(config.base.decision_count)[None, :]
    all_candidates = candidate_features[order, safe_parent]
    base_logits = _candidate_scores(
        base_params, tokens, all_candidates, config.base
    )
    local_features = []
    for decision in decisions:
        candidate_left = left[decision.order, safe_parent[:, decision.order]]
        candidate_right = right[decision.order, safe_parent[:, decision.order]]
        local_features.append(
            _local_candidate_features(
                local_encoder,
                values,
                tokens[:, decision.order],
                decision,
                candidate_left,
                candidate_right,
                config,
            )
        )
    features = jnp.stack(local_features, axis=1)
    selected = jnp.take_along_axis(
        features,
        actions[:, :, None, None, None],
        axis=2,
    )[:, :, 0]
    return {
        "actions": actions,
        "masks": masks,
        "valid": valid,
        "base_logits": base_logits,
        "base_phase": _global_configuration_phase(
            base_params, chosen_features, config.base
        ),
        "probability_features": features,
        "phase_features": jnp.sum(selected, axis=1),
    }


def many_cell_hamiltonian_conditioned_head_log_amplitudes(
    heads: dict[str, jax.Array],
    design: dict[str, jax.Array],
) -> jax.Array:
    """Evaluate cached exact amplitudes for active probability/phase heads."""

    logits = design["base_logits"] + jnp.einsum(
        "bdach,ch->bda",
        design["probability_features"],
        heads["probability_head"],
    )
    log_probabilities = jax.nn.log_softmax(
        jnp.where(design["masks"], logits, -jnp.inf), axis=-1
    )
    selected_log_p = jnp.take_along_axis(
        log_probabilities, design["actions"][..., None], axis=-1
    )[..., 0]
    phase = design["base_phase"] + jnp.einsum(
        "bch,ch->b", design["phase_features"], heads["phase_head"]
    )
    result = 0.5 * jnp.sum(selected_log_p, axis=1) + 1.0j * phase
    return jnp.where(design["valid"], result, -jnp.inf + 0.0j)


def sample_many_cell_hamiltonian_conditioned_vit(
    params: dict[str, object],
    key: jax.Array,
    sample_count: int,
    config: ManyCellHamiltonianConditionedViTConfig,
) -> jax.Array:
    """Sample exact valid Schur labels from the conditioned autoregression."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    count = int(sample_count)
    base = params["base"]
    local = params["local"]
    labels = jnp.zeros((count, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.base.target_q)
    feature_table = jnp.asarray(
        _candidate_features(config.base), dtype=jnp.float64
    )
    chosen_features = jnp.zeros(
        (count, config.base.decision_count, feature_table.shape[-1]),
        dtype=jnp.float64,
    )
    left, right, valid = (
        jnp.asarray(value) for value in split_tables(config.base.layout.tree)
    )
    batch_indices = jnp.arange(count)
    for decision, subkey in zip(
        tree_decisions(config.base.layout.tree),
        jax.random.split(key, config.base.decision_count),
        strict=True,
    ):
        tokens = _context_tokens(base, chosen_features, config.base)
        parent = labels[:, decision.node_index]
        candidate_left = left[decision.order, parent]
        candidate_right = right[decision.order, parent]
        mask = valid[decision.order, parent]
        candidate_features = feature_table[decision.order, parent]
        logits = _candidate_scores(
            base,
            tokens[:, decision.order : decision.order + 1],
            candidate_features[:, None, :, :],
            config.base,
        )[:, 0]
        local_features = _local_candidate_features(
            local,
            labels,
            tokens[:, decision.order],
            decision,
            candidate_left,
            candidate_right,
            config,
        )
        logits += jnp.einsum(
            "bach,ch->ba", local_features, local["probability_head"]
        )
        logits = jnp.where(mask, logits, -jnp.inf)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        selected_left = candidate_left[batch_indices, action]
        selected_right = candidate_right[batch_indices, action]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(selected_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(selected_right)
        chosen_features = chosen_features.at[:, decision.order].set(
            candidate_features[batch_indices, action]
        )
    return labels


__all__ = [
    "ManyCellHamiltonianConditionedViTConfig",
    "condition_many_cell_schur_vit",
    "init_many_cell_hamiltonian_conditioning_head",
    "many_cell_hamiltonian_conditioned_log_amplitudes",
    "many_cell_hamiltonian_conditioned_head_log_amplitudes",
    "many_cell_hamiltonian_conditioned_log_psi",
    "prepare_many_cell_hamiltonian_conditioned_head_design",
    "sample_many_cell_hamiltonian_conditioned_vit",
]
