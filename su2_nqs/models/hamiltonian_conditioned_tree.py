"""Autoregressive TreeGRU with bond-causal-cone residual features.

The Hamiltonian residual is deliberately kept inside each conditional
distribution.  Consequently it changes the variational tangent space without
introducing a global partition function or weakening the exact SU(2)
reachability mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from ._core import embed, gru_step, heads
from .compact_tree_gru import (
    CompactTreeGRUConfig,
    init_compact_tree_gru,
    split_tables,
)
from .tree_gru import tree_decisions


@dataclass(frozen=True, order=True)
class HamiltonianBond:
    """One SU(2)-scalar pair term, with an optional shared feature channel.

    ``first`` and ``second`` are physical leaf indices.  ``coupling`` is the
    coefficient multiplying the pair term and ``channel`` can distinguish,
    for example, square/triangular ``J1`` and ``J2`` shells while retaining
    parameter sharing within a shell.
    """

    first: int
    second: int
    coupling: float = 1.0
    channel: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.first, bool) or isinstance(self.second, bool):
            raise TypeError("bond endpoints must be integer leaf indices")
        if not isinstance(self.first, int) or not isinstance(self.second, int):
            raise TypeError("bond endpoints must be integer leaf indices")
        if self.first == self.second:
            raise ValueError("a Hamiltonian bond must join distinct leaves")
        if not math.isfinite(self.coupling):
            raise ValueError("bond coupling must be finite")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise TypeError("bond channel must be an integer")
        if self.channel < 0:
            raise ValueError("bond channel must be non-negative")


@dataclass(frozen=True)
class HamiltonianConditionedTreeGRUConfig:
    """Configuration for a compact TreeGRU plus local Hamiltonian features."""

    base: CompactTreeGRUConfig
    bonds: tuple[HamiltonianBond, ...]
    n_channels: int = 1
    local_embedding_dim: int = 8
    local_hidden_dim: int = 32
    train_local_encoder: bool = False

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        if self.local_embedding_dim < 1 or self.local_hidden_dim < 1:
            raise ValueError("local embedding and hidden dimensions must be positive")
        for bond in self.bonds:
            if not 0 <= bond.first < self.n or not 0 <= bond.second < self.n:
                raise ValueError("Hamiltonian bond endpoint lies outside the tree")
            if bond.channel >= self.n_channels:
                raise ValueError("Hamiltonian bond channel lies outside n_channels")

    @property
    def tree(self) -> FusionTree:
        return self.base.tree

    @property
    def target_q(self) -> int:
        return self.base.target_q

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    @property
    def hidden_dim(self) -> int:
        return self.base.hidden_dim


def bond_terms(
    bonds: tuple[tuple[int, int] | tuple[int, int, float], ...],
    *,
    channel: int = 0,
    scale: float = 1.0,
) -> tuple[HamiltonianBond, ...]:
    """Convert an existing lattice bond tuple to conditioned-model terms."""

    answer = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = scale
        elif len(bond) == 3:
            first, second, value = bond
            coupling = scale * value
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        answer.append(HamiltonianBond(int(first), int(second), float(coupling), channel))
    return tuple(answer)


def _tree_metadata(tree: FusionTree):
    parents: dict[TreeNode, FusionNode] = {}
    subtree_sizes: dict[TreeNode, int] = {}

    def visit(node: TreeNode) -> int:
        if isinstance(node, Leaf):
            subtree_sizes[node] = 1
            return 1
        parents[node.left] = node
        parents[node.right] = node
        size = visit(node.left) + visit(node.right)
        subtree_sizes[node] = size
        return size

    visit(tree.root)
    indices = {node: index for index, node in enumerate(tree.internal_nodes)}
    reveals = {len(tree.internal_nodes) - 1: -1}
    for decision in tree_decisions(tree):
        for child_index in (decision.left_index, decision.right_index):
            if child_index is not None:
                reveals[child_index] = decision.order
    return parents, subtree_sizes, indices, reveals


def _internal_leaf_path(
    tree: FusionTree,
    first: int,
    second: int,
) -> tuple[tuple[FusionNode, ...], tuple[int, ...]]:
    """Internal nodes on the unique tree path from ``first`` to ``second``.

    Roles are ``0`` on the first leg, ``1`` at the least common ancestor, and
    ``2`` on the second leg.  This path is precisely the label support exposed
    by recoupling a physical pair to siblings.
    """

    parents, _, _, _ = _tree_metadata(tree)

    def ancestors(leaf: Leaf) -> list[FusionNode]:
        result = []
        node: TreeNode = leaf
        while node in parents:
            parent = parents[node]
            result.append(parent)
            node = parent
        return result

    left = ancestors(tree.leaves[first])
    right = ancestors(tree.leaves[second])
    right_positions = {node: position for position, node in enumerate(right)}
    left_lca_position = next(
        position for position, node in enumerate(left) if node in right_positions
    )
    lca = left[left_lca_position]
    right_lca_position = right_positions[lca]
    left_leg = left[:left_lca_position]
    right_leg = tuple(reversed(right[:right_lca_position]))
    nodes = tuple(left_leg) + (lca,) + right_leg
    roles = (0,) * len(left_leg) + (1,) + (2,) * len(right_leg)
    return nodes, roles


@lru_cache(maxsize=None)
def bond_causal_cone_tables(
    tree: FusionTree,
    bonds: tuple[HamiltonianBond, ...],
) -> tuple[np.ndarray, ...]:
    """Padded label paths grouped by their autoregressive reveal decision.

    A bond feature is emitted only at the first conditional for which its full
    path is known.  This no-future-label rule is what keeps ancestral sampling
    exact despite using an explicitly non-chain physical bond graph.
    """

    _, subtree_sizes, indices, reveals = _tree_metadata(tree)
    grouped: list[list[tuple[HamiltonianBond, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]] = [
        [] for _ in range(max(tree.n_leaves - 1, 0))
    ]
    for bond in bonds:
        if bond.coupling == 0.0:
            continue
        nodes, roles = _internal_leaf_path(tree, bond.first, bond.second)
        label_indices = tuple(indices[node] for node in nodes)
        closure = max(reveals[index] for index in label_indices)
        # A path known before the first action is constant in a fixed target-q
        # sector, so it cannot supply a useful variational tangent direction.
        if closure < 0:
            continue
        sizes = tuple(subtree_sizes[node] for node in nodes)
        grouped[closure].append((bond, label_indices, roles, sizes))

    maximum_bonds = max((len(items) for items in grouped), default=0)
    maximum_bonds = max(maximum_bonds, 1)
    maximum_path = max(
        (
            len(item[1])
            for items in grouped
            for item in items
        ),
        default=1,
    )
    shape = (len(grouped), maximum_bonds, maximum_path)
    label_indices = np.zeros(shape, dtype=np.int32)
    path_mask = np.zeros(shape, dtype=bool)
    roles = np.zeros(shape, dtype=np.int32)
    distances = np.zeros(shape, dtype=np.int32)
    sizes = np.zeros(shape, dtype=np.int32)
    channels = np.zeros(shape[:2], dtype=np.int32)
    couplings = np.zeros(shape[:2], dtype=np.float64)
    bond_mask = np.zeros(shape[:2], dtype=bool)
    path_lengths = np.ones(shape[:2], dtype=np.int32)
    for decision, items in enumerate(grouped):
        for slot, (bond, indices_for_bond, roles_for_bond, sizes_for_bond) in enumerate(items):
            length = len(indices_for_bond)
            label_indices[decision, slot, :length] = indices_for_bond
            path_mask[decision, slot, :length] = True
            roles[decision, slot, :length] = roles_for_bond
            distances[decision, slot, :length] = tuple(
                min(position, length - 1 - position) for position in range(length)
            )
            sizes[decision, slot, :length] = sizes_for_bond
            channels[decision, slot] = bond.channel
            couplings[decision, slot] = bond.coupling
            bond_mask[decision, slot] = True
            path_lengths[decision, slot] = length
    return (
        label_indices,
        path_mask,
        roles,
        distances,
        sizes,
        channels,
        couplings,
        bond_mask,
        path_lengths,
    )


def _matrix(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    scale = (2.0 / (shape[0] + shape[1])) ** 0.5
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_hamiltonian_conditioned_tree_gru(
    key: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> dict[str, object]:
    """Initialize at exactly the corresponding unconditioned TreeGRU state.

    The local output heads start at zero.  Thus the initial wavefunction is
    bit-for-bit the base model, while the first tangent space already contains
    channel-resolved functions of every bond causal cone.
    """

    base_key, *local_keys = jax.random.split(key, 12)
    keys = iter(local_keys)
    embedding = config.local_embedding_dim
    hidden = config.local_hidden_dim
    n = config.n
    local = {
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
        "node_projection": _matrix(next(keys), (4 * embedding, hidden)),
        "pair_projection": _matrix(next(keys), (embedding, hidden)),
        "context_projection": _matrix(next(keys), (config.hidden_dim, hidden)),
        "global_projection": _matrix(next(keys), (2 * embedding, hidden)),
        "bias": jnp.zeros((hidden,), dtype=jnp.float64),
        "probability_head": jnp.zeros(
            (config.n_channels, hidden), dtype=jnp.float64
        ),
        "phase_head": jnp.zeros((config.n_channels, hidden), dtype=jnp.float64),
    }
    return {"base": init_compact_tree_gru(base_key, config.base), "local": local}


def condition_existing_tree_gru(
    base_params: dict[str, object],
    key: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> dict[str, object]:
    """Attach a zero-output local block to a trained compact TreeGRU.

    This is the entry point for the sampled-SR experiments: the returned state
    initially has exactly the same amplitudes as ``base_params``, so measured
    teacher-improvement recovery cannot be attributed to reinitialization.
    """

    initialized = init_hamiltonian_conditioned_tree_gru(key, config)
    return {"base": base_params, "local": initialized["local"]}


def _local_candidate_features(
    local: dict[str, jax.Array],
    labels: jax.Array,
    hidden: jax.Array,
    decision,
    candidates_left: jax.Array,
    candidates_right: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> jax.Array:
    """Encode candidates into channel-resolved causal bond features."""

    # Keeping the path encoder fixed during the first projection makes this a
    # well-conditioned linear tangent block.  Once the channel heads have
    # learned a useful operator-aligned direction, a later training stage may
    # opt into adapting the encoder as well.
    def encoder_parameter(name: str) -> jax.Array:
        value = local[name]
        return value if config.train_local_encoder else jax.lax.stop_gradient(value)

    tables = tuple(jnp.asarray(value) for value in bond_causal_cone_tables(config.tree, config.bonds))
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

    candidate_count = candidates_left.shape[0]
    candidate_labels = jnp.broadcast_to(labels, (candidate_count, config.n - 1))
    if decision.left_index is not None:
        candidate_labels = candidate_labels.at[:, decision.left_index].set(candidates_left)
    if decision.right_index is not None:
        candidate_labels = candidate_labels.at[:, decision.right_index].set(candidates_right)
    path_q = jnp.take(candidate_labels, indices, axis=1)
    path_q = jnp.where(path_mask[None, :, :], path_q, 0)

    node_inputs = jnp.concatenate(
        (
            encoder_parameter("q_embedding")[path_q],
            jnp.broadcast_to(
                encoder_parameter("role_embedding")[roles][None, :, :],
                (candidate_count,) + roles.shape + (config.local_embedding_dim,),
            ),
            jnp.broadcast_to(
                encoder_parameter("distance_embedding")[distances][None, :, :],
                (candidate_count,) + distances.shape + (config.local_embedding_dim,),
            ),
            jnp.broadcast_to(
                encoder_parameter("size_embedding")[sizes][None, :, :],
                (candidate_count,) + sizes.shape + (config.local_embedding_dim,),
            ),
        ),
        axis=-1,
    )
    node_features = jnp.tanh(node_inputs @ encoder_parameter("node_projection"))
    node_features = jnp.sum(
        node_features * path_mask[None, :, :, None], axis=2
    ) / jnp.sqrt(lengths[None, :, None])

    pair_mask = path_mask[:, :-1] & path_mask[:, 1:]
    pair_indices = path_q[:, :, :-1] * (config.n + 1) + path_q[:, :, 1:]
    pair_features = (
        encoder_parameter("pair_embedding")[pair_indices]
        @ encoder_parameter("pair_projection")
    )
    pair_denominator = jnp.sqrt(jnp.maximum(lengths - 1, 1))[None, :, None]
    pair_features = jnp.sum(
        pair_features * pair_mask[None, :, :, None], axis=2
    ) / pair_denominator

    global_inputs = jnp.concatenate(
        (
            encoder_parameter("channel_embedding")[channels],
            encoder_parameter("length_embedding")[lengths],
        ),
        axis=-1,
    )
    global_features = global_inputs @ encoder_parameter("global_projection")
    context = hidden @ encoder_parameter("context_projection")
    bond_features = jnp.tanh(
        node_features
        + pair_features
        + global_features[None, :, :]
        + context[None, None, :]
        + encoder_parameter("bias")
    )
    scale = couplings * active_bonds
    channel_features = jnp.zeros(
        (candidate_count, config.n_channels, config.local_hidden_dim),
        dtype=jnp.float64,
    )
    channel_features = channel_features.at[:, channels, :].add(
        bond_features * scale[None, :, None]
    )
    return channel_features


def _local_candidate_scores(
    local: dict[str, jax.Array],
    labels: jax.Array,
    hidden: jax.Array,
    decision,
    candidates_left: jax.Array,
    candidates_right: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Score all split candidates using only labels revealed by this step."""

    features = _local_candidate_features(
        local,
        labels,
        hidden,
        decision,
        candidates_left,
        candidates_right,
        config,
    )
    probability = jnp.einsum(
        "ach,ch->a", features, local["probability_head"]
    )
    phase = jnp.einsum("ach,ch->a", features, local["phase_head"])
    return probability, phase


def fixed_conditioned_path_data(
    params: dict[str, object],
    labels: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Precompute exact fixed-base/fixed-encoder data for channel-head fitting.

    Returns base logits, base phases, validity masks, selected actions, and
    channel-resolved local features for every autoregressive decision.  The
    representation is algebraically identical to
    :func:`hamiltonian_conditioned_tree_log_psi` when only the two local heads
    are trainable.
    """

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},), got {labels.shape}")
    base = params["base"]
    local = params["local"]
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    all_logits, all_phases, all_masks, all_actions, all_features = [], [], [], [], []
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        candidates_left = table_left[decision.order, parent_q]
        candidates_right = table_right[decision.order, parent_q]
        mask = table_valid[decision.order, parent_q]
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
        matches = mask & (candidates_left == desired_left) & (
            candidates_right == desired_right
        )
        action = jnp.argmax(matches).astype(jnp.int32)
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
        logits, phases = heads(base, hidden)
        features = _local_candidate_features(
            local,
            labels,
            hidden,
            decision,
            candidates_left,
            candidates_right,
            config,
        )
        all_logits.append(logits)
        all_phases.append(phases)
        all_masks.append(mask)
        all_actions.append(action)
        all_features.append(features)
        previous = action
    return (
        jnp.stack(all_logits),
        jnp.stack(all_phases),
        jnp.stack(all_masks),
        jnp.stack(all_actions),
        jnp.stack(all_features),
    )


def hamiltonian_conditioned_tree_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> jax.Array:
    """Exactly normalized complex amplitude with local bond-path residuals."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (config.n - 1,):
        raise ValueError(f"expected label shape ({config.n - 1},), got {labels.shape}")
    base = params["base"]
    table_left, table_right, table_valid = (
        jnp.asarray(value) for value in split_tables(config.tree)
    )
    hidden = base["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid_state = labels[-1] == config.target_q
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        parent_in_range = (parent_q >= 0) & (parent_q <= config.n)
        safe_parent = jnp.clip(parent_q, 0, config.n)
        desired_left = jnp.asarray(1, dtype=jnp.int32) if decision.left_index is None else labels[decision.left_index]
        desired_right = jnp.asarray(1, dtype=jnp.int32) if decision.right_index is None else labels[decision.right_index]
        candidates_left = table_left[decision.order, safe_parent]
        candidates_right = table_right[decision.order, safe_parent]
        mask = table_valid[decision.order, safe_parent]
        matches = mask & (candidates_left == desired_left) & (candidates_right == desired_right)
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
        logits, phases = heads(base, hidden)
        local_logits, local_phases = _local_candidate_scores(
            params["local"],
            labels,
            hidden,
            decision,
            candidates_left,
            candidates_right,
            config,
        )
        logits = jnp.where(mask, logits + local_logits, -1.0e30)
        valid_state = valid_state & parent_in_range & jnp.any(matches)
        log_probability += jax.nn.log_softmax(logits)[action]
        phase += phases[action] + local_phases[action]
        previous = action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid_state, value, -jnp.inf + 0j)


def hamiltonian_conditioned_tree_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: HamiltonianConditionedTreeGRUConfig,
) -> jax.Array:
    logs = jax.vmap(
        lambda state: hamiltonian_conditioned_tree_log_psi(params, state, config)
    )(states)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_hamiltonian_conditioned_tree(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: HamiltonianConditionedTreeGRUConfig,
) -> jax.Array:
    """Sample valid fusion labels from the conditioned model."""

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
        local_logits, _ = jax.vmap(
            lambda state, state_hidden, left, right: _local_candidate_scores(
                params["local"],
                state,
                state_hidden,
                decision,
                left,
                right,
                config,
            )
        )(labels, hidden, candidates_left, candidates_right)
        logits = jnp.where(mask, logits + local_logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        chosen_left = jnp.take_along_axis(candidates_left, action[:, None], axis=1)[:, 0]
        chosen_right = jnp.take_along_axis(candidates_right, action[:, None], axis=1)[:, 0]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(chosen_left)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(chosen_right)
        previous = action
    return labels
