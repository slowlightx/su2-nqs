"""Exactly normalized TreeGRU on representatives of a finite symmetry quotient."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._core import embed, gru_step, heads
from .compact_tree_gru import CompactTreeGRUConfig, split_tables
from .tree_gru import tree_decisions


@dataclass(frozen=True)
class TreeOrbitSupport:
    """Prefix trie for a fixed list of valid fusion-tree representatives."""

    labels: np.ndarray
    actions: np.ndarray
    prefix_masks: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class OrbitPushforwardSupport:
    """Compatible fusion paths partitioned into character orbits."""

    tree_support: TreeOrbitSupport
    orbit_index: np.ndarray
    character_gauge: np.ndarray
    n_orbits: int


def tree_orbit_support(
    config: CompactTreeGRUConfig,
    representative_labels: np.ndarray,
) -> TreeOrbitSupport:
    """Compile exact completion masks for symmetry-orbit representatives.

    The mask at decision ``d`` contains precisely the actions that finish at
    least one representative sharing the already generated prefix.  Hence the
    restricted autoregressive distribution is normalized without projection
    or rejection sampling.
    """

    labels = np.asarray(representative_labels)
    expected = (config.n - 1,)
    if labels.ndim != 2 or labels.shape[1:] != expected or len(labels) < 1:
        raise ValueError(
            f"representative_labels must have nonempty shape (n, {expected[0]})"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("fusion labels must use doubled integer quantum numbers")
    labels = np.asarray(labels, dtype=np.int32)
    if len({tuple(row) for row in labels}) != len(labels):
        raise ValueError("representative labels must be unique")
    if np.any(labels[:, -1] != config.target_q):
        raise ValueError("every representative must end in target_q")

    table_left, table_right, table_valid = split_tables(config.tree)
    decisions = tree_decisions(config.tree)
    actions = np.empty((len(labels), len(decisions)), dtype=np.int32)
    for decision in decisions:
        for row_index, row in enumerate(labels):
            parent_q = int(row[decision.node_index])
            if not 0 <= parent_q <= config.n:
                raise ValueError("a parent doubled spin lies outside the tree range")
            left_q = 1 if decision.left_index is None else int(row[decision.left_index])
            right_q = 1 if decision.right_index is None else int(row[decision.right_index])
            matches = np.flatnonzero(
                table_valid[decision.order, parent_q]
                & (table_left[decision.order, parent_q] == left_q)
                & (table_right[decision.order, parent_q] == right_q)
            )
            if len(matches) != 1:
                raise ValueError("a representative contains an invalid fusion split")
            actions[row_index, decision.order] = int(matches[0])

    prefix_masks = np.zeros(
        (len(labels), len(decisions), config.action_dim), dtype=bool
    )
    for decision in range(len(decisions)):
        groups: dict[tuple[int, ...], list[int]] = {}
        for row_index, row in enumerate(actions):
            groups.setdefault(tuple(int(value) for value in row[:decision]), []).append(
                row_index
            )
        for members in groups.values():
            allowed = actions[members, decision]
            prefix_masks[np.ix_(members, [decision], np.unique(allowed))] = True
    return TreeOrbitSupport(labels, actions, prefix_masks)


def orbit_pushforward_support(
    config: CompactTreeGRUConfig,
    compatible_labels: np.ndarray,
    orbit_index: np.ndarray,
    character_gauge: np.ndarray,
) -> OrbitPushforwardSupport:
    """Compile a quotient whose probabilities are pushed forward by orbits."""

    tree_support = tree_orbit_support(config, compatible_labels)
    membership = np.asarray(orbit_index)
    gauge = np.asarray(character_gauge, dtype=np.complex128)
    if membership.shape != (tree_support.size,) or not np.issubdtype(
        membership.dtype, np.integer
    ):
        raise ValueError("orbit_index must be one integer per compatible path")
    membership = np.asarray(membership, dtype=np.int32)
    if gauge.shape != (tree_support.size,) or np.any(np.abs(gauge) == 0.0):
        raise ValueError("character_gauge must be nonzero on every compatible path")
    unique = np.unique(membership)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("orbit indices must be contiguous from zero")
    gauge = gauge / np.abs(gauge)
    return OrbitPushforwardSupport(
        tree_support=tree_support,
        orbit_index=membership,
        character_gauge=gauge,
        n_orbits=len(unique),
    )


def orbit_quotient_log_psi(
    params: dict[str, object],
    representative_index: jax.Array | int,
    config: CompactTreeGRUConfig,
    support: TreeOrbitSupport,
) -> jax.Array:
    """Return the normalized log coefficient of one quotient-basis state."""

    index = jnp.asarray(representative_index, dtype=jnp.int32)
    safe_index = jnp.clip(index, 0, support.size - 1)
    labels = jnp.asarray(support.labels)[safe_index]
    actions = jnp.asarray(support.actions)[safe_index]
    masks = jnp.asarray(support.prefix_masks)[safe_index]
    hidden = params["h0"]
    previous = jnp.asarray(config.action_dim, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    for decision in tree_decisions(config.tree):
        parent_q = labels[decision.node_index]
        inputs = embed(
            params,
            (
                previous,
                parent_q,
                decision.left_leaves,
                decision.right_leaves,
                decision.depth,
                decision.order,
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        logits = jnp.where(masks[decision.order], logits, -1.0e30)
        action = actions[decision.order]
        log_probability += jax.nn.log_softmax(logits)[action]
        phase += phases[action]
        previous = action
    value = 0.5 * log_probability + 1j * phase
    return jnp.where((index >= 0) & (index < support.size), value, -jnp.inf + 0j)


def orbit_quotient_probabilities(
    params: dict[str, object],
    config: CompactTreeGRUConfig,
    support: TreeOrbitSupport,
) -> jax.Array:
    """Enumerate the exactly normalized Born distribution on the quotient."""

    indices = jnp.arange(support.size, dtype=jnp.int32)
    logs = jax.vmap(
        lambda index: orbit_quotient_log_psi(params, index, config, support)
    )(indices)
    return jnp.exp(2.0 * jnp.real(logs))


def sample_orbit_quotient(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: CompactTreeGRUConfig,
    support: TreeOrbitSupport,
) -> jax.Array:
    """Sample representative indices using the same exact prefix trie."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    actions = jnp.asarray(support.actions)
    candidates = jnp.ones((n_samples, support.size), dtype=bool)
    labels = jnp.zeros((n_samples, config.n - 1), dtype=jnp.int32)
    labels = labels.at[:, -1].set(config.target_q)
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), config.action_dim, dtype=jnp.int32)
    table_left, table_right, _ = (jnp.asarray(value) for value in split_tables(config.tree))
    for decision in tree_decisions(config.tree):
        parent_q = labels[:, decision.node_index]
        inputs = embed(
            params,
            (
                previous,
                parent_q,
                jnp.full((n_samples,), decision.left_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.right_leaves, dtype=jnp.int32),
                jnp.full((n_samples,), decision.depth, dtype=jnp.int32),
                jnp.full((n_samples,), decision.order, dtype=jnp.int32),
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        allowed = jnp.any(
            candidates[:, :, None]
            & jax.nn.one_hot(
                actions[:, decision.order], config.action_dim, dtype=bool
            )[None, :, :],
            axis=1,
        )
        logits = jnp.where(allowed, logits, -1.0e30)
        key, subkey = jax.random.split(key)
        chosen = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
        candidates &= actions[None, :, decision.order] == chosen[:, None]
        left_q = table_left[decision.order, parent_q, chosen]
        right_q = table_right[decision.order, parent_q, chosen]
        if decision.left_index is not None:
            labels = labels.at[:, decision.left_index].set(left_q)
        if decision.right_index is not None:
            labels = labels.at[:, decision.right_index].set(right_q)
        previous = chosen
    return jnp.argmax(candidates, axis=1).astype(jnp.int32)


def orbit_pushforward_state(
    params: dict[str, object],
    config: CompactTreeGRUConfig,
    support: OrbitPushforwardSupport,
) -> jax.Array:
    """Return normalized character-orbit coefficients by Born pushforward.

    The magnitude squared of an orbit coefficient is the total probability of
    its compatible fusion paths.  Its phase is their probability-weighted
    circular mean after removal of the character-basis gauge.  Orbit masses
    partition a normalized restricted autoregressive distribution, so the
    returned quotient state is normalized without a projection denominator.
    """

    indices = jnp.arange(support.tree_support.size, dtype=jnp.int32)
    logs = jax.vmap(
        lambda index: orbit_quotient_log_psi(
            params, index, config, support.tree_support
        )
    )(indices)
    probabilities = jnp.exp(2.0 * jnp.real(logs))
    membership = jnp.asarray(support.orbit_index)
    masses = jnp.zeros((support.n_orbits,), dtype=jnp.float64).at[membership].add(
        probabilities
    )
    aligned = probabilities * jnp.exp(1.0j * jnp.imag(logs)) * jnp.conj(
        jnp.asarray(support.character_gauge)
    )
    pooled = jnp.zeros((support.n_orbits,), dtype=jnp.complex128).at[
        membership
    ].add(aligned)
    unit_phase = jnp.exp(1.0j * jnp.angle(pooled))
    return jnp.sqrt(masses) * unit_phase


def sample_orbit_pushforward(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: CompactTreeGRUConfig,
    support: OrbitPushforwardSupport,
) -> jax.Array:
    """Sample orbit indices from their exact pushed-forward Born weights."""

    path_indices = sample_orbit_quotient(
        params, key, n_samples, config, support.tree_support
    )
    return jnp.asarray(support.orbit_index)[path_indices]


__all__ = [
    "TreeOrbitSupport",
    "OrbitPushforwardSupport",
    "orbit_pushforward_state",
    "orbit_pushforward_support",
    "orbit_quotient_log_psi",
    "orbit_quotient_probabilities",
    "sample_orbit_quotient",
    "sample_orbit_pushforward",
    "tree_orbit_support",
]
