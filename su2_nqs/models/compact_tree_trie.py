"""Exact shared-prefix evaluation for compact fusion-tree wavefunctions.

The ordinary batched evaluator repeats the complete recurrent history for
every requested tree state.  Hamiltonian-connected states often share long
root-first fusion-decision prefixes.  This module compiles those states into a
levelled trie and evaluates each distinct recurrent transition once.

The trie changes only the execution schedule.  It uses the native compact
TreeGRU masks and conditionals, and therefore represents exactly the same
normalized wavefunction as :func:`compact_tree_log_psi`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._core import embed, gru_step, heads
from .compact_tree_gru import CompactTreeGRUConfig, split_tables
from .tree_gru import tree_decisions


@dataclass(frozen=True)
class CompactTreeTrieLevel:
    """One root-first decision level of a compiled support trie."""

    # One entry per child edge.
    parent_indices: np.ndarray
    actions: np.ndarray
    # One recurrent evaluation per parent prefix, before choosing this action.
    previous_actions: np.ndarray
    parent_q: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(len(self.actions))

    @property
    def parent_evaluation_count(self) -> int:
        return int(len(self.previous_actions))


@dataclass(frozen=True)
class CompactTreeTrie:
    """Levelled trie for a finite ordered multiset of valid tree states.

    ``terminal_indices`` maps every input row, including duplicates, to its
    terminal node on the last level.  ``unique_labels`` retains first-seen
    ordering and is useful when a caller needs to scatter amplitudes back into
    Hamiltonian columns.
    """

    unique_labels: np.ndarray
    input_to_unique: np.ndarray
    terminal_indices: np.ndarray
    levels: tuple[CompactTreeTrieLevel, ...]
    input_count: int
    labels_actions_labels_bijection: bool

    @property
    def unique_count(self) -> int:
        return int(len(self.unique_labels))

    @property
    def edge_count(self) -> int:
        return int(sum(level.edge_count for level in self.levels))

    @property
    def parent_prefix_evaluation_count(self) -> int:
        """GRU/head evaluations after exact parent-prefix sharing."""

        return int(sum(level.parent_evaluation_count for level in self.levels))

    @property
    def direct_transition_count(self) -> int:
        """Transitions used by direct evaluation of every input row."""

        return int(self.input_count * len(self.levels))

    @property
    def unique_direct_transition_count(self) -> int:
        """Transitions after terminal deduplication but before prefix reuse."""

        return int(self.unique_count * len(self.levels))


def _validated_labels(
    labels: np.ndarray, config: CompactTreeGRUConfig
) -> np.ndarray:
    raw = np.asarray(labels)
    expected = config.n - 1
    if raw.ndim != 2 or raw.shape[1:] != (expected,) or len(raw) < 1:
        raise ValueError(
            f"labels must have nonempty shape (count, {expected})"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("tree labels must use doubled integers")
    checked = np.asarray(raw, dtype=np.int32)
    if np.any(checked < 0) or np.any(checked > config.n):
        raise ValueError("a doubled-spin label lies outside the tree range")
    if np.any(checked[:, -1] != config.target_q):
        raise ValueError("every tree state must end in target_q")
    return checked


def _state_actions(
    labels: np.ndarray, config: CompactTreeGRUConfig
) -> np.ndarray:
    table_left, table_right, table_valid = split_tables(config.tree)
    decisions = tree_decisions(config.tree)
    actions = np.empty((len(labels), len(decisions)), dtype=np.int32)
    for decision in decisions:
        for row_index, row in enumerate(labels):
            parent_q = int(row[decision.node_index])
            left_q = (
                1
                if decision.left_index is None
                else int(row[decision.left_index])
            )
            right_q = (
                1
                if decision.right_index is None
                else int(row[decision.right_index])
            )
            matches = np.flatnonzero(
                table_valid[decision.order, parent_q]
                & (table_left[decision.order, parent_q] == left_q)
                & (table_right[decision.order, parent_q] == right_q)
            )
            if len(matches) != 1:
                raise ValueError("labels contain an inadmissible fusion split")
            actions[row_index, decision.order] = int(matches[0])
    return actions


def _labels_from_actions(
    actions: np.ndarray, config: CompactTreeGRUConfig
) -> np.ndarray:
    """Reconstruct postorder doubled-spin labels from root-first actions."""

    words = np.asarray(actions)
    decisions = tree_decisions(config.tree)
    if words.ndim != 2 or words.shape[1] != len(decisions):
        raise ValueError("actions have the wrong root-first decision shape")
    table_left, table_right, table_valid = split_tables(config.tree)
    labels = np.zeros((len(words), config.n - 1), dtype=np.int32)
    labels[:, -1] = config.target_q
    for decision in decisions:
        for row_index, word in enumerate(words):
            parent_q = int(labels[row_index, decision.node_index])
            action = int(word[decision.order])
            if (
                action < 0
                or action >= config.action_dim
                or not table_valid[decision.order, parent_q, action]
            ):
                raise ValueError("actions contain an invalid fusion split")
            if decision.left_index is not None:
                labels[row_index, decision.left_index] = table_left[
                    decision.order, parent_q, action
                ]
            if decision.right_index is not None:
                labels[row_index, decision.right_index] = table_right[
                    decision.order, parent_q, action
                ]
    return labels


def compile_compact_tree_trie(
    labels: np.ndarray, config: CompactTreeGRUConfig
) -> CompactTreeTrie:
    """Compile valid tree labels into their exact decision-prefix trie."""

    checked = _validated_labels(labels, config)
    unique_rows: list[tuple[int, ...]] = []
    row_to_unique: dict[tuple[int, ...], int] = {}
    input_to_unique = np.empty(len(checked), dtype=np.int32)
    for index, row in enumerate(checked):
        key = tuple(int(value) for value in row)
        position = row_to_unique.get(key)
        if position is None:
            position = len(unique_rows)
            row_to_unique[key] = position
            unique_rows.append(key)
        input_to_unique[index] = position
    unique = np.asarray(unique_rows, dtype=np.int32)
    actions = _state_actions(unique, config)
    if len({tuple(map(int, row)) for row in actions}) != len(unique):
        raise RuntimeError("distinct tree states produced the same decision word")
    reconstructed = _labels_from_actions(actions, config)
    if not np.array_equal(reconstructed, unique):
        raise RuntimeError("labels-to-actions-to-labels is not bijective")

    decisions = tree_decisions(config.tree)
    previous_nodes: dict[tuple[int, ...], int] = {(): 0}
    levels: list[CompactTreeTrieLevel] = []
    for depth, decision in enumerate(decisions):
        current_nodes: dict[tuple[int, ...], int] = {}
        parent_indices: list[int] = []
        selected_actions: list[int] = []
        prefix_metadata: dict[tuple[int, ...], tuple[int, int]] = {}
        parent_metadata: list[tuple[int, int] | None] = [
            None
        ] * len(previous_nodes)
        for row, word in zip(unique, actions, strict=True):
            prefix = tuple(int(value) for value in word[: depth + 1])
            parent_prefix = prefix[:-1]
            parent_index = previous_nodes[parent_prefix]
            previous_action = (
                config.action_dim if depth == 0 else int(word[depth - 1])
            )
            parent_q = int(row[decision.node_index])
            action = int(word[depth])
            metadata = (parent_q, action)
            recurrent_metadata = (previous_action, parent_q)
            prior_parent_metadata = parent_metadata[parent_index]
            if prior_parent_metadata is None:
                parent_metadata[parent_index] = recurrent_metadata
            elif prior_parent_metadata != recurrent_metadata:
                raise RuntimeError(
                    "a parent prefix has inconsistent recurrent metadata"
                )
            if prefix in current_nodes:
                if prefix_metadata[prefix] != metadata:
                    raise RuntimeError(
                        "a decision prefix has inconsistent recurrent metadata"
                    )
                continue
            current_nodes[prefix] = len(current_nodes)
            prefix_metadata[prefix] = metadata
            parent_indices.append(parent_index)
            selected_actions.append(action)
        if any(metadata is None for metadata in parent_metadata):
            raise RuntimeError("a compiled parent prefix has no child action")
        previous_actions = [
            int(metadata[0]) for metadata in parent_metadata if metadata is not None
        ]
        parent_labels = [
            int(metadata[1]) for metadata in parent_metadata if metadata is not None
        ]
        levels.append(
            CompactTreeTrieLevel(
                parent_indices=np.asarray(parent_indices, dtype=np.int32),
                actions=np.asarray(selected_actions, dtype=np.int32),
                previous_actions=np.asarray(previous_actions, dtype=np.int32),
                parent_q=np.asarray(parent_labels, dtype=np.int32),
            )
        )
        previous_nodes = current_nodes

    unique_terminal = np.asarray(
        [previous_nodes[tuple(map(int, word))] for word in actions],
        dtype=np.int32,
    )
    terminal_indices = unique_terminal[input_to_unique]
    return CompactTreeTrie(
        unique_labels=unique,
        input_to_unique=input_to_unique,
        terminal_indices=terminal_indices,
        levels=tuple(levels),
        input_count=len(checked),
        labels_actions_labels_bijection=True,
    )


def compact_tree_trie_log_psi(
    params: dict[str, object],
    trie: CompactTreeTrie,
    config: CompactTreeGRUConfig,
):
    """Evaluate trie rows in their original order, including duplicates."""

    if len(trie.levels) != config.n - 1:
        raise ValueError("trie and model have different decision counts")
    _, _, table_valid = split_tables(config.tree)
    previous_hidden = jnp.asarray(params["h0"])[None, :]
    previous_logs = jnp.zeros((1,), dtype=jnp.complex128)
    for decision, level in zip(
        tree_decisions(config.tree), trie.levels, strict=True
    ):
        parent_indices = jnp.asarray(level.parent_indices)
        previous_actions = jnp.asarray(level.previous_actions)
        parent_q = jnp.asarray(level.parent_q)
        action = jnp.asarray(level.actions)
        inputs = embed(
            params,
            (
                previous_actions,
                parent_q,
                jnp.full_like(parent_q, decision.left_leaves),
                jnp.full_like(parent_q, decision.right_leaves),
                jnp.full_like(parent_q, decision.depth),
                jnp.full_like(parent_q, decision.order),
            ),
        )
        hidden = gru_step(params["gru"], previous_hidden, inputs)
        logits, phases = heads(params, hidden)
        mask = jnp.asarray(table_valid)[decision.order, parent_q]
        masked_logits = jnp.where(mask, logits, -1.0e30)
        conditional = (
            0.5
            * jnp.take_along_axis(
                jax.nn.log_softmax(masked_logits)[parent_indices],
                action[:, None],
                axis=1,
            )[:, 0]
            + 1.0j
            * jnp.take_along_axis(
                phases[parent_indices], action[:, None], axis=1
            )[:, 0]
        )
        previous_logs = previous_logs[parent_indices] + conditional
        previous_hidden = hidden[parent_indices]
    return previous_logs[jnp.asarray(trie.terminal_indices)]


__all__ = [
    "CompactTreeTrie",
    "CompactTreeTrieLevel",
    "compact_tree_trie_log_psi",
    "compile_compact_tree_trie",
]
