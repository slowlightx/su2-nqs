"""Reverse-local Hamiltonian-connected proposals for fusion-tree states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    sample_compact_tree,
)
from ..operators.tree_path_connectivity import (
    compile_tree_heisenberg_graph_connectivity_plan,
)
from ..su2.fusion_tree import FusionTree, FusionTreeState


StateLabels = tuple[int, ...]


@partial(jax.jit, static_argnames=("config",))
def _batched_probabilities(params, labels, config):
    logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(labels)
    return jnp.exp(2.0 * jnp.real(logs))


class TreeConnectedTransitionCache:
    """Lazy exact columns of the off-diagonal ``|H|^power`` Markov kernel."""

    def __init__(
        self,
        tree: FusionTree,
        bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
        *,
        target_q: int,
        amplitude_power: float = 2.0,
        zero_tolerance: float = 0.0,
    ) -> None:
        if not np.isfinite(amplitude_power) or amplitude_power <= 0.0:
            raise ValueError("amplitude_power must be positive and finite")
        if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be nonnegative and finite")
        if not tree.target_is_reachable(target_q):
            raise ValueError("target_q is unreachable on this tree")
        self.tree = tree
        self.bonds = tuple(bonds)
        self.target_q = int(target_q)
        self.amplitude_power = float(amplitude_power)
        self.zero_tolerance = float(zero_tolerance)
        self._graph_plan = compile_tree_heisenberg_graph_connectivity_plan(
            tree, self.bonds, target_q=self.target_q
        )
        self._columns: dict[StateLabels, tuple[tuple[StateLabels, ...], np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def cached_state_count(self) -> int:
        return len(self._columns)

    def column(
        self, labels: Sequence[int]
    ) -> tuple[tuple[StateLabels, ...], np.ndarray]:
        """Return off-diagonal targets and normalized probabilities."""

        key = tuple(map(int, labels))
        cached = self._columns.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        state = FusionTreeState(key)
        connected, coefficients = self._graph_plan.column(state)
        targets = []
        weights = []
        for candidate, coefficient in zip(connected, coefficients, strict=True):
            if candidate.labels == key or abs(coefficient) <= self.zero_tolerance:
                continue
            targets.append(candidate.labels)
            weights.append(abs(float(coefficient)) ** self.amplitude_power)
        if targets:
            probabilities = np.asarray(weights, dtype=np.float64)
            total = float(np.sum(probabilities))
            if not np.isfinite(total) or total <= 0.0:
                raise FloatingPointError("connected column normalization is invalid")
            probabilities /= total
            result = (tuple(targets), probabilities)
        else:
            result = ((key,), np.asarray([1.0], dtype=np.float64))
        self._columns[key] = result
        self.misses += 1
        return result


@dataclass(frozen=True)
class TreeConnectedProposalBatch:
    """Samples from ``q=(1-eta)p+eta K_H p`` and exact local ``p/q``."""

    labels: np.ndarray
    parent_labels: np.ndarray
    target_probabilities: np.ndarray
    proposal_probabilities: np.ndarray
    importance_weights: np.ndarray
    moved_fraction: float
    effective_sample_size_fraction: float
    maximum_importance_weight: float
    unique_probability_evaluation_count: int
    reverse_connection_count: int
    cached_state_count: int
    construction_seconds: float

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        parents = np.asarray(self.parent_labels)
        if labels.ndim != 2 or parents.shape != labels.shape or len(labels) < 1:
            raise ValueError("proposal labels and parents must share a nonempty matrix")
        for values in (
            self.target_probabilities,
            self.proposal_probabilities,
            self.importance_weights,
        ):
            if np.asarray(values).shape != (len(labels),):
                raise ValueError("proposal probability vectors must match samples")
        if np.any(np.asarray(self.proposal_probabilities) <= 0.0):
            raise ValueError("sampled proposal probabilities must be positive")
        if np.any(np.asarray(self.importance_weights) <= 0.0):
            raise ValueError("importance weights must be positive")


def _probability_lookup(
    params: Mapping[str, object],
    required: set[StateLabels],
    config: CompactTreeGRUConfig,
) -> dict[StateLabels, float]:
    ordered = tuple(sorted(required))
    probabilities = _batched_probabilities(
        params, jnp.asarray(ordered, dtype=jnp.int32), config
    )
    jax.block_until_ready(probabilities)
    concrete = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(concrete)) or np.any(concrete < 0.0):
        raise FloatingPointError("model probabilities are invalid")
    return {
        labels: float(probability)
        for labels, probability in zip(ordered, concrete, strict=True)
    }


def mixed_tree_connected_probabilities(
    params: Mapping[str, object],
    target_labels: Sequence[Sequence[int]],
    config: CompactTreeGRUConfig,
    cache: TreeConnectedTransitionCache,
    *,
    mixture_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Evaluate local ``q(y)`` and ``p(y)`` from reverse tree connections."""

    if not np.isfinite(mixture_weight) or not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in [0, 1]")
    if cache.tree != config.tree or cache.target_q != config.target_q:
        raise ValueError("transition cache and model config are incompatible")
    targets = tuple(tuple(map(int, row)) for row in target_labels)
    if not targets:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), {
            "unique_target_count": 0,
            "unique_probability_evaluation_count": 0,
            "reverse_connection_count": 0,
        }
    unique_targets = tuple(dict.fromkeys(targets))
    required = set(unique_targets)
    reverse: dict[StateLabels, tuple[tuple[StateLabels, float], ...]] = {}
    reverse_connection_count = 0
    for target in unique_targets:
        neighbors, _ = cache.column(target)
        if len(neighbors) == 1 and neighbors[0] == target:
            terms = ((target, 1.0),)
        else:
            local_terms = []
            for parent in neighbors:
                parent_targets, parent_probabilities = cache.column(parent)
                try:
                    position = parent_targets.index(target)
                except ValueError as error:
                    raise RuntimeError(
                        "tree Hamiltonian connectivity is not reverse symmetric"
                    ) from error
                local_terms.append(
                    (parent, float(parent_probabilities[position]))
                )
            terms = tuple(local_terms)
        reverse[target] = terms
        reverse_connection_count += len(terms)
        required.update(parent for parent, _ in terms)
    probabilities = _probability_lookup(params, required, config)
    proposal_lookup = {}
    for target in unique_targets:
        connected = sum(
            transition_probability * probabilities[parent]
            for parent, transition_probability in reverse[target]
        )
        proposal_lookup[target] = (
            (1.0 - mixture_weight) * probabilities[target]
            + mixture_weight * connected
        )
    proposal = np.asarray(
        [proposal_lookup[target] for target in targets], dtype=np.float64
    )
    base = np.asarray(
        [probabilities[target] for target in targets], dtype=np.float64
    )
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise FloatingPointError("reverse-local proposal probability is invalid")
    return proposal, base, {
        "unique_target_count": len(unique_targets),
        "unique_probability_evaluation_count": len(required),
        "reverse_connection_count": reverse_connection_count,
    }


def sample_mixed_tree_connected(
    params: Mapping[str, object],
    key: jax.Array,
    batch_size: int,
    config: CompactTreeGRUConfig,
    cache: TreeConnectedTransitionCache,
    *,
    mixture_weight: float,
) -> TreeConnectedProposalBatch:
    """Draw exact mixed connected samples and evaluate reverse-local weights."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(mixture_weight) or not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in [0, 1]")
    started = time.perf_counter()
    parent_key, transition_key = jax.random.split(key)
    parents = np.asarray(
        sample_compact_tree(params, parent_key, batch_size, config),
        dtype=np.int32,
    )
    seed_words = np.asarray(transition_key, dtype=np.uint32).reshape(-1)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(word) for word in seed_words])
    )
    targets = parents.copy()
    move = rng.random(batch_size) < mixture_weight
    for sample in np.flatnonzero(move):
        neighbors, probabilities = cache.column(parents[sample])
        choice = int(rng.choice(len(neighbors), p=probabilities))
        targets[sample] = np.asarray(neighbors[choice], dtype=np.int32)
    proposal, base, diagnostics = mixed_tree_connected_probabilities(
        params,
        targets,
        config,
        cache,
        mixture_weight=mixture_weight,
    )
    importance = base / proposal
    normalized = importance / np.sum(importance)
    ess_fraction = float(1.0 / np.sum(normalized**2) / batch_size)
    return TreeConnectedProposalBatch(
        labels=targets,
        parent_labels=parents,
        target_probabilities=base,
        proposal_probabilities=proposal,
        importance_weights=importance,
        moved_fraction=float(np.mean(np.any(targets != parents, axis=1))),
        effective_sample_size_fraction=ess_fraction,
        maximum_importance_weight=float(np.max(importance)),
        unique_probability_evaluation_count=int(
            diagnostics["unique_probability_evaluation_count"]
        ),
        reverse_connection_count=int(diagnostics["reverse_connection_count"]),
        cached_state_count=cache.cached_state_count,
        construction_seconds=float(time.perf_counter() - started),
    )


__all__ = [
    "TreeConnectedProposalBatch",
    "TreeConnectedTransitionCache",
    "mixed_tree_connected_probabilities",
    "sample_mixed_tree_connected",
]
