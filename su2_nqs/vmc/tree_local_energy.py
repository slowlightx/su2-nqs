"""Batched local energies for normalized compact fusion-tree models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.compact_tree_gru import CompactTreeGRUConfig, compact_tree_log_psi
from ..models.compact_tree_trie import (
    compact_tree_trie_log_psi,
    compile_compact_tree_trie,
)
from ..operators.tree_path_connectivity import (
    compile_tree_heisenberg_graph_connectivity_plan,
)
from ..su2.fusion_tree import FusionTreeState


@partial(jax.jit, static_argnames=("config",))
def _batched_compact_tree_log_psi(params, labels, config):
    return jax.vmap(lambda state: compact_tree_log_psi(params, state, config))(
        labels
    )


@partial(jax.jit, static_argnames=("config",))
def _padded_compact_tree_local_energies(
    params, center_labels, connected_labels, coefficients, mask, config
):
    """Evaluate a fixed-shape padded local-energy workload on device."""

    center_logs = _batched_compact_tree_log_psi(params, center_labels, config)
    flat_connected = connected_labels.reshape(
        (-1, connected_labels.shape[-1])
    )
    connected_logs = _batched_compact_tree_log_psi(
        params, flat_connected, config
    ).reshape(connected_labels.shape[:2])
    ratios = jnp.exp(connected_logs - center_logs[:, None])
    return jnp.sum(jnp.where(mask, coefficients * ratios, 0.0), axis=1)


@partial(jax.jit, static_argnames=("config", "born_weighted"))
def _padded_compact_tree_vmc_energy_gradient(
    params,
    center_labels,
    connected_labels,
    coefficients,
    mask,
    config,
    born_weighted,
):
    """Return a VMC log-derivative gradient on one fixed workload."""

    local = _padded_compact_tree_local_energies(
        params,
        center_labels,
        connected_labels,
        coefficients,
        mask,
        config,
    )
    center_logs = _batched_compact_tree_log_psi(
        params, center_labels, config
    )
    raw_probabilities = jnp.exp(2.0 * jnp.real(center_logs))
    normalization = jnp.sum(raw_probabilities)
    if born_weighted:
        weights = raw_probabilities / normalization
    else:
        weights = jnp.ones_like(raw_probabilities) / len(raw_probabilities)
    weights = jax.lax.stop_gradient(weights)
    mean = jnp.sum(weights * local)
    centered = jax.lax.stop_gradient(local - mean)

    def objective(candidate):
        candidate_logs = _batched_compact_tree_log_psi(
            candidate, center_labels, config
        )
        surrogate = 2.0 * jnp.real(
            jnp.sum(weights * jnp.conj(candidate_logs) * centered)
        )
        return (
            jax.lax.stop_gradient(jnp.real(mean))
            + surrogate
            - jax.lax.stop_gradient(surrogate)
        )

    energy, gradient = jax.value_and_grad(objective)(params)
    variance = jnp.real(jnp.sum(weights * jnp.abs(local - mean) ** 2))
    return energy, variance, normalization, gradient


@partial(jax.jit, static_argnames=("config",))
def _padded_compact_tree_importance_vmc_energy_gradient(
    params,
    center_labels,
    connected_labels,
    coefficients,
    mask,
    raw_importance_weights,
    config,
):
    """Return the self-normalized importance-weighted VMC gradient."""

    local = _padded_compact_tree_local_energies(
        params,
        center_labels,
        connected_labels,
        coefficients,
        mask,
        config,
    )
    center_logs = _batched_compact_tree_log_psi(
        params, center_labels, config
    )
    normalization = jnp.sum(jnp.exp(2.0 * jnp.real(center_logs)))
    weights = jax.lax.stop_gradient(
        raw_importance_weights / jnp.sum(raw_importance_weights)
    )
    mean = jnp.sum(weights * local)
    centered = jax.lax.stop_gradient(local - mean)

    def objective(candidate):
        candidate_logs = _batched_compact_tree_log_psi(
            candidate, center_labels, config
        )
        surrogate = 2.0 * jnp.real(
            jnp.sum(weights * jnp.conj(candidate_logs) * centered)
        )
        return (
            jax.lax.stop_gradient(jnp.real(mean))
            + surrogate
            - jax.lax.stop_gradient(surrogate)
        )

    energy, gradient = jax.value_and_grad(objective)(params)
    variance = jnp.real(jnp.sum(weights * jnp.abs(local - mean) ** 2))
    return energy, variance, normalization, gradient


@dataclass(frozen=True)
class TreeLocalEnergyBatch:
    """Host result and workload ledger for one sampled local-energy batch."""

    local_energies: np.ndarray
    connectivity_widths: np.ndarray
    unique_state_count: int
    total_connectivity_entries: int
    connectivity_seconds: float
    amplitude_compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        count = len(self.local_energies)
        if np.asarray(self.local_energies).shape != (count,):
            raise ValueError("local_energies must be a vector")
        if np.asarray(self.connectivity_widths).shape != (count,):
            raise ValueError("connectivity_widths must match the energy batch")
        if self.unique_state_count < 1 or self.total_connectivity_entries < 0:
            raise ValueError("invalid local-energy workload counts")


@dataclass(frozen=True)
class PaddedTreeConnectivityBatch:
    """Fixed-shape host connectivity for repeatable accelerator execution."""

    center_labels: np.ndarray
    connected_labels: np.ndarray
    coefficients: np.ndarray
    mask: np.ndarray
    connectivity_widths: np.ndarray
    n_sites: int
    target_q: int
    connectivity_seconds: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_labels)
        connected = np.asarray(self.connected_labels)
        coefficients = np.asarray(self.coefficients)
        mask = np.asarray(self.mask)
        widths = np.asarray(self.connectivity_widths)
        if centers.ndim != 2 or centers.shape[1] != self.n_sites - 1:
            raise ValueError("center_labels have the wrong fixed shape")
        expected = (centers.shape[0], connected.shape[1])
        if connected.shape != (*expected, self.n_sites - 1):
            raise ValueError("connected_labels have the wrong fixed shape")
        if coefficients.shape != expected or mask.shape != expected:
            raise ValueError("coefficients and mask must match padded states")
        if widths.shape != (centers.shape[0],):
            raise ValueError("connectivity_widths must match the batch")
        if np.any(widths < 0) or np.any(widths > connected.shape[1]):
            raise ValueError("connectivity width exceeds the padded capacity")

    @property
    def batch_size(self) -> int:
        return int(self.center_labels.shape[0])

    @property
    def maximum_width(self) -> int:
        return int(self.connected_labels.shape[1])


@dataclass(frozen=True)
class PaddedTreeLocalEnergyBatch:
    """Local energies and timings from a padded fixed-shape workload."""

    local_energies: np.ndarray
    connectivity_widths: np.ndarray
    maximum_width: int
    total_connectivity_entries: int
    connectivity_seconds: float
    amplitude_compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        count = len(self.local_energies)
        if np.asarray(self.local_energies).shape != (count,):
            raise ValueError("local_energies must be a vector")
        if np.asarray(self.connectivity_widths).shape != (count,):
            raise ValueError("connectivity_widths must match the energy batch")
        if self.maximum_width < 1 or self.total_connectivity_entries < 0:
            raise ValueError("invalid padded local-energy workload counts")


@dataclass(frozen=True)
class PaddedTreeVMCGradient:
    """Energy statistics and a real-parameter log-derivative gradient."""

    energy: float
    variance: float
    enumerated_normalization: float
    gradient: object
    gradient_norm: float
    compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        if self.variance < -1.0e-12:
            raise ValueError("variance cannot be negative")
        if self.gradient_norm < 0.0:
            raise ValueError("gradient norm cannot be negative")


@dataclass(frozen=True)
class TrieTreeLocalEnergyBatch:
    """Exact local energies evaluated on a shared decision-prefix trie."""

    local_energies: np.ndarray
    connectivity_widths: np.ndarray
    unique_state_count: int
    total_connectivity_entries: int
    trie_edge_count: int
    trie_parent_prefix_evaluation_count: int
    direct_transition_count: int
    unique_direct_transition_count: int
    connectivity_seconds: float
    amplitude_compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        count = len(self.local_energies)
        if np.asarray(self.local_energies).shape != (count,):
            raise ValueError("local_energies must be a vector")
        if np.asarray(self.connectivity_widths).shape != (count,):
            raise ValueError("connectivity_widths must match the energy batch")
        if min(
            self.unique_state_count,
            self.total_connectivity_entries,
            self.trie_edge_count,
            self.trie_parent_prefix_evaluation_count,
            self.direct_transition_count,
            self.unique_direct_transition_count,
        ) < 0:
            raise ValueError("trie workload counts cannot be negative")


@dataclass(frozen=True)
class TrieTreeVMCGradient:
    """Born-weighted VMC result with exact trie local energies."""

    energy: float
    variance: float
    enumerated_normalization: float
    gradient: object
    gradient_norm: float
    local_energy: TrieTreeLocalEnergyBatch
    gradient_compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        if self.variance < -1.0e-12:
            raise ValueError("variance cannot be negative")
        if self.gradient_norm < 0.0:
            raise ValueError("gradient norm cannot be negative")


def build_padded_tree_connectivity(
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    maximum_width: int,
) -> PaddedTreeConnectivityBatch:
    """Build exact tree connectivity in a fixed ``(batch, width, N-1)`` array.

    Padding entries repeat their corresponding valid center state, while their
    coefficient and mask are zero.  No invalid fusion state is fabricated or
    repaired.  A too-small registered width fails explicitly instead of
    truncating Hamiltonian connectivity.
    """

    array = np.asarray(labels, dtype=np.int32)
    expected_width = config.n - 1
    if array.ndim != 2 or array.shape[1] != expected_width or len(array) < 1:
        raise ValueError(
            f"labels must have shape (batch, {expected_width}) with nonzero batch"
        )
    if maximum_width < 1:
        raise ValueError("maximum_width must be positive")

    connected_labels = np.repeat(array[:, None, :], maximum_width, axis=1)
    coefficients = np.zeros((len(array), maximum_width), dtype=np.float64)
    mask = np.zeros((len(array), maximum_width), dtype=bool)
    widths = np.empty(len(array), dtype=np.int64)
    started = time.perf_counter()
    graph_plan = compile_tree_heisenberg_graph_connectivity_plan(
        config.tree, bonds, target_q=config.target_q
    )
    for sample, raw in enumerate(array):
        state = FusionTreeState(tuple(map(int, raw)))
        connected, values = graph_plan.column(state)
        width = len(connected)
        if width > maximum_width:
            raise ValueError(
                "exact connectivity width "
                f"{width} exceeds registered maximum_width={maximum_width} "
                f"for sample {sample}"
            )
        widths[sample] = width
        if width:
            connected_labels[sample, :width] = np.asarray(
                [candidate.labels for candidate in connected], dtype=np.int32
            )
            coefficients[sample, :width] = np.asarray(values, dtype=np.float64)
            mask[sample, :width] = True
    seconds = time.perf_counter() - started
    return PaddedTreeConnectivityBatch(
        center_labels=array,
        connected_labels=connected_labels,
        coefficients=coefficients,
        mask=mask,
        connectivity_widths=widths,
        n_sites=config.n,
        target_q=config.target_q,
        connectivity_seconds=float(seconds),
    )


def evaluate_padded_compact_tree_local_energy(
    params: Mapping[str, object],
    connectivity: PaddedTreeConnectivityBatch,
    config: CompactTreeGRUConfig,
) -> PaddedTreeLocalEnergyBatch:
    """Evaluate a prebuilt fixed-shape tree-connectivity workload."""

    if connectivity.n_sites != config.n:
        raise ValueError("connectivity and model have different site counts")
    if connectivity.target_q != config.target_q:
        raise ValueError("connectivity and model have different target sectors")
    started = time.perf_counter()
    local = _padded_compact_tree_local_energies(
        params,
        jnp.asarray(connectivity.center_labels),
        jnp.asarray(connectivity.connected_labels),
        jnp.asarray(connectivity.coefficients),
        jnp.asarray(connectivity.mask),
        config,
    )
    jax.block_until_ready(local)
    seconds = time.perf_counter() - started
    return PaddedTreeLocalEnergyBatch(
        local_energies=np.asarray(local),
        connectivity_widths=np.asarray(connectivity.connectivity_widths),
        maximum_width=connectivity.maximum_width,
        total_connectivity_entries=int(
            np.sum(connectivity.connectivity_widths)
        ),
        connectivity_seconds=connectivity.connectivity_seconds,
        amplitude_compile_and_evaluate_seconds=float(seconds),
    )


def compact_tree_local_energy_padded_batch(
    params: Mapping[str, object],
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    maximum_width: int,
) -> PaddedTreeLocalEnergyBatch:
    """Build and evaluate a complete fixed-shape local-energy batch."""

    connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width
    )
    return evaluate_padded_compact_tree_local_energy(
        params, connectivity, config
    )


def padded_compact_tree_vmc_energy_gradient(
    params: Mapping[str, object],
    connectivity: PaddedTreeConnectivityBatch,
    config: CompactTreeGRUConfig,
    *,
    born_weighted: bool = False,
) -> PaddedTreeVMCGradient:
    """Evaluate energy, variance, and the VMC gradient on a padded batch.

    ``born_weighted=False`` is the usual Monte Carlo estimator for samples
    drawn from the current normalized model.  ``born_weighted=True`` is an
    exact-small diagnostic and is valid only when ``center_labels`` exhausts
    the sector exactly once; its returned normalization makes that invariant
    auditable by the caller.
    """

    if connectivity.n_sites != config.n:
        raise ValueError("connectivity and model have different site counts")
    if connectivity.target_q != config.target_q:
        raise ValueError("connectivity and model have different target sectors")
    started = time.perf_counter()
    energy, variance, normalization, gradient = (
        _padded_compact_tree_vmc_energy_gradient(
            params,
            jnp.asarray(connectivity.center_labels),
            jnp.asarray(connectivity.connected_labels),
            jnp.asarray(connectivity.coefficients),
            jnp.asarray(connectivity.mask),
            config,
            bool(born_weighted),
        )
    )
    jax.block_until_ready((energy, variance, normalization, gradient))
    seconds = time.perf_counter() - started
    squared_norm = sum(
        jnp.sum(jnp.abs(leaf) ** 2)
        for leaf in jax.tree_util.tree_leaves(gradient)
    )
    return PaddedTreeVMCGradient(
        energy=float(energy),
        variance=float(variance),
        enumerated_normalization=float(normalization),
        gradient=gradient,
        gradient_norm=float(jnp.sqrt(squared_norm)),
        compile_and_evaluate_seconds=float(seconds),
    )


def padded_compact_tree_importance_vmc_energy_gradient(
    params: Mapping[str, object],
    connectivity: PaddedTreeConnectivityBatch,
    config: CompactTreeGRUConfig,
    importance_weights: Sequence[float] | jax.Array,
) -> PaddedTreeVMCGradient:
    """Evaluate a self-normalized ``p/q``-weighted VMC gradient.

    The supplied positive weights and the connectivity workload are treated as
    fixed observations.  This preserves the coherent local-energy phase while
    avoiding differentiation through the external proposal construction.
    """

    if connectivity.n_sites != config.n:
        raise ValueError("connectivity and model have different site counts")
    if connectivity.target_q != config.target_q:
        raise ValueError("connectivity and model have different target sectors")
    concrete_weights = np.asarray(importance_weights, dtype=np.float64)
    if concrete_weights.shape != (connectivity.batch_size,):
        raise ValueError("importance_weights must match the padded batch")
    if np.any(~np.isfinite(concrete_weights)) or np.any(concrete_weights <= 0.0):
        raise ValueError("importance_weights must be finite and positive")
    started = time.perf_counter()
    energy, variance, normalization, gradient = (
        _padded_compact_tree_importance_vmc_energy_gradient(
            params,
            jnp.asarray(connectivity.center_labels),
            jnp.asarray(connectivity.connected_labels),
            jnp.asarray(connectivity.coefficients),
            jnp.asarray(connectivity.mask),
            jnp.asarray(concrete_weights),
            config,
        )
    )
    jax.block_until_ready((energy, variance, normalization, gradient))
    seconds = time.perf_counter() - started
    squared_norm = sum(
        jnp.sum(jnp.abs(leaf) ** 2)
        for leaf in jax.tree_util.tree_leaves(gradient)
    )
    return PaddedTreeVMCGradient(
        energy=float(energy),
        variance=float(variance),
        enumerated_normalization=float(normalization),
        gradient=gradient,
        gradient_norm=float(jnp.sqrt(squared_norm)),
        compile_and_evaluate_seconds=float(seconds),
    )


def compact_tree_local_energy_batch(
    params: Mapping[str, object],
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
) -> TreeLocalEnergyBatch:
    """Evaluate ``(H psi)(x)/psi(x)`` for a batch of tree states.

    Connectivity is generated on the host without enumerating the sector.
    All distinct center and connected states are then evaluated in one
    normalized TreeGRU batch.  The tree Hamiltonian is real symmetric, so the
    ``H[new, old]`` columns returned by the connectivity kernel are the rows
    required by the local-energy ratio.
    """

    array = np.asarray(labels, dtype=np.int32)
    expected_width = config.n - 1
    if array.ndim != 2 or array.shape[1] != expected_width or len(array) < 1:
        raise ValueError(
            f"labels must have shape (batch, {expected_width}) with nonzero batch"
        )

    state_index: dict[tuple[int, ...], int] = {}
    ordered_states: list[tuple[int, ...]] = []

    def intern(state_labels: tuple[int, ...]) -> int:
        position = state_index.get(state_labels)
        if position is None:
            position = len(ordered_states)
            state_index[state_labels] = position
            ordered_states.append(state_labels)
        return position

    centers = [intern(tuple(map(int, row))) for row in array]
    connectivity_started = time.perf_counter()
    graph_plan = compile_tree_heisenberg_graph_connectivity_plan(
        config.tree, bonds, target_q=config.target_q
    )
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    widths = []
    for raw in array:
        state = FusionTreeState(tuple(map(int, raw)))
        connected, values = graph_plan.column(state)
        indices = np.asarray(
            [intern(candidate.labels) for candidate in connected], dtype=np.int64
        )
        coefficients = np.asarray(values, dtype=np.float64)
        rows.append((indices, coefficients))
        widths.append(len(indices))
    connectivity_seconds = time.perf_counter() - connectivity_started

    all_labels = jnp.asarray(np.asarray(ordered_states, dtype=np.int32))
    amplitude_started = time.perf_counter()
    all_logs = _batched_compact_tree_log_psi(params, all_labels, config)
    jax.block_until_ready(all_logs)
    amplitude_seconds = time.perf_counter() - amplitude_started
    logs = np.asarray(all_logs)
    local = np.empty(len(rows), dtype=np.complex128)
    for sample, ((indices, coefficients), center) in enumerate(
        zip(rows, centers, strict=True)
    ):
        local[sample] = np.sum(
            coefficients * np.exp(logs[indices] - logs[center])
        )
    return TreeLocalEnergyBatch(
        local_energies=local,
        connectivity_widths=np.asarray(widths, dtype=np.int64),
        unique_state_count=len(ordered_states),
        total_connectivity_entries=int(sum(widths)),
        connectivity_seconds=float(connectivity_seconds),
        amplitude_compile_and_evaluate_seconds=float(amplitude_seconds),
    )


def compact_tree_trie_local_energy_batch(
    params: Mapping[str, object],
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
) -> TrieTreeLocalEnergyBatch:
    """Evaluate exact local energies while sharing recurrent prefixes.

    Hamiltonian columns are first combined coherently by the existing
    tree-native kernel.  The union of center and connected states is then
    evaluated on a decision-prefix trie.  Consequently this routine has no
    additional estimator variance and preserves the exact local energy.
    """

    array = np.asarray(labels, dtype=np.int32)
    expected_width = config.n - 1
    if array.ndim != 2 or array.shape[1] != expected_width or len(array) < 1:
        raise ValueError(
            f"labels must have shape (batch, {expected_width}) with nonzero batch"
        )

    state_index: dict[tuple[int, ...], int] = {}
    ordered_states: list[tuple[int, ...]] = []

    def intern(state_labels: tuple[int, ...]) -> int:
        position = state_index.get(state_labels)
        if position is None:
            position = len(ordered_states)
            state_index[state_labels] = position
            ordered_states.append(state_labels)
        return position

    centers = np.asarray(
        [intern(tuple(map(int, row))) for row in array], dtype=np.int32
    )
    connectivity_started = time.perf_counter()
    graph_plan = compile_tree_heisenberg_graph_connectivity_plan(
        config.tree, bonds, target_q=config.target_q
    )
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    widths: list[int] = []
    for raw in array:
        state = FusionTreeState(tuple(map(int, raw)))
        connected, values = graph_plan.column(state)
        indices = np.asarray(
            [intern(candidate.labels) for candidate in connected], dtype=np.int32
        )
        rows.append((indices, np.asarray(values, dtype=np.float64)))
        widths.append(len(indices))
    connectivity_seconds = time.perf_counter() - connectivity_started

    support = np.asarray(ordered_states, dtype=np.int32)
    trie = compile_compact_tree_trie(support, config)
    amplitude_started = time.perf_counter()
    all_logs = compact_tree_trie_log_psi(params, trie, config)
    jax.block_until_ready(all_logs)
    amplitude_seconds = time.perf_counter() - amplitude_started
    concrete_logs = np.asarray(all_logs)
    local = np.empty(len(rows), dtype=np.complex128)
    for sample, ((indices, coefficients), center) in enumerate(
        zip(rows, centers, strict=True)
    ):
        local[sample] = np.sum(
            coefficients
            * np.exp(concrete_logs[indices] - concrete_logs[int(center)])
        )
    return TrieTreeLocalEnergyBatch(
        local_energies=local,
        connectivity_widths=np.asarray(widths, dtype=np.int64),
        unique_state_count=len(ordered_states),
        total_connectivity_entries=int(sum(widths)),
        trie_edge_count=trie.edge_count,
        trie_parent_prefix_evaluation_count=(
            trie.parent_prefix_evaluation_count
        ),
        direct_transition_count=trie.direct_transition_count,
        unique_direct_transition_count=trie.unique_direct_transition_count,
        connectivity_seconds=float(connectivity_seconds),
        amplitude_compile_and_evaluate_seconds=float(amplitude_seconds),
    )


def compact_tree_trie_vmc_energy_gradient(
    params: Mapping[str, object],
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    born_weighted: bool = False,
) -> TrieTreeVMCGradient:
    """Return the standard VMC gradient using exact trie local energies.

    ``born_weighted=True`` has the same exhaustive-support contract as the
    padded reference implementation: every sector state must occur exactly
    once.  As in the standard score-function estimator, the local energy is
    held fixed while differentiating the centered log-amplitude surrogate.
    """

    array = np.asarray(labels, dtype=np.int32)
    local_batch = compact_tree_trie_local_energy_batch(
        params, array, config, bonds
    )
    fixed_local = jnp.asarray(local_batch.local_energies)
    center_labels = jnp.asarray(array)
    started = time.perf_counter()

    center_logs = _batched_compact_tree_log_psi(params, center_labels, config)
    raw_probabilities = jnp.exp(2.0 * jnp.real(center_logs))
    normalization = jnp.sum(raw_probabilities)
    if born_weighted:
        weights = raw_probabilities / normalization
    else:
        weights = jnp.ones_like(raw_probabilities) / len(array)
    weights = jax.lax.stop_gradient(weights)
    mean = jnp.sum(weights * fixed_local)
    centered = jax.lax.stop_gradient(fixed_local - mean)

    def objective(candidate):
        candidate_logs = _batched_compact_tree_log_psi(
            candidate, center_labels, config
        )
        surrogate = 2.0 * jnp.real(
            jnp.sum(weights * jnp.conj(candidate_logs) * centered)
        )
        return (
            jax.lax.stop_gradient(jnp.real(mean))
            + surrogate
            - jax.lax.stop_gradient(surrogate)
        )

    energy, gradient = jax.value_and_grad(objective)(params)
    variance = jnp.real(jnp.sum(weights * jnp.abs(fixed_local - mean) ** 2))
    jax.block_until_ready((energy, variance, normalization, gradient))
    seconds = time.perf_counter() - started
    squared_norm = sum(
        jnp.sum(jnp.abs(leaf) ** 2)
        for leaf in jax.tree_util.tree_leaves(gradient)
    )
    return TrieTreeVMCGradient(
        energy=float(energy),
        variance=float(variance),
        enumerated_normalization=float(normalization),
        gradient=gradient,
        gradient_norm=float(jnp.sqrt(squared_norm)),
        local_energy=local_batch,
        gradient_compile_and_evaluate_seconds=float(seconds),
    )


__all__ = [
    "PaddedTreeConnectivityBatch",
    "PaddedTreeLocalEnergyBatch",
    "PaddedTreeVMCGradient",
    "TreeLocalEnergyBatch",
    "TrieTreeLocalEnergyBatch",
    "TrieTreeVMCGradient",
    "build_padded_tree_connectivity",
    "compact_tree_local_energy_batch",
    "compact_tree_local_energy_padded_batch",
    "compact_tree_trie_local_energy_batch",
    "compact_tree_trie_vmc_energy_gradient",
    "evaluate_padded_compact_tree_local_energy",
    "padded_compact_tree_importance_vmc_energy_gradient",
    "padded_compact_tree_vmc_energy_gradient",
]
