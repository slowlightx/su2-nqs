"""Minimal three-root sampled-Ritz gate for compact fusion-tree states.

This is an integration gate, not the P9v2 D1 physics engine.  It combines
three independent normalized :class:`CompactTreeGRU` roots in the total-spin
``q=0`` sector, draws an equal-stratum Born-mixture sample, builds exact
bucketed tree connectivity, evaluates ``H psi`` for every root, and applies
the sampled ``S/H/K`` plus horizontal-force algebra.

No spatial carrier or projector is present.  Consequently every result from
this module is explicitly labelled ``unprojected_q0_only`` and cannot certify
``Gamma.A1`` roots, root ownership, or spin-liquid physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
)
from .adaptive_tree import (
    DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    BucketedTreeConnectivityBatch,
    build_bucketed_tree_connectivity,
    sample_compact_tree_data_parallel,
)
from .sampled_multistate_force import (
    SampledKyFanForceReference,
    SampledKyFanHorizontalForce,
    prepare_sampled_ky_fan_force_reference,
    sampled_ky_fan_horizontal_force,
)


D1A_TREE_GATE_SCIENTIFIC_SCOPE = (
    "unprojected_q0_only; no Gamma.A1 carrier; infrastructure evidence only; "
    "not formal D1 and not root ownership"
)
D1A_TREE_GATE_PROPOSAL = "equal_stratum_uniform_three_root_Born_mixture"
D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL = False


@dataclass(frozen=True)
class StratifiedThreeRootTreeSamples:
    """Equal-count samples and the per-root counters needed for resumption."""

    labels: np.ndarray
    sampled_root_ids: np.ndarray
    root_global_indices: np.ndarray
    starting_root_sample_indices: tuple[int, int, int]
    next_root_sample_indices: tuple[int, int, int]
    samples_per_root: int
    sampling_seconds: float
    device_label: str

    @property
    def sample_count(self) -> int:
        return int(len(self.labels))


@dataclass(frozen=True)
class ThreeRootTreeExecution:
    """Measured work and conservative static-input memory ledger."""

    device_label: str
    n_sites: int
    root_count: int
    active_bucket_widths: tuple[int, ...]
    total_connectivity_entries: int
    allocated_connectivity_slots: int
    maximum_resident_connectivity_slots: int
    padded_state_amplitude_evaluations: int
    host_connectivity_bytes: int
    maximum_static_device_input_bytes_excluding_xla_workspace: int
    sampling_seconds: float
    connectivity_seconds: float
    action_compile_and_evaluate_seconds: float
    force_compile_and_evaluate_seconds: float


@dataclass(frozen=True)
class ThreeRootTreeRitzForceGate:
    """One sampled three-root forward/Ritz/residual/horizontal-force gate."""

    scientific_scope: str
    proposal_kind: str
    samples: StratifiedThreeRootTreeSamples
    proposal_probabilities: np.ndarray
    root_importance_ratios: np.ndarray
    root_ratio_kish_ess_fractions: np.ndarray
    maximum_root_importance_ratio: float
    center_log_amplitudes: np.ndarray
    hamiltonian_actions: np.ndarray
    reference: SampledKyFanForceReference
    force: SampledKyFanHorizontalForce
    execution: ThreeRootTreeExecution


@partial(jax.jit, static_argnames=("config",))
def _padded_root_log_and_action(
    params: Mapping[str, object],
    center_labels: jax.Array,
    connected_labels: jax.Array,
    coefficients: jax.Array,
    mask: jax.Array,
    config: CompactTreeGRUConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return center ``log(psi)`` and ``H psi`` for one padded workload."""

    center_logs = jax.vmap(
        lambda labels: compact_tree_log_psi(params, labels, config)
    )(center_labels)
    flat_connected = connected_labels.reshape(
        (-1, connected_labels.shape[-1])
    )
    connected_logs = jax.vmap(
        lambda labels: compact_tree_log_psi(params, labels, config)
    )(flat_connected).reshape(connected_labels.shape[:2])
    actions = jnp.sum(
        jnp.where(mask, coefficients * jnp.exp(connected_logs), 0.0), axis=1
    )
    return center_logs, actions


def _device_label(device: jax.Device) -> str:
    return f"{device.platform}:{device.process_index}:{device.id}:{device.device_kind}"


def _validated_single_device(
    devices: Sequence[jax.Device] | None,
) -> jax.Device:
    selected = tuple(jax.local_devices() if devices is None else devices)
    if len(selected) != 1:
        raise ValueError(
            "the preliminary three-root gate requires exactly one local device"
        )
    return selected[0]


def sample_stratified_three_root_compact_tree(
    params: Sequence[Mapping[str, object]],
    key: jax.Array,
    samples_per_root: int,
    config: CompactTreeGRUConfig,
    *,
    starting_root_sample_indices: Sequence[int] = (0, 0, 0),
    sampling_microbatch_size: int = 256,
    devices: Sequence[jax.Device] | None = None,
) -> StratifiedThreeRootTreeSamples:
    """Draw equal strata from three roots with independent indexed streams.

    Root ``a`` uses ``fold_in(key, a)`` and its own monotonically increasing
    global index.  A checkpoint can therefore store the fixed base key and
    ``next_root_sample_indices`` and reproduce the next unused sample on
    restart, independently of invocation boundaries.
    """

    roots = tuple(params)
    if len(roots) != 3:
        raise ValueError("the D1a integration gate requires exactly three roots")
    if samples_per_root < 1:
        raise ValueError("samples_per_root must be positive")
    starts = tuple(int(value) for value in starting_root_sample_indices)
    if len(starts) != 3 or any(value < 0 for value in starts):
        raise ValueError(
            "starting_root_sample_indices must contain three nonnegative integers"
        )
    device = _validated_single_device(devices)
    labels = []
    root_ids = []
    global_indices = []
    sampling_seconds = 0.0
    for root, root_params in enumerate(roots):
        sampled = sample_compact_tree_data_parallel(
            root_params,
            jax.random.fold_in(key, root),
            samples_per_root,
            config,
            global_index_start=starts[root],
            microbatch_size=sampling_microbatch_size,
            devices=(device,),
        )
        labels.append(np.asarray(sampled.labels, dtype=np.int32))
        root_ids.append(np.full(samples_per_root, root, dtype=np.int32))
        global_indices.append(
            np.asarray(sampled.global_indices, dtype=np.uint32)
        )
        sampling_seconds += sampled.compile_and_evaluate_seconds
    return StratifiedThreeRootTreeSamples(
        labels=np.concatenate(labels, axis=0),
        sampled_root_ids=np.concatenate(root_ids),
        root_global_indices=np.concatenate(global_indices),
        starting_root_sample_indices=starts,
        next_root_sample_indices=tuple(
            value + samples_per_root for value in starts
        ),
        samples_per_root=int(samples_per_root),
        sampling_seconds=float(sampling_seconds),
        device_label=_device_label(device),
    )


def _padded_chunk(
    connectivity,
    start: int,
    stop: int,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = stop - start
    if count < 1 or count > capacity:
        raise ValueError("invalid preliminary tree-action chunk")
    centers = np.repeat(connectivity.center_labels[start : start + 1], capacity, 0)
    connected = np.repeat(
        connectivity.connected_labels[start : start + 1], capacity, 0
    )
    coefficients = np.repeat(
        connectivity.coefficients[start : start + 1], capacity, 0
    )
    mask = np.repeat(connectivity.mask[start : start + 1], capacity, 0)
    centers[:count] = connectivity.center_labels[start:stop]
    connected[:count] = connectivity.connected_labels[start:stop]
    coefficients[:count] = connectivity.coefficients[start:stop]
    mask[:count] = connectivity.mask[start:stop]
    return centers, connected, coefficients, mask


def _three_root_tree_actions(
    params: tuple[Mapping[str, object], ...],
    connectivity: BucketedTreeConnectivityBatch,
    config: CompactTreeGRUConfig,
    *,
    microbatch_size: int,
    device: jax.Device,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    if microbatch_size < 1:
        raise ValueError("action microbatch_size must be positive")
    logs = np.empty((connectivity.batch_size, 3), dtype=np.complex128)
    actions = np.empty_like(logs)
    padded_evaluations = 0
    maximum_input_bytes = 0
    started = time.perf_counter()
    with jax.default_device(device):
        for bucket in connectivity.buckets:
            local = bucket.connectivity
            width = local.maximum_width
            for start in range(0, local.batch_size, microbatch_size):
                stop = min(start + microbatch_size, local.batch_size)
                count = stop - start
                centers, connected, coefficients, mask = _padded_chunk(
                    local, start, stop, microbatch_size
                )
                original_indices = np.asarray(
                    bucket.sample_indices[start:stop], dtype=np.int64
                )
                maximum_input_bytes = max(
                    maximum_input_bytes,
                    int(
                        centers.nbytes
                        + connected.nbytes
                        + coefficients.nbytes
                        + mask.nbytes
                    ),
                )
                padded_evaluations += 3 * microbatch_size * (1 + width)
                for root, root_params in enumerate(params):
                    root_logs, root_actions = _padded_root_log_and_action(
                        root_params,
                        jnp.asarray(centers),
                        jnp.asarray(connected),
                        jnp.asarray(coefficients),
                        jnp.asarray(mask),
                        config,
                    )
                    jax.block_until_ready((root_logs, root_actions))
                    logs[original_indices, root] = np.asarray(root_logs)[:count]
                    actions[original_indices, root] = np.asarray(root_actions)[:count]
    seconds = time.perf_counter() - started
    if np.any(~np.isfinite(logs)) or np.any(~np.isfinite(actions)):
        raise FloatingPointError("three-root tree action produced non-finite values")
    return logs, actions, float(seconds), padded_evaluations, maximum_input_bytes


def _uniform_root_mixture(logs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_probabilities = 2.0 * np.real(logs)
    maximum = np.max(log_probabilities, axis=1)
    proposal = np.exp(maximum) * np.mean(
        np.exp(log_probabilities - maximum[:, None]), axis=1
    )
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise FloatingPointError("shared root-mixture proposal is not finite and positive")
    ratios = np.exp(log_probabilities) / proposal[:, None]
    maximum_ratio = float(np.max(ratios))
    if maximum_ratio > 3.0 * (1.0 + 1.0e-10):
        raise FloatingPointError("uniform three-root importance ratio exceeds three")
    return proposal, ratios


def _kish_ess_fractions(ratios: np.ndarray) -> np.ndarray:
    count = ratios.shape[0]
    numerator = np.sum(ratios, axis=0) ** 2
    denominator = count * np.sum(ratios**2, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _host_connectivity_bytes(connectivity: BucketedTreeConnectivityBatch) -> int:
    total = int(connectivity.center_labels.nbytes + connectivity.connectivity_widths.nbytes)
    for bucket in connectivity.buckets:
        local = bucket.connectivity
        total += int(
            bucket.sample_indices.nbytes
            + local.center_labels.nbytes
            + local.connected_labels.nbytes
            + local.coefficients.nbytes
            + local.mask.nbytes
            + local.connectivity_widths.nbytes
        )
    return total


def compact_tree_three_root_ritz_force_gate(
    params: Sequence[Mapping[str, object]],
    key: jax.Array,
    samples_per_root: int,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    starting_root_sample_indices: Sequence[int] = (0, 0, 0),
    registered_widths: Sequence[int] = DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    sampling_microbatch_size: int = 256,
    action_microbatch_size: int = 8,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float | None = None,
    exact_normalized_overlap_diagonal: bool = (
        D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL
    ),
    devices: Sequence[jax.Device] | None = None,
) -> ThreeRootTreeRitzForceGate:
    """Run one preemptible-compatible three-root ``q=0`` integration gate.

    The three Born strata have equal size, hence every retained row has
    observation weight ``1 / (3 * samples_per_root)`` under the uniform root
    mixture.  This preliminary implementation deliberately uses one device
    and evaluates the three Hamiltonian actions sequentially, bounding peak
    padded-connectivity storage without claiming multi-GPU scaling.

    The force path uses the historical fully sampled overlap by default.  The
    exact ``S_aa=1`` control variate is valid for standalone Ritz estimates,
    but is intentionally rejected here because the corresponding corrected
    force has not passed independent-root phase-gauge validation.
    """

    if config.target_q != 0:
        raise ValueError("the D1a integration gate is restricted to target_q=0")
    if exact_normalized_overlap_diagonal:
        raise ValueError(
            "exact normalized overlap diagonals are Ritz-only and are not "
            "available in the D1a Ky-Fan force gate"
        )
    roots = tuple(params)
    if len(roots) != 3:
        raise ValueError("the D1a integration gate requires exactly three roots")
    device = _validated_single_device(devices)
    samples = sample_stratified_three_root_compact_tree(
        roots,
        key,
        samples_per_root,
        config,
        starting_root_sample_indices=starting_root_sample_indices,
        sampling_microbatch_size=sampling_microbatch_size,
        devices=(device,),
    )
    connectivity = build_bucketed_tree_connectivity(
        samples.labels, config, bonds, registered_widths
    )
    (
        logs,
        actions,
        action_seconds,
        padded_evaluations,
        maximum_input_bytes,
    ) = _three_root_tree_actions(
        roots,
        connectivity,
        config,
        microbatch_size=action_microbatch_size,
        device=device,
    )
    proposal, ratios = _uniform_root_mixture(logs)
    reference = prepare_sampled_ky_fan_force_reference(
        logs,
        actions,
        proposal,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
        exact_normalized_overlap_diagonal=(
            exact_normalized_overlap_diagonal
        ),
    )
    labels_on_device = jnp.asarray(samples.labels)

    def log_matrix(candidate: object) -> jax.Array:
        candidate_roots = tuple(candidate)
        return jnp.stack(
            [
                jax.vmap(
                    lambda labels: compact_tree_log_psi(
                        root_params, labels, config
                    )
                )(labels_on_device)
                for root_params in candidate_roots
            ],
            axis=1,
        )

    force_started = time.perf_counter()
    with jax.default_device(device):
        force = sampled_ky_fan_horizontal_force(roots, reference, log_matrix)
    force_seconds = time.perf_counter() - force_started
    active_widths = tuple(
        bucket.connectivity.maximum_width for bucket in connectivity.buckets
    )
    if not active_widths:
        raise RuntimeError("tree connectivity unexpectedly has no active bucket")
    if not math.isfinite(force.gradient_norm):
        raise FloatingPointError("three-root force norm is non-finite")
    execution = ThreeRootTreeExecution(
        device_label=_device_label(device),
        n_sites=config.n,
        root_count=3,
        active_bucket_widths=active_widths,
        total_connectivity_entries=connectivity.total_connectivity_entries,
        allocated_connectivity_slots=connectivity.allocated_connectivity_slots,
        maximum_resident_connectivity_slots=(
            action_microbatch_size * max(active_widths)
        ),
        padded_state_amplitude_evaluations=int(padded_evaluations),
        host_connectivity_bytes=_host_connectivity_bytes(connectivity),
        maximum_static_device_input_bytes_excluding_xla_workspace=int(
            maximum_input_bytes
        ),
        sampling_seconds=samples.sampling_seconds,
        connectivity_seconds=connectivity.connectivity_seconds,
        action_compile_and_evaluate_seconds=float(action_seconds),
        force_compile_and_evaluate_seconds=float(force_seconds),
    )
    return ThreeRootTreeRitzForceGate(
        scientific_scope=D1A_TREE_GATE_SCIENTIFIC_SCOPE,
        proposal_kind=D1A_TREE_GATE_PROPOSAL,
        samples=samples,
        proposal_probabilities=np.asarray(proposal, dtype=np.float64),
        root_importance_ratios=np.asarray(ratios, dtype=np.float64),
        root_ratio_kish_ess_fractions=np.asarray(
            _kish_ess_fractions(ratios), dtype=np.float64
        ),
        maximum_root_importance_ratio=float(np.max(ratios)),
        center_log_amplitudes=np.asarray(logs, dtype=np.complex128),
        hamiltonian_actions=np.asarray(actions, dtype=np.complex128),
        reference=reference,
        force=force,
        execution=execution,
    )


__all__ = [
    "D1A_TREE_GATE_PROPOSAL",
    "D1A_TREE_GATE_SCIENTIFIC_SCOPE",
    "D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL",
    "StratifiedThreeRootTreeSamples",
    "ThreeRootTreeExecution",
    "ThreeRootTreeRitzForceGate",
    "compact_tree_three_root_ritz_force_gate",
    "sample_stratified_three_root_compact_tree",
]
