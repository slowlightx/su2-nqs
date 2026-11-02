"""Resource-adaptive tree VMC energy-gradient execution.

The mathematical estimator is identical to :mod:`tree_local_energy`.  This
module changes only how a fixed sample batch is represented and executed:

* connectivity rows are assigned to a finite set of static width buckets;
* only one ``devices * microbatch`` chunk is resident on the accelerator;
* accelerator devices split the sample axis and combine *sums* with ``psum``;
* a two-pass estimator uses one global normalization and energy centre.

The existing padded implementation remains the CPU/default reference.  The
``auto`` entry point selects this backend only on an accelerator, while an
explicit ``legacy`` mode is always available for trajectory replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    sample_compact_tree_indexed,
)
from ..operators.tree_path_connectivity import (
    compile_tree_heisenberg_graph_connectivity_plan,
)
from ..su2.fusion_tree import FusionTreeState
from .tree_local_energy import (
    PaddedTreeConnectivityBatch,
    build_padded_tree_connectivity,
    padded_compact_tree_importance_vmc_energy_gradient,
    padded_compact_tree_vmc_energy_gradient,
)


WeightingMode = Literal["uniform", "exhaustive_born", "importance"]
ExecutionMode = Literal["auto", "legacy", "adaptive"]

# A deliberately finite shape registry.  Adding a width is an explicit choice:
# an overflow never truncates Hamiltonian connectivity or silently recompiles a
# new shape.  The small buckets also keep exact reference tests inexpensive.
DEFAULT_TREE_CONNECTIVITY_BUCKETS = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4096,
)


@dataclass(frozen=True)
class TreeConnectivityBucket:
    """A fixed-width connectivity batch plus its original sample positions."""

    sample_indices: np.ndarray
    connectivity: PaddedTreeConnectivityBatch

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices)
        if indices.shape != (self.connectivity.batch_size,):
            raise ValueError("sample_indices must match the bucket batch size")
        if np.any(indices < 0):
            raise ValueError("sample_indices cannot be negative")


@dataclass(frozen=True)
class BucketedTreeConnectivityBatch:
    """Exact connectivity partitioned among registered static widths."""

    center_labels: np.ndarray
    connectivity_widths: np.ndarray
    buckets: tuple[TreeConnectivityBucket, ...]
    registered_widths: tuple[int, ...]
    n_sites: int
    target_q: int
    connectivity_seconds: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_labels)
        widths = np.asarray(self.connectivity_widths)
        if centers.ndim != 2 or centers.shape[1] != self.n_sites - 1:
            raise ValueError("center_labels have the wrong fixed shape")
        if len(centers) < 1 or widths.shape != (len(centers),):
            raise ValueError("connectivity_widths must match a nonempty batch")
        registered = _validate_registered_widths(self.registered_widths)
        if registered != self.registered_widths:
            raise ValueError("registered_widths must be sorted and unique")
        if not self.buckets:
            raise ValueError("at least one nonempty bucket is required")
        observed: list[int] = []
        for bucket in self.buckets:
            if bucket.connectivity.n_sites != self.n_sites:
                raise ValueError("bucket and model have different site counts")
            if bucket.connectivity.target_q != self.target_q:
                raise ValueError("bucket and model have different target sectors")
            if bucket.connectivity.maximum_width not in registered:
                raise ValueError("bucket width is not registered")
            observed.extend(map(int, np.asarray(bucket.sample_indices)))
        if sorted(observed) != list(range(len(centers))):
            raise ValueError("buckets must partition every sample exactly once")

    @property
    def batch_size(self) -> int:
        return int(len(self.center_labels))

    @property
    def total_connectivity_entries(self) -> int:
        return int(np.sum(self.connectivity_widths))

    @property
    def allocated_connectivity_slots(self) -> int:
        return int(
            sum(
                bucket.connectivity.batch_size
                * bucket.connectivity.maximum_width
                for bucket in self.buckets
            )
        )

    @property
    def registered_fixed_width_slots(self) -> int:
        return int(self.batch_size * max(self.registered_widths))

    @property
    def legacy_connectivity_slots(self) -> int:
        """Compatibility alias for the registered fixed-width baseline."""

        return self.registered_fixed_width_slots


@dataclass(frozen=True)
class AdaptiveTreeExecution:
    """Auditable resource plan for one energy-gradient evaluation."""

    requested_mode: ExecutionMode
    selected_mode: Literal["legacy", "adaptive"]
    backend: str
    device_count: int
    device_labels: tuple[str, ...]
    microbatch_size_per_device: int
    chunk_count: int
    real_sample_count: int
    dummy_sample_count: int
    registered_widths: tuple[int, ...]
    active_bucket_widths: tuple[int, ...]
    total_connectivity_entries: int
    allocated_connectivity_slots: int
    registered_fixed_width_slots: int
    maximum_resident_connectivity_slots: int

    @property
    def bucket_fill_fraction(self) -> float:
        return self.total_connectivity_entries / self.allocated_connectivity_slots

    @property
    def slot_reduction(self) -> float:
        """Slot reduction versus the largest *registered* fixed width."""

        return (
            self.registered_fixed_width_slots
            / self.allocated_connectivity_slots
        )

    @property
    def maximum_resident_connectivity_slots_per_device(self) -> int:
        return self.maximum_resident_connectivity_slots // self.device_count


@dataclass(frozen=True)
class AdaptiveTreeVMCGradient:
    """VMC statistics, gradient, and execution diagnostics."""

    energy: float
    variance: float
    enumerated_normalization: float
    gradient: object
    gradient_norm: float
    execution: AdaptiveTreeExecution
    connectivity_seconds: float
    first_pass_compile_and_evaluate_seconds: float
    second_pass_compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        if self.variance < -1.0e-12:
            raise ValueError("variance cannot be negative")
        if self.gradient_norm < 0.0:
            raise ValueError("gradient norm cannot be negative")

    @property
    def compile_and_evaluate_seconds(self) -> float:
        return (
            self.first_pass_compile_and_evaluate_seconds
            + self.second_pass_compile_and_evaluate_seconds
        )


@dataclass(frozen=True)
class ParallelTreeSampleBatch:
    """Globally indexed samples and their sample-axis execution ledger."""

    labels: np.ndarray
    global_indices: np.ndarray
    backend: str
    device_count: int
    device_labels: tuple[str, ...]
    microbatch_size_per_device: int
    chunk_count: int
    dummy_sample_count: int
    compile_and_evaluate_seconds: float

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        indices = np.asarray(self.global_indices)
        if labels.ndim != 2 or indices.shape != (len(labels),):
            raise ValueError("labels and global_indices must share a batch axis")
        if self.device_count < 1 or self.microbatch_size_per_device < 1:
            raise ValueError("invalid parallel sampling resources")
        if self.chunk_count < 1 or self.dummy_sample_count < 0:
            raise ValueError("invalid parallel sampling workload counts")


def _validate_registered_widths(widths: Sequence[int]) -> tuple[int, ...]:
    concrete = tuple(int(width) for width in widths)
    if not concrete or any(width < 1 for width in concrete):
        raise ValueError("registered_widths must contain positive integers")
    if tuple(sorted(set(concrete))) != concrete:
        raise ValueError("registered_widths must be sorted and unique")
    return concrete


def _validate_labels(
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int32)
    expected_width = config.n - 1
    if array.ndim != 2 or array.shape[1] != expected_width or len(array) < 1:
        raise ValueError(
            f"labels must have shape (batch, {expected_width}) with nonzero batch"
        )
    return array


def build_bucketed_tree_connectivity(
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    registered_widths: Sequence[int] = DEFAULT_TREE_CONNECTIVITY_BUCKETS,
) -> BucketedTreeConnectivityBatch:
    """Build exact connectivity using only a finite registry of static widths.

    Each row is evaluated exactly once and assigned to the smallest registered
    width that can contain it.  Padding repeats a valid centre state and has
    zero coefficient/mask, so no invalid fusion labels reach the neural model.
    """

    array = _validate_labels(labels, config)
    widths_registry = _validate_registered_widths(registered_widths)
    rows_by_width: dict[
        int, list[tuple[int, tuple[FusionTreeState, ...], np.ndarray]]
    ] = {width: [] for width in widths_registry}
    exact_widths = np.empty(len(array), dtype=np.int64)
    started = time.perf_counter()
    graph_plan = compile_tree_heisenberg_graph_connectivity_plan(
        config.tree, bonds, target_q=config.target_q
    )
    for sample, raw in enumerate(array):
        state = FusionTreeState(tuple(map(int, raw)))
        connected, coefficients = graph_plan.column(state)
        exact_width = len(connected)
        bucket_width = next(
            (width for width in widths_registry if exact_width <= width), None
        )
        if bucket_width is None:
            raise ValueError(
                "exact connectivity width "
                f"{exact_width} exceeds largest registered width "
                f"{widths_registry[-1]} for sample {sample}"
            )
        exact_widths[sample] = exact_width
        rows_by_width[bucket_width].append(
            (
                sample,
                tuple(connected),
                np.asarray(coefficients, dtype=np.float64),
            )
        )

    buckets: list[TreeConnectivityBucket] = []
    for bucket_width in widths_registry:
        rows = rows_by_width[bucket_width]
        if not rows:
            continue
        sample_indices = np.asarray([row[0] for row in rows], dtype=np.int64)
        centers = array[sample_indices]
        connected_labels = np.repeat(
            centers[:, None, :], bucket_width, axis=1
        )
        coefficients = np.zeros((len(rows), bucket_width), dtype=np.float64)
        mask = np.zeros((len(rows), bucket_width), dtype=bool)
        local_widths = np.empty(len(rows), dtype=np.int64)
        for position, (_, connected, values) in enumerate(rows):
            width = len(connected)
            local_widths[position] = width
            if width:
                connected_labels[position, :width] = np.asarray(
                    [candidate.labels for candidate in connected],
                    dtype=np.int32,
                )
                coefficients[position, :width] = values
                mask[position, :width] = True
        buckets.append(
            TreeConnectivityBucket(
                sample_indices=sample_indices,
                connectivity=PaddedTreeConnectivityBatch(
                    center_labels=centers,
                    connected_labels=connected_labels,
                    coefficients=coefficients,
                    mask=mask,
                    connectivity_widths=local_widths,
                    n_sites=config.n,
                    target_q=config.target_q,
                    connectivity_seconds=0.0,
                ),
            )
        )
    seconds = time.perf_counter() - started
    return BucketedTreeConnectivityBatch(
        center_labels=array,
        connectivity_widths=exact_widths,
        buckets=tuple(buckets),
        registered_widths=widths_registry,
        n_sites=config.n,
        target_q=config.target_q,
        connectivity_seconds=float(seconds),
    )


def _device_local_energy_and_log_psi(
    params,
    center_labels,
    connected_labels,
    coefficients,
    connectivity_mask,
    *,
    config: CompactTreeGRUConfig,
):
    center_logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(center_labels)
    flat_connected = connected_labels.reshape(
        (-1, connected_labels.shape[-1])
    )
    connected_logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(flat_connected).reshape(connected_labels.shape[:2])
    ratios = jnp.exp(connected_logs - center_logs[:, None])
    local = jnp.sum(
        jnp.where(connectivity_mask, coefficients * ratios, 0.0), axis=1
    )
    return local, center_logs


@lru_cache(maxsize=None)
def _parallel_kernel_factory(
    config: CompactTreeGRUConfig,
    devices: tuple[jax.Device, ...],
):
    """Compile-cache pmapped kernels for one model/device topology."""

    def first_pass(
        params,
        center_labels,
        connected_labels,
        coefficients,
        connectivity_mask,
    ):
        return _device_local_energy_and_log_psi(
            params,
            center_labels,
            connected_labels,
            coefficients,
            connectivity_mask,
            config=config,
        )

    def second_pass(
        params,
        center_labels,
        local_energies,
        raw_outer_weights,
        sample_mask,
        weight_sum,
        mean_energy,
    ):
        normalized_weights = jax.lax.stop_gradient(
            jnp.where(sample_mask, raw_outer_weights / weight_sum, 0.0)
        )
        centered = jax.lax.stop_gradient(local_energies - mean_energy)

        def local_surrogate(candidate):
            logs = jax.vmap(
                lambda state: compact_tree_log_psi(candidate, state, config)
            )(center_labels)
            return 2.0 * jnp.real(
                jnp.sum(normalized_weights * jnp.conj(logs) * centered)
            )

        gradient = jax.grad(local_surrogate)(params)
        return jax.tree_util.tree_map(
            lambda leaf: jax.lax.psum(leaf, "tree_sample_devices"),
            gradient,
        )

    first = jax.pmap(
        first_pass,
        axis_name="tree_sample_devices",
        devices=devices,
    )
    second = jax.pmap(
        second_pass,
        axis_name="tree_sample_devices",
        devices=devices,
    )
    return first, second


@dataclass(frozen=True)
class _StreamingChunk:
    sample_indices: np.ndarray
    center_labels: np.ndarray
    local_energies: np.ndarray
    sample_mask: np.ndarray


def _padded_sharded_chunk(
    bucket: TreeConnectivityBucket,
    start: int,
    stop: int,
    *,
    device_count: int,
    microbatch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    connectivity = bucket.connectivity
    capacity = device_count * microbatch_size
    count = stop - start
    if count < 1 or count > capacity:
        raise ValueError("invalid streaming chunk bounds")
    source = slice(start, stop)
    # Every dummy is a copy of a valid state/connectivity row.  The separate
    # outer sample mask removes it from all global sufficient statistics.
    centers = np.repeat(connectivity.center_labels[start : start + 1], capacity, 0)
    connected = np.repeat(
        connectivity.connected_labels[start : start + 1], capacity, 0
    )
    coefficients = np.repeat(
        connectivity.coefficients[start : start + 1], capacity, 0
    )
    connection_mask = np.repeat(
        connectivity.mask[start : start + 1], capacity, 0
    )
    centers[:count] = connectivity.center_labels[source]
    connected[:count] = connectivity.connected_labels[source]
    coefficients[:count] = connectivity.coefficients[source]
    connection_mask[:count] = connectivity.mask[source]
    sample_mask = np.zeros(capacity, dtype=bool)
    sample_mask[:count] = True
    indices = np.asarray(bucket.sample_indices[source], dtype=np.int64)
    leading = (device_count, microbatch_size)
    return (
        centers.reshape((*leading, centers.shape[-1])),
        connected.reshape((*leading, *connected.shape[1:])),
        coefficients.reshape((*leading, coefficients.shape[-1])),
        connection_mask.reshape((*leading, connection_mask.shape[-1])),
        sample_mask.reshape(leading),
        indices,
    )


def _device_labels(devices: Sequence[jax.Device]) -> tuple[str, ...]:
    return tuple(
        f"{device.platform}:{device.process_index}:{device.id}" for device in devices
    )


def _validated_devices(
    devices: Sequence[jax.Device] | None,
) -> tuple[jax.Device, ...]:
    selected = tuple(jax.local_devices() if devices is None else devices)
    if not selected:
        raise ValueError("at least one local JAX device is required")
    if len(set(selected)) != len(selected):
        raise ValueError("selected JAX devices must be unique")
    if len({device.platform for device in selected}) != 1:
        raise ValueError("all devices must use the same JAX platform")
    if len({device.process_index for device in selected}) != 1:
        raise ValueError("this backend is single-host; devices must share a process")
    return selected


def _replicate_pytree_for_pmap(params: object, device_count: int) -> object:
    """Add one identical leading shard per pmap device, across JAX versions."""

    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf), (device_count, *np.shape(leaf))
        ),
        params,
    )


@lru_cache(maxsize=None)
def _parallel_sampler_factory(
    config: CompactTreeGRUConfig,
    devices: tuple[jax.Device, ...],
):
    def sample_shard(params, base_key, global_indices):
        return sample_compact_tree_indexed(
            params, base_key, global_indices, config
        )

    return jax.pmap(sample_shard, devices=devices)


def sample_compact_tree_data_parallel(
    params: Mapping[str, object],
    key: jax.Array,
    n_samples: int,
    config: CompactTreeGRUConfig,
    *,
    global_index_start: int = 0,
    microbatch_size: int = 256,
    devices: Sequence[jax.Device] | None = None,
) -> ParallelTreeSampleBatch:
    """Draw a device-count-invariant sample stream along the sample axis.

    Sample ``i`` uses ``fold_in(key, global_index_start + i)``.  Thus changing
    device count or microbatch partitioning does not change any retained row,
    which makes fixed-seed legacy/adaptive trajectory comparisons possible
    when both arms opt into this indexed sampler.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    if global_index_start < 0:
        raise ValueError("global_index_start cannot be negative")
    if global_index_start + n_samples > 2**32:
        raise ValueError("global sample indices must fit in uint32")
    selected_devices = _validated_devices(devices)
    device_count = len(selected_devices)
    capacity = device_count * microbatch_size
    replicated_params = _replicate_pytree_for_pmap(params, device_count)
    concrete_key = jnp.asarray(key)
    replicated_key = jnp.broadcast_to(
        concrete_key, (device_count, *concrete_key.shape)
    )
    kernel = _parallel_sampler_factory(config, selected_devices)
    labels = np.empty((n_samples, config.n - 1), dtype=np.int32)
    global_indices = np.arange(
        global_index_start,
        global_index_start + n_samples,
        dtype=np.uint32,
    )
    chunks = 0
    dummy_count = 0
    started = time.perf_counter()
    for offset in range(0, n_samples, capacity):
        count = min(capacity, n_samples - offset)
        padded_indices = np.arange(
            global_index_start + offset,
            global_index_start + offset + capacity,
            dtype=np.uint32,
        ).reshape((device_count, microbatch_size))
        sampled = kernel(
            replicated_params, replicated_key, jnp.asarray(padded_indices)
        )
        jax.block_until_ready(sampled)
        labels[offset : offset + count] = np.asarray(sampled).reshape(
            (capacity, config.n - 1)
        )[:count]
        chunks += 1
        dummy_count += capacity - count
    seconds = time.perf_counter() - started
    return ParallelTreeSampleBatch(
        labels=labels,
        global_indices=global_indices,
        backend=selected_devices[0].platform,
        device_count=device_count,
        device_labels=_device_labels(selected_devices),
        microbatch_size_per_device=microbatch_size,
        chunk_count=chunks,
        dummy_sample_count=dummy_count,
        compile_and_evaluate_seconds=float(seconds),
    )


def _gradient_norm(gradient: object) -> float:
    squared = sum(
        jnp.sum(jnp.abs(leaf) ** 2)
        for leaf in jax.tree_util.tree_leaves(gradient)
    )
    return float(jnp.sqrt(squared))


def adaptive_compact_tree_vmc_energy_gradient_from_connectivity(
    params: Mapping[str, object],
    connectivity: BucketedTreeConnectivityBatch,
    config: CompactTreeGRUConfig,
    *,
    weighting: WeightingMode = "uniform",
    importance_weights: Sequence[float] | jax.Array | None = None,
    microbatch_size: int = 8,
    devices: Sequence[jax.Device] | None = None,
    requested_mode: ExecutionMode = "adaptive",
) -> AdaptiveTreeVMCGradient:
    """Run the exact two-pass sample-parallel estimator on fixed connectivity.

    This routine is single-host.  All selected devices receive replicated
    parameters and disjoint sample shards; Hamiltonian/root dimensions are
    never partitioned across devices.
    """

    if connectivity.n_sites != config.n:
        raise ValueError("connectivity and model have different site counts")
    if connectivity.target_q != config.target_q:
        raise ValueError("connectivity and model have different target sectors")
    if weighting not in ("uniform", "exhaustive_born", "importance"):
        raise ValueError(
            "weighting must be uniform, exhaustive_born, or importance"
        )
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    selected_devices = _validated_devices(devices)

    external_weights: np.ndarray | None = None
    if weighting == "importance":
        if importance_weights is None:
            raise ValueError("importance weighting requires importance_weights")
        external_weights = np.asarray(importance_weights, dtype=np.float64)
        if external_weights.shape != (connectivity.batch_size,):
            raise ValueError("importance_weights must match the sample batch")
        if np.any(~np.isfinite(external_weights)) or np.any(
            external_weights <= 0.0
        ):
            raise ValueError("importance_weights must be finite and positive")
    elif importance_weights is not None:
        raise ValueError("importance_weights are valid only in importance mode")

    device_count = len(selected_devices)
    # A leading pmap axis is portable across the supported JAX range.  Newer
    # JAX releases removed ``device_put_replicated``; stacking once here lets
    # pmap place one identical parameter shard on each selected device without
    # relying on that deprecated API.
    replicated_params = _replicate_pytree_for_pmap(params, device_count)
    first_kernel, second_kernel = _parallel_kernel_factory(
        config, selected_devices
    )
    chunks: list[_StreamingChunk] = []
    all_local = np.empty(connectivity.batch_size, dtype=np.complex128)
    all_probabilities = np.empty(connectivity.batch_size, dtype=np.float64)
    chunk_count = 0
    dummy_count = 0
    first_started = time.perf_counter()
    capacity = device_count * microbatch_size
    for bucket in connectivity.buckets:
        for start in range(0, bucket.connectivity.batch_size, capacity):
            stop = min(start + capacity, bucket.connectivity.batch_size)
            (
                centers,
                connected,
                coefficients,
                connection_mask,
                sample_mask,
                indices,
            ) = _padded_sharded_chunk(
                bucket,
                start,
                stop,
                device_count=device_count,
                microbatch_size=microbatch_size,
            )
            local, logs = first_kernel(
                replicated_params,
                jnp.asarray(centers),
                jnp.asarray(connected),
                jnp.asarray(coefficients),
                jnp.asarray(connection_mask),
            )
            jax.block_until_ready((local, logs))
            local_host = np.asarray(local)
            probabilities = np.asarray(
                jnp.exp(2.0 * jnp.real(logs)), dtype=np.float64
            )
            flat_mask = sample_mask.reshape(-1)
            count = len(indices)
            real_local = local_host.reshape(-1)[:count]
            real_probabilities = probabilities.reshape(-1)[:count]
            all_local[indices] = real_local
            all_probabilities[indices] = real_probabilities
            chunks.append(
                _StreamingChunk(
                    sample_indices=indices,
                    center_labels=centers,
                    local_energies=local_host,
                    sample_mask=sample_mask,
                )
            )
            chunk_count += 1
            dummy_count += int(np.sum(~flat_mask))
    first_seconds = time.perf_counter() - first_started

    if weighting == "uniform":
        raw_outer = np.ones(connectivity.batch_size, dtype=np.float64)
    elif weighting == "exhaustive_born":
        raw_outer = all_probabilities.copy()
    else:
        assert external_weights is not None
        raw_outer = external_weights
    weight_sum = float(np.sum(raw_outer, dtype=np.float64))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise FloatingPointError("global sample weight is not finite and positive")
    mean = np.sum(raw_outer * all_local, dtype=np.complex128) / weight_sum
    variance = float(
        np.sum(
            raw_outer * np.abs(all_local - mean) ** 2,
            dtype=np.float64,
        )
        / weight_sum
    )

    accumulated_gradient = None
    second_started = time.perf_counter()
    replicated_weight_sum = jnp.full((device_count,), weight_sum)
    replicated_mean = jnp.full((device_count,), mean)
    for chunk in chunks:
        chunk_raw = np.zeros((device_count, microbatch_size), dtype=np.float64)
        real_count = len(chunk.sample_indices)
        chunk_raw.reshape(-1)[:real_count] = raw_outer[chunk.sample_indices]
        chunk_gradient_replicated = second_kernel(
            replicated_params,
            jnp.asarray(chunk.center_labels),
            jnp.asarray(chunk.local_energies),
            jnp.asarray(chunk_raw),
            jnp.asarray(chunk.sample_mask),
            replicated_weight_sum,
            replicated_mean,
        )
        chunk_gradient = jax.tree_util.tree_map(
            lambda leaf: leaf[0], chunk_gradient_replicated
        )
        if accumulated_gradient is None:
            accumulated_gradient = chunk_gradient
        else:
            accumulated_gradient = jax.tree_util.tree_map(
                jnp.add, accumulated_gradient, chunk_gradient
            )
    assert accumulated_gradient is not None
    jax.block_until_ready(accumulated_gradient)
    second_seconds = time.perf_counter() - second_started

    active_widths = tuple(
        bucket.connectivity.maximum_width for bucket in connectivity.buckets
    )
    maximum_resident = max(active_widths) * capacity
    execution = AdaptiveTreeExecution(
        requested_mode=requested_mode,
        selected_mode="adaptive",
        backend=selected_devices[0].platform,
        device_count=device_count,
        device_labels=_device_labels(selected_devices),
        microbatch_size_per_device=microbatch_size,
        chunk_count=chunk_count,
        real_sample_count=connectivity.batch_size,
        dummy_sample_count=dummy_count,
        registered_widths=connectivity.registered_widths,
        active_bucket_widths=active_widths,
        total_connectivity_entries=connectivity.total_connectivity_entries,
        allocated_connectivity_slots=connectivity.allocated_connectivity_slots,
        registered_fixed_width_slots=(
            connectivity.registered_fixed_width_slots
        ),
        maximum_resident_connectivity_slots=maximum_resident,
    )
    return AdaptiveTreeVMCGradient(
        energy=float(np.real(mean)),
        variance=max(variance, 0.0),
        enumerated_normalization=float(
            np.sum(all_probabilities, dtype=np.float64)
        ),
        gradient=accumulated_gradient,
        gradient_norm=_gradient_norm(accumulated_gradient),
        execution=execution,
        connectivity_seconds=connectivity.connectivity_seconds,
        first_pass_compile_and_evaluate_seconds=float(first_seconds),
        second_pass_compile_and_evaluate_seconds=float(second_seconds),
    )


def compact_tree_vmc_energy_gradient_auto(
    params: Mapping[str, object],
    labels: Sequence[Sequence[int]] | jax.Array,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    weighting: WeightingMode = "uniform",
    importance_weights: Sequence[float] | jax.Array | None = None,
    execution_mode: ExecutionMode = "auto",
    registered_widths: Sequence[int] = DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    microbatch_size: int = 8,
    devices: Sequence[jax.Device] | None = None,
) -> AdaptiveTreeVMCGradient:
    """Select the legacy CPU path or the adaptive accelerator path.

    ``auto`` preserves the established padded implementation on CPU and uses
    adaptive execution on GPU, including a single accelerator.  Set
    ``execution_mode='legacy'`` to replay the old path on any backend, or
    ``'adaptive'`` to exercise/sample-test the parallel path on CPU devices.
    """

    if execution_mode not in ("auto", "legacy", "adaptive"):
        raise ValueError("execution_mode must be auto, legacy, or adaptive")
    if weighting not in ("uniform", "exhaustive_born", "importance"):
        raise ValueError(
            "weighting must be uniform, exhaustive_born, or importance"
        )
    array = _validate_labels(labels, config)
    widths = _validate_registered_widths(registered_widths)
    selected_devices = _validated_devices(devices)
    backend = selected_devices[0].platform
    selected_mode: Literal["legacy", "adaptive"]
    if execution_mode == "auto":
        selected_mode = "adaptive" if backend == "gpu" else "legacy"
    else:
        selected_mode = execution_mode

    if selected_mode == "adaptive":
        connectivity = build_bucketed_tree_connectivity(
            array, config, bonds, widths
        )
        return adaptive_compact_tree_vmc_energy_gradient_from_connectivity(
            params,
            connectivity,
            config,
            weighting=weighting,
            importance_weights=importance_weights,
            microbatch_size=microbatch_size,
            devices=selected_devices,
            requested_mode=execution_mode,
        )

    if weighting == "importance":
        if importance_weights is None:
            raise ValueError("importance weighting requires importance_weights")
    elif importance_weights is not None:
        raise ValueError("importance_weights are valid only in importance mode")
    connectivity_started = time.perf_counter()
    padded = build_padded_tree_connectivity(
        array, config, bonds, maximum_width=widths[-1]
    )
    connectivity_seconds = time.perf_counter() - connectivity_started
    with jax.default_device(selected_devices[0]):
        if weighting == "importance":
            legacy = padded_compact_tree_importance_vmc_energy_gradient(
                params,
                padded,
                config,
                importance_weights,  # type: ignore[arg-type]
            )
        else:
            legacy = padded_compact_tree_vmc_energy_gradient(
                params,
                padded,
                config,
                born_weighted=(weighting == "exhaustive_born"),
            )
    execution = AdaptiveTreeExecution(
        requested_mode=execution_mode,
        selected_mode="legacy",
        backend=backend,
        device_count=1,
        device_labels=(_device_labels((selected_devices[0],))[0],),
        microbatch_size_per_device=len(array),
        chunk_count=1,
        real_sample_count=len(array),
        dummy_sample_count=0,
        registered_widths=widths,
        active_bucket_widths=(widths[-1],),
        total_connectivity_entries=int(np.sum(padded.connectivity_widths)),
        allocated_connectivity_slots=len(array) * widths[-1],
        registered_fixed_width_slots=len(array) * widths[-1],
        maximum_resident_connectivity_slots=len(array) * widths[-1],
    )
    return AdaptiveTreeVMCGradient(
        energy=legacy.energy,
        variance=legacy.variance,
        enumerated_normalization=legacy.enumerated_normalization,
        gradient=legacy.gradient,
        gradient_norm=legacy.gradient_norm,
        execution=execution,
        connectivity_seconds=float(connectivity_seconds),
        first_pass_compile_and_evaluate_seconds=legacy.compile_and_evaluate_seconds,
        second_pass_compile_and_evaluate_seconds=0.0,
    )


__all__ = [
    "AdaptiveTreeExecution",
    "AdaptiveTreeVMCGradient",
    "BucketedTreeConnectivityBatch",
    "DEFAULT_TREE_CONNECTIVITY_BUCKETS",
    "ExecutionMode",
    "ParallelTreeSampleBatch",
    "TreeConnectivityBucket",
    "WeightingMode",
    "adaptive_compact_tree_vmc_energy_gradient_from_connectivity",
    "build_bucketed_tree_connectivity",
    "compact_tree_vmc_energy_gradient_auto",
    "sample_compact_tree_data_parallel",
]
