"""Matrix-free local energies for many-cell SU(2) Schur models.

Connectivity is generated column by column with local reduced-vector
transport and recoupling in :mod:`su2_nqs.operators.tree_path_connectivity`.
No fusion-space Hamiltonian matrix and no computational-basis wavefunction is
formed.  The resulting fixed-shape arrays are evaluated as one JAX kernel on
CPU or accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from numbers import Integral
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    additive_log_amplitudes,
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from ..operators.tree_path_connectivity import (
    compile_tree_heisenberg_graph_connectivity_plan,
)
from ..operators.jax_tree_path_connectivity import (
    JaxTreeConnectivityPlan,
    compile_jax_tree_connectivity_plan,
    jax_coalesce_tree_connectivity,
    jax_tree_bond_connectivity_rectangular,
)
from ..su2.fusion_tree import FusionTreeState


@dataclass(frozen=True)
class ManyCellTreeConnectivity:
    """Padded exact ``H[new,old]`` columns for a batch of tree states."""

    center_labels: np.ndarray
    connected_labels: np.ndarray
    coefficients: np.ndarray
    mask: np.ndarray
    widths: np.ndarray
    physical_bonds: tuple[tuple[int, int, float], ...]
    model_bonds: tuple[tuple[int, int, float], ...]
    connectivity_seconds: float

    def __post_init__(self) -> None:
        centers = np.asarray(self.center_labels)
        connected = np.asarray(self.connected_labels)
        coefficients = np.asarray(self.coefficients)
        mask = np.asarray(self.mask)
        widths = np.asarray(self.widths)
        if centers.ndim != 2 or len(centers) < 1:
            raise ValueError("center_labels must be a nonempty rank-two array")
        if connected.ndim != 3 or connected.shape[0] != len(centers):
            raise ValueError("connected_labels have the wrong shape")
        if connected.shape[2] != centers.shape[1]:
            raise ValueError("center and connected label widths disagree")
        expected = connected.shape[:2]
        if coefficients.shape != expected or mask.shape != expected:
            raise ValueError("coefficients and mask must match connectivity")
        if widths.shape != (len(centers),):
            raise ValueError("one connectivity width is required per center")
        if np.any(widths < 0) or np.any(widths > connected.shape[1]):
            raise ValueError("a connectivity width exceeds the padded capacity")

    @property
    def maximum_width(self) -> int:
        return int(self.connected_labels.shape[1])

    @property
    def total_entries(self) -> int:
        return int(np.sum(self.widths))


@dataclass(frozen=True)
class ManyCellTreeLocalEnergy:
    """Device-evaluated amplitude ratios and local energies."""

    ratios: np.ndarray
    local_energies: np.ndarray
    widths: np.ndarray
    amplitude_seconds: float


@dataclass(frozen=True)
class ManyCellDeviceHamiltonianKernel:
    """Reusable full-graph JAX local-energy kernel with no host row building."""

    plan: JaxTreeConnectivityPlan
    config: ManyCellSchurViTConfig
    candidate_capacity: int | None
    connected_capacity: int
    amplitude_tile_size: int
    bucket_capacities: tuple[tuple[int, int], ...]
    bucket_bond_counts: tuple[int, ...]
    raw_transition_slots_per_sample: int
    compilation_seconds: float
    _kernel: object
    skip_padded_amplitudes: bool = False
    bucket_padded_amplitudes: bool = False
    _connectivity_kernel: object | None = None
    _ratio_kernel: object | None = None


@dataclass(frozen=True)
class ManyCellDeviceLocalEnergy:
    """Device-native local energies and an auditable Hamiltonian-work ledger."""

    local_energies: np.ndarray
    overflow: np.ndarray
    evaluation_seconds: float
    sample_count: int
    bond_count: int
    full_h_term_calls: int
    raw_transition_slots: int
    compact_transition_slots: int
    unique_widths: np.ndarray
    evaluated_connected_amplitudes: int | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.local_energies).shape != (self.sample_count,):
            raise ValueError("local_energies must match sample_count")
        if np.asarray(self.overflow).shape != (self.sample_count,):
            raise ValueError("overflow must match sample_count")
        if np.asarray(self.unique_widths).shape != (self.sample_count,):
            raise ValueError("unique_widths must match sample_count")
        if self.full_h_term_calls != self.sample_count * self.bond_count:
            raise ValueError("full_h_term_calls are inconsistent")


@dataclass(frozen=True)
class ManyCellAdditiveProposal:
    """Mixture-proposal samples for a coherent additive two-model state."""

    labels: jax.Array
    log_proposal_probability: jax.Array
    log_target_amplitude: jax.Array
    log_importance_weight: jax.Array
    carrier_draw: jax.Array


def apply_many_cell_tree_connectivity(
    workload: ManyCellTreeConnectivity,
    coefficients: Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Apply an exact host connectivity table as ``H[new, old]``.

    This is an exhaustive small-system reference helper.  Production VMC
    should use :func:`many_cell_device_hpsi`; the host scatter is useful when
    every reduced-basis coefficient is available and an independently
    auditable target vector is required.
    """

    vector = np.asarray(coefficients, dtype=np.complex128)
    if vector.shape != (len(workload.center_labels),):
        raise ValueError("coefficients must match the connectivity centers")
    positions = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(workload.center_labels)
    }
    if len(positions) != len(workload.center_labels):
        raise ValueError("connectivity centers must be unique")
    action = np.zeros_like(vector)
    for old, amplitude in enumerate(vector):
        for slot in range(int(workload.widths[old])):
            label = tuple(
                int(value)
                for value in workload.connected_labels[old, slot]
            )
            try:
                new = positions[label]
            except KeyError as error:
                raise ValueError(
                    "connectivity leaves the supplied exhaustive center basis"
                ) from error
            action[new] += workload.coefficients[old, slot] * amplitude
    return action


def sample_many_cell_additive_proposal(
    carrier_params: dict[str, object],
    residual_params: dict[str, object],
    residual_scale: complex | float,
    key: jax.Array,
    sample_count: int,
    config: ManyCellSchurViTConfig,
    *,
    carrier_probability: float = 0.5,
) -> ManyCellAdditiveProposal:
    """Sample an exact normalized mixture proposal for ``carrier+residual``.

    Coherent addition generally destroys autoregressive normalization.  The
    two normalized AR arms are therefore mixed only at the *proposal* level;
    the returned self-normalized importance weights target the coherent sum.
    This fills carrier nodes without pretending that the sum itself is an AR
    distribution.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    probability = float(carrier_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("carrier_probability must lie strictly between zero and one")
    arm_key, carrier_key, residual_key = jax.random.split(key, 3)
    carrier_labels = sample_many_cell_schur_vit(
        carrier_params, carrier_key, int(sample_count), config
    )
    residual_labels = sample_many_cell_schur_vit(
        residual_params, residual_key, int(sample_count), config
    )
    carrier_draw = jax.random.bernoulli(
        arm_key, probability, (int(sample_count),)
    )
    labels = jnp.where(carrier_draw[:, None], carrier_labels, residual_labels)
    carrier_logs = many_cell_schur_log_amplitudes(
        carrier_params, labels, config
    )
    residual_logs = many_cell_schur_log_amplitudes(
        residual_params, labels, config
    )
    log_probability = jnp.logaddexp(
        np.log(probability) + 2.0 * jnp.real(carrier_logs),
        np.log1p(-probability) + 2.0 * jnp.real(residual_logs),
    )
    target_logs = additive_log_amplitudes(
        carrier_logs, residual_logs, residual_scale
    )
    return ManyCellAdditiveProposal(
        labels=labels,
        log_proposal_probability=log_probability,
        log_target_amplitude=target_logs,
        log_importance_weight=2.0 * jnp.real(target_logs) - log_probability,
        carrier_draw=carrier_draw,
    )


def _normalize_physical_bonds(
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
) -> tuple[tuple[int, int, float], ...]:
    result = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        result.append((int(first), int(second), float(coupling)))
    return tuple(result)


def build_many_cell_tree_connectivity(
    labels: Sequence[Sequence[int]] | np.ndarray,
    config: ManyCellSchurViTConfig,
    physical_bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    maximum_width: int | None = None,
) -> ManyCellTreeConnectivity:
    """Build exact local-recoupling columns without materializing ``H``.

    If ``maximum_width`` is omitted, a first pass discovers the exact maximum.
    Production runs should preregister a fixed width; a too-small width fails
    rather than truncating any Hamiltonian column.
    """

    centers = np.asarray(labels, dtype=np.int32)
    if centers.ndim != 2 or centers.shape[1] != config.n - 1 or len(centers) < 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    if maximum_width is not None:
        if isinstance(maximum_width, bool) or not isinstance(maximum_width, Integral):
            raise TypeError("maximum_width must be an integer")
        if maximum_width < 1:
            raise ValueError("maximum_width must be positive")
    physical = _normalize_physical_bonds(physical_bonds)
    model = config.layout.remap_bonds(physical)
    plan = compile_tree_heisenberg_graph_connectivity_plan(
        config.layout.tree, model, target_q=config.target_q
    )
    started = time.perf_counter()
    columns = []
    values = []
    widths = np.empty(len(centers), dtype=np.int64)
    discovered = 1
    for sample, raw in enumerate(centers):
        connected, coefficient = plan.column(
            FusionTreeState(tuple(int(value) for value in raw))
        )
        rows = np.asarray([state.labels for state in connected], dtype=np.int32)
        if len(rows) == 0:
            rows = np.empty((0, config.n - 1), dtype=np.int32)
        columns.append(rows)
        values.append(np.asarray(coefficient, dtype=np.float64))
        widths[sample] = len(rows)
        discovered = max(discovered, len(rows))
    width = discovered if maximum_width is None else int(maximum_width)
    if discovered > width:
        offender = int(np.argmax(widths))
        raise ValueError(
            f"exact connectivity width {discovered} exceeds preregistered "
            f"maximum_width={width}; first offending maximum is sample {offender}"
        )
    connected_labels = np.repeat(centers[:, None, :], width, axis=1)
    coefficients = np.zeros((len(centers), width), dtype=np.float64)
    mask = np.zeros((len(centers), width), dtype=np.bool_)
    for sample, (rows, coefficient) in enumerate(zip(columns, values, strict=True)):
        count = len(rows)
        connected_labels[sample, :count] = rows
        coefficients[sample, :count] = coefficient
        mask[sample, :count] = True
    return ManyCellTreeConnectivity(
        center_labels=centers,
        connected_labels=connected_labels,
        coefficients=coefficients,
        mask=mask,
        widths=widths,
        physical_bonds=physical,
        model_bonds=model,
        connectivity_seconds=float(time.perf_counter() - started),
    )


@partial(jax.jit, static_argnames=("config",))
def _many_cell_kernel(params, centers, connected, coefficients, mask, config):
    center_logs = many_cell_schur_log_amplitudes(params, centers, config)
    flat = connected.reshape((-1, config.n - 1))
    connected_logs = many_cell_schur_log_amplitudes(
        params, flat, config
    ).reshape(connected.shape[:2])
    ratios = jnp.exp(connected_logs - center_logs[:, None])
    ratios = jnp.where(mask, ratios, 0.0 + 0.0j)
    return ratios, jnp.sum(coefficients * ratios, axis=1)


def evaluate_many_cell_tree_local_energy(
    params: dict[str, object],
    workload: ManyCellTreeConnectivity,
    config: ManyCellSchurViTConfig,
) -> ManyCellTreeLocalEnergy:
    """Evaluate pure-model matrix-free ratios and local energies on device."""

    started = time.perf_counter()
    centers = jnp.asarray(workload.center_labels)
    connected = jnp.asarray(workload.connected_labels)
    ratios, local = _many_cell_kernel(
        params,
        centers,
        connected,
        jnp.asarray(workload.coefficients),
        jnp.asarray(workload.mask),
        config,
    )
    ratios_host, local_host = jax.device_get((ratios, local))
    return ManyCellTreeLocalEnergy(
        ratios=np.asarray(ratios_host),
        local_energies=np.asarray(local_host),
        widths=np.asarray(workload.widths),
        amplitude_seconds=float(time.perf_counter() - started),
    )


def compile_many_cell_device_hamiltonian(
    config: ManyCellSchurViTConfig,
    physical_bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    *,
    candidate_capacity: int | None = None,
    connected_capacity: int | None = None,
    amplitude_tile_size: int | None = None,
    skip_padded_amplitudes: bool = False,
    bucket_padded_amplitudes: bool = False,
) -> ManyCellDeviceHamiltonianKernel:
    """Compile a full-H JAX kernel for changing autoregressive samples.

    Only state-independent geometry and SU(2) coefficient tables are prepared
    on the host. Every sampled center, vector transport, scalar contraction,
    amplitude ratio, and graph reduction runs inside the returned JIT kernel.
    Raw duplicate branches are summed exactly on device.

    With ``skip_padded_amplitudes=True`` the fixed buffers and overflow
    checks are unchanged, but scalar device conditionals skip empty neural
    tiles. Centers use ``lax.map``: an outer vmap would turn these conditionals
    into eager selects and evaluate their inactive branches. Fixed scan bounds
    retain JVP/reverse-mode support; no dynamic while-loop or host row rebuild.

    ``bucket_padded_amplitudes`` instead preserves batched connectivity and
    batched neural work. The checked evaluator transfers ONLY widths/overflow
    to select a power-of-two tile bucket, never rebuilding rows on the host.
    Its pure-JAX interface retains the equivalent padded action for tracing;
    adaptive dispatch and its extra scalar synchronization are timed explicitly.
    """

    if not isinstance(skip_padded_amplitudes,bool):
        raise TypeError('skip_padded_amplitudes must be boolean')
    if not isinstance(bucket_padded_amplitudes,bool) or (bucket_padded_amplitudes and skip_padded_amplitudes):
        raise ValueError('bucket_padded_amplitudes must be boolean and execution modes exclusive')
    if candidate_capacity is not None:
        if isinstance(candidate_capacity, bool) or not isinstance(
            candidate_capacity, Integral
        ):
            raise TypeError("candidate_capacity must be an integer or None")
        if candidate_capacity < 1:
            raise ValueError("candidate_capacity must be positive")
    if connected_capacity is not None:
        if isinstance(connected_capacity, bool) or not isinstance(
            connected_capacity, Integral
        ):
            raise TypeError("connected_capacity must be an integer or None")
        if connected_capacity < 1:
            raise ValueError("connected_capacity must be positive")
    if amplitude_tile_size is not None:
        if isinstance(amplitude_tile_size, bool) or not isinstance(
            amplitude_tile_size, Integral
        ):
            raise TypeError("amplitude_tile_size must be an integer or None")
        if amplitude_tile_size < 1:
            raise ValueError("amplitude_tile_size must be positive")
        if connected_capacity is not None and (
            amplitude_tile_size > connected_capacity
            or connected_capacity % amplitude_tile_size
        ):
            raise ValueError(
                "amplitude_tile_size must divide connected_capacity and not "
                "exceed it"
            )
    physical = _normalize_physical_bonds(physical_bonds)
    model_bonds = config.layout.remap_bonds(physical)
    started = time.perf_counter()
    plan = compile_jax_tree_connectivity_plan(
        config.layout.tree,
        model_bonds,
        target_q=config.target_q,
    )
    if candidate_capacity is None:
        grouped: dict[tuple[int, int], list[int]] = {}
        for bond_index, (left, right) in enumerate(
            zip(
                plan.left_candidate_capacity_bounds,
                plan.right_candidate_capacity_bounds,
                strict=True,
            )
        ):
            grouped.setdefault((int(left), int(right)), []).append(bond_index)
        bucket_rows = tuple(
            (
                capacities,
                np.asarray(indices, dtype=np.int32),
            )
            for capacities, indices in sorted(grouped.items())
        )
    else:
        bucket_rows = (
            (
                (int(candidate_capacity), int(candidate_capacity)),
                np.arange(plan.bond_count, dtype=np.int32),
            ),
        )
    raw_slots_per_sample = int(
        sum(
            len(indices) * left_capacity * right_capacity
            for (left_capacity, right_capacity), indices in bucket_rows
        )
    )
    compact_capacity = (
        raw_slots_per_sample
        if connected_capacity is None
        else int(connected_capacity)
    )
    tile_size = (
        compact_capacity
        if amplitude_tile_size is None
        else int(amplitude_tile_size)
    )
    if tile_size > compact_capacity or compact_capacity % tile_size:
        raise ValueError(
            "amplitude_tile_size must divide connected_capacity and not "
            "exceed it"
        )

    def uncompiled(params, center_labels, *, connectivity_only=False):
        centers = jnp.asarray(center_labels, dtype=jnp.int32)
        if centers.ndim != 2 or centers.shape[1] != config.n - 1:
            raise ValueError(f"center_labels must have shape (batch,{config.n - 1})")

        def one_center(center):
            raw_labels = []
            raw_coefficients = []
            raw_valid = []
            overflows = []
            for (left_capacity, right_capacity), bond_indices in bucket_rows:
                def apply_bond(_, bond_index):
                    connected, coefficients, valid, bond_overflow = (
                        jax_tree_bond_connectivity_rectangular(
                            center,
                            plan,
                            bond_index,
                            left_candidate_capacity=left_capacity,
                            right_candidate_capacity=right_capacity,
                        )
                    )
                    return None, (connected, coefficients, valid, bond_overflow)

                _, bucket = jax.lax.scan(
                    apply_bond,
                    None,
                    jnp.asarray(bond_indices),
                )
                labels, coefficients, valid, bond_overflow = bucket
                raw_labels.append(labels.reshape((-1, config.n - 1)))
                raw_coefficients.append(coefficients.reshape((-1,)))
                raw_valid.append(valid.reshape((-1,)))
                overflows.append(jnp.any(bond_overflow))
            connected, coefficients, valid, unique_count, compact_overflow = (
                jax_coalesce_tree_connectivity(
                    jnp.concatenate(raw_labels, axis=0),
                    jnp.concatenate(raw_coefficients, axis=0),
                    jnp.concatenate(raw_valid, axis=0),
                    n_sites=config.n,
                    output_capacity=compact_capacity,
                )
            )
            overflow=compact_overflow
            for candidate_overflow in overflows:
                overflow=overflow|candidate_overflow
            if connectivity_only:
                return connected,coefficients,valid,overflow,unique_count
            center_log=many_cell_schur_log_amplitudes(params,center[None,:],config)[0]
            if skip_padded_amplitudes:
                safe_connected=jnp.where(valid[:,None],connected,center[None,:])
                tiles=safe_connected.reshape((-1,tile_size,config.n-1))
                masks=valid.reshape((-1,tile_size))

                def active_tile(_,inputs):
                    tile,mask=inputs
                    logs=jax.lax.cond(jnp.any(mask),
                        lambda x: many_cell_schur_log_amplitudes(params,x,config),
                        lambda x: jnp.zeros((tile_size,),dtype=jnp.complex128),tile)
                    return None,logs

                _,tile_logs=jax.lax.scan(active_tile,None,(tiles,masks))
                connected_logs=tile_logs.reshape((compact_capacity,))
            elif tile_size == compact_capacity:
                connected_logs = many_cell_schur_log_amplitudes(
                    params, connected, config
                )
            else:
                connected_tiles = connected.reshape(
                    (compact_capacity // tile_size, tile_size, config.n - 1)
                )

                def evaluate_tile(_, tile):
                    return None, many_cell_schur_log_amplitudes(
                        params, tile, config
                    )

                _, tile_logs = jax.lax.scan(
                    evaluate_tile, None, connected_tiles
                )
                connected_logs = tile_logs.reshape((compact_capacity,))
            ratios = jnp.exp(connected_logs - center_log)
            local = jnp.sum(
                jnp.where(valid, coefficients * ratios, 0.0 + 0.0j)
            )
            return local, overflow, unique_count

        return (jax.lax.map(one_center,centers) if skip_padded_amplitudes and not connectivity_only
                else jax.vmap(one_center)(centers))

    def ratio_kernel(params,centers,connected,coefficients,valid):
        safe=jnp.where(valid[:,:,None],connected,centers[:,None,:])
        logs=many_cell_schur_log_amplitudes(params,safe.reshape((-1,config.n-1)),config)
        center_logs=many_cell_schur_log_amplitudes(params,centers,config)
        ratios=jnp.exp(logs.reshape(valid.shape)-center_logs[:,None])
        return jnp.sum(jnp.where(valid,coefficients*ratios,0.+0.j),axis=1)

    return ManyCellDeviceHamiltonianKernel(
        plan=plan,
        config=config,
        candidate_capacity=(
            None if candidate_capacity is None else int(candidate_capacity)
        ),
        connected_capacity=compact_capacity,
        amplitude_tile_size=tile_size,
        bucket_capacities=tuple(row[0] for row in bucket_rows),
        bucket_bond_counts=tuple(len(row[1]) for row in bucket_rows),
        raw_transition_slots_per_sample=raw_slots_per_sample,
        compilation_seconds=float(time.perf_counter() - started),
        _kernel=jax.jit(uncompiled),
        skip_padded_amplitudes=skip_padded_amplitudes,
        bucket_padded_amplitudes=bucket_padded_amplitudes,
        _connectivity_kernel=jax.jit(lambda centers:uncompiled(None,centers,connectivity_only=True)),
        _ratio_kernel=jax.jit(ratio_kernel),
    )


def evaluate_many_cell_device_local_energy(
    compiled: ManyCellDeviceHamiltonianKernel,
    params: dict[str, object],
    center_labels: Sequence[Sequence[int]] | np.ndarray | jax.Array,
    *,
    fail_on_overflow: bool = True,
) -> ManyCellDeviceLocalEnergy:
    """Evaluate one full-H batch with the accelerator-native recoupling plan."""

    centers = np.asarray(center_labels, dtype=np.int32)
    if centers.ndim != 2 or centers.shape[1] != compiled.config.n - 1:
        raise ValueError(
            f"center_labels must have shape (batch,{compiled.config.n - 1})"
        )
    if len(centers) < 1:
        raise ValueError("center_labels must be nonempty")
    started = time.perf_counter()
    selected_capacity=compiled.connected_capacity
    if compiled.bucket_padded_amplitudes:
        connected,coefficients,valid,overflow,unique_widths=compiled._connectivity_kernel(jnp.asarray(centers))
        concrete_overflow,concrete_widths=jax.device_get((overflow,unique_widths))
        if fail_on_overflow and np.any(concrete_overflow):
            raise RuntimeError('device recoupling capacity overflowed; no truncated batch accepted')
        units=max(1,(int(np.max(concrete_widths))+compiled.amplitude_tile_size-1)//compiled.amplitude_tile_size)
        selected_capacity=min(compiled.connected_capacity,
            compiled.amplitude_tile_size*2**(units-1).bit_length())
        local=compiled._ratio_kernel(params,jnp.asarray(centers),connected[:,:selected_capacity],
            coefficients[:,:selected_capacity],valid[:,:selected_capacity])
    else:
        local, overflow, unique_widths = compiled._kernel(params,jnp.asarray(centers))
    local_host, overflow_host, unique_host = jax.device_get(
        (local, overflow, unique_widths)
    )
    elapsed = float(time.perf_counter() - started)
    concrete_overflow = np.asarray(overflow_host, dtype=np.bool_)
    if fail_on_overflow and np.any(concrete_overflow):
        first = int(np.flatnonzero(concrete_overflow)[0])
        raise RuntimeError(
            "device recoupling candidate capacity overflowed for sample "
            f"{first}; increase the registered bucket capacity and recompile"
        )
    sample_count = len(centers)
    return ManyCellDeviceLocalEnergy(
        local_energies=np.asarray(local_host),
        overflow=concrete_overflow,
        evaluation_seconds=elapsed,
        sample_count=sample_count,
        bond_count=compiled.plan.bond_count,
        full_h_term_calls=sample_count * compiled.plan.bond_count,
        raw_transition_slots=(
            sample_count * compiled.raw_transition_slots_per_sample
        ),
        compact_transition_slots=sample_count * compiled.connected_capacity,
        unique_widths=np.asarray(unique_host, dtype=np.int32),
        evaluated_connected_amplitudes=(int(np.sum(
            (np.asarray(unique_host)+compiled.amplitude_tile_size-1)
            //compiled.amplitude_tile_size*compiled.amplitude_tile_size))
            if compiled.skip_padded_amplitudes else sample_count*selected_capacity),
    )


def many_cell_device_local_energies_jax(
    compiled: ManyCellDeviceHamiltonianKernel,
    params: dict[str, object],
    center_labels: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return local energies, overflow flags, and unique widths on device."""

    labels = jnp.asarray(center_labels, dtype=jnp.int32)
    if labels.ndim != 2 or labels.shape[1] != compiled.config.n - 1:
        raise ValueError(
            f"center_labels must have shape (batch,{compiled.config.n - 1})"
        )
    return compiled._kernel(params, labels)


def many_cell_device_hpsi(
    compiled: ManyCellDeviceHamiltonianKernel,
    params: dict[str, object],
    center_labels: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return centers, ``H psi``, overflow flags, and unique widths in JAX.

    This differentiable boundary is intended for sampled VMC force and SR
    code. Callers must run the checked local-energy evaluation on every batch
    before accepting an update; overflow flags are returned explicitly so a
    traced optimizer can never silently treat a truncated column as exact.
    """

    labels = jnp.asarray(center_labels, dtype=jnp.int32)
    local, overflow, unique_widths = many_cell_device_local_energies_jax(
        compiled, params, labels
    )
    centers = jnp.exp(
        many_cell_schur_log_amplitudes(params, labels, compiled.config)
    )
    return centers, local * centers, overflow, unique_widths


@partial(jax.jit, static_argnames=("config",))
def _external_carrier_kernel(
    carrier_centers: jax.Array,
    carrier_connected: jax.Array,
    residual_params,
    residual_scale: jax.Array,
    center_labels: jax.Array,
    connected_labels: jax.Array,
    coefficients: jax.Array,
    mask: jax.Array,
    config: ManyCellSchurViTConfig,
) -> tuple[jax.Array, jax.Array]:
    residual_center_logs = many_cell_schur_log_amplitudes(
        residual_params, center_labels, config
    )
    residual_connected_logs = many_cell_schur_log_amplitudes(
        residual_params,
        connected_labels.reshape((-1, config.n - 1)),
        config,
    ).reshape(connected_labels.shape[:2])
    center = carrier_centers + residual_scale * jnp.exp(residual_center_logs)
    connected = carrier_connected + residual_scale * jnp.exp(residual_connected_logs)
    ratios = connected / center[:, None]
    ratios = jnp.where(mask, ratios, 0.0 + 0.0j)
    return ratios, jnp.sum(coefficients * ratios, axis=1)


def evaluate_external_carrier_additive_local_energy(
    carrier_centers: np.ndarray,
    carrier_connected: np.ndarray,
    residual_params: dict[str, object],
    residual_scale: complex | float,
    workload: ManyCellTreeConnectivity,
    config: ManyCellSchurViTConfig,
) -> ManyCellTreeLocalEnergy:
    """Evaluate ``carrier + scale*ViT`` ratios in one padded JAX kernel.

    This interface is also the exact 4x4 replay boundary: the sealed carrier
    may be supplied as audited amplitudes in the same fusion basis.  At larger
    sizes the identical kernel accepts amplitudes from a separately sampled or
    learned carrier, without imposing its nodes on the residual channel.
    """

    centers = np.asarray(carrier_centers, dtype=np.complex128)
    connected = np.asarray(carrier_connected, dtype=np.complex128)
    if centers.shape != (len(workload.center_labels),):
        raise ValueError("carrier_centers have the wrong shape")
    if connected.shape != workload.mask.shape:
        raise ValueError("carrier_connected have the wrong shape")
    started = time.perf_counter()
    center_labels = jnp.asarray(workload.center_labels)
    connected_labels = jnp.asarray(workload.connected_labels)
    ratios, local = _external_carrier_kernel(
        jnp.asarray(centers),
        jnp.asarray(connected),
        residual_params,
        jnp.asarray(residual_scale, dtype=jnp.complex128),
        center_labels,
        connected_labels,
        jnp.asarray(workload.coefficients),
        jnp.asarray(workload.mask),
        config,
    )
    ratios_host, local_host = jax.device_get((ratios, local))
    return ManyCellTreeLocalEnergy(
        ratios=np.asarray(ratios_host),
        local_energies=np.asarray(local_host),
        widths=np.asarray(workload.widths),
        amplitude_seconds=float(time.perf_counter() - started),
    )


__all__ = [
    "apply_many_cell_tree_connectivity",
    "ManyCellAdditiveProposal",
    "ManyCellTreeConnectivity",
    "ManyCellTreeLocalEnergy",
    "ManyCellDeviceHamiltonianKernel",
    "ManyCellDeviceLocalEnergy",
    "build_many_cell_tree_connectivity",
    "compile_many_cell_device_hamiltonian",
    "evaluate_many_cell_device_local_energy",
    "many_cell_device_local_energies_jax",
    "many_cell_device_hpsi",
    "evaluate_external_carrier_additive_local_energy",
    "evaluate_many_cell_tree_local_energy",
    "sample_many_cell_additive_proposal",
]
