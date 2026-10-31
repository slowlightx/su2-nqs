"""Process-isolated target-GPU timing slices for the N=36 spin campaign.

Each static ``(system, arm, batch)`` shape is compiled in a fresh process.
This is an execution boundary only: the registered model, samples, full-H
work, replay count, and tolerances are inherited verbatim from the sealed
three-arm configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import (
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitude,
    fixed_sz_autoregressive_spatial_parameter_count,
    init_fixed_sz_autoregressive_spatial_vit,
    sample_fixed_sz_autoregressive_spatial_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_vit_parameter_count,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.many_cell_tree import (
    compile_many_cell_device_hamiltonian,
    evaluate_many_cell_device_local_energy,
)
from su2_nqs.vmc.spin_device import spin_device_hpsi

from .spin_scale_first_three_arm_preflight import ARM_IDS, ROOT, _bonds, _validate


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_accelerator_slice"
MERGE_TASK = "spin_scale_first_accelerator_merge"


def _shared_kwargs(shared: dict[str, object]) -> dict[str, object]:
    return {
        "embedding_dim": int(shared["embedding_dim"]),
        "head_count": int(shared["head_count"]),
        "depth": int(shared["depth"]),
        "mlp_ratio": int(shared["mlp_ratio"]),
        "phase_scale": float(shared["phase_scale"]),
        "initialization_scale": float(shared["initialization_scale"]),
        "direct_phase_initialization_scale": float(
            shared.get("direct_phase_initialization_scale", 0.0)
        ),
        "initial_distance_decay": float(shared["initial_distance_decay"]),
        "use_distance_envelope": True,
    }


def _selection(
    spec: dict[str, object], case_id: str, arm_id: str, batch_size: int
) -> tuple[dict[str, object], int, int]:
    """Return the registered case plus its case and batch indices."""

    _validate(spec)
    if arm_id not in ARM_IDS:
        raise ValueError(f"unregistered arm_id: {arm_id}")
    benchmark = spec.get("accelerator_benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("accelerator_benchmark is required for a timing slice")
    batches = [int(value) for value in benchmark["batch_sizes"]]
    if batch_size not in batches:
        raise ValueError(f"unregistered batch_size: {batch_size}")
    cases = list(spec["systems"])
    for index, case in enumerate(cases):
        if str(case["case_id"]) == case_id:
            return case, index, batches.index(batch_size)
    raise ValueError(f"unregistered case_id: {case_id}")


def _timing_summary(
    warm_times: list[float], batch_size: int, bond_count: int
) -> dict[str, object]:
    values = np.asarray(warm_times, dtype=np.float64)
    median = float(np.median(values))
    return {
        "warm_repetitions": len(warm_times),
        "warm_seconds": [float(value) for value in values],
        "warm_median_seconds": median,
        "warm_minimum_seconds": float(np.min(values)),
        "warm_maximum_seconds": float(np.max(values)),
        "samples_per_second": float(batch_size / median),
        "full_h_bond_terms_per_second": float(
            batch_size * bond_count / median
        ),
    }


def _check_replay(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> float:
    difference = float(
        np.max(np.abs(np.asarray(reference) - np.asarray(candidate)), initial=0.0)
    )
    if not np.allclose(reference, candidate, atol=atol, rtol=rtol):
        raise RuntimeError(
            "accelerator replay changed beyond the registered tolerance; "
            f"max_abs={difference}, atol={atol}, rtol={rtol}"
        )
    return difference


def _benchmark_su2(
    spec: dict[str, object],
    case: dict[str, object],
    *,
    seed: int,
    sample_key: jax.Array,
    batch_size: int,
    amplitude_tile_size: int | None,
    device_microbatch_size: int | None,
) -> tuple[dict[str, object], object]:
    geometry = str(case["geometry"])
    layout = rectangular_four_spin_layout(
        6, 6, triangular_coordinates=geometry == "triangular"
    )
    config = ManyCellSchurViTConfig(
        layout,
        target_q=0,
        geometry=geometry,
        **_shared_kwargs(spec["shared_spatial_backbone"]),
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(seed), config)
    bonds = _bonds(geometry, float(case["j2_over_j1"]))
    compiled = compile_many_cell_device_hamiltonian(
        config,
        bonds,
        connected_capacity=int(spec["local_canary"]["su2_connected_capacity"]),
        amplitude_tile_size=amplitude_tile_size,
    )
    sample_started = time.perf_counter()
    samples = sample_many_cell_schur_vit(
        params, sample_key, batch_size, config
    )
    jax.block_until_ready(samples)
    sampling_seconds = float(time.perf_counter() - sample_started)
    benchmark = spec["accelerator_benchmark"]
    microbatch_size = (
        int(benchmark["device_microbatch_size"])
        if device_microbatch_size is None
        else int(device_microbatch_size)
    )
    chunks = [
        samples[start : start + microbatch_size]
        for start in range(0, batch_size, microbatch_size)
    ]

    def evaluate():
        started = time.perf_counter()
        results = [
            evaluate_many_cell_device_local_energy(compiled, params, chunk)
            for chunk in chunks
        ]
        elapsed = float(time.perf_counter() - started)
        return (
            np.concatenate([row.local_energies for row in results]),
            np.concatenate([row.overflow for row in results]),
            np.concatenate([row.unique_widths for row in results]),
            elapsed,
        )

    first_local, first_overflow, first_widths, first_seconds = evaluate()
    atol = float(benchmark["replay_absolute_tolerance"])
    rtol = float(benchmark["replay_relative_tolerance"])
    warm_times = []
    replay_maximum = 0.0
    maximum_width = int(np.max(first_widths))
    overflow_count = int(np.count_nonzero(first_overflow))
    for _ in range(int(benchmark["warm_repetitions"])):
        replay_local, replay_overflow, replay_widths, elapsed = evaluate()
        replay_maximum = max(
            replay_maximum,
            _check_replay(
                first_local,
                replay_local,
                atol=atol,
                rtol=rtol,
            ),
        )
        maximum_width = max(maximum_width, int(np.max(replay_widths)))
        overflow_count += int(np.count_nonzero(replay_overflow))
        warm_times.append(elapsed)
    row = {
        "sampling_seconds": sampling_seconds,
        "shape_compile_or_first_seconds": first_seconds,
        "plan_compilation_seconds": compiled.compilation_seconds,
        "device_microbatch_size": microbatch_size,
        "device_microbatch_count": len(chunks),
        "amplitude_tile_size": compiled.amplitude_tile_size,
        "all_finite": bool(np.all(np.isfinite(first_local))),
        "overflow_count": overflow_count,
        "maximum_replay_abs_difference": replay_maximum,
        "maximum_unique_width": maximum_width,
        "connected_capacity": compiled.connected_capacity,
        "raw_transition_slots_per_sample": (
            compiled.raw_transition_slots_per_sample
        ),
        **_timing_summary(warm_times, batch_size, len(bonds)),
    }
    return row, (params, many_cell_schur_vit_parameter_count(params))


def _benchmark_physical(
    spec: dict[str, object],
    case: dict[str, object],
    *,
    seed: int,
    sample_key: jax.Array,
    batch_size: int,
    target_m2: int | None,
) -> tuple[dict[str, object], object]:
    geometry = str(case["geometry"])
    config = FixedSzAutoregressiveSpatialViTConfig(
        geometry,
        6,
        6,
        target_m2=target_m2,
        **_shared_kwargs(spec["shared_spatial_backbone"]),
    )
    params = init_fixed_sz_autoregressive_spatial_vit(
        jax.random.PRNGKey(seed), config
    )
    bonds = _bonds(geometry, float(case["j2_over_j1"]))
    sample_started = time.perf_counter()
    samples = sample_fixed_sz_autoregressive_spatial_vit(
        params, sample_key, batch_size, config
    )
    jax.block_until_ready(samples)
    sampling_seconds = float(time.perf_counter() - sample_started)

    def action(values, states):
        return spin_device_hpsi(
            values,
            states,
            bonds,
            config,
            amplitude_fn=lambda candidate, row, model: jnp.exp(
                fixed_sz_autoregressive_spatial_log_amplitude(
                    candidate, row, model
                )
            ),
        )

    kernel = jax.jit(action)
    benchmark = spec["accelerator_benchmark"]
    microbatch_size = int(benchmark["device_microbatch_size"])
    chunks = [
        samples[start : start + microbatch_size]
        for start in range(0, batch_size, microbatch_size)
    ]

    def evaluate() -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        rows = []
        for chunk in chunks:
            centers, hpsi = kernel(params, chunk)
            jax.block_until_ready(hpsi)
            rows.append(np.asarray(hpsi / centers))
        elapsed = float(time.perf_counter() - started)
        local = np.concatenate(rows)
        if not np.all(np.isfinite(local)):
            raise FloatingPointError("physical-H timing slice is non-finite")
        return local, elapsed

    first_local, first_seconds = evaluate()
    atol = float(benchmark["replay_absolute_tolerance"])
    rtol = float(benchmark["replay_relative_tolerance"])
    warm_times = []
    replay_maximum = 0.0
    for _ in range(int(benchmark["warm_repetitions"])):
        replay, elapsed = evaluate()
        replay_maximum = max(
            replay_maximum,
            _check_replay(first_local, replay, atol=atol, rtol=rtol),
        )
        warm_times.append(elapsed)
    row = {
        "sampling_seconds": sampling_seconds,
        "shape_compile_or_first_seconds": first_seconds,
        "device_microbatch_size": microbatch_size,
        "device_microbatch_count": len(chunks),
        "all_finite": True,
        "overflow_count": 0,
        "maximum_replay_abs_difference": replay_maximum,
        **_timing_summary(warm_times, batch_size, len(bonds)),
    }
    return row, (
        params,
        fixed_sz_autoregressive_spatial_parameter_count(params),
    )


def run_spin_scale_first_accelerator_slice(
    config_path: Path,
    output: Path,
    *,
    case_id: str,
    arm_id: str,
    batch_size: int,
    amplitude_tile_size: int | None = None,
    device_microbatch_size: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Measure one registered static shape in one fresh process."""

    spec = json.loads(config_path.read_text())
    case, case_index, batch_index = _selection(
        spec, case_id, arm_id, batch_size
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    seed = int(spec["seed"]) + 10 * case_index
    arm_index = ARM_IDS.index(arm_id)
    sample_key = jax.random.PRNGKey(
        seed + 1000 + arm_index + 10 * batch_index
    )
    started = time.perf_counter()
    registered_tiles = spec["accelerator_benchmark"].get(
        "su2_amplitude_tile_sizes", []
    )
    if amplitude_tile_size is not None:
        if arm_id != ARM_IDS[0]:
            raise ValueError("amplitude tiling applies only to the exact-SU2 arm")
        if int(amplitude_tile_size) not in registered_tiles:
            raise ValueError(
                f"unregistered SU2 amplitude tile size: {amplitude_tile_size}"
            )
        amplitude_tile_size = int(amplitude_tile_size)
    elif arm_id == ARM_IDS[0]:
        configured_tile = spec["accelerator_benchmark"].get(
            "su2_amplitude_tile_size"
        )
        amplitude_tile_size = (
            None if configured_tile is None else int(configured_tile)
        )
    if device_microbatch_size is not None:
        if arm_id != ARM_IDS[0]:
            raise ValueError(
                "SU2 device microbatch overrides apply only to the exact-SU2 arm"
            )
        registered_microbatches = spec["accelerator_benchmark"].get(
            "su2_device_microbatch_sizes", []
        )
        if int(device_microbatch_size) not in registered_microbatches:
            raise ValueError(
                f"unregistered SU2 device microbatch: {device_microbatch_size}"
            )
        if batch_size % int(device_microbatch_size):
            raise ValueError("device microbatch must divide the logical batch")
        device_microbatch_size = int(device_microbatch_size)
    if arm_id == ARM_IDS[0]:
        timing, (params, parameter_count) = _benchmark_su2(
            spec,
            case,
            seed=seed,
            sample_key=sample_key,
            batch_size=batch_size,
            amplitude_tile_size=amplitude_tile_size,
            device_microbatch_size=device_microbatch_size,
        )
    else:
        timing, (params, parameter_count) = _benchmark_physical(
            spec,
            case,
            seed=seed,
            sample_key=sample_key,
            batch_size=batch_size,
            target_m2=0 if arm_id == ARM_IDS[1] else None,
        )
    benchmark = spec["accelerator_benchmark"]
    runtime = numerical_runtime_identity()
    passed = bool(
        runtime["jax_backend"] == "gpu"
        and timing["all_finite"]
        and int(timing["overflow_count"]) == 0
        and int(timing["warm_repetitions"])
        == int(benchmark["warm_repetitions"])
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "case_id": case_id,
        "arm_id": arm_id,
        "batch_size": int(batch_size),
        "amplitude_tile_size": amplitude_tile_size,
        "device_microbatch_size": device_microbatch_size,
        "case_index": case_index,
        "batch_index": batch_index,
        "seed": seed,
        "parameter_count": int(parameter_count),
        "initial_params_sha256": pytree_sha256(params),
        "timing": timing,
        "wall_seconds": float(time.perf_counter() - started),
        "fresh_process_required": True,
        "optimization_started": False,
        "passed": passed,
        "runtime_identity": runtime,
        "claim_boundary": spec["claim_boundary"],
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(
        payload["source_manifest"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


def run_spin_scale_first_accelerator_merge(
    config_path: Path,
    input_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Fail closed unless all preregistered isolated timing slices pass."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    expected = {
        (str(case["case_id"]), arm_id, int(batch_size))
        for case in spec["systems"]
        for arm_id in ARM_IDS
        for batch_size in spec["accelerator_benchmark"]["batch_sizes"]
    }
    rows = []
    observed: dict[tuple[str, str, int], dict[str, object]] = {}
    for path in sorted(input_root.glob("*/*/batch*/results.json")):
        row = json.loads(path.read_text())
        if row.get("task") != TASK:
            raise ValueError(f"unexpected slice task: {path}")
        stored_hash = row.pop("result_sha256")
        if stored_hash != canonical_sha256(row):
            raise RuntimeError(f"slice seal failed: {path}")
        row["result_sha256"] = stored_hash
        if row["config_sha256"] != canonical_sha256(spec):
            raise RuntimeError(f"slice config seal differs: {path}")
        key = (row["case_id"], row["arm_id"], int(row["batch_size"]))
        if key in observed:
            raise RuntimeError(f"duplicate accelerator timing slice: {key}")
        observed[key] = row
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise RuntimeError(
            f"accelerator slice coverage mismatch; missing={missing}, extra={extra}"
        )
    rows = [observed[key] for key in sorted(expected)]
    gates = {
        "all_registered_slices_present_once": True,
        "all_slices_used_gpu_backend": all(
            row["runtime_identity"]["jax_backend"] == "gpu" for row in rows
        ),
        "all_slices_passed": all(bool(row["passed"]) for row in rows),
        "all_slices_used_fresh_process_contract": all(
            bool(row["fresh_process_required"]) for row in rows
        ),
        "no_optimization_was_started": not any(
            bool(row["optimization_started"]) for row in rows
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": MERGE_TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "slice_count": len(rows),
        "slice_manifest": {
            f"{row['case_id']}/{row['arm_id']}/batch{row['batch_size']}": row[
                "result_sha256"
            ]
            for row in rows
        },
        "slices": rows,
        "gate_results": gates,
        "accelerator_admission_passed": all(gates.values()),
        "optimization_started": False,
        "production_training_ready": False,
        "claim_boundary": spec["claim_boundary"],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(destination, payload)
    return payload


__all__ = [
    "MERGE_TASK",
    "TASK",
    "run_spin_scale_first_accelerator_merge",
    "run_spin_scale_first_accelerator_slice",
]
