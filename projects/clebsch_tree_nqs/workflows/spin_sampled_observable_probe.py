"""Sealed 6x6 sampled probe for full SU(2)-invariant structure factors.

The workflow consumes immutable ``sampled-runtime`` v2 results and their
final Markov-chain checkpoints.  It rebuilds the registered D4 U(1)/SU(2)
models through the runtime workflow's shared initializer, validates cached
amplitudes, and measures several standard ``N^-1`` structure factors using
one shared pair-swapped-amplitude batch.

These are random-initialization aggregation and cost diagnostics.  They are
not optimized energies, order parameters, or phase evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime_workflow
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    repo_relative_path,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.spin_observables import (
    SpinStructureFactorSpec,
    prepare_spin_structure_factor,
    reciprocal_phase_weights,
    spin_structure_factor_multi_local_values,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
PROBE_CHECKPOINT_SCHEMA_VERSION = 1
PROBE_RESULT_SCHEMA_VERSION = 1


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("observable probe requires schema_version=2 and domain='spin'")
    if spec.get("task") != "sampled_full_spin_observable_probe":
        raise ValueError("config task is not the sampled full-spin observable probe")
    inputs = spec.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("observable probe requires square and triangular inputs")
    seen: set[str] = set()
    geometries: set[str] = set()
    expected_names = {
        "square": ("q0", "neel", "stripe_x", "stripe_y"),
        "triangular": ("q0", "k", "m1", "m2", "m3"),
    }
    expected_conventions = {
        "square": "row_major_cartesian_column_row",
        "triangular": "row_major_triangular_phase_row_column",
    }
    for entry in inputs:
        if not isinstance(entry, dict):
            raise TypeError("every probe input must be an object")
        case_id = str(entry.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("probe case_id values must be nonempty and unique")
        seen.add(case_id)
        geometry = str(entry.get("geometry"))
        if geometry not in expected_names:
            raise ValueError("probe geometry must be square or triangular")
        geometries.add(geometry)
        if entry.get("coordinate_convention") != expected_conventions[geometry]:
            raise ValueError("probe coordinate convention does not match geometry")
        path = entry.get("runtime_result")
        if not isinstance(path, str) or not path.endswith("/results.json"):
            raise ValueError("runtime_result must be a repository-relative results.json")
        if not str(entry.get("expected_runtime_case_id", "")):
            raise ValueError("expected_runtime_case_id must be nonempty")
        modes = entry.get("modes")
        if not isinstance(modes, list):
            raise TypeError("modes must be a list")
        names = tuple(str(mode.get("name")) for mode in modes)
        if names != expected_names[geometry]:
            raise ValueError("probe modes must contain the registered q0/order channels")
        for mode in modes:
            vector = np.asarray(mode.get("wavevector"), dtype=np.float64)
            if vector.shape != (2,) or np.any(~np.isfinite(vector)):
                raise ValueError("every probe wavevector must be a finite 2-vector")
    if geometries != {"square", "triangular"}:
        raise ValueError("observable probe requires one input per geometry")
    probe = spec.get("probe")
    if not isinstance(probe, dict):
        raise TypeError("probe must be an object")
    for name in (
        "continuation_sweeps",
        "checkpoint_every_sweeps",
        "observable_batch_size",
        "minimum_local_samples",
        "block_length_sweeps",
    ):
        _positive_integer(probe.get(name), f"probe.{name}")
    if isinstance(probe.get("seed"), bool) or not isinstance(probe.get("seed"), int):
        raise TypeError("probe.seed must be an integer")
    for name in ("cached_amplitude_tolerance", "su2_q0_local_tolerance"):
        value = float(probe.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"probe.{name} must be positive and finite")


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _repository_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("probe inputs must remain inside the repository") from error
    return candidate


def _load_runtime_result(entry: dict[str, object]) -> tuple[dict[str, object], dict[str, object], Path]:
    result_path = _repository_path(str(entry["runtime_result"]))
    payload = json.loads(result_path.read_text())
    if (
        payload.get("schema_version") != runtime_workflow.RUNTIME_RESULT_SCHEMA_VERSION
        or payload.get("domain") != "spin"
        or payload.get("task") != "physical_mcmc_runtime_smoke"
        or payload.get("status") != "complete"
    ):
        raise RuntimeError(f"runtime input is not a complete v2 spin smoke: {result_path}")
    config = payload.get("config")
    source = payload.get("source_sha256")
    runtime_identity = payload.get("runtime_identity")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError(f"runtime config seal failed: {result_path}")
    if payload.get("source_manifest_sha256") != canonical_sha256(source):
        raise RuntimeError(f"runtime source seal failed: {result_path}")
    if payload.get("runtime_identity_sha256") != canonical_sha256(runtime_identity):
        raise RuntimeError(f"runtime identity seal failed: {result_path}")
    if payload.get("trial_manifest_sha256") != canonical_sha256(
        embedded_trial_manifest(payload)
    ):
        raise RuntimeError(f"runtime trial seal failed: {result_path}")
    current_runtime_source = runtime_workflow._source_manifest()
    if source != current_runtime_source or payload.get("source_manifest_sha256") != canonical_sha256(
        current_runtime_source
    ):
        raise RuntimeError(f"runtime input is stale relative to current source: {result_path}")
    current_identity = numerical_runtime_identity()
    if runtime_identity != current_identity:
        raise RuntimeError(
            "runtime input numerical identity differs from the active process; "
            "run the probe with the same JAX platform environment"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise RuntimeError("runtime input must contain exactly one case")
    case = cases[0]
    if (
        case.get("case_id") != entry["expected_runtime_case_id"]
        or case.get("geometry") != entry["geometry"]
        or int(case.get("n_sites", -1)) != 36
    ):
        raise RuntimeError("runtime input case identity does not match the probe config")
    trials = case.get("trials")
    if not isinstance(trials, list) or len(trials) != 2:
        raise RuntimeError("runtime input must contain exactly U1 and SU2 trials")
    if {str(trial.get("backbone")) for trial in trials} != {"u1", "su2"}:
        raise RuntimeError("runtime input trials must be the U1/SU2 pair")
    return payload, case, result_path


def _load_runtime_checkpoint(
    result_path: Path,
    payload: dict[str, object],
    case: dict[str, object],
    trial: dict[str, object],
) -> tuple[dict[str, object], Path]:
    checkpoint_record = trial.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise RuntimeError("runtime trial lacks checkpoint metadata")
    checkpoint_path = result_path.parent / str(checkpoint_record.get("path"))
    checkpoint = json.loads(checkpoint_path.read_text())
    if (
        checkpoint.get("schema_version")
        != runtime_workflow.RUNTIME_CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("status") != "complete"
    ):
        raise RuntimeError(f"runtime checkpoint is not complete v2: {checkpoint_path}")
    stored = checkpoint.get("checkpoint_sha256")
    unsealed = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if stored != canonical_sha256(unsealed):
        raise RuntimeError(f"runtime checkpoint seal failed: {checkpoint_path}")
    if stored != checkpoint_record.get("checkpoint_sha256"):
        raise RuntimeError("runtime result references a different checkpoint seal")
    identity = checkpoint.get("identity")
    expected_identity = {
        "config_sha256": payload["config_sha256"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "runtime_identity_sha256": payload["runtime_identity_sha256"],
        "case_id": case["case_id"],
        "arm_id": trial["arm_id"],
        "backbone": trial["backbone"],
        "initial_params_sha256": trial["initial_params_sha256"],
    }
    if identity != expected_identity:
        raise RuntimeError("runtime checkpoint identity differs from its trial")
    if (
        checkpoint.get("initial_chain_states_sha256")
        != trial.get("initial_chain_states_sha256")
        or checkpoint.get("common_profile_states_sha256")
        != trial.get("common_profile_states_sha256")
    ):
        raise RuntimeError("runtime checkpoint chain/profile identity mismatch")
    states = np.asarray(checkpoint.get("states"), dtype=np.int32)
    if states.shape != (16, 36) or np.any((states != 0) & (states != 1)):
        raise RuntimeError("runtime checkpoint states are not 16 physical 36-site chains")
    if np.any(np.sum(states, axis=1) != 18):
        raise RuntimeError("runtime checkpoint left the fixed-M2=0 support")
    if len(checkpoint.get("amplitudes", [])) != len(states):
        raise RuntimeError("runtime checkpoint cached amplitudes have the wrong size")
    return checkpoint, checkpoint_path


def _geometry_runtime(
    payload: dict[str, object], case: dict[str, object]
) -> tuple[tuple[tuple[int, int, float], ...], PhysicalSpatialProjector]:
    rows, columns = int(case["rows"]), int(case["columns"])
    ratio = float(case["j2_over_j1"])
    if case["geometry"] == "square":
        bonds = square_j1j2_bonds(rows, columns, j1=1.0, j2=ratio, periodic=True)
        group = rectangular_translation_group(rows, columns)
    elif case["geometry"] == "triangular":
        bonds = triangular_j1j2_bonds(
            rows, columns, j1=1.0, j2=ratio, periodic=True
        )
        group = triangular_hnf_translation_group(((columns, 0), (0, rows)))
    else:  # pragma: no cover - guarded by input validation
        raise ValueError(f"unsupported geometry: {case['geometry']!r}")
    if canonical_sha256(bonds) != case.get("bonds_sha256"):
        raise RuntimeError("reconstructed bonds differ from the runtime result")
    if canonical_sha256(group) != case.get("translation_group_sha256"):
        raise RuntimeError("reconstructed translation group differs from runtime result")
    projection = payload["config"]["projection"]
    projector = PhysicalSpatialProjector(
        group,
        orbit_chunk_size=int(projection["orbit_chunk_size"]),
        rematerialize_orbit=bool(projection["rematerialize_orbit"]),
    )
    return bonds, projector


def _mode_specs(
    entry: dict[str, object], *, rows: int, columns: int
) -> tuple[tuple[str, ...], tuple[SpinStructureFactorSpec, ...]]:
    if entry["geometry"] == "square":
        coordinates = tuple(
            (float(column), float(row))
            for row in range(rows)
            for column in range(columns)
        )
    else:
        coordinates = tuple(
            (float(row), float(column))
            for row in range(rows)
            for column in range(columns)
        )
    names = tuple(str(mode["name"]) for mode in entry["modes"])
    specs = tuple(
        prepare_spin_structure_factor(
            reciprocal_phase_weights(coordinates, mode["wavevector"])
        )
        for mode in entry["modes"]
    )
    if any(spec.normalization_kind != "structure_factor" for spec in specs):
        raise RuntimeError("sampled observable probe must use standard N^-1 S(q)")
    return names, specs


def _complex_pairs(values: np.ndarray) -> list:
    concrete = np.asarray(values, dtype=np.complex128)
    return np.stack((concrete.real, concrete.imag), axis=-1).tolist()


def _pairs_complex(values: object) -> np.ndarray:
    concrete = np.asarray(values, dtype=np.float64)
    if concrete.shape[-1] != 2:
        raise ValueError("complex-pair checkpoint data must end in length two")
    return concrete[..., 0] + 1.0j * concrete[..., 1]


def _probe_checkpoint_identity(
    *,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    case_id: str,
    trial: dict[str, object],
    runtime_checkpoint: dict[str, object],
) -> dict[str, object]:
    return {
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "runtime_identity_sha256": runtime_hash,
        "case_id": case_id,
        "arm_id": trial["arm_id"],
        "backbone": trial["backbone"],
        "initial_params_sha256": trial["initial_params_sha256"],
        "upstream_trial_payload_sha256": trial["trial_payload_sha256"],
        "upstream_checkpoint_sha256": runtime_checkpoint["checkpoint_sha256"],
    }


def _write_probe_checkpoint(
    path: Path,
    *,
    identity: dict[str, object],
    status: str,
    states: np.ndarray,
    amplitudes: np.ndarray,
    key: np.ndarray,
    completed_sweeps: int,
    diagnostics: list[dict[str, float]],
    local_values: np.ndarray,
    state_hashes: list[str],
    cold_observable_seconds: float,
    warm_observable_seconds: list[float],
    sampler_seconds: list[float],
    resume_count: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROBE_CHECKPOINT_SCHEMA_VERSION,
        "status": status,
        "identity": identity,
        "states": np.asarray(states, dtype=np.int32).tolist(),
        "amplitudes": _complex_pairs(amplitudes),
        "key": np.asarray(key, dtype=np.uint32).tolist(),
        "completed_sweeps": int(completed_sweeps),
        "diagnostics": diagnostics,
        "local_values": _complex_pairs(local_values),
        "state_hashes": state_hashes,
        "cold_observable_seconds": float(cold_observable_seconds),
        "warm_observable_seconds": [float(value) for value in warm_observable_seconds],
        "sampler_seconds": [float(value) for value in sampler_seconds],
        "resume_count": int(resume_count),
    }
    payload["checkpoint_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return payload


def _load_probe_checkpoint(
    path: Path, identity: dict[str, object]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != PROBE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"probe checkpoint schema mismatch: {path}")
    stored = payload.get("checkpoint_sha256")
    unsealed = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    if stored != canonical_sha256(unsealed):
        raise RuntimeError(f"probe checkpoint seal failed: {path}")
    if payload.get("identity") != identity:
        raise RuntimeError(f"probe checkpoint identity mismatch: {path}")
    return payload


def _summarize_mode(
    values: np.ndarray, *, block_length: int
) -> dict[str, object]:
    concrete = np.asarray(values, dtype=np.complex128)
    if concrete.ndim != 2 or min(concrete.shape) < 2:
        raise ValueError("mode series must have shape (sweeps>=2, chains>=2)")
    real = concrete.real
    correlation = runtime_workflow._autocorrelation_summary(real)
    ess = max(float(correlation["ess"]), 1.0)
    flattened = real.reshape(-1)
    iid_ess_mcse = float(np.std(flattened, ddof=1) / np.sqrt(ess))
    chain_means = np.mean(real, axis=0)
    chain_mcse = float(np.std(chain_means, ddof=1) / np.sqrt(len(chain_means)))
    usable = (real.shape[0] // block_length) * block_length
    blocked = real[:usable].reshape(
        usable // block_length, block_length, real.shape[1]
    ).mean(axis=1)
    blocked_flat = blocked.reshape(-1)
    blocked_mcse = float(
        np.std(blocked_flat, ddof=1) / np.sqrt(len(blocked_flat))
    )
    return {
        "mean_real": float(np.mean(real)),
        "mean_imaginary": float(np.mean(concrete.imag)),
        "maximum_abs_imaginary_local": float(np.max(np.abs(concrete.imag))),
        "standard_deviation_real": float(np.std(flattened, ddof=1)),
        "iid_ess_mcse": iid_ess_mcse,
        "chain_mean_mcse": chain_mcse,
        "blocked_chain_mcse": blocked_mcse,
        "block_length_sweeps": block_length,
        "block_count_per_chain": usable // block_length,
        **correlation,
    }


def _run_arm_probe(
    *,
    runtime: runtime_workflow._ArmRuntime,
    entry: dict[str, object],
    case: dict[str, object],
    runtime_trial: dict[str, object],
    runtime_checkpoint: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    spec: dict[str, object],
    checkpoint_path: Path,
    identity: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    probe = spec["probe"]
    rows, columns = int(case["rows"]), int(case["columns"])
    names, observable_specs = _mode_specs(entry, rows=rows, columns=columns)
    batch_size = int(probe["observable_batch_size"])
    upstream_states = np.asarray(runtime_checkpoint["states"], dtype=np.int32)
    upstream_amplitudes = runtime_workflow._pairs_complex(
        runtime_checkpoint["amplitudes"]
    )
    if len(upstream_states) % batch_size:
        raise ValueError("observable_batch_size must divide the upstream chain count")

    amplitude_batch = jax.jit(
        jax.vmap(lambda row: runtime.projected_amplitude(runtime.params, row))
    )
    rebuilt_amplitudes_device = amplitude_batch(jnp.asarray(upstream_states))
    jax.block_until_ready(rebuilt_amplitudes_device)
    rebuilt_amplitudes = np.asarray(rebuilt_amplitudes_device, dtype=np.complex128)
    cached_discrepancy = float(
        np.max(np.abs(rebuilt_amplitudes - upstream_amplitudes))
    )
    if cached_discrepancy > float(probe["cached_amplitude_tolerance"]):
        raise RuntimeError("rebuilt model does not reproduce upstream cached amplitudes")
    if pytree_sha256(runtime.params) != runtime_trial["initial_params_sha256"]:
        raise RuntimeError("rebuilt model parameters differ from the runtime trial")

    checkpoint = None if overwrite else _load_probe_checkpoint(checkpoint_path, identity)
    if checkpoint is None:
        states = upstream_states
        amplitudes = upstream_amplitudes
        source_key = jnp.asarray(runtime_checkpoint["key"], dtype=jnp.uint32)
        fold_value = int(
            canonical_sha256(
                {
                    "probe_seed": int(probe["seed"]),
                    "case_id": entry["case_id"],
                    "arm_id": runtime.arm_id,
                }
            )[:8],
            16,
        )
        key = np.asarray(jax.random.fold_in(source_key, fold_value), dtype=np.uint32)
        completed_sweeps = 0
        diagnostics: list[dict[str, float]] = []
        local_values = np.empty((0, len(states), len(names)), dtype=np.complex128)
        state_hashes: list[str] = []
        cold_observable_seconds = 0.0
        warm_observable_seconds: list[float] = []
        sampler_seconds: list[float] = []
        resume_count = 0
    else:
        states = np.asarray(checkpoint["states"], dtype=np.int32)
        amplitudes = _pairs_complex(checkpoint["amplitudes"])
        key = np.asarray(checkpoint["key"], dtype=np.uint32)
        completed_sweeps = int(checkpoint["completed_sweeps"])
        diagnostics = list(checkpoint["diagnostics"])
        local_values = _pairs_complex(checkpoint["local_values"])
        state_hashes = list(checkpoint["state_hashes"])
        cold_observable_seconds = float(checkpoint["cold_observable_seconds"])
        warm_observable_seconds = list(checkpoint["warm_observable_seconds"])
        sampler_seconds = list(checkpoint["sampler_seconds"])
        resume_count = int(checkpoint["resume_count"]) + 1

    def local_kernel(candidate, samples):
        return spin_structure_factor_multi_local_values(
            candidate,
            samples,
            observable_specs,
            amplitude_fn=runtime.projected_amplitude,
        )

    compiled_local = jax.jit(local_kernel)

    def measure() -> tuple[np.ndarray, float]:
        chunks = []
        started = time.perf_counter()
        for start in range(0, len(states), batch_size):
            values = compiled_local(
                runtime.params, jnp.asarray(states[start : start + batch_size])
            )
            jax.block_until_ready(values)
            chunks.append(np.asarray(values, dtype=np.complex128))
        return np.concatenate(chunks, axis=0), time.perf_counter() - started

    sampler_spec = runtime_trial["sampler"]
    proposals_per_sweep = int(sampler_spec["proposals_per_sweep"])
    bond_probability = float(sampler_spec["bond_proposal_probability"])
    exchange_edges = tuple((first, second) for first, second, _ in bonds)

    def advance(candidate, sweep_key, chain_states, chain_amplitudes):
        return run_exchange_sweeps(
            candidate,
            sweep_key,
            chain_states,
            amplitude_fn=runtime.projected_amplitude,
            particle_kind="spin",
            sweeps=proposals_per_sweep,
            amplitudes=chain_amplitudes,
            exchange_edges=exchange_edges,
            bond_proposal_probability=bond_probability,
        )

    compiled_advance = jax.jit(advance)

    def checkpoint_now(status: str) -> dict[str, object]:
        return _write_probe_checkpoint(
            checkpoint_path,
            identity=identity,
            status=status,
            states=states,
            amplitudes=amplitudes,
            key=key,
            completed_sweeps=completed_sweeps,
            diagnostics=diagnostics,
            local_values=local_values,
            state_hashes=state_hashes,
            cold_observable_seconds=cold_observable_seconds,
            warm_observable_seconds=warm_observable_seconds,
            sampler_seconds=sampler_seconds,
            resume_count=resume_count,
        )

    total_started = time.perf_counter()
    if len(local_values) == 0:
        values, elapsed = measure()
        cold_observable_seconds = elapsed
        local_values = values[None, :, :]
        state_hashes.append(array_sha256(states))
        checkpoint_now("running")

    while completed_sweeps < int(probe["continuation_sweeps"]):
        sweep_started = time.perf_counter()
        states_device, amplitudes_device, key_device, diagnostic_device = compiled_advance(
            runtime.params,
            jnp.asarray(key),
            jnp.asarray(states),
            jnp.asarray(amplitudes),
        )
        jax.block_until_ready(
            (states_device, amplitudes_device, key_device, diagnostic_device)
        )
        sampler_seconds.append(time.perf_counter() - sweep_started)
        states = np.asarray(states_device, dtype=np.int32)
        amplitudes = np.asarray(amplitudes_device, dtype=np.complex128)
        key = np.asarray(key_device, dtype=np.uint32)
        diagnostics.append(runtime_workflow._float_diagnostics(diagnostic_device))
        completed_sweeps += 1
        values, elapsed = measure()
        warm_observable_seconds.append(elapsed)
        local_values = np.concatenate((local_values, values[None, :, :]), axis=0)
        state_hashes.append(array_sha256(states))
        if completed_sweeps % int(probe["checkpoint_every_sweeps"]) == 0:
            checkpoint_now("running")

    final_checkpoint = checkpoint_now("complete")
    finite = bool(np.all(np.isfinite(local_values)))
    q0_index = names.index("q0")
    q0_maximum = float(np.max(np.abs(local_values[:, :, q0_index])))
    sample_count = int(local_values.shape[0] * local_values.shape[1])
    summaries = {
        name: _summarize_mode(
            local_values[:, :, index],
            block_length=int(probe["block_length_sweeps"]),
        )
        for index, name in enumerate(names)
    }
    su2_q0_passed = (
        q0_maximum <= float(probe["su2_q0_local_tolerance"])
        if runtime.backbone == "su2"
        else True
    )
    gates = {
        "upstream_cached_amplitudes_reproduced": cached_discrepancy
        <= float(probe["cached_amplitude_tolerance"]),
        "minimum_local_samples_reached": sample_count
        >= int(probe["minimum_local_samples"]),
        "all_local_values_finite": finite,
        "all_modes_use_standard_N_inverse_normalization": all(
            item.normalization_kind == "structure_factor"
            and item.normalization == rows * columns
            for item in observable_specs
        ),
        "su2_q0_local_values_below_tolerance": su2_q0_passed,
    }
    return {
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "projection": "translation",
        "feature_dim": runtime.feature_dim,
        "maximum_intermediate_q": runtime.maximum_intermediate_q,
        "seed": int(runtime_trial["seed"]),
        "parameter_count": runtime.parameter_count,
        "initial_params_sha256": pytree_sha256(runtime.params),
        "upstream": {
            "trial_payload_sha256": runtime_trial["trial_payload_sha256"],
            "checkpoint_sha256": runtime_checkpoint["checkpoint_sha256"],
            "checkpoint_final_state_sha256": array_sha256(upstream_states),
            "cached_amplitude_max_abs_discrepancy": cached_discrepancy,
        },
        "observable_contract": {
            "operator": "full_SU2_invariant_spin_structure_factor",
            "normalization": "N^-1_sum_ij",
            "mode_names": list(names),
            "multi_q_pair_swapped_amplitudes_shared": True,
            "state_stage": "random_initialization_runtime_checkpoint",
        },
        "sampling": {
            "upstream_final_snapshot_included": True,
            "continuation_sweeps": completed_sweeps,
            "chains": len(states),
            "total_local_samples_per_mode": sample_count,
            "samples_are_autocorrelated": True,
            "continuation_random_stream_independent_of_upstream": True,
            "diagnostics": runtime_workflow._summarize_diagnostics(diagnostics),
            "resume_count": resume_count,
        },
        "observables": {
            "mode_summaries": summaries,
            "q0_maximum_abs_local": q0_maximum,
            "nonfinite_local_count": int(np.count_nonzero(~np.isfinite(local_values))),
            "local_values_shape": list(local_values.shape),
        },
        "cost": {
            "cold_compile_and_initial_observable_seconds": cold_observable_seconds,
            "warm_observable_seconds": warm_observable_seconds,
            "warm_observable_total_seconds": float(sum(warm_observable_seconds)),
            "warm_observable_median_seconds": float(np.median(warm_observable_seconds)),
            "continuation_sampler_seconds": sampler_seconds,
            "continuation_sampler_total_seconds": float(sum(sampler_seconds)),
            "wall_seconds_this_invocation": time.perf_counter() - total_started,
        },
        "gates": gates,
        "all_readiness_gates_passed": all(gates.values()),
        "checkpoint": {
            "path": checkpoint_path.name,
            "status": final_checkpoint["status"],
            "checkpoint_sha256": final_checkpoint["checkpoint_sha256"],
        },
        "claim_boundary": (
            "random-state observable aggregation and cost readiness only; no "
            "energy accuracy, optimization efficiency, magnetic-order, or phase claim"
        ),
    }


def run_spin_sampled_observable_probe(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume the sealed square/triangular 6x6 observable probe."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    manifest = _source_manifest()
    source_hash = canonical_sha256(manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        payload = json.loads(result_path.read_text())
        observed = (
            payload.get("schema_version"),
            payload.get("config_sha256"),
            payload.get("source_manifest_sha256"),
            payload.get("runtime_identity_sha256"),
        )
        expected = (
            PROBE_RESULT_SCHEMA_VERSION,
            config_hash,
            source_hash,
            runtime_hash,
        )
        if observed != expected:
            raise RuntimeError(f"refusing stale sampled observable probe: {result_path}")
        if payload.get("status") == "complete":
            if payload.get("trial_manifest_sha256") != canonical_sha256(
                embedded_trial_manifest(payload)
            ):
                raise RuntimeError(f"sampled observable probe trial seal failed: {result_path}")
            print(f"reused immutable complete observable probe: {result_path}", flush=True)
            return payload
    else:
        payload: dict[str, Any] = {
            "schema_version": PROBE_RESULT_SCHEMA_VERSION,
            "domain": "spin",
            "task": "sampled_full_spin_observable_probe",
            "status": "running",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": repo_relative_path(config_path, root=ROOT),
            "config_sha256": config_hash,
            "config": spec,
            "source_sha256": manifest,
            "source_manifest_sha256": source_hash,
            "runtime_identity": runtime_identity,
            "runtime_identity_sha256": runtime_hash,
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "route_boundary": {
                "historical_project_imports": False,
                "direct_fusion_path_used": False,
                "hilbert_space_enumerated": False,
                "hamiltonian_diagonalization_performed": False,
                "upstream_runtime_v2_required": True,
                "full_su2_structure_factor_used": True,
                "random_state_only": True,
                "energy_or_phase_claimed": False,
            },
            "cases": [],
        }
        atomic_json(result_path, payload)

    completed_cases = {str(case["case_id"]): case for case in payload["cases"]}
    for entry in spec["inputs"]:
        runtime_payload, runtime_case, runtime_result_path = _load_runtime_result(entry)
        bonds, projector = _geometry_runtime(runtime_payload, runtime_case)
        case_id = str(entry["case_id"])
        if case_id in completed_cases:
            case_payload = completed_cases[case_id]
        else:
            case_payload = {
                "case_id": case_id,
                "geometry": runtime_case["geometry"],
                "rows": runtime_case["rows"],
                "columns": runtime_case["columns"],
                "n_sites": runtime_case["n_sites"],
                "j2_over_j1": runtime_case["j2_over_j1"],
                "upstream_result": repo_relative_path(runtime_result_path, root=ROOT),
                "upstream_config_sha256": runtime_payload["config_sha256"],
                "upstream_source_manifest_sha256": runtime_payload[
                    "source_manifest_sha256"
                ],
                "upstream_runtime_identity_sha256": runtime_payload[
                    "runtime_identity_sha256"
                ],
                "upstream_trial_manifest_sha256": runtime_payload[
                    "trial_manifest_sha256"
                ],
                "trials": [],
            }
            payload["cases"].append(case_payload)
            completed_cases[case_id] = case_payload
            atomic_json(result_path, payload)
        completed_arms = {str(trial["arm_id"]) for trial in case_payload["trials"]}
        arms_by_id = {
            str(arm["arm_id"]): arm for arm in runtime_payload["config"]["arms"]
        }
        for runtime_trial in runtime_case["trials"]:
            if str(runtime_trial["arm_id"]) in completed_arms:
                continue
            runtime_checkpoint, _ = _load_runtime_checkpoint(
                runtime_result_path,
                runtime_payload,
                runtime_case,
                runtime_trial,
            )
            arm = arms_by_id[str(runtime_trial["arm_id"])]
            rebuilt = runtime_workflow._initialize_arm(
                arm,
                rows=int(runtime_case["rows"]),
                columns=int(runtime_case["columns"]),
                projector=projector,
            )
            identity = _probe_checkpoint_identity(
                config_hash=config_hash,
                source_hash=source_hash,
                runtime_hash=runtime_hash,
                case_id=case_id,
                trial=runtime_trial,
                runtime_checkpoint=runtime_checkpoint,
            )
            checkpoint_path = output / f"checkpoint_{case_id}_{rebuilt.arm_id}.json"
            trial = _run_arm_probe(
                runtime=rebuilt,
                entry=entry,
                case=runtime_case,
                runtime_trial=runtime_trial,
                runtime_checkpoint=runtime_checkpoint,
                bonds=bonds,
                spec=spec,
                checkpoint_path=checkpoint_path,
                identity=identity,
                overwrite=overwrite,
            )
            trial["trial_payload_sha256"] = trial_payload_sha256(trial)
            case_payload["trials"].append(trial)
            atomic_json(result_path, payload)
            print(
                f"observable-probe {case_id} {trial['arm_id']}: "
                f"samples={trial['sampling']['total_local_samples_per_mode']} "
                f"q0max={trial['observables']['q0_maximum_abs_local']:.3e}",
                flush=True,
            )
    trial_manifest = embedded_trial_manifest(payload)
    payload["trial_manifest"] = trial_manifest
    payload["trial_manifest_sha256"] = canonical_sha256(trial_manifest)
    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_sampled_observable_probe"]
