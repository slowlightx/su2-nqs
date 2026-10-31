"""Resumable sampled-Adam preflight for projected physical-spin trees.

This workflow is the first optimization bridge built on the active physical-
input ClebschTree route.  It intentionally reuses the sealed sampled-runtime
model constructor without changing that workflow.  Persistent fixed-``M2=0``
exchange chains provide training samples, the Hamiltonian acts on demand, and
all microbatches share one global local-energy center.

The registered two-update campaign is a training-readiness preflight.  Its
energies, observables, and timings are not convergence or physics-win claims.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import shutil
import sys
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
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.physical_exchange import (
    initialize_spin_exchange_chains,
    run_exchange_sweeps,
)
from su2_nqs.vmc.physical_sampled_adam import (
    MAX_PHYSICAL_MICROBATCH_SIZE,
    SampledAdamState,
    initialize_sampled_adam,
    sampled_adam_update,
    sampled_vmc_value_and_grad,
)
from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.spin_observables import (
    SpinStructureFactorSpec,
    prepare_spin_structure_factor,
    reciprocal_phase_weights,
    spin_structure_factor_multi_local_values,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TRAINING_RESULT_SCHEMA_VERSION = 1
ALLOWED_BACKBONES = {"u1", "su2"}
CLAIM_BOUNDARY = (
    "sampled-training, checkpoint/resume, and fixed-parameter observable "
    "readiness only; no convergence, efficiency-advantage, magnetic-order, "
    "phase, QSL, or physics-win claim"
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_positive(value: object, name: str) -> float:
    concrete = float(value)
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return concrete


def _portable_config_path(config_path: Path) -> str:
    resolved = config_path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return f"external/{resolved.name}"


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("sampled training requires schema_version=1 and domain='spin'")
    if spec.get("task") != "physical_sampled_training":
        raise ValueError("config task is not physical_sampled_training")
    if spec.get("campaign_stage") not in {"readiness_preflight", "pilot"}:
        raise ValueError("campaign_stage must be readiness_preflight or pilot")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("config claim_boundary must match the enforced code boundary")

    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen_cases: set[str] = set()
    geometries: set[str] = set()
    expected_names = {
        "square": ("q0", "neel", "stripe_x", "stripe_y"),
        "triangular": ("q0", "k", "m1", "m2", "m3"),
    }
    expected_conventions = {
        "square": "row_major_cartesian_column_row",
        "triangular": "row_major_triangular_phase_row_column",
    }
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen_cases:
            raise ValueError("case_id values must be nonempty and unique")
        seen_cases.add(case_id)
        geometry = str(case.get("geometry"))
        if geometry not in expected_names:
            raise ValueError("sampled training supports square and triangular geometry")
        geometries.add(geometry)
        rows = _positive_integer(case.get("rows"), "case.rows")
        columns = _positive_integer(case.get("columns"), "case.columns")
        if rows * columns % 2:
            raise ValueError("fixed-M2=0 training requires an even site count")
        if geometry == "square" and (rows % 2 or columns % 2):
            raise ValueError("square Neel/stripe modes require even dimensions")
        if geometry == "triangular" and (rows % 6 or columns % 6):
            raise ValueError("triangular K/M modes require six-compatible dimensions")
        if case.get("periodic") is not True or case.get("ed") is not False:
            raise ValueError("training cases require periodic=true and ed=false")
        if case.get("support") != "fixed_m0":
            raise ValueError("training cases require fixed_m0 support")
        if case.get("coordinate_convention") != expected_conventions[geometry]:
            raise ValueError("coordinate convention does not match geometry")
        if not math.isfinite(float(case.get("j2_over_j1", np.nan))):
            raise ValueError("j2_over_j1 must be finite")
        modes = case.get("modes")
        if not isinstance(modes, list) or tuple(
            str(mode.get("name")) for mode in modes
        ) != expected_names[geometry]:
            raise ValueError("cases must register q0 and the geometry-specific modes")
        for mode in modes:
            vector = np.asarray(mode.get("wavevector"), dtype=np.float64)
            if vector.shape != (2,) or np.any(~np.isfinite(vector)):
                raise ValueError("every wavevector must be a finite two-vector")
    if geometries != {"square", "triangular"}:
        raise ValueError("the registered preflight requires square and triangular cases")

    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection.get("kind") != "translation":
        raise ValueError("sampled training requires translation projection")
    _positive_integer(projection.get("orbit_chunk_size"), "orbit_chunk_size")
    if not isinstance(projection.get("rematerialize_orbit"), bool):
        raise TypeError("rematerialize_orbit must be boolean")

    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty list")
    backbones: set[str] = set()
    seen_arms: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise TypeError("every arm must be an object")
        arm_id = str(arm.get("arm_id", ""))
        if not arm_id or arm_id in seen_arms:
            raise ValueError("arm_id values must be nonempty and unique")
        seen_arms.add(arm_id)
        backbone = str(arm.get("backbone"))
        if backbone not in ALLOWED_BACKBONES:
            raise ValueError("training backbones are restricted to u1 and su2")
        backbones.add(backbone)
        if arm.get("projection") != "translation":
            raise ValueError("every arm must use translation projection")
        if isinstance(arm.get("seed"), bool) or not isinstance(arm.get("seed"), int):
            raise TypeError("every arm seed must be an integer")
        model = arm.get("model")
        if not isinstance(model, dict):
            raise TypeError("every arm needs a model object")
        _positive_integer(model.get("feature_dim"), "feature_dim")
        _positive_integer(model.get("maximum_intermediate_q"), "maximum_intermediate_q")
        _finite_positive(model.get("initialization_scale"), "initialization_scale")
    if backbones != ALLOWED_BACKBONES:
        raise ValueError("sampled training requires the matched U1/SU2 pair")

    sampler = spec.get("sampler")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for name in (
        "chains",
        "initial_burn_sweeps",
        "decorrelation_sweeps_per_update",
        "proposals_per_sweep",
    ):
        _positive_integer(sampler.get(name), f"sampler.{name}")
    if isinstance(sampler.get("seed"), bool) or not isinstance(sampler.get("seed"), int):
        raise TypeError("sampler.seed must be an integer")
    bond_probability = float(sampler.get("bond_proposal_probability", np.nan))
    if not 0.0 <= bond_probability <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0,1]")

    optimizer = spec.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "local_adam":
        raise ValueError("optimizer must be local_adam")
    for name in ("updates", "microbatch_size", "checkpoint_every_updates"):
        _positive_integer(optimizer.get(name), f"optimizer.{name}")
    microbatch = int(optimizer["microbatch_size"])
    chains = int(sampler["chains"])
    if chains % microbatch or microbatch > MAX_PHYSICAL_MICROBATCH_SIZE:
        raise ValueError("microbatch_size must divide chains and remain at most eight")
    for name in ("learning_rate", "epsilon", "maximum_relative_update_norm"):
        _finite_positive(optimizer.get(name), f"optimizer.{name}")
    for name in ("beta1", "beta2"):
        value = float(optimizer.get(name, np.nan))
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"optimizer.{name} must lie in [0,1)")

    measurement = spec.get("measurement")
    if not isinstance(measurement, dict):
        raise TypeError("measurement must be an object")
    for name in (
        "every_updates",
        "burn_sweeps",
        "measurement_sweeps",
        "observable_batch_size",
    ):
        _positive_integer(measurement.get(name), f"measurement.{name}")
    observable_batch = int(measurement["observable_batch_size"])
    if chains % observable_batch:
        raise ValueError("observable_batch_size must divide the chain count")
    if isinstance(measurement.get("seed"), bool) or not isinstance(
        measurement.get("seed"), int
    ):
        raise TypeError("measurement.seed must be an integer")
    _finite_positive(
        measurement.get("su2_q0_local_tolerance"),
        "measurement.su2_q0_local_tolerance",
    )


def _geometry(
    case: dict[str, object], projection: dict[str, object]
) -> tuple[tuple[tuple[int, int, float], ...], tuple[tuple[int, ...], ...], PhysicalSpatialProjector]:
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
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"unknown geometry: {case['geometry']!r}")
    projector = PhysicalSpatialProjector(
        group,
        orbit_chunk_size=int(projection["orbit_chunk_size"]),
        rematerialize_orbit=bool(projection["rematerialize_orbit"]),
    )
    return bonds, group, projector


def _mode_specs(
    case: dict[str, object]
) -> tuple[tuple[str, ...], tuple[SpinStructureFactorSpec, ...]]:
    rows, columns = int(case["rows"]), int(case["columns"])
    if case["geometry"] == "square":
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
    names = tuple(str(mode["name"]) for mode in case["modes"])
    specs = tuple(
        prepare_spin_structure_factor(
            reciprocal_phase_weights(coordinates, mode["wavevector"])
        )
        for mode in case["modes"]
    )
    if any(item.normalization_kind != "structure_factor" for item in specs):
        raise RuntimeError("training observables must use standard N^-1 S(q)")
    return names, specs


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _sum_ledgers(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for row in rows:
        for key, value in row.items():
            result[key] = result.get(key, 0) + value
    return result


def _float_diagnostics(values: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def _compiled_kernels(
    runtime: runtime_workflow._ArmRuntime,
    bonds: tuple[tuple[int, int, float], ...],
    observable_specs: tuple[SpinStructureFactorSpec, ...],
) -> tuple[Callable, Callable, Callable, Callable]:
    """Build static-shape sampler, Hpsi, and full-S(q) kernels for one arm."""

    exchange_edges = tuple((first, second) for first, second, _ in bonds)

    def advance_uncached(
        candidate: object,
        key: jax.Array,
        states: jax.Array,
        attempts: int,
        bond_probability: float,
    ):
        return run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=runtime.projected_amplitude,
            particle_kind="spin",
            sweeps=attempts,
            amplitudes=None,
            exchange_edges=exchange_edges,
            bond_proposal_probability=bond_probability,
        )

    def advance_cached(
        candidate: object,
        key: jax.Array,
        states: jax.Array,
        amplitudes: jax.Array,
        attempts: int,
        bond_probability: float,
    ):
        return run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=runtime.projected_amplitude,
            particle_kind="spin",
            sweeps=attempts,
            amplitudes=amplitudes,
            exchange_edges=exchange_edges,
            bond_proposal_probability=bond_probability,
        )

    def projected_for_hpsi(candidate, state, _config):
        return runtime.projected_amplitude(candidate, state)

    def hpsi(candidate: object, batch: tuple[jax.Array, ...]):
        samples, connected, elements, valid = batch
        return spin_physical_hpsi(
            candidate,
            samples,
            connected,
            elements,
            valid,
            runtime.model_config,
            amplitude_fn=projected_for_hpsi,
        )

    def observable(candidate: object, samples: jax.Array):
        return spin_structure_factor_multi_local_values(
            candidate,
            samples,
            observable_specs,
            amplitude_fn=runtime.projected_amplitude,
        )

    # attempts and bond_probability are static campaign choices.  Marking them
    # static prevents accidental traced-to-Python conversions inside proposal
    # validation while still allowing params/chains to evolve.
    return (
        jax.jit(advance_uncached, static_argnums=(3, 4)),
        jax.jit(advance_cached, static_argnums=(4, 5)),
        jax.jit(hpsi),
        jax.jit(observable),
    )


def _hpsi_batches(
    states: np.ndarray,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    padded_width: int,
    microbatch_size: int,
) -> tuple[tuple[tuple[jax.Array, ...], ...], np.ndarray, float]:
    started = time.perf_counter()
    connected, elements, valid, widths = padded_spin_connectivity(
        states, bonds, padded_width=padded_width
    )
    elapsed = time.perf_counter() - started
    batches = []
    for start in range(0, len(states), microbatch_size):
        stop = start + microbatch_size
        batches.append(
            tuple(
                jnp.asarray(value[start:stop])
                for value in (states, connected, elements, valid)
            )
        )
    return tuple(batches), widths, elapsed


def _local_energy_chunks(
    params: object,
    states: np.ndarray,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    padded_width: int,
    microbatch_size: int,
    hpsi_fn: Callable,
) -> tuple[np.ndarray, float, float, int]:
    batches, _widths, connectivity_seconds = _hpsi_batches(
        states,
        bonds,
        padded_width=padded_width,
        microbatch_size=microbatch_size,
    )
    started = time.perf_counter()
    values: list[np.ndarray] = []
    for batch in batches:
        centers, actions = hpsi_fn(params, batch)
        jax.block_until_ready((centers, actions))
        values.append(
            np.asarray(actions, dtype=np.complex128)
            / np.asarray(centers, dtype=np.complex128)
        )
    elapsed = time.perf_counter() - started
    local = np.concatenate(values)
    return (
        local,
        connectivity_seconds,
        elapsed,
        int(np.count_nonzero(~np.isfinite(local))),
    )


def _fixed_parameter_measurement(
    *,
    runtime: runtime_workflow._ArmRuntime,
    params: object,
    training_states: np.ndarray,
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    names: tuple[str, ...],
    spec: dict[str, object],
    update: int,
    advance_uncached: Callable,
    advance_cached: Callable,
    hpsi_fn: Callable,
    observable_fn: Callable,
) -> tuple[dict[str, object], dict[str, int | float]]:
    """Measure energy and full S(q) without mutating the training trajectory."""

    sampler = dict(spec["sampler"])
    measurement = dict(spec["measurement"])
    chains = int(sampler["chains"])
    proposals = int(sampler["proposals_per_sweep"])
    bond_probability = float(sampler["bond_proposal_probability"])
    padded_width = len(bonds) + 1
    microbatch = int(spec["optimizer"]["microbatch_size"])
    observable_batch = int(measurement["observable_batch_size"])
    fixed_params_hash = pytree_sha256(params)
    original_training_hash = array_sha256(training_states)

    fold_value = int(
        canonical_sha256(
            {
                "measurement_seed": int(measurement["seed"]),
                "case_id": case["case_id"],
                "arm_id": runtime.arm_id,
                "update": update,
            }
        )[:8],
        16,
    )
    key = jax.random.fold_in(
        jax.random.PRNGKey(int(measurement["seed"])), fold_value
    )
    states = jnp.asarray(np.array(training_states, copy=True), dtype=jnp.int32)
    burn_attempts = int(measurement["burn_sweeps"]) * proposals
    sampler_started = time.perf_counter()
    states, amplitudes, key, burn_diagnostics = advance_uncached(
        params, key, states, burn_attempts, bond_probability
    )
    jax.block_until_ready((states, amplitudes, key, burn_diagnostics))
    sampler_seconds = time.perf_counter() - sampler_started
    diagnostics = [_float_diagnostics(burn_diagnostics)]
    local_series: list[np.ndarray] = []
    observable_series: list[np.ndarray] = []
    connectivity_seconds = 0.0
    hpsi_seconds = 0.0
    observable_seconds = 0.0
    for _ in range(int(measurement["measurement_sweeps"])):
        sampler_started = time.perf_counter()
        states, amplitudes, key, row = advance_cached(
            params, key, states, amplitudes, proposals, bond_probability
        )
        jax.block_until_ready((states, amplitudes, key, row))
        sampler_seconds += time.perf_counter() - sampler_started
        diagnostics.append(_float_diagnostics(row))
        host_states = np.asarray(states, dtype=np.int32)
        local, host_seconds, device_seconds, nonfinite = _local_energy_chunks(
            params,
            host_states,
            bonds,
            padded_width=padded_width,
            microbatch_size=microbatch,
            hpsi_fn=hpsi_fn,
        )
        if nonfinite:
            raise FloatingPointError("fixed-parameter measurement has non-finite energy")
        connectivity_seconds += host_seconds
        hpsi_seconds += device_seconds
        local_series.append(local)
        chunks: list[np.ndarray] = []
        started = time.perf_counter()
        for start in range(0, chains, observable_batch):
            values = observable_fn(
                params, jnp.asarray(host_states[start : start + observable_batch])
            )
            jax.block_until_ready(values)
            chunks.append(np.asarray(values, dtype=np.complex128))
        observable_seconds += time.perf_counter() - started
        observable_series.append(np.concatenate(chunks, axis=0))

    if fixed_params_hash != pytree_sha256(params):
        raise RuntimeError("fixed-parameter measurement mutated model parameters")
    if original_training_hash != array_sha256(training_states):
        raise RuntimeError("fixed-parameter measurement mutated training chains")
    local_values = np.stack(local_series, axis=0)
    mode_values = np.stack(observable_series, axis=0)
    if np.any(~np.isfinite(mode_values)):
        raise FloatingPointError("fixed-parameter measurement has non-finite S(q)")
    flat_local = local_values.reshape(-1)
    energy_mean = complex(np.mean(flat_local))
    energy_variance = float(
        np.mean(np.abs(flat_local - energy_mean) ** 2).real
    )
    chain_energy_means = np.mean(local_values.real, axis=0)
    energy_chain_mcse = float(
        np.std(chain_energy_means, ddof=1) / np.sqrt(len(chain_energy_means))
    )
    modes: dict[str, object] = {}
    for index, name in enumerate(names):
        concrete = mode_values[:, :, index]
        chain_means = np.mean(concrete.real, axis=0)
        modes[name] = {
            "mean_real": float(np.mean(concrete.real)),
            "mean_imaginary": float(np.mean(concrete.imag)),
            "maximum_abs_imaginary_local": float(np.max(np.abs(concrete.imag))),
            "chain_mean_mcse": float(
                np.std(chain_means, ddof=1) / np.sqrt(len(chain_means))
            ),
            "sample_count": int(concrete.size),
        }
    q0_maximum = float(np.max(np.abs(mode_values[:, :, names.index("q0")])))
    diagnostic_summary = runtime_workflow._summarize_diagnostics(diagnostics)
    result = {
        "update": int(update),
        "fixed_params_sha256": fixed_params_hash,
        "params_frozen_throughout_block": True,
        "training_chains_copied_and_not_mutated": True,
        "independent_measurement_random_stream": True,
        "energy_mean": float(energy_mean.real),
        "energy_per_site": float(energy_mean.real / (int(case["rows"]) * int(case["columns"]))),
        "energy_imaginary_abs": float(abs(energy_mean.imag)),
        "energy_variance": energy_variance,
        "energy_chain_mean_mcse": energy_chain_mcse,
        "full_spin_structure_factor": {
            "operator": "full_SU2_invariant_spin_structure_factor",
            "normalization": "N^-1_sum_ij",
            "mode_summaries": modes,
            "q0_maximum_abs_local": q0_maximum,
            "multi_q_pair_swapped_amplitudes_shared": True,
        },
        "sampler_diagnostics": diagnostic_summary,
        "samples_are_autocorrelated": True,
        "ess_combined_across_optimizer_steps": False,
        "timing": {
            "sampler_seconds": sampler_seconds,
            "connectivity_seconds_host": connectivity_seconds,
            "hpsi_seconds": hpsi_seconds,
            "full_spin_observable_seconds": observable_seconds,
        },
    }

    n_sites = int(case["rows"]) * int(case["columns"])
    measurement_sweeps = int(measurement["measurement_sweeps"])
    sample_count = measurement_sweeps * chains
    pair_count = n_sites * (n_sites - 1) // 2
    sampling_projected = chains * (
        1
        + proposals
        * (int(measurement["burn_sweeps"]) + measurement_sweeps)
    )
    hpsi_projected = sample_count * (1 + padded_width)
    observable_projected = sample_count * (1 + pair_count)
    ledger: dict[str, int | float] = {
        "measurement_sampler_projected_amplitude_evaluations": sampling_projected,
        "measurement_hpsi_projected_amplitude_evaluations": hpsi_projected,
        "full_spin_observable_projected_amplitude_evaluations": observable_projected,
        "measurement_base_tree_orbit_image_work_proxy": (
            (sampling_projected + hpsi_projected + observable_projected)
            * runtime.projector.group_order
        ),
        "fixed_parameter_energy_samples": sample_count,
        "fixed_parameter_full_spin_samples": sample_count,
        "measurement_sampler_seconds": sampler_seconds,
        "measurement_connectivity_seconds_host": connectivity_seconds,
        "measurement_hpsi_seconds": hpsi_seconds,
        "full_spin_observable_seconds": observable_seconds,
    }
    return result, ledger


def _run_identity(
    *,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    case: dict[str, object],
    arm: dict[str, object],
    runtime: runtime_workflow._ArmRuntime,
    bonds: tuple[tuple[int, int, float], ...],
    group: tuple[tuple[int, ...], ...],
    initial_chain_states_sha256: str,
    spec: dict[str, object],
) -> dict[str, object]:
    rows, columns = int(case["rows"]), int(case["columns"])
    return {
        "schema": "physical_sampled_spin_adam_v1",
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "runtime_identity_sha256": runtime_hash,
        "case_id": case["case_id"],
        "geometry": case["geometry"],
        "rows": rows,
        "columns": columns,
        "j2_over_j1": float(case["j2_over_j1"]),
        "bonds_sha256": canonical_sha256(bonds),
        "translation_group_sha256": canonical_sha256(group),
        "morton_order": list(morton_order(rows, columns)),
        "arm_id": arm["arm_id"],
        "backbone": arm["backbone"],
        "model": dict(arm["model"]),
        "model_seed": int(arm["seed"]),
        "initial_params_sha256": pytree_sha256(runtime.params),
        "parameter_count": runtime.parameter_count,
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "projection": dict(spec["projection"]),
        "sampler": dict(spec["sampler"]),
        "optimizer": dict(spec["optimizer"]),
        "measurement": dict(spec["measurement"]),
        "padded_connectivity_width": len(bonds) + 1,
        "amplitude_cache_checkpoint_policy": "absent_recompute_after_every_update",
    }


def _checkpoint_metadata(
    *,
    params: object,
    optimizer_state: SampledAdamState,
    states: np.ndarray,
    sampling_key: np.ndarray,
    initial_chain_states_sha256: str,
    initial_params_sha256: str,
    initial_burn_diagnostics: dict[str, float],
    initial_burn_work_ledger: dict[str, int | float],
    resume_count: int,
) -> dict[str, object]:
    return {
        "chain_states": np.asarray(states, dtype=np.int32),
        "chain_states_sha256": array_sha256(np.asarray(states, dtype=np.int32)),
        "sampling_key": np.asarray(sampling_key, dtype=np.uint32),
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "initial_params_sha256": initial_params_sha256,
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "initial_burn_completed": True,
        "initial_burn_diagnostics": initial_burn_diagnostics,
        "initial_burn_work_ledger": initial_burn_work_ledger,
        "resume_count": int(resume_count),
        "amplitude_cache_present": False,
        "amplitude_cache_policy": (
            "never checkpointed; recomputed from checkpoint params and chains"
        ),
    }


def _load_training_checkpoint(
    directory: Path,
    identity: dict[str, object],
    *,
    expected_initial_chain_hash: str,
    expected_initial_params_hash: str,
    n_sites: int,
    n_chains: int,
) -> dict[str, object] | None:
    if not directory.is_dir():
        return None
    payload = load_latest_resumable_checkpoint(
        directory, expected_run_identity=identity
    )
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError("training checkpoint metadata is invalid")
    if metadata.get("amplitude_cache_present") is not False or "amplitudes" in metadata:
        raise RuntimeError("training checkpoint must not retain amplitude caches")
    if metadata.get("initial_chain_states_sha256") != expected_initial_chain_hash:
        raise RuntimeError("training checkpoint initial-chain identity mismatch")
    if metadata.get("initial_params_sha256") != expected_initial_params_hash:
        raise RuntimeError("training checkpoint initial-parameter identity mismatch")
    states = np.asarray(metadata.get("chain_states"), dtype=np.int32)
    if states.shape != (n_chains, n_sites):
        raise RuntimeError("training checkpoint has the wrong chain shape")
    if np.any(np.sum(states, axis=1) != n_sites // 2):
        raise RuntimeError("training checkpoint left fixed-M2=0 support")
    if metadata.get("chain_states_sha256") != array_sha256(states):
        raise RuntimeError("training checkpoint chain-state seal mismatch")
    if metadata.get("params_sha256") != pytree_sha256(payload["params"]):
        raise RuntimeError("training checkpoint parameter seal mismatch")
    optimizer_state = payload["optimizer_state"]
    if not isinstance(optimizer_state, SampledAdamState):
        raise RuntimeError("training checkpoint optimizer type mismatch")
    if metadata.get("optimizer_state_sha256") != pytree_sha256(optimizer_state):
        raise RuntimeError("training checkpoint optimizer seal mismatch")
    if optimizer_state.step != int(payload["completed_steps"]):
        raise RuntimeError("training checkpoint optimizer/update counters disagree")
    if int(payload["next_sample_index"]) != int(payload["completed_steps"]) * n_chains:
        raise RuntimeError("training checkpoint sample counter is inconsistent")
    return payload


def _run_training_arm(
    *,
    runtime: runtime_workflow._ArmRuntime,
    arm: dict[str, object],
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    group: tuple[tuple[int, ...], ...],
    spec: dict[str, object],
    checkpoint_directory: Path,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    overwrite: bool,
    maximum_updates_this_invocation: int | None = None,
) -> dict[str, object]:
    """Run, interrupt, or resume one arm; the optional cap is for canary tests."""

    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    sampler = dict(spec["sampler"])
    optimizer = dict(spec["optimizer"])
    measurement = dict(spec["measurement"])
    chains = int(sampler["chains"])
    proposals = int(sampler["proposals_per_sweep"])
    bond_probability = float(sampler["bond_proposal_probability"])
    microbatch = int(optimizer["microbatch_size"])
    total_updates = int(optimizer["updates"])
    padded_width = len(bonds) + 1
    names, observable_specs = _mode_specs(case)
    advance_uncached, advance_cached, hpsi_fn, observable_fn = _compiled_kernels(
        runtime, bonds, observable_specs
    )

    seed_key = jax.random.PRNGKey(int(sampler["seed"]))
    chain_key, sampling_key_device = jax.random.split(seed_key)
    initial_states_device = initialize_spin_exchange_chains(
        chain_key,
        n_chains=chains,
        n_sites=n_sites,
        n_up=n_sites // 2,
    )
    initial_states = np.asarray(initial_states_device, dtype=np.int32)
    initial_chain_hash = array_sha256(initial_states)
    initial_params_hash = pytree_sha256(runtime.params)
    identity = _run_identity(
        config_hash=config_hash,
        source_hash=source_hash,
        runtime_hash=runtime_hash,
        case=case,
        arm=arm,
        runtime=runtime,
        bonds=bonds,
        group=group,
        initial_chain_states_sha256=initial_chain_hash,
        spec=spec,
    )
    if overwrite and checkpoint_directory.is_dir():
        shutil.rmtree(checkpoint_directory)
    checkpoint = None if overwrite else _load_training_checkpoint(
        checkpoint_directory,
        identity,
        expected_initial_chain_hash=initial_chain_hash,
        expected_initial_params_hash=initial_params_hash,
        n_sites=n_sites,
        n_chains=chains,
    )
    checkpoint_seconds: list[float] = []
    latest_generation: Path | None = None
    invocation_started = time.perf_counter()

    if checkpoint is None:
        params = runtime.params
        optimizer_state = initialize_sampled_adam(params)
        burn_attempts = int(sampler["initial_burn_sweeps"]) * proposals
        started = time.perf_counter()
        states_device, amplitudes_device, sampling_key_device, burn_device = (
            advance_uncached(
                params,
                sampling_key_device,
                initial_states_device,
                burn_attempts,
                bond_probability,
            )
        )
        jax.block_until_ready(
            (states_device, amplitudes_device, sampling_key_device, burn_device)
        )
        burn_seconds = time.perf_counter() - started
        states = np.asarray(states_device, dtype=np.int32)
        sampling_key = np.asarray(sampling_key_device, dtype=np.uint32)
        # This cache belongs to the initial params, but no optimizer update has
        # happened and it is not needed again.  Drop it before the first save.
        del amplitudes_device
        initial_burn_diagnostics = _float_diagnostics(burn_device)
        burn_projected = chains * (1 + burn_attempts)
        initial_burn_work: dict[str, int | float] = {
            "sampler_proposal_attempts": chains * burn_attempts,
            "sampler_projected_amplitude_evaluations": burn_projected,
            "base_tree_orbit_image_work_proxy": (
                burn_projected * runtime.projector.group_order
            ),
            "sampler_seconds": burn_seconds,
        }
        history: list[dict[str, object]] = []
        completed_updates = 0
        resume_count = 0
        metadata = _checkpoint_metadata(
            params=params,
            optimizer_state=optimizer_state,
            states=states,
            sampling_key=sampling_key,
            initial_chain_states_sha256=initial_chain_hash,
            initial_params_sha256=initial_params_hash,
            initial_burn_diagnostics=initial_burn_diagnostics,
            initial_burn_work_ledger=initial_burn_work,
            resume_count=resume_count,
        )
        started = time.perf_counter()
        latest_generation = save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state=optimizer_state,
            completed_steps=0,
            next_sample_index=0,
            run_identity=identity,
            history=history,
            metadata=metadata,
        )
        checkpoint_seconds.append(time.perf_counter() - started)
    else:
        params = checkpoint["params"]
        optimizer_state = checkpoint["optimizer_state"]
        completed_updates = int(checkpoint["completed_steps"])
        history = list(checkpoint["history"])
        checkpoint_metadata = dict(checkpoint["metadata"])
        states = np.asarray(checkpoint_metadata["chain_states"], dtype=np.int32)
        sampling_key = np.asarray(
            checkpoint_metadata["sampling_key"], dtype=np.uint32
        )
        initial_burn_diagnostics = dict(
            checkpoint_metadata["initial_burn_diagnostics"]
        )
        initial_burn_work = dict(checkpoint_metadata["initial_burn_work_ledger"])
        resume_count = int(checkpoint_metadata["resume_count"]) + 1
        latest_generation = Path(str(checkpoint["checkpoint_generation_path"]))

    start_update = completed_updates
    invocation_limit = total_updates
    if maximum_updates_this_invocation is not None:
        if maximum_updates_this_invocation < 0:
            raise ValueError("maximum_updates_this_invocation cannot be negative")
        invocation_limit = min(
            total_updates, completed_updates + maximum_updates_this_invocation
        )

    for update_index in range(completed_updates + 1, invocation_limit + 1):
        params_before_hash = pytree_sha256(params)
        previous_states = states.copy()
        sampling_attempts = (
            int(sampler["decorrelation_sweeps_per_update"]) * proposals
        )
        started = time.perf_counter()
        states_device, amplitudes_device, key_device, sampler_device = advance_uncached(
            params,
            jnp.asarray(sampling_key),
            jnp.asarray(states),
            sampling_attempts,
            bond_probability,
        )
        jax.block_until_ready(
            (states_device, amplitudes_device, key_device, sampler_device)
        )
        sampler_seconds = time.perf_counter() - started
        states = np.asarray(states_device, dtype=np.int32)
        sampling_key = np.asarray(key_device, dtype=np.uint32)
        sampler_diagnostics = _float_diagnostics(sampler_device)
        chain_move_fraction = float(np.mean(np.any(states != previous_states, axis=1)))

        batches, widths, connectivity_seconds = _hpsi_batches(
            states,
            bonds,
            padded_width=padded_width,
            microbatch_size=microbatch,
        )
        started = time.perf_counter()
        gradient_result = sampled_vmc_value_and_grad(
            params, batches, hpsi_fn=hpsi_fn
        )
        gradient_seconds = time.perf_counter() - started
        # Cached amplitudes were evaluated under params_before and are invalid
        # as soon as Adam commits params_after.  Never retain them in state.
        del amplitudes_device
        started = time.perf_counter()
        params, optimizer_state, adam_diagnostics = sampled_adam_update(
            params,
            gradient_result.gradient,
            optimizer_state,
            learning_rate=float(optimizer["learning_rate"]),
            max_relative_update=float(optimizer["maximum_relative_update_norm"]),
            beta1=float(optimizer["beta1"]),
            beta2=float(optimizer["beta2"]),
            epsilon=float(optimizer["epsilon"]),
        )
        jax.block_until_ready(params)
        optimizer_seconds = time.perf_counter() - started
        params_after_hash = pytree_sha256(params)
        if params_after_hash == params_before_hash:
            raise RuntimeError("Adam accepted an update without changing parameters")

        fixed_measurement = None
        measurement_work: dict[str, int | float] = {}
        if (
            update_index % int(measurement["every_updates"]) == 0
            or update_index == total_updates
        ):
            fixed_measurement, measurement_work = _fixed_parameter_measurement(
                runtime=runtime,
                params=params,
                training_states=states,
                case=case,
                bonds=bonds,
                names=names,
                spec=spec,
                update=update_index,
                advance_uncached=advance_uncached,
                advance_cached=advance_cached,
                hpsi_fn=hpsi_fn,
                observable_fn=observable_fn,
            )

        sampler_projected = chains * (1 + sampling_attempts)
        hpsi_projected = 2 * chains * (1 + padded_width)
        step_work: dict[str, int | float] = {
            "training_samples": chains,
            "sampler_proposal_attempts": chains * sampling_attempts,
            "sampler_projected_amplitude_evaluations": sampler_projected,
            "hpsi_two_pass_projected_amplitude_evaluations": hpsi_projected,
            "gradient_backward_microbatch_invocations": len(batches),
            "hpsi_actual_connected_configurations_per_pass": int(np.sum(widths)),
            "hpsi_padded_connected_configurations_per_pass": chains * padded_width,
            "base_tree_orbit_image_work_proxy": (
                (sampler_projected + hpsi_projected)
                * runtime.projector.group_order
            ),
            "sampler_seconds": sampler_seconds,
            "connectivity_seconds_host": connectivity_seconds,
            "gradient_two_pass_seconds": gradient_seconds,
            "optimizer_seconds": optimizer_seconds,
            **measurement_work,
        }
        row: dict[str, object] = {
            "update": update_index,
            "params_before_sha256": params_before_hash,
            "params_after_sha256": params_after_hash,
            "chain_states_sha256": array_sha256(states),
            "sampler": {
                **sampler_diagnostics,
                "chain_snapshot_move_fraction": chain_move_fraction,
            },
            "pre_update_sampled_energy": gradient_result.energy,
            "pre_update_sampled_energy_per_site": gradient_result.energy / n_sites,
            "pre_update_sampled_variance": gradient_result.variance,
            "pre_update_energy_imaginary_abs": gradient_result.energy_imaginary_abs,
            "minimum_center_magnitude": gradient_result.minimum_center_magnitude,
            "maximum_local_energy_magnitude": (
                gradient_result.maximum_local_energy_magnitude
            ),
            "nonfinite_local_count": gradient_result.nonfinite_local_count,
            "gradient": {
                "norm": gradient_result.gradient_norm,
                "sample_count": gradient_result.sample_count,
                "microbatch_count": gradient_result.microbatch_count,
                "maximum_microbatch_size": gradient_result.maximum_microbatch_size,
                "global_local_energy_center_shared_across_microbatches": True,
            },
            "adam": asdict(adam_diagnostics),
            "post_update_amplitude_cache": "discarded_and_not_checkpointed",
            "fixed_parameter_measurement": fixed_measurement,
            "work_ledger": step_work,
            "timing_scope": (
                "step-local; first encountered shape includes cold compilation"
            ),
        }
        history.append(row)
        completed_updates = update_index

        if (
            update_index % int(optimizer["checkpoint_every_updates"]) == 0
            or update_index == total_updates
            or update_index == invocation_limit
        ):
            metadata = _checkpoint_metadata(
                params=params,
                optimizer_state=optimizer_state,
                states=states,
                sampling_key=sampling_key,
                initial_chain_states_sha256=initial_chain_hash,
                initial_params_sha256=initial_params_hash,
                initial_burn_diagnostics=initial_burn_diagnostics,
                initial_burn_work_ledger=initial_burn_work,
                resume_count=resume_count,
            )
            started = time.perf_counter()
            latest_generation = save_resumable_checkpoint_generation(
                checkpoint_directory,
                params,
                optimizer_state=optimizer_state,
                completed_steps=completed_updates,
                next_sample_index=completed_updates * chains,
                run_identity=identity,
                history=history,
                metadata=metadata,
            )
            checkpoint_seconds.append(time.perf_counter() - started)

    complete = completed_updates == total_updates
    state_result: dict[str, object] = {
        "status": "complete" if complete else "running",
        "completed_updates": completed_updates,
        "requested_updates": total_updates,
        "updates_completed_this_invocation": completed_updates - start_update,
        "resume_count": resume_count,
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "chain_states_sha256": array_sha256(states),
        "sampling_key_sha256": array_sha256(np.asarray(sampling_key)),
        "history": history,
        "wall_seconds_this_invocation": time.perf_counter() - invocation_started,
        "checkpoint_seconds_this_invocation": checkpoint_seconds,
        "timing_claim_eligible": resume_count == 0,
    }
    if not complete:
        return state_result

    final_checkpoint = load_latest_resumable_checkpoint(
        checkpoint_directory, expected_run_identity=identity
    )
    final_metadata = dict(final_checkpoint["metadata"])
    if final_metadata.get("amplitude_cache_present") is not False:
        raise RuntimeError("final training checkpoint retained an amplitude cache")
    if int(final_checkpoint["completed_steps"]) != total_updates:
        raise RuntimeError("final training checkpoint is incomplete")
    if latest_generation is None:
        latest_generation = Path(str(final_checkpoint["checkpoint_generation_path"]))
    cumulative_work = _sum_ledgers(
        [initial_burn_work]
        + [dict(row["work_ledger"]) for row in history]
    )
    all_measurements = [
        row["fixed_parameter_measurement"]
        for row in history
        if row["fixed_parameter_measurement"] is not None
    ]
    q0_tolerance = float(measurement["su2_q0_local_tolerance"])
    gates = {
        "all_requested_updates_completed": len(history) == total_updates,
        "all_local_energies_and_gradients_finite": all(
            int(row["nonfinite_local_count"]) == 0
            and all(
                math.isfinite(float(value))
                for value in (
                    row["pre_update_sampled_energy"],
                    row["pre_update_sampled_variance"],
                    row["gradient"]["norm"],
                    row["minimum_center_magnitude"],
                    row["maximum_local_energy_magnitude"],
                )
            )
            for row in history
        ),
        "all_relative_updates_within_registered_limit": all(
            float(row["adam"]["relative_update_norm"])
            <= float(optimizer["maximum_relative_update_norm"])
            for row in history
        ),
        "all_updates_changed_parameters": all(
            row["params_before_sha256"] != row["params_after_sha256"]
            for row in history
        ),
        "all_updates_move_at_least_one_chain": all(
            float(row["sampler"]["chain_snapshot_move_fraction"]) > 0.0
            for row in history
        ),
        "fixed_parameter_measurements_present": bool(all_measurements),
        "fixed_parameter_measurements_finite": all(
            math.isfinite(float(item["energy_mean"]))
            and math.isfinite(float(item["energy_variance"]))
            and all(
                math.isfinite(float(summary["mean_real"]))
                for summary in item["full_spin_structure_factor"][
                    "mode_summaries"
                ].values()
            )
            for item in all_measurements
        ),
        "su2_q0_local_below_registered_tolerance": (
            all(
                float(item["full_spin_structure_factor"]["q0_maximum_abs_local"])
                <= q0_tolerance
                for item in all_measurements
            )
            if runtime.backbone == "su2"
            else True
        ),
        "amplitude_cache_absent_from_checkpoint": (
            final_metadata.get("amplitude_cache_present") is False
            and "amplitudes" not in final_metadata
        ),
        "microbatch_global_center_contract": all(
            bool(row["gradient"][
                "global_local_energy_center_shared_across_microbatches"
            ])
            for row in history
        ),
        "optimizer_step_matches_update_count": optimizer_state.step == total_updates,
    }
    generation_path = Path(str(final_checkpoint["checkpoint_generation_path"]))
    trial: dict[str, object] = {
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "projection": "translation",
        "feature_dim": runtime.feature_dim,
        "maximum_intermediate_q": runtime.maximum_intermediate_q,
        "seed": int(arm["seed"]),
        "parameter_count": runtime.parameter_count,
        "initial_params_sha256": initial_params_hash,
        "final_params_sha256": pytree_sha256(params),
        "initial_chain_states_sha256": initial_chain_hash,
        "final_chain_states_sha256": array_sha256(states),
        "translation_group_order": runtime.projector.group_order,
        "padded_connectivity_width": padded_width,
        "optimizer": {
            **optimizer,
            "completed_updates": completed_updates,
        },
        "sampler": {
            **sampler,
            "initial_burn_diagnostics": initial_burn_diagnostics,
            "resume_count": resume_count,
        },
        "history": history,
        "final_fixed_parameter_measurement": all_measurements[-1],
        "work_ledger": {
            **cumulative_work,
            "checkpoint_seconds_this_invocation": float(sum(checkpoint_seconds)),
            "host_peak_rss_bytes_process_high_watermark_proxy": _peak_rss_bytes(),
            "parameter_count_is_not_used_as_a_work_substitute": True,
        },
        "checkpoint": {
            "series_directory": checkpoint_directory.name,
            "latest_generation": generation_path.name,
            "latest_generation_sha256": file_sha256(generation_path),
            "run_identity_sha256": canonical_json_sha256(identity),
            "completed_updates": int(final_checkpoint["completed_steps"]),
            "params_sha256": final_metadata["params_sha256"],
            "optimizer_state_sha256": final_metadata["optimizer_state_sha256"],
            "chain_states_sha256": final_metadata["chain_states_sha256"],
            "amplitude_cache_present": final_metadata["amplitude_cache_present"],
        },
        "readiness_gates": gates,
        "all_training_readiness_gates_passed": all(gates.values()),
        "timing": {
            "wall_seconds_this_invocation": state_result[
                "wall_seconds_this_invocation"
            ],
            "checkpoint_seconds_this_invocation": checkpoint_seconds,
            "resume_count": resume_count,
            "timing_claim_eligible": resume_count == 0,
            "reason_if_ineligible": (
                None
                if resume_count == 0
                else "resumed wall time covers only the final process invocation"
            ),
        },
        "observable_claim_boundary": (
            "full S(q) is measured only in fixed-parameter blocks; ESS is not "
            "combined across optimizer updates, and this short preflight is not "
            "an order or phase estimate"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return trial


def run_spin_sampled_training(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume the registered square/triangular sampled-training preflight."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    source_manifest = _source_manifest()
    source_hash = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if overwrite:
        if result_path.exists():
            result_path.unlink()
        for case in spec["cases"]:
            for arm in spec["arms"]:
                candidate = output / (
                    f"checkpoint_{case['case_id']}_{arm['arm_id']}"
                )
                if candidate.is_dir():
                    shutil.rmtree(candidate)

    expected_header = (config_hash, source_hash, runtime_hash)
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        observed_header = (
            payload.get("config_sha256"),
            payload.get("source_manifest_sha256"),
            payload.get("runtime_identity_sha256"),
        )
        if (
            payload.get("schema_version") != TRAINING_RESULT_SCHEMA_VERSION
            or observed_header != expected_header
        ):
            raise RuntimeError(f"refusing stale sampled-training result: {result_path}")
        if payload.get("status") == "complete":
            manifest = embedded_trial_manifest(payload)
            if payload.get("trial_manifest_sha256") != canonical_sha256(manifest):
                raise RuntimeError(f"sampled-training trial seal failed: {result_path}")
            print(
                f"reused immutable complete sampled training: {result_path}",
                flush=True,
            )
            return payload
        if payload.get("status") not in {"running", "failed"}:
            raise RuntimeError(f"invalid sampled-training status: {result_path}")
        payload["status"] = "running"
        payload.pop("failure", None)
        atomic_json(result_path, payload)
    else:
        payload = {
            "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
            "domain": "spin",
            "task": "physical_sampled_training",
            "campaign_stage": spec["campaign_stage"],
            "status": "running",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": _portable_config_path(config_path),
            "config_sha256": config_hash,
            "config": spec,
            "source_sha256": source_manifest,
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
                "physical_spatial_projector_used": True,
                "on_demand_spin_hpsi_used": True,
                "persistent_physical_exchange_mcmc_used": True,
                "global_centered_microbatch_vmc_gradient_used": True,
                "full_SU2_invariant_structure_factor_used": True,
                "large_ED_reference_required": False,
                "energy_or_physics_win_claimed": False,
            },
            "matched_comparison_contract": {
                "same_balanced_tree_and_Morton_order": True,
                "same_feature_dim_and_intermediate_cutoff": True,
                "same_translation_projector": True,
                "same_initial_chain_array_and_sampler_random_stream": True,
                "same_samples_proposals_and_optimizer_schedule_per_update": True,
                "parameter_counts_are_reported_but_not_matched": True,
                "energy_to_target_requires_longer_held_out_runs": True,
            },
            "cases": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
        atomic_json(result_path, payload)

    try:
        projection = dict(spec["projection"])
        cases_by_id = {
            str(case_payload["case_id"]): case_payload
            for case_payload in payload["cases"]
        }
        for case in spec["cases"]:
            bonds, group, projector = _geometry(case, projection)
            case_id = str(case["case_id"])
            if case_id in cases_by_id:
                case_payload = cases_by_id[case_id]
            else:
                rows, columns = int(case["rows"]), int(case["columns"])
                case_payload = {
                    "case_id": case_id,
                    "geometry": case["geometry"],
                    "rows": rows,
                    "columns": columns,
                    "n_sites": rows * columns,
                    "j2_over_j1": float(case["j2_over_j1"]),
                    "periodic": True,
                    "support": "fixed_m0",
                    "ed_performed": False,
                    "bond_count": len(bonds),
                    "bonds_sha256": canonical_sha256(bonds),
                    "translation_group_order": projector.group_order,
                    "translation_group_sha256": canonical_sha256(group),
                    "site_order": list(morton_order(rows, columns)),
                    "observable_modes": [dict(mode) for mode in case["modes"]],
                    "trials": [],
                }
                payload["cases"].append(case_payload)
                cases_by_id[case_id] = case_payload
                atomic_json(result_path, payload)
            completed_arms = {
                str(trial["arm_id"]) for trial in case_payload["trials"]
            }
            for arm in spec["arms"]:
                if str(arm["arm_id"]) in completed_arms:
                    continue
                runtime = runtime_workflow._initialize_arm(
                    arm,
                    rows=int(case["rows"]),
                    columns=int(case["columns"]),
                    projector=projector,
                )
                checkpoint_directory = output / (
                    f"checkpoint_{case_id}_{runtime.arm_id}"
                )
                trial = _run_training_arm(
                    runtime=runtime,
                    arm=arm,
                    case=case,
                    bonds=bonds,
                    group=group,
                    spec=spec,
                    checkpoint_directory=checkpoint_directory,
                    config_hash=config_hash,
                    source_hash=source_hash,
                    runtime_hash=runtime_hash,
                    overwrite=overwrite,
                )
                if trial.get("status") == "running":  # pragma: no cover - test cap only
                    raise RuntimeError("production sampled-training arm stopped early")
                trial["trial_payload_sha256"] = trial_payload_sha256(trial)
                case_payload["trials"].append(trial)
                atomic_json(result_path, payload)
                print(
                    f"sampled-training {case_id} {runtime.arm_id}: "
                    f"E/N={trial['final_fixed_parameter_measurement']['energy_per_site']:.8f} "
                    f"grad={trial['history'][-1]['gradient']['norm']:.3e} "
                    f"ready={trial['all_training_readiness_gates_passed']}",
                    flush=True,
                )
        manifest = embedded_trial_manifest(payload)
        payload["trial_manifest"] = manifest
        payload["trial_manifest_sha256"] = canonical_sha256(manifest)
        payload["all_trials_training_ready"] = all(
            bool(trial["all_training_readiness_gates_passed"])
            for case in payload["cases"]
            for trial in case["trials"]
        )
        payload["status"] = "complete"
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_path, payload)
        return payload
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(result_path, payload)
        raise


__all__ = ["run_spin_sampled_training"]
