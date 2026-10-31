"""Local admission gate for the scale-first spin-only three-arm campaign."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitude,
    fixed_sz_autoregressive_spatial_log_amplitudes,
    fixed_sz_autoregressive_spatial_parameter_count,
    init_fixed_sz_autoregressive_spatial_vit,
    sample_fixed_sz_autoregressive_spatial_vit,
)
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_vit_parameter_count,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.many_cell_tree import (
    build_many_cell_tree_connectivity,
    compile_many_cell_device_hamiltonian,
    evaluate_many_cell_device_local_energy,
)
from su2_nqs.vmc.accelerator_sampled_force import (
    sampled_vmc_log_value_and_grad,
)
from su2_nqs.vmc.physical_sampled_sr import sampled_physical_sr_direction
from su2_nqs.vmc.scale_first_arms import SCALE_FIRST_ARM_IDS
from su2_nqs.vmc.spin_device import spin_device_hpsi


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_scale_first_three_arm_preflight"
ARM_IDS = SCALE_FIRST_ARM_IDS


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid scale-first three-arm config identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the three-arm contract must be preregistered")
    systems = spec.get("systems")
    if not isinstance(systems, list) or len(systems) != 2:
        raise ValueError("exactly two 6x6 systems are required")
    expected = {("triangular", 0.125), ("square", 0.5)}
    observed = set()
    for case in systems:
        if not isinstance(case, dict):
            raise TypeError("every system must be an object")
        geometry = str(case.get("geometry"))
        rows = _positive_integer(case.get("rows"), "rows")
        columns = _positive_integer(case.get("columns"), "columns")
        if (rows, columns) != (6, 6):
            raise ValueError("the admission systems must be 6x6")
        observed.add((geometry, float(case.get("j2_over_j1"))))
    if observed != expected:
        raise ValueError("the required triangular and square couplings changed")
    arms = spec.get("arms")
    if not isinstance(arms, list) or tuple(
        str(arm.get("arm_id")) for arm in arms if isinstance(arm, dict)
    ) != ARM_IDS:
        raise ValueError("the registered arm order or identity changed")
    shared = spec.get("shared_spatial_backbone")
    generic = spec.get("symmetry_unaware_control")
    if not isinstance(shared, dict) or not isinstance(generic, dict):
        raise TypeError("model contracts must be objects")
    for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
        _positive_integer(shared.get(name), f"shared.{name}")
    if generic != {
        "same_backbone_and_initial_arrays": True,
        "target_m2": None,
    }:
        raise ValueError("the symmetry-unaware arm must differ only by its mask")
    production = spec.get("production_contract")
    if not isinstance(production, dict):
        raise TypeError("production_contract must be an object")
    seeds = production.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("three distinct production seeds are required")
    if production.get("work_matched_view") != (
        "identical_samples_steps_and_full_H_bond_terms"
    ):
        raise ValueError("the work-matched estimator contract changed")
    if production.get("time_matched_view") != (
        "interpolate_preregistered_checkpoints_at_common_GPU_hours"
    ):
        raise ValueError("the time-matched estimator contract changed")
    canary = spec.get("local_canary")
    if not isinstance(canary, dict):
        raise TypeError("local_canary must be an object")
    _positive_integer(canary.get("sample_count_per_arm"), "sample_count_per_arm")
    _positive_integer(
        canary.get("su2_connected_capacity"), "su2_connected_capacity"
    )
    _positive_integer(
        canary.get("offline_capacity_audit_sample_count"),
        "offline_capacity_audit_sample_count",
    )
    optimizer = spec.get("optimizer_canary")
    if not isinstance(optimizer, dict) or optimizer.get("name") != (
        "matrix_free_sampled_SR_direction_only"
    ):
        raise ValueError("the optimizer canary must be matrix-free sampled SR")
    _positive_integer(
        optimizer.get("cg_max_iterations"), "optimizer.cg_max_iterations"
    )
    for name in (
        "damping",
        "cg_tolerance",
        "step_size",
        "maximum_relative_parameter_step",
    ):
        value = float(optimizer.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"optimizer.{name} must be positive and finite")
    benchmark = spec.get("accelerator_benchmark")
    if benchmark is not None:
        if not isinstance(benchmark, dict):
            raise TypeError("accelerator_benchmark must be an object")
        if benchmark.get("enabled") is not True:
            raise ValueError("accelerator_benchmark.enabled must be true")
        batch_sizes = benchmark.get("batch_sizes")
        if (
            not isinstance(batch_sizes, list)
            or not batch_sizes
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 2
                for value in batch_sizes
            )
            or batch_sizes != sorted(set(batch_sizes))
        ):
            raise ValueError(
                "accelerator_benchmark.batch_sizes must be unique sorted "
                "integers of at least two"
            )
        repetitions = _positive_integer(
            benchmark.get("warm_repetitions"),
            "accelerator_benchmark.warm_repetitions",
        )
        if repetitions < 3:
            raise ValueError("at least three warm repetitions are required")
        microbatch_size = _positive_integer(
            benchmark.get("device_microbatch_size", min(batch_sizes)),
            "accelerator_benchmark.device_microbatch_size",
        )
        if any(batch_size % microbatch_size for batch_size in batch_sizes):
            raise ValueError(
                "every accelerator batch size must be divisible by the "
                "device microbatch size"
            )
        tile_sizes = benchmark.get("su2_amplitude_tile_sizes")
        if tile_sizes is not None:
            connected_capacity = int(canary["su2_connected_capacity"])
            if (
                not isinstance(tile_sizes, list)
                or len(tile_sizes) < 3
                or tile_sizes != sorted(set(tile_sizes))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    or value > connected_capacity
                    or connected_capacity % value
                    for value in tile_sizes
                )
            ):
                raise ValueError(
                    "accelerator SU2 amplitude tile sizes must be at least "
                    "three unique sorted positive divisors of connected capacity"
                )
        su2_microbatches = benchmark.get("su2_device_microbatch_sizes")
        if su2_microbatches is not None and (
            not isinstance(su2_microbatches, list)
            or su2_microbatches != sorted(set(su2_microbatches))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or any(batch_size % value for batch_size in batch_sizes[-1:])
                for value in su2_microbatches
            )
        ):
            raise ValueError(
                "SU2 device microbatch sizes must be unique sorted positive "
                "divisors of the largest registered logical batch"
            )
        if benchmark.get("measurement") != "end_to_end_full_h_action":
            raise ValueError("the accelerator measurement contract changed")
        for name in ("replay_absolute_tolerance", "replay_relative_tolerance"):
            tolerance = float(benchmark.get(name, np.nan))
            if not np.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0e-10:
                raise ValueError(
                    f"accelerator_benchmark.{name} must lie in [0,1e-10]"
                )
    calibration = spec.get("training_calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise TypeError("training_calibration must be an object")
        if _positive_integer(
            calibration.get("sample_count"), "training_calibration.sample_count"
        ) not in benchmark["batch_sizes"]:
            raise ValueError(
                "training calibration sample count must be a registered batch"
            )
        if _positive_integer(
            calibration.get("total_steps"), "training_calibration.total_steps"
        ) < 3:
            raise ValueError(
                "training calibration requires compile plus at least two warm steps"
            )
        _positive_integer(calibration.get("seed"), "training_calibration.seed")
        _positive_integer(
            calibration.get("cg_max_iterations"),
            "training_calibration.cg_max_iterations",
        )
        tile_size = _positive_integer(
            calibration.get("su2_amplitude_tile_size"),
            "training_calibration.su2_amplitude_tile_size",
        )
        calibration_capacity = _positive_integer(
            calibration.get("su2_connected_capacity"),
            "training_calibration.su2_connected_capacity",
        )
        if tile_size not in benchmark.get("su2_amplitude_tile_sizes", []):
            raise ValueError("training calibration tile size is not registered")
        if calibration_capacity < int(canary["su2_connected_capacity"]):
            raise ValueError(
                "training calibration connected capacity cannot be smaller "
                "than the admitted canary capacity"
            )
        if calibration_capacity % tile_size:
            raise ValueError(
                "training calibration tile must divide connected capacity"
            )
        for name in (
            "damping",
            "cg_tolerance",
            "step_size",
            "maximum_relative_parameter_step",
        ):
            value = float(calibration.get(name, np.nan))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"training_calibration.{name} must be positive and finite"
                )


def _sr_direction_canary(
    params: object,
    samples: jax.Array,
    local_energies: np.ndarray,
    *,
    log_amplitude_fn,
    optimizer: dict[str, object],
) -> dict[str, object]:
    """Exercise a common sampled force/SR path without accepting an update."""

    started = time.perf_counter()
    gradient = sampled_vmc_log_value_and_grad(
        params,
        samples,
        local_energies,
        log_amplitude_fn=log_amplitude_fn,
    )
    direction = sampled_physical_sr_direction(
        params,
        samples,
        energy_gradient=gradient.gradient,
        log_amplitude_fn=log_amplitude_fn,
        damping=float(optimizer["damping"]),
        cg_tolerance=float(optimizer["cg_tolerance"]),
        cg_max_iterations=int(optimizer["cg_max_iterations"]),
        step_size=float(optimizer["step_size"]),
        maximum_relative_parameter_step=float(
            optimizer["maximum_relative_parameter_step"]
        ),
        raise_on_nonconvergence=False,
    )
    diagnostics = direction.diagnostics
    return {
        "direction_only_no_update_accepted": True,
        "physics_interpretation_forbidden": True,
        "sample_count": gradient.sample_count,
        "full_h_evaluations": 1,
        "energy": gradient.energy,
        "variance": gradient.variance,
        "energy_imaginary_abs": gradient.energy_imaginary_abs,
        "gradient_norm": gradient.gradient_norm,
        "all_finite": bool(
            np.isfinite(
                [
                    gradient.energy,
                    gradient.variance,
                    gradient.gradient_norm,
                    diagnostics.direction_norm,
                    diagnostics.relative_linear_residual,
                ]
            ).all()
        ),
        "cg_iterations": diagnostics.cg_iterations,
        "qgt_matvec_count": diagnostics.qgt_matvec_count,
        "linear_solve_converged": diagnostics.linear_solve_converged,
        "relative_linear_residual": diagnostics.relative_linear_residual,
        "direction_norm": diagnostics.direction_norm,
        "relative_parameter_step": diagnostics.relative_parameter_step,
        "directional_derivative": diagnostics.directional_derivative,
        "seconds": float(time.perf_counter() - started),
    }


def _bonds(geometry: str, j2: float):
    if geometry == "triangular":
        return triangular_hnf_j1j2_bonds(
            ((6, 0), (0, 6)), j1=1.0, j2=j2
        )
    return square_j1j2_bonds(6, 6, j1=1.0, j2=j2, periodic=True)


def _case_preflight(
    case: dict[str, object],
    shared: dict[str, object],
    generic: dict[str, object],
    canary: dict[str, object],
    optimizer: dict[str, object],
    seed: int,
    accelerator_benchmark: dict[str, object] | None = None,
) -> dict[str, object]:
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    triangular = geometry == "triangular"
    layout = rectangular_four_spin_layout(
        6, 6, triangular_coordinates=triangular
    )
    common = dict(
        embedding_dim=int(shared["embedding_dim"]),
        head_count=int(shared["head_count"]),
        depth=int(shared["depth"]),
        mlp_ratio=int(shared["mlp_ratio"]),
        phase_scale=float(shared["phase_scale"]),
        initialization_scale=float(shared["initialization_scale"]),
        initial_distance_decay=float(shared["initial_distance_decay"]),
        use_distance_envelope=True,
    )
    su2_config = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **common
    )
    fixed_config = FixedSzAutoregressiveSpatialViTConfig(
        geometry, 6, 6, target_m2=0, **common
    )
    generic_config = FixedSzAutoregressiveSpatialViTConfig(
        geometry, 6, 6, target_m2=None, **common
    )
    common_key = jax.random.PRNGKey(seed)
    su2_params = init_many_cell_schur_vit(common_key, su2_config)
    fixed_params = init_fixed_sz_autoregressive_spatial_vit(
        common_key, fixed_config
    )
    generic_params = init_fixed_sz_autoregressive_spatial_vit(
        common_key, generic_config
    )
    counts = {
        ARM_IDS[0]: many_cell_schur_vit_parameter_count(su2_params),
        ARM_IDS[1]: fixed_sz_autoregressive_spatial_parameter_count(fixed_params),
        ARM_IDS[2]: fixed_sz_autoregressive_spatial_parameter_count(
            generic_params
        ),
    }
    shared_initialization_identical = all(
        bool(np.array_equal(np.asarray(left), np.asarray(right)))
        for left, right in zip(
            jax.tree_util.tree_leaves(su2_params),
            jax.tree_util.tree_leaves(fixed_params),
            strict=True,
        )
    )
    all_three_initialization_identical = shared_initialization_identical and all(
        bool(np.array_equal(np.asarray(left), np.asarray(right)))
        for left, right in zip(
            jax.tree_util.tree_leaves(su2_params),
            jax.tree_util.tree_leaves(generic_params),
            strict=True,
        )
    )
    bonds = _bonds(geometry, j2)
    sample_count = int(canary["sample_count_per_arm"])
    su2_samples = sample_many_cell_schur_vit(
        su2_params,
        jax.random.PRNGKey(seed + 2),
        sample_count,
        su2_config,
    )
    compiled = compile_many_cell_device_hamiltonian(
        su2_config,
        bonds,
        connected_capacity=int(canary["su2_connected_capacity"]),
    )
    su2_first_action = evaluate_many_cell_device_local_energy(
        compiled, su2_params, su2_samples
    )
    su2_action = evaluate_many_cell_device_local_energy(
        compiled, su2_params, su2_samples
    )
    base_replay_atol = (
        0.0
        if accelerator_benchmark is None
        else float(accelerator_benchmark["replay_absolute_tolerance"])
    )
    base_replay_rtol = (
        0.0
        if accelerator_benchmark is None
        else float(accelerator_benchmark["replay_relative_tolerance"])
    )
    base_replay_difference = float(
        np.max(
            np.abs(
                su2_first_action.local_energies
                - su2_action.local_energies
            ),
            initial=0.0,
        )
    )
    if not np.allclose(
        su2_first_action.local_energies,
        su2_action.local_energies,
        atol=base_replay_atol,
        rtol=base_replay_rtol,
        equal_nan=False,
    ):
        raise RuntimeError(
            "SU2 device-H warm replay changed beyond the registered "
            f"tolerance; max_abs={base_replay_difference}"
        )
    audit_samples = sample_many_cell_schur_vit(
        su2_params,
        jax.random.PRNGKey(seed + 4),
        int(canary["offline_capacity_audit_sample_count"]),
        su2_config,
    )
    reference_audit = build_many_cell_tree_connectivity(
        np.asarray(audit_samples), su2_config, bonds
    )
    reference_widths = np.asarray(reference_audit.widths)
    fixed_samples = sample_fixed_sz_autoregressive_spatial_vit(
        fixed_params,
        jax.random.PRNGKey(seed + 3),
        sample_count,
        fixed_config,
    )

    def fixed_action(values, states):
        return spin_device_hpsi(
            values,
            states,
            bonds,
            fixed_config,
            amplitude_fn=lambda candidate, row, model: jnp.exp(
                fixed_sz_autoregressive_spatial_log_amplitude(
                    candidate, row, model
                )
            ),
        )

    fixed_kernel = jax.jit(fixed_action)
    started = time.perf_counter()
    fixed_centers, fixed_hpsi = fixed_kernel(fixed_params, fixed_samples)
    jax.block_until_ready(fixed_hpsi)
    fixed_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fixed_centers, fixed_hpsi = fixed_kernel(fixed_params, fixed_samples)
    jax.block_until_ready(fixed_hpsi)
    fixed_warm_seconds = time.perf_counter() - started

    generic_samples = sample_fixed_sz_autoregressive_spatial_vit(
        generic_params,
        jax.random.PRNGKey(seed + 6),
        sample_count,
        generic_config,
    )

    def generic_action(values, states):
        return spin_device_hpsi(
            values,
            states,
            bonds,
            generic_config,
            amplitude_fn=lambda candidate, row, model: jnp.exp(
                fixed_sz_autoregressive_spatial_log_amplitude(
                    candidate, row, model
                )
            ),
        )

    generic_kernel = jax.jit(generic_action)
    started = time.perf_counter()
    generic_centers, generic_hpsi = generic_kernel(generic_params, generic_samples)
    jax.block_until_ready(generic_hpsi)
    generic_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    generic_centers, generic_hpsi = generic_kernel(generic_params, generic_samples)
    jax.block_until_ready(generic_hpsi)
    generic_warm_seconds = time.perf_counter() - started
    fixed_local = np.asarray(fixed_hpsi / fixed_centers)
    generic_local = np.asarray(generic_hpsi / generic_centers)
    optimizer_canaries = {
        ARM_IDS[0]: _sr_direction_canary(
            su2_params,
            su2_samples,
            su2_action.local_energies,
            log_amplitude_fn=lambda values, states: (
                many_cell_schur_log_amplitudes(values, states, su2_config)
            ),
            optimizer=optimizer,
        ),
        ARM_IDS[1]: _sr_direction_canary(
            fixed_params,
            fixed_samples,
            fixed_local,
            log_amplitude_fn=lambda values, states: (
                fixed_sz_autoregressive_spatial_log_amplitudes(
                    values, states, fixed_config
                )
            ),
            optimizer=optimizer,
        ),
        ARM_IDS[2]: _sr_direction_canary(
            generic_params,
            generic_samples,
            generic_local,
            log_amplitude_fn=lambda values, states: (
                fixed_sz_autoregressive_spatial_log_amplitudes(
                    values, states, generic_config
                )
            ),
            optimizer=optimizer,
        ),
    }
    throughput = None
    if accelerator_benchmark is not None:
        warm_repetitions = int(accelerator_benchmark["warm_repetitions"])
        replay_atol = float(
            accelerator_benchmark["replay_absolute_tolerance"]
        )
        replay_rtol = float(
            accelerator_benchmark["replay_relative_tolerance"]
        )

        def replay_difference(reference, candidate, label: str) -> float:
            left = np.asarray(reference)
            right = np.asarray(candidate)
            difference = float(np.max(np.abs(left - right), initial=0.0))
            if not np.allclose(
                left,
                right,
                atol=replay_atol,
                rtol=replay_rtol,
                equal_nan=False,
            ):
                raise RuntimeError(
                    f"accelerator {label} warm replay changed beyond "
                    f"atol={replay_atol}, rtol={replay_rtol}; "
                    f"max_abs={difference}"
                )
            return difference

        def summarize(times: list[float], batch_size: int) -> dict[str, object]:
            values = np.asarray(times, dtype=np.float64)
            median = float(np.median(values))
            return {
                "warm_repetitions": len(times),
                "warm_seconds": [float(value) for value in values],
                "warm_median_seconds": median,
                "warm_minimum_seconds": float(np.min(values)),
                "warm_maximum_seconds": float(np.max(values)),
                "samples_per_second": float(batch_size / median),
                "full_h_bond_terms_per_second": float(
                    batch_size * len(bonds) / median
                ),
            }

        def physical_row(
            params: object,
            config: FixedSzAutoregressiveSpatialViTConfig,
            kernel,
            batch_size: int,
            key: jax.Array,
        ) -> dict[str, object]:
            sample_started = time.perf_counter()
            samples = sample_fixed_sz_autoregressive_spatial_vit(
                params, key, batch_size, config
            )
            jax.block_until_ready(samples)
            sampling_seconds = float(time.perf_counter() - sample_started)

            def evaluate():
                started = time.perf_counter()
                centers, action = kernel(params, samples)
                jax.block_until_ready(action)
                seconds = float(time.perf_counter() - started)
                local = np.asarray(action / centers)
                if not np.all(np.isfinite(local)):
                    raise FloatingPointError(
                        "accelerator physical-H benchmark is non-finite"
                    )
                return local, seconds

            first_local, first_seconds = evaluate()
            warm_times = []
            maximum_replay_difference = 0.0
            for _ in range(warm_repetitions):
                replay, seconds = evaluate()
                maximum_replay_difference = max(
                    maximum_replay_difference,
                    replay_difference(first_local, replay, "physical-H"),
                )
                warm_times.append(seconds)
            return {
                "batch_size": batch_size,
                "sampling_seconds": sampling_seconds,
                "shape_compile_or_first_seconds": first_seconds,
                "all_finite": True,
                "overflow_count": 0,
                "maximum_replay_abs_difference": maximum_replay_difference,
                "full_h_bond_terms_per_evaluation": batch_size * len(bonds),
                **summarize(warm_times, batch_size),
            }

        def su2_row(batch_size: int, key: jax.Array) -> dict[str, object]:
            sample_started = time.perf_counter()
            samples = sample_many_cell_schur_vit(
                su2_params, key, batch_size, su2_config
            )
            jax.block_until_ready(samples)
            sampling_seconds = float(time.perf_counter() - sample_started)
            first = evaluate_many_cell_device_local_energy(
                compiled, su2_params, samples
            )
            warm_times = []
            maximum_width = int(np.max(first.unique_widths))
            maximum_replay_difference = 0.0
            for _ in range(warm_repetitions):
                replay = evaluate_many_cell_device_local_energy(
                    compiled, su2_params, samples
                )
                maximum_replay_difference = max(
                    maximum_replay_difference,
                    replay_difference(
                        first.local_energies,
                        replay.local_energies,
                        "SU2-H",
                    ),
                )
                maximum_width = max(
                    maximum_width, int(np.max(replay.unique_widths))
                )
                warm_times.append(replay.evaluation_seconds)
            return {
                "batch_size": batch_size,
                "sampling_seconds": sampling_seconds,
                "shape_compile_or_first_seconds": first.evaluation_seconds,
                "all_finite": bool(
                    np.all(np.isfinite(first.local_energies))
                ),
                "overflow_count": int(np.count_nonzero(first.overflow)),
                "maximum_replay_abs_difference": maximum_replay_difference,
                "maximum_unique_width": maximum_width,
                "connected_capacity": compiled.connected_capacity,
                "full_h_bond_terms_per_evaluation": first.full_h_term_calls,
                **summarize(warm_times, batch_size),
            }

        benchmark_started = time.perf_counter()
        throughput = {arm_id: [] for arm_id in ARM_IDS}
        # The admission sweep deliberately compiles several static batch
        # shapes. Release executable caches between shapes so an 80 GB GPU is
        # tested on each workload independently rather than on their summed
        # compilation footprint.
        jax.clear_caches()
        gc.collect()
        for batch_index, batch_size in enumerate(
            accelerator_benchmark["batch_sizes"]
        ):
            batch_size = int(batch_size)
            throughput[ARM_IDS[0]].append(
                su2_row(
                    batch_size,
                    jax.random.PRNGKey(seed + 1000 + 10 * batch_index),
                )
            )
            throughput[ARM_IDS[1]].append(
                physical_row(
                    fixed_params,
                    fixed_config,
                    fixed_kernel,
                    batch_size,
                    jax.random.PRNGKey(seed + 1001 + 10 * batch_index),
                )
            )
            throughput[ARM_IDS[2]].append(
                physical_row(
                    generic_params,
                    generic_config,
                    generic_kernel,
                    batch_size,
                    jax.random.PRNGKey(seed + 1002 + 10 * batch_index),
                )
            )
            jax.clear_caches()
            gc.collect()
        throughput = {
            "measurement": accelerator_benchmark["measurement"],
            "compile_and_warm_are_separate": True,
            "same_samples_replayed_within_each_timing_row": True,
            "replay_absolute_tolerance": replay_atol,
            "replay_relative_tolerance": replay_rtol,
            "compiled_shape_caches_cleared_between_batch_sizes": True,
            "total_seconds": float(time.perf_counter() - benchmark_started),
            "arms": throughput,
        }
    return {
        "case_id": case["case_id"],
        "geometry": geometry,
        "j2_over_j1": j2,
        "bond_count": len(bonds),
        "parameter_counts": counts,
        "shared_parameter_pytree_identical": (
            jax.tree_util.tree_structure(su2_params)
            == jax.tree_util.tree_structure(fixed_params)
            and all(
                left.shape == right.shape
                for left, right in zip(
                    jax.tree_util.tree_leaves(su2_params),
                    jax.tree_util.tree_leaves(fixed_params),
                    strict=True,
                )
            )
        ),
        "shared_initial_parameter_arrays_identical": (
            shared_initialization_identical
        ),
        "all_three_initial_parameter_arrays_identical": (
            all_three_initialization_identical
        ),
        "su2_device_h": {
            "sample_count": su2_action.sample_count,
            "all_finite": bool(np.all(np.isfinite(su2_action.local_energies))),
            "overflow_count": int(np.count_nonzero(su2_action.overflow)),
            "plan_compilation_seconds": compiled.compilation_seconds,
            "first_compile_and_evaluate_seconds": (
                su2_first_action.evaluation_seconds
            ),
            "warm_evaluate_seconds": su2_action.evaluation_seconds,
            "warm_replay_max_abs_difference": base_replay_difference,
            "bucket_capacities": [list(row) for row in compiled.bucket_capacities],
            "bucket_bond_counts": list(compiled.bucket_bond_counts),
            "raw_transition_slots_per_sample": (
                compiled.raw_transition_slots_per_sample
            ),
            "connected_capacity": compiled.connected_capacity,
            "unique_widths": [int(value) for value in su2_action.unique_widths],
            "raw_to_compact_capacity_ratio": (
                compiled.raw_transition_slots_per_sample
                / compiled.connected_capacity
            ),
            "full_h_term_calls": su2_action.full_h_term_calls,
            "offline_reference_capacity_audit": {
                "not_on_training_critical_path": True,
                "sample_count": len(reference_widths),
                "minimum_unique_width": int(np.min(reference_widths)),
                "median_unique_width": float(np.median(reference_widths)),
                "p90_unique_width": float(np.quantile(reference_widths, 0.9)),
                "maximum_unique_width": int(np.max(reference_widths)),
                "capacity_headroom": int(
                    compiled.connected_capacity - np.max(reference_widths)
                ),
                "connectivity_seconds": reference_audit.connectivity_seconds,
            },
        },
        "fixed_sz_shared_device_h": {
            "sample_count": len(fixed_local),
            "all_finite": bool(np.all(np.isfinite(fixed_local))),
            "first_compile_and_evaluate_seconds": fixed_first_seconds,
            "warm_evaluate_seconds": fixed_warm_seconds,
            "full_h_term_calls": len(fixed_local) * len(bonds),
        },
        "symmetry_unaware_device_h": {
            "sample_count": len(generic_local),
            "all_finite": bool(np.all(np.isfinite(generic_local))),
            "first_compile_and_evaluate_seconds": generic_first_seconds,
            "warm_evaluate_seconds": generic_warm_seconds,
            "full_h_term_calls": len(generic_local) * len(bonds),
            "native_autoregressive_sampler": {
                "exactly_normalized": True,
                "magnetization_values": [
                    int(2 * np.sum(row) - generic_config.n_sites)
                    for row in np.asarray(generic_samples)
                ],
            },
        },
        "literature_raw_energy_per_site": case[
            "literature_raw_energy_per_site"
        ],
        "finite_exact_reference_per_site": case[
            "finite_exact_reference_per_site"
        ],
        "optimizer_canaries": optimizer_canaries,
        "accelerator_throughput": throughput,
    }


def run_spin_scale_first_three_arm_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run architecture/device-work admission only; perform no optimization."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    rows = [
        _case_preflight(
            case,
            spec["shared_spatial_backbone"],
            spec["symmetry_unaware_control"],
            spec["local_canary"],
            spec["optimizer_canary"],
            int(spec["seed"]) + 10 * index,
            spec.get("accelerator_benchmark"),
        )
        for index, case in enumerate(spec["systems"])
    ]
    counts = np.asarray(
        [list(row["parameter_counts"].values()) for row in rows], dtype=np.float64
    )
    spread = float((np.max(counts) - np.min(counts)) / np.mean(counts))
    gates = {
        "shared_su2_fixed_sz_parameter_pytree_identical": all(
            bool(row["shared_parameter_pytree_identical"]) for row in rows
        ),
        "shared_su2_fixed_sz_initialization_identical": all(
            bool(row["shared_initial_parameter_arrays_identical"])
            for row in rows
        ),
        "all_three_initial_parameter_arrays_identical": all(
            bool(row["all_three_initial_parameter_arrays_identical"])
            for row in rows
        ),
        "three_arm_parameter_budget_matched": spread
        <= float(spec["parameter_budget"]["maximum_fractional_spread"]),
        "all_device_h_actions_finite": all(
            bool(row[key]["all_finite"])
            for row in rows
            for key in (
                "su2_device_h",
                "fixed_sz_shared_device_h",
                "symmetry_unaware_device_h",
            )
        ),
        "no_su2_candidate_overflow": all(
            int(row["su2_device_h"]["overflow_count"]) == 0 for row in rows
        ),
        "registered_capacity_audit_has_headroom": all(
            int(
                row["su2_device_h"]["offline_reference_capacity_audit"][
                    "capacity_headroom"
                ]
            )
            > 0
            for row in rows
        ),
        "all_sampled_force_sr_directions_finite": all(
            bool(canary["all_finite"])
            and float(canary["directional_derivative"]) >= 0.0
            for row in rows
            for canary in row["optimizer_canaries"].values()
        ),
        "two_complementary_work_views_preregistered": True,
    }
    if spec.get("accelerator_benchmark") is not None:
        expected_batches = list(spec["accelerator_benchmark"]["batch_sizes"])
        gates["accelerator_throughput_sweep_complete"] = all(
            row["accelerator_throughput"] is not None
            and all(
                [entry["batch_size"] for entry in rows_by_arm] == expected_batches
                and all(
                    bool(entry["all_finite"])
                    and int(entry["overflow_count"]) == 0
                    and float(entry["warm_median_seconds"]) > 0.0
                    for entry in rows_by_arm
                )
                for rows_by_arm in row["accelerator_throughput"]["arms"].values()
            )
            for row in rows
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "cases": rows,
        "parameter_fractional_spread": spread,
        "gate_results": gates,
        "local_preflight_passed": all(gates.values()),
        "optimization_started": False,
        "production_training_ready": False,
        "remaining_training_blockers": [
            "accelerator timing on the target GPU architecture",
            "production sample/step schedule registration after timing",
        ],
        "claim_boundary": spec["claim_boundary"],
        "runtime_identity": numerical_runtime_identity(),
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


__all__ = ["TASK", "run_spin_scale_first_three_arm_preflight"]
