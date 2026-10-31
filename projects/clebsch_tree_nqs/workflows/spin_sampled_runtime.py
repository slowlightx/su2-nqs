"""On-demand physical-spin MCMC/runtime readiness smoke.

This workflow deliberately does not enumerate a Hilbert space and does not
diagonalize a Hamiltonian.  It combines the physical-input U(1)/SU(2)
Clebsch trees with an exact finite translation projector, fixed-``M2=0``
exchange Metropolis chains, and the padded on-demand ``H|psi>`` kernel.

The resulting artifact is a scaling/readiness diagnostic.  Random-state
energies, four-step-or-shorter sanity optimizations, and throughput numbers
from this workflow are not evidence of an energy or physics advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.route_policy import require_active_production_method
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.physical_tree import (
    PhysicalSpinTreeConfig,
    init_physical_spin_tree,
    physical_spin_tree_amplitude,
    physical_spin_tree_parameter_count,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.spin_control_tree import (
    SpinU1TreeConfig,
    init_spin_u1_tree,
    spin_control_tree_parameter_count,
    spin_u1_tree_amplitude,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.physical_exchange import (
    initialize_spin_exchange_chains,
    run_exchange_sweeps,
)
from su2_nqs.vmc.physical_vmc import physical_hpsi_vmc_loss_with_aux
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
ALLOWED_BACKBONES = {"u1", "su2"}
RUNTIME_RESULT_SCHEMA_VERSION = 2
RUNTIME_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _ArmRuntime:
    arm_id: str
    backbone: str
    feature_dim: int
    maximum_intermediate_q: int
    model_config: object
    params: object
    parameter_count: int
    projector: PhysicalSpatialProjector
    projected_amplitude: Callable[[object, jax.Array], jax.Array]


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


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("sampled runtime requires schema_version=2 and domain='spin'")
    if spec.get("task") != "physical_mcmc_runtime_smoke":
        raise ValueError("config task is not physical_mcmc_runtime_smoke")
    case = spec.get("case")
    if not isinstance(case, dict):
        raise TypeError("case must be an object")
    geometry = str(case.get("geometry"))
    if geometry not in {"square", "triangular"}:
        raise ValueError("sampled runtime supports square and triangular geometry")
    rows = _positive_integer(case.get("rows"), "case.rows")
    columns = _positive_integer(case.get("columns"), "case.columns")
    if (rows * columns) % 2:
        raise ValueError("fixed-M2=0 sampling requires an even site count")
    if geometry == "triangular" and (rows % 6 or columns % 6):
        raise ValueError("triangular K/M runtime diagnostics require 6-compatible dimensions")
    if geometry == "square" and (rows % 2 or columns % 2):
        raise ValueError("square Neel/stripe diagnostics require even dimensions")
    if str(case.get("support")) != "fixed_m0":
        raise ValueError("sampled runtime requires fixed_m0 support")
    if case.get("periodic") is not True:
        raise ValueError("sampled runtime requires a periodic torus")
    if case.get("ed") is not False:
        raise ValueError("sampled runtime config must explicitly disable ED")
    ratio = float(case.get("j2_over_j1", np.nan))
    if not np.isfinite(ratio):
        raise ValueError("j2_over_j1 must be finite")
    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection.get("kind") != "translation":
        raise ValueError("sampled runtime requires the translation projector")
    _positive_integer(projection.get("orbit_chunk_size"), "orbit_chunk_size")
    if not isinstance(projection.get("rematerialize_orbit"), bool):
        raise TypeError("rematerialize_orbit must be boolean")
    observable = spec.get("observable_contract")
    if not isinstance(observable, dict):
        raise TypeError("observable_contract must be an object")
    expected_estimators = (
        ["diagonal_Szz_K_star", "diagonal_Szz_M_star"]
        if geometry == "triangular"
        else ["diagonal_Szz_Neel_Q", "diagonal_Szz_stripe_star"]
    )
    if (
        observable.get("reported_estimators") != expected_estimators
        or observable.get("basis_estimator")
        != "diagonal_Szz_from_sampled_computational_basis_configurations"
        or observable.get("su2_invariant_structure_factor") is not False
    ):
        raise ValueError(
            "ordering observables must use the geometry-specific diagonal Szz labels"
        )
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty list")
    seen: set[str] = set()
    backbones: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise TypeError("every arm must be an object")
        arm_id = str(arm.get("arm_id", ""))
        if not arm_id or arm_id in seen:
            raise ValueError("arm_id values must be nonempty and unique")
        seen.add(arm_id)
        backbone = str(arm.get("backbone"))
        if backbone not in ALLOWED_BACKBONES:
            raise ValueError("runtime backbones are restricted to u1 and su2")
        backbones.add(backbone)
        if str(arm.get("projection")) != "translation":
            raise ValueError("every runtime arm must use translation projection")
        model = arm.get("model")
        if not isinstance(model, dict):
            raise TypeError("every arm needs a model object")
        _positive_integer(model.get("feature_dim"), "feature_dim")
        _positive_integer(
            model.get("maximum_intermediate_q"), "maximum_intermediate_q"
        )
        scale = float(model.get("initialization_scale", np.nan))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("initialization_scale must be positive and finite")
        if isinstance(arm.get("seed"), bool) or not isinstance(arm.get("seed"), int):
            raise TypeError("every arm seed must be an integer")
    if backbones != ALLOWED_BACKBONES:
        raise ValueError("runtime smoke requires both u1 and su2 arms")
    sampler = spec.get("sampler")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for name in (
        "chains",
        "burn_in_sweeps",
        "measurement_sweeps",
        "proposals_per_sweep",
        "checkpoint_every_sweeps",
    ):
        _positive_integer(sampler.get(name), f"sampler.{name}")
    if isinstance(sampler.get("seed"), bool) or not isinstance(
        sampler.get("seed"), int
    ):
        raise TypeError("sampler.seed must be an integer")
    bond_probability = float(sampler.get("bond_proposal_probability", np.nan))
    if not 0.0 <= bond_probability <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0,1]")
    profile = spec.get("profile")
    if not isinstance(profile, dict):
        raise TypeError("profile must be an object")
    batches = profile.get("batch_sizes")
    if (
        not isinstance(batches, list)
        or not batches
        or batches != sorted(set(batches))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in batches)
    ):
        raise ValueError("profile.batch_sizes must be unique increasing positive integers")
    if max(batches) > int(sampler["chains"]):
        raise ValueError("profile batch sizes cannot exceed the chain count")
    _positive_integer(profile.get("warm_repeats"), "profile.warm_repeats")


def _initialize_arm(
    arm: dict[str, object],
    *,
    rows: int,
    columns: int,
    projector: PhysicalSpatialProjector,
) -> _ArmRuntime:
    backbone = str(arm["backbone"])
    require_active_production_method(backbone)
    model = dict(arm["model"])
    n_sites = rows * columns
    tree = FusionTree.balanced(n_sites)
    order = tuple(morton_order(rows, columns))
    order_array = jnp.asarray(order, dtype=jnp.int32)
    feature_dim = int(model["feature_dim"])
    maximum_q = int(model["maximum_intermediate_q"])
    scale = float(model["initialization_scale"])
    key = jax.random.PRNGKey(int(arm["seed"]))
    if backbone == "su2":
        model_config = PhysicalSpinTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=feature_dim,
            maximum_intermediate_q=maximum_q,
            initialization_scale=scale,
        )
        params = init_physical_spin_tree(key, model_config)

        def base(candidate, state, config):
            return physical_spin_tree_amplitude(
                candidate, state[order_array], config
            )

        parameter_count = physical_spin_tree_parameter_count(params)
    elif backbone == "u1":
        model_config = SpinU1TreeConfig(
            tree,
            order,
            target_m2=0,
            feature_dim=feature_dim,
            maximum_intermediate_abs_m2=maximum_q,
            initialization_scale=scale,
        )
        params = init_spin_u1_tree(key, model_config)
        base = spin_u1_tree_amplitude
        parameter_count = spin_control_tree_parameter_count(params)
    else:  # pragma: no cover - guarded by validation and route policy
        raise ValueError(f"unknown backbone: {backbone!r}")

    def projected(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            model_config,
            projector,
            amplitude_fn=base,
        )

    return _ArmRuntime(
        arm_id=str(arm["arm_id"]),
        backbone=backbone,
        feature_dim=feature_dim,
        maximum_intermediate_q=maximum_q,
        model_config=model_config,
        params=params,
        parameter_count=parameter_count,
        projector=projector,
        projected_amplitude=projected,
    )


def _ordering_structure_factor_series(
    states: np.ndarray, *, geometry: str, rows: int, columns: int
) -> dict[str, np.ndarray]:
    """Return geometry-specific diagonal ``Szz`` ordering slow modes."""

    concrete = np.asarray(states, dtype=np.float64)
    if concrete.ndim != 2 or concrete.shape[1] != rows * columns:
        raise ValueError("states have the wrong lattice shape")
    spins = concrete - 0.5
    coordinates = np.asarray(
        [(row, column) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )

    def star(indices: tuple[tuple[int, int], ...]) -> np.ndarray:
        values = []
        for ky, kx in indices:
            phase = np.exp(
                2.0j
                * np.pi
                * (
                    ky * coordinates[:, 0] / rows
                    + kx * coordinates[:, 1] / columns
                )
            )
            values.append(np.abs(spins @ phase) ** 2 / (rows * columns))
        return np.max(np.stack(values, axis=1), axis=1)

    if geometry == "triangular":
        if rows % 3 or columns % 3 or rows % 2 or columns % 2:
            raise ValueError("K/M diagnostics require dimensions divisible by six")
        return {
            "diagonal_Szz_K_star": star(
                (
                    (rows // 3, columns // 3),
                    (2 * rows // 3, 2 * columns // 3),
                )
            ),
            "diagonal_Szz_M_star": star(
                (
                    (rows // 2, 0),
                    (0, columns // 2),
                    (rows // 2, columns // 2),
                )
            ),
        }
    if geometry == "square":
        if rows % 2 or columns % 2:
            raise ValueError("Neel/stripe diagnostics require even dimensions")
        return {
            "diagonal_Szz_Neel_Q": star(((rows // 2, columns // 2),)),
            "diagonal_Szz_stripe_star": star(
                ((rows // 2, 0), (0, columns // 2))
            ),
        }
    raise ValueError(f"unknown geometry: {geometry!r}")


def _autocorrelation_summary(series: np.ndarray) -> dict[str, object]:
    """Estimate multi-chain autocorrelation using an initial-positive window."""

    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("autocorrelation input must have shape (time>=2, chains>=2)")
    if np.any(~np.isfinite(values)):
        return {
            "finite": False,
            "tau_int": None,
            "ess": 0.0,
            "split_rhat": None,
            "window_lag": 0,
        }
    time_steps, chains = values.shape
    centered = values - np.mean(values, axis=0, keepdims=True)
    gamma_zero = float(np.mean(centered**2))
    if gamma_zero <= np.finfo(np.float64).tiny:
        tau_int = 1.0
        window_lag = 0
    else:
        correlations = [1.0]
        for lag in range(1, time_steps):
            correlations.append(
                float(np.mean(centered[:-lag] * centered[lag:]) / gamma_zero)
            )
        retained: list[float] = []
        lag = 1
        while lag < time_steps:
            pair = correlations[lag]
            if lag + 1 < time_steps:
                pair += correlations[lag + 1]
            if pair <= 0.0:
                break
            retained.extend(correlations[lag : min(lag + 2, time_steps)])
            lag += 2
        tau_int = max(1.0, 1.0 + 2.0 * float(np.sum(retained)))
        window_lag = len(retained)
    ess = min(float(time_steps * chains), float(time_steps * chains) / tau_int)

    half = time_steps // 2
    split_rhat: float | None
    if half < 2:
        split_rhat = None
    else:
        split = np.concatenate((values[:half].T, values[-half:].T), axis=0)
        within = float(np.mean(np.var(split, axis=1, ddof=1)))
        between = float(half * np.var(np.mean(split, axis=1), ddof=1))
        if within <= np.finfo(np.float64).tiny:
            split_rhat = 1.0 if between <= np.finfo(np.float64).tiny else None
        else:
            variance = (half - 1.0) * within / half + between / half
            split_rhat = float(np.sqrt(max(variance / within, 0.0)))
    return {
        "finite": True,
        "tau_int": tau_int,
        "ess": ess,
        "split_rhat": split_rhat,
        "window_lag": window_lag,
    }


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    concrete = np.asarray(values, dtype=np.complex128).reshape(-1)
    return [[float(value.real), float(value.imag)] for value in concrete]


def _pairs_complex(values: list[list[float]]) -> np.ndarray:
    return np.asarray(
        [complex(float(pair[0]), float(pair[1])) for pair in values],
        dtype=np.complex128,
    )


def _checkpoint_identity(
    *,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    case_id: str,
    runtime: _ArmRuntime,
) -> dict[str, object]:
    return {
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "runtime_identity_sha256": runtime_hash,
        "case_id": case_id,
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "initial_params_sha256": pytree_sha256(runtime.params),
    }


def _write_checkpoint(
    path: Path,
    *,
    identity: dict[str, object],
    initial_chain_states_sha256: str,
    common_profile_states_sha256: str,
    status: str,
    states: np.ndarray,
    amplitudes: np.ndarray,
    key: np.ndarray,
    burn_completed: int,
    measurement_completed: int,
    burn_diagnostics: list[dict[str, float]],
    measurement_diagnostics: list[dict[str, float]],
    energy_series: list[list[float]],
    ordering_first_series: list[list[float]],
    ordering_second_series: list[list[float]],
    snapshot_changed: list[list[bool]],
    performance_profile: list[dict[str, object]] | None,
    resume_count: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RUNTIME_CHECKPOINT_SCHEMA_VERSION,
        "status": status,
        "identity": identity,
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "common_profile_states_sha256": common_profile_states_sha256,
        "states": np.asarray(states, dtype=np.int32).tolist(),
        "amplitudes": _complex_pairs(amplitudes),
        "key": np.asarray(key, dtype=np.uint32).tolist(),
        "burn_completed": int(burn_completed),
        "measurement_completed": int(measurement_completed),
        "burn_diagnostics": burn_diagnostics,
        "measurement_diagnostics": measurement_diagnostics,
        "energy_series": energy_series,
        "ordering_first_series": ordering_first_series,
        "ordering_second_series": ordering_second_series,
        "snapshot_changed": snapshot_changed,
        "performance_profile": performance_profile,
        "resume_count": int(resume_count),
    }
    payload["checkpoint_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return payload


def _load_checkpoint(
    path: Path,
    identity: dict[str, object],
    *,
    initial_chain_states_sha256: str,
    common_profile_states_sha256: str,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != RUNTIME_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint schema mismatch: {path}; expected "
            f"{RUNTIME_CHECKPOINT_SCHEMA_VERSION}"
        )
    stored_hash = payload.pop("checkpoint_sha256", None)
    if stored_hash != canonical_sha256(payload):
        raise RuntimeError(f"checkpoint seal failed: {path}")
    payload["checkpoint_sha256"] = stored_hash
    if payload.get("identity") != identity:
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    if payload.get("initial_chain_states_sha256") != initial_chain_states_sha256:
        raise RuntimeError(f"checkpoint initial-chain identity mismatch: {path}")
    if payload.get("common_profile_states_sha256") != common_profile_states_sha256:
        raise RuntimeError(f"checkpoint profile-state identity mismatch: {path}")
    return payload


def _float_diagnostics(diagnostics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in diagnostics.items()}


def _profile_kernels(
    runtime: _ArmRuntime,
    states: np.ndarray,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    padded_width: int,
    batch_sizes: list[int],
    warm_repeats: int,
) -> tuple[list[dict[str, object]], Callable, Callable]:
    """Compile/profile static Hpsi and gradient shapes and return the kernels."""

    def projected_for_hpsi(candidate, state, _config):
        return runtime.projected_amplitude(candidate, state)

    def hpsi_kernel(candidate, samples, connected, elements, valid):
        return spin_physical_hpsi(
            candidate,
            samples,
            connected,
            elements,
            valid,
            runtime.model_config,
            amplitude_fn=projected_for_hpsi,
        )

    def gradient_kernel(candidate, samples, connected, elements, valid):
        def action(parameters):
            return hpsi_kernel(parameters, samples, connected, elements, valid)

        return physical_hpsi_vmc_loss_with_aux(candidate, hpsi_fn=action)

    compiled_hpsi = jax.jit(hpsi_kernel)
    compiled_gradient = jax.jit(
        jax.value_and_grad(gradient_kernel, has_aux=True)
    )
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        batch = np.asarray(states[:batch_size], dtype=np.int32)
        connectivity_started = time.perf_counter()
        connected, elements, valid, widths = padded_spin_connectivity(
            batch, bonds, padded_width=padded_width
        )
        connectivity_seconds = time.perf_counter() - connectivity_started
        arguments = tuple(
            jnp.asarray(value)
            for value in (batch, connected, elements, valid)
        )

        started = time.perf_counter()
        cold_hpsi = compiled_hpsi(runtime.params, *arguments)
        jax.block_until_ready(cold_hpsi)
        hpsi_cold_seconds = time.perf_counter() - started
        hpsi_warm = []
        for _ in range(warm_repeats):
            started = time.perf_counter()
            value = compiled_hpsi(runtime.params, *arguments)
            jax.block_until_ready(value)
            hpsi_warm.append(time.perf_counter() - started)

        started = time.perf_counter()
        cold_gradient = compiled_gradient(runtime.params, *arguments)
        jax.block_until_ready(cold_gradient)
        gradient_cold_seconds = time.perf_counter() - started
        gradient_warm = []
        for _ in range(warm_repeats):
            started = time.perf_counter()
            value = compiled_gradient(runtime.params, *arguments)
            jax.block_until_ready(value)
            gradient_warm.append(time.perf_counter() - started)
        (_, auxiliary), gradient = cold_gradient
        gradient_norm = float(
            np.linalg.norm(
                np.concatenate(
                    [
                        np.asarray(leaf, dtype=np.float64).reshape(-1)
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    ]
                )
            )
        )
        hpsi_median = float(np.median(hpsi_warm))
        gradient_median = float(np.median(gradient_warm))
        rows.append(
            {
                "batch_size": batch_size,
                "connectivity_seconds_host": connectivity_seconds,
                "padded_width": padded_width,
                "actual_width_minimum": int(np.min(widths)),
                "actual_width_maximum": int(np.max(widths)),
                "translation_group_order": runtime.projector.group_order,
                "orbit_work_proxy": int(
                    batch_size * padded_width * runtime.projector.group_order
                ),
                "hpsi_cold_compile_and_call_seconds": hpsi_cold_seconds,
                "hpsi_warm_seconds": hpsi_warm,
                "hpsi_warm_median_seconds": hpsi_median,
                "hpsi_warm_throughput_states_per_second": batch_size / hpsi_median,
                "gradient_cold_compile_and_call_seconds": gradient_cold_seconds,
                "gradient_warm_seconds": gradient_warm,
                "gradient_warm_median_seconds": gradient_median,
                "gradient_warm_throughput_states_per_second": (
                    batch_size / gradient_median
                ),
                "gradient_norm": gradient_norm,
                "profile_energy": float(auxiliary["energy"]),
                "profile_energy_imaginary_abs": float(
                    auxiliary["energy_imaginary_abs"]
                ),
                "profile_minimum_center_magnitude": float(
                    auxiliary["minimum_center_magnitude"]
                ),
                "profile_maximum_local_energy_magnitude": float(
                    auxiliary["maximum_local_energy_magnitude"]
                ),
                "profile_nonfinite_local_count": int(
                    auxiliary["nonfinite_local_count"]
                ),
            }
        )
    return rows, compiled_hpsi, compiled_gradient


def _local_energy_batch(
    runtime: _ArmRuntime,
    compiled_hpsi: Callable,
    states: np.ndarray,
    bonds: tuple[tuple[int, int, float], ...],
    *,
    padded_width: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    locals_: list[np.ndarray] = []
    seconds = 0.0
    for start in range(0, len(states), batch_size):
        batch = np.asarray(states[start : start + batch_size], dtype=np.int32)
        if len(batch) != batch_size:
            raise ValueError("chain count must be divisible by local-energy batch")
        connected, elements, valid, _ = padded_spin_connectivity(
            batch, bonds, padded_width=padded_width
        )
        started = time.perf_counter()
        centers, actions = compiled_hpsi(
            runtime.params,
            jnp.asarray(batch),
            jnp.asarray(connected),
            jnp.asarray(elements),
            jnp.asarray(valid),
        )
        jax.block_until_ready((centers, actions))
        seconds += time.perf_counter() - started
        with np.errstate(divide="ignore", invalid="ignore"):
            locals_.append(
                np.asarray(actions, dtype=np.complex128)
                / np.asarray(centers, dtype=np.complex128)
            )
    return np.concatenate(locals_), seconds


def _summarize_diagnostics(
    rows: list[dict[str, float]], *, snapshot_changed: np.ndarray | None = None
) -> dict[str, object]:
    if not rows:
        return {}
    result: dict[str, object] = {}
    for name in rows[0]:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_minimum"] = float(np.min(values))
        result[f"{name}_maximum"] = float(np.max(values))
    if snapshot_changed is not None:
        changed = np.asarray(snapshot_changed, dtype=bool)
        result["snapshot_transition_move_fraction"] = float(np.mean(changed))
        result["frozen_chain_fraction"] = float(np.mean(~np.any(changed, axis=0)))
    return result


def _profile_rows_are_finite(rows: list[dict[str, object]]) -> bool:
    """Return whether every profiled Hpsi/gradient result is numerically safe."""

    if not rows:
        return False
    scalar_names = (
        "hpsi_cold_compile_and_call_seconds",
        "hpsi_warm_median_seconds",
        "hpsi_warm_throughput_states_per_second",
        "gradient_cold_compile_and_call_seconds",
        "gradient_warm_median_seconds",
        "gradient_warm_throughput_states_per_second",
        "gradient_norm",
        "profile_energy",
        "profile_energy_imaginary_abs",
        "profile_minimum_center_magnitude",
        "profile_maximum_local_energy_magnitude",
    )
    return all(
        int(row["profile_nonfinite_local_count"]) == 0
        and all(np.isfinite(float(row[name])) for name in scalar_names)
        and all(np.isfinite(float(value)) for value in row["hpsi_warm_seconds"])
        and all(np.isfinite(float(value)) for value in row["gradient_warm_seconds"])
        for row in rows
    )


def _run_arm(
    runtime: _ArmRuntime,
    *,
    spec: dict[str, object],
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    checkpoint_path: Path,
    identity: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    sampler = dict(spec["sampler"])
    profile = dict(spec["profile"])
    chains = int(sampler["chains"])
    proposals_per_sweep = int(sampler["proposals_per_sweep"])
    checkpoint_every = int(sampler["checkpoint_every_sweeps"])
    padded_width = len(bonds) + 1
    initial_key = jax.random.PRNGKey(int(sampler["seed"]))
    chain_key, sampling_key = jax.random.split(initial_key)
    initial_states_device = initialize_spin_exchange_chains(
        chain_key,
        n_chains=chains,
        n_sites=n_sites,
        n_up=n_sites // 2,
    )
    initial_states = np.asarray(initial_states_device, dtype=np.int32)
    initial_chain_states_sha256 = array_sha256(initial_states)
    common_profile_states = np.asarray(
        initialize_spin_exchange_chains(
            initial_key,
            n_chains=chains,
            n_sites=n_sites,
            n_up=n_sites // 2,
        ),
        dtype=np.int32,
    )
    common_profile_states_sha256 = array_sha256(common_profile_states)
    checkpoint = (
        None
        if overwrite
        else _load_checkpoint(
            checkpoint_path,
            identity,
            initial_chain_states_sha256=initial_chain_states_sha256,
            common_profile_states_sha256=common_profile_states_sha256,
        )
    )
    if checkpoint is None:
        amplitude_batch = jax.jit(
            jax.vmap(lambda row: runtime.projected_amplitude(runtime.params, row))
        )
        amplitudes_device = amplitude_batch(initial_states_device)
        jax.block_until_ready(amplitudes_device)
        states = initial_states
        amplitudes = np.asarray(amplitudes_device, dtype=np.complex128)
        key = np.asarray(sampling_key, dtype=np.uint32)
        burn_completed = 0
        measurement_completed = 0
        burn_diagnostics: list[dict[str, float]] = []
        measurement_diagnostics: list[dict[str, float]] = []
        energy_series: list[list[float]] = []
        ordering_first_series: list[list[float]] = []
        ordering_second_series: list[list[float]] = []
        snapshot_changed: list[list[bool]] = []
        performance_profile = None
        resume_count = 0
    else:
        states = np.asarray(checkpoint["states"], dtype=np.int32)
        amplitudes = _pairs_complex(checkpoint["amplitudes"])
        key = np.asarray(checkpoint["key"], dtype=np.uint32)
        burn_completed = int(checkpoint["burn_completed"])
        measurement_completed = int(checkpoint["measurement_completed"])
        burn_diagnostics = list(checkpoint["burn_diagnostics"])
        measurement_diagnostics = list(checkpoint["measurement_diagnostics"])
        energy_series = list(checkpoint["energy_series"])
        ordering_first_series = list(checkpoint["ordering_first_series"])
        ordering_second_series = list(checkpoint["ordering_second_series"])
        snapshot_changed = list(checkpoint["snapshot_changed"])
        performance_profile = checkpoint["performance_profile"]
        resume_count = int(checkpoint["resume_count"]) + 1

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
            bond_proposal_probability=float(
                sampler["bond_proposal_probability"]
            ),
        )

    compiled_advance = jax.jit(advance)

    def one_sweep() -> dict[str, float]:
        nonlocal states, amplitudes, key
        states_device, amplitudes_device, key_device, diagnostics = compiled_advance(
            runtime.params,
            jnp.asarray(key),
            jnp.asarray(states),
            jnp.asarray(amplitudes),
        )
        jax.block_until_ready((states_device, amplitudes_device, diagnostics))
        states = np.asarray(states_device, dtype=np.int32)
        amplitudes = np.asarray(amplitudes_device, dtype=np.complex128)
        key = np.asarray(key_device, dtype=np.uint32)
        return _float_diagnostics(diagnostics)

    def checkpoint_now(status: str) -> dict[str, object]:
        return _write_checkpoint(
            checkpoint_path,
            identity=identity,
            initial_chain_states_sha256=initial_chain_states_sha256,
            common_profile_states_sha256=common_profile_states_sha256,
            status=status,
            states=states,
            amplitudes=amplitudes,
            key=key,
            burn_completed=burn_completed,
            measurement_completed=measurement_completed,
            burn_diagnostics=burn_diagnostics,
            measurement_diagnostics=measurement_diagnostics,
            energy_series=energy_series,
            ordering_first_series=ordering_first_series,
            ordering_second_series=ordering_second_series,
            snapshot_changed=snapshot_changed,
            performance_profile=performance_profile,
            resume_count=resume_count,
        )

    total_started = time.perf_counter()
    while burn_completed < int(sampler["burn_in_sweeps"]):
        burn_diagnostics.append(one_sweep())
        burn_completed += 1
        if burn_completed % checkpoint_every == 0:
            checkpoint_now("running")

    if performance_profile is None:
        performance_profile, compiled_hpsi, _ = _profile_kernels(
            runtime,
            common_profile_states,
            bonds,
            padded_width=padded_width,
            batch_sizes=[int(value) for value in profile["batch_sizes"]],
            warm_repeats=int(profile["warm_repeats"]),
        )
        checkpoint_now("running")
    else:
        # Recreate compiled kernels after a fresh-process resume.  The replayed
        # compile is intentionally not folded into the original profile.
        _, compiled_hpsi, _ = _profile_kernels(
            runtime,
            common_profile_states,
            bonds,
            padded_width=padded_width,
            batch_sizes=[max(int(value) for value in profile["batch_sizes"])],
            warm_repeats=1,
        )

    measurement_batch = max(int(value) for value in profile["batch_sizes"])
    if chains % measurement_batch:
        raise ValueError("chain count must be divisible by largest profile batch")
    previous_states = states.copy()
    local_energy_seconds = 0.0
    while measurement_completed < int(sampler["measurement_sweeps"]):
        diagnostics = one_sweep()
        measurement_diagnostics.append(diagnostics)
        changed = np.any(states != previous_states, axis=1)
        snapshot_changed.append(changed.tolist())
        previous_states = states.copy()
        local, elapsed = _local_energy_batch(
            runtime,
            compiled_hpsi,
            states,
            bonds,
            padded_width=padded_width,
            batch_size=measurement_batch,
        )
        local_energy_seconds += elapsed
        energy_series.append(np.real(local).astype(float).tolist())
        ordering = _ordering_structure_factor_series(
            states,
            geometry=str(case["geometry"]),
            rows=rows,
            columns=columns,
        )
        ordering_names = list(spec["observable_contract"]["reported_estimators"])
        ordering_first_series.append(
            ordering[ordering_names[0]].astype(float).tolist()
        )
        ordering_second_series.append(
            ordering[ordering_names[1]].astype(float).tolist()
        )
        measurement_completed += 1
        if measurement_completed % checkpoint_every == 0:
            checkpoint_now("running")

    energy = np.asarray(energy_series, dtype=np.float64)
    ordering_names = list(spec["observable_contract"]["reported_estimators"])
    ordering_first = np.asarray(ordering_first_series, dtype=np.float64)
    ordering_second = np.asarray(ordering_second_series, dtype=np.float64)
    changed_values = np.asarray(snapshot_changed, dtype=bool)
    energy_correlation = _autocorrelation_summary(energy)
    ordering_first_correlation = _autocorrelation_summary(ordering_first)
    ordering_second_correlation = _autocorrelation_summary(ordering_second)
    final_local, final_elapsed = _local_energy_batch(
        runtime,
        compiled_hpsi,
        states,
        bonds,
        padded_width=padded_width,
        batch_size=measurement_batch,
    )
    local_energy_seconds += final_elapsed
    imaginary_maximum = float(np.max(np.abs(np.imag(final_local))))
    nonfinite_local_count = int(np.count_nonzero(~np.isfinite(final_local)))
    measurement_summary = _summarize_diagnostics(
        measurement_diagnostics, snapshot_changed=changed_values
    )
    ess_values = [
        float(row["ess"])
        for row in (
            energy_correlation,
            ordering_first_correlation,
            ordering_second_correlation,
        )
    ]
    acceptance = float(
        measurement_summary.get("conditional_move_acceptance_mean", 0.0)
    )
    frozen = float(measurement_summary.get("frozen_chain_fraction", 1.0))
    zero_amplitude_maximum = float(
        measurement_summary.get("zero_amplitude_chain_fraction_maximum", 1.0)
    )
    gates = {
        "finite_local_energies": nonfinite_local_count == 0,
        "local_energy_imaginary_abs_below_1e_minus_10": (
            imaginary_maximum < 1.0e-10
        ),
        "profile_gradients_finite_and_no_nonfinite_local": (
            _profile_rows_are_finite(performance_profile)
        ),
        "conditional_move_acceptance_between_0p1_and_0p9": (
            0.1 <= acceptance <= 0.9
        ),
        "no_frozen_chains": frozen == 0.0,
        "no_zero_amplitude_chains": zero_amplitude_maximum == 0.0,
        "ess_each_above_50": min(ess_values) > 50.0,
        "profile_batches_1_2_4_8_present": {
            int(row["batch_size"]) for row in performance_profile
        }
        >= {1, 2, 4, 8},
    }
    checkpoint_payload = checkpoint_now("complete")
    energy_mean = float(np.mean(energy))
    energy_ess = max(float(energy_correlation["ess"]), 1.0)
    energy_mcse = float(np.std(energy, ddof=1) / np.sqrt(energy_ess))
    trial: dict[str, object] = {
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "projection": "translation",
        "feature_dim": runtime.feature_dim,
        "maximum_intermediate_q": runtime.maximum_intermediate_q,
        "seed": int(next(arm["seed"] for arm in spec["arms"] if arm["arm_id"] == runtime.arm_id)),
        "parameter_count": runtime.parameter_count,
        "initial_params_sha256": pytree_sha256(runtime.params),
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "common_profile_states_sha256": common_profile_states_sha256,
        "translation_group_order": runtime.projector.group_order,
        "padded_connectivity_width": padded_width,
        "sampler": {
            "chains": chains,
            "burn_in_sweeps": int(sampler["burn_in_sweeps"]),
            "measurement_sweeps": int(sampler["measurement_sweeps"]),
            "proposals_per_sweep": proposals_per_sweep,
            "bond_proposal_probability": float(
                sampler["bond_proposal_probability"]
            ),
            "global_proposal_probability": 1.0
            - float(sampler["bond_proposal_probability"]),
            "burn_diagnostics": _summarize_diagnostics(burn_diagnostics),
            "measurement_diagnostics": measurement_summary,
            "resume_count": resume_count,
        },
        "observables": {
            "estimator_contract": dict(spec["observable_contract"]),
            "energy_mean_random_state": energy_mean,
            "energy_per_site_random_state": energy_mean / n_sites,
            "energy_mcmc_standard_error": energy_mcse,
            "energy": energy_correlation,
            ordering_names[0]: {
                "mean": float(np.mean(ordering_first)),
                **ordering_first_correlation,
            },
            ordering_names[1]: {
                "mean": float(np.mean(ordering_second)),
                **ordering_second_correlation,
            },
            "local_energy_imaginary_abs_maximum": imaginary_maximum,
            "nonfinite_local_energy_count": nonfinite_local_count,
            "local_energy_kernel_seconds": local_energy_seconds,
            "interpretation": (
                "random-initialization mixing diagnostics only; diagonal Szz "
                "ordering estimators are slow-mode probes, not full SU(2)-"
                "invariant structure factors or phase estimates"
            ),
        },
        "performance_profile": performance_profile,
        "largest_configured_successful_batch": max(
            int(row["batch_size"]) for row in performance_profile
        ),
        "readiness_gates": gates,
        "all_readiness_gates_passed": all(gates.values()),
        "checkpoint": {
            "path": checkpoint_path.name,
            "status": checkpoint_payload["status"],
            "checkpoint_sha256": checkpoint_payload["checkpoint_sha256"],
            "deterministic_state_owned": [
                "chains",
                "cached_amplitudes",
                "jax_prng_key",
                "initial_chain_states_sha256",
                "common_profile_states_sha256",
                "burn_and_measure_counters",
                "time_series",
            ],
        },
        "wall_seconds_including_compile_profile_and_sampling": (
            time.perf_counter() - total_started
        ),
        "claim_boundary": (
            "on-demand runtime and Markov-chain readiness only; no energy "
            "accuracy, optimization, phase, or physics-win claim"
        ),
    }
    return trial


def run_spin_sampled_runtime(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume one physical-input spin MCMC/runtime smoke."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    manifest = _source_manifest()
    source_hash = canonical_sha256(manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    expected_header = (config_hash, source_hash, runtime_hash)
    if result_path.exists() and not overwrite:
        payload = json.loads(result_path.read_text())
        if payload.get("schema_version") != RUNTIME_RESULT_SCHEMA_VERSION:
            raise RuntimeError(
                f"sampled runtime schema mismatch: {result_path}; expected "
                f"{RUNTIME_RESULT_SCHEMA_VERSION}"
            )
        observed_header = (
            payload.get("config_sha256"),
            payload.get("source_manifest_sha256"),
            payload.get("runtime_identity_sha256"),
        )
        if observed_header != expected_header:
            raise RuntimeError(f"refusing stale sampled runtime: {result_path}")
        if payload.get("status") == "complete":
            if payload.get("trial_manifest_sha256") != canonical_sha256(
                embedded_trial_manifest(payload)
            ):
                raise RuntimeError(f"sampled runtime trial seal failed: {result_path}")
            print(f"reused immutable complete sampled runtime: {result_path}", flush=True)
            return payload
    else:
        payload = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "domain": "spin",
            "task": "physical_mcmc_runtime_smoke",
            "status": "running",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": _portable_config_path(config_path),
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
                "physical_spatial_projector_used": True,
                "on_demand_spin_hpsi_used": True,
                "energy_or_physics_win_claimed": False,
            },
            "cases": [],
        }
        atomic_json(result_path, payload)

    case_spec = dict(spec["case"])
    rows, columns = int(case_spec["rows"]), int(case_spec["columns"])
    n_sites = rows * columns
    geometry = str(case_spec["geometry"])
    if geometry == "triangular":
        bonds = triangular_j1j2_bonds(
            rows,
            columns,
            j1=1.0,
            j2=float(case_spec["j2_over_j1"]),
            periodic=True,
        )
        group = triangular_hnf_translation_group(((columns, 0), (0, rows)))
    elif geometry == "square":
        bonds = square_j1j2_bonds(
            rows,
            columns,
            j1=1.0,
            j2=float(case_spec["j2_over_j1"]),
            periodic=True,
        )
        group = rectangular_translation_group(rows, columns)
    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"unknown geometry: {geometry!r}")
    projection_spec = dict(spec["projection"])
    projector = PhysicalSpatialProjector(
        group,
        orbit_chunk_size=int(projection_spec["orbit_chunk_size"]),
        rematerialize_orbit=bool(projection_spec["rematerialize_orbit"]),
    )
    if payload["cases"]:
        case_payload = payload["cases"][0]
    else:
        case_payload = {
            "case_id": str(case_spec["case_id"]),
            "geometry": geometry,
            "rows": rows,
            "columns": columns,
            "n_sites": n_sites,
            "j2_over_j1": float(case_spec["j2_over_j1"]),
            "periodic": True,
            "support": "fixed_m0",
            "ed_performed": False,
            "bond_count": len(bonds),
            "bonds_sha256": canonical_sha256(bonds),
            "translation_group_order": projector.group_order,
            "translation_group_sha256": canonical_sha256(group),
            "site_order": list(morton_order(rows, columns)),
            "trials": [],
        }
        payload["cases"].append(case_payload)
        atomic_json(result_path, payload)
    completed = {str(trial["arm_id"]) for trial in case_payload["trials"]}
    for arm in spec["arms"]:
        if str(arm["arm_id"]) in completed:
            continue
        runtime = _initialize_arm(
            arm, rows=rows, columns=columns, projector=projector
        )
        identity = _checkpoint_identity(
            config_hash=config_hash,
            source_hash=source_hash,
            runtime_hash=runtime_hash,
            case_id=str(case_spec["case_id"]),
            runtime=runtime,
        )
        checkpoint_path = output / f"checkpoint_{runtime.arm_id}.json"
        trial = _run_arm(
            runtime,
            spec=spec,
            case=case_spec,
            bonds=bonds,
            checkpoint_path=checkpoint_path,
            identity=identity,
            overwrite=overwrite,
        )
        trial["trial_payload_sha256"] = trial_payload_sha256(trial)
        case_payload["trials"].append(trial)
        atomic_json(result_path, payload)
        print(
            f"sampled-runtime {case_spec['case_id']} {runtime.arm_id}: "
            f"accept={trial['sampler']['measurement_diagnostics']['conditional_move_acceptance_mean']:.3f} "
            f"ESS(E)={trial['observables']['energy']['ess']:.1f}",
            flush=True,
        )
    manifest_rows = embedded_trial_manifest(payload)
    payload["trial_manifest"] = manifest_rows
    payload["trial_manifest_sha256"] = canonical_sha256(manifest_rows)
    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_sampled_runtime"]
