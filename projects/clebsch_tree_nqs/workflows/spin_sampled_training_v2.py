"""Versioned center-only Adam learning-rate scan for 6x6 spin systems.

This workflow is intentionally separate from ``spin_sampled_training.py``.
The v1 workflow and its artifacts remain sealed; v2 opts into the versioned
center-only gradient backend and has its own result/checkpoint identities.

Each config contains one learning-rate candidate.  Configs in a scan family
share the same model, chain, sampling, and measurement seeds.  Learning rate
enters the run identity but never an RNG fold.  The registered family can
therefore compare three trajectories that start from exactly the same
parameters, post-burn chains, and training random key for each
geometry/backbone arm.

The scan is an optimizer diagnostic, not a physics or efficiency result.
Independent fixed-parameter blocks report only E/N and Var(H)/N at registered
updates.  Wall time is never a candidate-selection criterion.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime_workflow
from projects.clebsch_tree_nqs.workflows import spin_sampled_training as v1_helpers
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
from su2_nqs.vmc.physical_exchange import (
    initialize_spin_exchange_chains,
    run_exchange_sweeps,
)
from su2_nqs.vmc.physical_sampled_adam_center_only import (
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
from su2_nqs.vmc.spin_physical import spin_physical_hpsi


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TRAINING_V2_RESULT_SCHEMA_VERSION = 2
OPTIMIZER_BACKEND = "center_only_sampled_adam_v2"
TASK = "physical_sampled_optimizer_scan_v2"
ALLOWED_BACKBONES = {"u1", "su2"}
CLAIM_BOUNDARY = (
    "single-seed center-only Adam learning-rate diagnostic only; no converged-"
    "energy, wall-time efficiency, magnetic-order, phase, QSL, or physics-win claim"
)


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


def _finite_positive(value: object, name: str) -> float:
    concrete = float(value)
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return concrete


def _common_scan_profile(spec: dict[str, object]) -> dict[str, object]:
    """Return the choices that must be identical across learning rates."""

    profile = deepcopy(spec)
    profile.pop("candidate_id", None)
    optimizer = dict(profile["optimizer"])
    optimizer.pop("learning_rate", None)
    profile["optimizer"] = optimizer
    return profile


def common_scan_profile_sha256(spec: dict[str, object]) -> str:
    return canonical_sha256(_common_scan_profile(spec))


def _expected_selection_policy(final_update: int) -> dict[str, object]:
    """Return the fully preregistered scan-to-heldout decision policy."""

    return {
        "endpoint_updates": [0, final_update],
        "required_energy_lowering_per_site": 0.03,
        "minimum_energy_lowering_combined_mcse": 3.0,
        "maximum_variance_worsening_combined_mcse": 2.0,
        "geometry_common_lr": True,
        "candidate_gap_tie_combined_mcse": 2.0,
        "tie_choose_smaller_lr": True,
        "arm_specific_optimum_diagnostic_only": True,
        "endpoint_uncertainty": {
            "primary": "same_arm_same_chain_lineage_paired_difference_se",
            "retain_independent_endpoint_hypot_diagnostic": True,
        },
        "stability_gates": {
            "last10_accepted_move_fraction_median_minimum": 0.15,
            "relative_update_quantile": 0.99,
            "relative_update_quantile_maximum": 0.02,
        },
        "geometry_selection": {
            "score": "equal_weight_mean_u1_su2_signed_energy_lowering_per_site",
            "arm_weights": {"u1": 0.5, "su2": 0.5},
            "z_scores_averaged": False,
            "wall_time_used": False,
            "candidate_requires_both_arms_pass": [
                "energy_endpoint_gate",
                "variance_endpoint_gate",
                "stability_gate",
            ],
            "raw_best": "highest_geometry_score_among_eligible_candidates",
            "tie_reference": "raw_best_candidate",
            "same_arm_cross_lr_uncertainty": (
                "paired_E50_chain_mean_difference_se"
            ),
            "cross_arm_uncertainty_combination": "worst_case_sum_upper_bound",
            "geometry_score_difference_mcse_formula": "0.5*(se_u1+se_su2)",
            "tie_threshold_combined_mcse": 2.0,
            "tie_choose_smaller_lr": True,
        },
        "continuation_if_no_eligible_candidate": {
            "maximum_uses": 1,
            "resume_selected_step50_checkpoint": True,
            "from_update": final_update,
            "to_update": 2 * final_update,
            "pool_requires_both_arms_stability_gate": True,
            "pool_requires_both_arms_variance_endpoint_gate": True,
            "pool_minimum_each_arm_energy_lowering_paired_mcse": -2.0,
            "pool_geometry_score_must_be_strictly_positive": True,
            "selection_rule_reused_without_change": True,
            "only_selected_candidate_continues": True,
            "learning_rate_locked": True,
            "run_other_candidates": False,
            "promotion_0_to_100_each_arm": {
                "minimum_energy_lowering_per_site": 0.05,
                "minimum_energy_lowering_paired_mcse": 5.0,
                "minimum_variance_fractional_decrease": 0.25,
                "minimum_variance_decrease_paired_mcse": 3.0,
            },
            "failure_action": "transition_to_matrix_free_physical_sr",
        },
        "heldout_policy": {
            "common_learning_rate_fixed_per_geometry": True,
            "chains": 64,
            "updates": 200,
            "initial_seeds": [23, 47],
            "measurement_updates": [0, 100, 200],
            "full_spin_structure_factor_deferred_until_heldout": True,
            "full_spin_structure_factor_updates": [0, 100, 200],
            "each_seed_each_arm_safety_gates": {
                "minimum_energy_lowering_per_site": 0.05,
                "minimum_energy_lowering_paired_mcse": 5.0,
                "minimum_variance_fractional_decrease": 0.25,
                "minimum_variance_decrease_paired_mcse": 3.0,
            },
            "efficiency_claim_requires_both_arms_all_initial_seeds": True,
            "additional_confirmation_seed_if_pass": 89,
            "efficiency_claim_requires_confirmation_seed": True,
            "confirmation_seed_must_pass_each_arm_safety_gates": True,
            "time_to_target": {
                "metric": (
                    "first_registered_work_reaching_absolute_energy_per_site_target"
                ),
                "absolute_energy_per_site_targets": {
                    "square_6x6_j2_0p5": None,
                    "triangular_6x6_j2_0p125": None,
                },
                "must_be_frozen_from_convention_and_graph_matched_published_reference_before_heldout_run": True,
                "forbid_square_8x8_target_for_square_6x6": True,
                "heldout_config_forbidden_until_targets_are_nonnull": True,
            },
        },
    }


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("optimizer scan v2 requires schema_version=2 and domain='spin'")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    if spec.get("optimizer_backend") != OPTIMIZER_BACKEND:
        raise ValueError(f"optimizer_backend must be {OPTIMIZER_BACKEND}")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("config claim_boundary must match the enforced v2 boundary")
    for name in ("scan_family_id", "candidate_id"):
        if not isinstance(spec.get(name), str) or not str(spec[name]):
            raise ValueError(f"{name} must be a nonempty string")

    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen_cases:
            raise ValueError("case_id values must be nonempty and unique")
        seen_cases.add(case_id)
        geometry = str(case.get("geometry"))
        if geometry not in {"square", "triangular"}:
            raise ValueError("optimizer scan supports square and triangular geometry")
        rows = _positive_integer(case.get("rows"), "case.rows")
        columns = _positive_integer(case.get("columns"), "case.columns")
        if rows * columns % 2:
            raise ValueError("fixed-M2=0 training requires an even site count")
        if case.get("periodic") is not True or case.get("ed") is not False:
            raise ValueError("optimizer scan cases require periodic=true and ed=false")
        if case.get("support") != "fixed_m0":
            raise ValueError("optimizer scan cases require fixed_m0 support")
        expected_convention = (
            "row_major_cartesian_column_row"
            if geometry == "square"
            else "row_major_triangular_phase_row_column"
        )
        if case.get("coordinate_convention") != expected_convention:
            raise ValueError("coordinate convention does not match geometry")
        if not math.isfinite(float(case.get("j2_over_j1", np.nan))):
            raise ValueError("j2_over_j1 must be finite")

    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection.get("kind") != "translation":
        raise ValueError("optimizer scan requires translation projection")
    _positive_integer(projection.get("orbit_chunk_size"), "orbit_chunk_size")
    if not isinstance(projection.get("rematerialize_orbit"), bool):
        raise TypeError("rematerialize_orbit must be boolean")

    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty list")
    seen_arms: set[str] = set()
    backbones: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise TypeError("every arm must be an object")
        arm_id = str(arm.get("arm_id", ""))
        if not arm_id or arm_id in seen_arms:
            raise ValueError("arm_id values must be nonempty and unique")
        seen_arms.add(arm_id)
        backbone = str(arm.get("backbone"))
        if backbone not in ALLOWED_BACKBONES:
            raise ValueError("backbone must be u1 or su2")
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
        raise ValueError("registered optimizer scan requires matched U1/SU2 arms")

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
    probability = float(sampler.get("bond_proposal_probability", np.nan))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0,1]")

    optimizer = spec.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "local_adam":
        raise ValueError("optimizer must be local_adam")
    for name in ("updates", "microbatch_size", "checkpoint_every_updates"):
        _positive_integer(optimizer.get(name), f"optimizer.{name}")
    chains = int(sampler["chains"])
    microbatch = int(optimizer["microbatch_size"])
    if chains % microbatch or microbatch > MAX_PHYSICAL_MICROBATCH_SIZE:
        raise ValueError("microbatch_size must divide chains and remain at most eight")
    for name in ("learning_rate", "epsilon", "maximum_relative_update_norm"):
        _finite_positive(optimizer.get(name), f"optimizer.{name}")
    if float(optimizer["maximum_relative_update_norm"]) != 0.05:
        raise ValueError("optimizer scan v2 hard relative-update limit must be 0.05")
    for name in ("beta1", "beta2"):
        value = float(optimizer.get(name, np.nan))
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"optimizer.{name} must lie in [0,1)")

    measurement = spec.get("measurement")
    if not isinstance(measurement, dict):
        raise TypeError("measurement must be an object")
    if measurement.get("energy_variance_only") is not True:
        raise ValueError("v2 scan measurement must be energy_variance_only")
    for name in ("burn_sweeps", "measurement_sweeps"):
        _positive_integer(measurement.get(name), f"measurement.{name}")
    if isinstance(measurement.get("seed"), bool) or not isinstance(
        measurement.get("seed"), int
    ):
        raise TypeError("measurement.seed must be an integer")
    updates = measurement.get("updates")
    if (
        not isinstance(updates, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in updates)
        or updates != sorted(set(updates))
        or not updates
        or updates[0] != 0
        or updates[-1] != int(optimizer["updates"])
    ):
        raise ValueError("measurement.updates must be sorted unique and span 0..updates")

    diagnostics = spec.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError("diagnostics must be an object")
    expected_diagnostics = {
        "acceptance_window": 10,
        "low_acceptance_median_threshold": 0.15,
        "relative_update_quantile": 0.99,
        "relative_update_threshold": 0.02,
    }
    if diagnostics != expected_diagnostics:
        raise ValueError("diagnostics must match the preregistered stability policy")

    policy = spec.get("selection_policy")
    if not isinstance(policy, dict):
        raise TypeError("selection_policy must be an object")
    expected_policy = _expected_selection_policy(int(optimizer["updates"]))
    if policy != expected_policy:
        raise ValueError("selection_policy must exactly match the preregistered policy")


def _compiled_kernels(
    runtime: runtime_workflow._ArmRuntime,
    bonds: tuple[tuple[int, int, float], ...],
) -> tuple[Callable, Callable, Callable, Callable]:
    """Build sampler, Hpsi, and genuinely center-only static kernels."""

    exchange_edges = tuple((first, second) for first, second, _ in bonds)

    def advance_uncached(candidate, key, states, attempts, probability):
        return run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=runtime.projected_amplitude,
            particle_kind="spin",
            sweeps=attempts,
            amplitudes=None,
            exchange_edges=exchange_edges,
            bond_proposal_probability=probability,
        )

    def advance_cached(candidate, key, states, amplitudes, attempts, probability):
        return run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=runtime.projected_amplitude,
            particle_kind="spin",
            sweeps=attempts,
            amplitudes=amplitudes,
            exchange_edges=exchange_edges,
            bond_proposal_probability=probability,
        )

    def projected_for_hpsi(candidate, state, _config):
        return runtime.projected_amplitude(candidate, state)

    def hpsi(candidate, batch):
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

    def centers(candidate, batch):
        samples = batch[0]
        return jax.vmap(
            lambda state: runtime.projected_amplitude(candidate, state)
        )(samples)

    return (
        jax.jit(advance_uncached, static_argnums=(3, 4)),
        jax.jit(advance_cached, static_argnums=(4, 5)),
        jax.jit(hpsi),
        jax.jit(centers),
    )


def _measurement_fold(
    *, spec: dict[str, object], case_id: str, arm_id: str, update: int
) -> int:
    # Candidate/LR intentionally absent: all candidates use the same stream.
    digest = canonical_sha256(
        {
            "purpose": "center_only_optimizer_scan_fixed_energy_measurement",
            "measurement_seed": int(spec["measurement"]["seed"]),
            "scan_family_id": spec["scan_family_id"],
            "case_id": case_id,
            "arm_id": arm_id,
            "update": int(update),
        }
    )
    return int(digest[:8], 16)


def _fixed_energy_measurement(
    *,
    runtime: runtime_workflow._ArmRuntime,
    params: object,
    training_states: np.ndarray,
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    spec: dict[str, object],
    update: int,
    advance_uncached: Callable,
    advance_cached: Callable,
    hpsi_fn: Callable,
) -> tuple[dict[str, object], dict[str, int]]:
    sampler = dict(spec["sampler"])
    measurement = dict(spec["measurement"])
    chains = int(sampler["chains"])
    proposals = int(sampler["proposals_per_sweep"])
    probability = float(sampler["bond_proposal_probability"])
    microbatch = int(spec["optimizer"]["microbatch_size"])
    padded_width = len(bonds) + 1
    n_sites = int(case["rows"]) * int(case["columns"])
    params_hash = pytree_sha256(params)
    training_hash = array_sha256(training_states)

    measurement_fold = _measurement_fold(
        spec=spec,
        case_id=str(case["case_id"]),
        arm_id=runtime.arm_id,
        update=update,
    )
    key = jax.random.fold_in(
        jax.random.PRNGKey(int(measurement["seed"])),
        measurement_fold,
    )
    states = jnp.asarray(np.array(training_states, copy=True), dtype=jnp.int32)
    burn_attempts = int(measurement["burn_sweeps"]) * proposals
    states, amplitudes, key, burn_row = advance_uncached(
        params, key, states, burn_attempts, probability
    )
    jax.block_until_ready((states, amplitudes, key, burn_row))
    sampler_rows = [v1_helpers._float_diagnostics(burn_row)]
    local_rows: list[np.ndarray] = []
    actual_connected = 0
    for _ in range(int(measurement["measurement_sweeps"])):
        states, amplitudes, key, row = advance_cached(
            params, key, states, amplitudes, proposals, probability
        )
        jax.block_until_ready((states, amplitudes, key, row))
        sampler_rows.append(v1_helpers._float_diagnostics(row))
        host_states = np.asarray(states, dtype=np.int32)
        batches, widths, _seconds = v1_helpers._hpsi_batches(
            host_states,
            bonds,
            padded_width=padded_width,
            microbatch_size=microbatch,
        )
        actual_connected += int(np.sum(widths))
        chunks: list[np.ndarray] = []
        for batch in batches:
            centers, actions = hpsi_fn(params, batch)
            jax.block_until_ready((centers, actions))
            chunks.append(
                np.asarray(actions, dtype=np.complex128)
                / np.asarray(centers, dtype=np.complex128)
            )
        local_rows.append(np.concatenate(chunks))

    if params_hash != pytree_sha256(params):
        raise RuntimeError("fixed-energy measurement mutated model parameters")
    if training_hash != array_sha256(training_states):
        raise RuntimeError("fixed-energy measurement mutated training chains")
    local = np.stack(local_rows, axis=0)
    if np.any(~np.isfinite(local)):
        raise FloatingPointError("fixed-energy measurement has non-finite local energy")
    flat = local.reshape(-1)
    mean = complex(np.mean(flat))
    variance = float(np.mean(np.abs(flat - mean) ** 2).real)
    chain_energy = np.asarray(
        np.mean(local.real, axis=0) / n_sites, dtype=np.float64
    )
    energy_mcse = float(np.std(chain_energy, ddof=1) / np.sqrt(chains))
    chain_variance_contributions = np.asarray(
        np.mean(np.abs(local - mean) ** 2, axis=0) / n_sites,
        dtype=np.float64,
    )
    variance_mcse = float(
        np.std(chain_variance_contributions, ddof=1) / np.sqrt(chains)
    )
    diagnostics = runtime_workflow._summarize_diagnostics(sampler_rows)
    sample_count = int(local.size)
    result = {
        "update": int(update),
        "fixed_params_sha256": params_hash,
        "training_chains_sha256": training_hash,
        "params_frozen_throughout_block": True,
        "training_chains_copied_and_not_mutated": True,
        "independent_measurement_random_stream": True,
        "measurement_random_stream_fold": measurement_fold,
        "measurement_random_stream_identity_sha256": canonical_sha256(
            {
                "measurement_seed": int(measurement["seed"]),
                "fold": measurement_fold,
                "case_id": str(case["case_id"]),
                "arm_id": runtime.arm_id,
                "update": int(update),
            }
        ),
        "energy_per_site": float(mean.real / n_sites),
        "energy_per_site_chain_mcse": energy_mcse,
        "energy_per_site_chain_means": chain_energy.tolist(),
        "energy_per_site_chain_means_sha256": array_sha256(chain_energy),
        "energy_imaginary_abs": float(abs(mean.imag)),
        "variance_h_per_site": float(variance / n_sites),
        "variance_h_per_site_chain_mcse": variance_mcse,
        "variance_h_per_site_chain_contributions": (
            chain_variance_contributions.tolist()
        ),
        "variance_h_per_site_chain_contributions_sha256": array_sha256(
            chain_variance_contributions
        ),
        "chain_index_lineage": (
            "persistent_training_chain_index_then_independent_measurement_copy"
        ),
        "raw_measurement_sweeps_stored": False,
        "sample_count": sample_count,
        "chain_count": chains,
        "measurement_sweeps": int(measurement["measurement_sweeps"]),
        "samples_are_autocorrelated": True,
        "chain_mcse_does_not_pool_optimizer_updates": True,
        "sampler_diagnostics": diagnostics,
        "full_spin_structure_factor_computed": False,
    }
    sampling_projected = chains * (
        1
        + proposals
        * (int(measurement["burn_sweeps"]) + int(measurement["measurement_sweeps"]))
    )
    hpsi_projected = sample_count * (1 + padded_width)
    ledger = {
        "fixed_parameter_energy_samples": sample_count,
        "measurement_sampler_projected_amplitude_evaluations": sampling_projected,
        "measurement_hpsi_forward_projected_amplitude_evaluations": hpsi_projected,
        "measurement_hpsi_actual_connected_configurations": actual_connected,
        "measurement_hpsi_padded_connected_configurations": sample_count
        * padded_width,
        "measurement_base_tree_orbit_image_work_proxy": (
            (sampling_projected + hpsi_projected) * runtime.projector.group_order
        ),
    }
    return result, ledger


def _run_identity(
    *,
    config_hash: str,
    source_hash: str,
    runtime_hash: str,
    common_profile_hash: str,
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
        "schema": "physical_sampled_spin_adam_center_only_v2",
        "task": TASK,
        "optimizer_backend": OPTIMIZER_BACKEND,
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "runtime_identity_sha256": runtime_hash,
        "common_scan_profile_sha256": common_profile_hash,
        "scan_family_id": spec["scan_family_id"],
        "candidate_id": spec["candidate_id"],
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
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "projection": dict(spec["projection"]),
        "sampler": dict(spec["sampler"]),
        "optimizer": dict(spec["optimizer"]),
        "measurement": dict(spec["measurement"]),
        "diagnostics": dict(spec["diagnostics"]),
        "selection_policy": dict(spec["selection_policy"]),
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
    initial_post_burn_chain_states_sha256: str,
    initial_params_sha256: str,
    initial_training_key_sha256: str,
    initial_post_burn_training_key_sha256: str,
    initial_burn_diagnostics: dict[str, float],
    initial_burn_work_ledger: dict[str, int],
    measurements: list[dict[str, object]],
    resume_count: int,
) -> dict[str, object]:
    return {
        "optimizer_backend": OPTIMIZER_BACKEND,
        "chain_states": np.asarray(states, dtype=np.int32),
        "chain_states_sha256": array_sha256(np.asarray(states, dtype=np.int32)),
        "sampling_key": np.asarray(sampling_key, dtype=np.uint32),
        "sampling_key_sha256": array_sha256(np.asarray(sampling_key, dtype=np.uint32)),
        "initial_chain_states_sha256": initial_chain_states_sha256,
        "initial_post_burn_chain_states_sha256": initial_post_burn_chain_states_sha256,
        "initial_params_sha256": initial_params_sha256,
        "initial_training_key_sha256": initial_training_key_sha256,
        "initial_post_burn_training_key_sha256": (
            initial_post_burn_training_key_sha256
        ),
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "initial_burn_completed": True,
        "initial_burn_diagnostics": initial_burn_diagnostics,
        "initial_burn_work_ledger": initial_burn_work_ledger,
        "fixed_parameter_measurements": measurements,
        "resume_count": int(resume_count),
        "amplitude_cache_present": False,
        "amplitude_cache_policy": "never checkpointed; recompute after every update",
    }


def _load_checkpoint(
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
        raise RuntimeError("v2 checkpoint metadata is invalid")
    if metadata.get("optimizer_backend") != OPTIMIZER_BACKEND:
        raise RuntimeError("refusing non-v2 optimizer checkpoint")
    if metadata.get("amplitude_cache_present") is not False or "amplitudes" in metadata:
        raise RuntimeError("v2 checkpoint must not retain amplitude caches")
    if metadata.get("initial_chain_states_sha256") != expected_initial_chain_hash:
        raise RuntimeError("v2 checkpoint initial-chain identity mismatch")
    if metadata.get("initial_params_sha256") != expected_initial_params_hash:
        raise RuntimeError("v2 checkpoint initial-parameter identity mismatch")
    states = np.asarray(metadata.get("chain_states"), dtype=np.int32)
    if states.shape != (n_chains, n_sites):
        raise RuntimeError("v2 checkpoint has the wrong chain shape")
    if np.any(np.sum(states, axis=1) != n_sites // 2):
        raise RuntimeError("v2 checkpoint left fixed-M2=0 support")
    if metadata.get("chain_states_sha256") != array_sha256(states):
        raise RuntimeError("v2 checkpoint chain-state seal mismatch")
    sampling_key = np.asarray(metadata.get("sampling_key"), dtype=np.uint32)
    if metadata.get("sampling_key_sha256") != array_sha256(sampling_key):
        raise RuntimeError("v2 checkpoint training-key seal mismatch")
    if metadata.get("params_sha256") != pytree_sha256(payload["params"]):
        raise RuntimeError("v2 checkpoint parameter seal mismatch")
    state = payload["optimizer_state"]
    if not isinstance(state, SampledAdamState):
        raise RuntimeError("v2 checkpoint optimizer type mismatch")
    if metadata.get("optimizer_state_sha256") != pytree_sha256(state):
        raise RuntimeError("v2 checkpoint optimizer seal mismatch")
    if state.step != int(payload["completed_steps"]):
        raise RuntimeError("v2 checkpoint optimizer/update counters disagree")
    if int(payload["next_sample_index"]) != int(payload["completed_steps"]) * n_chains:
        raise RuntimeError("v2 checkpoint sample counter is inconsistent")
    measurements = metadata.get("fixed_parameter_measurements")
    if not isinstance(measurements, list) or not measurements or measurements[0].get(
        "update"
    ) != 0:
        raise RuntimeError("v2 checkpoint is missing the update-zero measurement")
    return payload


def _selection_diagnostics(
    history: list[dict[str, object]], spec: dict[str, object]
) -> dict[str, object]:
    diagnostics = dict(spec["diagnostics"])
    window = int(diagnostics["acceptance_window"])
    acceptance = np.asarray(
        [float(row["sampler"]["accepted_move_fraction"]) for row in history],
        dtype=np.float64,
    )
    relative = np.asarray(
        [float(row["adam"]["relative_update_norm"]) for row in history],
        dtype=np.float64,
    )
    if len(history):
        median_acceptance = float(np.median(acceptance[-window:]))
        relative_quantile = float(
            np.quantile(relative, float(diagnostics["relative_update_quantile"]))
        )
        relative_fraction = float(
            np.mean(relative <= float(diagnostics["relative_update_threshold"]))
        )
    else:
        median_acceptance = None
        relative_quantile = None
        relative_fraction = None
    acceptance_threshold = float(diagnostics["low_acceptance_median_threshold"])
    relative_threshold = float(diagnostics["relative_update_threshold"])
    return {
        "last_updates_window": min(window, len(history)),
        "last_updates_accepted_move_fraction_median": median_acceptance,
        "low_acceptance_median_threshold": acceptance_threshold,
        "low_acceptance_warning": (
            bool(float(median_acceptance) < acceptance_threshold)
            if len(history)
            else None
        ),
        "relative_update_quantile": float(diagnostics["relative_update_quantile"]),
        "relative_update_quantile_value": relative_quantile,
        "relative_update_threshold": relative_threshold,
        "relative_update_quantile_within_threshold": (
            bool(float(relative_quantile) <= relative_threshold)
            if len(history)
            else None
        ),
        "relative_updates_fraction_within_threshold": relative_fraction,
        "diagnostics_are_not_hard_stops_or_wall_time_filters": True,
    }


def _endpoint_policy_diagnostics(
    measurements: list[dict[str, object]], spec: dict[str, object]
) -> dict[str, object]:
    """Evaluate paired same-chain endpoints without selecting an LR."""

    policy = dict(spec["selection_policy"])
    start_update, final_update = (
        int(value) for value in policy["endpoint_updates"]
    )
    by_update = {int(item["update"]): item for item in measurements}
    if start_update not in by_update or final_update not in by_update:
        raise RuntimeError("registered selection-policy endpoints are missing")
    start = by_update[start_update]
    final = by_update[final_update]

    def checked_chain_vector(
        measurement: dict[str, object], value_name: str, hash_name: str
    ) -> np.ndarray:
        values = np.asarray(measurement.get(value_name), dtype=np.float64)
        expected_count = int(measurement["chain_count"])
        if values.shape != (expected_count,) or np.any(~np.isfinite(values)):
            raise RuntimeError(f"invalid endpoint chain vector: {value_name}")
        if measurement.get(hash_name) != array_sha256(values):
            raise RuntimeError(f"endpoint chain-vector seal failed: {value_name}")
        if measurement.get("chain_index_lineage") != (
            "persistent_training_chain_index_then_independent_measurement_copy"
        ):
            raise RuntimeError("endpoint chain-index lineage is not registered")
        return values

    energy_start_chains = checked_chain_vector(
        start,
        "energy_per_site_chain_means",
        "energy_per_site_chain_means_sha256",
    )
    energy_final_chains = checked_chain_vector(
        final,
        "energy_per_site_chain_means",
        "energy_per_site_chain_means_sha256",
    )
    variance_start_chains = checked_chain_vector(
        start,
        "variance_h_per_site_chain_contributions",
        "variance_h_per_site_chain_contributions_sha256",
    )
    variance_final_chains = checked_chain_vector(
        final,
        "variance_h_per_site_chain_contributions",
        "variance_h_per_site_chain_contributions_sha256",
    )
    chain_count = len(energy_start_chains)
    if not (
        len(energy_final_chains)
        == len(variance_start_chains)
        == len(variance_final_chains)
        == chain_count
        and chain_count >= 2
    ):
        raise RuntimeError("endpoint paired chain vectors have inconsistent lengths")
    energy_start = float(start["energy_per_site"])
    energy_final = float(final["energy_per_site"])
    energy_independent_mcse = math.hypot(
        float(start["energy_per_site_chain_mcse"]),
        float(final["energy_per_site_chain_mcse"]),
    )
    variance_start = float(start["variance_h_per_site"])
    variance_final = float(final["variance_h_per_site"])
    variance_independent_mcse = math.hypot(
        float(start["variance_h_per_site_chain_mcse"]),
        float(final["variance_h_per_site_chain_mcse"]),
    )
    energy_lowering_chains = energy_start_chains - energy_final_chains
    variance_worsening_chains = variance_final_chains - variance_start_chains
    energy_lowering = float(np.mean(energy_lowering_chains))
    variance_worsening = float(np.mean(variance_worsening_chains))
    if not np.isclose(
        energy_lowering, energy_start - energy_final, rtol=2.0e-12, atol=2.0e-14
    ):
        raise RuntimeError("paired energy lowering disagrees with endpoint means")
    if not np.isclose(
        variance_worsening,
        variance_final - variance_start,
        rtol=2.0e-12,
        atol=2.0e-14,
    ):
        raise RuntimeError("paired variance worsening disagrees with endpoint means")
    energy_paired_mcse = float(
        np.std(energy_lowering_chains, ddof=1) / np.sqrt(chain_count)
    )
    variance_paired_mcse = float(
        np.std(variance_worsening_chains, ddof=1) / np.sqrt(chain_count)
    )
    valid = (
        all(
            math.isfinite(value)
            for value in (
                energy_start,
                energy_final,
                energy_independent_mcse,
                energy_paired_mcse,
                variance_start,
                variance_final,
                variance_independent_mcse,
                variance_paired_mcse,
            )
        )
        and energy_paired_mcse > 0.0
        and variance_paired_mcse > 0.0
    )
    if valid:
        signed_energy_lowering_z: float | None = (
            energy_lowering / energy_paired_mcse
        )
        signed_variance_worsening_z: float | None = (
            variance_worsening / variance_paired_mcse
        )
        energy_gate = (
            energy_lowering
            >= float(policy["required_energy_lowering_per_site"])
            and signed_energy_lowering_z
            >= float(policy["minimum_energy_lowering_combined_mcse"])
        )
        variance_gate = signed_variance_worsening_z <= float(
            policy["maximum_variance_worsening_combined_mcse"]
        )
        invalid_reason = None
    else:
        signed_energy_lowering_z = None
        signed_variance_worsening_z = None
        energy_gate = False
        variance_gate = False
        invalid_reason = "nonfinite_or_nonpositive_paired_endpoint_mcse"
    independent_energy_z = (
        energy_lowering / energy_independent_mcse
        if energy_independent_mcse > 0.0
        else None
    )
    independent_variance_z = (
        variance_worsening / variance_independent_mcse
        if variance_independent_mcse > 0.0
        else None
    )
    return {
        "endpoint_updates": [start_update, final_update],
        "chain_count": chain_count,
        "primary_endpoint_uncertainty": "same_chain_lineage_paired_difference_se",
        "energy_per_site_start": energy_start,
        "energy_per_site_final": energy_final,
        "signed_energy_lowering_per_site": energy_lowering,
        "energy_combined_chain_mcse_per_site": energy_paired_mcse,
        "signed_energy_lowering_combined_mcse": signed_energy_lowering_z,
        "energy_paired_chain_mcse_per_site": energy_paired_mcse,
        "signed_energy_lowering_paired_mcse": signed_energy_lowering_z,
        "energy_independent_hypot_mcse_per_site": energy_independent_mcse,
        "signed_energy_lowering_independent_hypot_mcse": independent_energy_z,
        "required_energy_lowering_per_site": float(
            policy["required_energy_lowering_per_site"]
        ),
        "minimum_energy_lowering_combined_mcse": float(
            policy["minimum_energy_lowering_combined_mcse"]
        ),
        "energy_endpoint_gate_passed": bool(energy_gate),
        "variance_h_per_site_start": variance_start,
        "variance_h_per_site_final": variance_final,
        "signed_variance_worsening_per_site": variance_worsening,
        "variance_combined_chain_mcse_per_site": variance_paired_mcse,
        "signed_variance_worsening_combined_mcse": signed_variance_worsening_z,
        "variance_paired_chain_mcse_per_site": variance_paired_mcse,
        "signed_variance_worsening_paired_mcse": signed_variance_worsening_z,
        "variance_independent_hypot_mcse_per_site": variance_independent_mcse,
        "signed_variance_worsening_independent_hypot_mcse": (
            independent_variance_z
        ),
        "maximum_variance_worsening_combined_mcse": float(
            policy["maximum_variance_worsening_combined_mcse"]
        ),
        "variance_endpoint_gate_passed": bool(variance_gate),
        "policy_diagnostic_valid": bool(valid),
        "invalid_reason": invalid_reason,
        "eligible_for_cross_artifact_geometry_common_lr_analysis": bool(
            valid and energy_gate and variance_gate
        ),
        "geometry_common_lr_selection_deferred_to_cross_artifact_analysis": True,
        "candidate_gap_tie_combined_mcse": float(
            policy["candidate_gap_tie_combined_mcse"]
        ),
        "tie_choose_smaller_lr": True,
        "arm_specific_optimum_diagnostic_only": True,
        "diagnostic_does_not_stop_training_early": True,
    }


def _run_scan_arm(
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
    common_profile_hash: str,
    overwrite: bool,
    maximum_updates_this_invocation: int | None = None,
) -> dict[str, object]:
    """Run or resume one LR/geometry/backbone candidate exactly."""

    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    sampler = dict(spec["sampler"])
    optimizer = dict(spec["optimizer"])
    measurement = dict(spec["measurement"])
    chains = int(sampler["chains"])
    proposals = int(sampler["proposals_per_sweep"])
    probability = float(sampler["bond_proposal_probability"])
    microbatch = int(optimizer["microbatch_size"])
    total_updates = int(optimizer["updates"])
    padded_width = len(bonds) + 1
    advance_uncached, advance_cached, hpsi_fn, center_fn = _compiled_kernels(
        runtime, bonds
    )

    seed_key = jax.random.PRNGKey(int(sampler["seed"]))
    chain_key, sampling_key_device = jax.random.split(seed_key)
    initial_training_key = np.asarray(sampling_key_device, dtype=np.uint32)
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
        common_profile_hash=common_profile_hash,
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
    checkpoint = None if overwrite else _load_checkpoint(
        checkpoint_directory,
        identity,
        expected_initial_chain_hash=initial_chain_hash,
        expected_initial_params_hash=initial_params_hash,
        n_sites=n_sites,
        n_chains=chains,
    )
    latest_generation: Path | None = None
    invocation_started = time.perf_counter()

    if checkpoint is None:
        params = runtime.params
        optimizer_state = initialize_sampled_adam(params)
        burn_attempts = int(sampler["initial_burn_sweeps"]) * proposals
        states_device, amplitudes_device, sampling_key_device, burn_device = (
            advance_uncached(
                params,
                sampling_key_device,
                initial_states_device,
                burn_attempts,
                probability,
            )
        )
        jax.block_until_ready(
            (states_device, amplitudes_device, sampling_key_device, burn_device)
        )
        states = np.asarray(states_device, dtype=np.int32)
        sampling_key = np.asarray(sampling_key_device, dtype=np.uint32)
        del amplitudes_device
        initial_post_burn_hash = array_sha256(states)
        initial_post_burn_key_hash = array_sha256(sampling_key)
        initial_burn_diagnostics = v1_helpers._float_diagnostics(burn_device)
        burn_projected = chains * (1 + burn_attempts)
        initial_burn_work = {
            "sampler_proposal_attempts": chains * burn_attempts,
            "sampler_projected_amplitude_evaluations": burn_projected,
            "base_tree_orbit_image_work_proxy": burn_projected
            * runtime.projector.group_order,
        }
        try:
            update_zero, measurement_work = _fixed_energy_measurement(
                runtime=runtime,
                params=params,
                training_states=states,
                case=case,
                bonds=bonds,
                spec=spec,
                update=0,
                advance_uncached=advance_uncached,
                advance_cached=advance_cached,
                hpsi_fn=hpsi_fn,
            )
        except FloatingPointError as error:
            return {
                "status": "hard_failed",
                "candidate_id": spec["candidate_id"],
                "learning_rate": float(optimizer["learning_rate"]),
                "optimizer_backend": OPTIMIZER_BACKEND,
                "common_scan_profile_sha256": common_profile_hash,
                "arm_id": runtime.arm_id,
                "backbone": runtime.backbone,
                "seed": int(arm["seed"]),
                "completed_updates": 0,
                "requested_updates": total_updates,
                "initial_params_sha256": initial_params_hash,
                "initial_chain_states_sha256": initial_chain_hash,
                "initial_post_burn_chain_states_sha256": initial_post_burn_hash,
                "initial_training_key_sha256": array_sha256(initial_training_key),
                "initial_post_burn_training_key_sha256": (
                    initial_post_burn_key_hash
                ),
                "hard_failure": {
                    "category": "nonfinite_measurement",
                    "attempted_update": 0,
                    "message": str(error),
                },
                "history": [],
                "fixed_parameter_measurements": [],
                "selection_diagnostics": _selection_diagnostics([], spec),
                "latest_valid_checkpoint": None,
                "wall_time_used_for_candidate_selection": False,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        update_zero["work_ledger"] = measurement_work
        measurements = [update_zero]
        history: list[dict[str, object]] = []
        completed_updates = 0
        resume_count = 0
        metadata = _checkpoint_metadata(
            params=params,
            optimizer_state=optimizer_state,
            states=states,
            sampling_key=sampling_key,
            initial_chain_states_sha256=initial_chain_hash,
            initial_post_burn_chain_states_sha256=initial_post_burn_hash,
            initial_params_sha256=initial_params_hash,
            initial_training_key_sha256=array_sha256(initial_training_key),
            initial_post_burn_training_key_sha256=initial_post_burn_key_hash,
            initial_burn_diagnostics=initial_burn_diagnostics,
            initial_burn_work_ledger=initial_burn_work,
            measurements=measurements,
            resume_count=resume_count,
        )
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
    else:
        params = checkpoint["params"]
        optimizer_state = checkpoint["optimizer_state"]
        completed_updates = int(checkpoint["completed_steps"])
        history = list(checkpoint["history"])
        metadata = dict(checkpoint["metadata"])
        states = np.asarray(metadata["chain_states"], dtype=np.int32)
        sampling_key = np.asarray(metadata["sampling_key"], dtype=np.uint32)
        initial_post_burn_hash = str(
            metadata["initial_post_burn_chain_states_sha256"]
        )
        initial_post_burn_key_hash = str(
            metadata["initial_post_burn_training_key_sha256"]
        )
        initial_burn_diagnostics = dict(metadata["initial_burn_diagnostics"])
        initial_burn_work = dict(metadata["initial_burn_work_ledger"])
        measurements = list(metadata["fixed_parameter_measurements"])
        resume_count = int(metadata["resume_count"]) + 1
        latest_generation = Path(str(checkpoint["checkpoint_generation_path"]))

    start_update = completed_updates
    invocation_limit = total_updates
    if maximum_updates_this_invocation is not None:
        if maximum_updates_this_invocation < 0:
            raise ValueError("maximum_updates_this_invocation cannot be negative")
        invocation_limit = min(
            total_updates, completed_updates + maximum_updates_this_invocation
        )

    hard_failure: dict[str, object] | None = None
    for update_index in range(completed_updates + 1, invocation_limit + 1):
        params_before = params
        optimizer_state_before = optimizer_state
        states_before = np.array(states, copy=True)
        sampling_key_before = np.array(sampling_key, copy=True)
        params_before_hash = pytree_sha256(params)
        sampling_attempts = int(sampler["decorrelation_sweeps_per_update"]) * proposals
        states_device, amplitudes_device, key_device, sampler_device = advance_uncached(
            params,
            jnp.asarray(sampling_key),
            jnp.asarray(states),
            sampling_attempts,
            probability,
        )
        jax.block_until_ready(
            (states_device, amplitudes_device, key_device, sampler_device)
        )
        states = np.asarray(states_device, dtype=np.int32)
        sampling_key = np.asarray(key_device, dtype=np.uint32)
        sampler_diagnostics = v1_helpers._float_diagnostics(sampler_device)
        chain_move_fraction = float(
            np.mean(np.any(states != states_before, axis=1))
        )
        del amplitudes_device
        if chain_move_fraction <= 0.0:
            hard_failure = {
                "category": "frozen_chain",
                "attempted_update": update_index,
                "message": "no chain changed during the registered update",
            }
            states = states_before
            sampling_key = sampling_key_before
            break

        batches, widths, _connectivity_seconds = v1_helpers._hpsi_batches(
            states,
            bonds,
            padded_width=padded_width,
            microbatch_size=microbatch,
        )
        try:
            gradient_result = sampled_vmc_value_and_grad(
                params,
                batches,
                hpsi_fn=hpsi_fn,
                center_fn=center_fn,
            )
        except FloatingPointError as error:
            hard_failure = {
                "category": "nonfinite_or_zero_center",
                "attempted_update": update_index,
                "message": str(error),
            }
            states = states_before
            sampling_key = sampling_key_before
            break

        try:
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
        except FloatingPointError as error:
            category = (
                "relative_update_exceeded"
                if "relative update exceeds" in str(error)
                else "nonfinite_optimizer_update"
            )
            hard_failure = {
                "category": category,
                "attempted_update": update_index,
                "message": str(error),
            }
            params = params_before
            optimizer_state = optimizer_state_before
            states = states_before
            sampling_key = sampling_key_before
            break

        params_after_hash = pytree_sha256(params)
        if params_after_hash == params_before_hash:
            raise RuntimeError("Adam accepted an update without changing parameters")

        fixed_measurement = None
        measurement_work: dict[str, int] = {}
        if update_index in set(int(value) for value in measurement["updates"]):
            try:
                fixed_measurement, measurement_work = _fixed_energy_measurement(
                    runtime=runtime,
                    params=params,
                    training_states=states,
                    case=case,
                    bonds=bonds,
                    spec=spec,
                    update=update_index,
                    advance_uncached=advance_uncached,
                    advance_cached=advance_cached,
                    hpsi_fn=hpsi_fn,
                )
            except FloatingPointError as error:
                hard_failure = {
                    "category": "nonfinite_measurement",
                    "attempted_update": update_index,
                    "message": str(error),
                }
                params = params_before
                optimizer_state = optimizer_state_before
                states = states_before
                sampling_key = sampling_key_before
                break
            fixed_measurement["work_ledger"] = measurement_work
            measurements.append(fixed_measurement)

        sampler_projected = chains * (1 + sampling_attempts)
        hpsi_forward_projected = chains * (1 + padded_width)
        center_primal_projected = chains
        center_adjoint_proxy = chains
        legacy_two_hpsi = 2 * chains * (1 + padded_width)
        actual_forward = hpsi_forward_projected + center_primal_projected
        avoided_connected = chains * padded_width
        step_work: dict[str, int] = {
            "training_samples": chains,
            "sampler_proposal_attempts": chains * sampling_attempts,
            "sampler_projected_amplitude_evaluations": sampler_projected,
            "hpsi_forward_projected_amplitude_evaluations": hpsi_forward_projected,
            "center_only_primal_projected_amplitude_evaluations": center_primal_projected,
            "center_only_adjoint_center_equivalent_work_proxy": center_adjoint_proxy,
            "center_only_forward_projected_amplitude_evaluations": actual_forward,
            "legacy_two_hpsi_projected_amplitude_evaluations_proxy": legacy_two_hpsi,
            "avoided_second_hpsi_connected_projected_amplitude_evaluations": (
                avoided_connected
            ),
            "gradient_backward_microbatch_invocations": len(batches),
            "hpsi_actual_connected_configurations": int(np.sum(widths)),
            "hpsi_padded_connected_configurations": chains * padded_width,
            "base_tree_orbit_image_forward_work_proxy": (
                (sampler_projected + actual_forward) * runtime.projector.group_order
            ),
            "base_tree_orbit_image_forward_plus_adjoint_work_proxy": (
                (sampler_projected + actual_forward + center_adjoint_proxy)
                * runtime.projector.group_order
            ),
            **measurement_work,
        }
        row: dict[str, object] = {
            "update": update_index,
            "params_before_sha256": params_before_hash,
            "params_after_sha256": params_after_hash,
            "chain_states_sha256": array_sha256(states),
            "sampling_key_sha256": array_sha256(sampling_key),
            "sampler": {
                **sampler_diagnostics,
                "chain_snapshot_move_fraction": chain_move_fraction,
            },
            "pre_update_sampled_energy_per_site": gradient_result.energy / n_sites,
            "pre_update_sampled_variance_h_per_site": gradient_result.variance
            / n_sites,
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
                "connected_hpsi_recomputed_in_backward": False,
            },
            "adam": asdict(adam_diagnostics),
            "post_update_amplitude_cache": "discarded_and_not_checkpointed",
            "fixed_parameter_measurement": fixed_measurement,
            "work_ledger": step_work,
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
                initial_post_burn_chain_states_sha256=initial_post_burn_hash,
                initial_params_sha256=initial_params_hash,
                initial_training_key_sha256=array_sha256(initial_training_key),
                initial_post_burn_training_key_sha256=(
                    initial_post_burn_key_hash
                ),
                initial_burn_diagnostics=initial_burn_diagnostics,
                initial_burn_work_ledger=initial_burn_work,
                measurements=measurements,
                resume_count=resume_count,
            )
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

    complete = completed_updates == total_updates and hard_failure is None
    latest_valid_checkpoint: dict[str, object] | None = None
    if latest_generation is not None:
        latest_payload = load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=identity
        )
        latest_path = Path(str(latest_payload["checkpoint_generation_path"]))
        latest_valid_checkpoint = {
            "series_directory": checkpoint_directory.name,
            "generation": latest_path.name,
            "generation_sha256": file_sha256(latest_path),
            "run_identity_sha256": canonical_json_sha256(identity),
            "completed_updates": int(latest_payload["completed_steps"]),
        }
    state_result: dict[str, object] = {
        "status": (
            "hard_failed"
            if hard_failure is not None
            else "complete" if complete else "running"
        ),
        "completed_updates": completed_updates,
        "requested_updates": total_updates,
        "updates_completed_this_invocation": completed_updates - start_update,
        "resume_count": resume_count,
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(optimizer_state),
        "chain_states_sha256": array_sha256(states),
        "sampling_key_sha256": array_sha256(sampling_key),
        "history": history,
        "fixed_parameter_measurements": measurements,
        "selection_diagnostics": _selection_diagnostics(history, spec),
        "latest_valid_checkpoint": latest_valid_checkpoint,
        "hard_failure": hard_failure,
        "candidate_id": spec["candidate_id"],
        "learning_rate": float(optimizer["learning_rate"]),
        "optimizer_backend": OPTIMIZER_BACKEND,
        "common_scan_profile_sha256": common_profile_hash,
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "seed": int(arm["seed"]),
        "initial_params_sha256": initial_params_hash,
        "initial_chain_states_sha256": initial_chain_hash,
        "initial_post_burn_chain_states_sha256": initial_post_burn_hash,
        "initial_training_key_sha256": array_sha256(initial_training_key),
        "initial_post_burn_training_key_sha256": initial_post_burn_key_hash,
        "wall_seconds_this_invocation": time.perf_counter() - invocation_started,
        "wall_time_used_for_candidate_selection": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if not complete:
        return state_result

    final_checkpoint = load_latest_resumable_checkpoint(
        checkpoint_directory, expected_run_identity=identity
    )
    final_metadata = dict(final_checkpoint["metadata"])
    if int(final_checkpoint["completed_steps"]) != total_updates:
        raise RuntimeError("final v2 checkpoint is incomplete")
    if final_metadata.get("amplitude_cache_present") is not False:
        raise RuntimeError("final v2 checkpoint retained an amplitude cache")
    generation_path = Path(str(final_checkpoint["checkpoint_generation_path"]))
    if latest_generation is None:
        latest_generation = generation_path
    work = v1_helpers._sum_ledgers(
        [initial_burn_work, dict(measurements[0]["work_ledger"])]
        + [dict(row["work_ledger"]) for row in history]
    )
    measurement_updates = [int(item["update"]) for item in measurements]
    registered_measurements = [int(value) for value in measurement["updates"]]
    optimizer_diagnostics = _selection_diagnostics(history, spec)
    endpoint_policy = _endpoint_policy_diagnostics(measurements, spec)
    stability_gate_passed = (
        optimizer_diagnostics["low_acceptance_warning"] is False
        and optimizer_diagnostics["relative_update_quantile_within_threshold"]
        is True
    )
    endpoint_policy["stability_gate_passed"] = stability_gate_passed
    endpoint_policy[
        "eligible_for_cross_artifact_geometry_common_lr_analysis"
    ] = bool(
        endpoint_policy["eligible_for_cross_artifact_geometry_common_lr_analysis"]
        and stability_gate_passed
    )
    gates = {
        "all_requested_updates_completed": len(history) == total_updates,
        "all_local_energies_and_gradients_finite": all(
            int(row["nonfinite_local_count"]) == 0
            and all(
                math.isfinite(float(value))
                for value in (
                    row["pre_update_sampled_energy_per_site"],
                    row["pre_update_sampled_variance_h_per_site"],
                    row["gradient"]["norm"],
                    row["minimum_center_magnitude"],
                )
            )
            for row in history
        ),
        "all_relative_updates_within_hard_limit": all(
            float(row["adam"]["relative_update_norm"])
            <= float(optimizer["maximum_relative_update_norm"])
            for row in history
        ),
        "all_updates_move_at_least_one_chain": all(
            float(row["sampler"]["chain_snapshot_move_fraction"]) > 0.0
            for row in history
        ),
        "center_only_backward_contract": all(
            row["gradient"]["connected_hpsi_recomputed_in_backward"] is False
            for row in history
        ),
        "fixed_measurements_at_registered_updates": (
            measurement_updates == registered_measurements
        ),
        "fixed_measurements_finite": all(
            all(
                math.isfinite(float(item[name]))
                for name in (
                    "energy_per_site",
                    "energy_per_site_chain_mcse",
                    "variance_h_per_site",
                    "variance_h_per_site_chain_mcse",
                )
            )
            for item in measurements
        ),
        "endpoint_policy_combined_mcse_is_finite_and_nonzero": bool(
            endpoint_policy["policy_diagnostic_valid"]
        ),
        "amplitude_cache_absent_from_checkpoint": (
            final_metadata.get("amplitude_cache_present") is False
            and "amplitudes" not in final_metadata
        ),
        "optimizer_step_matches_update_count": optimizer_state.step == total_updates,
    }
    trial: dict[str, object] = {
        "status": "complete",
        "hard_failure": None,
        "candidate_id": spec["candidate_id"],
        "learning_rate": float(optimizer["learning_rate"]),
        "optimizer_backend": OPTIMIZER_BACKEND,
        "common_scan_profile_sha256": common_profile_hash,
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
        "initial_post_burn_chain_states_sha256": initial_post_burn_hash,
        "initial_training_key_sha256": array_sha256(initial_training_key),
        "initial_post_burn_training_key_sha256": initial_post_burn_key_hash,
        "final_chain_states_sha256": array_sha256(states),
        "translation_group_order": runtime.projector.group_order,
        "padded_connectivity_width": padded_width,
        "optimizer": {**optimizer, "completed_updates": completed_updates},
        "sampler": {
            **sampler,
            "initial_burn_diagnostics": initial_burn_diagnostics,
            "resume_count": resume_count,
        },
        "history": history,
        "fixed_parameter_measurements": measurements,
        "selection_diagnostics": optimizer_diagnostics,
        "endpoint_policy_diagnostics": endpoint_policy,
        "work_ledger": {
            **work,
            "host_peak_rss_bytes_process_high_watermark_proxy": (
                v1_helpers._peak_rss_bytes()
            ),
            "parameter_count_is_not_used_as_a_work_substitute": True,
            "wall_time_used_for_candidate_selection": False,
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
            "sampling_key_sha256": final_metadata["sampling_key_sha256"],
            "amplitude_cache_present": final_metadata["amplitude_cache_present"],
        },
        "readiness_gates": gates,
        "all_readiness_gates_passed": all(gates.values()),
        "wall_time": {
            "seconds_this_invocation": state_result["wall_seconds_this_invocation"],
            "candidate_selection_metric": False,
            "resume_count": resume_count,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return trial


def run_spin_sampled_training_v2(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume one candidate in the registered optimizer-scan family."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    common_profile_hash = common_scan_profile_sha256(spec)
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
                directory = output / f"checkpoint_{case['case_id']}_{arm['arm_id']}"
                if directory.is_dir():
                    shutil.rmtree(directory)

    header = (config_hash, source_hash, runtime_hash, common_profile_hash)
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        observed = (
            payload.get("config_sha256"),
            payload.get("source_manifest_sha256"),
            payload.get("runtime_identity_sha256"),
            payload.get("common_scan_profile_sha256"),
        )
        if payload.get("schema_version") != 2 or observed != header:
            raise RuntimeError(f"refusing stale optimizer-scan v2 result: {result_path}")
        if payload.get("status") == "complete":
            manifest = embedded_trial_manifest(payload)
            if payload.get("trial_manifest_sha256") != canonical_sha256(manifest):
                raise RuntimeError(f"optimizer-scan v2 trial seal failed: {result_path}")
            return payload
        if payload.get("status") not in {"running", "failed"}:
            raise RuntimeError(f"invalid optimizer-scan v2 status: {result_path}")
        payload["status"] = "running"
        payload.pop("failure", None)
        atomic_json(result_path, payload)
    else:
        payload = {
            "schema_version": TRAINING_V2_RESULT_SCHEMA_VERSION,
            "domain": "spin",
            "task": TASK,
            "optimizer_backend": OPTIMIZER_BACKEND,
            "campaign_stage": "local_optimizer_scan",
            "scan_family_id": spec["scan_family_id"],
            "candidate_id": spec["candidate_id"],
            "learning_rate": float(spec["optimizer"]["learning_rate"]),
            "status": "running",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": _portable_config_path(config_path),
            "config_sha256": config_hash,
            "config": spec,
            "common_scan_profile": _common_scan_profile(spec),
            "common_scan_profile_sha256": common_profile_hash,
            "source_sha256": source_manifest,
            "source_manifest_sha256": source_hash,
            "runtime_identity": runtime_identity,
            "runtime_identity_sha256": runtime_hash,
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "route_boundary": {
                "direct_fusion_path_used": False,
                "hilbert_space_enumerated": False,
                "hamiltonian_diagonalization_performed": False,
                "physical_translation_projector_used": True,
                "persistent_exchange_mcmc_used": True,
                "center_only_gradient_backend_used": True,
                "full_spin_structure_factor_computed": False,
                "large_ed_reference_required": False,
                "wall_time_used_for_selection": False,
                "energy_or_physics_win_claimed": False,
            },
            "matched_scan_contract": {
                "learning_rate_is_only_trajectory_choice_across_family": True,
                "same_initial_params_per_geometry_backbone_across_lr": True,
                "same_initial_and_post_burn_chains_across_lr": True,
                "same_training_random_key_and_schedule_across_lr": True,
                "same_independent_measurement_random_streams_across_lr": True,
                "all_non_hard_failed_candidates_run_all_updates": True,
                "wall_time_is_not_a_selection_metric": True,
                "endpoint_policy_is_diagnostic_and_never_an_early_stop": True,
                "geometry_common_lr_selection_is_cross_artifact_only": True,
            },
            "selection_policy": dict(spec["selection_policy"]),
            "cases": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
        atomic_json(result_path, payload)

    try:
        projection = dict(spec["projection"])
        cases_by_id = {
            str(item["case_id"]): item for item in payload["cases"]
        }
        for case in spec["cases"]:
            bonds, group, projector = v1_helpers._geometry(case, projection)
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
                    "trials": [],
                }
                payload["cases"].append(case_payload)
                cases_by_id[case_id] = case_payload
                atomic_json(result_path, payload)
            completed_arms = {str(item["arm_id"]) for item in case_payload["trials"]}
            for arm in spec["arms"]:
                if str(arm["arm_id"]) in completed_arms:
                    continue
                runtime = runtime_workflow._initialize_arm(
                    arm,
                    rows=int(case["rows"]),
                    columns=int(case["columns"]),
                    projector=projector,
                )
                directory = output / f"checkpoint_{case_id}_{runtime.arm_id}"
                trial = _run_scan_arm(
                    runtime=runtime,
                    arm=arm,
                    case=case,
                    bonds=bonds,
                    group=group,
                    spec=spec,
                    checkpoint_directory=directory,
                    config_hash=config_hash,
                    source_hash=source_hash,
                    runtime_hash=runtime_hash,
                    common_profile_hash=common_profile_hash,
                    overwrite=overwrite,
                )
                if trial["status"] == "running":  # pragma: no cover
                    raise RuntimeError("production optimizer-scan arm stopped early")
                trial["trial_payload_sha256"] = trial_payload_sha256(trial)
                case_payload["trials"].append(trial)
                atomic_json(result_path, payload)
        manifest = embedded_trial_manifest(payload)
        payload["trial_manifest"] = manifest
        payload["trial_manifest_sha256"] = canonical_sha256(manifest)
        trials = [trial for case in payload["cases"] for trial in case["trials"]]
        payload["terminal_trial_count"] = len(trials)
        payload["hard_failed_trial_count"] = sum(
            trial["status"] == "hard_failed" for trial in trials
        )
        payload["all_non_hard_failed_candidates_completed"] = all(
            trial["status"] in {"complete", "hard_failed"} for trial in trials
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


__all__ = [
    "CLAIM_BOUNDARY",
    "OPTIMIZER_BACKEND",
    "TASK",
    "common_scan_profile_sha256",
    "run_spin_sampled_training_v2",
]
