"""Matched-work N=36 preflight for the projected multiplicity residual.

The exact three-determinant pi/0-flux carrier is retained analytically.  A
physical-input ClebschTree residual is projected into full ``Gamma-A1`` on
every amplitude call and added coherently to that carrier.  The comparison
uses the same sealed physical configurations as the projected two-Lanczos and
fixed-``Sz`` ViT controls, and equalizes literal full-H configuration calls.

The local tier is an execution/conditioning gate, not an untouched physics
claim.  A remote tier may use all sealed chains only after this code path has
passed locally.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (
    _paired_chain_difference,
)
from projects.clebsch_tree_nqs.workflows.spin_g1_regularized_operator import (
    _matrices,
    _ratio_summary,
)
from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
    _batch_hpsi,
    _normalized_gradient_step,
)
from projects.clebsch_tree_nqs.workflows.spin_n36_su2_scalar_dressing import (
    _baseline_from_cache,
    _sealed_common_data,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.benchmarks.crossfit_regularized_operator import krylov_control_fit
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.spin_dimer_product_init import (
    init_spin_clebsch_tree_dimer_product,
)
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
)
from su2_nqs.lattices.pair_balanced_tree import pair_balanced_fusion_tree
from su2_nqs.lattices.physical_route_banks import (
    triangular_dimer_complete_route_orders,
)
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.projected_carrier_multiplicity_dressing import (
    ProjectedCarrierMultiplicityDressingConfig,
    certified_carrier_projected_residual_amplitude,
    init_projected_carrier_multiplicity_dressing,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
)
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
)
from su2_nqs.models.su2_route_vit import SU2RouteViTConfig, init_su2_route_vit
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig
from su2_nqs.vmc.common_sample_vmc import common_sample_vmc_value_and_grad
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_projected_multiplicity_preflight"
ARM_BASELINE = "projected_vmc_two_lanczos"
ARM_FIXED_SZ = "pi_flux_fixed_sz_spatial_vit"
ARM_SU2 = "pi_flux_exact_su2_projected_multiplicity"
N_SITES = 36
J2 = 0.125
ESS_ROUNDOFF_TOLERANCE = 1.0e-12


def _has_committed_checkpoint(directory: Path) -> bool:
    """Return whether a checkpoint series contains a committed artifact.

    An empty directory, or one containing only atomic-write temporary files,
    means that preemption happened before the first commit and is safe to
    initialize again.  A manifest or immutable generation is formal state and
    must be loaded (and therefore validated) rather than silently ignored.
    """

    return directory.is_dir() and (
        (directory / "LATEST.json").is_file()
        or any(directory.glob("step_*.pkl"))
    )


def _su2_work_cursor(completed_updates: int, batch_size: int) -> int:
    """Count registered SU(2) full-H rows, including the calibration pass."""

    if completed_updates < 0 or batch_size < 1:
        raise ValueError("invalid SU(2) work cursor")
    return (1 + completed_updates) * batch_size


def _all_numeric_values_finite(value: object) -> bool:
    """Recursively reject NaN/Inf while ignoring descriptive text fields."""

    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (int, float, complex, np.number)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return bool(np.isfinite(value))
    return True


def _valid_ess_fraction(value: object) -> bool:
    """Accept a probability ESS fraction up to roundoff above one."""

    candidate = float(value)
    return bool(
        np.isfinite(candidate)
        and 0.0 < candidate <= 1.0 + ESS_ROUNDOFF_TOLERANCE
    )


def _training_history_is_complete_and_finite(
    history: list[dict[str, object]], expected_steps: int
) -> bool:
    """Validate a one-indexed, gap-free common-sample training history."""

    required = (
        "training_energy_per_site",
        "training_variance_per_site",
        "training_ess_fraction",
        "gradient_norm",
        "applied_gradient_scale",
    )
    if len(history) != expected_steps:
        return False
    for expected_step, row in enumerate(history, start=1):
        if not isinstance(row, dict) or row.get("step") != expected_step:
            return False
        if any(name not in row for name in required):
            return False
        if not _all_numeric_values_finite(row):
            return False
        if float(row["training_variance_per_site"]) < 0.0:
            return False
        if not _valid_ess_fraction(row["training_ess_fraction"]):
            return False
        if float(row["gradient_norm"]) < 0.0:
            return False
        if float(row["applied_gradient_scale"]) < 0.0:
            return False
    return True


def _preflight_decision(
    *,
    execution_tier: str,
    baseline_summary: dict[str, object],
    fixed_summary: dict[str, object],
    su2_summary: dict[str, object],
    su2_minus_baseline: dict[str, object],
    su2_minus_fixed: dict[str, object],
    su2_history: list[dict[str, object]],
    fixed_history: list[dict[str, object]],
    training_steps: int,
    carrier_relative_error: float,
    maximum_carrier_relative_error: float,
    minimum_ess: float,
    energy_tolerance: float,
) -> dict[str, object]:
    """Evaluate preregistered pilot/repeat gates with auditable subchecks."""

    thresholds_valid = bool(
        np.isfinite(maximum_carrier_relative_error)
        and maximum_carrier_relative_error >= 0.0
        and np.isfinite(minimum_ess)
        and 0.0 < minimum_ess <= 1.0
        and np.isfinite(energy_tolerance)
        and energy_tolerance >= 0.0
    )
    summaries_finite = all(
        _all_numeric_values_finite(summary)
        for summary in (baseline_summary, fixed_summary, su2_summary)
    )
    summary_domains_valid = bool(
        summaries_finite
        and all(
            float(summary["variance_total"]) >= 0.0
            and _valid_ess_fraction(summary["importance_ess_fraction"])
            for summary in (baseline_summary, fixed_summary, su2_summary)
        )
    )
    paired_comparisons_finite = all(
        _all_numeric_values_finite(comparison)
        for comparison in (su2_minus_baseline, su2_minus_fixed)
    )
    paired_standard_errors_nonnegative = bool(
        paired_comparisons_finite
        and float(su2_minus_baseline["chain_standard_error_per_site"]) >= 0.0
        and float(su2_minus_fixed["chain_standard_error_per_site"]) >= 0.0
    )
    paired_comparisons_well_formed = bool(
        paired_comparisons_finite and paired_standard_errors_nonnegative
    )
    su2_history_ok = _training_history_is_complete_and_finite(
        su2_history, training_steps - 1
    )
    fixed_history_ok = _training_history_is_complete_and_finite(
        fixed_history, training_steps
    )
    carrier_certified = bool(
        thresholds_valid
        and np.isfinite(carrier_relative_error)
        and carrier_relative_error >= 0.0
        and carrier_relative_error <= maximum_carrier_relative_error
    )
    all_arms_ess = bool(
        thresholds_valid
        and summary_domains_valid
        and min(
            float(baseline_summary["importance_ess_fraction"]),
            float(fixed_summary["importance_ess_fraction"]),
            float(su2_summary["importance_ess_fraction"]),
        )
        >= minimum_ess
    )
    two_lanczos_upper_2se = float(
        su2_minus_baseline["difference_per_site"]
    ) + 2.0 * float(su2_minus_baseline["chain_standard_error_per_site"])
    fixed_sz_upper_2se = float(
        su2_minus_fixed["difference_per_site"]
    ) + 2.0 * float(su2_minus_fixed["chain_standard_error_per_site"])
    two_lanczos_noninferior = bool(
        thresholds_valid
        and paired_comparisons_well_formed
        and two_lanczos_upper_2se <= energy_tolerance
    )
    fixed_sz_noninferior = bool(
        thresholds_valid
        and paired_comparisons_well_formed
        and fixed_sz_upper_2se <= energy_tolerance
    )
    variance_below_two_lanczos = bool(
        summary_domains_valid
        and float(su2_summary["variance_total"])
        < float(baseline_summary["variance_total"])
    )
    variance_below_fixed_sz = bool(
        summary_domains_valid
        and float(su2_summary["variance_total"])
        < float(fixed_summary["variance_total"])
    )
    execution_integrity = {
        "registered_thresholds_valid": thresholds_valid,
        "summaries_finite": summaries_finite,
        "summary_domains_valid": summary_domains_valid,
        "paired_comparisons_finite": paired_comparisons_finite,
        "paired_standard_errors_nonnegative": paired_standard_errors_nonnegative,
        "su2_history_complete_and_finite": su2_history_ok,
        "fixed_sz_history_complete_and_finite": fixed_history_ok,
        "carrier_certified": carrier_certified,
        "all_arms_ess_passed": all_arms_ess,
    }
    fresh_repeat_checks = {
        **execution_integrity,
        "remote_gpu_tier": execution_tier == "remote_gpu_pilot",
        "noninferior_to_two_lanczos_at_registered_margin": two_lanczos_noninferior,
        "noninferior_to_fixed_sz_at_registered_margin": fixed_sz_noninferior,
        "variance_below_two_lanczos_point_estimate": variance_below_two_lanczos,
        "variance_below_fixed_sz_point_estimate": variance_below_fixed_sz,
    }
    baseline_difference = float(su2_minus_baseline["difference_per_site"])
    baseline_se = float(
        su2_minus_baseline["chain_standard_error_per_site"]
    )
    fixed_difference = float(su2_minus_fixed["difference_per_site"])
    fixed_se = float(su2_minus_fixed["chain_standard_error_per_site"])
    return {
        "su2_energy_below_two_lanczos": baseline_difference < 0.0,
        "su2_energy_below_fixed_sz": fixed_difference < 0.0,
        "su2_variance_below_two_lanczos": variance_below_two_lanczos,
        "su2_variance_below_fixed_sz": variance_below_fixed_sz,
        "su2_ess_gate_passed": bool(
            thresholds_valid
            and summary_domains_valid
            and float(su2_summary["importance_ess_fraction"]) >= minimum_ess
        ),
        "all_arms_ess_gate_passed": all_arms_ess,
        "su2_resolved_energy_win_vs_two_lanczos": bool(
            paired_comparisons_well_formed
            and baseline_difference + 2.0 * baseline_se < 0.0
        ),
        "su2_resolved_energy_loss_vs_two_lanczos": bool(
            paired_comparisons_well_formed
            and baseline_difference - 2.0 * baseline_se > 0.0
        ),
        "su2_resolved_energy_win_vs_fixed_sz": bool(
            paired_comparisons_well_formed
            and fixed_difference + 2.0 * fixed_se < 0.0
        ),
        "su2_resolved_energy_loss_vs_fixed_sz": bool(
            paired_comparisons_well_formed
            and fixed_difference - 2.0 * fixed_se > 0.0
        ),
        "su2_noninferior_to_two_lanczos_at_registered_margin": (
            two_lanczos_noninferior
        ),
        "su2_noninferior_to_fixed_sz_at_registered_margin": fixed_sz_noninferior,
        "variance_gate_is_point_estimate_only": True,
        "variance_claim_licensed": False,
        "remote_gpu_pilot_gate_checks": execution_integrity,
        "remote_gpu_pilot_licensed": bool(
            execution_tier == "local_sampled_preflight"
            and all(execution_integrity.values())
        ),
        "fresh_independent_repeat_gate_checks": fresh_repeat_checks,
        "fresh_independent_repeat_licensed": bool(
            all(fresh_repeat_checks.values())
        ),
    }


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 projected-multiplicity identity")
    if spec.get("execution_tier") not in {
        "local_sampled_preflight",
        "remote_gpu_pilot",
    }:
        raise ValueError("invalid execution_tier")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the matched-work run must be preregistered")
    for name in ("sealed_development_result", "sealed_audit_result"):
        if not isinstance(spec.get(name), str) or not str(spec[name]).endswith(".json"):
            raise ValueError(f"{name} must be a JSON path")
    for section in (
        "samples",
        "su2_model",
        "fixed_sz_model",
        "training",
        "gates",
    ):
        if not isinstance(spec.get(section), dict):
            raise TypeError(f"{section} must be an object")
    train_count = int(spec["samples"].get("training_count", 0))
    audit_count = int(spec["samples"].get("audit_count", 0))
    if min(train_count, audit_count) < 1 or train_count != audit_count:
        raise ValueError("equal positive training and audit counts are required")
    for name in ("training_start", "audit_start"):
        if int(spec["samples"].get(name, 0)) < 0:
            raise ValueError(f"samples.{name} must be nonnegative")
    steps = int(spec["training"].get("steps", 0))
    if steps != 5:
        raise ValueError("five neural steps exactly match the three-column baseline work")
    for name in ("su2_learning_rate", "fixed_sz_learning_rate"):
        value = float(spec["training"].get(name, 0.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"training.{name} must be positive")
    minimum_ess = float(spec["gates"].get("minimum_ess_fraction", np.nan))
    maximum_carrier_error = float(
        spec["gates"].get("maximum_carrier_relative_error", np.nan)
    )
    energy_tolerance = float(
        spec["gates"].get("competitive_energy_tolerance_per_site", np.nan)
    )
    if not np.isfinite(minimum_ess) or not 0.0 < minimum_ess <= 1.0:
        raise ValueError("gates.minimum_ess_fraction must be in (0, 1]")
    if not np.isfinite(maximum_carrier_error) or maximum_carrier_error < 0.0:
        raise ValueError(
            "gates.maximum_carrier_relative_error must be finite and nonnegative"
        )
    if not np.isfinite(energy_tolerance) or energy_tolerance < 0.0:
        raise ValueError(
            "gates.competitive_energy_tolerance_per_site must be finite and nonnegative"
        )


def _su2_configuration(spec: dict[str, object]):
    model = spec["su2_model"]
    routes = triangular_dimer_complete_route_orders(6, 6)
    tree = pair_balanced_fusion_tree(36)
    j1 = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    j2 = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    route_features = multiscale_bond_route_features(
        tree, routes, {"J1": j1, "J2": j2}
    )
    names = tuple(routes)
    spatial = SupportAwareSpatialCGConfig(
        backbone=SpinClebschTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=int(model["feature_dim"]),
            maximum_intermediate_q=int(model["maximum_intermediate_q"]),
            initialization_scale=float(model["initialization_scale"]),
        ),
        route_names=names,
        route_orders=tuple(routes.values()),
        route_features=route_features.standardized,
        feature_names=route_features.labels,
        reference_route=names.index("row_dimers"),
    )
    base = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=int(model["depth"]),
        mlp_ratio=int(model["mlp_ratio"]),
        modulation_rank=int(model["modulation_rank"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    multiplicity_features = multiplicity_spatial_features(
        tree, spatial.route_orders, j1, j2
    )
    multiplicity = SU2MultiplicitySpatialAttentionConfig(
        base=base,
        features=multiplicity_features,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    config = ProjectedCarrierMultiplicityDressingConfig(
        multiplicity=multiplicity,
        projector=PhysicalSpatialProjector(
            group,
            orbit_chunk_size=int(model["orbit_chunk_size"]),
            rematerialize_orbit=True,
        ),
    )
    ledger = {
        "route_names": list(names),
        "route_count": len(names),
        "feature_names": list(route_features.labels),
        "group_order": len(group),
        "maximum_intermediate_q": int(model["maximum_intermediate_q"]),
    }
    return config, tuple(j1) + tuple(
        triangular_j1j2_bonds(6, 6, j1=0.0, j2=J2, periodic=True)
    ), ledger


def _initialize_su2(
    spec: dict[str, object],
    config: ProjectedCarrierMultiplicityDressingConfig,
    train_states: np.ndarray,
    train_carrier: np.ndarray,
    carrier_amplitude,
):
    model = spec["su2_model"]
    seed = int(spec["seed"])
    frozen = init_su2_route_vit(jax.random.PRNGKey(seed), config.multiplicity.base)
    dimer = init_spin_clebsch_tree_dimer_product(
        config.multiplicity.base.spatial.backbone
    )
    floor = float(model["backbone_support_floor"])
    if floor:
        random = init_spin_clebsch_tree(
            jax.random.PRNGKey(seed + 1),
            config.multiplicity.base.spatial.backbone,
        )
        dimer = jax.tree_util.tree_map(
            lambda left, right: left + floor * right, dimer, random
        )
    frozen = {**frozen, "backbone": dimer}
    dressing = init_projected_carrier_multiplicity_dressing(
        jax.random.PRNGKey(seed + 2),
        frozen,
        config,
        initial_residual_coefficient=1.0,
    )
    head = dressing["multiplicity"]
    head = {
        **head,
        "control_head": float(model["multiplicity_control_scale"])
        * jax.random.normal(
            jax.random.PRNGKey(seed + 3),
            head["control_head"].shape,
            dtype=jnp.float64,
        ),
    }
    dressing = {**dressing, "multiplicity": head}

    def zero_carrier(state):
        return jnp.asarray(0.0, dtype=jnp.float64)

    def residual(candidate, state):
        return certified_carrier_projected_residual_amplitude(
            candidate,
            frozen,
            state,
            config,
            carrier_amplitude_fn=zero_carrier,
        )

    evaluate = jax.jit(
        lambda candidate, states: jax.vmap(
            lambda row: residual(candidate, row)
        )(states)
    )
    microbatch_size = int(
        spec["training"].get("gradient_microbatch_size", len(train_states))
    )
    initial_residual = np.concatenate(
        [
            np.asarray(
                evaluate(
                    dressing,
                    jnp.asarray(train_states[start : start + microbatch_size], dtype=jnp.int32),
                )
            )
            for start in range(0, len(train_states), microbatch_size)
        ]
    )
    carrier_rms = float(np.sqrt(np.mean(np.abs(train_carrier) ** 2)))
    residual_rms = float(np.sqrt(np.mean(np.abs(initial_residual) ** 2)))
    if min(carrier_rms, residual_rms) <= 1.0e-300:
        raise RuntimeError("carrier or projected residual is numerically empty")
    readout_scale = carrier_rms / residual_rms
    frozen = {
        **frozen,
        "backbone": {
            **frozen["backbone"],
            "readout": frozen["backbone"]["readout"] * readout_scale,
        },
    }
    dressing = {
        **dressing,
        "residual_coefficient": jnp.asarray(
            float(model["initial_residual_coefficient"]), dtype=jnp.float64
        ),
    }
    analytic = np.concatenate(
        [
            np.asarray(
                jax.vmap(carrier_amplitude)(
                    jnp.asarray(
                        train_states[start : start + microbatch_size],
                        dtype=jnp.int32,
                    )
                )
            )
            for start in range(0, len(train_states), microbatch_size)
        ]
    )
    carrier_error = float(
        np.linalg.norm(analytic - train_carrier)
        / max(np.linalg.norm(train_carrier), np.finfo(float).tiny)
    )
    return {"frozen": frozen, "dressing": dressing}, {
        "carrier_rms": carrier_rms,
        "initial_projected_residual_rms": residual_rms,
        "backbone_readout_scale": readout_scale,
        "analytic_carrier_cache_relative_error": carrier_error,
    }


def _matched_two_lanczos(
    development: dict[str, object],
    train_raw: np.ndarray,
    audit_raw: np.ndarray,
    *,
    train_slice: slice,
    audit_slice: slice,
) -> dict[str, object]:
    """Fit the nested three-column Krylov control on registered train rows."""

    words = tuple(tuple(word) for word in development["words"])
    means = np.asarray(development["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(development["standardization"]["scales"], dtype=np.float64)
    train_components, train_action, train_carrier = _matrices(
        np.asarray(train_raw[train_slice]),
        words,
        coupling=J2,
        means=means,
        scales=scales,
    )
    audit_components, audit_action, audit_carrier = _matrices(
        np.asarray(audit_raw[audit_slice]),
        words,
        coupling=J2,
        means=means,
        scales=scales,
    )
    control_basis = np.asarray(development["control_basis"], dtype=np.float64)
    fit = krylov_control_fit(
        train_components,
        train_action,
        control_basis=control_basis,
    )
    j1_index = words.index(("J1_shell",))
    j2_index = words.index(("J2_shell",))
    selected_train_raw = np.asarray(train_raw[train_slice])
    return {
        "coefficients": fit.coefficients,
        "training_energy_total": fit.training_energy,
        "gram_rank": fit.gram_rank,
        "gram_condition_number": fit.gram_condition_number,
        "train_state": train_components @ fit.coefficients,
        "train_action": train_action @ fit.coefficients,
        "audit_state": audit_components @ fit.coefficients,
        "audit_action": audit_action @ fit.coefficients,
        "train_carrier": train_carrier,
        "audit_carrier": audit_carrier,
        "train_carrier_h": (
            selected_train_raw[:, j1_index]
            + J2 * selected_train_raw[:, j2_index]
        ),
    }


def _train_su2(
    spec,
    output,
    train_states,
    train_carrier,
    train_carrier_h,
    audit_states,
    audit_carrier,
    bonds,
    config,
    source_manifest_sha256,
    runtime_identity_sha256,
):
    segment_started = time.perf_counter()
    orbit = build_n36_pi_zero_flux_orbit()

    def carrier(state):
        return projected_pi_zero_flux_a1_amplitudes_jax(
            state[None, :], orbit=orbit
        )[0]

    def amplitude(candidate, state):
        return certified_carrier_projected_residual_amplitude(
            candidate["dressing"],
            candidate["frozen"],
            state,
            config,
            carrier_amplitude_fn=carrier,
        )

    train_connectivity = tuple(
        jnp.asarray(value)
        for value in padded_spin_connectivity(train_states, bonds)[:3]
    )
    audit_connectivity = tuple(
        jnp.asarray(value)
        for value in padded_spin_connectivity(audit_states, bonds)[:3]
    )
    run_identity = {
        "task": TASK,
        "arm": ARM_SU2,
        "config_sha256": canonical_sha256(spec),
        "training_states_sha256": array_sha256(train_states),
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    checkpoint_directory = output / "checkpoints" / ARM_SU2
    restored = (
        load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=run_identity
        )
        if _has_committed_checkpoint(checkpoint_directory)
        else None
    )
    if restored is None:
        params, initialization = _initialize_su2(
            spec, config, train_states, train_carrier, carrier
        )
        history = []
        completed_steps = 0
        prior_wall_seconds = 0.0
        # The N=36 Reynolds residual can have moderate center amplitudes but
        # very large connected-state tails.  Spend the first registered full-H
        # pass on an action-norm gauge calibration, rather than hiding this
        # cost in initialization or clipping local energies after the fact.
        unit_dressing = {
            **params["dressing"],
            "residual_coefficient": jnp.asarray(1.0, dtype=jnp.float64),
        }

        def zero_carrier(state):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def residual_only(candidate, state):
            return certified_carrier_projected_residual_amplitude(
                candidate["dressing"],
                candidate["frozen"],
                state,
                config,
                carrier_amplitude_fn=zero_carrier,
            )

        calibration_params = {**params, "dressing": unit_dressing}
        residual_center, residual_action = _batch_hpsi(
            calibration_params,
            jnp.asarray(train_states),
            *train_connectivity,
            config,
            residual_only,
        )
        residual_center, residual_action = np.asarray(residual_center), np.asarray(
            residual_action
        )
        residual_action_rms = float(np.sqrt(np.mean(np.abs(residual_action) ** 2)))
        carrier_action_rms = float(
            np.sqrt(np.mean(np.abs(np.asarray(train_carrier_h)) ** 2))
        )
        if min(residual_action_rms, carrier_action_rms) <= 1.0e-300:
            raise RuntimeError("carrier or residual Hamiltonian action is empty")
        action_scale = min(1.0, carrier_action_rms / residual_action_rms)
        params = {
            **params,
            "frozen": {
                **params["frozen"],
                "backbone": {
                    **params["frozen"]["backbone"],
                    "readout": params["frozen"]["backbone"]["readout"]
                    * action_scale,
                },
            },
        }
        initialization = {
            **initialization,
            "precalibration_residual_center_rms": float(
                np.sqrt(np.mean(np.abs(residual_center) ** 2))
            ),
            "precalibration_residual_action_rms": residual_action_rms,
            "carrier_action_rms": carrier_action_rms,
            "action_norm_scale": action_scale,
        }
        calibration = {
            "event": "registered_full_h_action_norm_calibration",
            "full_h_configuration_calls": len(train_states),
            "action_norm_scale": action_scale,
        }
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state={"kind": "normalized_common_sample_gradient"},
            completed_steps=0,
            next_sample_index=_su2_work_cursor(0, len(train_states)),
            run_identity=run_identity,
            history=history,
            metadata={
                "exact_sector": "S0_Gamma_A1",
                "carrier": "analytic_three_determinant_pi0_flux",
                "registered_calibration_pass_complete": True,
                "initialization": initialization,
                "calibration": calibration,
                "accumulated_arm_wall_seconds": time.perf_counter()
                - segment_started,
                "completed_full_h_configuration_calls": len(train_states),
            },
        )
    else:
        completed_steps = int(restored["completed_steps"])
        update_count = int(spec["training"]["steps"]) - 1
        prior_wall_seconds = float(
            restored["metadata"].get("accumulated_arm_wall_seconds", np.nan)
        )
        if completed_steps > update_count:
            raise ValueError("SU(2) checkpoint exceeds the registered update count")
        if not np.isfinite(prior_wall_seconds) or prior_wall_seconds < 0.0:
            raise ValueError("invalid SU(2) checkpoint accumulated wall time")
        expected_cursor = _su2_work_cursor(completed_steps, len(train_states))
        if (
            int(restored["next_sample_index"]) != expected_cursor
            or int(
                restored["metadata"].get(
                    "completed_full_h_configuration_calls", -1
                )
            )
            != expected_cursor
        ):
            raise ValueError("inconsistent SU(2) checkpoint work cursor")
        params = restored["params"]
        history = list(restored["history"])
        initialization = dict(restored["metadata"]["initialization"])
        calibration = dict(restored["metadata"]["calibration"])
    connected, elements, valid = train_connectivity
    update_count = int(spec["training"]["steps"]) - 1
    for step in range(completed_steps + 1, update_count + 1):
        gradient = common_sample_vmc_value_and_grad(
            params,
            jnp.asarray(train_states),
            jnp.asarray(train_carrier),
            hpsi_fn=lambda candidate, states: _batch_hpsi(
                candidate,
                states,
                connected,
                elements,
                valid,
                config,
                amplitude,
            ),
            center_fn=lambda candidate, states: jax.vmap(
                lambda row: amplitude(candidate, row)
            )(states),
            gradient_microbatch_size=int(
                spec["training"].get(
                    "gradient_microbatch_size", len(train_states)
                )
            ),
        )
        params, norm, scale = _normalized_gradient_step(
            params,
            gradient.gradient,
            float(spec["training"]["su2_learning_rate"]),
        )
        params = {
            **params,
            "dressing": {
                **params["dressing"],
                "residual_coefficient": jnp.clip(
                    params["dressing"]["residual_coefficient"], -0.25, 0.25
                ),
            },
        }
        history.append(
            {
                "step": step,
                "training_energy_per_site": gradient.energy / N_SITES,
                "training_variance_per_site": gradient.variance / N_SITES,
                "training_ess_fraction": gradient.ess_fraction,
                "gradient_norm": norm,
                "applied_gradient_scale": scale,
                "residual_coefficient": float(
                    params["dressing"]["residual_coefficient"]
                ),
            }
        )
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state={"kind": "normalized_common_sample_gradient"},
            completed_steps=step,
            next_sample_index=_su2_work_cursor(step, len(train_states)),
            run_identity=run_identity,
            history=history,
            metadata={
                "exact_sector": "S0_Gamma_A1",
                "carrier": "analytic_three_determinant_pi0_flux",
                "registered_calibration_pass_complete": True,
                "initialization": initialization,
                "calibration": calibration,
                "accumulated_arm_wall_seconds": prior_wall_seconds
                + time.perf_counter()
                - segment_started,
                "completed_full_h_configuration_calls": _su2_work_cursor(
                    step, len(train_states)
                ),
            },
        )
    centers, actions = _batch_hpsi(
        params,
        jnp.asarray(audit_states),
        *audit_connectivity,
        config,
        amplitude,
    )
    jax.block_until_ready((centers, actions))
    accumulated_wall_seconds = (
        prior_wall_seconds + time.perf_counter() - segment_started
    )
    return {
        "audit_state": np.asarray(centers) / np.asarray(audit_carrier),
        "audit_action": np.asarray(actions) / np.asarray(audit_carrier),
        "history": history,
        "wall_seconds": accumulated_wall_seconds,
        "timing_scope": (
            "complete arm function including initialization, action-norm "
            "calibration, updates, and audit; prior successfully checkpointed "
            "segments are accumulated on resume"
        ),
        "calibration": calibration,
        "completed_full_h_configuration_calls_before_audit": _su2_work_cursor(
            update_count, len(train_states)
        ),
        "parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "parameter_sha256": pytree_sha256(params),
        "initialization": initialization,
    }


def _train_fixed_sz(
    spec,
    output,
    train_states,
    train_carrier,
    audit_states,
    audit_carrier,
    bonds,
    source_manifest_sha256,
    runtime_identity_sha256,
):
    segment_started = time.perf_counter()
    model = spec["fixed_sz_model"]
    config = FixedSzSpatialViTConfig(
        geometry="triangular",
        rows=6,
        columns=6,
        patch_rows=int(model["patch_rows"]),
        patch_columns=int(model["patch_columns"]),
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=int(model["depth"]),
        mlp_ratio=int(model["mlp_ratio"]),
        target_m2=0,
        initialization_scale=float(model["initialization_scale"]),
        use_distance_envelope=True,
    )
    projector = PhysicalSpatialProjector(
        triangular_hnf_gamma_a1_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(model["orbit_chunk_size"]),
        rematerialize_orbit=True,
    )
    orbit = build_n36_pi_zero_flux_orbit()
    params = init_fixed_sz_spatial_vit(
        jax.random.PRNGKey(int(spec["seed"]) + 31), config
    )
    initial = params

    def raw(candidate, state, _config):
        current = fixed_sz_spatial_vit_log_amplitude(candidate, state, config)
        reference = jax.lax.stop_gradient(
            fixed_sz_spatial_vit_log_amplitude(initial, state, config)
        )
        return jnp.exp(jnp.real(current - reference))

    def amplitude(candidate, state):
        carrier = projected_pi_zero_flux_a1_amplitudes_jax(
            state[None, :], orbit=orbit
        )[0]
        scalar = spatial_projected_amplitude(
            candidate, state, config, projector, amplitude_fn=raw
        )
        return carrier * scalar

    train_connectivity = tuple(
        jnp.asarray(value)
        for value in padded_spin_connectivity(train_states, bonds)[:3]
    )
    audit_connectivity = tuple(
        jnp.asarray(value)
        for value in padded_spin_connectivity(audit_states, bonds)[:3]
    )
    run_identity = {
        "task": TASK,
        "arm": ARM_FIXED_SZ,
        "config_sha256": canonical_sha256(spec),
        "training_states_sha256": array_sha256(train_states),
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    checkpoint_directory = output / "checkpoints" / ARM_FIXED_SZ
    restored = (
        load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=run_identity
        )
        if _has_committed_checkpoint(checkpoint_directory)
        else None
    )
    if restored is None:
        history = []
        completed_steps = 0
        prior_wall_seconds = 0.0
    else:
        completed_steps = int(restored["completed_steps"])
        registered_steps = int(spec["training"]["steps"])
        prior_wall_seconds = float(
            restored["metadata"].get("accumulated_arm_wall_seconds", np.nan)
        )
        if completed_steps > registered_steps:
            raise ValueError("fixed-Sz checkpoint exceeds the registered update count")
        if not np.isfinite(prior_wall_seconds) or prior_wall_seconds < 0.0:
            raise ValueError("invalid fixed-Sz checkpoint accumulated wall time")
        expected_cursor = completed_steps * len(train_states)
        if (
            int(restored["next_sample_index"]) != expected_cursor
            or int(
                restored["metadata"].get(
                    "completed_full_h_configuration_calls", -1
                )
            )
            != expected_cursor
        ):
            raise ValueError("inconsistent fixed-Sz checkpoint work cursor")
        params = restored["params"]
        history = list(restored["history"])
    connected, elements, valid = train_connectivity
    for step in range(completed_steps + 1, int(spec["training"]["steps"]) + 1):
        gradient = common_sample_vmc_value_and_grad(
            params,
            jnp.asarray(train_states),
            jnp.asarray(train_carrier),
            hpsi_fn=lambda candidate, states: _batch_hpsi(
                candidate,
                states,
                connected,
                elements,
                valid,
                config,
                amplitude,
            ),
            center_fn=lambda candidate, states: jax.vmap(
                lambda row: amplitude(candidate, row)
            )(states),
            gradient_microbatch_size=int(
                spec["training"].get(
                    "gradient_microbatch_size", len(train_states)
                )
            ),
        )
        params, norm, scale = _normalized_gradient_step(
            params,
            gradient.gradient,
            float(spec["training"]["fixed_sz_learning_rate"]),
        )
        history.append(
            {
                "step": step,
                "training_energy_per_site": gradient.energy / N_SITES,
                "training_variance_per_site": gradient.variance / N_SITES,
                "training_ess_fraction": gradient.ess_fraction,
                "gradient_norm": norm,
                "applied_gradient_scale": scale,
            }
        )
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state={"kind": "normalized_common_sample_gradient"},
            completed_steps=step,
            next_sample_index=step * len(train_states),
            run_identity=run_identity,
            history=history,
            metadata={
                "spin_sector": "fixed_Sz0_not_exact_S2",
                "accumulated_arm_wall_seconds": prior_wall_seconds
                + time.perf_counter()
                - segment_started,
                "completed_full_h_configuration_calls": step
                * len(train_states),
            },
        )
    centers, actions = _batch_hpsi(
        params,
        jnp.asarray(audit_states),
        *audit_connectivity,
        config,
        amplitude,
    )
    jax.block_until_ready((centers, actions))
    accumulated_wall_seconds = (
        prior_wall_seconds + time.perf_counter() - segment_started
    )
    return {
        "audit_state": np.asarray(centers) / np.asarray(audit_carrier),
        "audit_action": np.asarray(actions) / np.asarray(audit_carrier),
        "history": history,
        "wall_seconds": accumulated_wall_seconds,
        "timing_scope": (
            "complete arm function including deterministic initialization, "
            "updates, and audit; prior successfully checkpointed segments "
            "are accumulated on resume"
        ),
        "completed_full_h_configuration_calls_before_audit": len(train_states)
        * int(spec["training"]["steps"]),
        "parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "parameter_sha256": pytree_sha256(params),
    }


def _accelerator_ledger(wall_seconds: float) -> dict[str, object]:
    gpu_count = len([device for device in jax.devices() if device.platform == "gpu"])
    return {
        "instrumented_arm_wall_seconds": float(wall_seconds),
        "gpu_count": gpu_count,
        "instrumented_arm_gpu_hours": float(
            wall_seconds * gpu_count / 3600.0
        ),
        "cpu_wall_hours": float(wall_seconds / 3600.0) if gpu_count == 0 else 0.0,
        "platforms": sorted({device.platform for device in jax.devices()}),
        "accounting_boundary": (
            "Arm-function timer including cold initialization in an uninterrupted "
            "run. Resumed results accumulate only successfully checkpointed prior "
            "segments, so authoritative allocated GPU-hours require Slurm sacct."
        ),
    }


def run_spin_n36_projected_multiplicity_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    workflow_started = time.perf_counter()
    spec = json.loads(config_path.read_text())
    _validate(spec)
    source_manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    source_manifest_sha256 = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_identity_sha256 = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists: {result_path}")
    development, audit, train_states_all, train_raw, audit_states_all, audit_raw = (
        _sealed_common_data(spec)
    )
    full_baseline = _baseline_from_cache(
        development,
        audit,
        train_raw,
        audit_raw,
        require_sealed_reproduction=True,
    )
    train_count = int(spec["samples"]["training_count"])
    audit_count = int(spec["samples"]["audit_count"])
    train_start = int(spec["samples"].get("training_start", 0))
    audit_start = int(spec["samples"].get("audit_start", 0))
    train_slice = slice(train_start, train_start + train_count)
    audit_slice = slice(audit_start, audit_start + audit_count)
    if (
        train_start + train_count > len(train_states_all)
        or audit_start + audit_count > len(audit_states_all)
    ):
        raise ValueError("requested sample count exceeds the sealed cache")
    train_states = np.asarray(train_states_all[train_slice], dtype=np.int8)
    audit_states = np.asarray(audit_states_all[audit_slice], dtype=np.int8)
    selected_train_raw = np.asarray(train_raw[train_slice])
    selected_audit_raw = np.asarray(audit_raw[audit_slice])
    for name, observed in (
        ("training_raw_actions_sha256", array_sha256(selected_train_raw)),
        ("audit_raw_actions_sha256", array_sha256(selected_audit_raw)),
    ):
        expected = spec["samples"].get(name)
        if expected is not None and expected != observed:
            raise RuntimeError(f"sealed selected cache mismatch: {name}")
    matched_baseline = _matched_two_lanczos(
        development,
        train_raw,
        audit_raw,
        train_slice=train_slice,
        audit_slice=audit_slice,
    )
    frozen_baseline_summary = _ratio_summary(
        np.asarray(full_baseline["audit_state"][audit_slice]),
        np.asarray(full_baseline["audit_action"][audit_slice]),
        chain_count=audit_count,
        n_sites=N_SITES,
    )
    frozen_coefficients = np.asarray(
        development["frozen_two_lanczos"]["coefficients"], dtype=np.float64
    )
    coefficient_relative_change = float(
        np.linalg.norm(matched_baseline["coefficients"] - frozen_coefficients)
        / np.linalg.norm(frozen_coefficients)
    )
    train_carrier = np.asarray(matched_baseline["train_carrier"])
    audit_carrier = np.asarray(matched_baseline["audit_carrier"])
    baseline_state = np.asarray(matched_baseline["audit_state"])
    baseline_action = np.asarray(matched_baseline["audit_action"])
    chain_count = audit_count
    baseline_summary = _ratio_summary(
        baseline_state,
        baseline_action,
        chain_count=chain_count,
        n_sites=N_SITES,
    )
    su2_config, bonds, route_ledger = _su2_configuration(spec)
    su2 = _train_su2(
        spec,
        output,
        train_states,
        train_carrier,
        np.asarray(matched_baseline["train_carrier_h"]),
        audit_states,
        audit_carrier,
        bonds,
        su2_config,
        source_manifest_sha256,
        runtime_identity_sha256,
    )
    fixed = _train_fixed_sz(
        spec,
        output,
        train_states,
        train_carrier,
        audit_states,
        audit_carrier,
        bonds,
        source_manifest_sha256,
        runtime_identity_sha256,
    )
    su2_summary = _ratio_summary(
        su2["audit_state"], su2["audit_action"], chain_count=chain_count, n_sites=N_SITES
    )
    fixed_summary = _ratio_summary(
        fixed["audit_state"],
        fixed["audit_action"],
        chain_count=chain_count,
        n_sites=N_SITES,
    )
    baseline_arm = {
        "audit_state": baseline_state,
        "audit_action": baseline_action,
    }
    su2_minus_baseline = _paired_chain_difference(
        su2["audit_state"],
        su2["audit_action"],
        baseline_state,
        baseline_action,
        chain_count=chain_count,
    )
    su2_minus_fixed = _paired_chain_difference(
        su2["audit_state"],
        su2["audit_action"],
        fixed["audit_state"],
        fixed["audit_action"],
        chain_count=chain_count,
    )
    fixed_minus_baseline = _paired_chain_difference(
        fixed["audit_state"],
        fixed["audit_action"],
        baseline_arm["audit_state"],
        baseline_arm["audit_action"],
        chain_count=chain_count,
    )
    baseline_calls = 3 * (train_count + audit_count)
    neural_calls = int(spec["training"]["steps"]) * train_count + audit_count
    if baseline_calls != neural_calls:
        raise RuntimeError("the three arms do not have equal full-H call counts")
    tolerance = float(spec["gates"]["competitive_energy_tolerance_per_site"])
    minimum_ess = float(spec["gates"]["minimum_ess_fraction"])
    arms = {
        ARM_BASELINE: {
            "summary": baseline_summary,
            "registered_train_block_fit": {
                "coefficients": np.asarray(
                    matched_baseline["coefficients"]
                ).tolist(),
                "training_energy_total": matched_baseline[
                    "training_energy_total"
                ],
                "gram_rank": matched_baseline["gram_rank"],
                "gram_condition_number": matched_baseline[
                    "gram_condition_number"
                ],
                "relative_change_from_512_row_frozen_coefficients": (
                    coefficient_relative_change
                ),
                "frozen_512_row_coefficients_on_selected_audit": (
                    frozen_baseline_summary
                ),
            },
            "full_h_configuration_calls": baseline_calls,
            "accelerator": {
                "current_run_wall_seconds": 0.0,
                "gpu_count": 0,
                "current_run_gpu_hours": 0.0,
                "cached_production_work_reused": True,
                "historical_cache_provenance": spec.get(
                    "baseline_cache_provenance"
                ),
            },
            "exact_su2": True,
            "exact_gamma_a1": True,
        },
        ARM_FIXED_SZ: {
            "summary": fixed_summary,
            "full_h_configuration_calls": neural_calls,
            "accelerator": _accelerator_ledger(fixed["wall_seconds"]),
            "parameter_count": fixed["parameter_count"],
            "parameter_sha256": fixed["parameter_sha256"],
            "training_history": fixed["history"],
            "timing_scope": fixed["timing_scope"],
            "exact_su2": False,
            "exact_gamma_a1": True,
        },
        ARM_SU2: {
            "summary": su2_summary,
            "full_h_configuration_calls": neural_calls,
            "accelerator": _accelerator_ledger(su2["wall_seconds"]),
            "parameter_count": su2["parameter_count"],
            "parameter_sha256": su2["parameter_sha256"],
            "training_history": su2["history"],
            "initialization": su2["initialization"],
            "calibration": su2["calibration"],
            "timing_scope": su2["timing_scope"],
            "exact_su2": True,
            "exact_gamma_a1": True,
            "analytic_three_determinant_carrier": True,
            "configuration_dependent_physical_residual": True,
        },
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "execution_tier": spec["execution_tier"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": spec["claim_boundary"],
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic torus",
            "j2_over_j1": J2,
            "route_ledger": route_ledger,
        },
        "common_samples": {
            "training_count": train_count,
            "audit_count": audit_count,
            "training_start": train_start,
            "audit_start": audit_start,
            "effective_chain_count": chain_count,
            "training_sha256": array_sha256(train_states),
            "audit_sha256": array_sha256(audit_states),
            "training_raw_actions_sha256": array_sha256(selected_train_raw),
            "audit_raw_actions_sha256": array_sha256(selected_audit_raw),
            "identical_configurations_for_all_arms": True,
            "reuses_previously_sealed_chain_cache": True,
            "remote_block_indices_are_disjoint_from_local_preflight": (
                spec["execution_tier"] != "remote_gpu_pilot"
                or (train_start >= 4 and audit_start >= 4)
            ),
            "training_configuration_overlap_with_local_first_four": int(
                sum(
                    any(np.array_equal(row, prior) for prior in train_states_all[:4])
                    for row in train_states
                )
            ),
            "audit_configuration_overlap_with_local_first_four": int(
                sum(
                    any(np.array_equal(row, prior) for prior in audit_states_all[:4])
                    for row in audit_states
                )
            ),
        },
        "arms": arms,
        "paired_comparisons": {
            "su2_minus_two_lanczos": su2_minus_baseline,
            "su2_minus_fixed_sz_vit": su2_minus_fixed,
            "fixed_sz_vit_minus_two_lanczos": fixed_minus_baseline,
        },
        "work_contract": {
            "same_samples": True,
            "same_full_h_configuration_calls": True,
            "full_h_configuration_calls_per_arm": baseline_calls,
            "same_gpu_hours": False,
            "gpu_hours_are_measured_as_an_efficiency_axis_not_forced_equal": True,
            "arm_gpu_hours_are_instrumented_complete_arm_function_times": True,
            "authoritative_total_allocated_gpu_hours_require_slurm_sacct": True,
            "baseline_cached_gpu_hours_not_reconstructed": True,
            "baseline_historical_cache_cost_is_only_an_upper_bound": (
                "baseline_cache_provenance" in spec
            ),
        },
        "execution_accounting": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "workflow_instrumented_wall_seconds": time.perf_counter()
            - workflow_started,
            "gpu_count": len(
                [device for device in jax.devices() if device.platform == "gpu"]
            ),
        },
        "decision": _preflight_decision(
            execution_tier=str(spec["execution_tier"]),
            baseline_summary=baseline_summary,
            fixed_summary=fixed_summary,
            su2_summary=su2_summary,
            su2_minus_baseline=su2_minus_baseline,
            su2_minus_fixed=su2_minus_fixed,
            su2_history=su2["history"],
            fixed_history=fixed["history"],
            training_steps=int(spec["training"]["steps"]),
            carrier_relative_error=float(
                su2["initialization"]["analytic_carrier_cache_relative_error"]
            ),
            maximum_carrier_relative_error=float(
                spec["gates"]["maximum_carrier_relative_error"]
            ),
            minimum_ess=minimum_ess,
            energy_tolerance=tolerance,
        ),
        "interpretation_boundary": {
            "local_preflight_is_not_an_untouched_physics_claim": True,
            "four_or_sixty_four_samples_cannot_establish_a_scaling_win": True,
            "near_exact_n16_result_is_not_used_as_an_n36_target": True,
        },
        "hamiltonian_diagonalization_performed": False,
        "runtime_identity": runtime_identity,
    }
    payload["source_manifest"] = source_manifest
    payload["source_manifest_sha256"] = source_manifest_sha256
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_n36_projected_multiplicity_preflight"]
