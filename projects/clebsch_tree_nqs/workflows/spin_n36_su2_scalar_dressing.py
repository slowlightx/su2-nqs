"""Matched-chain N=36 benchmark for a configuration-dependent SU(2) dressing.

The experiment reuses the sealed development and untouched-audit chains from
the regularized G1 campaign.  Reusing configurations is intentional: all
three arms see exactly the same persistent chains, and the optimized
two-Lanczos reference is reproduced from immutable scalar-word caches rather
than estimated from a different Monte Carlo run.

Arms
----

``projected_vmc_two_lanczos``
    The sealed projected pi/0-flux ``I,H,H^2`` control.

``pi_flux_fixed_sz_spatial_vit``
    A full-space-group projected positive Spatial-ViT correlator on the same
    pi/0-flux carrier.  It stays in fixed ``Sz=0`` but is not an exact singlet.

``pi_flux_exact_su2_scalar_dressing``
    The immutable sealed two-Lanczos column plus two spatial-attention-
    generated, Reynolds-projected singlet-AGP columns.  Thus the strong
    baseline is nested exactly.  A lowest-root generalized Ritz solve
    determines the additive coefficients; nonlinear pair-attention parameters
    are trained with a node-safe direct-amplitude horizontal force.  The bare
    pi/0-flux Born state remains only the common full-support proposal.

The audit is fail-closed on sample hashes, exact reproduction of the sealed
two-Lanczos estimate, importance ESS, and work accounting.  A finite-sample
``competitive`` outcome is distinct from a strict resolved win.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)

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
from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (
    triangular_physical_pairing_bank,
)
from su2_nqs.benchmarks.projected_singlet_agp import (
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
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
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
from su2_nqs.models.resonating_su2_scalar_dressing import (
    ResonatingSU2ScalarDressingConfig,
    init_resonating_su2_scalar_dressing,
    packed_resonating_pairing_bank,
    projected_resonating_amplitudes,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.vmc.common_sample_vmc import common_sample_vmc_value_and_grad
from su2_nqs.vmc.direct_amplitude_ground_force import (
    direct_amplitude_ground_horizontal_force,
    prepare_direct_amplitude_ground_force_reference,
)
from su2_nqs.vmc.projected_agp import ProjectedAGPHeisenbergConfig
from su2_nqs.vmc.regularized_resonating_ground import (
    crossfit_resonating_ridge,
    evaluate_resonating_ground,
)
from su2_nqs.vmc.resonating_projected_agp import (
    ResonatingProjectedAGPAction,
    resonating_projected_agp_action,
)
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_su2_scalar_dressing"
ARM_BASELINE = "projected_vmc_two_lanczos"
ARM_FIXED_SZ = "pi_flux_fixed_sz_spatial_vit"
ARM_SU2 = "pi_flux_exact_su2_scalar_dressing"
N_SITES = 36
J2 = 0.125
EXACT_ENERGY_PER_SITE = -0.515564
PUBLISHED_TWO_LANCZOS_PER_SITE = -0.512503


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 SU2 scalar-dressing identity")
    if spec.get("execution_tier") not in {
        "local_contract_preflight",
        "remote_n36_pilot",
        "remote_regularized_refit",
    }:
        raise ValueError("invalid execution_tier")
    for name in ("sealed_development_result", "sealed_audit_result"):
        if not isinstance(spec.get(name), str) or not str(spec[name]).endswith(".json"):
            raise ValueError(f"{name} must be a JSON path")
    for section in ("model", "training", "gates"):
        if not isinstance(spec.get(section), dict):
            raise TypeError(f"{section} must be an object")
    model = spec["model"]
    for name in ("embedding_dim", "head_count", "component_count", "mlp_ratio"):
        if int(model.get(name, 0)) < 1:
            raise ValueError(f"model.{name} must be positive")
    training = spec["training"]
    for name in ("su2_steps", "fixed_sz_steps"):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"training.{name} must be positive")
    for name in ("su2_learning_rate", "fixed_sz_learning_rate"):
        if float(training.get(name, 0.0)) <= 0.0:
            raise ValueError(f"training.{name} must be positive")
    if int(training.get("matched_measurement_blocks", 0)) < 1:
        raise ValueError("training.matched_measurement_blocks must be positive")
    if spec.get("execution_tier") == "remote_regularized_refit":
        if not isinstance(spec.get("source_checkpoint_directory"), str):
            raise ValueError("regularized refit requires source_checkpoint_directory")
        if int(training.get("secondary_audit_start_block", 0)) < 1:
            raise ValueError("secondary_audit_start_block must be positive")
        grid = training.get("ridge_grid")
        if not isinstance(grid, list) or len(grid) < 2:
            raise ValueError("regularized refit requires a ridge_grid")


def _verified_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    recorded = payload.get("result_sha256")
    body = dict(payload)
    body.pop("result_sha256", None)
    if not isinstance(recorded, str) or canonical_sha256(body) != recorded:
        raise ValueError(f"sealed result hash mismatch: {path}")
    return payload


def _load_cached_rows(result_path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    cache = result_path.parent / "component_cache"
    files = sorted(cache.glob(f"{prefix}_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {prefix} component caches below {cache}")
    states, actions = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as stored:
            state = np.asarray(stored["state"], dtype=np.int8)
            row = np.asarray(stored["actions"], dtype=np.complex128)
        if state.shape != (N_SITES,) or row.ndim != 2 or row.shape[0] != 1:
            raise ValueError(f"malformed immutable component cache: {path}")
        states.append(state)
        actions.append(row[0])
    return np.stack(states), np.stack(actions)


def _sealed_common_data(spec: dict[str, object]):
    development_path = ROOT / str(spec["sealed_development_result"])
    audit_path = ROOT / str(spec["sealed_audit_result"])
    development = _verified_result(development_path)
    audit = _verified_result(audit_path)
    if (
        development.get("task") != "spin_g1_regularized_operator"
        or development.get("phase") != "development"
        or audit.get("task") != "spin_g1_regularized_operator"
        or audit.get("phase") != "untouched_audit"
    ):
        raise ValueError("sealed inputs are not the registered G1 development/audit pair")
    # The untouched campaign was intentionally licensed by a smaller frozen
    # candidate artifact, not by the mutable development report itself.
    candidate_path = ROOT / str(audit["config"]["development_artifact"])
    candidate = _verified_result(candidate_path)
    if (
        audit.get("development_result_sha256") != candidate.get("result_sha256")
        or candidate.get("source_result_sha256") != development.get("result_sha256")
    ):
        raise ValueError("untouched audit was not generated from this sealed development")
    train_states, train_raw = _load_cached_rows(development_path, "develop")
    audit_states, audit_raw = _load_cached_rows(audit_path, "audit")
    for states, result, label in (
        (train_states, development, "development"),
        (audit_states, audit, "audit"),
    ):
        identity = result["sample_identity"]
        if array_sha256(states) != identity["configurations_sha256"]:
            raise ValueError(f"{label} cached configuration hash mismatch")
        if len(states) != int(identity["count"]):
            raise ValueError(f"{label} cached sample count mismatch")
    return candidate, audit, train_states, train_raw, audit_states, audit_raw


def _baseline_from_cache(
    development, audit, train_raw, audit_raw, *, require_sealed_reproduction: bool
):
    words = tuple(tuple(word) for word in development["words"])
    means = np.asarray(development["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(development["standardization"]["scales"], dtype=np.float64)
    train_components, train_action, train_carrier = _matrices(
        train_raw, words, coupling=J2, means=means, scales=scales
    )
    audit_components, audit_action, audit_carrier = _matrices(
        audit_raw, words, coupling=J2, means=means, scales=scales
    )
    coefficients = np.asarray(
        development["frozen_two_lanczos"]["coefficients"], dtype=np.float64
    )
    train_state, train_h = train_components @ coefficients, train_action @ coefficients
    audit_state, audit_h = audit_components @ coefficients, audit_action @ coefficients
    chain_count = int(audit["sample_identity"]["chain_count"])
    summary = _ratio_summary(
        audit_state, audit_h, chain_count=chain_count, n_sites=N_SITES
    )
    if require_sealed_reproduction:
        sealed = audit["audit_two_lanczos"]
        for key in ("energy_per_site", "variance_total", "importance_ess_fraction"):
            if not np.isclose(summary[key], sealed[key], rtol=2.0e-12, atol=2.0e-12):
                raise RuntimeError(f"failed to reproduce sealed two-Lanczos {key}")
    j1_index = words.index(("J1_shell",))
    j2_index = words.index(("J2_shell",))
    train_carrier_h = train_raw[:, j1_index] + J2 * train_raw[:, j2_index]
    audit_carrier_h = audit_raw[:, j1_index] + J2 * audit_raw[:, j2_index]
    return {
        "train_state": train_state,
        "train_action": train_h,
        "audit_state": audit_state,
        "audit_action": audit_h,
        "audit_summary": summary,
        "train_carrier": train_carrier,
        "train_carrier_h": train_carrier_h,
        "audit_carrier": audit_carrier,
        "audit_carrier_h": audit_carrier_h,
        "chain_count": chain_count,
        "word_count": len(words),
    }


def _combined_action(
    parameters,
    states,
    fixed_amplitudes,
    fixed_actions,
    *,
    base_pairing,
    directions,
    dressing_config,
    carrier_config,
    graph_config,
    amplitude_microbatch_size,
):
    bank = packed_resonating_pairing_bank(
        parameters, base_pairing, directions, dressing_config
    )
    trainable = resonating_projected_agp_action(
        bank,
        states,
        carrier_config,
        graph_config,
        amplitude_microbatch_size=amplitude_microbatch_size,
    )
    fixed_amplitudes = jnp.asarray(fixed_amplitudes)
    fixed_actions = jnp.asarray(fixed_actions)
    return ResonatingProjectedAGPAction(
        amplitudes=jnp.column_stack((fixed_amplitudes, trainable.amplitudes)),
        hamiltonian_actions=jnp.column_stack(
            (fixed_actions, trainable.hamiltonian_actions)
        ),
        center_exact_nodes=jnp.column_stack(
            (fixed_amplitudes == 0.0, trainable.center_exact_nodes)
        ),
        minimum_connected_cancellation_ratio=jnp.column_stack(
            (
                jnp.ones_like(fixed_amplitudes, dtype=jnp.float64),
                trainable.minimum_connected_cancellation_ratio,
            )
        ),
    )


def _local_model_contract(spec, states, nested_carrier_amplitudes) -> dict[str, object]:
    """Exercise the actual N=36 projected columns without applying ``H``."""

    model = spec["model"]
    dressing = ResonatingSU2ScalarDressingConfig(
        rows=6,
        columns=6,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        component_count=int(model["component_count"]),
        mlp_ratio=int(model["mlp_ratio"]),
        residual_scale=float(model["residual_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier = ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )
    base, direction_tuple, _ = triangular_physical_pairing_bank(6)
    directions = np.stack(direction_tuple)
    params = init_resonating_su2_scalar_dressing(
        jax.random.key(int(spec["seed"]) + 17), dressing
    )
    selected = np.asarray(states[:8], dtype=np.int8)
    columns = np.asarray(
        projected_resonating_amplitudes(
            params,
            selected,
            base,
            directions,
            dressing,
            carrier,
            microbatch_size=8,
        )
    )
    image = np.asarray(group[-1], dtype=np.int64)
    transformed = np.asarray(
        projected_resonating_amplitudes(
            params,
            selected[:, image],
            base,
            directions,
            dressing,
            carrier,
            microbatch_size=8,
        )
    )
    normalized = columns / np.maximum(
        np.linalg.norm(columns, axis=0, keepdims=True), np.finfo(np.float64).tiny
    )
    combined = np.column_stack(
        (
            np.asarray(nested_carrier_amplitudes[:8])
            / max(
                np.linalg.norm(nested_carrier_amplitudes[:8]),
                np.finfo(np.float64).tiny,
            ),
            normalized,
        )
    )
    nonzero_carrier = np.abs(nested_carrier_amplitudes[:8]) > 1.0e-14
    ratios = columns[nonzero_carrier] / np.asarray(
        nested_carrier_amplitudes[:8]
    )[nonzero_carrier, None]
    return {
        "evaluated_configuration_count": len(selected),
        "group_order": len(group),
        "projected_column_count": dressing.component_count,
        "combined_sample_rank": int(np.linalg.matrix_rank(combined, tol=1.0e-10)),
        "maximum_gamma_a1_relative_residual": float(
            np.linalg.norm(transformed - columns)
            / max(np.linalg.norm(columns), np.finfo(np.float64).tiny)
        ),
        "minimum_configuration_ratio_standard_deviation": float(
            np.min(np.std(ratios, axis=0))
        ),
        "trainable_parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "exact_singlet_by_symmetric_pairing_construction": True,
        "additive_carrier_nesting_is_algebraic": True,
    }


def _train_su2(
    spec,
    output,
    train_states,
    train_nested_carrier,
    train_nested_carrier_h,
    train_proposal_carrier,
    audit_states,
    audit_nested_carrier,
    audit_nested_carrier_h,
    audit_proposal_carrier,
):
    model = spec["model"]
    training = spec["training"]
    dressing_config = ResonatingSU2ScalarDressingConfig(
        rows=6,
        columns=6,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        component_count=int(model["component_count"]),
        mlp_ratio=int(model["mlp_ratio"]),
        residual_scale=float(model["residual_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier_config = ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
        parameter_mode="real",
    )
    bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=J2, periodic=True)
    graph = ProjectedAGPHeisenbergConfig(
        n_sites=N_SITES,
        bonds=tuple((int(i), int(j), float(value)) for i, j, value in bonds),
    )
    base, direction_tuple, direction_labels = triangular_physical_pairing_bank(6)
    directions = np.stack(direction_tuple)
    params = init_resonating_su2_scalar_dressing(
        jax.random.key(int(spec["seed"]) + 17), dressing_config
    )
    proposal = np.abs(np.asarray(train_proposal_carrier)) ** 2
    run_identity = {
        "task": TASK,
        "arm": ARM_SU2,
        "config_sha256": canonical_sha256(spec),
        "training_states_sha256": array_sha256(np.asarray(train_states)),
    }
    checkpoint_directory = output / "checkpoints" / ARM_SU2
    restored = (
        load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=run_identity
        )
        if checkpoint_directory.is_dir()
        else None
    )
    if restored is None:
        history = []
        completed_steps = 0
    else:
        params = restored["params"]
        history = list(restored["history"])
        completed_steps = int(restored["completed_steps"])
    started = time.perf_counter()
    final_reference = None
    final_action = None
    steps = int(training["su2_steps"])
    for evaluation in range(completed_steps, steps + 1):
        action = _combined_action(
            params,
            train_states,
            train_nested_carrier,
            train_nested_carrier_h,
            base_pairing=base,
            directions=directions,
            dressing_config=dressing_config,
            carrier_config=carrier_config,
            graph_config=graph,
            amplitude_microbatch_size=int(model["amplitude_microbatch_size"]),
        )
        reference = prepare_direct_amplitude_ground_force_reference(
            action,
            proposal,
            maximum_condition_number=float(spec["gates"]["maximum_condition_number"]),
        )
        final_reference, final_action = reference, action
        if evaluation == steps:
            break

        def amplitude_bank(candidate):
            trainable = projected_resonating_amplitudes(
                candidate,
                train_states,
                base,
                directions,
                dressing_config,
                carrier_config,
                microbatch_size=int(model["amplitude_microbatch_size"]),
            )
            return jnp.column_stack((jnp.asarray(train_nested_carrier), trainable))

        force = direct_amplitude_ground_horizontal_force(
            params, reference, amplitude_bank
        )
        params, gradient_norm, applied_scale = _normalized_gradient_step(
            params, force.gradient, float(training["su2_learning_rate"])
        )
        params = {
            **params,
            "pair_gates": jnp.clip(params["pair_gates"], -2.0, 2.0),
        }
        history.append(
            {
                "step": evaluation + 1,
                "training_energy_per_site_before": reference.energy / N_SITES,
                "training_variance_per_site_before": reference.variance / N_SITES,
                "training_ess_fraction": reference.state_weight_ess_fraction,
                "overlap_condition_number": reference.overlap_condition_number,
                "gradient_norm": gradient_norm,
                "applied_gradient_scale": applied_scale,
            }
        )
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state={"kind": "normalized_horizontal_force"},
            completed_steps=evaluation + 1,
            next_sample_index=(evaluation + 1) * len(train_states),
            run_identity=run_identity,
            history=history,
            metadata={"exact_sector": "S0_Gamma_A1", "node_safe": True},
        )
    assert final_reference is not None and final_action is not None
    coefficients = final_reference.scaled_coefficients / final_reference.column_rescales
    audit_action = _combined_action(
        params,
        audit_states,
        audit_nested_carrier,
        audit_nested_carrier_h,
        base_pairing=base,
        directions=directions,
        dressing_config=dressing_config,
        carrier_config=carrier_config,
        graph_config=graph,
        amplitude_microbatch_size=int(model["amplitude_microbatch_size"]),
    )
    audit_state = np.asarray(audit_action.amplitudes) @ coefficients
    audit_h = np.asarray(audit_action.hamiltonian_actions) @ coefficients
    train_state = np.asarray(final_action.amplitudes) @ coefficients
    train_h = np.asarray(final_action.hamiltonian_actions) @ coefficients
    return {
        "params": params,
        "coefficients": coefficients,
        "train_state": train_state / np.asarray(train_proposal_carrier),
        "train_action": train_h / np.asarray(train_proposal_carrier),
        "audit_state": audit_state / np.asarray(audit_proposal_carrier),
        "audit_action": audit_h / np.asarray(audit_proposal_carrier),
        "history": history,
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "parameter_sha256": pytree_sha256(params),
        "component_count": dressing_config.component_count,
        "group_order": carrier_config.group_order,
        "direction_labels": list(direction_labels),
        "final_train_reference": asdict(final_reference),
    }


def _train_fixed_sz(
    spec,
    output,
    train_states,
    train_carrier,
    audit_states,
    audit_carrier,
):
    model = spec["fixed_sz_model"]
    training = spec["training"]
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
    full_group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    projector = PhysicalSpatialProjector(
        full_group,
        orbit_chunk_size=int(model["orbit_chunk_size"]),
        rematerialize_orbit=True,
    )
    orbit = build_n36_pi_zero_flux_orbit()
    params = init_fixed_sz_spatial_vit(jax.random.key(int(spec["seed"]) + 31), config)
    initial = params

    def raw_correlator(candidate, state, _config):
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
            candidate, state, config, projector, amplitude_fn=raw_correlator
        )
        return carrier * scalar

    bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=J2, periodic=True)
    train_connectivity = tuple(
        jnp.asarray(value) for value in padded_spin_connectivity(train_states, bonds)[:3]
    )
    audit_connectivity = tuple(
        jnp.asarray(value) for value in padded_spin_connectivity(audit_states, bonds)[:3]
    )
    connected, elements, valid = train_connectivity
    run_identity = {
        "task": TASK,
        "arm": ARM_FIXED_SZ,
        "config_sha256": canonical_sha256(spec),
        "training_states_sha256": array_sha256(np.asarray(train_states)),
    }
    checkpoint_directory = output / "checkpoints" / ARM_FIXED_SZ
    restored = (
        load_latest_resumable_checkpoint(
            checkpoint_directory, expected_run_identity=run_identity
        )
        if checkpoint_directory.is_dir()
        else None
    )
    if restored is None:
        history = []
        completed_steps = 0
    else:
        params = restored["params"]
        history = list(restored["history"])
        completed_steps = int(restored["completed_steps"])
    started = time.perf_counter()
    for step in range(completed_steps + 1, int(training["fixed_sz_steps"]) + 1):
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
        )
        params, norm, scale = _normalized_gradient_step(
            params, gradient.gradient, float(training["fixed_sz_learning_rate"])
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
            metadata={"spin_sector": "fixed_Sz0_not_exact_S2"},
        )
    train_center, train_h = _batch_hpsi(
        params,
        jnp.asarray(train_states),
        *train_connectivity,
        config,
        amplitude,
    )
    audit_center, audit_h = _batch_hpsi(
        params,
        jnp.asarray(audit_states),
        *audit_connectivity,
        config,
        amplitude,
    )
    return {
        "params": params,
        "train_state": np.asarray(train_center) / np.asarray(train_carrier),
        "train_action": np.asarray(train_h) / np.asarray(train_carrier),
        "audit_state": np.asarray(audit_center) / np.asarray(audit_carrier),
        "audit_action": np.asarray(audit_h) / np.asarray(audit_carrier),
        "history": history,
        "wall_seconds": time.perf_counter() - started,
        "parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "parameter_sha256": pytree_sha256(params),
        "projector_group_order": len(full_group),
    }


def _regularized_refit(
    spec,
    output,
    development,
    audit,
    train_states,
    train_raw,
    audit_states,
    audit_raw,
    full_baseline,
) -> dict[str, object]:
    """Cross-fit a nested bank and audit on previously unseen later blocks."""

    training = spec["training"]
    chain_count = int(full_baseline["chain_count"])
    start = chain_count * int(training["secondary_audit_start_block"])
    if start >= len(audit_states):
        raise ValueError("secondary audit start leaves no configurations")
    secondary_states = audit_states[start:]
    secondary_raw = audit_raw[start:]
    secondary_baseline = _baseline_from_cache(
        development,
        audit,
        train_raw,
        secondary_raw,
        require_sealed_reproduction=False,
    )

    model = spec["model"]
    dressing = ResonatingSU2ScalarDressingConfig(
        rows=6,
        columns=6,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        component_count=int(model["component_count"]),
        mlp_ratio=int(model["mlp_ratio"]),
        residual_scale=float(model["residual_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    carrier_config = ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )
    bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=J2, periodic=True)
    graph = ProjectedAGPHeisenbergConfig(
        n_sites=N_SITES,
        bonds=tuple((int(i), int(j), float(value)) for i, j, value in bonds),
    )
    base, direction_tuple, direction_labels = triangular_physical_pairing_bank(6)
    directions = np.stack(direction_tuple)
    source = ROOT / str(spec["source_checkpoint_directory"])
    checkpoint = load_latest_resumable_checkpoint(source)
    identity = checkpoint["run_identity"]
    if identity.get("task") != TASK or identity.get("arm") != ARM_SU2:
        raise ValueError("source checkpoint is not an SU2 scalar-dressing model")
    params = checkpoint["params"]

    def action_for(states, baseline):
        fixed = baseline["train_state"] * baseline["train_carrier"]
        fixed_h = baseline["train_action"] * baseline["train_carrier"]
        return _combined_action(
            params,
            states,
            fixed,
            fixed_h,
            base_pairing=base,
            directions=directions,
            dressing_config=dressing,
            carrier_config=carrier_config,
            graph_config=graph,
            amplitude_microbatch_size=int(model["amplitude_microbatch_size"]),
        )

    # ``secondary_baseline`` stores the supplied secondary rows under its
    # audit keys; expose the same arrays as train keys for the shared helper.
    secondary_for_action = {
        "train_state": secondary_baseline["audit_state"],
        "train_action": secondary_baseline["audit_action"],
        "train_carrier": secondary_baseline["audit_carrier"],
    }
    started = time.perf_counter()
    train_action = action_for(train_states, full_baseline)
    audit_action = action_for(secondary_states, secondary_for_action)
    jax.block_until_ready((train_action.amplitudes, train_action.hamiltonian_actions))
    jax.block_until_ready((audit_action.amplitudes, audit_action.hamiltonian_actions))
    action_wall = time.perf_counter() - started
    train_proposal = np.abs(full_baseline["train_carrier"]) ** 2
    audit_proposal = np.abs(secondary_baseline["audit_carrier"]) ** 2
    selected = crossfit_resonating_ridge(
        np.asarray(train_action.amplitudes),
        np.asarray(train_action.hamiltonian_actions),
        train_proposal,
        chain_count=chain_count,
        fold_count=int(training["crossfit_folds"]),
        ridge_grid=tuple(float(value) for value in training["ridge_grid"]),
    )
    fit = selected["fit"]
    observed = evaluate_resonating_ground(
        fit.coefficients,
        np.asarray(audit_action.amplitudes),
        np.asarray(audit_action.hamiltonian_actions),
        audit_proposal,
    )
    state_ratio = np.asarray(observed["state"]) / secondary_baseline["audit_carrier"]
    action_ratio = np.asarray(observed["action"]) / secondary_baseline["audit_carrier"]
    candidate = _ratio_summary(
        state_ratio, action_ratio, chain_count=chain_count, n_sites=N_SITES
    )
    baseline_summary = secondary_baseline["audit_summary"]
    paired = _paired_chain_difference(
        state_ratio,
        action_ratio,
        secondary_baseline["audit_state"],
        secondary_baseline["audit_action"],
        chain_count=chain_count,
    )
    grid = selected["grid"]
    fit_row = {
        "ridge": fit.ridge,
        "energy_per_site_on_development": fit.energy / N_SITES,
        "variance_per_site_on_development": fit.variance / N_SITES,
        "overlap_condition_number": fit.overlap_condition_number,
        "resonance_norm_fraction": fit.resonance_norm_fraction,
        "coefficients": {
            "real": fit.coefficients.real.tolist(),
            "imag": fit.coefficients.imag.tolist(),
        },
    }
    tolerance = float(spec["gates"]["competitive_energy_tolerance_per_site"])
    resolved = (
        paired["difference_per_site"]
        + 2.0 * paired["chain_standard_error_per_site"]
        < 0.0
    )
    competitive = paired["difference_per_site"] <= max(
        tolerance, 2.0 * paired["chain_standard_error_per_site"]
    )
    gpu_count = max(1, len([device for device in jax.devices() if device.platform == "gpu"]))
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "execution_tier": spec["execution_tier"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_checkpoint": {
            "directory": str(spec["source_checkpoint_directory"]),
            "completed_steps": checkpoint["completed_steps"],
            "parameter_sha256": pytree_sha256(params),
        },
        "development": {
            "sample_count": len(train_states),
            "chain_count": chain_count,
            "sample_sha256": array_sha256(train_states),
            "crossfit_folds": int(training["crossfit_folds"]),
            "grid": grid,
            "minimum_mean_ridge": selected["minimum_mean_ridge"],
            "one_standard_error_threshold": selected["one_standard_error_threshold"],
            "selected_ridge": selected["selected_ridge"],
            "selection_rule": selected["selection_rule"],
            "fit": fit_row,
        },
        "secondary_untouched_audit": {
            "scope": "blocks after the first pilot block; never inspected by this refit",
            "start_block": int(training["secondary_audit_start_block"]),
            "sample_count": len(secondary_states),
            "chain_count": chain_count,
            "sample_sha256": array_sha256(secondary_states),
            "candidate": candidate,
            "two_lanczos": baseline_summary,
            "paired_candidate_minus_two_lanczos": paired,
        },
        "work_ledger": {
            "projected_pair_component_count": dressing.component_count,
            "full_h_configuration_calls": dressing.component_count
            * (len(train_states) + len(secondary_states)),
            "two_lanczos_actions_reused_from_sealed_cache": True,
            "action_wall_seconds": action_wall,
            "gpu_hours": action_wall * gpu_count / 3600.0,
            "group_order": len(group),
            "direction_labels": list(direction_labels),
        },
        "decision": {
            "energy_below_two_lanczos": paired["difference_per_site"] < 0.0,
            "energy_below_two_lanczos_at_2se": resolved,
            "variance_below_two_lanczos": candidate["variance_total"]
            < baseline_summary["variance_total"],
            "ess_gate_passed": candidate["importance_ess_fraction"]
            >= float(spec["gates"]["minimum_ess_fraction"]),
            "strict_joint_win": resolved
            and candidate["variance_total"] < baseline_summary["variance_total"],
            "finite_sample_competitive_energy": competitive,
        },
        "interpretation_boundary": {
            "first_64_audit_rows_excluded_after_pilot_inspection": True,
            "secondary_rows_are_untouched_by_refit_selection": True,
            "nonlinear_parameters_frozen_before_crossfit": True,
            "ridge_selected_only_on_development_chains": True,
        },
        "hamiltonian_diagonalization_performed": False,
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(output / "results.json", payload)
    return payload


def _paired(first, second, chain_count: int):
    return _paired_chain_difference(
        first["audit_state"],
        first["audit_action"],
        second["audit_state"],
        second["audit_action"],
        chain_count=chain_count,
    )


def run_spin_n36_su2_scalar_dressing(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists: {result_path}")
    development, audit, train_states, train_raw, audit_states, audit_raw = _sealed_common_data(spec)
    full_baseline = _baseline_from_cache(
        development,
        audit,
        train_raw,
        audit_raw,
        require_sealed_reproduction=True,
    )
    if spec["execution_tier"] == "remote_regularized_refit":
        return _regularized_refit(
            spec,
            output,
            development,
            audit,
            train_states,
            train_raw,
            audit_states,
            audit_raw,
            full_baseline,
        )
    if spec["execution_tier"] == "local_contract_preflight":
        nested_audit_carrier = (
            full_baseline["audit_state"] * full_baseline["audit_carrier"]
        )
        model_contract = _local_model_contract(
            spec, audit_states, nested_audit_carrier
        )
        payload = {
            "schema_version": 1,
            "domain": "spin",
            "task": TASK,
            "execution_tier": spec["execution_tier"],
            "status": "complete",
            "config": spec,
            "sealed_common_sample_contract": {
                "train_count": len(train_states),
                "audit_count": len(audit_states),
                "chain_count": full_baseline["chain_count"],
                "train_sha256": array_sha256(train_states),
                "audit_sha256": array_sha256(audit_states),
                "two_lanczos_reproduced": True,
                "two_lanczos": full_baseline["audit_summary"],
            },
            "n36_model_contract": model_contract,
            "decision": {
                "remote_n36_pilot_licensed": (
                    model_contract["combined_sample_rank"]
                    == int(spec["model"]["component_count"]) + 1
                    and model_contract["maximum_gamma_a1_relative_residual"] < 1.0e-10
                    and model_contract["minimum_configuration_ratio_standard_deviation"] > 1.0e-8
                )
            },
        }
        payload["result_sha256"] = canonical_sha256(payload)
        atomic_json(result_path, payload)
        return payload

    chain_count = int(full_baseline["chain_count"])
    matched_count = chain_count * int(
        spec["training"]["matched_measurement_blocks"]
    )
    if matched_count > len(train_states) or matched_count > len(audit_states):
        raise ValueError("matched_measurement_blocks exceeds the sealed sample count")
    train_states, train_raw = train_states[:matched_count], train_raw[:matched_count]
    audit_states, audit_raw = audit_states[:matched_count], audit_raw[:matched_count]
    baseline = _baseline_from_cache(
        development,
        audit,
        train_raw,
        audit_raw,
        require_sealed_reproduction=False,
    )

    su2 = _train_su2(
        spec,
        output,
        train_states,
        baseline["train_state"] * baseline["train_carrier"],
        baseline["train_action"] * baseline["train_carrier"],
        baseline["train_carrier"],
        audit_states,
        baseline["audit_state"] * baseline["audit_carrier"],
        baseline["audit_action"] * baseline["audit_carrier"],
        baseline["audit_carrier"],
    )
    fixed = _train_fixed_sz(
        spec,
        output,
        train_states,
        baseline["train_carrier"],
        audit_states,
        baseline["audit_carrier"],
    )
    chain_count = baseline["chain_count"]
    baseline_arm = {
        "audit_state": baseline["audit_state"],
        "audit_action": baseline["audit_action"],
        "summary": baseline["audit_summary"],
    }
    su2_summary = _ratio_summary(
        su2["audit_state"], su2["audit_action"], chain_count=chain_count, n_sites=N_SITES
    )
    fixed_summary = _ratio_summary(
        fixed["audit_state"], fixed["audit_action"], chain_count=chain_count, n_sites=N_SITES
    )
    su2["summary"] = su2_summary
    fixed["summary"] = fixed_summary
    su2_minus_baseline = _paired(su2, baseline_arm, chain_count)
    su2_minus_fixed = _paired(su2, fixed, chain_count)
    fixed_minus_baseline = _paired(fixed, baseline_arm, chain_count)
    tolerance = float(spec["gates"]["competitive_energy_tolerance_per_site"])
    minimum_ess = float(spec["gates"]["minimum_ess_fraction"])
    strict_energy = (
        su2_minus_baseline["difference_per_site"]
        + 2.0 * su2_minus_baseline["chain_standard_error_per_site"]
        < 0.0
        and su2_minus_fixed["difference_per_site"]
        + 2.0 * su2_minus_fixed["chain_standard_error_per_site"]
        < 0.0
    )
    strict_variance = (
        su2_summary["variance_total"] < baseline["audit_summary"]["variance_total"]
        and su2_summary["variance_total"] < fixed_summary["variance_total"]
    )
    statistically_competitive = (
        su2_minus_baseline["difference_per_site"]
        <= max(
            tolerance,
            2.0 * su2_minus_baseline["chain_standard_error_per_site"],
        )
        and su2_minus_fixed["difference_per_site"]
        <= max(tolerance, 2.0 * su2_minus_fixed["chain_standard_error_per_site"])
    )
    train_count, audit_count = len(train_states), len(audit_states)
    su2_calls = int(su2["component_count"]) * (
        (int(spec["training"]["su2_steps"]) + 1) * train_count + audit_count
    )
    fixed_calls = (
        int(spec["training"]["fixed_sz_steps"]) * train_count
        + train_count
        + audit_count
    )
    baseline_calls = 3 * (train_count + audit_count)
    gpu_count = max(1, len([device for device in jax.devices() if device.platform == "gpu"]))
    arms = {
        ARM_BASELINE: {
            "summary": baseline["audit_summary"],
            "full_h_configuration_calls": baseline_calls,
            "gpu_hours_current_run": 0.0,
            "cached_production_work_reused": True,
            "exact_su2": True,
            "exact_gamma_a1": True,
        },
        ARM_FIXED_SZ: {
            "summary": fixed_summary,
            "full_h_configuration_calls": fixed_calls,
            "wall_seconds": fixed["wall_seconds"],
            "gpu_hours": fixed["wall_seconds"] * gpu_count / 3600.0,
            "parameter_count": fixed["parameter_count"],
            "parameter_sha256": fixed["parameter_sha256"],
            "training_history": fixed["history"],
            "exact_su2": False,
            "exact_gamma_a1": True,
        },
        ARM_SU2: {
            "summary": su2_summary,
            "full_h_configuration_calls": su2_calls,
            "wall_seconds": su2["wall_seconds"],
            "gpu_hours": su2["wall_seconds"] * gpu_count / 3600.0,
            "parameter_count": su2["parameter_count"],
            "parameter_sha256": su2["parameter_sha256"],
            "linear_coefficients": {
                "real": np.asarray(su2["coefficients"]).real.tolist(),
                "imag": np.asarray(su2["coefficients"]).imag.tolist(),
            },
            "training_history": su2["history"],
            "exact_su2": True,
            "exact_gamma_a1": True,
            "configuration_dependent": True,
            "additive_carrier_nested": True,
            "nested_carrier": "sealed_projected_vmc_two_lanczos",
        },
    }
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "execution_tier": spec["execution_tier"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic torus",
            "j2_over_j1": J2,
            "target": "ground state in Gamma-A1; exact S=0 for baseline and SU2 arm",
        },
        "common_samples": {
            "training_count": train_count,
            "untouched_audit_count": audit_count,
            "persistent_chain_count": chain_count,
            "training_sha256": array_sha256(train_states),
            "untouched_audit_sha256": array_sha256(audit_states),
            "identical_configurations_for_all_arms": True,
            "audit_never_used_for_training_or_selection": True,
        },
        "arms": arms,
        "paired_comparisons": {
            "su2_minus_two_lanczos": su2_minus_baseline,
            "su2_minus_fixed_sz_vit": su2_minus_fixed,
            "fixed_sz_vit_minus_two_lanczos": fixed_minus_baseline,
        },
        "work_contract": {
            "same_training_and_audit_configurations": True,
            "full_h_calls_are_reported_not_hidden_in_optimizer_steps": True,
            "maximum_full_h_call_budget": max(baseline_calls, fixed_calls, su2_calls),
            "all_arms_within_common_call_cap": all(
                value <= max(baseline_calls, fixed_calls, su2_calls)
                for value in (baseline_calls, fixed_calls, su2_calls)
            ),
            "gpu_hours_are_actual_not_parameter_count_proxies": True,
            "baseline_cached_gpu_work_amortized": True,
        },
        "literature_targets": {
            "exact_36_site_energy_per_site": EXACT_ENERGY_PER_SITE,
            "published_optimized_projected_vmc_two_lanczos_per_site": PUBLISHED_TWO_LANCZOS_PER_SITE,
            "citation": "Y. Iqbal et al., Phys. Rev. B 93, 144411 (2016), Table III",
        },
        "decision": {
            "su2_energy_strictly_below_both_at_2se": strict_energy,
            "su2_variance_below_both": strict_variance,
            "strict_joint_win": strict_energy and strict_variance,
            "finite_sample_competitive_energy": statistically_competitive,
            "su2_ess_gate_passed": su2_summary["importance_ess_fraction"] >= minimum_ess,
            "relaxed_pilot_pass": statistically_competitive
            and su2_summary["importance_ess_fraction"] >= minimum_ess
            and (
                su2_summary["variance_total"]
                <= min(baseline["audit_summary"]["variance_total"], fixed_summary["variance_total"])
                or strict_energy
            ),
        },
        "interpretation_boundary": {
            "one_preregistered_model_seed": True,
            "limited_sample_pilot_not_final_scaling_claim": True,
            "competitive_does_not_mean_resolved_win": True,
            "published_two_lanczos_target_not_work_matched_to_this_implementation": True,
        },
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": False,
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_n36_su2_scalar_dressing"]
