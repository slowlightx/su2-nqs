"""Physical-carrier three-arm pilot on the triangular 6x6 torus.

The experiment deliberately replaces the historical six-scalar ViT tangent
with three actual wavefunctions:

1. a from-scratch fixed-Sz Spatial-ViT;
2. the same Spatial-ViT used as a positive correlator around the exact
   projected pi/0-flux carrier; and
3. an exact-SU(2) reduced-multiplicity spatial-attention state first distilled
   toward that carrier and then energy-trained.

All energy updates use configurations drawn from the immutable projected
parton state and self-normalized importance weights.  Consequently an arm is
interpretable only when its reported importance ESS passes the preregistered
gate.  Distillation uses no Hamiltonian calls and is recorded separately from
the matched VMC ledger.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
    _batch_hpsi,
    _load_reference,
    _normalized_gradient_step,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import (
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
    init_su2_multiplicity_spatial_attention,
    su2_multiplicity_spatial_attention_amplitudes,
)
from su2_nqs.vmc.common_sample_vmc import (
    common_sample_vmc_value_and_grad,
    paired_common_sample_energy_difference,
)
from su2_nqs.vmc.fixed_wavefunction_mcmc import (
    sample_fixed_wavefunction_exchange,
)
from su2_nqs.vmc.resumable_checkpoint import (
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_physical_three_arm"
ARM_IDS = (
    "from_scratch_fixed_sz_spatial_vit",
    "pi_flux_spatial_vit_correlator",
    "pi_flux_distilled_su2_multiplicity_spatial",
)
EXACT_ENERGY_PER_SITE = -0.515564
GCNN_ENERGY_PER_SITE = -0.515386


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 physical three-arm identity")
    for name in ("reference", "sampling", "model", "distillation", "training", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    sampling = spec["sampling"]
    for name in (
        "replicates",
        "training_count",
        "audit_count",
        "burn_sweeps",
        "decorrelation_sweeps",
    ):
        if int(sampling.get(name, 0)) < 1:
            raise ValueError(f"sampling.{name} must be positive")
    if int(sampling["replicates"]) < 2:
        raise ValueError("the comparison requires at least two replicates")
    for section in ("distillation", "training"):
        if int(spec[section].get("steps", 0)) < 1:
            raise ValueError(f"{section}.steps must be positive")
        if float(spec[section].get("learning_rate", 0.0)) <= 0.0:
            raise ValueError(f"{section}.learning_rate must be positive")


def _parton_samples(
    amplitude_fn,
    *,
    count: int,
    burn_sweeps: int,
    decorrelation_sweeps: int,
    seed: int,
) -> np.ndarray:
    sampled = sample_fixed_wavefunction_exchange(
        amplitude_fn,
        n_sites=36,
        n_up=18,
        chain_count=int(count),
        burn_sweeps=int(burn_sweeps),
        measurement_blocks=1,
        decorrelation_sweeps=int(decorrelation_sweeps),
        seed=int(seed),
    )
    return np.asarray(sampled.configurations, dtype=np.int32)


def _teacher_born_fidelity(model, teacher) -> jax.Array:
    """Estimate normalized overlap from samples distributed as ``|teacher|^2``."""

    model = jnp.asarray(model, dtype=jnp.complex128)
    teacher = jnp.asarray(teacher, dtype=jnp.complex128)
    ratio = model / teacher
    rms = jax.lax.stop_gradient(
        jnp.sqrt(jnp.maximum(jnp.mean(jnp.abs(ratio) ** 2), 1.0e-300))
    )
    scaled = ratio / rms
    return jnp.abs(jnp.mean(scaled)) ** 2 / jnp.maximum(
        jnp.mean(jnp.abs(scaled) ** 2), jnp.asarray(1.0e-300)
    )


def _distill(
    params,
    amplitude_fn,
    train_states,
    train_teacher,
    audit_states,
    audit_teacher,
    *,
    steps: int,
    learning_rate: float,
):
    started = time.perf_counter()
    train = jnp.asarray(train_states, dtype=jnp.int32)
    audit = jnp.asarray(audit_states, dtype=jnp.int32)
    teacher = jnp.asarray(train_teacher, dtype=jnp.complex128)

    def loss(candidate):
        predicted = jax.vmap(lambda row: amplitude_fn(candidate, row))(train)
        fidelity = _teacher_born_fidelity(predicted, teacher)
        return -jnp.log(jnp.maximum(fidelity, jnp.asarray(1.0e-15)))

    history = []
    for step in range(1, int(steps) + 1):
        value, gradient = jax.value_and_grad(loss)(params)
        norm = float(
            np.sqrt(
                sum(
                    float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
                    for leaf in jax.tree_util.tree_leaves(gradient)
                )
            )
        )
        base_scale = 0.0 if norm <= 1.0e-12 else float(learning_rate) / norm
        accepted = False
        scale = 0.0
        candidate_value = float(np.asarray(value))
        for backtrack in range(8):
            trial_scale = base_scale * 0.5**backtrack
            candidate = jax.tree_util.tree_map(
                lambda parameter, direction: parameter - trial_scale * direction,
                params,
                gradient,
            )
            trial_value = float(np.asarray(loss(candidate)))
            if np.isfinite(trial_value) and trial_value <= float(np.asarray(value)):
                params = candidate
                candidate_value = trial_value
                scale = trial_scale
                accepted = True
                break
        history.append(
            {
                "step": step,
                "negative_log_sample_overlap_before": float(np.asarray(value)),
                "negative_log_sample_overlap_after": candidate_value,
                "gradient_norm": norm,
                "applied_gradient_scale": scale,
                "backtracking_accepted": accepted,
            }
        )
    train_model = jax.vmap(lambda row: amplitude_fn(params, row))(train)
    audit_model = jax.vmap(lambda row: amplitude_fn(params, row))(audit)
    return params, history, {
        "train_sample_fidelity": float(
            np.asarray(_teacher_born_fidelity(train_model, train_teacher))
        ),
        "heldout_sample_fidelity": float(
            np.asarray(_teacher_born_fidelity(audit_model, audit_teacher))
        ),
        "hamiltonian_configuration_calls": 0,
        "objective": "parton-Born-sample normalized overlap",
        "wall_seconds": time.perf_counter() - started,
    }


def _audit(params, amplitude_fn, config, states, reference, connectivity):
    connected, elements, valid = connectivity
    centers, actions = _batch_hpsi(
        params,
        jnp.asarray(states),
        connected,
        elements,
        valid,
        config,
        amplitude_fn,
    )
    center_host = np.asarray(centers, dtype=np.complex128)
    action_host = np.asarray(actions, dtype=np.complex128)
    reference_host = np.asarray(reference, dtype=np.complex128)
    local = action_host / center_host
    raw = np.abs(center_host / reference_host) ** 2
    weights = raw / np.sum(raw)
    energy = complex(np.sum(weights * local))
    variance = float(np.sum(weights * np.abs(local - energy) ** 2).real)
    ess = float(1.0 / np.sum(weights**2))
    return {
        "energy_total": float(energy.real),
        "energy_per_site": float(energy.real / 36.0),
        "variance_per_site": variance / 36.0,
        "importance_ess_fraction": ess / len(states),
        "maximum_normalized_weight": float(np.max(weights)),
        "maximum_abs_imaginary_local_energy": float(np.max(np.abs(local.imag))),
    }, center_host, action_host


def _train_arm(
    *,
    arm_id,
    params,
    amplitude_fn,
    config,
    train_states,
    train_reference,
    train_connectivity,
    audit_states,
    audit_reference,
    audit_connectivity,
    steps,
    learning_rate,
    checkpoint_directory,
    run_identity,
):
    initial, initial_centers, initial_actions = _audit(
        params,
        amplitude_fn,
        config,
        audit_states,
        audit_reference,
        audit_connectivity,
    )
    train_connected, train_elements, train_valid = train_connectivity
    history = []
    started = time.perf_counter()
    for step in range(1, int(steps) + 1):
        gradient = common_sample_vmc_value_and_grad(
            params,
            jnp.asarray(train_states),
            jnp.asarray(train_reference),
            hpsi_fn=lambda candidate, states: _batch_hpsi(
                candidate,
                states,
                train_connected,
                train_elements,
                train_valid,
                config,
                amplitude_fn,
            ),
            center_fn=lambda candidate, states: jax.vmap(
                lambda row: amplitude_fn(candidate, row)
            )(states),
        )
        params, norm, scale = _normalized_gradient_step(
            params, gradient.gradient, float(learning_rate)
        )
        history.append(
            {
                "step": step,
                "training_energy_total": gradient.energy,
                "training_variance": gradient.variance,
                "training_importance_ess_fraction": gradient.ess_fraction,
                "gradient_norm": norm,
                "applied_gradient_scale": scale,
            }
        )
        save_resumable_checkpoint_generation(
            checkpoint_directory,
            params,
            optimizer_state={"kind": "normalized_gradient_descent"},
            completed_steps=step,
            next_sample_index=step * len(train_states),
            run_identity={**run_identity, "arm_id": arm_id},
            history=history,
            metadata={"proposal": "projected_pi_zero_flux_a1"},
        )
    final, centers, actions = _audit(
        params,
        amplitude_fn,
        config,
        audit_states,
        audit_reference,
        audit_connectivity,
    )
    paired_change = paired_common_sample_energy_difference(
        centers,
        actions,
        initial_centers,
        initial_actions,
        audit_reference,
    )
    paired_change_row = asdict(paired_change)
    paired_change_row["difference_per_site"] = paired_change.difference / 36.0
    paired_change_row["standard_error_per_site"] = (
        paired_change.standard_error / 36.0
    )
    return params, centers, actions, {
        "arm_id": arm_id,
        "trainable_parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "initial_audit": initial,
        "training_history": history,
        "final_audit": final,
        "paired_final_minus_initial": paired_change_row,
        "wall_seconds": time.perf_counter() - started,
        "work_ledger": {
            "training_steps": int(steps),
            "shared_samples_per_step": len(train_states),
            "full_h_configuration_calls_training": int(steps) * len(train_states),
            "full_h_configuration_calls_audit": 2 * len(audit_states),
            "full_h_configuration_calls_total": int(steps) * len(train_states)
            + 2 * len(audit_states),
        },
    }


def run_spin_n36_physical_three_arm(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    basin_spec, base_config, bonds, route_ledger, _burn, final = _load_reference(spec)
    frozen = final["params"]
    orbit = build_n36_pi_zero_flux_orbit()
    sampling = spec["sampling"]
    model = spec["model"]
    vit_config = FixedSzSpatialViTConfig(
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
        triangular_hnf_translation_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(model["translation_orbit_chunk_size"]),
        rematerialize_orbit=True,
    )
    j1_bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    j2_bonds = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    features = multiplicity_spatial_features(
        base_config.spatial.backbone.tree,
        base_config.spatial.route_orders,
        j1_bonds,
        j2_bonds,
    )
    su2_config = SU2MultiplicitySpatialAttentionConfig(
        base=base_config,
        features=features,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )

    def parton_host(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states, orbit=orbit, microbatch_size=64
        ).amplitudes

    replicates = []
    for replicate_index in range(int(sampling["replicates"])):
        replicate_seed = int(spec["seed"]) + 1000 * replicate_index
        train_states = _parton_samples(
            parton_host,
            count=int(sampling["training_count"]),
            burn_sweeps=int(sampling["burn_sweeps"]),
            decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
            seed=replicate_seed + 1,
        )
        audit_states = _parton_samples(
            parton_host,
            count=int(sampling["audit_count"]),
            burn_sweeps=int(sampling["burn_sweeps"]),
            decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
            seed=replicate_seed + 2,
        )
        train_reference = parton_host(train_states)
        audit_reference = parton_host(audit_states)
        train_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(train_states, bonds)[:3]
        )
        audit_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(audit_states, bonds)[:3]
        )

        initial_vit = init_fixed_sz_spatial_vit(
            jax.random.PRNGKey(replicate_seed + 11), vit_config
        )

        def raw_vit(candidate, state, _config):
            value = fixed_sz_spatial_vit_log_amplitude(candidate, state, vit_config)
            return jnp.exp(value)

        def pure_vit(candidate, state):
            return spatial_projected_amplitude(
                candidate,
                state,
                vit_config,
                projector,
                amplitude_fn=raw_vit,
            )

        def raw_correlator(candidate, state, _config):
            current = fixed_sz_spatial_vit_log_amplitude(candidate, state, vit_config)
            reference = jax.lax.stop_gradient(
                fixed_sz_spatial_vit_log_amplitude(initial_vit, state, vit_config)
            )
            return jnp.exp(jnp.real(current - reference))

        def anchored_vit(candidate, state):
            carrier = projected_pi_zero_flux_a1_amplitudes_jax(
                state[None, :], orbit=orbit
            )[0]
            correlator = spatial_projected_amplitude(
                candidate,
                state,
                vit_config,
                projector,
                amplitude_fn=raw_correlator,
            )
            return carrier * correlator

        attention = init_su2_multiplicity_spatial_attention(
            jax.random.PRNGKey(replicate_seed + 17), frozen, su2_config
        )
        initial_su2 = {
            "backbone": frozen["backbone"],
            "attention": attention,
        }

        def su2_amplitude(candidate, state):
            def raw(local_candidate, row, _config):
                base_params = {
                    **frozen,
                    "backbone": local_candidate["backbone"],
                }
                return su2_multiplicity_spatial_attention_amplitudes(
                    local_candidate["attention"],
                    base_params,
                    row[None, :],
                    su2_config,
                )[0]

            return spatial_projected_amplitude(
                candidate,
                state,
                su2_config,
                projector,
                amplitude_fn=raw,
            )

        distilled_su2, distill_history, distill_audit = _distill(
            initial_su2,
            su2_amplitude,
            train_states,
            train_reference,
            audit_states,
            audit_reference,
            steps=int(spec["distillation"]["steps"]),
            learning_rate=float(spec["distillation"]["learning_rate"]),
        )
        runtime = {
            ARM_IDS[0]: (initial_vit, pure_vit, vit_config),
            ARM_IDS[1]: (initial_vit, anchored_vit, vit_config),
            ARM_IDS[2]: (distilled_su2, su2_amplitude, su2_config),
        }
        run_identity = {
            "task": TASK,
            "config_sha256": canonical_sha256(spec),
            "replicate_index": replicate_index,
            "reference_checkpoint_sha256": pytree_sha256(frozen),
            "train_states_sha256": canonical_sha256(train_states.tolist()),
            "audit_states_sha256": canonical_sha256(audit_states.tolist()),
        }
        arms = []
        evaluated = {}
        for arm_id in ARM_IDS:
            params, amplitude_fn, local_config = runtime[arm_id]
            _params, centers, actions, row = _train_arm(
                arm_id=arm_id,
                params=params,
                amplitude_fn=amplitude_fn,
                config=local_config,
                train_states=train_states,
                train_reference=train_reference,
                train_connectivity=train_connectivity,
                audit_states=audit_states,
                audit_reference=audit_reference,
                audit_connectivity=audit_connectivity,
                steps=int(spec["training"]["steps"]),
                learning_rate=float(spec["training"]["learning_rate"]),
                checkpoint_directory=output
                / "checkpoints"
                / f"replicate_{replicate_index}"
                / arm_id,
                run_identity=run_identity,
            )
            if arm_id == ARM_IDS[2]:
                row["carrier_distillation"] = {
                    "history": distill_history,
                    **distill_audit,
                }
            arms.append(row)
            evaluated[arm_id] = (centers, actions)
            print(
                f"replicate={replicate_index} {arm_id} "
                f"E/N={row['final_audit']['energy_per_site']:.8f} "
                f"ESS={row['final_audit']['importance_ess_fraction']:.3f}",
                flush=True,
            )
        paired = {}
        for first_index in range(len(ARM_IDS)):
            for second_index in range(first_index + 1, len(ARM_IDS)):
                first = ARM_IDS[first_index]
                second = ARM_IDS[second_index]
                comparison = paired_common_sample_energy_difference(
                    *evaluated[first], *evaluated[second], audit_reference
                )
                row = asdict(comparison)
                row["difference_per_site"] = comparison.difference / 36.0
                row["standard_error_per_site"] = comparison.standard_error / 36.0
                paired[f"{first}_minus_{second}"] = row
        replicate = {
            "replicate_index": replicate_index,
            "train_sample_count": len(train_states),
            "audit_sample_count": len(audit_states),
            "arms": arms,
            "paired_comparisons": paired,
        }
        replicates.append(replicate)
        atomic_json(output / f"replicate_{replicate_index}.json", replicate)

    aggregate = {}
    for arm_id in ARM_IDS:
        values = np.asarray(
            [
                next(row for row in rep["arms"] if row["arm_id"] == arm_id)[
                    "final_audit"
                ]["energy_per_site"]
                for rep in replicates
            ]
        )
        minimum_arm_ess = min(
            next(row for row in rep["arms"] if row["arm_id"] == arm_id)[
                "final_audit"
            ]["importance_ess_fraction"]
            for rep in replicates
        )
        paired_updates = np.asarray(
            [
                next(row for row in rep["arms"] if row["arm_id"] == arm_id)[
                    "paired_final_minus_initial"
                ]["difference_per_site"]
                for rep in replicates
            ]
        )
        aggregate[arm_id] = {
            "mean_energy_per_site": float(np.mean(values)),
            "replicate_standard_error_per_site": float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            ),
            "gap_above_exact_per_site": float(np.mean(values) - EXACT_ENERGY_PER_SITE),
            "individual_energy_per_site": values.tolist(),
            "minimum_final_importance_ess_fraction": minimum_arm_ess,
            "ess_gate_passed": minimum_arm_ess
            >= float(spec["gates"]["importance_ess_fraction_minimum"]),
            "paired_final_minus_initial_mean_per_site": float(
                np.mean(paired_updates)
            ),
            "paired_final_minus_initial_replicate_standard_error_per_site": float(
                np.std(paired_updates, ddof=1) / np.sqrt(len(paired_updates))
            ),
            "paired_final_minus_initial_negative_replicates": int(
                np.count_nonzero(paired_updates < 0.0)
            ),
        }
    minimum_ess = min(
        row["final_audit"]["importance_ess_fraction"]
        for rep in replicates
        for row in rep["arms"]
    )
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic torus",
            "j2_over_j1": 0.125,
            "proposal": "exact S=0, Gamma-A1 projected pi/0-flux Born state",
        },
        "arm_contract": {
            ARM_IDS[0]: "independent fixed-Sz translation-projected Spatial-ViT",
            ARM_IDS[1]: "projected pi/0-flux times positive translation-scalar Spatial-ViT correlator",
            ARM_IDS[2]: "exact-SU2 translation-projected reduced multiplicity spatial attention, parton distilled",
            "historical_six_scalar_residual_used": False,
            "same_parton_proposal_samples_and_full_h_calls": True,
        },
        "literature_targets": {
            "exact_energy_per_site": EXACT_ENERGY_PER_SITE,
            "raw_space_group_gcnn_energy_per_site": GCNN_ENERGY_PER_SITE,
            "projected_pi_flux_energy_per_site": -0.5081858380989991,
        },
        "reference": {
            "checkpoint_sha256": pytree_sha256(frozen),
            "route_ledger": route_ledger,
            "no_ed_repeated": True,
        },
        "replicates": replicates,
        "aggregate": aggregate,
        "gates": {
            "minimum_final_importance_ess_fraction": minimum_ess,
            "importance_ess_fraction_minimum": float(
                spec["gates"]["importance_ess_fraction_minimum"]
            ),
            "all_arms_ess_gate_passed": minimum_ess
            >= float(spec["gates"]["importance_ess_fraction_minimum"]),
        },
        "interpretation_boundary": {
            "local_short_horizon_pilot": True,
            "distillation_is_unmatched_forward_only_pretraining": True,
            "absolute_energy_requires_ess_gate": True,
            "not_a_scaling_or_phase_claim": True,
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
    atomic_json(result_path, payload)
    return payload


__all__ = ["ARM_IDS", "TASK", "run_spin_n36_physical_three_arm"]
