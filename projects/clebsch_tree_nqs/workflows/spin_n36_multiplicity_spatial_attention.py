"""Repeated paired N=36 audit of SU(2)-scalar multiplicity attention."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
    _batch_hpsi,
    _known_mixture_samples,
    _load_reference,
    _matched_residual_control,
    _nematic_diagonal_series,
    _normalized_gradient_step,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
    scramble_multiplicity_spatial_features,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.fixed_sz_vit_adapter import (
    FixedSzViTAdapterConfig,
    fixed_sz_vit_adapter_amplitude,
    init_fixed_sz_vit_adapter,
    seeded_fixed_sz_vit_adapter_directions,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
    init_su2_multiplicity_spatial_attention,
    su2_multiplicity_spatial_attention_amplitudes,
    su2_multiplicity_spatial_attention_parameter_count,
)
from su2_nqs.models.su2_route_vit import su2_route_vit_amplitudes
from su2_nqs.vmc.common_sample_vmc import (
    common_sample_vmc_value_and_grad,
    paired_common_sample_energy_difference,
)
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint_generation
from su2_nqs.vmc.spin_physical import padded_spin_connectivity


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_multiplicity_spatial_attention"
ARM_IDS = (
    "su2_multiplicity_spatial",
    "su2_scrambled_geometry",
    "fixed_sz_spatial_vit",
)
EXACT_ENERGY_PER_SITE = -0.515564
GCNN_ENERGY_PER_SITE = -0.515386
GCNN_UNCERTAINTY = 0.000009


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid multiplicity-spatial attention identity")
    for name in ("reference", "common_sample", "model", "training", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    common = spec["common_sample"]
    for name in ("replicates", "calibration_count", "training_count", "audit_count"):
        if int(common.get(name, 0)) < 1:
            raise ValueError(f"common_sample.{name} must be positive")
    if int(common["replicates"]) < 2:
        raise ValueError("paired validation requires multiple independent replicates")
    probabilities = (
        float(common.get("reference_dimer_probability", 0.0)),
        float(common.get("exploratory_dimer_probability", 0.0)),
    )
    if min(probabilities) <= 0.0 or sum(probabilities) >= 1.0:
        raise ValueError("common-sample bridge probabilities are invalid")
    if int(spec["training"].get("steps", 0)) < 1:
        raise ValueError("training.steps must be positive")
    for name in ("train_reduced_backbone", "include_fixed_sz_control"):
        if name in spec["training"] and not isinstance(
            spec["training"][name], bool
        ):
            raise TypeError(f"training.{name} must be boolean")


def _summary(centers, actions, reference, states) -> dict[str, float]:
    centers = np.asarray(centers, dtype=np.complex128)
    actions = np.asarray(actions, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    local = actions / centers
    raw = np.abs(centers / reference) ** 2
    weights = raw / np.sum(raw)
    mean = complex(np.sum(weights * local))
    variance = float(np.sum(weights * np.abs(local - mean) ** 2).real)
    ess = float(1.0 / np.sum(weights**2))
    q1, q2 = _nematic_diagonal_series(states)
    return {
        "energy_total": float(mean.real),
        "energy_per_site": float(mean.real / states.shape[1]),
        "variance_per_site": variance / states.shape[1],
        "importance_ess_fraction": ess / len(states),
        "maximum_imaginary_local_energy": float(np.max(np.abs(local.imag))),
        "diagonal_nematic_q_squared": float(np.sum(weights * (q1 * q1 + q2 * q2))),
    }


def _train(
    *,
    arm_id,
    params,
    amplitude_fn,
    model_config,
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
    train_connected, train_elements, train_valid = train_connectivity
    audit_connected, audit_elements, audit_valid = audit_connectivity
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
                model_config,
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
            metadata={"common_sample_training_count": len(train_states)},
        )
    centers, actions = _batch_hpsi(
        params,
        jnp.asarray(audit_states),
        audit_connected,
        audit_elements,
        audit_valid,
        model_config,
        amplitude_fn,
    )
    return params, np.asarray(centers), np.asarray(actions), {
        "arm_id": arm_id,
        "trainable_parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "training_history": history,
        "final_audit": _summary(centers, actions, audit_reference, audit_states),
        "wall_seconds": time.perf_counter() - started,
        "work_ledger": {
            "training_steps": int(steps),
            "shared_samples_per_step": len(train_states),
            "full_h_configuration_calls_training": int(steps) * len(train_states),
            "full_h_configuration_calls_audit": len(audit_states),
            "full_h_configuration_calls_total": int(steps) * len(train_states)
            + len(audit_states),
        },
    }


def _aggregate(replicates: list[dict[str, object]]) -> dict[str, object]:
    pair_names = tuple(replicates[0]["paired_comparisons"])
    result = {}
    for name in pair_names:
        rows = [replicate["paired_comparisons"][name] for replicate in replicates]
        values = np.asarray(
            [row["difference_per_site"] for row in rows], dtype=np.float64
        )
        result[name] = {
            "mean_difference_per_site": float(np.mean(values)),
            "replicate_standard_error_per_site": float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            ),
            "negative_replicate_count": int(np.count_nonzero(values < 0.0)),
            "replicate_count": len(values),
            "individual_differences_per_site": values.tolist(),
            "individual_paired_standard_errors_per_site": [
                row["standard_error_per_site"] for row in rows
            ],
        }
    return result


def run_spin_n36_multiplicity_spatial_attention(
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
    j1_bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    j2_bonds = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    features = multiplicity_spatial_features(
        base_config.spatial.backbone.tree,
        base_config.spatial.route_orders,
        j1_bonds,
        j2_bonds,
    )
    scrambled = scramble_multiplicity_spatial_features(
        features, seed=int(spec["seed"]) + 19
    )
    model = spec["model"]
    spatial_config = SU2MultiplicitySpatialAttentionConfig(
        base=base_config,
        features=features,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    scrambled_config = SU2MultiplicitySpatialAttentionConfig(
        base=base_config,
        features=scrambled,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    attention_seed = jax.random.PRNGKey(int(spec["seed"]) + 23)
    initial_spatial_attention = init_su2_multiplicity_spatial_attention(
        attention_seed, frozen, spatial_config
    )
    initial_scrambled_attention = init_su2_multiplicity_spatial_attention(
        attention_seed, frozen, scrambled_config
    )
    projector = PhysicalSpatialProjector(
        triangular_hnf_translation_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(basin_spec["model"]["translation_orbit_chunk_size"]),
        rematerialize_orbit=True,
    )

    def base_raw(candidate, state, _config):
        return su2_route_vit_amplitudes(candidate, state[None, :], base_config)[0]

    def base_amplitude(state):
        return spatial_projected_amplitude(
            frozen, state, base_config, projector, amplitude_fn=base_raw
        )

    def attention_amplitude(
        candidate, state, *, local_config, base_params=frozen
    ):
        def raw(local_candidate, row, _config):
            return su2_multiplicity_spatial_attention_amplitudes(
                local_candidate, base_params, row[None, :], local_config
            )[0]

        return spatial_projected_amplitude(
            candidate, state, local_config, projector, amplitude_fn=raw
        )

    fixed_base = FixedSzSpatialViTConfig(
        geometry="triangular",
        rows=6,
        columns=6,
        patch_rows=2,
        patch_columns=2,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=1,
        mlp_ratio=2,
        target_m2=0,
        initialization_scale=float(model["initialization_scale"]),
        use_distance_envelope=True,
    )
    fixed_frozen = init_fixed_sz_spatial_vit(
        jax.random.PRNGKey(int(spec["seed"]) + 29), fixed_base
    )
    fixed_frozen = {
        **fixed_frozen,
        "log_cosh_head": {
            **fixed_frozen["log_cosh_head"],
            "weight_imag": jnp.zeros_like(fixed_frozen["log_cosh_head"]["weight_imag"]),
            "bias_imag": jnp.zeros_like(fixed_frozen["log_cosh_head"]["bias_imag"]),
        },
    }
    fixed_config = FixedSzViTAdapterConfig(
        base=fixed_base,
        directions=seeded_fixed_sz_vit_adapter_directions(
            int(model["embedding_dim"]),
            6,
            seed=int(spec["seed"]) + 31,
            real_only=True,
        ),
        labels=tuple(f"spatial-{index}" for index in range(6)),
        adapter_scale=float(model["modulation_scale"]),
    )
    common = spec["common_sample"]
    training = spec["training"]
    joint_training = bool(training.get("train_reduced_backbone", False))
    include_fixed = bool(training.get("include_fixed_sz_control", True))
    arm_ids = ARM_IDS if include_fixed else ARM_IDS[:2]
    replicates = []
    for replicate_index in range(int(common["replicates"])):
        seed = int(spec["seed"]) + 1000 * (replicate_index + 1)
        calibration_states, calibration_density, calibration_ledger = _known_mixture_samples(
            seed=seed + 1,
            count=int(common["calibration_count"]),
            reference_dimer_probability=float(common["reference_dimer_probability"]),
            exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
        )
        train_states, train_density, train_ledger = _known_mixture_samples(
            seed=seed + 2,
            count=int(common["training_count"]),
            reference_dimer_probability=float(common["reference_dimer_probability"]),
            exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
        )
        audit_states, audit_density, audit_ledger = _known_mixture_samples(
            seed=seed + 3,
            count=int(common["audit_count"]),
            reference_dimer_probability=float(common["reference_dimer_probability"]),
            exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
        )
        train_reference = np.sqrt(train_density).astype(np.complex128)
        audit_reference = np.sqrt(audit_density).astype(np.complex128)
        train_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(train_states, bonds)[:3]
        )
        audit_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(audit_states, bonds)[:3]
        )
        audit_connected, audit_elements, audit_valid = audit_connectivity
        base_centers, base_actions = _batch_hpsi(
            {},
            jnp.asarray(audit_states),
            audit_connected,
            audit_elements,
            audit_valid,
            base_config,
            lambda _candidate, state: base_amplitude(state),
        )
        fixed_zero = init_fixed_sz_vit_adapter(fixed_config)

        def fixed_raw(candidate, state, _config):
            return fixed_sz_vit_adapter_amplitude(
                candidate, fixed_frozen, state, fixed_config
            )

        def fixed_projected(candidate, state):
            return spatial_projected_amplitude(
                candidate,
                state,
                fixed_config,
                projector,
                amplitude_fn=fixed_raw,
            )

        fixed_registration = None
        if include_fixed:
            fixed_zero, fixed_amplitude, fixed_registration = _matched_residual_control(
                zero_params=fixed_zero,
                vit_amplitude_fn=fixed_projected,
                base_amplitude_fn=base_amplitude,
                calibration_states=calibration_states,
                proposal_density=calibration_density,
            )
        if joint_training:
            initial_spatial = {
                "backbone": frozen["backbone"],
                "attention": initial_spatial_attention,
            }
            initial_scrambled = {
                "backbone": frozen["backbone"],
                "attention": initial_scrambled_attention,
            }

            def joint_attention(candidate, state, *, local_config):
                base_params = {**frozen, "backbone": candidate["backbone"]}
                return attention_amplitude(
                    candidate["attention"],
                    state,
                    local_config=local_config,
                    base_params=base_params,
                )

            runtime = {
                "su2_multiplicity_spatial": (
                    initial_spatial,
                    lambda candidate, state: joint_attention(
                        candidate, state, local_config=spatial_config
                    ),
                    spatial_config,
                ),
                "su2_scrambled_geometry": (
                    initial_scrambled,
                    lambda candidate, state: joint_attention(
                        candidate, state, local_config=scrambled_config
                    ),
                    scrambled_config,
                ),
            }
        else:
            runtime = {
                "su2_multiplicity_spatial": (
                    initial_spatial_attention,
                    lambda candidate, state: attention_amplitude(
                        candidate, state, local_config=spatial_config
                    ),
                    spatial_config,
                ),
                "su2_scrambled_geometry": (
                    initial_scrambled_attention,
                    lambda candidate, state: attention_amplitude(
                        candidate, state, local_config=scrambled_config
                    ),
                    scrambled_config,
                ),
            }
        if include_fixed:
            runtime["fixed_sz_spatial_vit"] = (
                fixed_zero,
                fixed_amplitude,
                fixed_config,
            )
        run_identity = {
            "task": TASK,
            "workflow_revision": (
                "joint_backbone_node_reduced_attention_paired_v2"
                if joint_training
                else "node_reduced_attention_paired_v1"
            ),
            "config_sha256": canonical_sha256(spec),
            "reference_params_sha256": pytree_sha256(frozen),
            "replicate_index": replicate_index,
            "training_states_sha256": canonical_sha256(train_states.tolist()),
            "audit_states_sha256": canonical_sha256(audit_states.tolist()),
        }
        arm_rows = []
        evaluated = {
            "common_su2_base": (
                np.asarray(base_centers), np.asarray(base_actions)
            )
        }
        for arm_id in arm_ids:
            params, amplitude_fn, local_config = runtime[arm_id]
            final_params, centers, actions, row = _train(
                arm_id=arm_id,
                params=params,
                amplitude_fn=amplitude_fn,
                model_config=local_config,
                train_states=train_states,
                train_reference=train_reference,
                train_connectivity=train_connectivity,
                audit_states=audit_states,
                audit_reference=audit_reference,
                audit_connectivity=audit_connectivity,
                steps=int(training["steps"]),
                learning_rate=float(training["learning_rate"]),
                checkpoint_directory=output
                / "checkpoints"
                / f"replicate_{replicate_index}"
                / arm_id,
                run_identity=run_identity,
            )
            del final_params
            evaluated[arm_id] = (centers, actions)
            arm_rows.append(row)
            print(
                f"replicate={replicate_index}",
                arm_id,
                f"E/N={row['final_audit']['energy_per_site']:.8f}",
                flush=True,
            )
        pairs = {
            "spatial_minus_base": ("su2_multiplicity_spatial", "common_su2_base"),
            "scrambled_minus_base": ("su2_scrambled_geometry", "common_su2_base"),
            "spatial_minus_scrambled": (
                "su2_multiplicity_spatial",
                "su2_scrambled_geometry",
            ),
        }
        if include_fixed:
            pairs.update(
                {
                    "fixed_sz_minus_base": (
                        "fixed_sz_spatial_vit",
                        "common_su2_base",
                    ),
                    "spatial_minus_fixed_sz": (
                        "su2_multiplicity_spatial",
                        "fixed_sz_spatial_vit",
                    ),
                }
            )
        paired = {}
        for name, (first, second) in pairs.items():
            comparison = paired_common_sample_energy_difference(
                *evaluated[first], *evaluated[second], audit_reference
            )
            row = asdict(comparison)
            row["difference_per_site"] = comparison.difference / 36.0
            row["standard_error_per_site"] = comparison.standard_error / 36.0
            paired[name] = row
        replicate = {
            "replicate_index": replicate_index,
            "sample_ledger": {
                "calibration": calibration_ledger,
                "training": train_ledger,
                "audit": audit_ledger,
            },
            "base_audit": _summary(
                base_centers, base_actions, audit_reference, audit_states
            ),
            "fixed_sz_registration": fixed_registration,
            "arms": arm_rows,
            "paired_comparisons": paired,
        }
        replicates.append(replicate)
        atomic_json(output / f"replicate_{replicate_index}.json", replicate)
    aggregate = _aggregate(replicates)
    arm_energy_means = {}
    for arm_id in arm_ids:
        values = np.asarray(
            [
                next(row for row in rep["arms"] if row["arm_id"] == arm_id)[
                    "final_audit"
                ]["energy_per_site"]
                for rep in replicates
            ]
        )
        arm_energy_means[arm_id] = {
            "mean_energy_per_site": float(np.mean(values)),
            "replicate_standard_error_per_site": float(
                np.std(values, ddof=1) / np.sqrt(len(values))
            ),
            "gap_above_exact_per_site": float(np.mean(values) - EXACT_ENERGY_PER_SITE),
            "gap_above_gcnn_per_site": float(np.mean(values) - GCNN_ENERGY_PER_SITE),
        }
    minimum_ess = min(
        row["final_audit"]["importance_ess_fraction"]
        for replicate in replicates
        for row in replicate["arms"]
    )
    maximum_imaginary = max(
        row["final_audit"]["maximum_imaginary_local_energy"]
        for replicate in replicates
        for row in replicate["arms"]
    )
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "model_case": {
            "hamiltonian": "spin-1/2 triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic torus",
            "j1": 1.0,
            "j2_over_j1": 0.125,
            "j1_bond_count": len(j1_bonds),
            "j2_bond_count": len(j2_bonds),
            "target_sector": "S=0, M=0, translation projected",
        },
        "literature_targets": {
            "exact_36_site_energy_per_site": EXACT_ENERGY_PER_SITE,
            "raw_space_group_gcnn_energy_per_site": GCNN_ENERGY_PER_SITE,
            "raw_space_group_gcnn_uncertainty": GCNN_UNCERTAINTY,
            "citation": "C. Roth, A. Szabo, and A. H. MacDonald, Phys. Rev. B 108, 054410 (2023), Table III",
            "doi": "https://doi.org/10.1103/PhysRevB.108.054410",
        },
        "reference": {
            "checkpoint_sha256": pytree_sha256(frozen),
            "route_ledger": route_ledger,
            "no_ed_repeated": True,
            "important_boundary": "reference is a sampler-admitted near-dimer checkpoint, not a converged variational state",
        },
        "attention_contract": {
            "acts_only_on_static_support_descriptors_and_reduced_multiplicity_maps": True,
            "magnetic_indices_carried_only_by_fixed_cg_tensors": True,
            "zero_head_exactly_nests_common_su2_base": True,
            "geometry_control": "independent node permutation on every route with descriptor marginals preserved",
            "trainable_parameter_count": su2_multiplicity_spatial_attention_parameter_count(
                initial_spatial_attention
            ),
            "reduced_backbone_parameter_count": int(
                sum(
                    np.prod(leaf.shape)
                    for leaf in jax.tree_util.tree_leaves(frozen["backbone"])
                )
            ),
            "reduced_backbone_jointly_trained": joint_training,
            "joint_trainable_parameter_count": int(
                su2_multiplicity_spatial_attention_parameter_count(
                    initial_spatial_attention
                )
                + (
                    sum(
                        np.prod(leaf.shape)
                        for leaf in jax.tree_util.tree_leaves(frozen["backbone"])
                    )
                    if joint_training
                    else 0
                )
            ),
        },
        "replicates": replicates,
        "paired_aggregate": aggregate,
        "arm_energy_aggregate": arm_energy_means,
        "gates": {
            "minimum_importance_ess_fraction": minimum_ess,
            "maximum_imaginary_local_energy": maximum_imaginary,
            "ess_gate_passed": minimum_ess
            >= float(spec["gates"]["importance_ess_fraction_minimum"]),
            "real_local_energy_gate_passed": maximum_imaginary <= 1.0e-10,
        },
        "interpretation_boundary": {
            "paired_pilot_not_publication_energy_claim": True,
            "full_gamma_a1_projected": False,
            "fixed_sz_control_is_a_six_scalar_common_base_correlator": include_fixed,
            "independent_fixed_sz_control_included": include_fixed,
        },
    }
    result["gates"]["all_admission_gates_passed"] = bool(
        result["gates"]["ess_gate_passed"]
        and result["gates"]["real_local_energy_gate_passed"]
    )
    atomic_json(result_path, result)
    return result


__all__ = ["run_spin_n36_multiplicity_spatial_attention"]
