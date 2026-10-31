"""Explicit projected-parton plus exact-SU(2) multiplicity-channel pilot.

This workflow repairs the symmetry mismatch in the earlier multiplicative
``pi-flux * fixed-Sz ViT`` experiment.  The immutable projected pi/0-flux
state is retained as an explicit coherent channel, while the trainable
correction is a translation-projected SU(2)-scalar ClebschTree spatial-
attention state.  The zero mixing head is exactly the bare parton carrier, so
the paired final-minus-initial estimator directly tests whether the new
SU(2) channel adds variational value.

The N=36 pilot projects translations but not the full D6 point group on the
residual channel.  It therefore establishes exact ``S=0`` and ``k=0`` only;
full ``Gamma-A1`` closure is a separate preregistered architecture gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
    _load_reference,
)
from projects.clebsch_tree_nqs.workflows.spin_n36_physical_three_arm import (
    EXACT_ENERGY_PER_SITE,
    GCNN_ENERGY_PER_SITE,
    _parton_samples,
    _train_arm,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.coherent_su2_carrier import (
    CoherentSU2CarrierConfig,
    coherent_su2_carrier_amplitudes,
    coherent_su2_carrier_coefficient,
    init_coherent_su2_carrier_head,
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
from su2_nqs.vmc.spin_physical import padded_spin_connectivity


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_su2_coherent_carrier"
ARM_ID = "pi_flux_plus_coherent_su2_multiplicity_attention"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 coherent-SU2 carrier identity")
    for name in ("reference", "sampling", "model", "training", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    sampling = spec["sampling"]
    for name in (
        "replicates",
        "calibration_count",
        "training_count",
        "audit_count",
        "burn_sweeps",
        "decorrelation_sweeps",
    ):
        if int(sampling.get(name, 0)) < 1:
            raise ValueError(f"sampling.{name} must be positive")
    if int(sampling["replicates"]) < 2:
        raise ValueError("the paired pilot requires at least two replicates")
    if int(spec["training"].get("steps", 0)) < 2:
        raise ValueError("training.steps must be at least two for joint activation")
    if float(spec["training"].get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("training.learning_rate must be positive")


def rms_match_residual_scale(
    carrier_amplitudes: np.ndarray,
    residual_amplitudes: np.ndarray,
    *,
    target_relative_norm: float = 1.0,
) -> float:
    """Match Hilbert norms from carrier-Born samples, without energy calls.

    For samples distributed as ``|carrier|^2``, the second moment of
    ``residual/carrier`` estimates ``||residual||^2/||carrier||^2``.  This is
    the relevant projective scale; matching raw amplitudes on the same sample
    double-counts the carrier Born weight and can catastrophically miss a
    heavy-tailed residual.
    """

    carrier = np.asarray(carrier_amplitudes, dtype=np.complex128)
    residual = np.asarray(residual_amplitudes, dtype=np.complex128)
    if carrier.shape != residual.shape or carrier.ndim != 1 or carrier.size < 1:
        raise ValueError("carrier and residual must be equal nonempty vectors")
    if not np.isfinite(target_relative_norm) or target_relative_norm <= 0.0:
        raise ValueError("target_relative_norm must be positive and finite")
    if np.any(np.abs(carrier) <= np.finfo(np.float64).tiny):
        raise FloatingPointError("carrier-Born calibration contains a zero center")
    ratio_rms = float(np.sqrt(np.mean(np.abs(residual / carrier) ** 2)))
    if not np.isfinite(ratio_rms) or ratio_rms <= 0.0:
        raise FloatingPointError("residual/carrier RMS is not positive and finite")
    return float(target_relative_norm) / ratio_rms


def run_spin_n36_su2_coherent_carrier(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    _basin, base_config, bonds, route_ledger, _burn, final = _load_reference(spec)
    frozen = final["params"]
    orbit = build_n36_pi_zero_flux_orbit()
    sampling = spec["sampling"]
    model = spec["model"]
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
        calibration_states = _parton_samples(
            parton_host,
            count=int(sampling["calibration_count"]),
            burn_sweeps=int(sampling["burn_sweeps"]),
            decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
            seed=replicate_seed + 3,
        )
        audit_states = _parton_samples(
            parton_host,
            count=int(sampling["audit_count"]),
            burn_sweeps=int(sampling["burn_sweeps"]),
            decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
            seed=replicate_seed + 2,
        )
        train_reference = parton_host(train_states)
        calibration_reference = parton_host(calibration_states)
        audit_reference = parton_host(audit_states)
        train_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(train_states, bonds)[:3]
        )
        audit_connectivity = tuple(
            jnp.asarray(value)
            for value in padded_spin_connectivity(audit_states, bonds)[:3]
        )

        attention = init_su2_multiplicity_spatial_attention(
            jax.random.PRNGKey(replicate_seed + 17), frozen, su2_config
        )
        residual_params = {
            "backbone": frozen["backbone"],
            "attention": attention,
        }

        def residual_amplitude(candidate, state):
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

        calibration_residual = np.asarray(
            jax.vmap(lambda row: residual_amplitude(residual_params, row))(
                jnp.asarray(calibration_states, dtype=jnp.int32)
            )
        )
        residual_scale = rms_match_residual_scale(
            calibration_reference,
            calibration_residual,
            target_relative_norm=float(model["maximum_residual_norm_fraction"]),
        )
        mixture_config = CoherentSU2CarrierConfig(
            n_sites=36,
            residual_scale=residual_scale,
            complex_mixing=False,
        )
        initial_params = {
            "head": init_coherent_su2_carrier_head(mixture_config),
            **residual_params,
        }

        def coherent_amplitude(candidate, state):
            carrier = projected_pi_zero_flux_a1_amplitudes_jax(
                state[None, :], orbit=orbit
            )[0]
            residual = residual_amplitude(candidate, state)
            return coherent_su2_carrier_amplitudes(
                candidate["head"], carrier, residual, mixture_config
            )

        final_params, _centers, _actions, row = _train_arm(
            arm_id=ARM_ID,
            params=initial_params,
            amplitude_fn=coherent_amplitude,
            config=su2_config,
            train_states=train_states,
            train_reference=train_reference,
            train_connectivity=train_connectivity,
            audit_states=audit_states,
            audit_reference=audit_reference,
            audit_connectivity=audit_connectivity,
            steps=int(spec["training"]["steps"]),
            learning_rate=float(spec["training"]["learning_rate"]),
            checkpoint_directory=(
                output / "checkpoints" / f"replicate_{replicate_index}"
            ),
            run_identity={
                "task": TASK,
                "config_sha256": canonical_sha256(spec),
                "replicate_index": replicate_index,
                "reference_checkpoint_sha256": pytree_sha256(frozen),
                "train_states_sha256": canonical_sha256(train_states.tolist()),
                "audit_states_sha256": canonical_sha256(audit_states.tolist()),
            },
        )
        coefficient = coherent_su2_carrier_coefficient(
            final_params["head"], mixture_config
        )
        row["channel_calibration"] = {
            "residual_scale": residual_scale,
            "calibration_sample_count": len(calibration_states),
            "maximum_residual_norm_fraction": float(
                model["maximum_residual_norm_fraction"]
            ),
            "uncalibrated_residual_to_carrier_ratio_rms": float(
                np.sqrt(
                    np.mean(
                        np.abs(calibration_residual / calibration_reference) ** 2
                    )
                )
            ),
            "metric": "parton-Born E[|residual/carrier|^2]; no Hamiltonian calls",
            "initial_exact_carrier_nesting": True,
        }
        row["final_mixing_coordinate"] = float(
            np.asarray(final_params["head"]["coordinates"])[0]
        )
        row["final_residual_coefficient"] = float(np.asarray(coefficient).real)
        replicate = {
            "replicate_index": replicate_index,
            "train_sample_count": len(train_states),
            "calibration_sample_count": len(calibration_states),
            "audit_sample_count": len(audit_states),
            "arm": row,
        }
        replicates.append(replicate)
        atomic_json(output / f"replicate_{replicate_index}.json", replicate)
        print(
            f"replicate={replicate_index} {ARM_ID} "
            f"E/N={row['final_audit']['energy_per_site']:.8f} "
            f"dE/N={row['paired_final_minus_initial']['difference_per_site']:.8f} "
            f"ESS={row['final_audit']['importance_ess_fraction']:.3f}",
            flush=True,
        )

    energies = np.asarray(
        [replicate["arm"]["final_audit"]["energy_per_site"] for replicate in replicates]
    )
    changes = np.asarray(
        [
            replicate["arm"]["paired_final_minus_initial"]["difference_per_site"]
            for replicate in replicates
        ]
    )
    minimum_ess = min(
        replicate["arm"]["final_audit"]["importance_ess_fraction"]
        for replicate in replicates
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
        "architecture_contract": {
            "formula": "psi = psi_pi_flux_A1 + lambda * psi_SU2_multiplicity_attention",
            "carrier_is_explicit_and_frozen": True,
            "zero_head_exactly_equals_bare_carrier": True,
            "configuration_wise_product_used": False,
            "residual_total_spin_sector": "exact S=0, M=0",
            "residual_lattice_sector": "exact k=0; full D6 not yet projected",
            "jointly_trainable_reduced_backbone_and_spatial_attention": True,
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
        "aggregate": {
            "mean_energy_per_site": float(np.mean(energies)),
            "replicate_standard_error_per_site": float(
                np.std(energies, ddof=1) / np.sqrt(len(energies))
            ),
            "gap_above_exact_per_site": float(
                np.mean(energies) - EXACT_ENERGY_PER_SITE
            ),
            "paired_final_minus_bare_carrier_mean_per_site": float(
                np.mean(changes)
            ),
            "paired_change_replicate_standard_error_per_site": float(
                np.std(changes, ddof=1) / np.sqrt(len(changes))
            ),
            "paired_change_negative_replicates": int(
                np.count_nonzero(changes < 0.0)
            ),
            "minimum_final_importance_ess_fraction": minimum_ess,
            "ess_gate_passed": minimum_ess
            >= float(spec["gates"]["importance_ess_fraction_minimum"]),
        },
        "interpretation_boundary": {
            "local_short_horizon_pilot": True,
            "paired_change_is_the_primary_observable": True,
            "full_gamma_a1_residual_not_yet_established": True,
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


__all__ = [
    "ARM_ID",
    "TASK",
    "rms_match_residual_scale",
    "run_spin_n36_su2_coherent_carrier",
]
