"""Untouched-audit canary for a physical-distance-bounded SU(2) frame carrier."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.coherent_plaquette_frames import distinct_partition_orbit
from su2_nqs.benchmarks.coherent_plaquette_sampling import (
    build_linear_resonance_sampled_action,
    linear_resonance_energy_variance,
    ordered_frame_orbit_with_weights,
    sample_coherent_plaquette_frames,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.physical_trust_region import (
    bound_physical_state_update,
    sampled_physical_state_distance,
)


TASK = "spin_n36_coherent_frame_canary"
TRIANGULAR_6X6 = ((6, 0), (0, 6))


jax.config.update("jax_enable_x64", True)


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid coherent-frame canary identity")
    controls = spec.get("controls")
    optimizer = spec.get("optimizer")
    gates = spec.get("gates")
    if not all(isinstance(value, dict) for value in (controls, optimizer, gates)):
        raise TypeError("controls, optimizer, and gates must be objects")
    development = controls.get("development_seeds")
    audit = controls.get("untouched_audit_seeds")
    if not isinstance(development, dict) or set(development) != {
        "square_6x6_j2_0.5",
        "triangular_6x6_j2_0.125",
    }:
        raise ValueError("one development seed is required for each case")
    if (
        not isinstance(audit, list)
        or len(audit) < 4
        or len(set(audit)) != len(audit)
        or set(int(value) for value in development.values()).intersection(audit)
    ):
        raise ValueError("at least four fresh untouched audit seeds are required")
    if int(controls.get("sample_count_per_seed", 0)) < 128:
        raise ValueError("sample_count_per_seed must be at least 128")
    for name in (
        "finite_difference_step",
        "learning_rate",
        "maximum_fubini_study_distance",
        "maximum_symmetric_kl",
    ):
        value = float(optimizer.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"optimizer.{name} must be positive and finite")
    if float(gates.get("minimum_absolute_parameter_update", 0.0)) < 0.0:
        raise ValueError("minimum_absolute_parameter_update must be nonnegative")
    if float(gates.get("minimum_energy_z_score", 0.0)) < 0.0:
        raise ValueError("minimum_energy_z_score must be nonnegative")
    if spec.get("remote_training_on_pass") is not False:
        raise ValueError("the canary must not submit remote work")


def _case(case_id: str):
    if case_id == "square_6x6_j2_0.5":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=False)
        group = square_gamma_a1_group(6)
        bonds = square_j1j2_bonds(6, 6, j1=1.0, j2=0.5, periodic=True)
    elif case_id == "triangular_6x6_j2_0.125":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=True)
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_6X6)
        bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_6X6, j1=1.0, j2=0.125
        )
    else:
        raise ValueError(f"unknown coherent-frame canary case: {case_id}")
    proposal_frames = distinct_partition_orbit(layout.physical_cells, group)
    ordered_frames, frame_weights = ordered_frame_orbit_with_weights(
        layout.physical_cells, group
    )
    return proposal_frames, ordered_frames, frame_weights, bonds


def _sampled_action(case_id: str, *, seed: int, sample_count: int):
    proposal_frames, ordered_frames, frame_weights, bonds = _case(case_id)
    batch = sample_coherent_plaquette_frames(
        proposal_frames, sample_count=sample_count, seed=seed
    )
    action = build_linear_resonance_sampled_action(
        batch, ordered_frames, frame_weights, bonds
    )
    return batch, action, len(proposal_frames), len(ordered_frames)


def _run_case(case_id: str, spec: dict[str, object]) -> dict[str, object]:
    controls = spec["controls"]
    optimizer = spec["optimizer"]
    count = int(controls["sample_count_per_seed"])
    development_seed = int(controls["development_seeds"][case_id])
    _batch, development, proposal_count, ordered_count = _sampled_action(
        case_id, seed=development_seed, sample_count=count
    )
    epsilon = float(optimizer["finite_difference_step"])
    energy_plus = linear_resonance_energy_variance(development, epsilon)[
        "energy_total"
    ]
    energy_minus = linear_resonance_energy_variance(development, -epsilon)[
        "energy_total"
    ]
    gradient = float((energy_plus - energy_minus) / (2.0 * epsilon))
    raw_direction = jnp.asarray(
        [float(optimizer["learning_rate"]) * gradient], dtype=jnp.float64
    )
    params = jnp.asarray([0.0], dtype=jnp.float64)
    sample_indices = jnp.arange(count, dtype=jnp.int32)

    def development_center(candidate, indices):
        values = (
            jnp.asarray(development.base_amplitude)
            + candidate[0] * jnp.asarray(development.response_amplitude)
        )
        return values[indices]

    bounded = bound_physical_state_update(
        params,
        raw_direction,
        sample_indices,
        center_fn=development_center,
        maximum_fubini_study_distance=float(
            optimizer["maximum_fubini_study_distance"]
        ),
        maximum_symmetric_kl=float(optimizer["maximum_symmetric_kl"]),
    )
    update = float(np.asarray(bounded.direction)[0])
    final_coefficient = -update
    development_initial = linear_resonance_energy_variance(development, 0.0)
    development_final = linear_resonance_energy_variance(
        development, final_coefficient
    )

    audit_rows = []
    for seed in controls["untouched_audit_seeds"]:
        seed = int(seed)
        _audit_batch, audit, _proposal_count, _ordered_count = _sampled_action(
            case_id, seed=seed, sample_count=count
        )

        def audit_center(candidate, indices):
            values = (
                jnp.asarray(audit.base_amplitude)
                + candidate[0] * jnp.asarray(audit.response_amplitude)
            )
            return values[indices]

        distance = sampled_physical_state_distance(
            params,
            jnp.asarray([final_coefficient], dtype=jnp.float64),
            sample_indices,
            center_fn=audit_center,
        )
        initial = linear_resonance_energy_variance(audit, 0.0)
        final = linear_resonance_energy_variance(audit, final_coefficient)
        audit_rows.append(
            {
                "sample_seed": seed,
                "initial": initial,
                "final": final,
                "paired_energy_change_per_site": float(
                    final["energy_per_site"] - initial["energy_per_site"]
                ),
                "paired_variance_change_per_site": float(
                    final["variance_per_site"] - initial["variance_per_site"]
                ),
                "physical_distance": asdict(distance),
                "full_h_configuration_calls": int(
                    2 * audit.full_h_configuration_calls
                ),
            }
        )
    energy_changes = [row["paired_energy_change_per_site"] for row in audit_rows]
    variance_changes = [row["paired_variance_change_per_site"] for row in audit_rows]
    energy_mean = float(np.mean(energy_changes))
    energy_se = _standard_error(energy_changes)
    variance_mean = float(np.mean(variance_changes))
    variance_se = _standard_error(variance_changes)
    energy_z = -energy_mean / max(energy_se, np.finfo(float).tiny)
    maximum_audit_fs = max(
        row["physical_distance"]["fubini_study_distance"] for row in audit_rows
    )
    maximum_audit_kl = max(
        row["physical_distance"]["symmetric_kl"] for row in audit_rows
    )
    gates = spec["gates"]
    passed = bool(
        abs(update) >= float(gates["minimum_absolute_parameter_update"])
        and energy_mean < 0.0
        and energy_z >= float(gates["minimum_energy_z_score"])
        and variance_mean <= float(gates["maximum_variance_z_score"])
        * max(variance_se, np.finfo(float).tiny)
        and maximum_audit_fs
        <= float(optimizer["maximum_fubini_study_distance"])
        and maximum_audit_kl <= float(optimizer["maximum_symmetric_kl"])
    )
    return {
        "case_id": case_id,
        "proposal_frame_count": proposal_count,
        "ordered_frame_class_count": ordered_count,
        "development_seed": development_seed,
        "development_gradient": gradient,
        "raw_parameter_direction": float(raw_direction[0]),
        "bounded_parameter_update": update,
        "final_response_coefficient": final_coefficient,
        "development_physical_trust": {
            "scale": bounded.scale,
            "distance": asdict(bounded.distance),
            "maximum_fubini_study_distance": bounded.maximum_fubini_study_distance,
            "maximum_symmetric_kl": bounded.maximum_symmetric_kl,
        },
        "development_initial": development_initial,
        "development_final": development_final,
        "audit_rows": audit_rows,
        "audit_summary": {
            "paired_energy_change_per_site_mean": energy_mean,
            "paired_energy_change_per_site_standard_error": energy_se,
            "energy_decrease_z_score": energy_z,
            "paired_variance_change_per_site_mean": variance_mean,
            "paired_variance_change_per_site_standard_error": variance_se,
            "maximum_fubini_study_distance": maximum_audit_fs,
            "maximum_symmetric_kl": maximum_audit_kl,
        },
        "passed": passed,
    }


def run_spin_n36_coherent_frame_canary(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train on development samples, then open untouched audit samples once."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    source_path = Path(str(spec["source_n36_support_gate"]))
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    source = json.loads(source_path.read_text())
    if (
        source.get("task") != "spin_n36_coherent_frame_support"
        or source.get("all_support_gates_passed") is not True
        or source.get("bounded_su2_canary_allowed") is not True
    ):
        raise RuntimeError("the sealed N36 coherent-frame support gate did not pass")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    cases = [
        _run_case(case_id, spec)
        for case_id in ("square_6x6_j2_0.5", "triangular_6x6_j2_0.125")
    ]
    passed = bool(all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_n36_support_result_sha256": source["result_sha256"],
        "ansatz": {
            "base": "coherent translated/rotated plaquette A1 products",
            "additive_response": "two local B1 singlets in every frame, projected to Gamma-A1",
            "total_spin": "exact S=0",
            "magnetic_indices_learned": False,
            "parameter_count": 1,
        },
        "cases": cases,
        "all_untouched_energy_gates_passed": passed,
        "remote_pilot_reapplication_allowed": passed,
        "remote_production_allowed": False,
        "decision": (
            "prepare_matched_three_arm_remote_pilot_with_physical_trust_radius"
            if passed
            else "stop_coherent_frame_route_no_untouched_variational_gain"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n36_coherent_frame_canary"]
