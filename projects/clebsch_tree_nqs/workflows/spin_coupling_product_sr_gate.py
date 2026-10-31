"""Final exact-N16 product-FS/SR gate for the existing shared-coupling VAP.

No architecture or state definition is changed.  Three training couplings
define a product of independent physical rays.  Held couplings are evaluated
only after the fixed-budget endpoint is saved, so they cannot tune training.
The dense singlet VAP is validation-only and is deliberately not relabelled as
an N36 implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.coupling_conditioned_spatial_vit import wrap_spatial_vit_parameters
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.vmc.exact_state_sr import train_product_state_sr_trust_region
from su2_nqs.vmc.matched_state_step import matched_state_step

from .spin_coupling_stationarity import _saved_operator_state
from .spin_coupling_transfer import response_metrics
from .spin_symmetry_readout import density_readout

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_coupling_product_sr_gate.json"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("task") != "spin_coupling_product_sr_gate":
        raise ValueError("wrong product-SR task")
    source = ROOT / str(spec["source"])
    if not source.is_dir():
        raise FileNotFoundError(source)
    if spec["training_couplings"] != [0.2, 0.5, 0.8]:
        raise ValueError("training split changed")
    if spec["held_out_couplings"] != [0.35, 0.65]:
        raise ValueError("held split changed")
    if int(spec["optimization"]["steps"]) < 1:
        raise ValueError("fixed optimization budget must be positive")
    if "FiLM" in str(spec.get("scope", "")):
        raise ValueError("this gate may not introduce a conditioning control")


def _sealed_source(spec: dict[str, object]) -> tuple[Path, dict[str, object]]:
    source = ROOT / str(spec["source"])
    result = json.loads((source / "results.json").read_text())
    unsigned = dict(result)
    seal = unsigned.pop("result_sha256")
    if canonical_sha256(unsigned) != seal or result.get("status") != "complete":
        raise ValueError("invalid source result seal")
    if result["config"]["parents"] != spec["parents"]:
        raise ValueError("source parents changed")
    for relative, digest in result["artifacts"].items():
        if file_sha256(source / relative) != digest:
            raise ValueError("source artifact changed: " + relative)
    return source, result


def _registration(spec: dict[str, object], source: Path, result: dict[str, object]) -> dict[str, object]:
    inputs = {
        str((source / name).relative_to(ROOT)): file_sha256(source / name)
        for name in ("results.json", "registration.json", "operators.npz", "states.npz", "endpoints_seal.json")
    }
    record = {
        "config": spec,
        "inputs": inputs,
        "source_result_sha256": result["result_sha256"],
        "runtime": numerical_runtime_identity(),
        "sources": transitive_local_source_manifest(
            (Path(__file__),), root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
        ),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["registration_sha256"] = canonical_sha256(record)
    return record


def _physical(reduced: np.ndarray, operators: dict[str, np.ndarray]) -> np.ndarray:
    values = reduced[operators["orbit_id"]] / np.sqrt(operators["orbit_sizes"][operators["orbit_id"]])
    return values / np.linalg.norm(values)


def _rayleigh(vector: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    vector = np.array(vector, dtype=np.complex128, copy=True)
    vector /= np.linalg.norm(vector)
    action = matrix @ vector
    energy = float(np.vdot(vector, action).real)
    return energy / 16, float(np.linalg.norm(action - energy * vector) ** 2 / 16)


def _readout(vector: np.ndarray, operators: dict[str, np.ndarray]) -> tuple[dict[str, object], np.ndarray]:
    rho = reduced_density_matrix(
        _physical(vector, operators), operators["basis"], (0, 1, 4, 5)
    )
    report, _ = density_readout(rho)
    return report, rho


def _readout_distances(left: dict[str, object], right: dict[str, object],
                       left_rho: np.ndarray, right_rho: np.ndarray) -> dict[str, float]:
    trace = float(np.sum(np.abs(np.linalg.eigvalsh(left_rho - right_rho))) / 2)
    tv = float(sum(abs(left["sectors"][str(q)]["p"] - right["sectors"][str(q)]["p"])
                   for q in (0, 2, 4)) / 2)
    return {"plaquette_trace_distance_to_pointwise": trace,
            "p_q_TV_to_pointwise": tv}


def _group_monitor(x: jax.Array, state, h0: jax.Array, h2: jax.Array,
                   training_js: jax.Array, conditional_mask: jax.Array) -> dict[str, object]:
    def objective(values):
        rays = jax.vmap(lambda j: state(values, j, True))(training_js)
        actions = jax.vmap(lambda j, ray: (h0 + j * h2) @ ray)(training_js, rays)
        return jnp.mean(jnp.sum(jnp.conj(rays) * actions, axis=1).real) / 16

    gradient = jax.grad(objective)(x)
    result: dict[str, object] = {"mean_energy_per_site": float(objective(x)),
                                "coordinate_gradient_norm": float(jnp.linalg.norm(gradient))}
    for label, mask in (("common_backbone", 1.0 - conditional_mask),
                        ("conditional_direction", conditional_mask)):
        group_gradient = gradient * mask
        norm = float(jnp.linalg.norm(group_gradient))
        if norm == 0.0:
            result[label] = {"coordinate_gradient_norm": 0.0, "product_FS_speed": 0.0,
                             "unit_FS_descent_slope_per_site": 0.0}
            continue
        direction = -group_gradient / norm
        speeds = []
        for coupling in np.asarray(training_js):
            ray, tangent = jax.jvp(lambda values: state(values, coupling, True),
                                   (x,), (direction,))
            tangent = tangent - ray * jnp.vdot(ray, tangent)
            speeds.append(jnp.vdot(tangent, tangent).real)
        speed = float(jnp.sqrt(jnp.mean(jnp.asarray(speeds))))
        result[label] = {
            "coordinate_gradient_norm": norm,
            "product_FS_speed": speed,
            "unit_FS_descent_slope_per_site": float(-jnp.vdot(gradient, group_gradient).real / (norm * max(speed, 1e-30))),
        }
    return result


def _point_probe(x: jax.Array, coupling: float, state, matrix: np.ndarray,
                 probe: dict[str, object]) -> dict[str, object]:
    def energy(values):
        ray = state(values, coupling, True)
        return jnp.vdot(ray, jnp.asarray(matrix) @ ray).real / 16
    gradient = jax.grad(energy)(x)
    norm = float(jnp.linalg.norm(gradient))
    parent = np.asarray(state(x, coupling, True))
    if norm == 0.0:
        return {"matched": False, "status": "zero_gradient", "delta_energy_per_site": 0.0,
                "coordinate_gradient_norm": 0.0}
    direction = -gradient / norm
    matched = matched_state_step(
        lambda alpha: np.asarray(state(x + alpha * direction, coupling, True)),
        parent, np.ones(len(parent)), float(probe["target_fs"]),
        float(probe["maximum_normalized_parameter_step"]), int(probe["bisection_steps"]),
    )
    before = float(energy(x))
    after, _ = _rayleigh(matched.amplitudes, matrix)
    return {**matched.as_dict(), "coordinate_gradient_norm": norm,
            "energy_before_per_site": before, "energy_after_per_site": after,
            "delta_energy_per_site": after - before}


def _audit_endpoint(x: jax.Array, state, operators: dict[str, np.ndarray],
                    source_states: dict[str, np.ndarray], source_result: dict[str, object],
                    parent: str, spec: dict[str, object]) -> dict[str, object]:
    h0, h2 = operators["h0"], operators["h2"]
    rows: dict[str, object] = {}
    for coupling in spec["all_couplings"]:
        matrix = h0 + coupling * h2
        vector = np.asarray(state(x, coupling, True))
        energy, variance = _rayleigh(vector, matrix)
        readout, rho = _readout(vector, operators)
        point_key = parent + f"__singlet_vap__point_{coupling:.2f}__" + str(coupling)
        point = source_states[point_key]
        point_reduced = np.zeros(len(operators["orbit_sizes"]), dtype=np.complex128)
        np.add.at(point_reduced, operators["orbit_id"], point / np.sqrt(operators["orbit_sizes"][operators["orbit_id"]]))
        point_reduced /= np.linalg.norm(point_reduced)
        point_readout, point_rho = _readout(point_reduced, operators)
        response = response_metrics(lambda values, j: state(values, j, True), x, coupling,
                                    jnp.asarray(h0), jnp.asarray(h2), 1e-4)
        rows[str(coupling)] = {
            "energy_per_site": energy,
            "variance_per_site": variance,
            "multiplicity_readout": readout,
            "readout_distance": _readout_distances(readout, point_readout, rho, point_rho),
            "response": response,
            "point_FS_probe": _point_probe(x, coupling, state, matrix, spec["probe"]),
        }
    return rows


def _gate(parent: str, before: dict[str, object], after: dict[str, object],
          training: dict[str, object], spec: dict[str, object]) -> dict[str, object]:
    limits = spec["gate"]
    history = training["history"]
    first_speed = history[0]["natural_direction_product_fs_speed"]
    first_slope = history[0]["damped_natural_gradient_squared"] / max(first_speed, 1e-30) / 16
    final_slope = training["final_damped_natural_gradient_squared"] / max(
        training["final_natural_direction_product_fs_speed"], 1e-30) / 16
    checks: dict[str, object] = {}
    checks["common_physical_gradient"] = {
        "initial_unit_natural_slope_per_site": first_slope,
        "final_unit_natural_slope_per_site": final_slope,
        "fraction": final_slope / max(first_slope, 1e-30),
        "pass": final_slope <= limits["maximum_final_unit_natural_slope_per_site"]
                and final_slope <= limits["maximum_natural_slope_fraction_of_initial"] * first_slope,
    }
    deltas = [after[str(j)]["point_FS_probe"]["delta_energy_per_site"] for j in spec["all_couplings"]]
    systematic = sum(value < limits["systematic_probe_drop_per_site"] for value in deltas)
    checks["pointwise_FS_probes"] = {
        "deltas_per_site": deltas, "mean_delta_per_site": float(np.mean(deltas)),
        "systematic_drop_count": systematic,
        "pass": systematic <= limits["maximum_systematic_probe_count"]
                and float(np.mean(deltas)) >= limits["minimum_mean_probe_delta_per_site"],
    }
    held = []
    for coupling in spec["held_out_couplings"]:
        old, new = before[str(coupling)], after[str(coupling)]
        held.append({
            "coupling": coupling,
            "energy_change": new["energy_per_site"] - old["energy_per_site"],
            "variance_change": new["variance_per_site"] - old["variance_per_site"],
            "trace_distance_change": new["readout_distance"]["plaquette_trace_distance_to_pointwise"] - old["readout_distance"]["plaquette_trace_distance_to_pointwise"],
            "p_q_TV_change": new["readout_distance"]["p_q_TV_to_pointwise"] - old["readout_distance"]["p_q_TV_to_pointwise"],
        })
    checks["held_out_non_degradation"] = {
        "points": held,
        "pass": all(row["energy_change"] <= limits["held_energy_worsening_tolerance"]
                    and row["variance_change"] <= limits["held_variance_worsening_tolerance"]
                    and row["trace_distance_change"] <= limits["multiplicity_trace_distance_worsening_tolerance"]
                    and row["p_q_TV_change"] <= limits["multiplicity_probability_tv_worsening_tolerance"]
                    for row in held),
    }
    old_hf = max(abs(before[str(j)]["response"]["hellmann_feynman_defect"]) for j in spec["held_out_couplings"])
    new_hf = max(abs(after[str(j)]["response"]["hellmann_feynman_defect"]) for j in spec["held_out_couplings"])
    old_positive = max(0.0, max(before[str(j)]["response"]["energy_curvature_per_site"] for j in spec["all_couplings"]))
    new_positive = max(0.0, max(after[str(j)]["response"]["energy_curvature_per_site"] for j in spec["all_couplings"]))
    checks["response_regularization"] = {
        "maximum_held_abs_HF_defect_before": old_hf,
        "maximum_held_abs_HF_defect_after": new_hf,
        "maximum_positive_curvature_before": old_positive,
        "maximum_positive_curvature_after": new_positive,
        "pass": new_hf <= limits["maximum_held_hf_defect_fraction"] * old_hf
                and new_positive <= max(limits["positive_curvature_absolute_tolerance"],
                                        limits["maximum_positive_curvature_fraction"] * old_positive),
    }
    return {"parent": parent, "checks": checks,
            "pass": all(block["pass"] for block in checks.values())}


def run(config: Path = CONFIG) -> dict[str, object]:
    spec = json.loads(config.read_text())
    _validate(spec)
    source, source_result = _sealed_source(spec)
    output = ROOT / str(spec["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("immutable product-SR gate already exists")
    output.mkdir(parents=True, exist_ok=True)
    registration = _registration(spec, source, source_result)
    atomic_json(output / "registration.json", registration)
    start = perf_counter()
    def timeout(*_):
        raise TimeoutError("registered product-SR wall cap reached")
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(spec["wall_seconds_cap"]))
    try:
        with np.load(source / "operators.npz", allow_pickle=False) as archive:
            operators = {key: archive[key].copy() for key in archive.files}
        with np.load(source / "states.npz", allow_pickle=False) as archive:
            source_states = {key: archive[key].copy() for key in archive.files}
        cfg = FixedSzSpatialViTConfig(**source_result["model_config"])
        template = wrap_spatial_vit_parameters(init_fixed_sz_spatial_vit(jax.random.PRNGKey(0), cfg))
        template_flat, unravel = ravel_pytree(template)
        marker = jax.tree.map(jnp.zeros_like, template)
        marker["coupling_bias"] = jnp.ones_like(marker["coupling_bias"])
        conditional_mask, _ = ravel_pytree(marker)
        gauge = conditional_mask / jnp.sqrt(jnp.sum(conditional_mask))
        canonicalize = jax.jit(lambda values: values - gauge * jnp.vdot(gauge, values).real)
        if float(jnp.linalg.norm(canonicalize(canonicalize(template_flat)) - canonicalize(template_flat))) > 1e-13:
            raise RuntimeError("coupling gauge map is not idempotent")
        state = jax.jit(_saved_operator_state(cfg, operators, unravel))
        h0, h2 = jnp.asarray(operators["h0"]), jnp.asarray(operators["h2"])
        training_js = jnp.asarray(spec["training_couplings"])
        eigenvalues, eigenvectors = np.linalg.eigh(operators["p0"])
        isometry_np = eigenvectors[:, eigenvalues > 0.5]
        if (isometry_np.shape != (len(operators["p0"]), 31)
                or np.max(abs(isometry_np.conj().T @ isometry_np - np.eye(31))) > 1e-11
                or np.max(abs(isometry_np @ isometry_np.conj().T - operators["p0"])) > 1e-10):
            raise RuntimeError("saved N16 singlet projector does not factor to rank 31")
        isometry = jnp.asarray(isometry_np)
        training_matrices = jax.vmap(
            lambda j: isometry.conj().T @ (h0 + j * h2) @ isometry
        )(training_js)
        records: dict[str, object] = {}
        all_pass = True
        for parent in spec["parents"]:
            name = parent + "__singlet_vap__shared"
            identity = {"run": name, "registration_sha256": source_result["registration_sha256"]}
            initial = load_parameter_artifact(source / name / "parameters.json", template,
                                              expected_identity=identity)
            initial_flat, _ = ravel_pytree(initial)
            canonical_flat = canonicalize(initial_flat)
            gauge_replay = max(float(jnp.max(jnp.abs(state(initial_flat, j, True) - state(canonical_flat, j, True))))
                               for j in training_js)
            if gauge_replay > 2e-11:
                raise RuntimeError(f"uniform coupling gauge is not physical: {gauge_replay}")
            initial = unravel(canonical_flat)
            initial_monitor = _group_monitor(canonical_flat, state, h0, h2, training_js, conditional_mask)
            before = _audit_endpoint(canonical_flat, state, operators, source_states, source_result, parent, spec)

            def states_fn(parameters):
                values, _ = ravel_pytree(parameters)
                return jax.vmap(
                    lambda j: isometry.conj().T @ state(values, j, True)
                )(training_js)

            trained, training = train_product_state_sr_trust_region(
                initial,
                states_fn=states_fn,
                hamiltonian_action=lambda rays: jnp.einsum("kij,kj->ki", training_matrices, rays),
                canonicalize_flat=canonicalize,
                project_flat_tangent=canonicalize,
                **spec["optimization"],
            )
            final_flat, _ = ravel_pytree(trained)
            final_monitor = _group_monitor(final_flat, state, h0, h2, training_js, conditional_mask)
            destination = output / parent
            destination.mkdir()
            save_parameter_artifact(destination / "parameters.json", trained,
                                    identity={"task": spec["task"], "parent": parent,
                                              "registration_sha256": registration["registration_sha256"]})
            # The held audit begins only after the immutable endpoint is written.
            after = _audit_endpoint(final_flat, state, operators, source_states, source_result, parent, spec)
            decision = _gate(parent, before, after, training, spec)
            all_pass = all_pass and decision["pass"]
            record = {
                "parent": parent,
                "initial_parameters_sha256": pytree_sha256(initial),
                "final_parameters_sha256": pytree_sha256(trained),
                "uniform_coupling_gauge_replay_max": gauge_replay,
                "uniform_coupling_gauge_removed": float(jnp.vdot(gauge, final_flat).real),
                "initial_direction_monitor": initial_monitor,
                "final_direction_monitor": final_monitor,
                "training": training,
                "before": before,
                "after": after,
                "gate": decision,
            }
            atomic_json(destination / "results.json", record)
            records[parent] = record
            print(json.dumps({"parent": parent, "pass": decision["pass"],
                              "accepted": training["accepted_steps"],
                              "energy": training["final_mean_energy"] / 16,
                              "seconds": perf_counter() - start}), flush=True)
        result = {
            "status": "complete",
            "task": spec["task"],
            "config": spec,
            "model_config": asdict(cfg),
            "registration_sha256": registration["registration_sha256"],
            "records": records,
            "n16_gate_pass": all_pass,
            "affine_conditioning_capacity_test_licensed": all_pass,
            "n36_status": ("eligible_for_equivalent_sampled-SU2 preflight, but dense VAP is forbidden"
                            if all_pass else "closed: N16 optimized gate failed"),
            "n36_training_run": False,
            "new_head": False,
            "new_ED_calls": 0,
            "new_samples": 0,
            "remote_released": False,
            "seconds": perf_counter() - start,
            "runtime": numerical_runtime_identity(),
            "interpretation": "Only a pass licenses affine-bias versus matched coupling-token/FiLM. A fail attributes no defect to affine capacity because the registered stationarity/non-degradation gate was not met.",
        }
        result["result_sha256"] = canonical_sha256(result)
        atomic_json(output / "results.json", result)
        return result
    except Exception as error:
        atomic_json(output / "failure.json", {"error": repr(error), "seconds": perf_counter() - start})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)
