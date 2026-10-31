"""Local exact-support benchmark for the minimal support-aware Spatial-CG."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    coo_matvec,
    reynolds_project_state,
)
from su2_nqs.benchmarks.support_aware_spatial_cg import (
    spatial_cg_route_spec,
    support_aware_spatial_cg_oracle,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.energy_targets import observed_energy_to_target
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_routes import globally_scrambled_support_routes
from su2_nqs.models.spin_clebsch_tree import (
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.models.support_aware_spatial_cg import (
    SupportAwareSpatialCGConfig,
    init_support_aware_spatial_cg,
    support_aware_spatial_cg_amplitudes,
    support_aware_spatial_cg_component_matrix,
    support_aware_spatial_cg_parameter_count,
    support_aware_spatial_cg_route_coefficients,
)
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "support_aware_spatial_cg_local_pilot"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("Spatial-CG pilot requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    model = spec.get("model")
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("optimization.seeds must be nonempty")
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be nonempty")
    required = {
        "single_tree",
        "spatial_cg",
        "scrambled_spatial_cg",
        "single_tree_work_control",
    }
    if {str(arm.get("arm_id")) for arm in arms if isinstance(arm, dict)} != required:
        raise ValueError("the four preregistered comparison arms are required")
    for arm in arms:
        if not isinstance(arm, dict) or int(arm.get("steps", 0)) < 1:
            raise ValueError("every arm requires a positive step count")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        if case.get("geometry") not in {"square", "triangular"}:
            raise ValueError("case geometry must be square or triangular")
        if (case.get("rows"), case.get("columns")) != (4, 4):
            raise ValueError("the local pilot is preregistered on 4x4")
        if case.get("projection") != "translation":
            raise ValueError("the pilot requires the common translation sector")
        if "exact_energy_anchor" not in case:
            raise ValueError("every case needs a stored exact energy anchor")


def _scrambled_config(
    physical: SupportAwareSpatialCGConfig,
    *,
    seed: int,
) -> tuple[SupportAwareSpatialCGConfig, list[int]]:
    orders = dict(zip(physical.route_names, physical.route_orders, strict=True))
    preserve = physical.route_names[physical.reference_route]
    rng = np.random.default_rng(seed)
    for _ in range(256):
        permutation = tuple(int(value) for value in rng.permutation(physical.n_sites))
        try:
            scrambled = globally_scrambled_support_routes(
                orders, permutation, preserve_name=preserve
            )
        except ValueError:
            continue
        return (
            SupportAwareSpatialCGConfig(
                backbone=physical.backbone,
                route_names=physical.route_names,
                route_orders=tuple(scrambled[name] for name in physical.route_names),
                # Deliberately keep the physical-route descriptors.  The
                # control destroys their relation to the actual supports.
                route_features=physical.route_features,
                feature_names=physical.feature_names,
                reference_route=physical.reference_route,
            ),
            list(permutation),
        )
    raise RuntimeError("failed to construct a distinct scrambled route bank")


def _projected_component_diagnostics(
    params: dict[str, object],
    config: SupportAwareSpatialCGConfig,
    basis: jax.Array,
    indices: jax.Array,
    data: dict[str, object],
    mixture_energy: float,
) -> dict[str, object]:
    components = np.asarray(
        support_aware_spatial_cg_component_matrix(params, basis, config),
        dtype=np.float64,
    )
    projected = np.column_stack(
        [
            np.asarray(reynolds_project_state(components[:, route], indices))
            for route in range(config.n_routes)
        ]
    )
    norms = np.linalg.norm(projected, axis=0)
    if np.any(norms <= 1.0e-14):
        raise RuntimeError("a projected route component has zero norm")
    normalized = projected / norms
    singular = np.linalg.svd(normalized, compute_uv=False)
    ratios = singular / singular[0]
    route_energies = []
    for route in range(config.n_routes):
        state = normalized[:, route]
        route_energies.append(
            float(np.vdot(state, data["hamiltonian"] @ state).real)
        )
    coefficients = np.asarray(
        support_aware_spatial_cg_route_coefficients(params, config),
        dtype=np.float64,
    )
    squared = coefficients**2
    participation = float(
        squared.sum() ** 2 / np.sum(squared**2)
    ) if np.any(squared) else 0.0
    best = float(min(route_energies))
    return {
        "route_names": list(config.route_names),
        "route_coefficients": coefficients.tolist(),
        "coefficient_participation_ratio": participation,
        "projected_component_singular_value_ratios": ratios.tolist(),
        "projected_component_rank_at_1e_minus_10": int(
            np.count_nonzero(ratios > 1.0e-10)
        ),
        "single_route_energies": {
            name: energy
            for name, energy in zip(config.route_names, route_energies, strict=True)
        },
        "best_single_route_energy": best,
        "coherent_gain_below_best_single_route": best - mixture_energy,
    }


def _train_trial(
    arm: dict[str, object],
    *,
    seed: int,
    spec: dict[str, object],
    data: dict[str, object],
    physical_config: SupportAwareSpatialCGConfig,
    scrambled_config: SupportAwareSpatialCGConfig,
) -> dict[str, object]:
    arm_id = str(arm["arm_id"])
    is_spatial = arm_id in {"spatial_cg", "scrambled_spatial_cg"}
    active_config = scrambled_config if arm_id == "scrambled_spatial_cg" else physical_config
    initialized = init_support_aware_spatial_cg(
        jax.random.PRNGKey(seed), active_config
    )
    params = initialized if is_spatial else initialized["backbone"]
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    reference_order = jnp.asarray(
        physical_config.route_orders[physical_config.reference_route],
        dtype=jnp.int32,
    )
    indices = jnp.asarray(
        data["tables"][str(spec["projection"])], dtype=jnp.int32
    )
    h_rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    h_columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    h_values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    if is_spatial:
        def raw_amplitudes(candidate):
            return support_aware_spatial_cg_amplitudes(
                candidate, basis, active_config
            )
    else:
        def raw_amplitudes(candidate):
            return spin_clebsch_tree_amplitudes(
                candidate,
                basis[:, reference_order],
                physical_config.backbone,
            )

    def amplitudes(candidate):
        return reynolds_project_state(raw_amplitudes(candidate), indices)

    def loss(candidate):
        state = amplitudes(candidate)
        action = coo_matvec(h_rows, h_columns, h_values, state)
        return jnp.vdot(state, action).real / jnp.vdot(state, state).real

    paired_baseline = spin_clebsch_tree_amplitudes(
        initialized["backbone"],
        basis[:, reference_order],
        physical_config.backbone,
    )
    paired_spatial = support_aware_spatial_cg_amplitudes(
        initialized, basis, active_config
    )
    initial_match = float(
        jnp.max(jnp.abs(paired_spatial - paired_baseline))
    )
    if initial_match > 1.0e-13:
        raise RuntimeError("Spatial-CG no longer nests the paired Morton baseline")
    query_gradient_norm = None
    if is_spatial:
        initial_gradient = jax.grad(loss)(params)
        query_gradient_norm = float(
            jnp.linalg.norm(initial_gradient["geometry_query"])
        )
        if not np.isfinite(query_gradient_norm) or query_gradient_norm <= 1.0e-14:
            raise RuntimeError("geometry-query energy gradient is zero or nonfinite")

    started = time.perf_counter()
    trained, training = train_exact_adam(
        params,
        loss_fn=loss,
        steps=int(arm["steps"]),
        learning_rate=float(spec["optimization"]["learning_rate"]),
        record_every=int(spec["optimization"]["record_every"]),
    )
    raw = np.asarray(raw_amplitudes(trained), dtype=np.float64)
    metrics = _metrics(raw, data, projection=str(spec["projection"]))
    wall = time.perf_counter() - started
    route_evaluations = active_config.n_routes if is_spatial else 1
    targets = observed_energy_to_target(
        training,
        final_energy=float(metrics["energy"]),
        exact_energy=float(data["exact_energy"]),
        n_sites=int(data["n_sites"]),
        total_steps=int(arm["steps"]),
        targets_per_site=spec["optimization"]["energy_error_per_site_targets"],
    )
    for target in targets:
        updates = target["first_observed_completed_updates"]
        target["raw_tree_configuration_evaluations_to_target"] = (
            None
            if updates is None
            else int(updates) * len(data["basis"]) * route_evaluations
        )
    result: dict[str, object] = {
        "arm_id": arm_id,
        "seed": seed,
        "steps": int(arm["steps"]),
        "route_evaluations_per_configuration": route_evaluations,
        "parameter_count": (
            support_aware_spatial_cg_parameter_count(trained)
            if is_spatial
            else spin_clebsch_tree_parameter_count(trained)
        ),
        "initial_backbone_sha256": pytree_sha256(initialized["backbone"]),
        "initial_raw_amplitudes_sha256": array_sha256(
            np.asarray(paired_baseline)
        ),
        "initial_spatial_to_morton_max_abs_error": initial_match,
        "initial_geometry_query_gradient_norm": query_gradient_norm,
        "training": training,
        "trial_wall_seconds": wall,
        "metrics": metrics,
        "energy_to_target": targets,
        "work_ledger": {
            "basis_size": len(data["basis"]),
            "route_tree_contractions_per_loss": len(data["basis"])
            * route_evaluations,
            "translation_gathers_per_loss": len(data["basis"])
            * int(indices.shape[0]),
            "hamiltonian_coo_entries_per_loss": len(data["h_values"]),
            "total_route_tree_contractions_over_updates": len(data["basis"])
            * route_evaluations
            * int(arm["steps"]),
        },
    }
    if is_spatial:
        result["geometry_query"] = np.asarray(
            trained["geometry_query"], dtype=np.float64
        ).tolist()
        result["route_diagnostics"] = _projected_component_diagnostics(
            trained,
            active_config,
            basis,
            indices,
            data,
            float(metrics["energy"]),
        )
    return result


def _summaries(cases: list[dict[str, object]]) -> dict[str, object]:
    def paired(
        by_arm: dict[str, list[dict[str, object]]],
        candidate: str,
        control: str,
    ) -> dict[str, object]:
        candidate_by_seed = {
            int(row["seed"]): row for row in by_arm[candidate]
        }
        control_by_seed = {int(row["seed"]): row for row in by_arm[control]}
        if set(candidate_by_seed) != set(control_by_seed):
            raise RuntimeError("paired Spatial-CG arms have different seeds")
        rows = []
        for seed in sorted(candidate_by_seed):
            candidate_row = candidate_by_seed[seed]
            control_row = control_by_seed[seed]
            candidate_error = float(
                candidate_row["metrics"]["energy_error_per_site"]
            )
            control_error = float(
                control_row["metrics"]["energy_error_per_site"]
            )
            rows.append(
                {
                    "seed": seed,
                    "candidate_minus_control_energy_error_per_site": (
                        candidate_error - control_error
                    ),
                    "candidate_over_control_energy_error_ratio": (
                        candidate_error / control_error
                    ),
                    "candidate_minus_control_variance_per_site": float(
                        candidate_row["metrics"]["variance_per_site"]
                        - control_row["metrics"]["variance_per_site"]
                    ),
                }
            )
        deltas = [
            row["candidate_minus_control_energy_error_per_site"] for row in rows
        ]
        ratios = [row["candidate_over_control_energy_error_ratio"] for row in rows]
        variance_deltas = [
            row["candidate_minus_control_variance_per_site"] for row in rows
        ]
        return {
            "candidate_arm": candidate,
            "control_arm": control,
            "seed_count": len(rows),
            "candidate_energy_win_count": sum(delta < 0.0 for delta in deltas),
            "median_paired_energy_error_per_site_delta": float(np.median(deltas)),
            "median_paired_energy_error_ratio": float(np.median(ratios)),
            "median_paired_variance_per_site_delta": float(
                np.median(variance_deltas)
            ),
            "per_seed": rows,
        }

    result: dict[str, object] = {}
    for case in cases:
        by_arm: dict[str, list[dict[str, object]]] = {}
        for trial in case["trials"]:
            by_arm.setdefault(str(trial["arm_id"]), []).append(trial)
        arms = {}
        for arm_id, trials in by_arm.items():
            arms[arm_id] = {
                "seed_count": len(trials),
                "median_energy_error_per_site": float(
                    np.median(
                        [trial["metrics"]["energy_error_per_site"] for trial in trials]
                    )
                ),
                "median_variance_per_site": float(
                    np.median(
                        [trial["metrics"]["variance_per_site"] for trial in trials]
                    )
                ),
                "median_wall_seconds": float(
                    np.median([trial["trial_wall_seconds"] for trial in trials])
                ),
            }
        result[str(case["case_id"])] = {
            "arms": arms,
            "paired_comparisons": {
                "spatial_vs_same_step_single_tree": paired(
                    by_arm, "spatial_cg", "single_tree"
                ),
                "spatial_vs_work_control_single_tree": paired(
                    by_arm, "spatial_cg", "single_tree_work_control"
                ),
                "physical_vs_scrambled_spatial_cg": paired(
                    by_arm, "spatial_cg", "scrambled_spatial_cg"
                ),
            },
        }
    return result


def run_spin_spatial_cg(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the N=8 oracle and paired square/triangular 4x4 pilot."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    runtime = numerical_runtime_identity()
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "exact_small_oracles": {
            lattice: support_aware_spatial_cg_oracle(
                lattice,
                seed=int(spec["oracle_seed"]),
                tangent_count=int(spec["oracle_tangent_count"]),
            )
            for lattice in ("square", "triangular")
        },
        "cases": [],
        "claim_boundary": (
            "Exact-support local CPU pilot. Four-by-four triangular is not "
            "K-compatible; energy and variance are architecture gates, not "
            "a QSL, 120-degree-order, scaling, or GPU-efficiency claim."
        ),
    }
    if not all(row["passed"] for row in payload["exact_small_oracles"].values()):
        raise RuntimeError("an N=8 exact architecture gate failed")
    atomic_json(result_path, payload)

    for case_spec in spec["cases"]:
        preparation_started = time.perf_counter()
        adapted = dict(case_spec)
        adapted["support"] = "fixed_m0"
        data = _prepare_case(adapted, "fixed_m0")
        model = spec["model"]
        physical_config, route_ledger = spatial_cg_route_spec(
            str(case_spec["geometry"]),
            int(case_spec["rows"]),
            int(case_spec["columns"]),
            feature_dim=int(model["feature_dim"]),
            maximum_intermediate_q=int(model["maximum_intermediate_q"]),
            initialization_scale=float(model["initialization_scale"]),
        )
        scrambled_config, scramble = _scrambled_config(
            physical_config, seed=int(case_spec["scramble_seed"])
        )
        case: dict[str, object] = {
            "case_id": str(case_spec["case_id"]),
            "geometry": data["geometry"],
            "rows": data["rows"],
            "columns": data["columns"],
            "j2_over_j1": data["j2_over_j1"],
            "projection": str(case_spec["projection"]),
            "basis_size": len(data["basis"]),
            "hamiltonian_nnz": int(data["hamiltonian"].nnz),
            "exact_energy_anchor": data["exact_energy"],
            "anchor_provenance": data["anchor_provenance"],
            "route_ledger": route_ledger,
            "scramble_permutation": scramble,
            "preparation_seconds": time.perf_counter() - preparation_started,
            "trials": [],
        }
        payload["cases"].append(case)
        atomic_json(result_path, payload)
        for seed in spec["optimization"]["seeds"]:
            for arm in spec["arms"]:
                trial = _train_trial(
                    arm,
                    seed=int(seed),
                    spec={**spec, "projection": case_spec["projection"]},
                    data=data,
                    physical_config=physical_config,
                    scrambled_config=scrambled_config,
                )
                case["trials"].append(trial)
                metrics = trial["metrics"]
                print(
                    f"{case['case_id']} {trial['arm_id']} seed={seed}: "
                    f"dE/N={metrics['energy_error_per_site']:.3e}, "
                    f"var/N={metrics['variance_per_site']:.3e}, "
                    f"wall={trial['trial_wall_seconds']:.2f}s",
                    flush=True,
                )
                atomic_json(result_path, payload)

    payload["summary"] = _summaries(payload["cases"])
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    # Seal the complete numerical payload while excluding the self-referential
    # field itself.  Intermediate ``running`` snapshots intentionally remain
    # unsealed so that a result seal can never be mistaken for a checkpoint.
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_spatial_cg"]
