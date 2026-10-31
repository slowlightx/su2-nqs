"""Frozen-backbone capacity diagnosis for the support-aware route bank.

This follow-up separates three questions that joint training confounds:

1. does selecting a different single route help;
2. does an unconstrained coherent route span contain a lower-energy state;
3. can the two shell-locality query parameters recover that headroom?

The single-tree backbone is deterministically replayed from the sealed pilot;
no eigensolve or new energy target is performed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.clebsch_tree_exact import (
    coo_matvec,
    reynolds_project_state,
)
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.models.spin_clebsch_tree import spin_clebsch_tree_amplitudes
from su2_nqs.models.support_aware_spatial_cg import (
    SupportAwareSpatialCGConfig,
    init_support_aware_spatial_cg,
    support_aware_spatial_cg_component_matrix,
)
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_spatial_cg import (
    _scrambled_config,
    _validate,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "support_aware_spatial_cg_frozen_backbone_capacity"


def _projected_components(
    backbone: object,
    config: SupportAwareSpatialCGConfig,
    basis: jax.Array,
    indices: jax.Array,
) -> np.ndarray:
    params = {
        "backbone": backbone,
        "geometry_query": jnp.zeros(
            (config.n_support_features,), dtype=jnp.float64
        ),
    }
    raw = np.asarray(
        support_aware_spatial_cg_component_matrix(params, basis, config),
        dtype=np.float64,
    )
    return np.column_stack(
        [
            np.asarray(reynolds_project_state(raw[:, route], indices))
            for route in range(config.n_routes)
        ]
    )


def _route_span_diagnosis(
    components: np.ndarray,
    config: SupportAwareSpatialCGConfig,
    data: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    norms = np.linalg.norm(components, axis=0)
    if np.any(norms <= 1.0e-14):
        raise RuntimeError("a projected route component has zero norm")
    normalized = components / norms
    route_energies = np.asarray(
        [
            np.vdot(normalized[:, route], data["hamiltonian"] @ normalized[:, route]).real
            for route in range(config.n_routes)
        ],
        dtype=np.float64,
    )
    reference_energy = float(route_energies[config.reference_route])

    left, singular, _ = np.linalg.svd(components, full_matrices=False)
    ratios = singular / singular[0]
    retained = ratios > 1.0e-10
    span = left[:, retained]
    reduced = span.T @ (data["hamiltonian"] @ span)
    reduced = 0.5 * (reduced + reduced.T)
    ritz_energy = float(np.linalg.eigvalsh(reduced)[0])

    matrix = jnp.asarray(components, dtype=jnp.float64)
    features = jnp.asarray(config.route_features, dtype=jnp.float64)
    h_rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    h_columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    h_values = jnp.asarray(data["h_values"], dtype=jnp.float64)
    reference = int(config.reference_route)

    def query_loss(query):
        residual = jnp.tanh(features @ query) / jnp.sqrt(
            jnp.asarray(config.n_routes, dtype=jnp.float64)
        )
        coefficients = residual.at[reference].add(1.0 - jnp.sum(residual))
        state = matrix @ coefficients
        action = coo_matvec(h_rows, h_columns, h_values, state)
        return jnp.vdot(state, action).real / jnp.vdot(state, state).real

    value_gradient = jax.jit(jax.value_and_grad(query_loss))
    zero = np.zeros(config.n_support_features, dtype=np.float64)
    value_gradient(jnp.asarray(zero))[0].block_until_ready()
    rng = np.random.default_rng(seed)
    starts = [zero]
    for scale in (0.1, 0.5, 1.5):
        starts.extend(
            (
                scale * np.ones(config.n_support_features),
                -scale * np.ones(config.n_support_features),
                scale * rng.normal(size=config.n_support_features),
            )
        )

    def objective(query: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_gradient(jnp.asarray(query, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    candidates = []
    for start in starts:
        optimized = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-8.0, 8.0)] * config.n_support_features,
            options={"maxiter": 500, "ftol": 1.0e-14, "gtol": 1.0e-11},
        )
        candidates.append(optimized)
    best = min(candidates, key=lambda result: float(result.fun))
    query = np.asarray(best.x, dtype=np.float64)
    residual = np.tanh(np.asarray(config.route_features) @ query) / np.sqrt(
        config.n_routes
    )
    coefficients = residual.copy()
    coefficients[reference] += 1.0 - residual.sum()
    squared = coefficients**2
    participation = float(squared.sum() ** 2 / np.sum(squared**2))
    return {
        "route_names": list(config.route_names),
        "reference_route": config.route_names[reference],
        "reference_energy": reference_energy,
        "single_route_energies": {
            name: float(value)
            for name, value in zip(config.route_names, route_energies, strict=True)
        },
        "best_single_route_energy": float(np.min(route_energies)),
        "best_single_route_name": config.route_names[int(np.argmin(route_energies))],
        "span_singular_value_ratios": ratios.tolist(),
        "span_rank_at_1e_minus_10": int(np.count_nonzero(retained)),
        "unconstrained_route_span_ritz_energy": ritz_energy,
        "unconstrained_ritz_gain_below_reference": reference_energy - ritz_energy,
        "optimized_two_feature_query_energy": float(best.fun),
        "optimized_two_feature_query": query.tolist(),
        "optimized_route_coefficients": coefficients.tolist(),
        "optimized_coefficient_participation_ratio": participation,
        "two_feature_gain_below_reference": reference_energy - float(best.fun),
        "fraction_of_ritz_gain_recovered": (
            (reference_energy - float(best.fun)) / (reference_energy - ritz_energy)
            if reference_energy - ritz_energy > 1.0e-14
            else 0.0
        ),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimizer_start_count": len(starts),
    }


def run_spin_spatial_cg_capacity(
    config_path: Path,
    pilot_result_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Replay paired single trees and resolve physical/scrambled route headroom."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    pilot = json.loads(pilot_result_path.read_text())
    if pilot.get("status") != "complete":
        raise ValueError("pilot result must be complete")
    pilot_without_seal = dict(pilot)
    pilot_result_sha256 = pilot_without_seal.pop("result_sha256", None)
    if (
        not isinstance(pilot_result_sha256, str)
        or pilot_result_sha256 != canonical_sha256(pilot_without_seal)
    ):
        raise ValueError("pilot result payload seal is missing or broken")
    if pilot.get("config_sha256") != canonical_sha256(spec):
        raise ValueError("pilot result does not match the supplied config")
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
    single_steps = next(
        int(arm["steps"])
        for arm in spec["arms"]
        if arm["arm_id"] == "single_tree"
    )
    pilot_cases = {case["case_id"]: case for case in pilot["cases"]}
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "pilot_result_path": str(pilot_result_path),
        "pilot_result_sha256": pilot_result_sha256,
        "pilot_config_sha256": pilot["config_sha256"],
        "pilot_source_manifest_sha256": pilot["source_manifest_sha256"],
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "cases": [],
        "claim_boundary": (
            "Frozen-backbone exact-support capacity diagnosis. Ritz energies "
            "are variational upper bounds within a small route span, not a "
            "scalable trainable ansatz or time-to-target result."
        ),
    }
    atomic_json(result_path, payload)
    for case_spec in spec["cases"]:
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
        scrambled_config, permutation = _scrambled_config(
            physical_config, seed=int(case_spec["scramble_seed"])
        )
        basis = jnp.asarray(data["basis"], dtype=jnp.int32)
        indices = jnp.asarray(
            data["tables"][str(case_spec["projection"])], dtype=jnp.int32
        )
        reference_order = jnp.asarray(
            physical_config.route_orders[physical_config.reference_route],
            dtype=jnp.int32,
        )
        h_rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
        h_columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
        h_values = jnp.asarray(data["h_values"], dtype=jnp.float64)

        def raw_amplitudes(candidate):
            return spin_clebsch_tree_amplitudes(
                candidate,
                basis[:, reference_order],
                physical_config.backbone,
            )

        def loss(candidate):
            state = reynolds_project_state(raw_amplitudes(candidate), indices)
            action = coo_matvec(h_rows, h_columns, h_values, state)
            return jnp.vdot(state, action).real / jnp.vdot(state, state).real

        case: dict[str, object] = {
            "case_id": str(case_spec["case_id"]),
            "geometry": str(case_spec["geometry"]),
            "j2_over_j1": float(case_spec["j2_over_j1"]),
            "route_ledger": route_ledger,
            "scramble_permutation": permutation,
            "trials": [],
        }
        payload["cases"].append(case)
        for seed in spec["optimization"]["seeds"]:
            started = time.perf_counter()
            initialized = init_support_aware_spatial_cg(
                jax.random.PRNGKey(int(seed)), physical_config
            )["backbone"]
            trained, training = train_exact_adam(
                initialized,
                loss_fn=loss,
                steps=single_steps,
                learning_rate=float(spec["optimization"]["learning_rate"]),
                record_every=int(spec["optimization"]["record_every"]),
            )
            raw = np.asarray(raw_amplitudes(trained), dtype=np.float64)
            baseline_metrics = _metrics(
                raw, data, projection=str(case_spec["projection"])
            )
            pilot_trial = next(
                trial
                for trial in pilot_cases[str(case_spec["case_id"])]["trials"]
                if trial["arm_id"] == "single_tree" and trial["seed"] == seed
            )
            replay_error = abs(
                float(baseline_metrics["energy"])
                - float(pilot_trial["metrics"]["energy"])
            )
            if replay_error > 2.0e-11:
                raise RuntimeError("single-tree deterministic replay disagrees with pilot")
            physical_components = _projected_components(
                trained, physical_config, basis, indices
            )
            scrambled_components = _projected_components(
                trained, scrambled_config, basis, indices
            )
            physical = _route_span_diagnosis(
                physical_components,
                physical_config,
                data,
                seed=1000 + int(seed),
            )
            scrambled = _route_span_diagnosis(
                scrambled_components,
                scrambled_config,
                data,
                seed=2000 + int(seed),
            )
            trial = {
                "seed": int(seed),
                "single_tree_steps": single_steps,
                "single_tree_replay_energy_error": replay_error,
                "single_tree_metrics": baseline_metrics,
                "single_tree_training": training,
                "physical_routes": physical,
                "scrambled_routes": scrambled,
                "physical_minus_scrambled_ritz_energy": (
                    physical["unconstrained_route_span_ritz_energy"]
                    - scrambled["unconstrained_route_span_ritz_energy"]
                ),
                "physical_minus_scrambled_two_feature_energy": (
                    physical["optimized_two_feature_query_energy"]
                    - scrambled["optimized_two_feature_query_energy"]
                ),
                "wall_seconds": time.perf_counter() - started,
            }
            case["trials"].append(trial)
            print(
                f"{case['case_id']} seed={seed}: "
                f"baseline={baseline_metrics['energy']:.8f}, "
                f"physical query={physical['optimized_two_feature_query_energy']:.8f}, "
                f"physical Ritz={physical['unconstrained_route_span_ritz_energy']:.8f}, "
                f"scrambled Ritz={scrambled['unconstrained_route_span_ritz_energy']:.8f}",
                flush=True,
            )
            atomic_json(result_path, payload)

    summary = {}
    for case in payload["cases"]:
        trials = case["trials"]
        ritz_deltas = [
            row["physical_minus_scrambled_ritz_energy"] for row in trials
        ]
        query_deltas = [
            row["physical_minus_scrambled_two_feature_energy"] for row in trials
        ]
        summary[case["case_id"]] = {
            "median_physical_two_feature_gain_below_reference": float(
                np.median(
                    [row["physical_routes"]["two_feature_gain_below_reference"] for row in trials]
                )
            ),
            "median_physical_ritz_gain_below_reference": float(
                np.median(
                    [row["physical_routes"]["unconstrained_ritz_gain_below_reference"] for row in trials]
                )
            ),
            "median_fraction_of_physical_ritz_gain_recovered": float(
                np.median(
                    [row["physical_routes"]["fraction_of_ritz_gain_recovered"] for row in trials]
                )
            ),
            "median_physical_minus_scrambled_ritz_energy": float(
                np.median(ritz_deltas)
            ),
            "physical_ritz_win_count_over_scrambled": sum(
                delta < 0.0 for delta in ritz_deltas
            ),
            "median_physical_minus_scrambled_two_feature_energy": float(
                np.median(query_deltas)
            ),
            "physical_two_feature_win_count_over_scrambled": sum(
                delta < 0.0 for delta in query_deltas
            ),
            "seed_count": len(trials),
        }
    payload["summary"] = summary
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_spatial_cg_capacity"]
