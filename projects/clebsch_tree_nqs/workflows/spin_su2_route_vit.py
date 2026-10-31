"""Local variational pilot for the exact-SU(2) route-token Transformer.

The comparison starts every D=4 arm from the same warmed single-tree state.
It then spends the same number of additional Adam updates on

* the original single tree;
* a Transformer that changes only coherent route weights; and
* a Transformer that also conditions low-rank reduced CG tensors.

A wider single tree is recorded as a roughly parameter-matched control.  The
runner constructs the Hamiltonian in fixed ``M=0`` but performs no eigensolve;
stored exact-small energies are used only as scalar anchors.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import coo_matvec, reynolds_project_state
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spin_clebsch_tree import (
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
    su2_route_vit_backbone_parameter_count,
    su2_route_vit_controls,
    su2_route_vit_parameter_count,
)
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_local_variational_pilot"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("SU2 Route-ViT pilot requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    model = spec.get("model")
    optimization = spec.get("optimization")
    cases = spec.get("cases")
    if not isinstance(model, dict) or not isinstance(optimization, dict):
        raise TypeError("model and optimization must be objects")
    for field in (
        "base_feature_dim",
        "matched_feature_dim",
        "maximum_intermediate_q",
        "embedding_dim",
        "head_count",
        "depth",
        "mlp_ratio",
        "modulation_rank",
    ):
        if isinstance(model.get(field), bool) or not isinstance(model.get(field), int):
            raise TypeError(f"model.{field} must be an integer")
    if model["embedding_dim"] % model["head_count"]:
        raise ValueError("embedding_dim must be divisible by head_count")
    for field in ("initialization_scale", "modulation_scale"):
        if float(model.get(field, 0.0)) <= 0.0:
            raise ValueError(f"model.{field} must be positive")
    for field in ("warmup_steps", "comparison_steps", "record_every"):
        if isinstance(optimization.get(field), bool) or int(optimization.get(field, 0)) < 1:
            raise ValueError(f"optimization.{field} must be positive")
    for field in ("learning_rate", "route_learning_rate"):
        if float(optimization.get(field, 0.0)) <= 0.0:
            raise ValueError(f"optimization.{field} must be positive")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("optimization.seeds must be nonempty integers")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if not isinstance(case, dict) or case.get("geometry") not in {
            "square",
            "triangular",
        }:
            raise ValueError("every case must select square or triangular geometry")
        if (case.get("rows"), case.get("columns")) != (4, 4):
            raise ValueError("the local pilot is registered on 4x4")
        if case.get("projection") != "translation":
            raise ValueError("the local pilot requires translation projection")
        if "exact_energy_anchor" not in case:
            raise ValueError("every case needs a stored exact energy anchor")


def _loss_parts(data: dict[str, object], projection: str):
    indices = jnp.asarray(data["tables"][projection], dtype=jnp.int32)
    rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def rayleigh(raw: jax.Array) -> jax.Array:
        state = reynolds_project_state(raw, indices)
        action = coo_matvec(rows, columns, values, state)
        return jnp.vdot(state, action).real / jnp.vdot(state, state).real

    return rayleigh


def _multiscale_spatial_config(
    spatial: SupportAwareSpatialCGConfig,
    geometry: str,
) -> tuple[SupportAwareSpatialCGConfig, dict[str, object]]:
    """Replace mean-distance route tokens by scale-resolved bond histograms."""

    rows = columns = 4
    if geometry == "square":
        bond_shells = {
            "J1": square_j1j2_bonds(
                rows, columns, j1=1.0, j2=0.0, periodic=True
            ),
            "J2": square_j1j2_bonds(
                rows, columns, j1=0.0, j2=1.0, periodic=True
            ),
        }
    elif geometry == "triangular":
        bond_shells = {
            "J1": triangular_j1j2_bonds(
                rows, columns, j1=1.0, j2=0.0, periodic=True
            ),
            "J2": triangular_j1j2_bonds(
                rows, columns, j1=0.0, j2=1.0, periodic=True
            ),
        }
    else:  # pragma: no cover - caller is config validated
        raise ValueError("geometry must be square or triangular")
    orders = dict(zip(spatial.route_names, spatial.route_orders, strict=True))
    features = multiscale_bond_route_features(
        spatial.backbone.tree, orders, bond_shells
    )
    enhanced = SupportAwareSpatialCGConfig(
        backbone=spatial.backbone,
        route_names=spatial.route_names,
        route_orders=spatial.route_orders,
        route_features=features.standardized,
        feature_names=features.labels,
        reference_route=spatial.reference_route,
    )
    return enhanced, {
        "feature_mode": "multiscale_bond_locality_histogram",
        "feature_names": list(features.labels),
        "raw_histograms": [list(row) for row in features.raw_histograms],
        "standardized": [list(row) for row in features.standardized],
        "unique_token_count": len(set(features.standardized)),
    }


def _train(
    params: object,
    raw_amplitudes,
    rayleigh,
    *,
    steps: int,
    learning_rate: float,
    record_every: int,
):
    def loss(candidate):
        return rayleigh(raw_amplitudes(candidate))

    return train_exact_adam(
        params,
        loss_fn=loss,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
    )


def _route_diagnostics(params: object, config: SU2RouteViTConfig) -> dict[str, object]:
    coefficients, controls = su2_route_vit_controls(params, config)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    squared = coefficients**2
    return {
        "route_names": list(config.spatial.route_names),
        "reference_route": config.spatial.route_names[
            config.spatial.reference_route
        ],
        "coefficients": coefficients.tolist(),
        "coefficient_participation_ratio": float(
            squared.sum() ** 2 / max(float(np.sum(squared**2)), 1.0e-30)
        ),
        "modulation_controls": controls.tolist(),
        "modulation_control_rms": float(
            np.sqrt(np.mean(controls**2)) if controls.size else 0.0
        ),
    }


def _run_trial(
    spec: dict[str, object],
    case_spec: dict[str, object],
    data: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    model = spec["model"]
    optimization = spec["optimization"]
    projection = str(case_spec["projection"])
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    rayleigh = _loss_parts(data, projection)

    base_spatial, ledger = spatial_cg_route_spec(
        str(case_spec["geometry"]),
        4,
        4,
        feature_dim=int(model["base_feature_dim"]),
        maximum_intermediate_q=int(model["maximum_intermediate_q"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    base_spatial, token_ledger = _multiscale_spatial_config(
        base_spatial, str(case_spec["geometry"])
    )
    ledger = {**ledger, "route_vit_tokens": token_ledger}
    reference_order = jnp.asarray(
        base_spatial.route_orders[base_spatial.reference_route], dtype=jnp.int32
    )

    def base_raw(candidate):
        return spin_clebsch_tree_amplitudes(
            candidate, basis[:, reference_order], base_spatial.backbone
        )

    initial = init_spin_clebsch_tree(
        jax.random.PRNGKey(seed), base_spatial.backbone
    )
    warmed, warm_trace = _train(
        initial,
        base_raw,
        rayleigh,
        steps=int(optimization["warmup_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    warm_metrics = _metrics(
        np.asarray(base_raw(warmed)), data, projection=projection
    )

    continued, continued_trace = _train(
        warmed,
        base_raw,
        rayleigh,
        steps=int(optimization["comparison_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    arms: list[dict[str, object]] = [
        {
            "arm_id": "single_tree_d4",
            "architecture": "single_clebsch_tree",
            "parameter_count": spin_clebsch_tree_parameter_count(continued),
            "warm_metrics": warm_metrics,
            "metrics": _metrics(
                np.asarray(base_raw(continued)), data, projection=projection
            ),
            "warm_training": warm_trace,
            "comparison_training": continued_trace,
        }
    ]

    for rank, arm_id in (
        (0, "route_vit_coefficients_d4"),
        (int(model["modulation_rank"]), "route_vit_conditioned_d4"),
    ):
        route_config = SU2RouteViTConfig(
            spatial=base_spatial,
            embedding_dim=int(model["embedding_dim"]),
            head_count=int(model["head_count"]),
            depth=int(model["depth"]),
            mlp_ratio=int(model["mlp_ratio"]),
            modulation_rank=rank,
            modulation_scale=float(model["modulation_scale"]),
            initialization_scale=float(model["initialization_scale"]),
        )
        initialized = init_su2_route_vit(
            jax.random.PRNGKey(seed + 1000 + rank), route_config
        )
        initialized = {**initialized, "backbone": warmed}

        def route_raw(candidate):
            return su2_route_vit_amplitudes(candidate, basis, route_config)

        nested_error = float(
            np.max(
                np.abs(
                    np.asarray(route_raw(initialized))
                    - np.asarray(base_raw(warmed))
                )
            )
        )
        trained, trace = _train(
            initialized,
            route_raw,
            rayleigh,
            steps=int(optimization["comparison_steps"]),
            learning_rate=float(optimization["route_learning_rate"]),
            record_every=int(optimization["record_every"]),
        )
        arms.append(
            {
                "arm_id": arm_id,
                "architecture": "exact_su2_route_token_transformer",
                "modulation_rank": rank,
                "parameter_count": su2_route_vit_parameter_count(trained),
                "backbone_parameter_count": su2_route_vit_backbone_parameter_count(
                    trained
                ),
                "nested_warm_state_max_abs_error": nested_error,
                "metrics": _metrics(
                    np.asarray(route_raw(trained)), data, projection=projection
                ),
                "comparison_training": trace,
                "route_diagnostics": _route_diagnostics(trained, route_config),
            }
        )

    matched_spatial, _matched_ledger = spatial_cg_route_spec(
        str(case_spec["geometry"]),
        4,
        4,
        feature_dim=int(model["matched_feature_dim"]),
        maximum_intermediate_q=int(model["maximum_intermediate_q"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    matched_order = jnp.asarray(
        matched_spatial.route_orders[matched_spatial.reference_route],
        dtype=jnp.int32,
    )

    def matched_raw(candidate):
        return spin_clebsch_tree_amplitudes(
            candidate, basis[:, matched_order], matched_spatial.backbone
        )

    matched = init_spin_clebsch_tree(
        jax.random.PRNGKey(seed), matched_spatial.backbone
    )
    matched, matched_trace = _train(
        matched,
        matched_raw,
        rayleigh,
        steps=int(optimization["warmup_steps"])
        + int(optimization["comparison_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    arms.append(
        {
            "arm_id": "single_tree_matched_width",
            "architecture": "single_clebsch_tree",
            "feature_dim": int(model["matched_feature_dim"]),
            "parameter_count": spin_clebsch_tree_parameter_count(matched),
            "metrics": _metrics(
                np.asarray(matched_raw(matched)), data, projection=projection
            ),
            "training": matched_trace,
        }
    )
    return {
        "seed": seed,
        "route_ledger": ledger,
        "warmup_steps": int(optimization["warmup_steps"]),
        "comparison_steps": int(optimization["comparison_steps"]),
        "arms": arms,
        "wall_seconds": time.perf_counter() - started,
    }


def _summary(cases: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for case in cases:
        arm_ids = [row["arm_id"] for row in case["trials"][0]["arms"]]
        by_arm = {}
        for arm_id in arm_ids:
            rows = [
                next(arm for arm in trial["arms"] if arm["arm_id"] == arm_id)
                for trial in case["trials"]
            ]
            errors = np.asarray(
                [row["metrics"]["energy_error_per_site"] for row in rows]
            )
            energies = np.asarray([row["metrics"]["energy"] for row in rows])
            by_arm[arm_id] = {
                "parameter_count": int(rows[0]["parameter_count"]),
                "median_energy": float(np.median(energies)),
                "median_energy_error_per_site": float(np.median(errors)),
                "minimum_energy_error_per_site": float(np.min(errors)),
                "seed_count": len(rows),
            }
        baseline = by_arm["single_tree_d4"]["median_energy"]
        for row in by_arm.values():
            row["median_energy_gain_below_same_work_d4"] = (
                baseline - row["median_energy"]
            )
        paired = {}
        baseline_rows = {
            int(trial["seed"]): next(
                arm
                for arm in trial["arms"]
                if arm["arm_id"] == "single_tree_d4"
            )
            for trial in case["trials"]
        }
        for arm_id in arm_ids:
            if arm_id == "single_tree_d4":
                continue
            rows = []
            for trial in case["trials"]:
                seed = int(trial["seed"])
                candidate = next(
                    arm for arm in trial["arms"] if arm["arm_id"] == arm_id
                )
                baseline_row = baseline_rows[seed]
                delta = float(candidate["metrics"]["energy"]) - float(
                    baseline_row["metrics"]["energy"]
                )
                baseline_error = float(
                    baseline_row["metrics"]["energy_error_per_site"]
                )
                candidate_error = float(
                    candidate["metrics"]["energy_error_per_site"]
                )
                rows.append(
                    {
                        "seed": seed,
                        "candidate_minus_d4_energy": delta,
                        "candidate_over_d4_energy_error_ratio": (
                            candidate_error / baseline_error
                            if baseline_error > 0.0
                            else None
                        ),
                    }
                )
            paired[arm_id] = {
                "energy_win_count": sum(
                    row["candidate_minus_d4_energy"] < 0.0 for row in rows
                ),
                "seed_count": len(rows),
                "median_candidate_minus_d4_energy": float(
                    np.median(
                        [row["candidate_minus_d4_energy"] for row in rows]
                    )
                ),
                "median_energy_error_ratio": float(
                    np.median(
                        [
                            row["candidate_over_d4_energy_error_ratio"]
                            for row in rows
                            if row["candidate_over_d4_energy_error_ratio"] is not None
                        ]
                    )
                ),
                "per_seed": rows,
            }
        result[case["case_id"]] = {"arms": by_arm, "paired_vs_d4": paired}
    return result


def run_spin_su2_route_vit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the local exact-SU(2) Route-ViT pilot."""

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
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "running",
        "created_utc": now,
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": False,
        "cases": [],
        "claim_boundary": (
            "Local 4x4 exact-support optimization of a route-token Transformer. "
            "It establishes only a variational small-system comparison; it is "
            "not a thermodynamic, QSL, or scaling result."
        ),
    }
    atomic_json(result_path, payload)
    for case_spec in spec["cases"]:
        prepared = dict(case_spec)
        prepared["support"] = "fixed_m0"
        data = _prepare_case(prepared, "fixed_m0")
        case = {
            "case_id": str(case_spec["case_id"]),
            "geometry": str(case_spec["geometry"]),
            "j2_over_j1": float(case_spec["j2_over_j1"]),
            "trials": [],
        }
        payload["cases"].append(case)
        for seed in spec["optimization"]["seeds"]:
            trial = _run_trial(spec, case_spec, data, seed=int(seed))
            case["trials"].append(trial)
            print(
                case["case_id"],
                f"seed={seed}",
                ", ".join(
                    f"{arm['arm_id']}={arm['metrics']['energy']:.9f}"
                    for arm in trial["arms"]
                ),
                flush=True,
            )
            atomic_json(result_path, payload)
    payload["summary"] = _summary(payload["cases"])
    triangular = next(
        (
            value
            for key, value in payload["summary"].items()
            if key.startswith("triangular_")
        ),
        None,
    )
    conditioned = (
        None
        if triangular is None
        else triangular["paired_vs_d4"]["route_vit_conditioned_d4"]
    )
    if conditioned is not None and conditioned["energy_win_count"] == conditioned["seed_count"]:
        payload["promotion_decision"] = (
            "promote_conditioned_route_vit_to_triangular_n36_sampled_training; "
            "retain_square_as_negative_control"
        )
    else:
        payload["promotion_decision"] = (
            "do_not_promote_without_additional_triangular_replication"
        )
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["runtime_identity_sha256"] = canonical_sha256(
        payload["runtime_identity"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_su2_route_vit"]
