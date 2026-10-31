"""Zero-parameter physical-route extension of triangular SU(2) Route-ViT."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.physical_route_banks import (
    triangular_dimer_complete_route_orders,
)
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spin_clebsch_tree import (
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.models.su2_route_vit import (
    su2_route_vit_amplitudes,
    su2_route_vit_controls,
)
from su2_nqs.models.su2_route_vit_causal_controls import (
    su2_route_local_gate_amplitudes,
    su2_route_local_gate_controls,
)
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit import (
    _loss_parts,
    _summary,
    _train,
)
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_causal_scan import (
    _route_arm,
    _route_config,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_physical_route_extension"


def _validate(spec: dict[str, object]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
    ):
        raise ValueError("invalid physical-route scan identity")
    model = spec.get("model")
    optimization = spec.get("optimization")
    cases = spec.get("cases")
    if not isinstance(model, dict) or not isinstance(optimization, dict):
        raise TypeError("model and optimization must be objects")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if (
            not isinstance(case, dict)
            or case.get("geometry") != "triangular"
            or (case.get("rows"), case.get("columns")) != (4, 4)
            or case.get("projection") != "translation"
            or "exact_energy_anchor" not in case
        ):
            raise ValueError("physical-route cases must be triangular 4x4 anchors")


def _spatial_banks(
    model: dict[str, object],
) -> tuple[
    SupportAwareSpatialCGConfig,
    SupportAwareSpatialCGConfig,
    dict[str, object],
]:
    base, _ledger = spatial_cg_route_spec(
        "triangular",
        4,
        4,
        feature_dim=int(model["base_feature_dim"]),
        maximum_intermediate_q=int(model["maximum_intermediate_q"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    complete_orders = triangular_dimer_complete_route_orders(4, 4)
    shells = {
        "J1": triangular_j1j2_bonds(
            4, 4, j1=1.0, j2=0.0, periodic=True
        ),
        "J2": triangular_j1j2_bonds(
            4, 4, j1=0.0, j2=1.0, periodic=True
        ),
    }
    features = multiscale_bond_route_features(
        base.backbone.tree, complete_orders, shells
    )
    names = tuple(complete_orders)
    rows_by_name = dict(zip(names, features.standardized, strict=True))
    base_names = tuple(base.route_names)
    reference_name = base.route_names[base.reference_route]
    base_joint = SupportAwareSpatialCGConfig(
        backbone=base.backbone,
        route_names=base_names,
        route_orders=base.route_orders,
        route_features=tuple(rows_by_name[name] for name in base_names),
        feature_names=features.labels,
        reference_route=base_names.index(reference_name),
    )
    complete = SupportAwareSpatialCGConfig(
        backbone=base.backbone,
        route_names=names,
        route_orders=tuple(complete_orders.values()),
        route_features=features.standardized,
        feature_names=features.labels,
        reference_route=names.index(reference_name),
    )
    return base_joint, complete, {
        "base_route_names": list(base_names),
        "complete_route_names": list(names),
        "added_route": "diagonal_dimers",
        "added_trainable_parameters": 0,
        "joint_feature_standardization": True,
        "feature_names": list(features.labels),
        "raw_histograms": [list(row) for row in features.raw_histograms],
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
    base_spatial, complete_spatial, ledger = _spatial_banks(model)
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
    warm_raw = np.asarray(base_raw(warmed))
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
            "warm_metrics": _metrics(warm_raw, data, projection=projection),
            "metrics": _metrics(
                np.asarray(base_raw(continued)), data, projection=projection
            ),
            "warm_training": warm_trace,
            "comparison_training": continued_trace,
        }
    ]
    common = {
        "warmed": warmed,
        "warm_raw": warm_raw,
        "basis": basis,
        "data": data,
        "projection": projection,
        "rayleigh": rayleigh,
        "seed": seed + 1000 + int(model["modulation_rank"]),
        "steps": int(optimization["comparison_steps"]),
        "learning_rate": float(optimization["route_learning_rate"]),
        "record_every": int(optimization["record_every"]),
        "token_alignment": "physical",
    }
    for arm_id, spatial, amplitudes_fn, controls_fn, mixer in (
        (
            "route_vit_base_8_routes_d4",
            base_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
        ),
        (
            "route_vit_dimer_complete_9_routes_d4",
            complete_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
        ),
        (
            "route_local_dimer_complete_9_routes_d4",
            complete_spatial,
            su2_route_local_gate_amplitudes,
            su2_route_local_gate_controls,
            "route_local_gate",
        ),
    ):
        arms.append(
            _route_arm(
                arm_id=arm_id,
                config=_route_config(
                    spatial, model, int(model["modulation_rank"])
                ),
                amplitudes_fn=amplitudes_fn,
                controls_fn=controls_fn,
                route_mixer=mixer,
                **common,
            )
        )
    counts = {
        arm["parameter_count"]
        for arm in arms
        if arm["arm_id"].startswith("route_")
    }
    if len(counts) != 1:
        raise RuntimeError("physical route comparison is not parameter matched")
    return {
        "seed": seed,
        "route_ledger": ledger,
        "warmup_steps": int(optimization["warmup_steps"]),
        "comparison_steps": int(optimization["comparison_steps"]),
        "arms": arms,
        "wall_seconds": time.perf_counter() - started,
    }


def _physical_comparisons(cases: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    base_id = "route_vit_base_8_routes_d4"
    for case in cases:
        rows = []
        for trial in case["trials"]:
            by_id = {arm["arm_id"]: arm for arm in trial["arms"]}
            base_energy = float(by_id[base_id]["metrics"]["energy"])
            candidate = by_id["route_vit_dimer_complete_9_routes_d4"]
            rows.append(
                {
                    "seed": int(trial["seed"]),
                    "complete_minus_base_energy": float(
                        candidate["metrics"]["energy"]
                    )
                    - base_energy,
                    "added_trainable_parameters": int(
                        candidate["parameter_count"] - by_id[base_id]["parameter_count"]
                    ),
                }
            )
        output[str(case["case_id"])] = {
            "complete_bank_win_count": sum(
                row["complete_minus_base_energy"] < 0.0 for row in rows
            ),
            "seed_count": len(rows),
            "median_complete_minus_base_energy": float(
                np.median([row["complete_minus_base_energy"] for row in rows])
            ),
            "per_seed": rows,
        }
    return output


def run_spin_su2_route_vit_physical_routes(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
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
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": False,
        "cases": [],
        "claim_boundary": (
            "Finite triangular-4x4 zero-parameter route-bank comparison. "
            "It tests dimer-orientation completeness, not a thermodynamic phase claim."
        ),
    }
    atomic_json(result_path, payload)
    for case_spec in spec["cases"]:
        prepared = {**case_spec, "support": "fixed_m0"}
        data = _prepare_case(prepared, "fixed_m0")
        case = {
            "case_id": str(case_spec["case_id"]),
            "geometry": "triangular",
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
    payload["physical_route_comparisons"] = _physical_comparisons(payload["cases"])
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["runtime_identity_sha256"] = canonical_sha256(payload["runtime_identity"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_su2_route_vit_physical_routes"]
