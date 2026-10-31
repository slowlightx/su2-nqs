"""Causal locality controls and coupling scan for exact-SU(2) Route-ViT.

This workflow extends the archived local pilot without mutating it.  Every
comparison starts from the same warmed D=4 ClebschTree and includes two
matched causal controls:

* correct route tokens with no cross-route read (route-local gate), and
* global attention with the physical token-to-route association scrambled.

The Hamiltonian is constructed in fixed M=0, but the workflow performs no
eigensolve.  Finite-torus exact energies must be registered as scalar anchors;
thermodynamic references are carried separately and never used as targets.
"""

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
from su2_nqs.lattices.route_vit_control_features import (
    maximally_mismatched_cyclic_permutation,
    permute_route_features,
)
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
from su2_nqs.models.su2_route_vit_causal_controls import (
    su2_route_local_gate_amplitudes,
    su2_route_local_gate_controls,
)
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit import (
    _loss_parts,
    _multiscale_spatial_config,
    _summary,
    _train,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_causal_coupling_scan"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("causal scan requires schema_version=1 and spin")
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
        if isinstance(optimization.get(field), bool) or int(
            optimization.get(field, 0)
        ) < 1:
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
            raise ValueError("the exact causal scan is registered on 4x4")
        if case.get("projection") != "translation":
            raise ValueError("the causal scan requires translation projection")
        if "exact_energy_anchor" not in case:
            raise ValueError("every case needs a finite-4x4 exact energy anchor")


def _scrambled_spatial_config(
    spatial: SupportAwareSpatialCGConfig,
) -> tuple[SupportAwareSpatialCGConfig, dict[str, object]]:
    permutation = maximally_mismatched_cyclic_permutation(spatial.route_features)
    scrambled = SupportAwareSpatialCGConfig(
        backbone=spatial.backbone,
        route_names=spatial.route_names,
        route_orders=spatial.route_orders,
        route_features=permute_route_features(
            spatial.route_features, permutation
        ),
        feature_names=spatial.feature_names,
        reference_route=spatial.reference_route,
    )
    values = np.asarray(spatial.route_features, dtype=np.float64)
    altered = np.asarray(scrambled.route_features, dtype=np.float64)
    return scrambled, {
        "permutation_target_to_source": list(permutation),
        "target_route_to_source_route": {
            spatial.route_names[target]: spatial.route_names[source]
            for target, source in enumerate(permutation)
        },
        "token_multiset_preserved": True,
        "fixed_route_count": sum(
            target == source for target, source in enumerate(permutation)
        ),
        "feature_mismatch_l2": float(np.linalg.norm(values - altered)),
    }


def _route_config(
    spatial: SupportAwareSpatialCGConfig,
    model: dict[str, object],
    rank: int,
) -> SU2RouteViTConfig:
    return SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=int(model["depth"]),
        mlp_ratio=int(model["mlp_ratio"]),
        modulation_rank=rank,
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )


def _route_diagnostics(
    params: object,
    config: SU2RouteViTConfig,
    controls_fn,
    *,
    route_mixer: str,
    token_alignment: str,
) -> dict[str, object]:
    coefficients, controls = controls_fn(params, config)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    squared = coefficients**2
    return {
        "route_names": list(config.spatial.route_names),
        "reference_route": config.spatial.route_names[
            config.spatial.reference_route
        ],
        "route_mixer": route_mixer,
        "token_alignment": token_alignment,
        "coefficients": coefficients.tolist(),
        "coefficient_participation_ratio": float(
            squared.sum() ** 2 / max(float(np.sum(squared**2)), 1.0e-30)
        ),
        "modulation_controls": controls.tolist(),
        "modulation_control_rms": float(
            np.sqrt(np.mean(controls**2)) if controls.size else 0.0
        ),
    }


def _route_arm(
    *,
    arm_id: str,
    warmed: object,
    warm_raw: np.ndarray,
    basis: jax.Array,
    data: dict[str, object],
    projection: str,
    rayleigh,
    config: SU2RouteViTConfig,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    amplitudes_fn,
    controls_fn,
    route_mixer: str,
    token_alignment: str,
) -> dict[str, object]:
    initialized = init_su2_route_vit(jax.random.PRNGKey(seed), config)
    initialized = {**initialized, "backbone": warmed}

    def raw(candidate):
        return amplitudes_fn(candidate, basis, config)

    nested_error = float(
        np.max(np.abs(np.asarray(raw(initialized)) - warm_raw))
    )
    trained, trace = _train(
        initialized,
        raw,
        rayleigh,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
    )
    return {
        "arm_id": arm_id,
        "architecture": "exact_su2_route_conditioned_clebsch_tree",
        "route_mixer": route_mixer,
        "token_alignment": token_alignment,
        "modulation_rank": config.modulation_rank,
        "parameter_count": su2_route_vit_parameter_count(trained),
        "backbone_parameter_count": su2_route_vit_backbone_parameter_count(
            trained
        ),
        "nested_warm_state_max_abs_error": nested_error,
        "metrics": _metrics(np.asarray(raw(trained)), data, projection=projection),
        "comparison_training": trace,
        "route_diagnostics": _route_diagnostics(
            trained,
            config,
            controls_fn,
            route_mixer=route_mixer,
            token_alignment=token_alignment,
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
    scrambled_spatial, scramble_ledger = _scrambled_spatial_config(base_spatial)
    ledger = {
        **ledger,
        "route_vit_tokens": token_ledger,
        "scrambled_token_control": scramble_ledger,
    }
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
    warm_metrics = _metrics(warm_raw, data, projection=projection)
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

    rank = int(model["modulation_rank"])
    common = {
        "warmed": warmed,
        "warm_raw": warm_raw,
        "basis": basis,
        "data": data,
        "projection": projection,
        "rayleigh": rayleigh,
        "steps": int(optimization["comparison_steps"]),
        "learning_rate": float(optimization["route_learning_rate"]),
        "record_every": int(optimization["record_every"]),
    }
    arms.append(
        _route_arm(
            arm_id="route_vit_coefficients_d4",
            config=_route_config(base_spatial, model, 0),
            seed=seed + 1000,
            amplitudes_fn=su2_route_vit_amplitudes,
            controls_fn=su2_route_vit_controls,
            route_mixer="global_attention",
            token_alignment="physical",
            **common,
        )
    )
    for arm_id, spatial, amplitudes_fn, controls_fn, mixer, alignment in (
        (
            "route_vit_conditioned_d4",
            base_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
            "physical",
        ),
        (
            "route_vit_scrambled_tokens_d4",
            scrambled_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
            "scrambled_multiset_preserving",
        ),
        (
            "route_local_conditioned_d4",
            base_spatial,
            su2_route_local_gate_amplitudes,
            su2_route_local_gate_controls,
            "route_local_gate",
            "physical",
        ),
    ):
        arms.append(
            _route_arm(
                arm_id=arm_id,
                config=_route_config(spatial, model, rank),
                seed=seed + 1000 + rank,
                amplitudes_fn=amplitudes_fn,
                controls_fn=controls_fn,
                route_mixer=mixer,
                token_alignment=alignment,
                **common,
            )
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


def _causal_comparisons(cases: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    global_id = "route_vit_conditioned_d4"
    controls = (
        "route_vit_scrambled_tokens_d4",
        "route_local_conditioned_d4",
    )
    for case in cases:
        comparisons = {}
        for control_id in controls:
            rows = []
            for trial in case["trials"]:
                by_id = {arm["arm_id"]: arm for arm in trial["arms"]}
                delta = float(by_id[control_id]["metrics"]["energy"]) - float(
                    by_id[global_id]["metrics"]["energy"]
                )
                rows.append({"seed": int(trial["seed"]), "control_minus_global": delta})
            comparisons[control_id] = {
                "global_win_count": sum(row["control_minus_global"] > 0.0 for row in rows),
                "seed_count": len(rows),
                "median_control_minus_global_energy": float(
                    np.median([row["control_minus_global"] for row in rows])
                ),
                "per_seed": rows,
            }
        result[str(case["case_id"])] = comparisons
    return result


def run_spin_su2_route_vit_causal_scan(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the exact-small causal locality comparison."""

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
            "Finite 4x4 exact-support causal and coupling comparison. "
            "Thermodynamic references are contextual overlays, not finite-torus "
            "targets. No phase-boundary, QSL, or scaling claim is made."
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
            "exact_energy_anchor": float(case_spec["exact_energy_anchor"]),
            "anchor_provenance": str(case_spec["anchor_provenance"]),
            "thermodynamic_references": case_spec.get(
                "thermodynamic_references", []
            ),
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
    payload["causal_comparisons"] = _causal_comparisons(payload["cases"])
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["runtime_identity_sha256"] = canonical_sha256(
        payload["runtime_identity"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_su2_route_vit_causal_scan"]
