"""Local exact-support frontier for fixed-``Sz`` factored Spatial-ViT states.

The workflow reuses the active fixed-magnetisation Hamiltonian, exact lattice
projector, observables, and stored energy anchors.  It performs no Hamiltonian
diagonalisation.  Its purpose is to establish the conventional Transformer
baseline and quantify spin contamination before any SU(2)-ViT construction.
"""

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
from su2_nqs.benchmarks.fixed_sz_spatial_vit import (
    fixed_sz_spatial_vit_oracle,
)
from su2_nqs.benchmarks.spin_sector import (
    singlet_projector_action,
    spin_purity_metrics,
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
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.spin import _metrics, _prepare_case


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "fixed_sz_factored_spatial_vit_local_frontier"


def _capacity_key(arm: dict[str, object]) -> tuple[int, int, int, int]:
    return (
        int(arm["embedding_dim"]),
        int(arm["head_count"]),
        int(arm["depth"]),
        int(arm.get("mlp_ratio", 2)),
    )


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("Spatial-ViT frontier requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    if int(optimization.get("steps", 0)) < 1:
        raise ValueError("optimization.steps must be positive")
    if float(optimization.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimization.learning_rate must be positive")
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("optimization.seeds must be nonempty")
    model = spec.get("model")
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    if (int(model.get("patch_rows", 0)), int(model.get("patch_columns", 0))) != (
        2,
        2,
    ):
        raise ValueError("the 4x4 validation frontier requires 2x2 patches")
    arms = spec.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("at least two capacity/control arms are required")
    arm_ids = [str(arm.get("arm_id")) for arm in arms if isinstance(arm, dict)]
    if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arms must be objects with unique arm_id values")
    modes_by_capacity: dict[tuple[int, int, int, int], set[bool]] = {}
    for arm in arms:
        if not isinstance(arm.get("use_distance_envelope"), bool):
            raise TypeError("every arm must select use_distance_envelope")
        capacity = _capacity_key(arm)
        if capacity[0] % capacity[1]:
            raise ValueError("every embedding_dim must be divisible by head_count")
        modes_by_capacity.setdefault(capacity, set()).add(
            bool(arm["use_distance_envelope"])
        )
    if not any(modes == {False, True} for modes in modes_by_capacity.values()):
        raise ValueError("one capacity must pair flat and spatial distance kernels")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        if case.get("geometry") not in {"square", "triangular"}:
            raise ValueError("case geometry must be square or triangular")
        if (case.get("rows"), case.get("columns")) != (4, 4):
            raise ValueError("the local frontier is preregistered on 4x4")
        if case.get("projection") != "translation":
            raise ValueError("the primary local frontier requires translations")
        if "exact_energy_anchor" not in case:
            raise ValueError("every case requires a stored exact energy anchor")


def _model_config(
    spec: dict[str, object],
    arm: dict[str, object],
    case: dict[str, object],
) -> FixedSzSpatialViTConfig:
    model = spec["model"]
    return FixedSzSpatialViTConfig(
        geometry=str(case["geometry"]),
        rows=int(case["rows"]),
        columns=int(case["columns"]),
        patch_rows=int(model["patch_rows"]),
        patch_columns=int(model["patch_columns"]),
        embedding_dim=int(arm["embedding_dim"]),
        head_count=int(arm["head_count"]),
        depth=int(arm["depth"]),
        mlp_ratio=int(arm.get("mlp_ratio", model.get("mlp_ratio", 2))),
        target_m2=0,
        initialization_scale=float(model["initialization_scale"]),
        initial_distance_decay=float(model["initial_distance_decay"]),
        use_distance_envelope=bool(arm["use_distance_envelope"]),
    )


def _projected_normalized_state(
    raw: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(
        reynolds_project_state(jnp.asarray(raw), jnp.asarray(indices))
    )
    norm = float(np.vdot(selected, selected).real)
    if not np.isfinite(norm) or norm <= 1.0e-28:
        raise RuntimeError("Spatial-ViT projected state has zero norm")
    return selected / np.sqrt(norm)


def _spin_projection_diagnostics(
    state: np.ndarray,
    data: dict[str, object],
    *,
    raw_energy: float,
) -> dict[str, object]:
    n_sites = int(data["n_sites"])
    purity = spin_purity_metrics(
        state,
        data["spin_squared"],
        maximum_spin=n_sites // 2,
    )
    singlet = singlet_projector_action(
        state,
        data["spin_squared"],
        maximum_spin=n_sites // 2,
    )
    weight = float(np.vdot(singlet, singlet).real)
    if weight <= 1.0e-12:
        return {
            **purity.__dict__,
            "singlet_projected_energy": None,
            "singlet_projected_energy_error_per_site": None,
            "singlet_projected_variance_per_site": None,
            "fraction_of_raw_energy_error_explained_by_spin_projection": None,
        }
    singlet /= np.sqrt(weight)
    action = np.asarray(data["hamiltonian"] @ singlet)
    energy = float(np.vdot(singlet, action).real)
    residual = action - energy * singlet
    exact = float(data["exact_energy"])
    denominator = raw_energy - exact
    fraction = None
    if abs(denominator) > 1.0e-14:
        fraction = (raw_energy - energy) / denominator
    return {
        **purity.__dict__,
        "singlet_projected_energy": energy,
        "singlet_projected_energy_error_per_site": (energy - exact) / n_sites,
        "singlet_projected_variance_per_site": float(
            np.vdot(residual, residual).real
        )
        / n_sites,
        "fraction_of_raw_energy_error_explained_by_spin_projection": fraction,
    }


def _distance_diagnostics(params: dict[str, object]) -> dict[str, object]:
    decay = [
        np.asarray(jax.nn.softplus(layer["raw_distance_decay"])).tolist()
        for layer in params["layers"]
    ]
    factors = [
        {
            "minimum": float(np.min(np.asarray(layer["relative_factor"]))),
            "maximum": float(np.max(np.asarray(layer["relative_factor"]))),
            "rms": float(
                np.sqrt(np.mean(np.asarray(layer["relative_factor"]) ** 2))
            ),
        }
        for layer in params["layers"]
    ]
    return {"distance_decay_by_layer_head": decay, "relative_factor_by_layer": factors}


def _train_trial(
    arm: dict[str, object],
    *,
    seed: int,
    spec: dict[str, object],
    case_spec: dict[str, object],
    data: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    config = _model_config(spec, arm, case_spec)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(seed), config)
    parameter_count = fixed_sz_spatial_vit_parameter_count(params)
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    projection = str(case_spec["projection"])
    indices = jnp.asarray(data["tables"][projection], dtype=jnp.int32)
    rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def raw_amplitudes(candidate):
        return fixed_sz_spatial_vit_amplitudes(candidate, basis, config)

    def amplitudes(candidate):
        return reynolds_project_state(raw_amplitudes(candidate), indices)

    def loss(candidate):
        state = amplitudes(candidate)
        action = coo_matvec(rows, columns, values, state)
        return jnp.vdot(state, action).real / jnp.vdot(state, state).real

    initial_params_sha256 = pytree_sha256(params)
    initial_raw = np.asarray(raw_amplitudes(params), dtype=np.complex128)
    initial_raw_sha256 = array_sha256(initial_raw)
    optimization = spec["optimization"]
    optimizer_started = time.perf_counter()
    trained, training = train_exact_adam(
        params,
        loss_fn=loss,
        steps=int(optimization["steps"]),
        learning_rate=float(arm.get("learning_rate", optimization["learning_rate"])),
        record_every=int(optimization["record_every"]),
    )
    optimizer_seconds = time.perf_counter() - optimizer_started
    raw = np.asarray(raw_amplitudes(trained), dtype=np.complex128)
    metrics = _metrics(raw, data, projection=projection)
    state = _projected_normalized_state(raw, np.asarray(indices))
    spin = _spin_projection_diagnostics(
        state,
        data,
        raw_energy=float(metrics["energy"]),
    )
    group_order = len(data["groups"][projection])
    result = {
        "arm_id": str(arm["arm_id"]),
        "seed": int(seed),
        "projection": projection,
        "support": "fixed_m0",
        "architecture": "factored_spatial_vit",
        "use_distance_envelope": config.use_distance_envelope,
        "embedding_dim": config.embedding_dim,
        "head_count": config.head_count,
        "depth": config.depth,
        "mlp_ratio": config.mlp_ratio,
        "patch_shape": [config.patch_rows, config.patch_columns],
        "patch_token_count": config.token_count,
        "parameter_count": parameter_count,
        "initial_params_sha256": initial_params_sha256,
        "initial_raw_amplitudes_sha256": initial_raw_sha256,
        "paired_backbone_id": canonical_sha256(
            {
                "case": str(case_spec["case_id"]),
                "seed": int(seed),
                "capacity": list(_capacity_key(arm)),
                "initialization_scale": config.initialization_scale,
            }
        ),
        "training": training,
        "optimizer_wall_seconds": optimizer_seconds,
        "trial_wall_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "spin_sector_diagnostics": spin,
        "spatial_kernel_diagnostics": _distance_diagnostics(trained),
        "work_ledger_per_loss": {
            "raw_fixed_m0_configurations": len(data["basis"]),
            "patch_tokens_per_configuration": config.token_count,
            "factored_attention_head_pairs_per_configuration": (
                config.depth
                * config.head_count
                * config.token_count
                * config.token_count
            ),
            "exact_translation_group_gathers": group_order * len(data["basis"]),
            "scalable_translation_coset_amplitude_evaluations": (
                config.patch_site_count
            ),
            "hamiltonian_coo_entries": len(data["h_values"]),
        },
    }
    result["energy_to_target"] = observed_energy_to_target(
        training,
        final_energy=float(metrics["energy"]),
        exact_energy=float(data["exact_energy"]),
        n_sites=int(data["n_sites"]),
        total_steps=int(optimization["steps"]),
        targets_per_site=optimization["energy_error_per_site_targets"],
    )
    return result


def _summary(cases: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for case in cases:
        by_arm: dict[str, list[dict[str, object]]] = {}
        for trial in case["trials"]:
            by_arm.setdefault(str(trial["arm_id"]), []).append(trial)
        arms = {}
        for arm_id, trials in by_arm.items():
            arms[arm_id] = {
                "parameter_count": int(trials[0]["parameter_count"]),
                "seed_count": len(trials),
                "median_energy_error_per_site": float(
                    np.median(
                        [row["metrics"]["energy_error_per_site"] for row in trials]
                    )
                ),
                "median_variance_per_site": float(
                    np.median([row["metrics"]["variance_per_site"] for row in trials])
                ),
                "median_singlet_weight": float(
                    np.median(
                        [
                            row["spin_sector_diagnostics"]["singlet_weight"]
                            for row in trials
                        ]
                    )
                ),
                "median_spin_projection_explainable_fraction": float(
                    np.median(
                        [
                            row["spin_sector_diagnostics"][
                                "fraction_of_raw_energy_error_explained_by_spin_projection"
                            ]
                            for row in trials
                        ]
                    )
                ),
                "median_trial_wall_seconds": float(
                    np.median([row["trial_wall_seconds"] for row in trials])
                ),
            }

        paired = []
        signatures: dict[tuple[int, int, int, int], dict[bool, str]] = {}
        for arm_id, trials in by_arm.items():
            first = trials[0]
            signature = (
                int(first["embedding_dim"]),
                int(first["head_count"]),
                int(first["depth"]),
                int(first["mlp_ratio"]),
            )
            signatures.setdefault(signature, {})[
                bool(first["use_distance_envelope"])
            ] = arm_id
        for signature, modes in signatures.items():
            if set(modes) != {False, True}:
                continue
            flat = {int(row["seed"]): row for row in by_arm[modes[False]]}
            spatial = {int(row["seed"]): row for row in by_arm[modes[True]]}
            if set(flat) != set(spatial):
                raise RuntimeError("flat/spatial arms do not share paired seeds")
            rows = []
            for seed in sorted(flat):
                flat_error = float(flat[seed]["metrics"]["energy_error_per_site"])
                spatial_error = float(
                    spatial[seed]["metrics"]["energy_error_per_site"]
                )
                rows.append(
                    {
                        "seed": seed,
                        "spatial_minus_flat_energy_error_per_site": (
                            spatial_error - flat_error
                        ),
                        "spatial_over_flat_energy_error_ratio": (
                            spatial_error / flat_error
                        ),
                        "spatial_minus_flat_singlet_weight": float(
                            spatial[seed]["spin_sector_diagnostics"]["singlet_weight"]
                            - flat[seed]["spin_sector_diagnostics"]["singlet_weight"]
                        ),
                    }
                )
            deltas = [row["spatial_minus_flat_energy_error_per_site"] for row in rows]
            paired.append(
                {
                    "capacity": list(signature),
                    "flat_arm_id": modes[False],
                    "spatial_arm_id": modes[True],
                    "seed_count": len(rows),
                    "spatial_energy_win_count": sum(delta < 0.0 for delta in deltas),
                    "median_paired_energy_error_per_site_delta": float(
                        np.median(deltas)
                    ),
                    "median_paired_energy_error_ratio": float(
                        np.median(
                            [row["spatial_over_flat_energy_error_ratio"] for row in rows]
                        )
                    ),
                    "per_seed": rows,
                }
            )
        result[str(case["case_id"])] = {
            "arms": arms,
            "paired_spatial_vs_flat": paired,
        }
    return result


def run_spin_spatial_vit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run exact-small oracles and the paired square/triangular 4x4 frontier."""

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
        "hamiltonian_diagonalization_performed": False,
        "exact_small_oracles": {
            geometry: fixed_sz_spatial_vit_oracle(geometry)
            for geometry in ("square", "triangular")
        },
        "cases": [],
        "claim_boundary": (
            "Validation-scale 4x4 fixed-Sz factored Spatial-ViT. The 2x2 "
            "patch is chosen to retain four tokens; this is not a reproduction "
            "of the production 20x20 architecture, a scaling claim, or a QSL "
            "result. Spin projection is a diagnostic, not a scalable ansatz arm."
        ),
    }
    if not all(row["passed"] for row in payload["exact_small_oracles"].values()):
        raise RuntimeError("a fixed-Sz Spatial-ViT exact-small oracle failed")
    atomic_json(result_path, payload)
    for case_spec in spec["cases"]:
        preparation_started = time.perf_counter()
        adapted = {**case_spec, "support": "fixed_m0"}
        data = _prepare_case(adapted, "fixed_m0")
        case: dict[str, object] = {
            "case_id": str(case_spec["case_id"]),
            "geometry": str(data["geometry"]),
            "rows": int(data["rows"]),
            "columns": int(data["columns"]),
            "j2_over_j1": float(data["j2_over_j1"]),
            "projection": str(case_spec["projection"]),
            "basis_size": len(data["basis"]),
            "hamiltonian_nnz": int(data["hamiltonian"].nnz),
            "exact_energy_anchor": float(data["exact_energy"]),
            "anchor_provenance": str(data["anchor_provenance"]),
            "preparation_seconds": time.perf_counter() - preparation_started,
            "trials": [],
        }
        payload["cases"].append(case)
        atomic_json(result_path, payload)
        for arm in spec["arms"]:
            for seed in spec["optimization"]["seeds"]:
                trial = _train_trial(
                    arm,
                    seed=int(seed),
                    spec=spec,
                    case_spec=case_spec,
                    data=data,
                )
                case["trials"].append(trial)
                print(
                    f"{case['case_id']} {trial['arm_id']} seed={seed}: "
                    f"P={trial['parameter_count']}, "
                    f"dE/N={trial['metrics']['energy_error_per_site']:.3e}, "
                    f"w0={trial['spin_sector_diagnostics']['singlet_weight']:.4f}, "
                    f"wall={trial['trial_wall_seconds']:.2f}s",
                    flush=True,
                )
                atomic_json(result_path, payload)
    payload["summary"] = _summary(payload["cases"])
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_spatial_vit"]
