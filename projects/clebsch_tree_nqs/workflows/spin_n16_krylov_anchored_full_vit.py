"""Run the N=16 Krylov-anchored full Spatial-ViT decision gate."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_basis_matrix
from su2_nqs.benchmarks.krylov_anchored_full_vit import (
    a1_orbit_coordinates,
    a1_orbit_hamiltonian,
    a1_orbit_reduction,
    anchored_unit_residual_state,
    fixed_sz_projected_residual,
    lowest_krylov_state,
    matched_backbone_initializations,
    scrambled_coordinate_layout,
    su2_projected_residual,
)
from su2_nqs.benchmarks.many_cell_replay import (
    TRIANGULAR_N16_SUPPORTS,
    balanced_tree_to_rotated_coefficients,
    rotated_to_balanced_tree_coefficients,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import init_many_cell_schur_vit
from su2_nqs.vmc.adam import train_exact_adam


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_krylov_anchored_full_vit"
SUPERCELL = ((4, 0), (0, 4))


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 Krylov-anchored full-ViT identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the comparison must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("the exact-small gate is fixed to triangular 4x4")
    if float(spec.get("j2_over_j1", np.nan)) != 0.125:
        raise ValueError("the registered target is J2/J1=0.125")
    if spec.get("arms") != [
        "krylov_anchored_exact_su2_a1_full_vit",
        "krylov_anchored_fixed_sz_a1_full_vit",
        "degree_four_krylov",
        "spatial_scrambled_exact_su2_a1_full_vit",
    ]:
        raise ValueError("the four-arm identity or order changed")
    architecture = spec.get("architecture")
    optimization = spec.get("optimization")
    source = spec.get("source")
    gates = spec.get("gates")
    for name, block in (
        ("architecture", architecture),
        ("optimization", optimization),
        ("source", source),
        ("gates", gates),
    ):
        if not isinstance(block, dict):
            raise TypeError(f"{name} must be an object")
    for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
        if isinstance(architecture.get(name), bool) or int(
            architecture.get(name, 0)
        ) < 1:
            raise ValueError(f"architecture.{name} must be positive")
    if int(architecture["embedding_dim"]) % int(architecture["head_count"]):
        raise ValueError("embedding_dim must be divisible by head_count")
    if architecture.get("use_distance_envelope") is not False:
        raise ValueError(
            "the exact scramble requires the support-distance envelope disabled"
        )
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("optimization.seeds must be nonempty and distinct")
    if any(isinstance(seed, bool) or int(seed) < 0 for seed in seeds):
        raise ValueError("optimization seeds must be nonnegative integers")
    if int(optimization.get("steps", 0)) < 1:
        raise ValueError("optimization.steps must be positive")
    if float(optimization.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimization.learning_rate must be positive")
    if float(optimization.get("initial_residual_scale", 0.0)) <= 0.0:
        raise ValueError("initial_residual_scale must be positive")
    if int(spec.get("scramble_seed", -1)) < 0:
        raise ValueError("scramble_seed must be nonnegative")


def _metrics(
    state: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    n_sites: int = 16,
) -> dict[str, float]:
    vector = np.array(state, dtype=np.complex128, copy=True)
    vector /= np.linalg.norm(vector)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    variance = float(np.vdot(residual, residual).real)
    return {
        "energy_total": energy,
        "energy_per_site": energy / n_sites,
        "variance_total": variance,
        "variance_per_site": variance / n_sites,
    }


def _parameter_count(params: object) -> int:
    return int(sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)))


def _load_source(spec: dict[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(spec["result"])
    bundle_path = ROOT / str(spec["bundle"])
    result = json.loads(result_path.read_text())
    seal = str(spec["result_sha256"])
    if result.get("result_sha256") != seal or canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError("source result seal changed")
    if file_sha256(bundle_path) != str(spec["bundle_sha256"]):
        raise RuntimeError("source state bundle seal changed")
    return result, bundle_path


def _balanced_a1_basis(
    coefficient_a1_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layout = matched_backbone_initializations(
        0,
        embedding_dim=4,
        head_count=1,
        depth=1,
        mlp_ratio=1,
        phase_scale=0.25,
        initialization_scale=0.1,
        use_distance_envelope=False,
    )[2].layout
    columns = []
    labels = None
    for index in range(coefficient_a1_basis.shape[1]):
        balanced, current = rotated_to_balanced_tree_coefficients(
            coefficient_a1_basis[:, index], layout
        )
        columns.append(balanced)
        if labels is None:
            labels = current
        elif not np.array_equal(labels, current):
            raise RuntimeError("balanced tree label order changed between columns")
    basis = np.column_stack(columns)
    if np.max(np.abs(basis.conj().T @ basis - np.eye(basis.shape[1]))) > 2.0e-11:
        raise RuntimeError("balanced A1 coefficient basis is not isometric")
    assert labels is not None
    return basis, labels


def _train(
    initial_model: object,
    state_fn,
    hamiltonian: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    initial_residual_scale: float,
) -> tuple[dict[str, object], np.ndarray]:
    matrix = jnp.asarray(hamiltonian)
    initial = {
        "model": initial_model,
        "residual_scale": jnp.asarray(initial_residual_scale, dtype=jnp.float64),
    }

    def loss(params):
        vector = state_fn(params)
        return jnp.real(jnp.vdot(vector, matrix @ vector) / jnp.vdot(vector, vector))

    initial_energy = float(loss(initial))
    started = time.perf_counter()
    trained, optimizer = train_exact_adam(
        initial,
        loss_fn=loss,
        steps=int(steps),
        learning_rate=float(learning_rate),
        record_every=max(1, int(steps) // 8),
    )
    wall = time.perf_counter() - started
    state = np.array(state_fn(trained), copy=True)
    state /= np.linalg.norm(state)
    return {
        "parameter_count": _parameter_count(trained),
        "initial_energy_total": initial_energy,
        "residual_scale": float(trained["residual_scale"]),
        "optimizer": optimizer,
        "wall_seconds": wall,
        "full_h_calls": int(steps + 2),
        "metrics": _metrics(state, hamiltonian),
    }, state


def run_spin_n16_krylov_anchored_full_vit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the four-arm exact-small decision gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    source_result, bundle_path = _load_source(spec["source"])
    with np.load(bundle_path, allow_pickle=False) as arrays:
        coefficient_a1 = np.asarray(arrays["coefficient_a1_basis"])
        physical_a1 = np.asarray(arrays["physical_a1_basis"])
        projected_vmc = np.asarray(arrays["coupling_0.125_carrier_a1"])
        two_lanczos = np.asarray(arrays["coupling_0.125_two_lanczos_a1"])
        h_a1 = np.asarray(arrays["coupling_0.125_h_a1"])
    projected_vmc /= np.linalg.norm(projected_vmc)
    two_lanczos /= np.linalg.norm(two_lanczos)
    balanced_a1, labels = _balanced_a1_basis(coefficient_a1)

    schur_basis, configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_SUPPORTS
    )
    if schur_basis.shape != (12870, 1430) or physical_a1.shape != (12870, 14):
        raise RuntimeError("sealed N16 basis dimensions changed")
    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    permutation_indices = physical_permutation_index_table(configurations, group)
    orbit = a1_orbit_reduction(permutation_indices)
    bonds = triangular_hnf_j1j2_bonds(SUPERCELL, j1=1.0, j2=0.125)
    rows, columns, elements, checked = graph_hamiltonian_coo(16, 0, bonds)
    if checked != configurations:
        raise RuntimeError("fixed-Sz basis order changed")
    h_orbit = a1_orbit_hamiltonian(rows, columns, elements, orbit)
    configurations_jax = jnp.asarray(configurations, dtype=jnp.int32)
    labels_jax = jnp.asarray(labels, dtype=jnp.int32)
    balanced_a1_jax = jnp.asarray(balanced_a1)

    physical_two_lanczos = physical_a1 @ two_lanczos
    carrier_orbit = np.array(
        a1_orbit_coordinates(jnp.asarray(physical_two_lanczos), orbit),
        copy=True,
    )
    carrier_orbit /= np.linalg.norm(carrier_orbit)
    singlet_columns_orbit = np.column_stack(
        [
            np.asarray(a1_orbit_coordinates(jnp.asarray(physical_a1[:, index]), orbit))
            for index in range(physical_a1.shape[1])
        ]
    )
    singlet_from_orbit = singlet_columns_orbit.conj().T

    block_check = singlet_from_orbit @ h_orbit @ singlet_columns_orbit
    operator_replay_error = float(np.max(np.abs(block_check - h_a1)))
    carrier_replay_error = float(
        np.linalg.norm(singlet_from_orbit @ carrier_orbit - two_lanczos)
    )
    if operator_replay_error > 2.0e-10 or carrier_replay_error > 2.0e-10:
        raise RuntimeError("sealed A1 source does not replay in orbit coordinates")

    degree_four, krylov_rank = lowest_krylov_state(
        projected_vmc, h_a1, degree=4
    )
    baselines = {
        "projected_vmc": _metrics(projected_vmc, h_a1),
        "two_lanczos": _metrics(two_lanczos, h_a1),
        "degree_four_krylov": {
            **_metrics(degree_four, h_a1),
            "rank": int(krylov_rank),
            "definition": "lowest Ritz state in span{psi,Hpsi,...,H^4 psi} from the sealed projected-VMC carrier",
        },
    }

    architecture = spec["architecture"]
    optimization = spec["optimization"]
    shared = {
        "embedding_dim": int(architecture["embedding_dim"]),
        "head_count": int(architecture["head_count"]),
        "depth": int(architecture["depth"]),
        "mlp_ratio": int(architecture["mlp_ratio"]),
        "phase_scale": float(architecture["phase_scale"]),
        "initialization_scale": float(architecture["initialization_scale"]),
        "use_distance_envelope": bool(architecture["use_distance_envelope"]),
    }
    scramble = np.random.default_rng(int(spec["scramble_seed"])).permutation(16)
    runs = {
        "krylov_anchored_exact_su2_a1_full_vit": [],
        "krylov_anchored_fixed_sz_a1_full_vit": [],
        "spatial_scrambled_exact_su2_a1_full_vit": [],
    }
    best_states: dict[str, tuple[float, np.ndarray]] = {}
    started = time.perf_counter()
    for seed_value in optimization["seeds"]:
        seed = int(seed_value)
        su2_initial, fixed_initial, su2_config, fixed_config = (
            matched_backbone_initializations(seed, **shared)
        )
        scrambled_layout = scrambled_coordinate_layout(su2_config.layout, scramble)
        scrambled_config = replace(su2_config, layout=scrambled_layout)
        scrambled_initial = init_many_cell_schur_vit(
            jax.random.PRNGKey(seed), scrambled_config
        )
        if any(
            not np.array_equal(left, right)
            for left, right in zip(
                jax.tree_util.tree_leaves(su2_initial),
                jax.tree_util.tree_leaves(scrambled_initial),
                strict=True,
            )
        ):
            raise RuntimeError("spatial scramble changed the initial arrays")

        def su2_state(params):
            residual = su2_projected_residual(
                params["model"], labels_jax, su2_config, balanced_a1_jax
            )
            return anchored_unit_residual_state(
                jnp.asarray(two_lanczos), residual, params["residual_scale"]
            )

        def scrambled_state(params):
            residual = su2_projected_residual(
                params["model"], labels_jax, scrambled_config, balanced_a1_jax
            )
            return anchored_unit_residual_state(
                jnp.asarray(two_lanczos), residual, params["residual_scale"]
            )

        def fixed_state(params):
            residual = fixed_sz_projected_residual(
                params["model"], configurations_jax, fixed_config, orbit
            )
            return anchored_unit_residual_state(
                jnp.asarray(carrier_orbit), residual, params["residual_scale"]
            )

        specifications = (
            (
                "krylov_anchored_exact_su2_a1_full_vit",
                su2_initial,
                su2_state,
                h_a1,
            ),
            (
                "krylov_anchored_fixed_sz_a1_full_vit",
                fixed_initial,
                fixed_state,
                h_orbit,
            ),
            (
                "spatial_scrambled_exact_su2_a1_full_vit",
                scrambled_initial,
                scrambled_state,
                h_a1,
            ),
        )
        for arm_id, initial, state_fn, hamiltonian in specifications:
            report, state = _train(
                initial,
                state_fn,
                hamiltonian,
                steps=int(optimization["steps"]),
                learning_rate=float(optimization["learning_rate"]),
                initial_residual_scale=float(
                    optimization["initial_residual_scale"]
                ),
            )
            report["arm_id"] = arm_id
            report["seed"] = seed
            if arm_id == "krylov_anchored_fixed_sz_a1_full_vit":
                normalized = state / np.linalg.norm(state)
                report["singlet_weight"] = float(
                    np.linalg.norm(singlet_from_orbit @ normalized) ** 2
                )
                report["exact_sz0_and_gamma_a1"] = True
                report["exact_total_spin"] = False
            else:
                report["singlet_weight"] = 1.0
                report["exact_sz0_and_gamma_a1"] = True
                report["exact_total_spin"] = True
            runs[arm_id].append(report)
            old = best_states.get(arm_id)
            energy = float(report["metrics"]["energy_total"])
            if old is None or energy < old[0]:
                best_states[arm_id] = (energy, state)

    summary = {}
    for arm_id, rows_for_arm in runs.items():
        energies = np.asarray(
            [row["metrics"]["energy_per_site"] for row in rows_for_arm]
        )
        variances = np.asarray(
            [row["metrics"]["variance_per_site"] for row in rows_for_arm]
        )
        summary[arm_id] = {
            "median_energy_per_site": float(np.median(energies)),
            "best_energy_per_site": float(np.min(energies)),
            "median_variance_per_site": float(np.median(variances)),
            "best_variance_per_site": float(np.min(variances)),
            "all_energy_per_site": energies.tolist(),
            "all_variance_per_site": variances.tolist(),
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in rows_for_arm])
            ),
            "parameter_count": int(rows_for_arm[0]["parameter_count"]),
            "full_h_calls_per_seed": int(rows_for_arm[0]["full_h_calls"]),
        }
    su2 = summary["krylov_anchored_exact_su2_a1_full_vit"]
    fixed = summary["krylov_anchored_fixed_sz_a1_full_vit"]
    scrambled = summary["spatial_scrambled_exact_su2_a1_full_vit"]
    krylov = baselines["degree_four_krylov"]
    tolerance = float(spec["gates"]["minimum_strict_improvement_per_site"])
    gates = {
        "matched_su2_fixed_parameter_count": (
            su2["parameter_count"] == fixed["parameter_count"]
        ),
        "su2_median_energy_beats_degree_four_krylov": (
            su2["median_energy_per_site"]
            < krylov["energy_per_site"] - tolerance
        ),
        "su2_median_variance_beats_degree_four_krylov": (
            su2["median_variance_per_site"]
            < krylov["variance_per_site"] - tolerance
        ),
        "su2_median_energy_beats_fixed_sz": (
            su2["median_energy_per_site"]
            < fixed["median_energy_per_site"] - tolerance
        ),
        "physical_spatial_labels_beat_scramble_energy": (
            su2["median_energy_per_site"]
            < scrambled["median_energy_per_site"] - tolerance
        ),
        "physical_spatial_labels_beat_scramble_variance": (
            su2["median_variance_per_site"]
            < scrambled["median_variance_per_site"] - tolerance
        ),
    }
    promotion_passed = bool(
        gates["matched_su2_fixed_parameter_count"]
        and gates["su2_median_energy_beats_degree_four_krylov"]
        and gates["su2_median_variance_beats_degree_four_krylov"]
    )
    saved = {
        "degree_four_krylov_a1": degree_four,
        "two_lanczos_a1": two_lanczos,
        "balanced_a1_basis": balanced_a1,
        "balanced_labels": labels,
        "fixed_sz_orbit_id": np.asarray(orbit.orbit_id),
        "fixed_sz_orbit_sizes": np.asarray(orbit.orbit_sizes),
        "scramble_permutation": scramble,
    }
    for arm_id, (_energy, state) in best_states.items():
        saved[f"best_{arm_id}"] = state
    _atomic_npz(state_path, saved)

    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "system": {
            "geometry": "triangular_4x4_periodic",
            "n_sites": 16,
            "j2_over_j1": 0.125,
            "fixed_sz_dimension": len(configurations),
            "complete_singlet_dimension": len(labels),
            "singlet_gamma_a1_dimension": physical_a1.shape[1],
            "fixed_sz_gamma_a1_orbit_dimension": orbit.reduced_dimension,
            "space_group_order": len(group),
        },
        "representation_audit": {
            "sealed_operator_replay_max_abs_error": operator_replay_error,
            "two_lanczos_coordinate_replay_l2_error": carrier_replay_error,
            "balanced_a1_isometry_max_abs_error": float(
                np.max(
                    np.abs(
                        balanced_a1.conj().T @ balanced_a1
                        - np.eye(balanced_a1.shape[1])
                    )
                )
            ),
            "a1_enforced_inside_every_loss": True,
            "su2_magnetic_indices_visible_to_network": False,
            "shared_su2_fixed_backbone_arrays_bitwise_identical_at_initialization": True,
        },
        "baselines": baselines,
        "runs": runs,
        "summary": summary,
        "comparisons": {
            "su2_minus_degree_four_energy_per_site": float(
                su2["median_energy_per_site"] - krylov["energy_per_site"]
            ),
            "su2_minus_degree_four_variance_per_site": float(
                su2["median_variance_per_site"] - krylov["variance_per_site"]
            ),
            "su2_minus_fixed_sz_energy_per_site": float(
                su2["median_energy_per_site"] - fixed["median_energy_per_site"]
            ),
            "su2_minus_spatial_scramble_energy_per_site": float(
                su2["median_energy_per_site"]
                - scrambled["median_energy_per_site"]
            ),
        },
        "gates": {**gates, "promotion_passed": promotion_passed},
        "decision": (
            "license_sampled_additive_carrier_preflight"
            if promotion_passed
            else "hold_n36_change_state_manifold_or_training"
        ),
        "repeated_many_body_ed_performed": False,
        "sealed_n16_representation_reused": True,
        "source_result_sha256": source_result["result_sha256"],
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": spec["claim_boundary"],
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


__all__ = ["TASK", "run_spin_n16_krylov_anchored_full_vit"]
