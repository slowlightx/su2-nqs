"""Unified local physics audit for exact-SU(2) Route-ViT states.

One workflow performs the five physics-facing stages requested by the project:

P1
    paired translation and common-Gamma-A1 Route-ViT state/observable audit;
P2
    scale-resolved local-RDM error against a cached N=16 exact state;
P3
    N=12 symmetry-resolved triangular bond/spin spectroscopy;
P4
    fidelity, static, magnetic, nematic, X-bond, and chiral responses;
P5
    Gamma-A1 physical error-channel tomography and small-subspace Ritz gains.

The N=12 ``((6,4),(0,2))`` HNF cell preserves full D6 and the K/M/X stars,
but its 36 J2 periodic images alias to 18 distinct edges of weight two.  Every
result is therefore labelled a finite thin-torus mechanism diagnostic, never a
phase-boundary or thermodynamic-order claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import time

import jax
import numpy as np

from su2_nqs.benchmarks.exact_reference_cache import (
    ExactReferenceCacheMiss,
    load_or_recompute_exact_reference,
)
from su2_nqs.benchmarks.physical_interpretation import (
    FixedSzPairActions,
    error_channel_tomography,
    fixed_sector_to_full_state,
    hnf_oriented_triangles,
    local_rdm_error_profile,
    local_rdm_reference_cache,
    triangular_cluster_supports,
    triangular_error_channel_actions,
    triangular_periodic_observable_panel,
)
from su2_nqs.benchmarks.spectral_physics import compile_sector_spin_vector
from su2_nqs.benchmarks.triangular_hnf_space_group import (
    d6_character_projectors,
    exact_gamma_a1_spin_multiplicity_dimension,
    hnf_fraction_character_numerator,
    momentum_projector_matrix,
    reciprocal_little_group,
)
from su2_nqs.benchmarks.triangular_monopole_branch import (
    bond_center_orientation_operators,
    little_group_character_projector,
    maximum_low_subspace_source_fraction,
    projector_basis,
    source_spectrum,
)
from su2_nqs.benchmarks.low_energy_subspace import matrix_free_chirality_action
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_hnf import (
    J1_POSITIVE,
    hnf_sites,
    triangular_hnf_j1j2_bonds,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

from projects.clebsch_tree_nqs.workflows.route_vit_exact_state import (
    train_exact_route_vit_trial,
)
from projects.clebsch_tree_nqs.workflows.spin import (
    _metrics,
    _prepare_case,
    _project,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects" / "clebsch_tree_nqs"
TASK = "exact_spin_physics_interpretation"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("physics interpretation requires schema_version=1 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    case = spec.get("p1_case")
    model = spec.get("model")
    optimization = spec.get("optimization")
    spectroscopy = spec.get("spectroscopy")
    if not all(isinstance(value, dict) for value in (case, model, optimization, spectroscopy)):
        raise TypeError("p1_case, model, optimization, and spectroscopy must be objects")
    assert isinstance(case, dict)
    if (case.get("geometry"), case.get("rows"), case.get("columns")) != (
        "triangular",
        4,
        4,
    ):
        raise ValueError("P1/P2/P5 are registered on triangular 4x4")
    if abs(float(case.get("j2_over_j1", -1.0)) - 0.075) > 1.0e-14:
        raise ValueError("P1/P2/P5 are registered at J2/J1=0.075")
    assert isinstance(optimization, dict)
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seeds
    ):
        raise ValueError("optimization.seeds must be nonempty integers")
    projections = optimization.get("training_projections")
    if projections != ["translation", "gamma_a1"]:
        raise ValueError("training_projections must be translation then gamma_a1")
    assert isinstance(spectroscopy, dict)
    supercell = spectroscopy.get("supercell_hnf")
    if supercell != [[6, 4], [0, 2]]:
        raise ValueError("P3/P4 use the registered full-D6 N12 HNF ((6,4),(0,2))")
    couplings = tuple(float(value) for value in spectroscopy.get("j2_values", ()))
    if couplings != (0.05, 0.075, 0.1, 0.125, 0.15):
        raise ValueError("P3/P4 require the preregistered five-coupling grid")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _normalized_projected_state(
    raw: np.ndarray, data: dict[str, object], projection: str
) -> np.ndarray:
    selected = np.asarray(_project(raw, data, projection), dtype=np.complex128)
    norm = float(np.linalg.norm(selected))
    if norm <= 1.0e-14:
        raise ValueError("projected state has zero numerical norm")
    return selected / norm


def _cached_n16_ground(
    data: dict[str, object], cache_root: Path, *, recompute_if_missing: bool
):
    identity = {
        "purpose": "triangular_4x4_j2_0p075_fixed_m0_ground",
        "basis_sha256": data["basis_sha256"],
        "hamiltonian_coo_sha256": data["hamiltonian_coo_sha256"],
        "solver": "scipy.sparse.linalg.eigsh_SA_tol_1e-13_v1",
    }

    def compute():
        from scipy.sparse.linalg import eigsh

        values, vectors = eigsh(
            data["hamiltonian"], k=1, which="SA", tol=1.0e-13, maxiter=200000
        )
        ordering = np.argsort(values)
        energy = float(values[ordering[0]])
        state = np.asarray(vectors[:, ordering[0]], dtype=np.complex128)
        state /= np.linalg.norm(state)
        residual = np.asarray(data["hamiltonian"] @ state - energy * state)
        return {
            "energy": np.asarray(energy, dtype=np.float64),
            "state": state,
            "residual_norm": np.asarray(np.linalg.norm(residual), dtype=np.float64),
        }

    try:
        return load_or_recompute_exact_reference(cache_root, identity)
    except ExactReferenceCacheMiss:
        if not recompute_if_missing:
            raise
        return load_or_recompute_exact_reference(
            cache_root, identity, recompute=True, compute=compute
        )


def _structure_factor_errors(
    metrics: dict[str, object], exact_metrics: dict[str, object]
) -> dict[str, float]:
    current = np.asarray(
        metrics["physical_observables"]["spin_structure_factor_grid"],
        dtype=np.float64,
    )
    exact = np.asarray(
        exact_metrics["physical_observables"]["spin_structure_factor_grid"],
        dtype=np.float64,
    )
    difference = current - exact
    m_values = np.asarray((current[2, 0], current[0, 2], current[2, 2]))
    exact_m = np.asarray((exact[2, 0], exact[0, 2], exact[2, 2]))
    return {
        "rms_error": float(np.sqrt(np.mean(difference**2))),
        "maximum_abs_error": float(np.max(np.abs(difference))),
        "uniform_structure_factor": float(current[0, 0]),
        "structure_factor_sum": float(np.sum(current)),
        "structure_factor_sum_rule_error": float(abs(np.sum(current) - 12.0)),
        "minimum_structure_factor": float(np.min(current)),
        "M_star_spread": float(np.ptp(m_values)),
        "exact_M_star_spread": float(np.ptp(exact_m)),
        "M_star_mean_error": float(np.mean(m_values) - np.mean(exact_m)),
    }


def _dimer_star_spread(panel: dict[str, object]) -> float:
    values = [
        float(panel["dimer_M"][name]["leading_strength"])
        for name in ("M1", "M2", "M3")
    ]
    return float(np.ptp(values))


def _gamma_sector_decomposition(
    state: np.ndarray,
    data: dict[str, object],
    exact: np.ndarray,
    exact_metrics: dict[str, object],
    full_exact: np.ndarray,
    supports: dict[str, tuple[tuple[int, ...], ...]],
    reference_rdms: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, object]:
    """Split a translation state into Gamma-A1 and orthogonal components."""

    vector = np.asarray(state, dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    image = np.asarray(_project(vector, data, "gamma_a1"), dtype=np.complex128)
    weight = float(np.vdot(image, image).real)
    if not 0.0 < weight <= 1.0 + 1.0e-10:
        raise RuntimeError("Gamma-A1 projector weight lies outside [0,1]")
    gamma = image / math.sqrt(weight)
    gamma_metrics = _metrics(gamma, data, projection="none")
    complement = vector - image
    complement_weight = float(np.vdot(complement, complement).real)
    complement_energy = None
    complement_error_contribution = 0.0
    if complement_weight > 1.0e-14:
        complement /= math.sqrt(complement_weight)
        complement_energy = float(
            np.vdot(complement, data["hamiltonian"] @ complement).real
        )
        complement_error_contribution = complement_weight * (
            complement_energy - float(data["exact_energy"])
        )
    gamma_error_contribution = weight * (
        float(gamma_metrics["energy"]) - float(data["exact_energy"])
    )
    total_error = gamma_error_contribution + complement_error_contribution
    full_gamma = fixed_sector_to_full_state(gamma, data["basis"])
    return {
        "weight": weight,
        "complement_weight": complement_weight,
        "projector_closure_error": float(
            np.linalg.norm(_project(gamma, data, "gamma_a1") - gamma)
        ),
        "total_fidelity_to_exact": float(abs(np.vdot(exact, vector)) ** 2),
        "normalized_gamma_component_fidelity_to_exact": float(
            abs(np.vdot(exact, gamma)) ** 2
        ),
        "gamma_component_energy": float(gamma_metrics["energy"]),
        "gamma_component_energy_error_per_site": float(
            gamma_metrics["energy_error_per_site"]
        ),
        "complement_energy": complement_energy,
        "energy_error_decomposition": {
            "gamma_component_contribution": gamma_error_contribution,
            "orthogonal_sector_contribution": complement_error_contribution,
            "closure_error": float(
                total_error
                - (
                    float(np.vdot(vector, data["hamiltonian"] @ vector).real)
                    - float(data["exact_energy"])
                )
            ),
            "orthogonal_sector_fraction": (
                complement_error_contribution / total_error
                if total_error > 1.0e-15
                else 0.0
            ),
        },
        "structure_factor_error": _structure_factor_errors(
            gamma_metrics, exact_metrics
        ),
        "local_rdm_error": local_rdm_error_profile(
            full_gamma,
            full_exact,
            16,
            supports,
            reference_rdms=reference_rdms,
        ),
    }


def _run_p1_p2_p5(
    spec: dict[str, object], output: Path, cache_root: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    case_spec = dict(spec["p1_case"])
    case_spec["support"] = "fixed_m0"
    data = _prepare_case(case_spec, "fixed_m0")
    record = _cached_n16_ground(
        data,
        cache_root,
        recompute_if_missing=bool(spec["exact_reference"]["recompute_if_missing"]),
    )
    exact = np.asarray(record.arrays["state"], dtype=np.complex128).copy()
    exact /= np.linalg.norm(exact)
    exact_metrics = _metrics(exact, data, projection="none")
    supercell = ((4, 0), (0, 4))
    pair_actions = FixedSzPairActions(data["basis"])
    exact_panel = triangular_periodic_observable_panel(
        exact, data["basis"], supercell, pair_actions=pair_actions
    )
    exact_panel["hamiltonian_energy_from_directional_bonds_per_site"] = float(
        sum(exact_panel["orientation_bond_energy_per_site"])
        + 0.075
        * sum(exact_panel["second_neighbor_orientation_bond_energy_per_site"])
    )
    exact_panel["directional_energy_closure_error"] = float(
        exact_panel["hamiltonian_energy_from_directional_bonds_per_site"]
        - exact_metrics["energy_per_site"]
    )
    full_exact = fixed_sector_to_full_state(exact, data["basis"])
    supports = triangular_cluster_supports(supercell)
    reference_rdms = local_rdm_reference_cache(full_exact, 16, supports)

    arrays: dict[str, np.ndarray] = {
        "basis": np.asarray(data["basis"], dtype=np.int8),
        "exact_ground": exact,
    }
    trials = []
    for projection in spec["optimization"]["training_projections"]:
        for seed in spec["optimization"]["seeds"]:
            started = time.perf_counter()
            trained = train_exact_route_vit_trial(
                model=spec["model"],
                optimization=spec["optimization"],
                data=data,
                seed=int(seed),
                projection=str(projection),
            )
            arm_rows = []
            for arm in trained["arms"]:
                arm_id = str(arm["arm_id"])
                raw = np.asarray(arm.pop("raw_state"), dtype=np.complex128)
                params = arm.pop("params")
                state = _normalized_projected_state(raw, data, str(projection))
                state_key = f"seed{seed}_{projection}_{arm_id}"
                arrays[state_key] = state
                metrics = _metrics(raw, data, projection=str(projection))
                panel = triangular_periodic_observable_panel(
                    state,
                    data["basis"],
                    supercell,
                    pair_actions=pair_actions,
                )
                panel["hamiltonian_energy_from_directional_bonds_per_site"] = float(
                    sum(panel["orientation_bond_energy_per_site"])
                    + 0.075
                    * sum(panel["second_neighbor_orientation_bond_energy_per_site"])
                )
                panel["directional_energy_closure_error"] = float(
                    panel["hamiltonian_energy_from_directional_bonds_per_site"]
                    - metrics["energy_per_site"]
                )
                row: dict[str, object] = {
                    **arm,
                    "state_bundle_key": state_key,
                    "state_sha256": array_sha256(state),
                    "params_sha256": pytree_sha256(params),
                    "metrics": metrics,
                    "periodic_observables": panel,
                    "structure_factor_error": _structure_factor_errors(
                        metrics, exact_metrics
                    ),
                    "dimer_M_star_leading_spread": _dimer_star_spread(panel),
                }
                checkpoint = output / "checkpoints" / f"{state_key}.pkl"
                save_checkpoint(
                    checkpoint,
                    params,
                    metadata={
                        "task": TASK,
                        "case_id": case_spec["case_id"],
                        "seed": int(seed),
                        "training_projection": str(projection),
                        "arm_id": arm_id,
                        "params_sha256": row["params_sha256"],
                        "state_sha256": row["state_sha256"],
                    },
                )
                row["checkpoint"] = str(
                    checkpoint.relative_to(ROOT)
                    if checkpoint.is_relative_to(ROOT)
                    else checkpoint
                )
                row["checkpoint_sha256"] = file_sha256(checkpoint)
                if projection == "gamma_a1":
                    full = fixed_sector_to_full_state(state, data["basis"])
                    row["local_rdm_error"] = local_rdm_error_profile(
                        full,
                        full_exact,
                        16,
                        supports,
                        reference_rdms=reference_rdms,
                    )
                    channels = triangular_error_channel_actions(
                        state,
                        data["basis"],
                        supercell,
                        pair_actions=pair_actions,
                    )
                    row["error_channel_tomography"] = error_channel_tomography(
                        state,
                        exact,
                        data["hamiltonian"],
                        channels,
                    )
                else:
                    row["gamma_a1_decomposition"] = _gamma_sector_decomposition(
                        state,
                        data,
                        exact,
                        exact_metrics,
                        full_exact,
                        supports,
                        reference_rdms,
                    )
                arm_rows.append(row)
            trials.append(
                {
                    "seed": int(seed),
                    "training_projection": str(projection),
                    "route_ledger": trained["route_ledger"],
                    "arms": arm_rows,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            print(
                "P1",
                projection,
                f"seed={seed}",
                ", ".join(
                    f"{row['arm_id']}={row['metrics']['energy']:.9f}"
                    for row in arm_rows
                ),
                flush=True,
            )
    return (
        {
            "case_id": case_spec["case_id"],
            "geometry": "triangular",
            "rows": 4,
            "columns": 4,
            "j2_over_j1": 0.075,
            "basis_size": len(data["basis"]),
            "group_orders": {name: len(group) for name, group in data["groups"].items()},
            "exact_gamma_a1_singlet_dimension": int(
                exact_gamma_a1_spin_multiplicity_dimension(supercell, target_q=0)[
                    "dimension"
                ]
            ),
            "exact_reference": {
                "cache_key": record.cache_key,
                "cache_status": record.status,
                "energy": float(record.arrays["energy"]),
                "residual_norm": float(record.arrays["residual_norm"]),
                "metrics": exact_metrics,
                "periodic_observables": exact_panel,
                "state_sha256": array_sha256(exact),
            },
            "local_support_orbits": {
                name: {
                    "support_size": len(rows[0]),
                    "orbit_size": len(rows),
                }
                for name, rows in supports.items()
            },
            "trials": trials,
            "claim_boundary": (
                "Exact finite triangular 4x4 mechanism comparison. This torus lacks "
                "the 120-degree K point; M-star, local-RDM, and response diagnostics "
                "cannot establish a QSL, 120-AFM order, or a phase boundary."
            ),
        },
        arrays,
    )


def _recover_p1_p2_p5_from_state_bundle(
    spec: dict[str, object],
    output: Path,
    cache_root: Path,
    state_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Recover completed physics analysis after a post-training seal failure.

    The portable normalized states and trusted parameter checkpoints are
    verified.  Training histories and raw-state projection weights cannot be
    reconstructed from this bundle, so the recovered artifact marks them
    unavailable rather than silently replacing them with normalized-state
    values.
    """

    with np.load(state_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    case_spec = dict(spec["p1_case"])
    case_spec["support"] = "fixed_m0"
    data = _prepare_case(case_spec, "fixed_m0")
    if not np.array_equal(arrays["basis"], np.asarray(data["basis"], dtype=np.int8)):
        raise RuntimeError("saved state bundle basis does not match the physical case")
    exact = np.asarray(arrays["exact_ground"], dtype=np.complex128)
    exact /= np.linalg.norm(exact)
    record = _cached_n16_ground(
        data,
        cache_root,
        recompute_if_missing=bool(spec["exact_reference"]["recompute_if_missing"]),
    )
    if array_sha256(exact) != array_sha256(np.asarray(record.arrays["state"])):
        raise RuntimeError("saved exact ground state does not match the verified cache")
    exact_metrics = _metrics(exact, data, projection="none")
    supercell = ((4, 0), (0, 4))
    pair_actions = FixedSzPairActions(data["basis"])
    exact_panel = triangular_periodic_observable_panel(
        exact, data["basis"], supercell, pair_actions=pair_actions
    )
    exact_panel["hamiltonian_energy_from_directional_bonds_per_site"] = float(
        sum(exact_panel["orientation_bond_energy_per_site"])
        + 0.075
        * sum(exact_panel["second_neighbor_orientation_bond_energy_per_site"])
    )
    exact_panel["directional_energy_closure_error"] = float(
        exact_panel["hamiltonian_energy_from_directional_bonds_per_site"]
        - exact_metrics["energy_per_site"]
    )
    full_exact = fixed_sector_to_full_state(exact, data["basis"])
    supports = triangular_cluster_supports(supercell)
    reference_rdms = local_rdm_reference_cache(full_exact, 16, supports)
    arm_ids = (
        "single_tree_d4",
        "route_vit_coefficients_d4",
        "route_vit_conditioned_d4",
        "route_vit_scrambled_tokens_d4",
        "route_local_conditioned_d4",
        "single_tree_matched_width",
    )
    trials = []
    for projection in spec["optimization"]["training_projections"]:
        for seed in spec["optimization"]["seeds"]:
            arm_rows = []
            for arm_id in arm_ids:
                key = f"seed{seed}_{projection}_{arm_id}"
                if key not in arrays:
                    raise RuntimeError(f"saved state bundle is missing {key}")
                state = np.asarray(arrays[key], dtype=np.complex128).copy()
                stored_state_hash = array_sha256(state)
                checkpoint = output / "checkpoints" / f"{key}.pkl"
                restored = load_checkpoint(checkpoint)
                params = restored["params"]
                metadata = restored.get("metadata", {})
                state_hash = stored_state_hash
                params_hash = pytree_sha256(params)
                if metadata.get("state_sha256") != state_hash:
                    raise RuntimeError(f"checkpoint state hash failed for {key}")
                if metadata.get("params_sha256") != params_hash:
                    raise RuntimeError(f"checkpoint parameter hash failed for {key}")
                state /= np.linalg.norm(state)
                metrics = _metrics(state, data, projection="none")
                metrics["selected_weight_of_raw"] = None
                metrics["raw_projection_weight_unavailable_after_recovery"] = True
                metrics["projection_closure_error"] = float(
                    np.linalg.norm(_project(state, data, str(projection)) - state)
                )
                panel = triangular_periodic_observable_panel(
                    state,
                    data["basis"],
                    supercell,
                    pair_actions=pair_actions,
                )
                panel["hamiltonian_energy_from_directional_bonds_per_site"] = float(
                    sum(panel["orientation_bond_energy_per_site"])
                    + 0.075
                    * sum(panel["second_neighbor_orientation_bond_energy_per_site"])
                )
                panel["directional_energy_closure_error"] = float(
                    panel["hamiltonian_energy_from_directional_bonds_per_site"]
                    - metrics["energy_per_site"]
                )
                row: dict[str, object] = {
                    "arm_id": arm_id,
                    "state_bundle_key": key,
                    "state_sha256": state_hash,
                    "params_sha256": params_hash,
                    "parameter_count": int(
                        sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params))
                    ),
                    "checkpoint": str(
                        checkpoint.resolve().relative_to(ROOT.resolve())
                        if checkpoint.resolve().is_relative_to(ROOT.resolve())
                        else checkpoint.resolve()
                    ),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "metrics": metrics,
                    "periodic_observables": panel,
                    "structure_factor_error": _structure_factor_errors(
                        metrics, exact_metrics
                    ),
                    "dimer_M_star_leading_spread": _dimer_star_spread(panel),
                    "training_history_status": "unavailable_after_post_training_seal_failure",
                }
                if projection == "gamma_a1":
                    full = fixed_sector_to_full_state(state, data["basis"])
                    row["local_rdm_error"] = local_rdm_error_profile(
                        full,
                        full_exact,
                        16,
                        supports,
                        reference_rdms=reference_rdms,
                    )
                    channels = triangular_error_channel_actions(
                        state,
                        data["basis"],
                        supercell,
                        pair_actions=pair_actions,
                    )
                    row["error_channel_tomography"] = error_channel_tomography(
                        state, exact, data["hamiltonian"], channels
                    )
                else:
                    row["gamma_a1_decomposition"] = _gamma_sector_decomposition(
                        state,
                        data,
                        exact,
                        exact_metrics,
                        full_exact,
                        supports,
                        reference_rdms,
                    )
                arm_rows.append(row)
            trials.append(
                {
                    "seed": int(seed),
                    "training_projection": str(projection),
                    "arms": arm_rows,
                    "recovered_from_completed_state_bundle": True,
                }
            )
    return (
        {
            "case_id": case_spec["case_id"],
            "geometry": "triangular",
            "rows": 4,
            "columns": 4,
            "j2_over_j1": 0.075,
            "basis_size": len(data["basis"]),
            "group_orders": {name: len(group) for name, group in data["groups"].items()},
            "exact_gamma_a1_singlet_dimension": int(
                exact_gamma_a1_spin_multiplicity_dimension(supercell, target_q=0)[
                    "dimension"
                ]
            ),
            "exact_reference": {
                "cache_key": record.cache_key,
                "cache_status": record.status,
                "energy": float(record.arrays["energy"]),
                "residual_norm": float(record.arrays["residual_norm"]),
                "metrics": exact_metrics,
                "periodic_observables": exact_panel,
                "state_sha256": array_sha256(exact),
            },
            "local_support_orbits": {
                name: {"support_size": len(rows[0]), "orbit_size": len(rows)}
                for name, rows in supports.items()
            },
            "trials": trials,
            "state_bundle_recovery": {
                "used": True,
                "reason": "all 36 checkpoints and 38 state arrays predated JSON path seal failure",
                "training_histories_recoverable": False,
                "raw_projection_weights_recoverable": False,
            },
            "claim_boundary": (
                "Exact finite triangular 4x4 mechanism comparison. This torus lacks "
                "the 120-degree K point; M-star, local-RDM, and response diagnostics "
                "cannot establish a QSL, 120-AFM order, or a phase boundary."
            ),
        },
        arrays,
    )


def _sector_eigensystem(
    hamiltonian: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    projected = basis.conj().T @ hamiltonian @ basis
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = np.linalg.eigh(projected)
    return values, basis @ vectors


def _response_summary(result, values: np.ndarray, ground_energy: float) -> dict[str, object]:
    weights = np.asarray(result.weights, dtype=np.float64)
    gaps = np.asarray(values, dtype=np.float64) - float(ground_energy)
    inverse = float(np.sum(weights / gaps))
    static = 2.0 * inverse
    fidelity = float(np.sum(weights / gaps**2))
    cumulative = np.cumsum(weights) / result.m0
    return {
        "m0": result.m0,
        "m1": result.m1,
        "mean_excitation": result.mean_excitation,
        "lowest_gap": result.lowest_gap,
        "lowest_pole_fraction": result.lowest_pole_fraction,
        "first_three_pole_fraction": float(cumulative[min(2, len(cumulative) - 1)]),
        "static_response": static,
        "fidelity_response": fidelity,
        "harmonic_gap": result.m0 / inverse,
        "m0_closure_error": result.m0_closure_error,
        "m1_closure_error": result.m1_closure_error,
    }


def _all_plus_basis(supercell, momentum, *, target_q: int) -> tuple[np.ndarray, np.ndarray]:
    characters = {name: 1 for name in reciprocal_little_group(supercell, momentum)}
    projector = little_group_character_projector(
        supercell, momentum, characters, target_q=target_q
    )
    return projector, projector_basis(projector)


def _cached_n12_eigensystem(
    supercell,
    j2: float,
    cache_root: Path,
    *,
    recompute_if_missing: bool,
):
    bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=float(j2))
    identity = {
        "purpose": "triangular_N12_HNF_6_4_0_2_q0_q2_eigensystem",
        "j2_over_j1": float(j2),
        "bonds": [list(row) for row in bonds],
        "solver": "numpy.linalg.eigh_dense_fusion_v1",
    }

    def compute():
        h0 = heisenberg_graph_fusion_matrix(12, bonds, target_q=0).toarray()
        h2 = heisenberg_graph_fusion_matrix(12, bonds, target_q=2).toarray()
        values0, vectors0 = np.linalg.eigh(h0)
        values2, vectors2 = np.linalg.eigh(h2)
        return {
            "values_q0": values0,
            "vectors_q0": vectors0,
            "values_q2": values2,
            "vectors_q2": vectors2,
        }

    try:
        record = load_or_recompute_exact_reference(cache_root, identity)
    except ExactReferenceCacheMiss:
        if not recompute_if_missing:
            raise
        record = load_or_recompute_exact_reference(
            cache_root, identity, recompute=True, compute=compute
        )
    h0 = heisenberg_graph_fusion_matrix(12, bonds, target_q=0).toarray()
    h2 = heisenberg_graph_fusion_matrix(12, bonds, target_q=2).toarray()
    return record, h0, h2


def _run_p3_p4(
    spec: dict[str, object], cache_root: Path
) -> dict[str, object]:
    supercell = ((6, 4), (0, 2))
    gamma = (Fraction(0), Fraction(0))
    x = (Fraction(1, 6), Fraction(1, 6))
    k = (Fraction(1, 3), Fraction(1, 3))
    m = (Fraction(1, 2), Fraction(0))
    p_gamma = momentum_projector_matrix(supercell, (0, 0), target_q=0)
    d6 = d6_character_projectors(supercell, target_q=0)
    p_e2 = 0.5 * (p_gamma @ d6["E2"] + (p_gamma @ d6["E2"]).conj().T)
    p_a2 = 0.5 * (p_gamma @ d6["A2"] + (p_gamma @ d6["A2"]).conj().T)
    b_e2 = projector_basis(p_e2)
    b_a2 = projector_basis(p_a2)
    p_x_plus = little_group_character_projector(
        supercell, x, {"C6^0": 1, "Mx*C6^4": 1}, target_q=0
    )
    p_x_minus = little_group_character_projector(
        supercell, x, {"C6^0": 1, "Mx*C6^4": -1}, target_q=0
    )
    b_x_plus = projector_basis(p_x_plus)
    b_x_minus = projector_basis(p_x_minus)
    p_k_triplet, b_k_triplet = _all_plus_basis(supercell, k, target_q=2)
    p_m_triplet, b_m_triplet = _all_plus_basis(supercell, m, target_q=2)
    c_e2 = np.asarray(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
            [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
            [0.0, -2.0 / math.sqrt(6.0)],
        ]
    )
    c_x_plus = np.asarray(
        [
            [1.0 / math.sqrt(2.0), 0.0],
            [1.0 / math.sqrt(2.0), 0.0],
            [0.0, 1.0],
        ]
    )
    frozen_x = np.asarray((0.995836786770, 0.091154232575))
    sites = hnf_sites(supercell)
    h2_shell = heisenberg_graph_fusion_matrix(
        12,
        triangular_hnf_j1j2_bonds(supercell, j1=0.0, j2=1.0),
        target_q=0,
    ).toarray()
    triangles = hnf_oriented_triangles(supercell)
    cases = []
    for j2 in spec["spectroscopy"]["j2_values"]:
        record, h0, h2 = _cached_n12_eigensystem(
            supercell,
            float(j2),
            cache_root,
            recompute_if_missing=bool(spec["exact_reference"]["recompute_if_missing"]),
        )
        values0 = np.asarray(record.arrays["values_q0"])
        vectors0 = np.asarray(record.arrays["vectors_q0"])
        ground_energy = float(values0[0])
        ground = vectors0[:, 0]

        e_e2, v_e2 = _sector_eigensystem(h0, b_e2)
        gamma_bonds = bond_center_orientation_operators(
            supercell, gamma, J1_POSITIVE, target_q=0
        )
        sources_e2 = np.column_stack([operator @ ground for operator in gamma_bonds]) @ c_e2
        e2_fraction, _ = maximum_low_subspace_source_fraction(
            sources_e2, v_e2[:, :2]
        )
        e2_result = source_spectrum(
            h0, ground_energy, sources_e2[:, 0], e_e2, v_e2
        )
        e2_summary = _response_summary(e2_result, e_e2, ground_energy)
        e2_summary.update(
            {
                "lowest_doublet_fraction": e2_fraction,
                "doublet_splitting": float(e_e2[1] - e_e2[0]),
                "next_E2_gap": float(e_e2[2] - ground_energy),
            }
        )

        e_xp, v_xp = _sector_eigensystem(h0, b_x_plus)
        x_bonds = bond_center_orientation_operators(
            supercell, x, J1_POSITIVE, target_q=0
        )
        raw_x = np.column_stack([operator @ ground for operator in x_bonds])
        sources_xp = p_x_plus @ (raw_x @ c_x_plus)
        fraction_x1, _ = maximum_low_subspace_source_fraction(
            sources_xp, v_xp[:, :1]
        )
        fraction_x3, _ = maximum_low_subspace_source_fraction(
            sources_xp, v_xp[:, : min(3, v_xp.shape[1])]
        )
        frozen_source = sources_xp @ frozen_x
        xp_result = source_spectrum(
            h0, ground_energy, frozen_source, e_xp, v_xp
        )
        xp_summary = _response_summary(xp_result, e_xp, ground_energy)
        xp_summary.update(
            {
                "adaptive_lowest_fraction": fraction_x1,
                "adaptive_first_three_fraction": fraction_x3,
                "frozen_coefficients_in_Xplus_basis": frozen_x.tolist(),
                "frozen_orientation_coefficients": (c_x_plus @ frozen_x).tolist(),
            }
        )

        e_xm, v_xm = _sector_eigensystem(h0, b_x_minus)
        sources_xm = p_x_minus @ raw_x
        _fraction_xm, coefficients_xm = maximum_low_subspace_source_fraction(
            sources_xm, v_xm[:, :1]
        )
        xm_source = sources_xm @ coefficients_xm
        xm_result = source_spectrum(
            h0, ground_energy, xm_source, e_xm, v_xm
        )
        xm_summary = _response_summary(xm_result, e_xm, ground_energy)

        spin_rows = {}
        for name, momentum, projector, sector_basis in (
            ("K", k, p_k_triplet, b_k_triplet),
            ("M", m, p_m_triplet, b_m_triplet),
        ):
            weights = np.asarray(
                [
                    np.exp(
                        2.0j
                        * np.pi
                        * float(momentum[0] * site[0] + momentum[1] * site[1])
                    )
                    for site in sites
                ]
            )
            operator = compile_sector_spin_vector(
                weights, q_new=2, q_old=0
            ).matrix / math.sqrt(3.0)
            source = np.asarray(operator @ ground)
            source_closure = float(
                np.linalg.norm(source - projector @ source)
                / max(np.linalg.norm(source), 1.0e-30)
            )
            energies, states = _sector_eigensystem(h2, sector_basis)
            result = source_spectrum(
                h2, ground_energy, source, energies, states
            )
            row = _response_summary(result, energies, ground_energy)
            row.update(
                {
                    "source_sector_closure_error": source_closure,
                    "spin_structure_factor_isotropic": 3.0 * result.m0 / 12.0,
                    "static_response_isotropic_per_site": 3.0 * row["static_response"] / 12.0,
                    "fidelity_response_isotropic_per_site": 3.0 * row["fidelity_response"] / 12.0,
                }
            )
            spin_rows[name] = row

        h2_mean = complex(np.vdot(ground, h2_shell @ ground))
        j2_source = h2_shell @ ground - h2_mean * ground
        fidelity_result = source_spectrum(
            h0,
            ground_energy,
            j2_source,
            values0[1:],
            vectors0[:, 1:],
        )
        fidelity_summary = _response_summary(
            fidelity_result, values0[1:], ground_energy
        )
        fidelity_summary.update(
            {
                "dE_dJ2": float(h2_mean.real),
                "d2E_dJ2_squared": -float(fidelity_summary["static_response"]),
            }
        )

        e_a2, v_a2 = _sector_eigensystem(h0, b_a2)
        chirality_source = matrix_free_chirality_action(
            ground,
            12,
            triangles,
            target_q=0,
        )
        chirality_source = p_a2 @ chirality_source
        chirality_result = source_spectrum(
            h0, ground_energy, chirality_source, e_a2, v_a2
        )
        chirality_summary = _response_summary(
            chirality_result, e_a2, ground_energy
        )
        cases.append(
            {
                "j2_over_j1": float(j2),
                "ground_energy": ground_energy,
                "ground_energy_per_site": ground_energy / 12.0,
                "cache_key": record.cache_key,
                "cache_status": record.status,
                "Gamma_E2_bond_nematic": e2_summary,
                "X_plus_bond_branch": xp_summary,
                "X_minus_control": xm_summary,
                "spin_response": spin_rows,
                "J2_fidelity_response": fidelity_summary,
                "Gamma_A2_uniform_chirality_response": chirality_summary,
                "projector_H_commutator_max": float(
                    max(
                        np.linalg.norm(h0 @ p_e2 - p_e2 @ h0),
                        np.linalg.norm(h0 @ p_x_plus - p_x_plus @ h0),
                        np.linalg.norm(h0 @ p_x_minus - p_x_minus @ h0),
                        np.linalg.norm(h2 @ p_k_triplet - p_k_triplet @ h2),
                        np.linalg.norm(h2 @ p_m_triplet - p_m_triplet @ h2),
                        np.linalg.norm(h0 @ p_a2 - p_a2 @ h0),
                    )
                ),
            }
        )
        print(
            "P3/P4",
            f"J2={float(j2):.3f}",
            f"Gamma-E2={e2_summary['lowest_gap']:.6f}",
            f"X+={xp_summary['lowest_gap']:.6f}",
            f"Ktriplet={spin_rows['K']['lowest_gap']:.6f}",
            f"Mtriplet={spin_rows['M']['lowest_gap']:.6f}",
            flush=True,
        )
    return {
        "supercell_hnf": [[6, 4], [0, 2]],
        "n_sites": 12,
        "singlet_multiplicity_dimension": 132,
        "triplet_multiplicity_dimension": 297,
        "geometry_warning": (
            "Full-D6 thin HNF torus: 36 J1 edges are distinct, while 36 J2 "
            "periodic images alias to 18 distinct edges of weight two."
        ),
        "sector_definitions": {
            "Gamma_E2": {"rank": int(b_e2.shape[1])},
            "X_plus": {
                "momentum_fraction": [1, 6, 1, 6],
                "character_numerator_over_12": list(
                    hnf_fraction_character_numerator(supercell, x)
                ),
                "rank": int(b_x_plus.shape[1]),
                "little_group_characters": {"C6^0": 1, "Mx*C6^4": 1},
            },
            "X_minus": {"rank": int(b_x_minus.shape[1])},
            "K_triplet_all_plus": {"rank": int(b_k_triplet.shape[1])},
            "M_triplet_all_plus": {"rank": int(b_m_triplet.shape[1])},
            "Gamma_A2_chirality": {"rank": int(b_a2.shape[1])},
        },
        "projector_audit": {
            "maximum_hermiticity_error": float(
                max(
                    np.linalg.norm(projector - projector.conj().T)
                    for projector in (p_e2, p_x_plus, p_x_minus, p_k_triplet, p_m_triplet, p_a2)
                )
            ),
            "maximum_idempotency_error": float(
                max(
                    np.linalg.norm(projector @ projector - projector)
                    for projector in (p_e2, p_x_plus, p_x_minus, p_k_triplet, p_m_triplet, p_a2)
                )
            ),
        },
        "cases": cases,
        "claim_boundary": (
            "The X+ label denotes an exact lattice momentum/little-group sector and "
            "a bond-response branch. Calling it a monopole remains a candidate "
            "interpretation pending an independent PSG quantum-number match. N=12 "
            "soft modes and gap crossings do not establish thermodynamic order or a "
            "phase boundary."
        ),
    }


def _summary(p1: dict[str, object], p3: dict[str, object]) -> dict[str, object]:
    arm_ids = (
        "single_tree_d4",
        "route_vit_coefficients_d4",
        "route_vit_conditioned_d4",
        "route_vit_scrambled_tokens_d4",
        "route_local_conditioned_d4",
        "single_tree_matched_width",
    )
    projections = {}
    for projection in ("translation", "gamma_a1"):
        rows = {}
        selected = [
            trial for trial in p1["trials"] if trial["training_projection"] == projection
        ]
        for arm_id in arm_ids:
            arms = [
                next(row for row in trial["arms"] if row["arm_id"] == arm_id)
                for trial in selected
            ]
            rows[arm_id] = {
                "median_energy_error_per_site": float(
                    np.median([row["metrics"]["energy_error_per_site"] for row in arms])
                ),
                "median_structure_factor_rms_error": float(
                    np.median([row["structure_factor_error"]["rms_error"] for row in arms])
                ),
                "median_M_star_spread": float(
                    np.median([row["structure_factor_error"]["M_star_spread"] for row in arms])
                ),
            }
            if projection == "gamma_a1":
                rows[arm_id].update(
                    {
                        "median_rhombus_RDM_trace_distance": float(
                            np.median(
                                [
                                    row["local_rdm_error"]["rhombus_4"]["trace_distance_mean"]
                                    for row in arms
                                ]
                            )
                        ),
                        "median_half_torus_RDM_trace_distance": float(
                            np.median(
                                [
                                    row["local_rdm_error"]["half_torus_8"]["trace_distance_mean"]
                                    for row in arms
                                ]
                            )
                        ),
                        "median_physical_channel_error_fraction": float(
                            np.median(
                                [
                                    row["error_channel_tomography"]["combined"]["exact_error_fraction"]
                                    for row in arms
                                ]
                            )
                        ),
                    }
                )
            else:
                rows[arm_id].update(
                    {
                        "median_gamma_a1_weight": float(
                            np.median(
                                [row["gamma_a1_decomposition"]["weight"] for row in arms]
                            )
                        ),
                        "median_gamma_component_fidelity_to_exact": float(
                            np.median(
                                [
                                    row["gamma_a1_decomposition"][
                                        "normalized_gamma_component_fidelity_to_exact"
                                    ]
                                    for row in arms
                                ]
                            )
                        ),
                        "median_orthogonal_sector_energy_error_fraction": float(
                            np.median(
                                [
                                    row["gamma_a1_decomposition"][
                                        "energy_error_decomposition"
                                    ]["orthogonal_sector_fraction"]
                                    for row in arms
                                ]
                            )
                        ),
                        "median_gamma_component_half_torus_RDM_trace_distance": float(
                            np.median(
                                [
                                    row["gamma_a1_decomposition"]["local_rdm_error"][
                                        "half_torus_8"
                                    ]["trace_distance_mean"]
                                    for row in arms
                                ]
                            )
                        ),
                    }
                )
        projections[projection] = rows
    p3_cases = p3["cases"]
    return {
        "p1_p2_p5_by_projection_and_arm": projections,
        "p3_p4_trends": {
            "Gamma_E2_gap_first_last": [
                p3_cases[0]["Gamma_E2_bond_nematic"]["lowest_gap"],
                p3_cases[-1]["Gamma_E2_bond_nematic"]["lowest_gap"],
            ],
            "Gamma_E2_static_response_first_last": [
                p3_cases[0]["Gamma_E2_bond_nematic"]["static_response"],
                p3_cases[-1]["Gamma_E2_bond_nematic"]["static_response"],
            ],
            "X_plus_gap_first_last": [
                p3_cases[0]["X_plus_bond_branch"]["lowest_gap"],
                p3_cases[-1]["X_plus_bond_branch"]["lowest_gap"],
            ],
            "K_M_triplet_gap_crossing_bracket": [0.1, 0.125],
        },
    }


def run_spin_physics_interpretation(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run, checkpoint, and seal the complete local physics interpretation."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    source_manifest = transitive_local_source_manifest(
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
        "source_manifest": source_manifest,
        "source_manifest_sha256": canonical_sha256(source_manifest),
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": True,
        "exact_diagonalization_scope": (
            "One cached N=16 fixed-M0 ground vector and five cached N=12 q=0/q=2 "
            "multiplicity-space eigensystems; no large-system ED."
        ),
    }
    atomic_json(result_path, payload)
    cache_root = PROJECT / str(spec["exact_reference"]["cache_directory"])
    expected_state_count = 2 + (
        len(spec["optimization"]["training_projections"])
        * len(spec["optimization"]["seeds"])
        * 6
    )
    if state_path.is_file():
        with np.load(state_path, allow_pickle=False) as archive:
            complete_bundle = len(archive.files) == expected_state_count
        complete_checkpoints = len(tuple((output / "checkpoints").glob("*.pkl"))) == 36
    else:
        complete_bundle = False
        complete_checkpoints = False
    if complete_bundle and complete_checkpoints:
        p1, arrays = _recover_p1_p2_p5_from_state_bundle(
            spec, output, cache_root, state_path
        )
    else:
        p1, arrays = _run_p1_p2_p5(spec, output, cache_root)
        _atomic_npz(state_path, arrays)
    payload["state_bundle"] = {
        "path": str(
            state_path.resolve().relative_to(ROOT.resolve())
            if state_path.resolve().is_relative_to(ROOT.resolve())
            else state_path.resolve()
        ),
        "sha256": file_sha256(state_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
            }
            for name, value in arrays.items()
        },
    }
    payload["P1_P2_P5"] = p1
    atomic_json(result_path, payload)
    p3 = _run_p3_p4(spec, cache_root)
    payload["P3_P4"] = p3
    payload["summary"] = _summary(p1, p3)
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_physics_interpretation"]
