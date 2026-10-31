"""Read-only audit of symmetry-labelled ``E2`` resonance carriers.

The workflow never retrains a state or diagonalizes a Hamiltonian.  It reuses
the sealed N=16 state bundle, constructs the convention-fixed bond-nematic
doublet ``N_a``, and compares four nested Rayleigh--Ritz spaces:

``baseline``
    the cached variational state;
``e2_pair``
    ``span{psi, (K_A1-<K_A1>)psi}``;
``hamiltonian_krylov``
    the generic same-one-direction energy control ``span{psi,(H-E)psi}``;
``joint``
    all three directions.

Equal-time ``m0`` and the SMA gap are reported separately from susceptibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import physical_permutation_index_table
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.triangular_hnf_space_group import (
    exact_gamma_a1_spin_multiplicity_dimension,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity
from su2_nqs.lattices.triangular_e2 import e2_orientation_representation
from su2_nqs.lattices.triangular_hnf import (
    J1_POSITIVE,
    J2_POSITIVE,
    hnf_point_group_permutations,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.triangular_resonance import (
    connected_triangular_a1_pair_action,
    connected_triangular_e2_pair_action,
    triangular_e2_actions,
    triangular_e2_pair_action,
    triangular_e2_moments,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs

from projects.clebsch_tree_nqs.workflows.spin import _prepare_case, _project


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_e2_resonance_carrier_audit"


def _normalized(vector: np.ndarray) -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if norm <= 1.0e-14:
        raise ValueError("state has zero numerical norm")
    return state / norm


def _ritz(
    state: np.ndarray,
    directions: Sequence[np.ndarray],
    hamiltonian,
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Solve the lowest Ritz vector in a small, possibly dependent span."""

    vectors = [_normalized(state)] + [np.asarray(row, dtype=np.complex128) for row in directions]
    matrix = np.column_stack(vectors)
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] <= 0.0:
        raise ValueError("Ritz span has zero norm")
    retained = singular / singular[0] > tolerance
    basis = left[:, retained]
    action = np.column_stack([hamiltonian @ basis[:, index] for index in range(basis.shape[1])])
    projected = basis.conj().T @ action
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors_small = np.linalg.eigh(projected)
    result = _normalized(basis @ vectors_small[:, 0])
    phase = np.vdot(_normalized(state), result)
    if abs(phase) > 1.0e-14:
        result *= np.exp(-1.0j * np.angle(phase))
    return {
        "state": result,
        "energy": float(values[0]),
        "retained_dimension": int(np.count_nonzero(retained)),
        "singular_value_ratios": (singular / singular[0]).tolist(),
    }


def _state_metrics(
    state: np.ndarray,
    *,
    exact: np.ndarray,
    exact_energy: float,
    exact_m0: float,
    exact_sma_gap: float,
    data: dict[str, object],
    pair_actions: FixedSzPairActions,
    supercell,
) -> dict[str, float]:
    vector = _normalized(state)
    action = data["hamiltonian"] @ vector
    energy = float(np.vdot(vector, action).real)
    variance = float(np.vdot(action, action).real - energy**2)
    moments = triangular_e2_moments(
        vector,
        pair_actions.pair,
        supercell,
        lambda candidate: data["hamiltonian"] @ candidate,
    )
    gamma_image = np.asarray(_project(vector, data, "gamma_a1"), dtype=np.complex128)
    return {
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_error_per_site": (energy - exact_energy) / int(data["n_sites"]),
        "variance": max(variance, 0.0),
        "fidelity_to_exact": float(abs(np.vdot(exact, vector)) ** 2),
        "gamma_a1_weight": float(np.vdot(gamma_image, gamma_image).real),
        "e2_m0": moments.m0,
        "e2_m0_error": moments.m0 - exact_m0,
        "e2_m0_abs_error": abs(moments.m0 - exact_m0),
        "e2_m1": moments.m1,
        "e2_sma_gap": moments.sma_gap,
        "e2_sma_gap_error": moments.sma_gap - exact_sma_gap,
        "e2_sma_gap_abs_error": abs(moments.sma_gap - exact_sma_gap),
    }


def _n12_symmetry_certificate() -> dict[str, float]:
    """Certify the E2 doublet and A1 pair in an independent full-D6 sector."""

    supercell = ((6, 4), (0, 2))
    basis = tuple(enumerate_sz_configs(12, 6))
    group = triangular_hnf_gamma_a1_group(supercell)
    table = physical_permutation_index_table(basis, group)
    rng = np.random.default_rng(2026091803)
    raw = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    gamma = np.mean(raw[table], axis=0)
    gamma = _normalized(gamma)
    actions = FixedSzPairActions(basis)
    doublet = np.stack(triangular_e2_actions(gamma, actions.pair, supercell))
    covariance_errors = []
    for _name, operation, site_image in hnf_point_group_permutations(supercell):
        indices = physical_permutation_index_table(basis, (site_image,))[0]
        expected = e2_orientation_representation(operation) @ doublet
        covariance_errors.append(float(np.linalg.norm(doublet[:, indices] - expected)))
    pair = triangular_e2_pair_action(gamma, actions.pair, supercell)
    pair_norm = max(float(np.linalg.norm(pair)), 1.0e-30)
    pair_leakage = max(
        float(np.linalg.norm(pair[indices] - pair) / pair_norm) for indices in table
    )
    cross = triangular_e2_pair_action(
        gamma,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    cross_norm = max(float(np.linalg.norm(cross)), 1.0e-30)
    cross_leakage = max(
        float(np.linalg.norm(cross[indices] - cross) / cross_norm) for indices in table
    )
    return {
        "gamma_input_invariance_max_abs_norm": float(
            max(np.linalg.norm(gamma[indices] - gamma) for indices in table)
        ),
        "e2_operator_covariance_max_abs_norm": max(covariance_errors),
        "j1_j1_pair_a1_relative_leakage": pair_leakage,
        "j1_j2_pair_a1_relative_leakage": cross_leakage,
    }


def run_spin_e2_resonance_audit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the registered cached-state carrier audit."""

    spec = json.loads(config_path.read_text())
    if spec.get("schema_version") != 1 or spec.get("task") != TASK:
        raise ValueError(f"config must select schema_version=1 and task={TASK}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    state_path = ROOT / str(spec["state_bundle"])
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    case = dict(spec["case"])
    data = _prepare_case(case, "fixed_m0")
    supercell = ((4, 0), (0, 4))
    pair_actions = FixedSzPairActions(data["basis"])
    bundle = np.load(state_path)
    stored_basis = np.asarray(bundle["basis"], dtype=np.int8)
    if not np.array_equal(stored_basis, np.asarray(data["basis"], dtype=np.int8)):
        raise RuntimeError("cached state bundle and reconstructed basis differ")
    exact = _normalized(bundle["exact_ground"])
    exact_energy = float(np.vdot(exact, data["hamiltonian"] @ exact).real)
    if abs(exact_energy - float(case["exact_energy_anchor"])) > 1.0e-10:
        raise RuntimeError("cached exact state fails the registered energy anchor")
    exact_moments = triangular_e2_moments(
        exact,
        pair_actions.pair,
        supercell,
        lambda candidate: data["hamiltonian"] @ candidate,
    )

    rows = []
    for projection in spec["projections"]:
        for seed in spec["seeds"]:
            key = f"seed{int(seed)}_{projection}_route_vit_conditioned_d4"
            if key not in bundle:
                raise KeyError(f"cached state is absent: {key}")
            baseline = _normalized(bundle[key])
            baseline_energy = float(
                np.vdot(baseline, data["hamiltonian"] @ baseline).real
            )
            carrier, carrier_mean = connected_triangular_e2_pair_action(
                baseline, pair_actions.pair, supercell
            )
            carrier_12, carrier_12_mean = connected_triangular_e2_pair_action(
                baseline,
                pair_actions.pair,
                supercell,
                first_displacements=J1_POSITIVE,
                second_displacements=J2_POSITIVE,
            )
            carrier_22, carrier_22_mean = connected_triangular_e2_pair_action(
                baseline,
                pair_actions.pair,
                supercell,
                first_displacements=J2_POSITIVE,
            )
            a1_11, a1_11_mean = connected_triangular_a1_pair_action(
                baseline,
                pair_actions.pair,
                supercell,
                first_displacements=J1_POSITIVE,
            )
            a1_12, a1_12_mean = connected_triangular_a1_pair_action(
                baseline,
                pair_actions.pair,
                supercell,
                first_displacements=J1_POSITIVE,
                second_displacements=J2_POSITIVE,
            )
            a1_22, a1_22_mean = connected_triangular_a1_pair_action(
                baseline,
                pair_actions.pair,
                supercell,
                first_displacements=J2_POSITIVE,
            )
            residual = data["hamiltonian"] @ baseline - baseline_energy * baseline
            krylov_directions = [residual]
            for _ in range(2):
                following = data["hamiltonian"] @ krylov_directions[-1]
                following -= np.vdot(baseline, following) * baseline
                krylov_directions.append(following)
            variants = {
                "baseline": {"state": baseline, "retained_dimension": 1},
                "e2_pair": _ritz(baseline, (carrier,), data["hamiltonian"]),
                "e2_shell_bank": _ritz(
                    baseline,
                    (carrier, carrier_12, carrier_22),
                    data["hamiltonian"],
                ),
                "a1_shell_bank_same_work": _ritz(
                    baseline,
                    (a1_11, a1_12, a1_22),
                    data["hamiltonian"],
                ),
                "joint_e2_a1_shell_banks": _ritz(
                    baseline,
                    (carrier, carrier_12, carrier_22, a1_11, a1_12, a1_22),
                    data["hamiltonian"],
                ),
                "hamiltonian_krylov": _ritz(
                    baseline, (residual,), data["hamiltonian"]
                ),
                "hamiltonian_krylov_matched": _ritz(
                    baseline, krylov_directions, data["hamiltonian"]
                ),
                "joint": _ritz(
                    baseline, (carrier, residual), data["hamiltonian"]
                ),
                "joint_shell_bank_matched_krylov": _ritz(
                    baseline,
                    (carrier, carrier_12, carrier_22, *krylov_directions),
                    data["hamiltonian"],
                ),
            }
            variant_rows = {}
            for name, variant in variants.items():
                state = np.asarray(variant.pop("state"), dtype=np.complex128)
                variant_rows[name] = {
                    **variant,
                    **_state_metrics(
                        state,
                        exact=exact,
                        exact_energy=exact_energy,
                        exact_m0=exact_moments.m0,
                        exact_sma_gap=exact_moments.sma_gap,
                        data=data,
                        pair_actions=pair_actions,
                        supercell=supercell,
                    ),
                }
            base = variant_rows["baseline"]
            carrier_row = variant_rows["e2_pair"]
            generic = variant_rows["hamiltonian_krylov"]
            joint = variant_rows["joint"]
            bank = variant_rows["e2_shell_bank"]
            matched_generic = variant_rows["hamiltonian_krylov_matched"]
            matched_joint = variant_rows["joint_shell_bank_matched_krylov"]
            same_work = variant_rows["a1_shell_bank_same_work"]
            channel_joint = variant_rows["joint_e2_a1_shell_banks"]
            rows.append(
                {
                    "seed": int(seed),
                    "projection": str(projection),
                    "state_bundle_key": key,
                    "carrier_expectation": {
                        "real": float(carrier_mean.real),
                        "imaginary": float(carrier_mean.imag),
                    },
                    "cross_shell_carrier_expectation": {
                        "real": float(carrier_12_mean.real),
                        "imaginary": float(carrier_12_mean.imag),
                    },
                    "j2_shell_carrier_expectation": {
                        "real": float(carrier_22_mean.real),
                        "imaginary": float(carrier_22_mean.imag),
                    },
                    "a1_pair_expectations": {
                        "G11": {"real": float(a1_11_mean.real), "imaginary": float(a1_11_mean.imag)},
                        "G12": {"real": float(a1_12_mean.real), "imaginary": float(a1_12_mean.imag)},
                        "G22": {"real": float(a1_22_mean.real), "imaginary": float(a1_22_mean.imag)},
                    },
                    "variants": variant_rows,
                    "gates": {
                        "carrier_lowers_energy": carrier_row["energy"] < base["energy"] - 1.0e-12,
                        "carrier_repairs_equal_time_m0": carrier_row["e2_m0_abs_error"] < base["e2_m0_abs_error"],
                        "carrier_repairs_sma_gap": carrier_row["e2_sma_gap_abs_error"] < base["e2_sma_gap_abs_error"],
                        "carrier_improves_fidelity": carrier_row["fidelity_to_exact"] > base["fidelity_to_exact"],
                        "carrier_beats_generic_energy_control": carrier_row["energy"] < generic["energy"],
                        "carrier_adds_after_generic_control": joint["energy"] < generic["energy"] - 1.0e-12,
                        "gamma_weight_preserved_to_1e_minus_10": abs(carrier_row["gamma_a1_weight"] - base["gamma_a1_weight"]) < 1.0e-10,
                        "shell_bank_lowers_energy": bank["energy"] < base["energy"] - 1.0e-12,
                        "shell_bank_repairs_equal_time_m0": bank["e2_m0_abs_error"] < base["e2_m0_abs_error"],
                        "shell_bank_repairs_sma_gap": bank["e2_sma_gap_abs_error"] < base["e2_sma_gap_abs_error"],
                        "shell_bank_beats_dimension_matched_krylov": bank["energy"] < matched_generic["energy"],
                        "shell_bank_adds_after_dimension_matched_krylov": matched_joint["energy"] < matched_generic["energy"] - 1.0e-12,
                        "e2_shell_bank_beats_same_work_a1_energy": bank["energy"] < same_work["energy"] - 1.0e-12,
                        "e2_shell_bank_beats_same_work_a1_m0": bank["e2_m0_abs_error"] < same_work["e2_m0_abs_error"],
                        "e2_shell_bank_beats_same_work_a1_sma_gap": bank["e2_sma_gap_abs_error"] < same_work["e2_sma_gap_abs_error"],
                        "e2_adds_after_same_work_a1": channel_joint["energy"] < same_work["energy"] - 1.0e-12,
                        "a1_adds_after_e2": channel_joint["energy"] < bank["energy"] - 1.0e-12,
                    },
                }
            )

    by_projection = {}
    for projection in spec["projections"]:
        selected = [row for row in rows if row["projection"] == projection]
        by_projection[str(projection)] = {
            "trial_count": len(selected),
            "carrier_energy_win_count": sum(row["gates"]["carrier_lowers_energy"] for row in selected),
            "carrier_m0_repair_count": sum(row["gates"]["carrier_repairs_equal_time_m0"] for row in selected),
            "carrier_joint_win_count": sum(row["gates"]["carrier_lowers_energy"] and row["gates"]["carrier_repairs_equal_time_m0"] for row in selected),
            "carrier_beats_generic_count": sum(row["gates"]["carrier_beats_generic_energy_control"] for row in selected),
            "carrier_adds_after_generic_count": sum(row["gates"]["carrier_adds_after_generic_control"] for row in selected),
            "shell_bank_joint_win_count": sum(row["gates"]["shell_bank_lowers_energy"] and row["gates"]["shell_bank_repairs_equal_time_m0"] for row in selected),
            "shell_bank_energy_m0_sma_joint_win_count": sum(
                row["gates"]["shell_bank_lowers_energy"]
                and row["gates"]["shell_bank_repairs_equal_time_m0"]
                and row["gates"]["shell_bank_repairs_sma_gap"]
                for row in selected
            ),
            "shell_bank_beats_matched_krylov_count": sum(row["gates"]["shell_bank_beats_dimension_matched_krylov"] for row in selected),
            "shell_bank_adds_after_matched_krylov_count": sum(row["gates"]["shell_bank_adds_after_dimension_matched_krylov"] for row in selected),
            "e2_beats_same_work_a1_energy_count": sum(row["gates"]["e2_shell_bank_beats_same_work_a1_energy"] for row in selected),
            "e2_beats_same_work_a1_all_physics_count": sum(
                row["gates"]["e2_shell_bank_beats_same_work_a1_energy"]
                and row["gates"]["e2_shell_bank_beats_same_work_a1_m0"]
                and row["gates"]["e2_shell_bank_beats_same_work_a1_sma_gap"]
                for row in selected
            ),
            "e2_adds_after_same_work_a1_count": sum(row["gates"]["e2_adds_after_same_work_a1"] for row in selected),
            "a1_adds_after_e2_count": sum(row["gates"]["a1_adds_after_e2"] for row in selected),
            "median_carrier_energy_gain": float(
                np.median([
                    row["variants"]["baseline"]["energy"] - row["variants"]["e2_pair"]["energy"]
                    for row in selected
                ])
            ),
            "median_carrier_m0_abs_error_reduction": float(
                np.median([
                    row["variants"]["baseline"]["e2_m0_abs_error"] - row["variants"]["e2_pair"]["e2_m0_abs_error"]
                    for row in selected
                ])
            ),
            "median_shell_bank_energy_gain": float(
                np.median([
                    row["variants"]["baseline"]["energy"] - row["variants"]["e2_shell_bank"]["energy"]
                    for row in selected
                ])
            ),
            "median_shell_bank_m0_abs_error_reduction": float(
                np.median([
                    row["variants"]["baseline"]["e2_m0_abs_error"] - row["variants"]["e2_shell_bank"]["e2_m0_abs_error"]
                    for row in selected
                ])
            ),
            "median_e2_energy_advantage_over_same_work_a1": float(
                np.median([
                    row["variants"]["a1_shell_bank_same_work"]["energy"]
                    - row["variants"]["e2_shell_bank"]["energy"]
                    for row in selected
                ])
            ),
        }

    dimensions = {
        "n12_gamma_a1": exact_gamma_a1_spin_multiplicity_dimension(
            ((6, 4), (0, 2)), target_q=0
        ),
        "n16_gamma_a1": exact_gamma_a1_spin_multiplicity_dimension(
            supercell, target_q=0
        ),
        "n36_gamma_a1": exact_gamma_a1_spin_multiplicity_dimension(
            ((6, 0), (0, 6)), target_q=0
        ),
    }
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "state_bundle": {
            "path": str(state_path.relative_to(ROOT)),
            "sha256": file_sha256(state_path),
            "reused_without_training": True,
        },
        "exact_reference": {
            "energy": exact_energy,
            "e2_m0": exact_moments.m0,
            "e2_m1": exact_moments.m1,
            "e2_sma_gap": exact_moments.sma_gap,
        },
        "gamma_a1_dimensions": dimensions,
        "n12_exact_symmetry_certificate": _n12_symmetry_certificate(),
        "trials": rows,
        "summary": by_projection,
        "interpretation_boundary": {
            "e2_m0_is_equal_time_weight_not_static_susceptibility": True,
            "n16_is_exact_support_but_not_thermodynamic": True,
            "n36_dimension_is_structural_not_an_energy_benchmark": True,
            "no_new_ed_or_training_performed": True,
        },
        "runtime_identity": numerical_runtime_identity(),
    }
    result["result_sha256"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", result)
    return result


__all__ = ["TASK", "run_spin_e2_resonance_audit"]
