"""Exact-support carrier-versus-operator factorial test on triangular 4x4.

This workflow changes the singlet pair kernel before the projected determinant
is formed.  It compares a three-coordinate physical deformation bank with
Hamiltonian-blind random-pair controls, both with and without the identical
two-Lanczos dressing.  No diagonalization is performed; registered exact
energies are used only after optimization as error anchors.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.adaptive_projected_pair_carrier import (
    AdaptiveProjectedPairCarrier,
    batched_singlet_agp_amplitudes,
    generic_adaptive_pair_control,
    triangular_physical_adaptive_pair_carrier,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    fixed_sz_sparse_hamiltonian,
    spin_raising_relative_norm,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    _operator_vit_trials,
    triangular_pi_flux_operator_components,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_adaptive_projected_pair_carrier"


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise FloatingPointError("candidate carrier has zero or nonfinite norm")
    return values / norm


def _energy_variance(state: np.ndarray, hamiltonian) -> tuple[float, float]:
    vector = _normalized(state)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def _ritz_state(
    seed: np.ndarray, hamiltonian, degree: int
) -> tuple[np.ndarray, float, float, int]:
    if degree not in (0, 1, 2):
        raise ValueError("registered local Ritz degree must be 0, 1, or 2")
    columns = [_normalized(seed)]
    for _ in range(degree):
        columns.append(np.asarray(hamiltonian @ columns[-1]))
    matrix = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(1.0e-14, float(singular[0]) * 1.0e-12)
    rank = int(np.count_nonzero(singular > tolerance))
    basis = left[:, :rank]
    projected = basis.conj().T @ (hamiltonian @ basis)
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    state = _normalized(basis @ vectors[:, 0])
    energy, variance = _energy_variance(state, hamiltonian)
    return state, energy, variance, rank


def _state_diagnostics(
    state: np.ndarray,
    *,
    reference: np.ndarray,
    family: AdaptiveProjectedPairCarrier,
    basis: Sequence[int],
) -> dict[str, float]:
    vector = _normalized(state)
    fixed = _normalized(reference)
    overlap = np.vdot(fixed, vector)
    aligned = vector * np.exp(-1.0j * np.angle(overlap))
    important = np.abs(vector) > 1.0e-13 * float(np.max(np.abs(vector)))
    sign_changed = important & (np.real(aligned) * np.real(fixed) < 0.0)
    probability = np.abs(vector) ** 2
    leakage = max(
        float(np.linalg.norm(vector[index] - vector))
        for index in family.permutation_indices
    )
    return {
        "squared_overlap_with_fixed_carrier": float(abs(overlap) ** 2),
        "optimized_probability_on_sign_changed_support": float(
            np.sum(probability[sign_changed])
        ),
        "gamma_a1_relative_residual": leakage,
        "splus_relative_norm": float(
            spin_raising_relative_norm(vector[:, None].real, basis, 16)[0]
        ),
    }


def _optimize(
    family: AdaptiveProjectedPairCarrier,
    hamiltonian,
    basis: Sequence[int],
    *,
    degree: int,
    starts: Sequence[np.ndarray],
    bound: float,
    maxiter: int,
) -> dict[str, object]:
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            return _ritz_state(family.amplitudes(parameters), hamiltonian, degree)[1]
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1.0e6

    started = time.perf_counter()
    results = [
        minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=tuple((-bound, bound) for _ in range(family.parameter_count)),
            options={"maxiter": maxiter, "ftol": 1.0e-13, "gtol": 1.0e-8},
        )
        for start in starts
    ]
    best = min(results, key=lambda row: float(row.fun))
    fixed = family.amplitudes(np.zeros(family.parameter_count))
    state, energy, variance, rank = _ritz_state(
        family.amplitudes(best.x), hamiltonian, degree
    )
    return {
        "degree": degree,
        "parameters": np.asarray(best.x).tolist(),
        "parameter_labels": list(family.deformation_labels),
        "energy_total": energy,
        "energy_per_site": energy / 16.0,
        "variance_total": variance,
        "variance_per_site": variance / 16.0,
        "ritz_rank": rank,
        "objective_evaluations": evaluations,
        "estimated_full_h_matvecs": evaluations * (2 * degree + 2),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "wall_seconds": time.perf_counter() - started,
        **_state_diagnostics(
            state, reference=fixed, family=family, basis=basis
        ),
    }


def _optimize_shared_carrier(
    family: AdaptiveProjectedPairCarrier,
    problems: Sequence[tuple[float, object, Sequence[int], float]],
    *,
    starts: Sequence[np.ndarray],
    bound: float,
    maxiter: int,
) -> dict[str, object]:
    """Fit one carrier across couplings with a separate two-Lanczos Ritz solve."""

    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            carrier = family.amplitudes(parameters)
            return float(
                np.mean(
                    [
                        _ritz_state(carrier, hamiltonian, 2)[1] / 16.0
                        for _coupling, hamiltonian, _basis, _exact in problems
                    ]
                )
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            return 1.0e6

    started = time.perf_counter()
    results = [
        minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=tuple((-bound, bound) for _ in range(family.parameter_count)),
            options={"maxiter": maxiter, "ftol": 1.0e-13, "gtol": 1.0e-8},
        )
        for start in starts
    ]
    best = min(results, key=lambda row: float(row.fun))
    carrier = family.amplitudes(best.x)
    rows = []
    for coupling, hamiltonian, basis, exact in problems:
        state, energy, variance, rank = _ritz_state(carrier, hamiltonian, 2)
        rows.append(
            {
                "j2_over_j1": coupling,
                "energy_per_site": energy / 16.0,
                "variance_per_site": variance / 16.0,
                "gap_above_exact_per_site": (energy - exact) / 16.0,
                "ritz_rank": rank,
                **_state_diagnostics(
                    state,
                    reference=family.amplitudes(
                        np.zeros(family.parameter_count)
                    ),
                    family=family,
                    basis=basis,
                ),
            }
        )
    return {
        "parameters": np.asarray(best.x).tolist(),
        "parameter_labels": list(family.deformation_labels),
        "rows": rows,
        "mean_energy_per_site": float(
            np.mean([row["energy_per_site"] for row in rows])
        ),
        "mean_gap_above_exact_per_site": float(
            np.mean([row["gap_above_exact_per_site"] for row in rows])
        ),
        "maximum_gap_above_exact_per_site": float(
            np.max([row["gap_above_exact_per_site"] for row in rows])
        ),
        "objective_evaluations": evaluations,
        "estimated_full_h_matvecs": evaluations * len(problems) * 6,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "wall_seconds": time.perf_counter() - started,
    }


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid adaptive-pair carrier config identity")
    if not isinstance(spec.get("couplings"), list) or not spec["couplings"]:
        raise ValueError("couplings must be a nonempty list")
    if len(spec.get("generic_seeds", [])) < 2:
        raise ValueError("at least two Hamiltonian-blind generic controls are required")


def _literature_comparison(path: Path, coupling: float) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("phase_scan", {}).get("rows", [])
    matched = [row for row in rows if abs(float(row["j2_over_j1"]) - coupling) < 1.0e-12]
    if len(matched) != 1:
        return {}
    row = matched[0]
    return {
        "optimized_jastrow_two_lanczos_energy_per_site": float(
            row["jastrow_two_lanczos_energy_per_site"]
        ),
        "existing_operator_vit_energy_per_site": float(
            row["operator_vit_energy_per_site"]
        ),
        "existing_operator_vit_variance_per_site": float(
            row["operator_variance_per_site"]
        ),
    }


def run_spin_adaptive_projected_pair_carrier(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    case = SMALL_CASES["square4"]
    controls = tuple(int(value) for value in spec["generic_seeds"])
    optimizer = spec["optimizer"]
    start_scale = float(optimizer["start_scale"])
    starts = [np.zeros(3)]
    starts.extend(
        (
            np.asarray([0.05, -0.05, 0.05]),
            np.asarray([-0.05, 0.05, -0.05]),
            np.asarray([-0.39, -0.25, 0.40]),
        )
    )
    starts.extend(
        np.random.default_rng(int(seed)).normal(0.0, start_scale, size=3)
        for seed in spec["start_seeds"]
    )
    comparison_path = ROOT / str(spec["comparison_artifact"])
    rows = []
    shared_problems = []
    shared_physical = None
    for cell in spec["couplings"]:
        coupling = float(cell["j2_over_j1"])
        exact = float(cell["exact_energy_anchor"])
        hamiltonian, basis = fixed_sz_sparse_hamiltonian(case, coupling)
        configurations = np.asarray(
            [
                configuration_from_repository_integer(value, case.n_sites)
                for value in basis
            ],
            dtype=np.int8,
        )
        physical = triangular_physical_adaptive_pair_carrier(
            configurations, case
        )
        if shared_physical is None:
            shared_physical = physical
        shared_problems.append((coupling, hamiltonian, basis, exact))
        fixed = physical.amplitudes(np.zeros(physical.parameter_count))
        fixed_rows = {}
        for degree in (0, 2):
            state, energy, variance, rank = _ritz_state(
                fixed, hamiltonian, degree
            )
            fixed_rows[str(degree)] = {
                "degree": degree,
                "energy_total": energy,
                "energy_per_site": energy / 16.0,
                "variance_total": variance,
                "variance_per_site": variance / 16.0,
                "ritz_rank": rank,
                **_state_diagnostics(
                    state, reference=fixed, family=physical, basis=basis
                ),
            }
        physical_rows = {
            str(degree): _optimize(
                physical,
                hamiltonian,
                basis,
                degree=degree,
                starts=starts,
                bound=float(optimizer["bound"]),
                maxiter=int(optimizer["maxiter"]),
            )
            for degree in (0, 2)
        }
        generic_rows = []
        for seed in controls:
            generic = generic_adaptive_pair_control(physical, seed=seed)
            generic_rows.append(
                {
                    "seed": seed,
                    "degree_2": _optimize(
                        generic,
                        hamiltonian,
                        basis,
                        degree=2,
                        starts=starts,
                        bound=float(optimizer["bound"]),
                        maxiter=int(optimizer["maxiter"]),
                    ),
                }
            )
        generic_energies = np.asarray(
            [row["degree_2"]["energy_per_site"] for row in generic_rows]
        )
        comparison = _literature_comparison(comparison_path, coupling)
        row = {
            "j2_over_j1": coupling,
            "exact_energy_anchor_total": exact,
            "exact_energy_anchor_per_site": exact / 16.0,
            "fixed_carrier": fixed_rows,
            "physical_adaptive_carrier": physical_rows,
            "generic_adaptive_controls": generic_rows,
            "generic_degree_2_energy_per_site_median": float(
                np.median(generic_energies)
            ),
            "generic_degree_2_energy_per_site_range": [
                float(np.min(generic_energies)),
                float(np.max(generic_energies)),
            ],
            "comparison_artifact": comparison,
        }
        adaptive = physical_rows["2"]["energy_per_site"]
        fixed_energy = fixed_rows["2"]["energy_per_site"]
        row["comparisons"] = {
            "physical_adaptive_minus_fixed_two_lanczos_per_site": (
                adaptive - fixed_energy
            ),
            "physical_adaptive_gap_above_exact_per_site": adaptive - exact / 16.0,
            "physical_adaptive_minus_generic_median_per_site": (
                adaptive - float(np.median(generic_energies))
            ),
            **(
                {
                    "physical_adaptive_minus_jastrow_two_lanczos_per_site": (
                        adaptive
                        - comparison[
                            "optimized_jastrow_two_lanczos_energy_per_site"
                        ]
                    ),
                    "physical_adaptive_minus_existing_operator_vit_per_site": (
                        adaptive - comparison["existing_operator_vit_energy_per_site"]
                    ),
                }
                if comparison
                else {}
            ),
        }
        rows.append(row)
        print(
            f"J2={coupling:.3f} fixed2={fixed_energy:.10f} "
            f"adaptive2={adaptive:.10f} generic_med={np.median(generic_energies):.10f}",
            flush=True,
        )

    assert shared_physical is not None
    shared_physical_result = _optimize_shared_carrier(
        shared_physical,
        shared_problems,
        starts=starts,
        bound=float(optimizer["bound"]),
        maxiter=int(optimizer["maxiter"]),
    )
    shared_generic_results = []
    for seed in controls:
        generic = generic_adaptive_pair_control(shared_physical, seed=seed)
        shared_generic_results.append(
            {
                "seed": seed,
                **_optimize_shared_carrier(
                    generic,
                    shared_problems,
                    starts=starts,
                    bound=float(optimizer["bound"]),
                    maxiter=int(optimizer["maxiter"]),
                ),
            }
        )
    generic_shared_mean_gaps = np.asarray(
        [row["mean_gap_above_exact_per_site"] for row in shared_generic_results]
    )
    operator_spec = spec["operator_factorial"]
    shared_carrier = shared_physical.amplitudes(
        np.asarray(shared_physical_result["parameters"], dtype=np.float64)
    )
    operator_factorial_rows = []
    for coupling, _hamiltonian, _basis, exact in shared_problems:
        data = triangular_pi_flux_operator_components(
            coupling, carrier_override=shared_carrier
        )
        trial_seeds = (
            tuple(int(value) for value in operator_spec["seeds"])
            if abs(coupling - 0.125) < 1.0e-12
            else (int(operator_spec["seeds"][0]),)
        )
        trials = _operator_vit_trials(
            data,
            seeds=trial_seeds,
            steps=int(operator_spec["steps"]),
            learning_rate=float(operator_spec["learning_rate"]),
        )
        energies = np.asarray([row["energy_per_site"] for row in trials])
        variances = np.asarray([row["variance_per_site"] for row in trials])
        old = _literature_comparison(comparison_path, coupling)
        operator_factorial_rows.append(
            {
                "j2_over_j1": coupling,
                "trials": trials,
                "mean_energy_per_site": float(np.mean(energies)),
                "minimum_energy_per_site": float(np.min(energies)),
                "mean_variance_per_site": float(np.mean(variances)),
                "mean_gap_above_exact_per_site": float(
                    np.mean(energies) - exact / 16.0
                ),
                "adaptive_carrier_operator_minus_fixed_carrier_operator_per_site": (
                    float(np.mean(energies))
                    - old["existing_operator_vit_energy_per_site"]
                    if old
                    else None
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "architecture": {
            "carrier": "Gutzwiller-projected real singlet AGP",
            "spin_sector": "exact S=0",
            "lattice_sector": "exact Gamma-A1 Reynolds projection",
            "physical_coordinates": [
                "A1 nearest-neighbour pairing",
                "A1 next-nearest-neighbour pairing",
                "E2 nearest-neighbour pairing before nonlinear A1 projection",
            ],
            "operator_control": "identical degree-two Lanczos dressing",
            "generic_control": "Hamiltonian-blind random symmetric pair directions",
        },
        "rows": rows,
        "shared_carrier_transfer": {
            "contract": (
                "one common projected-pair carrier across every J2; only the "
                "three two-Lanczos Ritz coefficients are refit per Hamiltonian"
            ),
            "physical": shared_physical_result,
            "generic_controls": shared_generic_results,
            "generic_median_mean_gap_above_exact_per_site": float(
                np.median(generic_shared_mean_gaps)
            ),
            "physical_minus_generic_median_mean_gap_per_site": float(
                shared_physical_result["mean_gap_above_exact_per_site"]
                - np.median(generic_shared_mean_gaps)
            ),
        },
        "carrier_by_operator_factorial": {
            "contract": (
                "the shared adaptive carrier replaces the fixed pi-flux carrier; "
                "operator labels, descriptors, depth, steps, and learning rate "
                "are unchanged"
            ),
            "rows": operator_factorial_rows,
        },
        "gates": {
            "physical_adaptive_beats_fixed_two_lanczos_all_points": all(
                row["comparisons"][
                    "physical_adaptive_minus_fixed_two_lanczos_per_site"
                ]
                < -1.0e-10
                for row in rows
            ),
            "physical_labels_beat_generic_median_all_points": all(
                row["comparisons"][
                    "physical_adaptive_minus_generic_median_per_site"
                ]
                < 0.0
                for row in rows
            ),
            "all_states_exact_singlets_and_a1": all(
                row["physical_adaptive_carrier"]["2"]["splus_relative_norm"]
                < 1.0e-10
                and row["physical_adaptive_carrier"]["2"][
                    "gamma_a1_relative_residual"
                ]
                < 1.0e-10
                for row in rows
            ),
            "shared_physical_carrier_beats_shared_generic_median": (
                shared_physical_result["mean_gap_above_exact_per_site"]
                < float(np.median(generic_shared_mean_gaps))
            ),
            "adaptive_carrier_improves_existing_operator_vit_all_points": all(
                row[
                    "adaptive_carrier_operator_minus_fixed_carrier_operator_per_site"
                ]
                is not None
                and row[
                    "adaptive_carrier_operator_minus_fixed_carrier_operator_per_site"
                ]
                < 0.0
                for row in operator_factorial_rows
            ),
        },
        "interpretation_boundary": {
            "exact_support_architecture_test": True,
            "hamiltonian_diagonalization_performed": False,
            "generic_control_is_not_a_scalable_final_ansatz": True,
            "remote_scale_up_allowed_only_after_transfer_and_control_gate": True,
        },
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


def _pair_sector_functions(basis: np.ndarray, isometry: np.ndarray,
                           hamiltonian: np.ndarray):
    """Reuse the existing JAX AGP and a SAVED COMPLETE singlet/A1 sector.

    U^dagger raw(C) gives the coefficients of the physical Reynolds projection
    because raw(C) is already an exact singlet. This exact-small shortcut is
    NOT a new scalable projector, and must agree with explicit averaging.
    Only the off-diagonal real symmetric pair kernel is trained. Normalizing
    its packed vector removes the homogeneous pairing scale gauge.
    """
    import jax
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import (
        ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes,
    )
    bits, u, h = np.asarray(basis), np.asarray(isometry), np.asarray(hamiltonian)
    if (bits.ndim != 2 or bits.shape[1] % 2 or
            np.any((bits != 0) & (bits != 1)) or
            np.any(bits.sum(axis=1) != bits.shape[1]//2) or
            u.ndim != 2 or u.shape[0] != len(bits) or h.shape != (u.shape[1],)*2 or
            not np.isfinite(u).all() or not np.isfinite(h).all() or
            np.linalg.norm(u.conj().T@u-np.eye(u.shape[1])) > 1e-10 or
            np.linalg.norm(h-h.conj().T) > 1e-10):
        raise ValueError('half-filled bits, orthonormal saved sector and Hermitian H required')
    cfg = ProjectedSingletAGPConfig(bits.shape[1], (tuple(range(bits.shape[1])),))
    x, uj, hj = jnp.asarray(bits), jnp.asarray(u), jnp.asarray(h)

    def coefficients(p):
        packed = p/jnp.linalg.norm(p)
        raw = projected_singlet_agp_amplitudes({'pairing_real': packed}, x, cfg).amplitudes
        c = uj.conj().T@raw
        return c/jnp.linalg.norm(c)

    def energy(p):
        c = coefficients(p)
        return jnp.vdot(c, hj@c).real/bits.shape[1]

    return jax.jit(coefficients), jax.jit(jax.value_and_grad(energy))


def run_pair_quality(config_path: Path) -> dict:
    """One bounded nonlinear pair-family refinement, followed by exact readout.

    No reference wavefunction/readout enters the energy objective or endpoint
    selection. Cached N16 operators avoid new many-body ED. Every run is the
    SAME unrestricted projected real-singlet AGP family, not a distinct phase.
    All numerical outputs are immutable. Large-size sampling is not released.
    """
    from su2_nqs.benchmarks.spin_operator_information import (
        joint_spin_operator_split, three_bond_readout,
    )
    from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
    from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
    from su2_nqs.benchmarks.readout_quality import regional_readout_distance
    from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

    config_path = config_path.resolve()
    spec = json.loads(config_path.read_text())
    if (spec['j2_over_j1'] != .125 or spec['n_sites'] != 16 or
            spec['optimizer']['maxiter'] > 96 or spec['optimizer']['maxfun'] > 160 or
            len(spec['initialization_seeds']) != 2 or
            spec['optimizer']['wall_seconds'] > 600 or
            spec['optimizer']['objective'] != 'bare_projected_AGP_energy_only'):
        raise ValueError('this registered budget is only a two-initialization N16 quality probe')
    out = ROOT/spec['output']
    if out.exists():
        raise FileExistsError('pair quality output exists; preserve checkpoints, no overwrite')
    screen_path = ROOT/spec['candidate_archive']/'results.json'
    screen = json.loads(screen_path.read_text())
    bundle = ROOT/screen['state_bundle']['path']
    if file_sha256(bundle) != screen['state_bundle']['sha256']:
        raise ValueError('saved complete symmetry support changed')
    old_path = ROOT/spec['optimized_pair_archive']
    old = json.loads(old_path.read_text()); body = dict(old); seal = body.pop('result_sha256')
    if canonical_sha256(body) != seal:
        raise ValueError('old pair optimizer seal fails')
    old_row = next(r for r in old['rows'] if r['j2_over_j1'] == .125)
    with np.load(bundle) as z:
        bits, u, h = z['basis'], z['physical_a1_basis'], z['h_a1']
    if u.shape != (12870, 14):
        raise ValueError('this probe requires the saved COMPLETE triangular N16 S0/A1 sector')
    family = triangular_physical_adaptive_pair_carrier(bits, SMALL_CASES['square4'])
    start_matrix = family.pairing(old_row['physical_adaptive_carrier']['0']['parameters'])
    upper = np.triu_indices(16, 1)
    initial = start_matrix[upper]; initial = initial/np.linalg.norm(initial)
    coefficients, objective = _pair_sector_functions(bits, u, h)
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    out.mkdir(parents=True)
    for path in sources:
        dst = out/'source_snapshots'/path
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/path, dst)
    atomic_json(out/'registration.json', dict(config=spec, sources=sources,
        inputs={str(p.relative_to(ROOT)): file_sha256(p)
                for p in (config_path, screen_path, bundle, old_path)},
        registered_utc=datetime.now(timezone.utc).isoformat(),
        protocol='One real symmetric projected AGP family, full pair-kernel refinement. Two initializations are optimizer checks, never two phases. Energy-only fixed budget, then exact full-support readout. No reference or readout target in training.',
        full_pair_parameters=120, homogeneous_scale_removed=True,
        cached_complete_sector_dimension=14, no_new_many_body_ED=True,
        no_new_head=True, no_ViT_training=True, statistical_SE=0.,
        no_untouched_MC_claim=True))
    started = time.perf_counter()
    initial_e, initial_g = objective(initial)
    initial_e, initial_g = float(initial_e), np.asarray(initial_g)
    replay_error = abs(initial_e-old_row['physical_adaptive_carrier']['0']['energy_per_site'])
    if replay_error > 2e-11 or not np.isfinite(initial_g).all():
        raise ValueError('energy replay or polynomial determinant gradient failed')
    check_rng = np.random.default_rng(spec['gradient_check_seed'])
    fd_errors = []
    for _ in range(3):
        direction = check_rng.normal(size=120); direction /= np.linalg.norm(direction)
        eps = 2e-5
        fd = (float(objective(initial+eps*direction)[0])
              -float(objective(initial-eps*direction)[0]))/(2*eps)
        fd_errors.append(abs(fd-float(initial_g@direction)))
    if max(fd_errors) > 2e-7:
        raise ValueError('independent finite-difference gradient calibration failed')

    snapshots = {}; trajectories = []
    for run_index, seed in enumerate(spec['initialization_seeds']):
        p0 = initial.copy()
        if run_index:
            rng = np.random.default_rng(seed)
            p0 += spec['optimizer']['perturbation_scale']*rng.normal(size=120)/np.sqrt(120)
        calls = 0; iterations = 0; history = []; last = p0.copy()

        def freeze(p, iteration):
            c = np.asarray(coefficients(p)); e = float(np.vdot(c, h@c).real)/16
            variance = float(np.linalg.norm(h@c-16*e*c)**2)/16
            label = f'run{run_index}_step{iteration}'
            snapshots[label] = dict(parameters=p.copy(), coefficients=c)
            history.append(dict(iteration=iteration, energy_per_site=e,
                                variance_per_site=variance, label=label))
            np.savez_compressed(out/f'checkpoint_run{run_index}.npz',
                                pairing_real=p, coefficients=c, iteration=iteration)

        def fun(p):
            nonlocal calls
            calls += 1
            value, gradient = objective(p)
            value, gradient = float(value), np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError('nonfinite pair energy/gradient; no node flooring')
            return value, gradient

        def callback(p):
            nonlocal iterations, last
            iterations += 1; last = p.copy()
            if iterations in spec['optimizer']['milestones']:
                freeze(p, iterations)
            if time.perf_counter()-started > spec['optimizer']['wall_seconds']:
                raise TimeoutError('registered pair-quality wall budget reached')

        freeze(p0, 0)
        try:
            opt = minimize(fun, p0, jac=True, method='L-BFGS-B', callback=callback,
                options=dict(maxiter=spec['optimizer']['maxiter'],
                             maxfun=spec['optimizer']['maxfun'], gtol=2e-7, ftol=1e-13))
            last = np.asarray(opt.x)
            message, success = str(opt.message), bool(opt.success)
        except TimeoutError as exc:
            message, success = str(exc), False
        if not history or history[-1]['iteration'] != iterations:
            freeze(last, iterations)
        # Save an unambiguous final endpoint even after early optimizer convergence.
        final_label = f'run{run_index}_final'
        snapshots[final_label] = dict(parameters=last, coefficients=np.asarray(coefficients(last)))
        _, grad = fun(last)
        trajectories.append(dict(seed=seed, objective_calls=calls, iterations=iterations,
            optimizer_success=success, optimizer_message=message, history=history,
            final_label=final_label, final_gradient_norm=float(np.linalg.norm(grad)),
            scale_gauge_gradient_dot=float(grad@last),
            same_variational_family=True))
        atomic_json(out/'progress.json', dict(trajectories=trajectories,
                    seconds=time.perf_counter()-started, phase_claim=False))
        if time.perf_counter()-started > spec['optimizer']['wall_seconds']:
            break

    # Training is over. Reference amplitudes are only read for the frozen audit.
    with np.load(bundle) as z:
        vectors = {name: z[name+'_state'] for name in spec['saved_controls']}
        reference = z['saved_reference_state']
    checks = []
    for name, row in snapshots.items():
        vectors[name] = u@row['coefficients']
    for row in trajectories:
        p = snapshots[row['final_label']]['parameters']
        matrix = np.zeros((16,16)); matrix[upper] = p; matrix[(upper[1],upper[0])] = p
        raw = batched_singlet_agp_amplitudes(bits,matrix)
        explicit = _normalized(np.mean(raw[family.permutation_indices], axis=0))
        checks.append(float(np.linalg.norm(explicit-vectors[row['final_label']]
                           *np.exp(1j*np.angle(np.vdot(vectors[row['final_label']],explicit))))))
        for degree in (1,2):
            c, _, _, _ = _ritz_state(snapshots[row['final_label']]['coefficients'],h,degree)
            vectors[row['final_label']+f'_k{degree}'] = u@c
    if max(checks) > 2e-10:
        raise ValueError('cached projector differs from independent explicit Reynolds amplitudes')
    vectors['saved_reference'] = reference
    records = {}; arrays = dict(basis=bits)
    split = joint_spin_operator_split(6, (0,2,4), (3,3,0))
    for name, psi in vectors.items():
        psi = _normalized(psi); c = u.conj().T@psi
        # Some old controls need not belong to this sector. Fail instead of repair.
        if np.linalg.norm(psi-u@c) > 1e-10:
            raise ValueError('registered old control is not in the common exact S0/A1 support')
        e, variance = _energy_variance(c,h)
        rho = reduced_density_matrix(psi,bits,spec['six_site_region'])
        bond = three_bond_readout(rho,(0,2,4))
        if abs(bond['high_q3']-split.measure(rho)['high']) > 3e-12:
            raise ValueError('three-bond/projector identity failed on physical state')
        records[name] = dict(energy_per_site=e/16, variance_per_site=variance/16,
            readout=physical_spin_block_readout(rho), three_bond=bond,
            projector_q3=split.measure(rho), statistical_SE=0.)
        arrays[name+'_state'] = psi; arrays[name+'_rho'] = rho
    for name, row in snapshots.items():
        arrays[name+'_pairing_real'] = row['parameters']
    def distance(a,b):
        return dict(**regional_readout_distance(arrays[a+'_rho'],arrays[b+'_rho']),
            high_q3=records[a]['three_bond']['high_q3']-records[b]['three_bond']['high_q3'],
            X_connected=records[a]['three_bond']['orbits']['X']['connected']-records[b]['three_bond']['orbits']['X']['connected'],
            Y_connected=records[a]['three_bond']['orbits']['Y']['connected']-records[b]['three_bond']['orbits']['Y']['connected'])
    comparison = None
    if len(trajectories) == 2:
        a,b = [t['final_label'] for t in trajectories]
        comparison = dict(endpoint_distance=distance(a,b),
            own_last_checkpoint_change={t['final_label']:distance(t['final_label'],
                t['history'][-2 if t['history'][-1]['iteration']==t['iterations'] and len(t['history'])>1 else -1]['label'])
                for t in trajectories},
            interpretation='Two initializations of ONE family test quality/optimization, not two physical phases.')
    result = dict(config=spec, trajectories=trajectories, states=records,
        errors_to_reference={n:distance(n,'saved_reference') for n in vectors},
        same_family_comparison=comparison, initial_energy_replay_error=replay_error,
        gradient_fd_errors=fd_errors, maximum_explicit_projection_error=max(checks),
        seconds=time.perf_counter()-started, new_many_body_ED=0, new_MC_samples=0,
        optimized_pair_family_count=1, new_heads=0, phase_claim=False,
        scalable_projector_claim=False, size_evolution_released=False,
        warning='Exact N16 quality probe on a torus missing K. No phase identification, no N36 precision guarantee. Finite refinement differences are empirical rather than rigorous bias bounds.')
    np.savez_compressed(out/'states_readouts.npz', **arrays)
    atomic_json(out/'results.json',result)
    atomic_json(out/'completion.json',dict(status='complete',
        outputs={p.name:file_sha256(p) for p in out.iterdir() if p.suffix in ('.json','.npz')
                 and p.name != 'completion.json'}, new_many_body_ED=0, phase_claim=False))
    print(json.dumps(dict(seconds=result['seconds'], endpoints={t['final_label']:
        {k:records[t['final_label']][k] for k in ('energy_per_site','variance_per_site')}
        for t in trajectories}, projection=max(checks))),flush=True)
    return result


def _tiled_pair_amplitudes(parameters, configurations, config, *, tile_size: int = 8):
    """Exact Reynolds amplitudes with a bounded, differentiable scan.

    This only changes batching of the existing signed-determinant evaluator.
    Padding repeats a valid configuration and is discarded, never interpreted
    as an extra sample. No spatial images, determinants or nodes are removed.
    """
    import jax
    import jax.numpy as jnp
    from su2_nqs.models.projected_singlet_agp import projected_singlet_agp_amplitudes

    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    raw = jnp.asarray(configurations)
    if raw.ndim != 2 or not len(raw) or raw.shape[1] != config.n_sites:
        raise ValueError('nonempty physical configuration matrix required')
    # Value checks must precede JIT at the public workflow boundary.
    if not isinstance(raw, jax.core.Tracer):
        host = np.asarray(raw)
        if host.dtype.kind not in 'iu' or np.any((host != 0) & (host != 1)) or np.any(host.sum(1) != config.n_sites//2):
            raise ValueError('integer binary half-filled configurations required')
    bits = raw.astype(jnp.int8)
    padding = (-len(bits)) % tile_size
    padded = jnp.concatenate((bits, jnp.repeat(bits[-1:], padding, axis=0)))
    tiles = padded.reshape((-1, tile_size, config.n_sites))

    def evaluate(_, batch):
        values = projected_singlet_agp_amplitudes(
            {'pairing_real': parameters}, batch, config).amplitudes
        return None, values

    _, values = jax.lax.scan(evaluate, None, tiles)
    return values.reshape(-1)[:len(bits)]


def _pair_query_functions(config, plan, *, amplitude_scale: float, tile_size: int):
    """Node-safe full-H action and both-sided Rayleigh gradient on a query plan."""
    import jax
    import jax.numpy as jnp
    from su2_nqs.vmc.physical_h_queries import gather_physical_h_action

    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError('positive finite amplitude scale required')
    queries = jnp.asarray(plan.configurations)

    def action(parameters):
        amplitudes = _tiled_pair_amplitudes(parameters, queries, config,
            tile_size=tile_size) / amplitude_scale
        return gather_physical_h_action(plan, amplitudes)

    def objective(parameters, proposal_density):
        psi, hpsi = action(parameters)
        weights = 1 / proposal_density
        norm = jnp.sum(jnp.abs(psi)**2 * weights)
        return jnp.real(jnp.sum(jnp.conj(psi)*hpsi*weights)) / norm / config.n_sites

    return jax.jit(action), jax.jit(jax.value_and_grad(objective))


def run_pair_n36_cost(config_path: Path) -> dict[str, object]:
    """Bounded complete-projection/H/gradient preflight; random probes are not VMC.

    The full real symmetric off-diagonal kernel is the trainable family. The
    old three physical directions provide test deformations only; N16 fitted
    parameters are neither transferred nor reoptimized. No H eigensolve occurs.
    """
    import resource
    import jax
    import jax.numpy as jnp
    from su2_nqs.benchmarks.adaptive_projected_pair_carrier import triangular_physical_pairing_bank
    from su2_nqs.benchmarks.projected_singlet_agp import triangular_hnf_gamma_a1_group
    from su2_nqs.benchmarks.triangular_induced_parton import (
        build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax)
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from su2_nqs.frontier.provenance import file_sha256

    spec = json.loads(config_path.read_text())
    if spec.get('task') != 'spin_pair_n36_cost' or spec.get('n_sites') != 36:
        raise ValueError('only the registered triangular N36 cost task is supported')
    out = ROOT/spec['output']
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = [Path(__file__), ROOT/'su2_nqs/models/projected_singlet_agp.py',
        ROOT/'su2_nqs/benchmarks/adaptive_projected_pair_carrier.py',
        ROOT/'su2_nqs/benchmarks/triangular_induced_parton.py',
        ROOT/'su2_nqs/vmc/physical_h_queries.py']
    registration = dict(config=spec, config_sha256=file_sha256(config_path),
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in sources},
        runtime=numerical_runtime_identity(), new_many_body_ED=0, new_heads=0)
    atomic_json(out/'registration.json', registration)
    base, directions, labels = triangular_physical_pairing_bank(6)
    group = triangular_hnf_gamma_a1_group(((6,0),(0,6)))
    config = ProjectedSingletAGPConfig(36, group)
    upper = np.triu_indices(36, 1)
    p0 = base[upper]
    coefficients = np.asarray(spec['probe_deformation'])
    pairing = base + np.einsum('k,kij->ij', coefficients, np.asarray(directions))
    probe = pairing[upper]
    generator = np.random.default_rng(spec['probe_seed'])
    bits = np.zeros((max(spec['center_batches']), 36), np.int8)
    for row in bits:
        row[generator.choice(36, 18, replace=False)] = 1
    orbit = build_n36_pi_zero_flux_orbit()
    fast = jax.jit(lambda x: projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit))
    fast_values = np.asarray(fast(jnp.asarray(bits)))
    scale = float(np.sqrt(np.mean(np.abs(fast_values)**2)))
    amplitude = jax.jit(lambda p,x: _tiled_pair_amplitudes(p,x,config,
        tile_size=spec['tile_size'])/scale)

    def timed(fn, *args):
        beginning = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        return result, time.perf_counter()-beginning

    checks = {}; measurements = []
    values, first_time = timed(amplitude, jnp.asarray(probe), jnp.asarray(bits))
    _, warm_time = timed(amplitude, jnp.asarray(probe), jnp.asarray(bits))
    images = np.asarray(group)
    explicit = batched_singlet_agp_amplitudes(bits[:,images].reshape(-1,36), pairing)
    explicit = explicit.reshape(len(bits), len(group)).mean(1)/scale
    checks['independent_numpy_projection_relative_error'] = float(
        np.linalg.norm(np.asarray(values)-explicit)/np.linalg.norm(explicit))
    baseline = np.asarray(amplitude(jnp.asarray(p0),jnp.asarray(bits)))
    fast_scaled = fast_values/scale
    phase = np.vdot(fast_scaled, baseline)/np.vdot(fast_scaled,fast_scaled)
    checks['zero_deformation_vs_three_determinant_relative_error'] = float(
        np.linalg.norm(baseline-phase*fast_scaled)/np.linalg.norm(baseline))
    checks['three_determinant_global_factor'] = float(np.real(phase))
    transformed = bits[:, images[len(group)//3]]
    check_a1 = np.asarray(amplitude(jnp.asarray(probe), jnp.asarray(transformed)))
    checks['sampled_a1_relative_error'] = float(np.linalg.norm(check_a1-values)/np.linalg.norm(values))
    measurements.append(dict(kind='complete_projection', centers=len(bits),
        first_seconds=first_time, warm_seconds=warm_time,
        determinant_terms=len(bits)*len(group)))
    bonds = triangular_j1j2_bonds(6,6,j1=1.,j2=spec['j2_over_j1'],periodic=True)
    for count in spec['center_batches']:
        if time.perf_counter()-started > spec['wall_seconds']:
            break
        plan = compile_physical_h_queries(bits[:count],bonds)
        action, gradient = _pair_query_functions(config, plan,
            amplitude_scale=scale, tile_size=spec['tile_size'])
        density = jnp.asarray(np.abs(baseline[:count])**2+spec['positive_floor'])
        (psi,hpsi), initial = timed(action,jnp.asarray(probe))
        _, warm = timed(action,jnp.asarray(probe))
        (energy, grad), grad_initial = timed(gradient,jnp.asarray(probe),density)
        _, grad_warm = timed(gradient,jnp.asarray(probe),density)
        measurements.append(dict(kind='full_H_and_energy_gradient', centers=count,
            queries=plan.diagnostics, determinant_terms=len(plan.configurations)*len(group),
            first_H_seconds=initial, warm_H_seconds=warm,
            first_gradient_seconds=grad_initial, warm_gradient_seconds=grad_warm,
            gradient_parameter_count=len(probe), gradient_finite=bool(np.all(np.isfinite(grad))),
            warning='random configurations: objective values are NOT physical energy estimates'))
        if count == spec['center_batches'][0]:
            independent = batched_singlet_agp_amplitudes(
                plan.configurations[:,images].reshape(-1,36),pairing)
            independent = independent.reshape(len(plan.configurations),len(group)).mean(1)/scale
            from su2_nqs.vmc.physical_h_queries import gather_physical_h_action
            refpsi, refh = gather_physical_h_action(plan, jnp.asarray(independent))
            checks['independent_full_H_relative_error'] = float(
                np.linalg.norm(np.asarray(hpsi-refh))/np.linalg.norm(np.asarray(refh)))
            direction = generator.normal(size=len(probe)); direction /= np.linalg.norm(direction)
            eps=spec['finite_difference_step']
            def scalar(p):
                a,b=action(jnp.asarray(p));w=1/np.asarray(density)
                return float(np.sum(np.conj(a)*b*w).real/np.sum(np.abs(a)**2*w)/36)
            checks['energy_gradient_directional_FD_error'] = abs(
                (scalar(probe+eps*direction)-scalar(probe-eps*direction))/(2*eps)
                -float(np.asarray(grad)@direction))
        atomic_json(out/'progress.json',dict(measurements=measurements,
            checks=checks, seconds=time.perf_counter()-started))
    save_checkpoint(out/'carrier.pkl',{'pairing_real':probe},
        metadata=dict(model='triangular_6x6_PBC_J1J2',j2=spec['j2_over_j1'],
            amplitude_scale=scale,group_order=len(group),parameter_count=len(probe)))
    restored=load_checkpoint(out/'carrier.pkl')['params']['pairing_real']
    checks['checkpoint_maximum_scaled_amplitude_error']=float(np.max(np.abs(
        np.asarray(amplitude(jnp.asarray(restored),jnp.asarray(bits)))-values)))
    passed = (all(checks[key]<spec['relative_tolerance'] for key in (
        'independent_numpy_projection_relative_error',
        'zero_deformation_vs_three_determinant_relative_error',
        'sampled_a1_relative_error','independent_full_H_relative_error'))
        and checks['energy_gradient_directional_FD_error']<spec['gradient_tolerance']
        and checks['checkpoint_maximum_scaled_amplitude_error']==0
        and all(r.get('gradient_finite',True) for r in measurements))
    result=dict(config=spec, checks=checks, measurements=measurements,
        checks_pass=passed, seconds=time.perf_counter()-started,
        peak_process_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
        parameterization='full 630-entry real symmetric singlet pair kernel',
        test_direction_labels=labels, group_order=len(group),
        new_many_body_ED=0,new_heads=0,training_steps=0,phase_claim=False)
    np.savez_compressed(out/'probe.npz',configurations=bits,base_pairing_real=p0,
        probe_pairing_real=probe,amplitude_scale=scale,probe_amplitudes=np.asarray(values))
    atomic_json(out/'results.json',result)
    atomic_json(out/'completion.json',dict(status='complete',
        outputs={p.name:file_sha256(p) for p in out.iterdir()
            if p.name!='completion.json' and p.suffix in ('.json','.npz','.pkl')}))
    print(json.dumps(result),flush=True)
    if not passed:
        raise ValueError('preflight invariant failed; no optimization released')
    return result


def _pair_moment_functions(config, plan, *, amplitude_scale: float, tile_size: int):
    """Blockwise numerator/norm derivatives for one GLOBAL Rayleigh quotient.

    Averaging block energies would be wrong. Accumulate N,D,dN,dD, then use
    (dN-E*dD)/D. The full-H numerator differentiates BOTH coherent factors.
    Norm-only derivatives evaluate centers, not the large H-connected set.
    """
    import jax
    import jax.numpy as jnp
    from su2_nqs.vmc.physical_h_queries import gather_physical_h_action
    queries=jnp.asarray(plan.configurations);centers=jnp.asarray(plan.centers)

    def numerator(p,q):
        all_values=_tiled_pair_amplitudes(p,queries,config,tile_size=tile_size)/amplitude_scale
        a,b=gather_physical_h_action(plan,all_values)
        return jnp.sum(jnp.real(jnp.conj(a)*b)/q), (
            jnp.sum(jnp.abs(a)**2/q),jnp.sum(jnp.abs(b)**2/q),a,b)

    def norm(p,q):
        a=_tiled_pair_amplitudes(p,centers,config,tile_size=tile_size)/amplitude_scale
        return jnp.sum(jnp.abs(a)**2/q)

    return (jax.jit(jax.value_and_grad(numerator,has_aux=True)),
            jax.jit(jax.grad(norm)))


def _pair_weighted_moments(psi, hpsi, density, n_sites: int) -> np.ndarray:
    """Self-normalized direct importance E and residual moment; valid at nodes."""
    a,b,q=np.asarray(psi),np.asarray(hpsi),np.asarray(density)
    if a.shape!=b.shape or q.shape!=a.shape or np.any(q<=0) or not np.isfinite(q).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching amplitudes and positive proposal density required')
    d=np.sum(np.abs(a)**2/q)
    if not np.isfinite(d) or d<=0:
        raise FloatingPointError('unresolved physical norm')
    e=np.sum(np.conj(a)*b/q).real/d
    residual=np.sum(np.abs(b-e*a)**2/q)/d
    return np.array([e/n_sites,residual/n_sites])


def _pair_chain_comparison(arrays, density, chains: int, n_sites: int) -> dict:
    """Pooled ratios and matched delete-WHOLE-chain differences, not chain ratios."""
    labels=list(arrays);n=len(density);groups=np.arange(n)%chains
    central={label:_pair_weighted_moments(*arrays[label],density,n_sites) for label in labels}
    deletes={label:np.array([_pair_weighted_moments(
        arrays[label][0][groups!=c],arrays[label][1][groups!=c],density[groups!=c],n_sites)
        for c in range(chains)]) for label in labels}
    def errors(jack):
        return np.sqrt((chains-1)*np.mean((jack-jack.mean(0))**2,axis=0))
    states={}
    for label in labels:
        se=errors(deletes[label]);w=np.abs(arrays[label][0])**2/density
        states[label]=dict(energy_per_site=float(central[label][0]),
            variance_per_site=float(central[label][1]),energy_chain_SE=float(se[0]),
            variance_chain_SE=float(se[1]),weight_ESS_fraction=float(w.sum()**2/(w@w*n)),
            maximum_weight_fraction=float(w.max()/w.sum()))
    paired={}
    for before,after in zip(labels[:-1],labels[1:]):
        delta=central[after]-central[before];se=errors(deletes[after]-deletes[before])
        paired[after+'-minus-'+before]=dict(energy=float(delta[0]),energy_chain_SE=float(se[0]),
            variance=float(delta[1]),variance_chain_SE=float(se[1]))
    return dict(states=states,paired=paired,independent_chains=chains,records=n,
        ratio_bias_not_removed=True,equilibrium_certified=False)


def run_pair_n36_fit(config_path: Path) -> dict[str, object]:
    """One bounded FULL pair-family fit; freeze first, then open a fresh audit.

    The old three-coordinate remote canary is not repeated. No neural head,
    readout-target loss, ED, spatial shortcut or audit-based endpoint selection.
    This is a feasibility/quality pilot, not optimized Z2 phase evidence.
    """
    import jax
    import jax.numpy as jnp
    from su2_nqs.benchmarks.projected_singlet_agp import triangular_hnf_gamma_a1_group
    from su2_nqs.benchmarks.triangular_induced_parton import (
        build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax)
    from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
    from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.vmc.checkpoint import save_checkpoint
    from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from su2_nqs.vmc.regional_density import (regional_query_plan,
        conditional_regional_matrices,regional_density_summary)
    from su2_nqs.frontier.provenance import file_sha256

    spec=json.loads(config_path.read_text())
    if spec.get('task')!='spin_pair_n36_fit' or spec.get('n_sites')!=36:
        raise ValueError('only the registered N36 pair pilot is supported')
    if spec['training_seed']==spec['audit_seed']:
        raise ValueError('training/audit seeds must be disjoint')
    cost=ROOT/spec['cost_source'];completion=json.loads((cost/'completion.json').read_text())
    for name,digest in completion['outputs'].items():
        if file_sha256(cost/name)!=digest:
            raise ValueError('sealed cost preflight changed')
    preflight=json.loads((cost/'results.json').read_text())
    if not preflight['checks_pass']:
        raise ValueError('complete preflight must pass before optimization')
    out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    atomic_json(out/'registration.json',dict(config=spec,
        config_sha256=file_sha256(config_path),workflow_sha256=file_sha256(Path(__file__)),
        preflight_completion_sha256=file_sha256(cost/'completion.json'),
        runtime=numerical_runtime_identity(),protocol='fit energy only; freeze physical endpoint before independent audit',
        audit_selects_endpoint=False))
    with np.load(cost/'probe.npz') as data:
        p0=data['base_pairing_real'].copy();scale=float(data['amplitude_scale'])
        probe_bits=data['configurations'].copy();probe_p=data['probe_pairing_real'].copy()
        saved_probe=data['probe_amplitudes'].copy()
    cfg=ProjectedSingletAGPConfig(36,triangular_hnf_gamma_a1_group(((6,0),(0,6))))
    amplitude=jax.jit(lambda p,x:_tiled_pair_amplitudes(p,x,cfg,tile_size=spec['tile_size'])/scale)
    replay=float(np.max(abs(np.asarray(amplitude(jnp.asarray(probe_p),jnp.asarray(probe_bits)))-saved_probe)))
    if replay!=0:
        raise ValueError('cost-checkpoint numerical amplitude replay changed')
    orbit=build_n36_pi_zero_flux_orbit()
    fast=jax.jit(lambda x:-projected_pi_zero_flux_a1_amplitudes_jax(x,orbit=orbit)/scale)
    base=lambda x:np.asarray(fast(jnp.asarray(x)))
    floor=spec['positive_floor']
    # The full-support floor changes the proposal ONLY, never physical psi.
    proposal=lambda x:np.sqrt(np.abs(base(x))**2+floor)
    training=sample_fixed_wavefunction_exchange(proposal,n_sites=36,n_up=18,
        chain_count=spec['chains'],burn_sweeps=spec['burn_sweeps'],
        measurement_blocks=spec['training_blocks'],decorrelation_sweeps=spec['decorrelation_sweeps'],
        seed=spec['training_seed'])
    x=training.configurations;q=np.abs(base(x))**2+floor
    bonds=triangular_j1j2_bonds(6,6,j1=1.,j2=.125,periodic=True)
    blocks=[]
    for begin in range(0,len(x),spec['H_batch_size']):
        plan=compile_physical_h_queries(x[begin:begin+spec['H_batch_size']],bonds)
        functions=_pair_moment_functions(cfg,plan,amplitude_scale=scale,tile_size=spec['tile_size'])
        blocks.append((begin,plan,functions))
    history=[];best=dict(parameters=p0.copy(),energy=np.inf);count=0
    norm0=np.linalg.norm(p0)

    def evaluate(raw):
        nonlocal count,best
        if count>=spec['objective_call_cap'] or time.perf_counter()-started>spec['training_wall_seconds']:
            raise StopIteration('fixed training budget exhausted')
        count+=1
        p=raw*norm0/np.linalg.norm(raw)
        n=d=m2=0.;gn=np.zeros_like(p);gd=np.zeros_like(p);states=[]
        for begin,plan,(numerator,normgradient) in blocks:
            density=jnp.asarray(q[begin:begin+len(plan.centers)])
            ((num,(den,second,a,b)),gradient)=numerator(jnp.asarray(p),density)
            gradnorm=normgradient(jnp.asarray(p),density)
            n+=float(num);d+=float(den);m2+=float(second)
            gn+=np.asarray(gradient);gd+=np.asarray(gradnorm)
            states.append(np.asarray(a))
        e=n/d;g=(gn-e*gd)/d/36
        # Explicitly remove the homogeneous pairing-scale gauge.
        g=(g-p*(g@p)/(p@p))*norm0/np.linalg.norm(raw)
        row=dict(call=count,energy_per_site=e/36,variance_per_site=max(0.,m2/d-e*e)/36,
            gradient_norm=float(np.linalg.norm(g)),scale_gauge_dot=float(g@raw),
            seconds=time.perf_counter()-started)
        history.append(row)
        if row['energy_per_site']<best['energy']:
            best=dict(parameters=p.copy(),energy=row['energy_per_site'])
        save_checkpoint(out/'training.pkl',{'pairing_real':p},
            metadata=dict(objective_calls=count,history=history,
                best_parameters=best['parameters'],bounded_budget=True))
        atomic_json(out/'progress.json',dict(history=history,seconds=time.perf_counter()-started))
        print(json.dumps(row),flush=True)
        return e/36,g

    try:
        minimize(evaluate,p0,method='L-BFGS-B',jac=True,
            bounds=[(float(p-v),float(p+v)) for p,v in zip(p0,np.full_like(p0,spec['entry_bound']))],
            options=dict(maxiter=spec['iteration_cap'],maxfun=spec['objective_call_cap'],
                maxls=8,ftol=1e-10,gtol=1e-7))
    except StopIteration:
        pass
    if not history:
        raise RuntimeError('no training objective completed')
    candidate=best['parameters']
    baseline=np.asarray(amplitude(jnp.asarray(p0),jnp.asarray(x)))
    def distance(p):
        a=np.asarray(amplitude(jnp.asarray(p),jnp.asarray(x)))
        overlap=np.sum(np.conj(a)*baseline/q)
        fidelity=abs(overlap)**2/(np.sum(abs(a)**2/q)*np.sum(abs(baseline)**2/q))
        return float(np.sqrt(max(0.,1-min(1.,fidelity))))
    fraction=1.
    for _ in range(20):
        endpoint=p0+fraction*(candidate-p0)
        if distance(endpoint)<=spec['maximum_training_sine_FS']:
            break
        fraction*=.5
    else:
        raise RuntimeError('bounded physical endpoint could not be calibrated')
    params=dict(carrier=p0,half_step=.5*(p0+endpoint),endpoint=endpoint)
    endpoint_distance=distance(endpoint)
    # Frozen checkpoint comes BEFORE generating even one audit sample.
    for label,p in params.items():
        save_checkpoint(out/(label+'.pkl'),{'pairing_real':p},metadata=dict(
            j2=.125,n_sites=36,group_order=432,amplitude_scale=scale,
            selected_using='training energy only, final FS cap uses training configurations',audit_opened=False))
    frozen={label:file_sha256(out/(label+'.pkl')) for label in params}
    atomic_json(out/'frozen_endpoint.json',dict(checkpoints=frozen,
        training_FS=endpoint_distance,accepted_fraction=fraction,objective_calls=count,
        audit_seed=spec['audit_seed'],audit_opened=False))
    frozen_time=time.perf_counter()-started
    def endpoint_amplitudes(states):
        return np.asarray(amplitude(jnp.asarray(endpoint),jnp.asarray(states)))
    # Include both frozen states AND a strictly positive floor, including nodes.
    def audit_density(states):
        return np.abs(base(states))**2+np.abs(endpoint_amplitudes(states))**2+floor
    audit=sample_fixed_wavefunction_exchange(lambda a:np.sqrt(audit_density(a)),
        n_sites=36,n_up=18,chain_count=spec['chains'],burn_sweeps=spec['burn_sweeps'],
        measurement_blocks=spec['audit_blocks'],decorrelation_sweeps=spec['decorrelation_sweeps'],
        seed=spec['audit_seed'])
    ax=audit.configurations;aq=audit_density(ax);arrays={label:([],[]) for label in params}
    full_H_centers=0;determinants=0
    for begin in range(0,len(ax),spec['H_batch_size']):
        plan=compile_physical_h_queries(ax[begin:begin+spec['H_batch_size']],bonds)
        action,_=_pair_query_functions(cfg,plan,amplitude_scale=scale,tile_size=spec['tile_size'])
        for label,p in params.items():
            a,b=action(jnp.asarray(p));arrays[label][0].append(np.asarray(a));arrays[label][1].append(np.asarray(b))
            full_H_centers+=len(plan.centers);determinants+=len(plan.configurations)*432
    arrays={label:tuple(np.concatenate(a) for a in items) for label,items in arrays.items()}
    moments=_pair_chain_comparison(arrays,aq,spec['chains'],36)
    saved=dict(training_configurations=x,training_density=q,audit_configurations=ax,audit_density=aq)
    for label,p in params.items():
        saved[label+'_pairing_real']=p;saved[label+'_psi']=arrays[label][0];saved[label+'_Hpsi']=arrays[label][1]
    regions={};groups=np.arange(len(ax))%spec['chains']
    for name,region in spec['regions'].items():
        if time.perf_counter()-started>spec['total_wall_seconds']:
            regions[name]=dict(status='not_evaluated_fixed_wall_budget');continue
        plan=regional_query_plan(ax,region['sites']);rq=audit_density(plan.configurations)
        rows={};deletes={}
        for label,p in params.items():
            amps=np.asarray(amplitude(jnp.asarray(p),jnp.asarray(plan.configurations)))
            summary=regional_density_summary(conditional_regional_matrices(plan,amps,rq),groups)
            saved[name+'_'+label+'_rho']=summary['rho']
            saved[name+'_'+label+'_jackknife_rhos']=summary['jackknife_rhos']
            read=physical_spin_block_readout(summary['rho'])
            def values(rho):
                v=three_bond_readout(rho,region['left'])
                return np.array([v['orbits']['Y'][k] for k in ('raw','connected')]+[v['high_q3']])
            center=values(summary['rho']);jack=np.array([values(r) for r in summary['jackknife_rhos']])
            se=np.sqrt((spec['chains']-1)*np.mean((jack-jack.mean(0))**2,axis=0))
            rows[label]=dict(readout=read,observables=dict(zip(('Y_raw','Y_connected','high_q3'),
                [dict(value=float(v),chain_SE=float(s)) for v,s in zip(center,se)])),
                weight_ESS_fraction=summary['weight_ess_fraction'],strict_spectrum_claim=False)
            deletes[label]=jack
        paired={}
        for before,after in (('carrier','half_step'),('half_step','endpoint'),('carrier','endpoint')):
            delta=np.array([rows[after]['observables'][k]['value']-rows[before]['observables'][k]['value']
                for k in ('Y_raw','Y_connected','high_q3')])
            jack=deletes[after]-deletes[before]
            se=np.sqrt((spec['chains']-1)*np.mean((jack-jack.mean(0))**2,axis=0))
            paired[after+'-minus-'+before]=dict(zip(('Y_raw','Y_connected','high_q3'),
                [dict(value=float(v),chain_SE=float(s)) for v,s in zip(delta,se)]))
        regions[name]=dict(sites=region['sites'],left=region['left'],states=rows,paired=paired,
            warning='Within-family quality path, not two competing phases; raw RDMs not twirled')
    for label,digest in frozen.items():
        if file_sha256(out/(label+'.pkl'))!=digest:
            raise RuntimeError('audit changed a frozen endpoint')
    np.savez_compressed(out/'samples_states_readouts.npz',**saved)
    delta=np.array([moments['states']['endpoint'][k]-moments['states']['carrier'][k]
        for k in ('energy_per_site','variance_per_site')])
    result=dict(config=spec,history=history,moments=moments,regions=regions,
        cost_checkpoint_replay_maximum_error=replay,objective_calls=count,
        parameter_count=630,optimized_candidate_families=1,new_heads=0,new_many_body_ED=0,
        frozen_after_seconds=frozen_time,seconds=time.perf_counter()-started,
        training_H_centers=count*len(x),audit_H_centers=full_H_centers,
        audit_projected_determinant_terms=determinants,
        training_sine_FS=endpoint_distance,accepted_fraction=fraction,
        audit_selection=False,energy_and_variance_central_values_improve=bool(np.all(delta<0)),
        phase_claim=False,equilibrium_certified=False,
        warnings=['Fixed pilot budget is not convergence proof.',
            'Whole-chain SE does not bound equilibration bias or unobserved tails.',
            'Pair kernel is a variational singlet AGP, not proof of a distinct Z2 liquid.',
            'Archived K2 energy uses different chains: contextual reference, not paired ranking.'])
    atomic_json(out/'results.json',result)
    atomic_json(out/'completion.json',dict(status='complete',
        outputs={p.name:file_sha256(p) for p in out.iterdir()
            if p.name!='completion.json' and p.suffix in ('.json','.pkl','.npz')}))
    print(json.dumps(dict(seconds=result['seconds'],moments=moments,accepted_fraction=fraction)),flush=True)
    return result


def _pair_dynamic_functions(config, *, amplitude_scale: float, tile_size: int):
    """Reusable full-H kernel: query values/maps are data, not JIT closures.

    Query padding repeats a legal state; no padded state enters a moment.
    The real kernel's Jacobian is used only for the importance FS metric.
    Full-H numerator derivatives still include BOTH coherent factors.
    """
    import jax
    import jax.numpy as jnp
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError('positive finite amplitude scale required')

    def amplitude(p, x):
        return _tiled_pair_amplitudes(p, x, config, tile_size=tile_size) / amplitude_scale

    def action(p, queries, centers, indices, elements):
        values = amplitude(p, queries)
        return values[centers], jnp.sum(values[indices] * elements, axis=1)

    def numerator(p, queries, centers, indices, elements, q):
        a, b = action(p, queries, centers, indices, elements)
        return jnp.sum(jnp.real(jnp.conj(a) * b) / q), (
            jnp.sum(abs(a)**2 / q), jnp.sum(abs(b)**2 / q), a, b)

    def norm(p, x, q):
        return jnp.sum(abs(amplitude(p, x))**2 / q)

    return dict(amplitude=jax.jit(amplitude), action=jax.jit(action),
        numerator=jax.jit(jax.value_and_grad(numerator, has_aux=True)),
        norm_gradient=jax.jit(jax.grad(norm)),
        center_jacobian=jax.jit(jax.jacrev(lambda p, x: amplitude(p, x).real)))


def _pair_padded_query_arguments(plan, *, bucket: int = 128):
    """Pack an exact coalesced plan into a reusable bucket of physical queries."""
    import jax.numpy as jnp
    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 1:
        raise ValueError('query bucket must be a positive integer')
    count = len(plan.configurations)
    capacity = ((count + bucket - 1) // bucket) * bucket
    queries = np.concatenate((plan.configurations,
        np.repeat(plan.configurations[-1:], capacity-count, axis=0)))
    # A fixed maximum width avoids compilation on every connectivity count.
    width = plan.diagnostics['bond_count'] + 1
    indices = np.zeros((len(plan.centers), width), np.int32)
    elements = np.zeros(indices.shape)
    actual_width = plan.connected_indices.shape[1]
    indices[:, :actual_width] = plan.connected_indices
    elements[:, :actual_width] = np.where(plan.valid, plan.elements, 0.)
    return tuple(jnp.asarray(a) for a in (queries, plan.center_indices, indices, elements))


def _pair_sine_fs(old, new, density) -> float:
    """Node-safe normalized physical overlap on a shared positive proposal."""
    a, b, q = np.asarray(old), np.asarray(new), np.asarray(density)
    if a.shape != b.shape or a.shape != q.shape or np.any(q <= 0):
        raise ValueError('matching amplitudes and positive density required')
    norm = np.sum(abs(a)**2/q) * np.sum(abs(b)**2/q)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError('unresolved overlap norm')
    fidelity = abs(np.sum(a.conj()*b/q))**2/norm
    return float(np.sqrt(max(0., 1.-min(1., fidelity))))


def _pair_generic_perturbation(parent: np.ndarray, *, relative_norm: float,
        seed: int) -> np.ndarray:
    """Perturb primitive real pairing entries, not the physical symmetry sector.

    The subsequent singlet/A1 decoder is unchanged. A symmetry-stabilized
    primitive pairing can have a low-rank first-order physical tangent despite
    having many parameters. Remove only its homogeneous scale gauge here;
    calibrate the finite physical distance separately, without energy selection.
    """
    p = np.asarray(parent, float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0:
        raise ValueError('nonzero finite packed real pairing required')
    if not np.isfinite(relative_norm) or relative_norm <= 0:
        raise ValueError('positive finite relative perturbation norm required')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('nonnegative integer initialization seed required')
    direction = np.random.default_rng(seed).normal(size=p.shape)
    direction -= p * (direction @ p) / (p @ p)
    direction *= relative_norm * np.linalg.norm(p) / np.linalg.norm(direction)
    return direction


def _pair_registered_coupling(spec: dict[str, object]) -> float:
    """Validate the finite nonnegative physical coupling, including ``J2=0``.

    This changes the Hamiltonian data only, not the exact singlet/A1 carrier.
    Historical configurations at ``J2/J1=.125`` retain their meaning.
    """
    value = spec.get('j2_over_j1')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('j2_over_j1 must be a finite nonnegative number')
    coupling = float(value)
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError('j2_over_j1 must be a finite nonnegative number')
    return coupling


def run_pair_n36_continue(config_path: Path, *, resume: bool = False) -> dict[str, object]:
    """Refresh samples and optimize ONE complete pairing family with SR.

    The previous endpoint, physical kernel and all432 A1 images are retained.
    Samples/geometry are refreshed every step; an independent cheap physical
    distance set limits updates, not the final audit. Endpoint selection uses
    the fixed last completed step, never the untouched audit. Checkpoints at
    completed-step boundaries contain the exact seed schedule/optimizer state.
    No additional architecture, readout-target loss, N16 optimization or ED.
    """
    import jax
    import jax.numpy as jnp
    from su2_nqs.benchmarks.projected_singlet_agp import triangular_hnf_gamma_a1_group
    from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout
    from su2_nqs.benchmarks.spin_operator_information import three_bond_readout
    from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
    from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange
    from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries
    from su2_nqs.vmc.sr import importance_amplitude_metric, importance_hermitian_force
    from su2_nqs.vmc.regional_density import (regional_query_plan,
        conditional_regional_matrices, regional_density_summary)
    from su2_nqs.frontier.provenance import file_sha256

    spec = json.loads(config_path.read_text())
    if spec.get('task') != 'spin_pair_n36_continue' or spec.get('n_sites') != 36:
        raise ValueError('this protocol supports triangular6x6 PBC only')
    coupling = _pair_registered_coupling(spec)
    for key in ('step_cap','chains','burn_sweeps','audit_burn_sweeps','decorrelation_sweeps',
            'training_blocks','geometry_blocks','audit_blocks','line_search_cap','tile_size','H_batch_size'):
        if isinstance(spec[key],bool) or not isinstance(spec[key],int) or spec[key]<1:
            raise ValueError(key+' must be a positive integer')
    if spec['chains']<2 or spec['training_blocks']*spec['chains']%spec['H_batch_size']:
        raise ValueError('independent chains and complete full-H batches required')
    force_estimator=spec.get('force_estimator','both_sided_quadrature')
    if force_estimator not in ('both_sided_quadrature','hermitian_center'):
        raise ValueError('unrecognized physical force estimator')
    for key in ('positive_floor','relative_damping','absolute_damping','maximum_sine_FS',
            'maximum_variance_ratio','step_size_total_energy','training_wall_seconds','total_wall_seconds'):
        if not np.isfinite(spec[key]) or spec[key]<=0:
            raise ValueError(key+' must be positive finite')
    train_seeds = {spec['training_seed']+i for i in range(spec['step_cap'])}
    geometry_seeds = {spec['geometry_seed']+i for i in range(spec['step_cap'])}
    audit_seeds = set(spec['audit_seeds'])
    if (train_seeds & geometry_seeds or train_seeds & audit_seeds
            or geometry_seeds & audit_seeds or len(audit_seeds) != len(spec['audit_seeds'])):
        raise ValueError('training/geometry/audit seed schedules must be disjoint')
    noise_spec = spec.get('initial_pair_noise')
    if noise_spec:
        if ({noise_spec['seed'], noise_spec['calibration_seed']} &
                (train_seeds | geometry_seeds | audit_seeds)
                or noise_spec['seed'] == noise_spec['calibration_seed']):
            raise ValueError('initialization seeds must be separate from training/audit')
        if not 0 < noise_spec['maximum_sine_FS'] <= spec['maximum_sine_FS']:
            raise ValueError('initialization physical radius exceeds registered update radius')
    parent = ROOT/spec['parent_source']; out = ROOT/spec['output']
    sealed = json.loads((parent/'completion.json').read_text())
    for name, digest in sealed['outputs'].items():
        if file_sha256(parent/name) != digest:
            raise ValueError('parent evidence seal changed')
    initial_checkpoint = parent/'endpoint.pkl'
    starting = spec.get('starting_checkpoint')
    if starting:
        initial_checkpoint = ROOT/starting['path']
        if file_sha256(initial_checkpoint) != starting['sha256']:
            raise ValueError('sealed starting checkpoint changed')
    previous = load_checkpoint(initial_checkpoint)
    if previous['metadata'].get('n_sites') != 36:
        raise ValueError('starting checkpoint must be a compatible N36 pair state')
    initial = np.asarray(previous['params']['pairing_real'], float)
    base = np.asarray(load_checkpoint(parent/'carrier.pkl')['params']['pairing_real'], float)
    scale = float(previous['metadata']['amplitude_scale'])
    controls={}
    for label,item in spec.get('frozen_controls',{}).items():
        control_path=ROOT/item['path']
        if label in ('original_carrier','previous_endpoint','endpoint') or file_sha256(control_path)!=item['sha256']:
            raise ValueError('control label or frozen checkpoint seal invalid')
        controls[label]=np.asarray(load_checkpoint(control_path)['params']['pairing_real'],float)
    sources = [Path(__file__), ROOT/'su2_nqs/vmc/sr.py',
        ROOT/'su2_nqs/models/projected_singlet_agp.py', ROOT/'su2_nqs/vmc/physical_h_queries.py',
        ROOT/'su2_nqs/vmc/fixed_wavefunction_mcmc.py', ROOT/'su2_nqs/vmc/checkpoint.py',
        ROOT/'su2_nqs/vmc/regional_density.py', ROOT/'su2_nqs/benchmarks/spin_operator_information.py',
        ROOT/'su2_nqs/benchmarks/tensor_network_readout.py',
        ROOT/'su2_nqs/benchmarks/projected_singlet_agp.py',
        ROOT/'su2_nqs/lattices/triangular_j1j2.py', ROOT/'su2_nqs/operators/spin_config.py']
    if spec.get('structure_factor_readout', False):
        sources.append(ROOT/'su2_nqs/benchmarks/singlet_physics_readout.py')
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}
    if resume:
        registration = json.loads((out/'registration.json').read_text())
        if (registration['config_sha256'] != file_sha256(config_path)
                or registration['source_sha256'] != hashes):
            raise ValueError('resume requires unchanged registered config and source')
        if (out/'frozen_endpoint.json').exists():
            raise ValueError('frozen audit phase cannot be reopened for training')
        restored = load_checkpoint(out/'training.pkl')
        p = np.asarray(restored['params']['pairing_real'], float)
        state = restored['optimizer_state']
        history = state['history']; elapsed_before = float(state['elapsed_seconds'])
        ledger = state['ledger']; start_step = int(state['next_step'])
    else:
        out.mkdir(parents=True, exist_ok=False)
        for path in sources:
            target = out/'sources'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        atomic_json(out/'registration.json', dict(config=spec,
            config_sha256=file_sha256(config_path), source_sha256=hashes,
            parent_completion_sha256=file_sha256(parent/'completion.json'),
            starting_checkpoint_path=str(initial_checkpoint.relative_to(ROOT)),
            starting_checkpoint_sha256=file_sha256(initial_checkpoint),
            starting_checkpoint_coupling=previous['metadata'].get('j2'),
            runtime=numerical_runtime_identity(), audit_selects_endpoint=False,
            protocol='last completed SR step; independent calibration; audit AFTER freeze'))
        p = initial.copy(); history = []; elapsed_before = 0.; start_step = 0
        ledger = dict(training_H_centers=0, audit_H_centers=0,
            physical_unique_queries=0, executed_padded_queries=0)
    began = time.perf_counter()
    elapsed = lambda: elapsed_before + time.perf_counter()-began
    cfg = ProjectedSingletAGPConfig(36, triangular_hnf_gamma_a1_group(((6,0),(0,6))))
    fns = _pair_dynamic_functions(cfg, amplitude_scale=scale, tile_size=spec['tile_size'])
    amp = lambda params, x: np.asarray(fns['amplitude'](jnp.asarray(params), jnp.asarray(x)))
    bonds = triangular_j1j2_bonds(6,6,j1=1.,j2=coupling,periodic=True)
    gauge_norm = float(np.linalg.norm(initial)); floor = spec['positive_floor']
    cost = ROOT/json.loads((parent/'results.json').read_text())['config']['cost_source']
    cost_completion=json.loads((cost/'completion.json').read_text())
    if any(file_sha256(cost/name)!=digest for name,digest in cost_completion['outputs'].items()):
        raise ValueError('cost preflight seal changed')
    with np.load(cost/'probe.npz') as cache:
        probe_p=cache['probe_pairing_real']; probe_x=cache['configurations']
        expected=cache['probe_amplitudes']
        replay=float(np.max(abs(amp(probe_p,probe_x)-expected)))
    probe_plan=compile_physical_h_queries(probe_x[:spec['H_batch_size']],bonds)
    ref_action,_=_pair_query_functions(cfg,probe_plan,amplitude_scale=scale,tile_size=spec['tile_size'])
    pair_ref=ref_action(probe_p)
    pair_new=fns['action'](probe_p,*_pair_padded_query_arguments(probe_plan))
    action_error=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(pair_ref,pair_new))
    if replay>1e-11 or action_error>1e-11:
        raise ValueError('N36 complete-projection/H checkpoint replay failed')
    atomic_json(out/'validation.json',dict(checkpoint_amplitude_maximum_error=replay,
        dynamic_full_H_maximum_error=action_error,tolerance=1e-11))

    def retracted(raw):
        return raw * gauge_norm / np.linalg.norm(raw)

    def draw(density, *, seed, blocks, burn=None):
        return sample_fixed_wavefunction_exchange(lambda x: np.sqrt(density(x)),
            n_sites=36,n_up=18,chain_count=spec['chains'],
            burn_sweeps=spec['burn_sweeps'] if burn is None else burn,
            measurement_blocks=blocks,decorrelation_sweeps=spec['decorrelation_sweeps'],seed=seed)

    def plans(x):
        result = []
        for i in range(0,len(x),spec['H_batch_size']):
            plan = compile_physical_h_queries(x[i:i+spec['H_batch_size']],bonds)
            result.append((i, plan, _pair_padded_query_arguments(plan)))
        return result

    def evaluate(params, x, q, query_plans, *, gradient=False, phase='training'):
        aa=[]; bb=[]; num=den=0.; gn=np.zeros_like(params); gd=np.zeros_like(params)
        for i, plan, args in query_plans:
            density = jnp.asarray(q[i:i+len(plan.centers)])
            if gradient:
                ((n,(d,_,a,b)), g) = fns['numerator'](jnp.asarray(params),*args,density)
                normg = fns['norm_gradient'](jnp.asarray(params),jnp.asarray(plan.centers),density)
                num+=float(n); den+=float(d); gn+=np.asarray(g); gd+=np.asarray(normg)
            else:
                a,b = fns['action'](jnp.asarray(params),*args)
            aa.append(np.asarray(a)); bb.append(np.asarray(b))
            ledger[phase+'_H_centers'] += len(plan.centers)
            ledger['physical_unique_queries'] += len(plan.configurations)
            ledger['executed_padded_queries'] += len(args[0])
        a,b=np.concatenate(aa),np.concatenate(bb)
        moments=_pair_weighted_moments(a,b,q,36)
        grad=None if not gradient else (gn-(num/den)*gd)/den/36
        return moments,grad,(a,b)

    def checkpoint(next_step):
        save_checkpoint(out/'training.pkl', {'pairing_real':p},
            optimizer_state=dict(next_step=next_step,history=history,ledger=ledger,
                elapsed_seconds=elapsed(),gauge_norm=gauge_norm),
            metadata=dict(amplitude_scale=scale,n_sites=36,j2=coupling,
                optimizer='importance-amplitude SR; fresh deterministic samples per completed step',
                resume_contract='atomic completed-step boundary; unfinished step repeats deterministically'))
        atomic_json(out/'progress.json', dict(history=history,ledger=ledger,seconds=elapsed()))

    if noise_spec and not resume:
        perturbation = _pair_generic_perturbation(initial,
            relative_norm=noise_spec['relative_norm'], seed=noise_spec['seed'])
        init_density = lambda x: abs(amp(initial,x))**2 + floor
        init_sample = draw(init_density, seed=noise_spec['calibration_seed'],
            blocks=spec['training_blocks'])
        ix = init_sample.configurations; iq = init_density(ix); ia = amp(initial,ix)
        fraction = 1.
        for _ in range(16):
            p = retracted(initial + fraction * perturbation)
            init_distance = _pair_sine_fs(ia,amp(p,ix),iq)
            if init_distance <= noise_spec['maximum_sine_FS']:
                break
            fraction *= .5
        else:
            raise ValueError('generic initialization did not fit its physical trust radius')
        # An amplitude-only diagnostic, not an energy-based seed search.
        ranks = {}
        for label, values in (('parent', initial), ('initialized', p)):
            j = np.concatenate([np.asarray(fns['center_jacobian'](jnp.asarray(values),
                jnp.asarray(ix[i:i+spec['tile_size']]))) for i in range(0,64,spec['tile_size'])])
            m = importance_amplitude_metric(amp(values,ix[:64]),j,iq[:64])
            eig = np.linalg.eigvalsh(m)
            ranks[label] = int(np.count_nonzero(eig > max(eig[-1]*1e-8, 1e-12)))
        save_checkpoint(out/'initialization.pkl', {'pairing_real':p}, metadata=dict(
            amplitude_scale=scale,n_sites=36,group_order=432,spin_sector='q=0',
            selected_using='one preregistered generic seed; amplitude-only physical radius'))
        atomic_json(out/'initialization.json',dict(seed=noise_spec['seed'],
            calibration_seed=noise_spec['calibration_seed'],accepted_fraction=fraction,
            sine_FS=init_distance,physical_metric_rank_on_64=ranks,
            energy_used_for_selection=False,physical_symmetry_unchanged=True,
            seconds=elapsed()))
        print(json.dumps(dict(initialization_sine_FS=init_distance,
            physical_metric_rank_on_64=ranks)),flush=True)
    checkpoint(start_step)
    for step in range(start_step, spec['step_cap']):
        if elapsed() > spec['training_wall_seconds']:
            break
        step_start = time.perf_counter(); parent_p = p.copy()
        density = lambda x: abs(amp(parent_p,x))**2 + floor
        sample = draw(density, seed=spec['training_seed']+step, blocks=spec['training_blocks'])
        x=sample.configurations; q=density(x); query_plans=plans(x)
        old, g, arrays = evaluate(p,x,q,query_plans,
            gradient=force_estimator=='both_sided_quadrature')
        jac=np.concatenate([np.asarray(fns['center_jacobian'](jnp.asarray(p),
            jnp.asarray(x[i:i+spec['tile_size']]))) for i in range(0,len(x),spec['tile_size'])])
        metric=importance_amplitude_metric(arrays[0],jac,q)
        if force_estimator=='hermitian_center':
            g=importance_hermitian_force(arrays[0],jac,arrays[1],q,n_sites=36)
        # Homogeneous F scale is a redundant physical direction, not a bond degree of freedom.
        unit=p/np.linalg.norm(p); projector=np.eye(len(p))-np.outer(unit,unit)
        metric=projector@metric@projector; g=projector@g
        metric_eigenvalues=np.linalg.eigvalsh(metric)
        metric_rank=int(np.count_nonzero(metric_eigenvalues>
            max(metric_eigenvalues[-1]*1e-8,1e-12)))
        damping=spec['relative_damping']*np.trace(metric)/len(p)+spec['absolute_damping']
        direction=np.linalg.solve(metric+damping*np.eye(len(p)),g)
        direction=projector@direction
        delta=-spec['step_size_total_energy']*36*direction
        tangent_fs=np.sqrt(max(0.,delta@metric@delta/2))
        if tangent_fs>spec['maximum_sine_FS']:
            delta*=spec['maximum_sine_FS']/tangent_fs
        fs_tangent=np.sqrt(max(0.,delta@metric@delta/2))
        force_bound=2*np.sqrt(old[1]/36)*fs_tangent
        force_ratio=abs(float(g@delta))/max(force_bound,np.finfo(float).tiny)
        if force_estimator=='hermitian_center' and force_ratio>1.000001:
            raise ValueError('canonical force is inconsistent with the sampled physical metric')
        # Independent cheap configurations calibrate physical distance, not energy selection.
        cal=draw(density,seed=spec['geometry_seed']+step,blocks=spec['geometry_blocks'])
        cx=cal.configurations; cq=density(cx); ca=amp(p,cx)
        trial_fraction=1.; accepted=False; tries=[]; best_candidate=None
        changed=old; training_fs=calibration_fs=0.
        for attempt in range(spec['line_search_cap']):
            candidate=retracted(p+trial_fraction*delta)
            training_fs=_pair_sine_fs(arrays[0],amp(candidate,x),q)
            calibration_fs=_pair_sine_fs(ca,amp(candidate,cx),cq)
            if max(training_fs,calibration_fs)>spec['maximum_sine_FS']*1.02:
                tries.append(dict(fraction=trial_fraction,status='physical_radius')); trial_fraction*=.5; continue
            new,_,_=evaluate(candidate,x,q,query_plans)
            sufficient=(new[0]<old[0] and new[1]<=spec['maximum_variance_ratio']*old[1])
            tries.append(dict(fraction=trial_fraction,energy_per_site=float(new[0]),
                variance_per_site=float(new[1]),status='accepted' if sufficient else 'training_objective'))
            if sufficient:
                if best_candidate is None or new[0]<best_candidate[1][0]:
                    best_candidate=(candidate,new,training_fs,calibration_fs,trial_fraction)
                if not spec.get('secant_curvature_step',False):
                    break
                # At most one curvature probe after the first acceptable step.
                # Predicted slope comes from the canonical physical force;
                # actual energies choose the step, never the untouched audit.
                if sum(t.get('status')=='accepted' for t in tries)>1:
                    break
                slope=float(g@(candidate-p));curvature=float(new[0]-old[0]-slope)
                optimum=-slope/(2*curvature) if curvature>0 else 1.
                if .1<optimum<.9:
                    trial_fraction*=optimum; continue
                break
            trial_fraction*=.5
        if best_candidate is not None:
            p,changed,training_fs,calibration_fs,trial_fraction=best_candidate;accepted=True
        row=dict(step=step+1,accepted=accepted,energy_before=float(old[0]),
            variance_before=float(old[1]),energy_after=float(changed[0]),variance_after=float(changed[1]),
            gradient_norm=float(np.linalg.norm(g)),damping=float(damping),
            physical_metric_rank=metric_rank,
            force_estimator=force_estimator,force_Cauchy_ratio=force_ratio,
            predicted_energy_change_per_site=float(g@(trial_fraction*delta)),
            training_sine_FS=training_fs,calibration_sine_FS=calibration_fs,
            training_acceptance=sample.acceptance_fraction,geometry_acceptance=cal.acceptance_fraction,
            trials=tries,step_seconds=time.perf_counter()-step_start,seconds=elapsed())
        history.append(row); checkpoint(step+1); print(json.dumps(row),flush=True)
        if step+1 in spec.get('milestone_steps', ()):
            save_checkpoint(out/f'milestone_step{step+1:03d}.pkl', {'pairing_real':p},
                metadata=dict(n_sites=36,j2=coupling,group_order=432,amplitude_scale=scale,
                    completed_step=step+1,selected_using='preregistered quality milestone; no audit'))
        np.savez_compressed(out/'last_training_batch.npz',configurations=x,density=q,
            parent_pairing_real=parent_p,gradient=g,metric=metric,applied_pairing_real=p)
    # Freeze LAST completed step. Neither audit chooses an endpoint or extends training.
    params=dict(original_carrier=base,previous_endpoint=initial,**controls,endpoint=p.copy())
    for label, values in params.items():
        save_checkpoint(out/(label+'.pkl'), {'pairing_real':values}, metadata=dict(
            n_sites=36,j2=coupling,group_order=432,amplitude_scale=scale,audit_opened=False,
            selected_using='last completed preregistered SR step, never audit'))
    frozen={label:file_sha256(out/(label+'.pkl')) for label in params}
    atomic_json(out/'frozen_endpoint.json',dict(checkpoints=frozen,completed_steps=len(history),
        seconds=elapsed(),audit_seeds=spec['audit_seeds'],audit_opened=False))
    audits=[]; saved={label+'_pairing_real':v for label,v in params.items()}
    pooled_arrays={label:([],[]) for label in params}; pooled_density=[]
    for number,seed in enumerate(spec['audit_seeds']):
        if elapsed()>spec['total_wall_seconds']:
            break
        adensity=lambda x:sum(abs(amp(v,x))**2 for v in params.values())+floor
        audit=draw(adensity,seed=seed,blocks=spec['audit_blocks'],burn=spec['audit_burn_sweeps'])
        ax=audit.configurations; aq=adensity(ax); query_plans=plans(ax); arrays={}
        for label,values in params.items():
            _,_,arrays[label]=evaluate(values,ax,aq,query_plans,phase='audit')
            pooled_arrays[label][0].append(arrays[label][0]); pooled_arrays[label][1].append(arrays[label][1])
            saved[f'audit{number}_{label}_psi']=arrays[label][0]
            saved[f'audit{number}_{label}_Hpsi']=arrays[label][1]
        pooled_density.append(aq); saved[f'audit{number}_configurations']=ax; saved[f'audit{number}_density']=aq
        report=_pair_chain_comparison(arrays,aq,spec['chains'],36)
        report['paired']['endpoint-minus-previous_endpoint']=_pair_chain_comparison(
            {k:arrays[k] for k in ('previous_endpoint','endpoint')},aq,spec['chains'],36)['paired']['endpoint-minus-previous_endpoint']
        report['endpoint_minus_original']=_pair_chain_comparison(
            {k:arrays[k] for k in ('original_carrier','endpoint')},aq,spec['chains'],36)['paired']['endpoint-minus-original_carrier']
        report.update(seed=seed,acceptance=audit.acceptance_fraction,regions={})
        # Same-family quality-dependence only; no other candidate or phase label.
        groups=np.arange(len(ax))%spec['chains']
        if spec.get('structure_factor_readout', False):
            from su2_nqs.benchmarks.singlet_physics_readout import (
                triangular_torus_structure_records, importance_jackknife_observables)
            records=triangular_torus_structure_records(ax,6,6,strict_singlet=True)
            report['structure_factors']={}
            for label,(a,_) in arrays.items():
                sf=importance_jackknife_observables(records,a,aq,groups)
                for name,item in sf['observables'].items():
                    saved[f'audit{number}_{label}_{name}_jackknife']=item.pop('jackknife_values')
                report['structure_factors'][label]=sf
        for name,region in spec['regions'].items():
            plan=regional_query_plan(ax,region['sites']); rq=adensity(plan.configurations)
            for label,values in params.items():
                values_a=amp(values,plan.configurations)
                summary=regional_density_summary(conditional_regional_matrices(plan,values_a,rq),groups)
                rho=summary['rho']; jacks=summary['jackknife_rhos']
                observables={}
                if len(region['sites']) == 6:
                    def observe(r):
                        t=three_bond_readout(r,region['left'])
                        return np.array([t['orbits']['Y'][k] for k in ('raw','connected')]+[t['high_q3']])
                    central=observe(rho); jack=np.array([observe(r) for r in jacks])
                    se=np.sqrt((spec['chains']-1)*np.mean((jack-jack.mean(0))**2,axis=0))
                    observables=dict(zip(('Y_raw','Y_connected','high_q3'),[
                        dict(value=float(v),chain_SE=float(s)) for v,s in zip(central,se)]))
                report['regions'].setdefault(name,{})[label]=dict(
                    readout=physical_spin_block_readout(rho),
                    observables=observables,
                    strict_spectrum_claim=False)
                saved[f'audit{number}_{name}_{label}_rho']=rho
                saved[f'audit{number}_{name}_{label}_jackknife_rhos']=jacks
        audits.append(report)
        atomic_json(out/f'audit{number}.json',report)
        np.savez_compressed(out/'samples_states_readouts.npz',**saved)
        print(json.dumps(dict(audit_seed=seed,moments=report['states'],paired=report['paired'])),flush=True)
    # Reorder blocks into block x ALL independent chains before pooled jackknife.
    if audits:
        def interleave(chunks):
            return np.concatenate([a.reshape(spec['audit_blocks'],spec['chains']) for a in chunks],axis=1).reshape(-1)
        all_arrays={k:tuple(interleave(a) for a in ab) for k,ab in pooled_arrays.items()}
        all_q=interleave(pooled_density)
        comparison=_pair_chain_comparison(all_arrays,all_q,len(audits)*spec['chains'],36)
        comparison['endpoint_minus_previous']=_pair_chain_comparison(
            {k:all_arrays[k] for k in ('previous_endpoint','endpoint')},all_q,
            len(audits)*spec['chains'],36)['paired']['endpoint-minus-previous_endpoint']
        comparison['paired']['endpoint-minus-previous_endpoint']=comparison['endpoint_minus_previous']
        comparison['endpoint_minus_original']=_pair_chain_comparison(
            {k:all_arrays[k] for k in ('original_carrier','endpoint')},all_q,
            len(audits)*spec['chains'],36)['paired']['endpoint-minus-original_carrier']
    else:
        comparison=dict(status='no_audit_completed_within_budget')
    for label,digest in frozen.items():
        if file_sha256(out/(label+'.pkl'))!=digest:
            raise RuntimeError('audit modified frozen physical state')
    result=dict(config=spec,history=history,audits=audits,pooled=comparison,ledger=ledger,
        seconds=elapsed(),completed_steps=len(history),accepted_steps=sum(h['accepted'] for h in history),
        parameter_count=630,candidate_families=1,new_heads=0,new_many_body_ED=0,
        remote_jobs=0,phase_claim=False,equilibrium_certified=False,convergence_certified=False,
        force_estimator=force_estimator,
        checkpoint_amplitude_maximum_error=replay,dynamic_full_H_maximum_error=action_error,
        warnings=['Chain SE does not certify equilibrium or unseen tails.',
            'Importance ESS is not Markov-chain ESS.',
            'Physical differences along this quality path are not differences between phases.',
            'No state is selected using the independent audit.'])
    atomic_json(out/'results.json',result)
    atomic_json(out/'completion.json',dict(status='complete',outputs={
        p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(completed_steps=len(history),seconds=elapsed(),pooled=comparison)),flush=True)
    return result


__all__ = ["TASK", "run_spin_adaptive_projected_pair_carrier", "run_pair_quality", "run_pair_n36_cost", "run_pair_n36_fit", "run_pair_n36_continue"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explicit bounded pair-family quality refinement; old broad factorial runs use the project runner.')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pair-quality',type=Path)
    modes.add_argument('--n36-cost',type=Path)
    modes.add_argument('--n36-fit',type=Path)
    modes.add_argument('--n36-continue',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.pair_quality:
        run_pair_quality(args.pair_quality)
    elif args.n36_cost:
        run_pair_n36_cost(args.n36_cost)
    elif args.n36_fit:
        run_pair_n36_fit(args.n36_fit)
    else:
        run_pair_n36_continue(args.n36_continue,resume=args.resume)
