"""Exact-small gate for a physical-amplitude simplex/F-move dressing.

This workflow deliberately keeps the projected pi-flux/Krylov carrier in the
physical spin-amplitude representation.  A local SU(2)-scalar recoupling
operator is applied there and only its output is projected to ``Gamma-A1``.
The construction therefore tests a new operator direction, rather than an
attempt to encode the carrier as one global fusion-coefficient tensor.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import computational_basis
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.benchmarks.triangular_state_manifold import (
    TRIANGULAR_N16_RHOMBUS_SUPPORTS,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.simplex_f_resonance import (
    simplex_connected_f_resonance_action,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_physical_fmove_carrier_gate"
SUPERCELL = ((4, 0), (0, 4))
N_SITES = 16


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid physical F-move carrier gate identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the local comparison must be preregistered")
    source = spec.get("source")
    gates = spec.get("gates")
    if not isinstance(source, dict) or not isinstance(gates, dict):
        raise TypeError("source and gates must be objects")
    for name in ("result", "result_sha256", "bundle", "bundle_sha256"):
        if not isinstance(source.get(name), str) or not source[name]:
            raise ValueError(f"source.{name} must be a nonempty string")
    if int(spec.get("scramble_seed", -1)) < 0:
        raise ValueError("scramble_seed must be nonnegative")
    if spec.get("remote_training_on_pass") is not False:
        raise ValueError("this exact-small gate must not submit remote work")


def _load_source(spec: dict[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    source = spec["source"]
    assert isinstance(source, dict)
    result_path = ROOT / str(source["result"])
    bundle_path = ROOT / str(source["bundle"])
    result = json.loads(result_path.read_text())
    seal = str(source["result_sha256"])
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    if result.get("result_sha256") != seal or canonical_sha256(body) != seal:
        raise RuntimeError("source result seal changed")
    if file_sha256(bundle_path) != str(source["bundle_sha256"]):
        raise RuntimeError("source state bundle seal changed")
    with np.load(bundle_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return result, arrays


def _fixed_sz_configurations() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((state >> (N_SITES - 1 - site)) & 1 for site in range(N_SITES))
        for state in computational_basis(N_SITES, N_SITES // 2)
    )


def _cyclic_simplex_action(
    state: np.ndarray,
    pairs: FixedSzPairActions,
    cells: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> np.ndarray:
    answer = np.zeros_like(np.asarray(state, dtype=np.complex128))
    a, b, c = cells
    for left, middle, right in ((a, b, c), (b, c, a), (c, a, b)):
        answer += simplex_connected_f_resonance_action(
            state, pairs.pair, left, middle, right
        )
    return answer / 3.0


def _compressed_operator(
    physical_a1: np.ndarray,
    pairs: FixedSzPairActions,
    cells: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    images = np.column_stack(
        [
            _cyclic_simplex_action(physical_a1[:, column], pairs, cells)
            for column in range(physical_a1.shape[1])
        ]
    )
    block = physical_a1.conj().T @ images
    block = 0.5 * (block + block.conj().T)
    return np.asarray(block), images


def _normalized(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state has zero or nonfinite norm")
    return vector / norm


def _metrics(state: np.ndarray, hamiltonian: np.ndarray) -> dict[str, float]:
    vector = _normalized(state)
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return {
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real / N_SITES),
    }


def _krylov_columns(
    carrier: np.ndarray, hamiltonian: np.ndarray, degree: int
) -> np.ndarray:
    columns = [_normalized(carrier)]
    for _ in range(int(degree)):
        columns.append(hamiltonian @ columns[-1])
    return np.column_stack(columns)


def _ritz(
    columns: np.ndarray, hamiltonian: np.ndarray, *, tolerance: float = 1.0e-11
) -> tuple[np.ndarray, int]:
    left, singular, _ = np.linalg.svd(np.asarray(columns), full_matrices=False)
    retained = singular > max(1.0e-14, tolerance * singular[0])
    basis = left[:, retained]
    projected = basis.conj().T @ hamiltonian @ basis
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    return _normalized(basis @ vectors[:, 0]), int(np.sum(retained))


def _outside_fraction(direction: np.ndarray, columns: np.ndarray) -> float:
    vector = np.asarray(direction, dtype=np.complex128)
    left, singular, _ = np.linalg.svd(np.asarray(columns), full_matrices=False)
    retained = singular > max(1.0e-14, 1.0e-11 * singular[0])
    basis = left[:, retained]
    residual = vector - basis @ (basis.conj().T @ vector)
    return float(
        np.vdot(residual, residual).real
        / max(np.vdot(vector, vector).real, 1.0e-30)
    )


def run_spin_n16_physical_fmove_carrier_gate(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Run and seal the physical-amplitude F-move carrier comparison."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    source_result, arrays = _load_source(spec)
    physical_a1 = np.asarray(arrays["physical_a1_basis"], dtype=np.complex128)
    carrier = _normalized(arrays["coupling_0.125_carrier_a1"])
    sealed_two = _normalized(arrays["coupling_0.125_two_lanczos_a1"])
    hamiltonian = np.asarray(arrays["coupling_0.125_h_a1"], dtype=np.complex128)
    if physical_a1.shape != (12870, 14) or hamiltonian.shape != (14, 14):
        raise RuntimeError("sealed N16 physical/A1 dimensions changed")

    configurations = _fixed_sz_configurations()
    pairs = FixedSzPairActions(configurations)
    # Coarse coordinates (0,0),(1,0),(1,1) form an elementary triangle for
    # the repository's J1 directions (1,0),(0,1),(1,1).  The tempting
    # (0,0),(1,0),(0,1) choice is *not* a closed simplex in this convention.
    cells = tuple(
        tuple(TRIANGULAR_N16_RHOMBUS_SUPPORTS[index])
        for index in (0, 1, 3)
    )
    physical_operator, unprojected_images = _compressed_operator(
        physical_a1, pairs, cells
    )
    scramble = np.random.default_rng(int(spec["scramble_seed"])).permutation(N_SITES)
    scrambled_cells = tuple(
        tuple(int(scramble[site]) for site in cell) for cell in cells
    )
    scrambled_operator, _ = _compressed_operator(
        physical_a1, pairs, scrambled_cells
    )

    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    group_image_cells = tuple(
        tuple(int(group[17][site]) for site in cell) for cell in cells
    )
    image_operator, _ = _compressed_operator(physical_a1, pairs, group_image_cells)
    reynolds_matrix_element_error = float(
        np.max(np.abs(image_operator - physical_operator))
    )
    projected_images = physical_a1 @ physical_operator
    spatial_projection_fraction = float(
        np.linalg.norm(projected_images) ** 2
        / max(np.linalg.norm(unprojected_images) ** 2, 1.0e-30)
    )
    maximum_splus = max(
        spin_raising_relative_norm(unprojected_images[:, column], configurations)
        for column in range(unprojected_images.shape[1])
    )

    k2_columns = _krylov_columns(carrier, hamiltonian, 2)
    replay_two, k2_rank = _ritz(k2_columns, hamiltonian)
    k3_columns = _krylov_columns(carrier, hamiltonian, 3)
    k4_columns = _krylov_columns(carrier, hamiltonian, 4)
    k5_columns = _krylov_columns(carrier, hamiltonian, 5)
    k3, k3_rank = _ritz(k3_columns, hamiltonian)
    k4, k4_rank = _ritz(k4_columns, hamiltonian)
    k5, k5_rank = _ritz(k5_columns, hamiltonian)
    replay_fidelity = float(abs(np.vdot(replay_two, sealed_two)) ** 2)

    r_two = physical_operator @ sealed_two
    r_two_scrambled = scrambled_operator @ sealed_two
    k2_plus_r, k2_plus_r_rank = _ritz(
        np.column_stack((k2_columns, r_two)), hamiltonian
    )
    k2_plus_scrambled, k2_plus_scrambled_rank = _ritz(
        np.column_stack((k2_columns, r_two_scrambled)), hamiltonian
    )
    r_four = physical_operator @ k4
    r_four_scrambled = scrambled_operator @ k4
    k4_plus_r, k4_plus_r_rank = _ritz(
        np.column_stack((k4_columns, r_four)), hamiltonian
    )
    k4_plus_scrambled, k4_plus_scrambled_rank = _ritz(
        np.column_stack((k4_columns, r_four_scrambled)), hamiltonian
    )
    states = {
        "projected_pi_flux": carrier,
        "two_lanczos": sealed_two,
        "degree_three_krylov": k3,
        "degree_four_krylov": k4,
        "degree_five_krylov": k5,
        "two_lanczos_plus_physical_fmove": k2_plus_r,
        "two_lanczos_plus_scrambled_fmove": k2_plus_scrambled,
        "degree_four_plus_physical_fmove": k4_plus_r,
        "degree_four_plus_scrambled_fmove": k4_plus_scrambled,
    }
    metrics = {name: _metrics(state, hamiltonian) for name, state in states.items()}
    depth_ladder = []
    for degree in range(7):
        columns = _krylov_columns(carrier, hamiltonian, degree)
        krylov_state, krylov_rank = _ritz(columns, hamiltonian)
        next_state, next_rank = _ritz(
            _krylov_columns(carrier, hamiltonian, degree + 1), hamiltonian
        )
        physical_state, physical_rank = _ritz(
            np.column_stack((columns, physical_operator @ krylov_state)),
            hamiltonian,
        )
        scramble_state, scramble_rank = _ritz(
            np.column_stack((columns, scrambled_operator @ krylov_state)),
            hamiltonian,
        )
        krylov_row = _metrics(krylov_state, hamiltonian)
        next_row = _metrics(next_state, hamiltonian)
        physical_row = _metrics(physical_state, hamiltonian)
        scramble_row = _metrics(scramble_state, hamiltonian)
        depth_ladder.append(
            {
                "carrier_krylov_degree": degree,
                "carrier_krylov_rank": krylov_rank,
                "next_krylov_rank": next_rank,
                "physical_extension_rank": physical_rank,
                "scrambled_extension_rank": scramble_rank,
                "carrier_krylov": krylov_row,
                "next_degree_krylov": next_row,
                "physical_fmove_extension": physical_row,
                "scrambled_fmove_extension": scramble_row,
                "physical_gain_vs_same_degree_per_site": float(
                    physical_row["energy_per_site"]
                    - krylov_row["energy_per_site"]
                ),
                "physical_minus_next_krylov_per_site": float(
                    physical_row["energy_per_site"] - next_row["energy_per_site"]
                ),
                "physical_minus_scramble_per_site": float(
                    physical_row["energy_per_site"]
                    - scramble_row["energy_per_site"]
                ),
            }
        )
    physical_small = metrics["two_lanczos_plus_physical_fmove"]
    scramble_small = metrics["two_lanczos_plus_scrambled_fmove"]
    physical_large = metrics["degree_four_plus_physical_fmove"]
    scramble_large = metrics["degree_four_plus_scrambled_fmove"]
    k3_metrics = metrics["degree_three_krylov"]
    k5_metrics = metrics["degree_five_krylov"]
    gates_spec = spec["gates"]
    assert isinstance(gates_spec, dict)
    minimum_outside = float(gates_spec["minimum_non_krylov_norm_fraction"])
    diagnostics = {
        "fmove_definition": (
            "P_GammaA1 [ cyclic_sum_{ABC} ({Q_A.Q_B,Q_B.Q_C}/2 "
            "- |B| Q_A.Q_C/4) / 3 ] P_GammaA1"
        ),
        "physical_representation": (
            "operator acts on fixed-Sz physical amplitudes; only the local "
            "recoupling action is projected to Gamma-A1"
        ),
        "cell_supports": [list(cell) for cell in cells],
        "scrambled_cell_supports": [list(cell) for cell in scrambled_cells],
        "maximum_group_image_matrix_element_error": reynolds_matrix_element_error,
        "maximum_unprojected_spin_raising_relative_norm": float(maximum_splus),
        "spatial_a1_projection_norm_fraction": spatial_projection_fraction,
        "operator_hermiticity_error": float(
            np.max(np.abs(physical_operator - physical_operator.conj().T))
        ),
        "two_lanczos_replay_fidelity": replay_fidelity,
        "two_lanczos_krylov_rank": k2_rank,
        "degree_three_krylov_rank": k3_rank,
        "degree_four_krylov_rank": k4_rank,
        "degree_five_krylov_rank": k5_rank,
        "two_lanczos_plus_physical_rank": k2_plus_r_rank,
        "two_lanczos_plus_scrambled_rank": k2_plus_scrambled_rank,
        "degree_four_plus_physical_rank": k4_plus_r_rank,
        "degree_four_plus_scrambled_rank": k4_plus_scrambled_rank,
        "r_two_outside_k2_fraction": _outside_fraction(r_two, k2_columns),
        "r_two_outside_k3_fraction": _outside_fraction(r_two, k3_columns),
        "r_four_outside_k4_fraction": _outside_fraction(r_four, k4_columns),
        "scrambled_r_four_outside_k4_fraction": _outside_fraction(
            r_four_scrambled, k4_columns
        ),
    }
    gates = {
        "exact_symmetry_contract": bool(
            reynolds_matrix_element_error < 2.0e-10
            and maximum_splus < 2.0e-10
            and diagnostics["operator_hermiticity_error"] < 2.0e-10
            and replay_fidelity > 1.0 - 2.0e-10
        ),
        "non_krylov_direction_present": bool(
            diagnostics["r_four_outside_k4_fraction"] > minimum_outside
        ),
        "physical_beats_matched_scramble_small": bool(
            physical_small["energy_per_site"] < scramble_small["energy_per_site"]
        ),
        "physical_beats_next_krylov_small": bool(
            physical_small["energy_per_site"] < k3_metrics["energy_per_site"]
            and physical_small["variance_total"] <= k3_metrics["variance_total"]
        ),
        "physical_beats_matched_scramble_large": bool(
            physical_large["energy_per_site"] < scramble_large["energy_per_site"]
        ),
        "physical_beats_next_krylov_large": bool(
            physical_large["energy_per_site"] < k5_metrics["energy_per_site"]
            and physical_large["variance_total"] <= k5_metrics["variance_total"]
        ),
    }
    gates["n36_sampled_preflight_licensed"] = bool(
        gates["exact_symmetry_contract"]
        and gates["non_krylov_direction_present"]
        and gates["physical_beats_matched_scramble_large"]
        and gates["physical_beats_next_krylov_large"]
    )
    exact_anchor = source_result["frontier"]["coupling_summaries"]["0.125"][
        "exact_anchor_energy_per_site"
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "model_case": {
            "hamiltonian": "triangular J1-J2 Heisenberg",
            "boundary": "4x4 periodic",
            "j2_over_j1": 0.125,
            "sector": "exact S=0, Gamma-A1",
            "strong_carrier": "projected pi-flux with Krylov/Lanczos anchors",
            "exact_anchor_energy_per_site": exact_anchor,
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
        "matched_rank_depth_ladder": depth_ladder,
        "comparisons": {
            "physical_minus_scramble_small_energy_per_site": float(
                physical_small["energy_per_site"] - scramble_small["energy_per_site"]
            ),
            "physical_minus_degree_three_energy_per_site": float(
                physical_small["energy_per_site"] - k3_metrics["energy_per_site"]
            ),
            "physical_minus_scramble_large_energy_per_site": float(
                physical_large["energy_per_site"] - scramble_large["energy_per_site"]
            ),
            "physical_minus_degree_five_energy_per_site": float(
                physical_large["energy_per_site"] - k5_metrics["energy_per_site"]
            ),
        },
        "gates": gates,
        "decision": {
            "status": (
                "pass" if gates["n36_sampled_preflight_licensed"] else "fail"
            ),
            "remote_training_licensed": False,
            "interpretation": (
                "A pass licenses an N36 sampled Reynolds operator preflight, "
                "not a variational or scaling claim. A fail falsifies this "
                "specific local recoupling direction, not physical-amplitude "
                "operator dressing in general."
            ),
        },
        "runtime": numerical_runtime_identity(),
        "source_manifest": transitive_local_source_manifest(
            (Path(__file__).resolve(),),
            root=ROOT,
            package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
        ),
    }
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    np.savez_compressed(
        output / "states.npz",
        physical_fmove_operator_a1=physical_operator,
        scrambled_fmove_operator_a1=scrambled_operator,
        **states,
    )
    return payload


__all__ = ["TASK", "_validate", "run_spin_n16_physical_fmove_carrier_gate"]
