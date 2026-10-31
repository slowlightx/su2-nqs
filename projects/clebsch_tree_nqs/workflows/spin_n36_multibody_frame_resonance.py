"""Cross-fitted N=36 gate for collective plaquette-singlet resonance.

The old coherent-frame canary varied only the channel containing exactly two
local B1 singlets.  This workflow opens the complete local singlet-number bank
``k=0..n_cells`` while keeping every column in exact ``S=0, Gamma-A1``.  One
development stream fixes a single Ritz direction; a sampled FS/KL radius then
shortens that direction before any untouched stream is opened.  Geometry-
scrambled frame orbits provide matched-dimensional causal controls.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.coherent_plaquette_frames import (
    distinct_partition_orbit,
    nongeometric_site_permutation,
)
from su2_nqs.benchmarks.coherent_plaquette_sampling import (
    ResonanceBankSampledAction,
    build_resonance_bank_sampled_action,
    ordered_frame_orbit_with_weights,
    resonance_bank_energy_variance,
    sample_coherent_plaquette_frames,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.physical_trust_region import (
    bound_physical_state_update,
    sampled_physical_state_distance,
)


TASK = "spin_n36_multibody_frame_resonance"
TRIANGULAR_6X6 = ((6, 0), (0, 6))


jax.config.update("jax_enable_x64", True)


def _standard_error(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.std(data, ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid multibody-frame gate identity")
    controls = spec.get("controls")
    trust = spec.get("physical_trust")
    gates = spec.get("gates")
    if not all(isinstance(value, dict) for value in (controls, trust, gates)):
        raise TypeError("controls, physical_trust, and gates must be objects")
    development = controls.get("development_seeds")
    required = {"square_6x6_j2_0.5", "triangular_6x6_j2_0.125"}
    if not isinstance(development, dict) or set(development) != required:
        raise ValueError("one development seed is required for each case")
    audit = tuple(int(seed) for seed in controls.get("untouched_audit_seeds", ()))
    scrambles = tuple(int(seed) for seed in controls.get("scramble_seeds", ()))
    if len(audit) < 4 or len(set(audit)) != len(audit):
        raise ValueError("at least four distinct untouched seeds are required")
    if len(scrambles) < 4 or len(set(scrambles)) != len(scrambles):
        raise ValueError("at least four distinct scramble seeds are required")
    if set(audit).intersection(scrambles) or set(audit).intersection(
        int(value) for value in development.values()
    ):
        raise ValueError("development, audit, and scramble seeds must be disjoint")
    if int(controls.get("sample_count_per_seed", 0)) < 128:
        raise ValueError("sample_count_per_seed must be at least 128")
    cutoff = float(spec.get("relative_overlap_cutoff", np.nan))
    if not np.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("relative_overlap_cutoff must lie in (0,1)")
    for name in ("maximum_fubini_study_distance", "maximum_symmetric_kl"):
        value = float(trust.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"physical_trust.{name} must be positive")
    if float(gates.get("minimum_energy_z_score", np.nan)) < 0.0:
        raise ValueError("minimum_energy_z_score must be nonnegative")
    if spec.get("remote_training_on_pass") is not False:
        raise ValueError("this local gate must not submit remote work")


def _case_data(case_id: str):
    if case_id == "square_6x6_j2_0.5":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=False)
        group = square_gamma_a1_group(6)
        bonds = square_j1j2_bonds(6, 6, j1=1.0, j2=0.5, periodic=True)
    elif case_id == "triangular_6x6_j2_0.125":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=True)
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_6X6)
        bonds = triangular_hnf_j1j2_bonds(TRIANGULAR_6X6, j1=1.0, j2=0.125)
    else:
        raise ValueError(f"unknown multibody-frame case: {case_id}")
    return layout.physical_cells, tuple(group), tuple(bonds)


def _frame_bank(cells, group, scramble_seed: int | None):
    if scramble_seed is not None:
        image = nongeometric_site_permutation(
            n_sites=sum(len(cell) for cell in cells),
            group=group,
            seed=int(scramble_seed),
        )
        cells = tuple(tuple(image[site] for site in cell) for cell in cells)
    proposal = distinct_partition_orbit(cells, group)
    ordered, weights = ordered_frame_orbit_with_weights(cells, group)
    return proposal, ordered, weights


def _sample_action(
    cells,
    group,
    bonds,
    *,
    seed: int,
    sample_count: int,
    scramble_seed: int | None,
) -> ResonanceBankSampledAction:
    proposal, ordered, weights = _frame_bank(cells, group, scramble_seed)
    batch = sample_coherent_plaquette_frames(
        proposal, sample_count=sample_count, seed=int(seed)
    )
    return build_resonance_bank_sampled_action(batch, ordered, weights, bonds)


def _weighted_matrices(
    action: ResonanceBankSampledAction, active: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.asarray(action.amplitudes[:, active], dtype=np.complex128)
    h_columns = np.asarray(action.h_amplitudes[:, active], dtype=np.complex128)
    weight = 1.0 / np.asarray(action.proposal_probability, dtype=np.float64)
    overlap = columns.conj().T @ (weight[:, None] * columns) / len(weight)
    hamiltonian = columns.conj().T @ (weight[:, None] * h_columns) / len(weight)
    return (
        (overlap + overlap.conj().T) / 2.0,
        (hamiltonian + hamiltonian.conj().T) / 2.0,
    )


def _ritz_coefficients(
    action: ResonanceBankSampledAction,
    active: tuple[int, ...],
    relative_cutoff: float,
) -> tuple[np.ndarray, dict[str, object]]:
    overlap, hamiltonian = _weighted_matrices(action, active)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    maximum = float(eigenvalues[-1])
    retained = eigenvalues > relative_cutoff * maximum
    if not np.any(retained):
        raise RuntimeError("resonance bank has zero retained overlap rank")
    whitening = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
    reduced = whitening.conj().T @ hamiltonian @ whitening
    energies, vectors = np.linalg.eigh((reduced + reduced.conj().T) / 2.0)
    reduced_coefficients = whitening @ vectors[:, 0]
    full = np.zeros(action.amplitudes.shape[1], dtype=np.complex128)
    full[np.asarray(active, dtype=np.int64)] = reduced_coefficients
    base = np.zeros_like(full)
    base[0] = 1.0
    overlap_full, _ = _weighted_matrices(
        action, tuple(range(action.amplitudes.shape[1]))
    )
    phase = np.vdot(base, overlap_full @ full)
    if abs(phase) > 1.0e-14:
        full *= np.exp(-1j * np.angle(phase))
    return full, {
        "active_counts": list(active),
        "overlap_eigenvalues": eigenvalues.tolist(),
        "retained_rank": int(np.sum(retained)),
        "raw_ritz_energy_total": float(energies[0]),
    }


def _bounded_coefficients(
    action: ResonanceBankSampledAction,
    candidate: np.ndarray,
    spec: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    base = np.zeros(action.amplitudes.shape[1], dtype=np.complex128)
    base[0] = 1.0
    active_samples = np.flatnonzero(
        np.abs(action.amplitudes @ base) > 1.0e-14
    ).astype(np.int32)
    if len(active_samples) < 2:
        raise RuntimeError("coherent base has insufficient nonzero trust samples")
    amplitudes = jnp.asarray(action.amplitudes)

    def center(coefficients, indices):
        return amplitudes[indices] @ coefficients

    direction = jnp.asarray(base - candidate)
    trust = spec["physical_trust"]
    bounded = bound_physical_state_update(
        jnp.asarray(base),
        direction,
        jnp.asarray(active_samples),
        center_fn=center,
        maximum_fubini_study_distance=float(
            trust["maximum_fubini_study_distance"]
        ),
        maximum_symmetric_kl=float(trust["maximum_symmetric_kl"]),
    )
    final = base - np.asarray(bounded.direction)
    return final, {
        "scale": bounded.scale,
        "distance": asdict(bounded.distance),
        "nonzero_trust_sample_count": int(len(active_samples)),
    }


def _distance_on_action(
    action: ResonanceBankSampledAction, coefficients: np.ndarray
) -> dict[str, object]:
    base = jnp.zeros(action.amplitudes.shape[1], dtype=jnp.complex128).at[0].set(1.0)
    active = np.flatnonzero(
        np.abs(np.asarray(action.amplitudes)[:, 0]) > 1.0e-14
    ).astype(np.int32)
    amplitudes = jnp.asarray(action.amplitudes)

    def center(values, indices):
        return amplitudes[indices] @ values

    return asdict(
        sampled_physical_state_distance(
            base,
            jnp.asarray(coefficients),
            jnp.asarray(active),
            center_fn=center,
        )
    )


def _fit_and_audit(
    cells,
    group,
    bonds,
    *,
    spec: dict[str, object],
    development_seed: int,
    scramble_seed: int | None,
    active: tuple[int, ...],
) -> dict[str, object]:
    controls = spec["controls"]
    sample_count = int(controls["sample_count_per_seed"])
    development = _sample_action(
        cells,
        group,
        bonds,
        seed=development_seed,
        sample_count=sample_count,
        scramble_seed=scramble_seed,
    )
    raw, fit = _ritz_coefficients(
        development, active, float(spec["relative_overlap_cutoff"])
    )
    coefficients, trust = _bounded_coefficients(development, raw, spec)
    base = np.zeros_like(coefficients)
    base[0] = 1.0
    development_initial = resonance_bank_energy_variance(development, base)
    development_final = resonance_bank_energy_variance(development, coefficients)
    rows = []
    for seed in controls["untouched_audit_seeds"]:
        audit = _sample_action(
            cells,
            group,
            bonds,
            seed=int(seed),
            sample_count=sample_count,
            scramble_seed=scramble_seed,
        )
        initial = resonance_bank_energy_variance(audit, base)
        final = resonance_bank_energy_variance(audit, coefficients)
        rows.append(
            {
                "seed": int(seed),
                "initial": initial,
                "final": final,
                "paired_energy_change_per_site": float(
                    final["energy_per_site"] - initial["energy_per_site"]
                ),
                "paired_variance_change_per_site": float(
                    final["variance_per_site"] - initial["variance_per_site"]
                ),
                "physical_distance": _distance_on_action(audit, coefficients),
            }
        )
    energy = [row["paired_energy_change_per_site"] for row in rows]
    variance = [row["paired_variance_change_per_site"] for row in rows]
    summary = {
        "paired_energy_change_per_site_mean": float(np.mean(energy)),
        "paired_energy_change_per_site_standard_error": _standard_error(energy),
        "paired_variance_change_per_site_mean": float(np.mean(variance)),
        "paired_variance_change_per_site_standard_error": _standard_error(variance),
        "maximum_fubini_study_distance": float(
            max(row["physical_distance"]["fubini_study_distance"] for row in rows)
        ),
        "maximum_symmetric_kl": float(
            max(row["physical_distance"]["symmetric_kl"] for row in rows)
        ),
    }
    summary["energy_decrease_z_score"] = float(
        -summary["paired_energy_change_per_site_mean"]
        / max(
            summary["paired_energy_change_per_site_standard_error"],
            np.finfo(float).tiny,
        )
    )
    return {
        "scramble_seed": scramble_seed,
        "fit": fit,
        "bounded_coefficients_real": np.real_if_close(coefficients).real.tolist(),
        "maximum_coefficient_imaginary_part": float(
            np.max(np.abs(coefficients.imag), initial=0.0)
        ),
        "development_trust": trust,
        "development_initial": development_initial,
        "development_final": development_final,
        "audit_rows": rows,
        "audit_summary": summary,
    }


def _run_case(case_id: str, spec: dict[str, object]) -> dict[str, object]:
    cells, group, bonds = _case_data(case_id)
    development_seed = int(spec["controls"]["development_seeds"][case_id])
    n_cells = len(cells)
    pair = _fit_and_audit(
        cells,
        group,
        bonds,
        spec=spec,
        development_seed=development_seed,
        scramble_seed=None,
        active=(0, 2),
    )
    physical = _fit_and_audit(
        cells,
        group,
        bonds,
        spec=spec,
        development_seed=development_seed,
        scramble_seed=None,
        active=tuple(range(n_cells + 1)),
    )
    controls = [
        _fit_and_audit(
            cells,
            group,
            bonds,
            spec=spec,
            development_seed=development_seed,
            scramble_seed=int(seed),
            active=tuple(range(n_cells + 1)),
        )
        for seed in spec["controls"]["scramble_seeds"]
    ]
    p = physical["audit_summary"]
    pair_summary = pair["audit_summary"]
    control_changes = [
        row["audit_summary"]["paired_energy_change_per_site_mean"]
        for row in controls
    ]
    trust = spec["physical_trust"]
    gates = spec["gates"]
    passed = bool(
        p["paired_energy_change_per_site_mean"] < 0.0
        and p["energy_decrease_z_score"] >= float(gates["minimum_energy_z_score"])
        and p["paired_variance_change_per_site_mean"]
        <= float(gates["maximum_variance_z_score"])
        * max(
            p["paired_variance_change_per_site_standard_error"],
            np.finfo(float).tiny,
        )
        and p["paired_energy_change_per_site_mean"]
        < pair_summary["paired_energy_change_per_site_mean"]
        - float(gates["minimum_incremental_energy_gain_per_site"])
        and p["paired_energy_change_per_site_mean"]
        < float(np.median(control_changes))
        - float(gates["minimum_scramble_advantage_per_site"])
        and p["maximum_fubini_study_distance"]
        <= float(trust["maximum_fubini_study_distance"])
        and p["maximum_symmetric_kl"] <= float(trust["maximum_symmetric_kl"])
    )
    return {
        "case_id": case_id,
        "n_sites": sum(len(cell) for cell in cells),
        "cell_count": n_cells,
        "channel_count": n_cells + 1,
        "development_seed": development_seed,
        "pair_b1_control": pair,
        "physical_multibody_bank": physical,
        "scrambled_multibody_banks": controls,
        "median_scrambled_energy_change_per_site": float(np.median(control_changes)),
        "passed": passed,
    }


def run_spin_n36_multibody_frame_resonance(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the physical, restricted, and geometry-scrambled N=36 gates."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    source_path = Path(str(spec["source_n36_support_gate"]))
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    source = json.loads(source_path.read_text())
    if (
        source.get("task") != "spin_n36_coherent_frame_support"
        or source.get("all_support_gates_passed") is not True
    ):
        raise RuntimeError("the sealed coherent-frame support gate did not pass")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    cases = [
        _run_case(case_id, spec)
        for case_id in ("square_6x6_j2_0.5", "triangular_6x6_j2_0.125")
    ]
    passed = bool(all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_n36_support_result_sha256": source["result_sha256"],
        "ansatz": {
            "columns": "full k=0..9 local-B1 singlet-number bank",
            "spin_sector": "exact S=0 term by term",
            "space_sector": "full ordered-frame Gamma-A1 average term by term",
            "magnetic_indices_learned": False,
            "fitted_object": "one cross-fitted coefficient direction",
        },
        "cases": cases,
        "all_multibody_gates_passed": passed,
        "remote_pilot_reapplication_allowed": passed,
        "remote_production_allowed": False,
        "decision": (
            "implement_shared_multiframe_schur_vit_and_reapply_for_bounded_remote_pilot"
            if passed
            else "freeze_coherent_local_singlet_resonance_route"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n36_multibody_frame_resonance"]
