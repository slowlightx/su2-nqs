"""Complete multiplicity-space SU(2)-Spatial-ViT on triangular 4x4.

This benchmark changes the variational state manifold rather than appending a
response head.  Four complete four-spin Schur cells span all 1430 singlets.
The ViT sees only reduced ``(q,a)`` and inter-cell doubled-spin labels; a full
triangular-space-group Reynolds projection enforces ``Gamma-A1``.

The exact ground state is used only for post-training audit.  Training uses
the Hamiltonian Rayleigh quotient and either a standalone Schur state or an
additive residual around a frozen, independently trained SU(2)/A1 carrier.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.carrier_anchored_schur import carrier_anchored_additive_state
from ..models.su2_four_cell_schur_vit import (
    SU2FourCellSchurViTConfig,
    exhaustive_four_cell_actions,
    init_su2_four_cell_schur_vit,
    su2_four_cell_schur_log_amplitudes,
    su2_four_cell_schur_vit_parameter_count,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    coo_matvec,
    graph_hamiltonian_coo,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .four_cell_schur import four_cell_schur_basis_matrix
from .fusion_spatial_bridge import spin_raising_relative_norm


jax.config.update("jax_enable_x64", True)


TRIANGULAR_N16_SUPERCELL = ((4, 0), (0, 4))
TRIANGULAR_N16_RHOMBUS_SUPPORTS = (
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (8, 9, 12, 13),
    (10, 11, 14, 15),
)
ARM_IDS = (
    "standalone_complete_multiplicity_vit",
    "operator_anchored_complete_multiplicity_vit",
)


def triangular_rhombus_schur_config(
    *,
    embedding_dim: int = 12,
    head_count: int = 3,
    depth: int = 1,
) -> SU2FourCellSchurViTConfig:
    """Return Cartesian centers for four 2x2 rhombi on the triangular torus."""

    root_three = np.sqrt(3.0)
    # Convert oblique triangular coordinates (u,v) to
    # (x,y)=(u+v/2,sqrt(3)v/2) before spatial attention.
    return SU2FourCellSchurViTConfig(
        "square_plaquette",
        (
            (0.75, 0.25 * root_three),
            (2.75, 0.25 * root_three),
            (1.75, 1.25 * root_three),
            (3.75, 1.25 * root_three),
        ),
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-14 or not np.isfinite(norm):
        raise ValueError("state must have finite nonzero norm")
    return result / norm


def _energy_variance(state: np.ndarray, hamiltonian) -> tuple[float, float]:
    vector = _normalize(state)
    action = np.asarray(hamiltonian @ vector)
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def phase_magnitude_tomography(
    reference: np.ndarray,
    candidate: np.ndarray,
    hamiltonian,
    *,
    relative_node_threshold: float = 1.0e-14,
) -> dict[str, float | int]:
    """Separate physical-basis magnitude and phase/sign errors.

    The two oracle states exchange either magnitudes or phases with the exact
    reference.  This is a post-hoc diagnostic only and is never used by the
    optimizer.  All phase metrics are invariant to a global candidate phase.
    """

    if not 0.0 < relative_node_threshold < 1.0:
        raise ValueError("relative_node_threshold must lie in (0,1)")
    exact = _normalize(reference)
    trial = _normalize(candidate)
    overlap = np.vdot(exact, trial)
    if abs(overlap) > 0.0:
        trial = trial * np.exp(-1.0j * np.angle(overlap))
    exact_active = np.abs(exact) > relative_node_threshold * np.max(np.abs(exact))
    trial_active = np.abs(trial) > relative_node_threshold * np.max(np.abs(trial))
    exact_phase = np.ones_like(exact)
    trial_phase = np.ones_like(trial)
    exact_phase[exact_active] = exact[exact_active] / np.abs(exact[exact_active])
    trial_phase[trial_active] = trial[trial_active] / np.abs(trial[trial_active])
    exact_phase_trial_magnitude = _normalize(np.abs(trial) * exact_phase)
    trial_phase_exact_magnitude = _normalize(np.abs(exact) * trial_phase)
    exact_energy, _ = _energy_variance(exact, hamiltonian)
    energy, variance = _energy_variance(trial, hamiltonian)
    amplitude_oracle_energy, amplitude_oracle_variance = _energy_variance(
        exact_phase_trial_magnitude, hamiltonian
    )
    phase_oracle_energy, phase_oracle_variance = _energy_variance(
        trial_phase_exact_magnitude, hamiltonian
    )
    relative_phase = np.conj(exact_phase) * trial_phase
    exact_probability = np.abs(exact) ** 2
    return {
        "energy_total": energy,
        "energy_error_total": energy - exact_energy,
        "variance_total": variance,
        "ground_fidelity": float(abs(np.vdot(exact, trial)) ** 2),
        "magnitude_fidelity": float(np.sum(np.abs(exact) * np.abs(trial)) ** 2),
        "exact_weighted_phase_coherence": float(
            abs(np.sum(exact_probability * relative_phase)) ** 2
        ),
        "exact_weighted_real_sign_agreement": float(
            np.sum(exact_probability * (np.real(relative_phase) > 0.0))
        ),
        "candidate_node_count": int(np.count_nonzero(~trial_active)),
        "exact_probability_on_candidate_nodes": float(
            np.sum(exact_probability[~trial_active])
        ),
        "exact_phase_candidate_magnitude_energy_total": amplitude_oracle_energy,
        "exact_phase_candidate_magnitude_energy_error_total": (
            amplitude_oracle_energy - exact_energy
        ),
        "exact_phase_candidate_magnitude_variance_total": amplitude_oracle_variance,
        "candidate_phase_exact_magnitude_energy_total": phase_oracle_energy,
        "candidate_phase_exact_magnitude_energy_error_total": (
            phase_oracle_energy - exact_energy
        ),
        "candidate_phase_exact_magnitude_variance_total": phase_oracle_variance,
    }


def _metric_panel(
    state: np.ndarray,
    exact_state: np.ndarray,
    hamiltonian,
    configurations: Sequence[Sequence[int]],
    permutation_indices: np.ndarray,
) -> dict[str, float | int]:
    vector = _normalize(state)
    projected = np.asarray(
        reynolds_project_state(jnp.asarray(vector), jnp.asarray(permutation_indices))
    )
    panel = phase_magnitude_tomography(exact_state, vector, hamiltonian)
    panel.update(
        {
            "spin_raising_relative_norm": spin_raising_relative_norm(
                vector, configurations
            ),
            "spatial_projector_closure": float(
                np.linalg.norm(projected - vector) / np.linalg.norm(vector)
            ),
        }
    )
    return panel


def _checkpoint_loss(optimizer: Mapping[str, object], step: int) -> float:
    for row in optimizer["history"]:
        if int(row["step"]) == int(step):
            return float(row["loss_before_update"])
    raise RuntimeError(f"optimizer history does not contain step {step}")


def triangular_n16_complete_manifold_frontier(
    state_bundle: Mapping[str, np.ndarray],
    *,
    seeds: tuple[int, ...],
    steps: int,
    matched_step: int,
    learning_rate: float,
    initial_residual_scale: float,
    embedding_dim: int,
    head_count: int,
    depth: int,
    coupling: float = 0.125,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Train standalone and carrier-anchored complete N=16 Schur ViTs."""

    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct and nonnegative")
    if not 0 < matched_step < steps or matched_step % 4:
        raise ValueError("matched_step must be below steps and divisible by four")
    if learning_rate <= 0.0 or initial_residual_scale <= 0.0:
        raise ValueError("learning rate and initial residual scale must be positive")
    required = {
        "basis",
        "exact_ground",
        "optimized_pi_flux_jastrow_two_lanczos",
        "pi_flux_fixed_sz_gamma_a1_spatial_vit",
        "pi_flux_exact_su2_a1_operator_vit",
    }
    missing = sorted(required - set(state_bundle))
    if missing:
        raise ValueError(f"state bundle is missing arrays: {missing}")

    config = triangular_rhombus_schur_config(
        embedding_dim=embedding_dim, head_count=head_count, depth=depth
    )
    basis_matrix, configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_RHOMBUS_SUPPORTS
    )
    stored_basis = np.asarray(state_bundle["basis"], dtype=np.int8)
    if stored_basis.shape != (len(configurations), 16):
        raise RuntimeError("frozen exact-state bundle has a different basis shape")
    stored_positions = {
        tuple(int(bit) for bit in row): index
        for index, row in enumerate(stored_basis)
    }
    if len(stored_positions) != len(configurations) or set(stored_positions) != set(
        configurations
    ):
        raise RuntimeError("frozen and Schur fixed-M configuration sets disagree")
    source_order = np.asarray(
        [stored_positions[configuration] for configuration in configurations],
        dtype=np.int64,
    )
    ordered_bundle = {
        name: (
            np.asarray(array)[source_order]
            if np.asarray(array).ndim == 1
            and np.asarray(array).shape[0] == len(configurations)
            else np.asarray(array)
        )
        for name, array in state_bundle.items()
    }
    bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N16_SUPERCELL, j1=1.0, j2=float(coupling)
    )
    rows, columns, values, checked = graph_hamiltonian_coo(16, 0, bonds)
    if checked != configurations:
        raise RuntimeError("Schur and Hamiltonian fixed-M bases disagree")
    hamiltonian = coo_matrix(
        (values, (rows, columns)), shape=(len(configurations), len(configurations))
    ).tocsr()
    exact_state = _normalize(ordered_bundle["exact_ground"])
    anchor = _normalize(ordered_bundle["pi_flux_exact_su2_a1_operator_vit"])
    exact_energy, exact_variance = _energy_variance(exact_state, hamiltonian)
    anchor_energy, anchor_variance = _energy_variance(anchor, hamiltonian)

    group = triangular_hnf_gamma_a1_group(TRIANGULAR_N16_SUPERCELL)
    permutation_indices = physical_permutation_index_table(configurations, group)
    matrix = jnp.asarray(np.asarray(basis_matrix))
    indices = jnp.asarray(permutation_indices)
    row = jnp.asarray(rows)
    column = jnp.asarray(columns)
    elements = jnp.asarray(values)
    actions = exhaustive_four_cell_actions(config)
    anchor_jax = jnp.asarray(anchor)

    def residual_state(vit_params):
        coefficients = jnp.exp(
            su2_four_cell_schur_log_amplitudes(vit_params, actions, config)
        )
        return reynolds_project_state(matrix @ coefficients, indices)

    def standalone_state(params):
        return residual_state(params)

    def anchored_state(params):
        residual = residual_state(params["vit"])
        return carrier_anchored_additive_state(
            anchor_jax, residual, params["residual_scale"]
        )

    def loss(state_fn, params):
        vector = state_fn(params)
        action = coo_matvec(row, column, elements, vector)
        return jnp.vdot(vector, action).real / jnp.vdot(vector, vector).real

    runs: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARM_IDS}
    best: dict[str, tuple[float, np.ndarray]] = {}
    record_every = matched_step // 4
    for seed in seeds:
        initial_vit = init_su2_four_cell_schur_vit(
            jax.random.PRNGKey(seed), config
        )
        specifications = (
            (ARM_IDS[0], initial_vit, standalone_state),
            (
                ARM_IDS[1],
                {
                    "vit": initial_vit,
                    "residual_scale": jnp.asarray(
                        initial_residual_scale, dtype=jnp.float64
                    ),
                },
                anchored_state,
            ),
        )
        for arm_id, initial_params, state_fn in specifications:
            initial_energy = float(loss(state_fn, initial_params))
            started = time.perf_counter()
            trained, optimizer = train_exact_adam(
                initial_params,
                loss_fn=lambda candidate, fn=state_fn: loss(fn, candidate),
                steps=steps,
                learning_rate=learning_rate,
                record_every=record_every,
            )
            wall_seconds = time.perf_counter() - started
            state = _normalize(np.asarray(state_fn(trained)))
            metrics = _metric_panel(
                state,
                exact_state,
                hamiltonian,
                configurations,
                permutation_indices,
            )
            parameter_count = (
                su2_four_cell_schur_vit_parameter_count(trained)
                if arm_id == ARM_IDS[0]
                else su2_four_cell_schur_vit_parameter_count(trained["vit"]) + 1
            )
            report: dict[str, object] = {
                "arm_id": arm_id,
                "seed": int(seed),
                "parameter_count": parameter_count,
                "initial_energy_total": initial_energy,
                "matched_step": matched_step,
                "matched_step_energy_total": _checkpoint_loss(
                    optimizer, matched_step
                ),
                "steps": steps,
                "full_h_calls": steps + 2,
                "wall_seconds": wall_seconds,
                "optimizer": optimizer,
                "metrics": metrics,
            }
            if arm_id == ARM_IDS[1]:
                report["initial_residual_scale"] = initial_residual_scale
                report["final_residual_scale"] = float(trained["residual_scale"])
            runs[arm_id].append(report)
            previous = best.get(arm_id)
            if previous is None or metrics["energy_total"] < previous[0]:
                best[arm_id] = (float(metrics["energy_total"]), state)

    frozen_diagnostics = {
        name: phase_magnitude_tomography(
            exact_state, np.asarray(ordered_bundle[name]), hamiltonian
        )
        for name in required
        if name not in {"basis", "exact_ground"}
    }
    summary = {}
    for arm_id, reports in runs.items():
        final_energies = np.asarray(
            [row["metrics"]["energy_total"] for row in reports]
        )
        final_variances = np.asarray(
            [row["metrics"]["variance_total"] for row in reports]
        )
        matched_energies = np.asarray(
            [row["matched_step_energy_total"] for row in reports]
        )
        summary[arm_id] = {
            "median_energy_total": float(np.median(final_energies)),
            "median_energy_error_total": float(np.median(final_energies - exact_energy)),
            "best_energy_total": float(np.min(final_energies)),
            "best_energy_error_total": float(np.min(final_energies - exact_energy)),
            "median_variance_total": float(np.median(final_variances)),
            "median_matched_step_energy_total": float(np.median(matched_energies)),
            "all_seed_energy_errors_total": (final_energies - exact_energy).tolist(),
            "all_seed_matched_step_energy_errors_total": (
                matched_energies - exact_energy
            ).tolist(),
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in reports])
            ),
        }
    anchored = summary[ARM_IDS[1]]
    comparator_energy = anchor_energy
    comparator_variance = anchor_variance
    gates = {
        "frozen_anchor_has_no_candidate_nodes": frozen_diagnostics[
            "pi_flux_exact_su2_a1_operator_vit"
        ]["exact_probability_on_candidate_nodes"]
        < 1.0e-15,
        "frozen_anchor_phase_oracle_is_exact": abs(
            frozen_diagnostics["pi_flux_exact_su2_a1_operator_vit"][
                "candidate_phase_exact_magnitude_energy_error_total"
            ]
        )
        < 1.0e-12,
        "anchored_median_energy_beats_frozen_operator_vit": (
            anchored["median_energy_total"] < comparator_energy
        ),
        "anchored_median_variance_beats_frozen_operator_vit": (
            anchored["median_variance_total"] < comparator_variance
        ),
        "anchored_majority_of_seeds_beat_frozen_operator_vit": sum(
            row["metrics"]["energy_total"] < comparator_energy
            for row in runs[ARM_IDS[1]]
        )
        >= (len(seeds) // 2 + 1),
        "anchored_exact_su2_a1_closure": max(
            max(
                row["metrics"]["spin_raising_relative_norm"],
                row["metrics"]["spatial_projector_closure"],
            )
            for row in runs[ARM_IDS[1]]
        )
        < 1.0e-11,
    }
    states = {
        "standalone_best": best[ARM_IDS[0]][1],
        "operator_anchored_best": best[ARM_IDS[1]][1],
    }
    return (
        {
            "case_id": "triangular_4x4_periodic_j2_0125",
            "n_sites": 16,
            "fixed_m_basis_size": len(configurations),
            "frozen_bundle_basis_reordered": bool(
                not np.array_equal(source_order, np.arange(len(configurations)))
            ),
            "complete_singlet_dimension": basis_matrix.shape[1],
            "space_group_order": len(group),
            "local_cells": [list(support) for support in TRIANGULAR_N16_RHOMBUS_SUPPORTS],
            "hamiltonian_nnz": int(hamiltonian.nnz),
            "exact_ground": {
                "energy_total": exact_energy,
                "variance_total": exact_variance,
            },
            "frozen_operator_vit_comparator": {
                "energy_total": anchor_energy,
                "energy_error_total": anchor_energy - exact_energy,
                "variance_total": anchor_variance,
            },
            "frozen_state_bottleneck_audit": frozen_diagnostics,
            "runs": runs,
            "summary": summary,
            "promotion_gates": gates,
            "promotion_gates_passed": bool(all(gates.values())),
            "interpretation_rule": (
                "Prefer the complete configuration-dependent multiplicity manifold "
                "when the frozen carrier has exact phase/sign and zero nodal mass, "
                "and the anchored arm lowers both median energy and variance."
            ),
            "claim_boundary": (
                "Exact-support triangular 4x4 state-manifold test. It establishes "
                "small-system variational capacity and symmetry closure, not sampled "
                "6x6 scaling or a thermodynamic QSL identification."
            ),
        },
        states,
    )


__all__ = [
    "ARM_IDS",
    "TRIANGULAR_N16_RHOMBUS_SUPPORTS",
    "TRIANGULAR_N16_SUPERCELL",
    "phase_magnitude_tomography",
    "triangular_n16_complete_manifold_frontier",
    "triangular_rhombus_schur_config",
]
