"""Exact 4x4 replay gate for the scalable many-cell Schur representation."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_vit_parameter_count,
)
from ..su2.local_schur_fusion import four_cell_schur_states
from ..su2.many_cell_schur import ManyCellSchurLayout, rectangular_four_spin_layout
from ..vmc.many_cell_tree import (
    build_many_cell_tree_connectivity,
    evaluate_external_carrier_additive_local_energy,
)
from ..vmc.adam import train_exact_adam
from .four_cell_schur import four_cell_schur_basis_matrix


TRIANGULAR_N16_SUPERCELL = ((4, 0), (0, 4))
TRIANGULAR_N16_SUPPORTS = (
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (8, 9, 12, 13),
    (10, 11, 14, 15),
)
_Q0_ROTATION = np.asarray(
    ((np.sqrt(3.0) / 2.0, 0.5), (0.5, -np.sqrt(3.0) / 2.0)),
    dtype=np.float64,
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have finite nonzero norm")
    return value / norm


def _four_cell_key_from_tree(
    layout: ManyCellSchurLayout,
    labels: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    decoded = layout.decode(labels)
    if layout.n_cells != 4 or decoded.intercell_q.shape != (3,):
        raise ValueError("the sealed replay requires exactly four cells")
    left_q, right_q, root_q = map(int, decoded.intercell_q)
    if root_q != 0 or left_q != right_q:
        raise ValueError("tree state is not a four-cell singlet")
    return (
        tuple(int(value) for value in decoded.cell_q),
        tuple(int(value) for value in decoded.cell_a),
        left_q,
    )


def _local_rotation_options(q: int, unrotated_a: int) -> tuple[tuple[int, float], ...]:
    if q == 0:
        return tuple(
            (rotated_a, float(_Q0_ROTATION[unrotated_a, rotated_a]))
            for rotated_a in range(2)
        )
    return ((unrotated_a, 1.0),)


def rotated_to_balanced_tree_coefficients(
    rotated: Sequence[complex],
    layout: ManyCellSchurLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local D4-labelled coefficients to a physical balanced tree.

    The only nontrivial change is the documented two-by-two q=0 plaquette
    rotation in :mod:`su2_nqs.su2.local_schur`; q=2 and q=4 multiplicities and
    the inter-cell association are already identical.
    """

    source_states = four_cell_schur_states("square_plaquette")
    source = np.asarray(rotated, dtype=np.complex128)
    if source.shape != (len(source_states),):
        raise ValueError(f"expected {len(source_states)} rotated coefficients")
    source_index = {
        (state.cell_q, state.cell_a, state.pair_q): index
        for index, state in enumerate(source_states)
    }
    tree_states = layout.tree.enumerate_states(0)
    coefficients = np.zeros(len(tree_states), dtype=np.complex128)
    labels = np.asarray([state.labels for state in tree_states], dtype=np.int32)
    for index, state in enumerate(tree_states):
        cell_q, unrotated_a, pair_q = _four_cell_key_from_tree(layout, state.labels)
        options = [
            _local_rotation_options(q, a)
            for q, a in zip(cell_q, unrotated_a, strict=True)
        ]
        value = 0.0 + 0.0j
        for choice in product(*options):
            rotated_a = tuple(item[0] for item in choice)
            weight = float(np.prod([item[1] for item in choice]))
            value += weight * source[source_index[(cell_q, rotated_a, pair_q)]]
        coefficients[index] = value
    return coefficients, labels


def balanced_tree_to_rotated_coefficients(
    balanced: Sequence[complex],
    layout: ManyCellSchurLayout,
) -> np.ndarray:
    """Inverse of :func:`rotated_to_balanced_tree_coefficients`."""

    tree_states = layout.tree.enumerate_states(0)
    source = np.asarray(balanced, dtype=np.complex128)
    if source.shape != (len(tree_states),):
        raise ValueError(f"expected {len(tree_states)} balanced coefficients")
    balanced_index = {
        _four_cell_key_from_tree(layout, state.labels): index
        for index, state in enumerate(tree_states)
    }
    rotated_states = four_cell_schur_states("square_plaquette")
    output = np.zeros(len(rotated_states), dtype=np.complex128)
    for index, state in enumerate(rotated_states):
        options = []
        for q, rotated_a in zip(state.cell_q, state.cell_a, strict=True):
            if q == 0:
                options.append(
                    tuple(
                        (unrotated_a, float(_Q0_ROTATION[unrotated_a, rotated_a]))
                        for unrotated_a in range(2)
                    )
                )
            else:
                options.append(((rotated_a, 1.0),))
        value = 0.0 + 0.0j
        for choice in product(*options):
            unrotated_a = tuple(item[0] for item in choice)
            weight = float(np.prod([item[1] for item in choice]))
            key = (state.cell_q, unrotated_a, state.pair_q)
            value += weight * source[balanced_index[key]]
        output[index] = value
    return output


def _matrix_free_action(workload, coefficients: np.ndarray) -> np.ndarray:
    positions = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(workload.center_labels)
    }
    action = np.zeros_like(coefficients, dtype=np.complex128)
    for old in range(len(coefficients)):
        for slot in range(int(workload.widths[old])):
            new = tuple(int(value) for value in workload.connected_labels[old, slot])
            action[positions[new]] += workload.coefficients[old, slot] * coefficients[old]
    return action


def distill_many_cell_vit(
    labels: np.ndarray,
    target_coefficients: np.ndarray,
    workload,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    embedding_dim: int = 16,
    head_count: int = 4,
    depth: int = 2,
) -> tuple[dict[str, object], np.ndarray]:
    """Fit the transferable ViT to a sealed coefficient vector without ``H``.

    The fidelity gate is deliberately distinct from the exact basis replay.
    Passing the latter proves the representation and local operator; only this
    distillation audit tests whether the shared neural parameterization can
    carry the state to a larger lattice.
    """

    if steps < 1 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    config = ManyCellSchurViTConfig(
        layout,
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
    )
    target = _normalize(np.asarray(target_coefficients, dtype=np.complex128))
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.shape != (len(target), 15):
        raise ValueError("N16 labels and target coefficients disagree")
    labels_jax = jnp.asarray(label_array)
    target_jax = jnp.asarray(target)
    initial = init_many_cell_schur_vit(jax.random.PRNGKey(int(seed)), config)

    def loss(params):
        candidate = jnp.exp(
            many_cell_schur_log_amplitudes(params, labels_jax, config)
        )
        overlap = jnp.vdot(target_jax, candidate)
        fidelity = jnp.abs(overlap) ** 2 / jnp.vdot(
            candidate, candidate
        ).real
        return -jnp.log(
            jnp.maximum(fidelity, jnp.asarray(1.0e-15))
        )

    trained, optimizer = train_exact_adam(
        initial,
        loss_fn=loss,
        steps=int(steps),
        learning_rate=float(learning_rate),
        record_every=max(1, int(steps) // 6),
    )
    candidate = np.exp(
        np.asarray(
            many_cell_schur_log_amplitudes(trained, labels_jax, config)
        )
    )
    overlap = np.vdot(target, candidate)
    if abs(overlap) > 0.0:
        candidate *= np.exp(-1.0j * np.angle(overlap))
    candidate = _normalize(candidate)
    action = _matrix_free_action(workload, candidate)
    energy = float(np.vdot(candidate, action).real)
    residual = action - energy * candidate
    fidelity = float(abs(np.vdot(target, candidate)) ** 2)
    report: dict[str, object] = {
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "embedding_dim": int(embedding_dim),
        "head_count": int(head_count),
        "depth": int(depth),
        "parameter_count": many_cell_schur_vit_parameter_count(trained),
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "coefficient_max_abs_error": float(
            np.max(np.abs(candidate - target), initial=0.0)
        ),
        "coefficient_relative_l2_error": float(np.linalg.norm(candidate - target)),
        "magnitude_fidelity": float(np.sum(np.abs(candidate) * np.abs(target)) ** 2),
        "energy_total": energy,
        "variance_total": float(np.vdot(residual, residual).real),
        "optimizer": optimizer,
        "hamiltonian_calls_during_distillation": 0,
    }
    return report, candidate


def replay_sealed_triangular_n16_state(
    sealed_state: Sequence[complex],
    *,
    j2_over_j1: float = 0.125,
    sealed_energy: float | None = None,
    maximum_width: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Replay every sealed physical amplitude and audit matrix-free ``H``.

    This function performs no diagonalization.  The sealed state is merely
    changed from the complete local-D4 Schur basis to an equally complete
    physical fusion-tree basis.  Hamiltonian moments are then evaluated from
    local recoupling columns.
    """

    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    if layout.physical_cells != TRIANGULAR_N16_SUPPORTS:
        raise RuntimeError("the N16 cell partition changed")
    physical = _normalize(np.asarray(sealed_state, dtype=np.complex128))
    basis, configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_SUPPORTS
    )
    if physical.shape != (len(configurations),):
        raise ValueError("sealed state does not use the expected fixed-M basis")
    rotated = np.asarray(basis.T.conj() @ physical)
    balanced, labels = rotated_to_balanced_tree_coefficients(rotated, layout)
    reconstructed_rotated = balanced_tree_to_rotated_coefficients(balanced, layout)
    reconstructed = np.asarray(basis @ reconstructed_rotated)
    overlap = np.vdot(physical, reconstructed)
    if abs(overlap) > 0.0:
        reconstructed *= np.exp(-1.0j * np.angle(overlap))
        balanced *= np.exp(-1.0j * np.angle(overlap))
    difference = reconstructed - physical

    bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N16_SUPERCELL, j1=1.0, j2=float(j2_over_j1)
    )
    config = ManyCellSchurViTConfig(
        layout, embedding_dim=8, head_count=2, depth=1
    )
    workload = build_many_cell_tree_connectivity(
        labels, config, bonds, maximum_width=maximum_width
    )
    action = _matrix_free_action(workload, balanced)
    norm = float(np.vdot(balanced, balanced).real)
    energy = float(np.vdot(balanced, action).real / norm)
    residual = action - energy * balanced
    variance = float(np.vdot(residual, residual).real / norm)

    threshold = 1.0e-13 * float(np.max(np.abs(balanced)))
    active = np.flatnonzero(np.abs(balanced) > threshold)
    active_workload = build_many_cell_tree_connectivity(
        labels[active],
        config,
        bonds,
        maximum_width=workload.maximum_width,
    )
    lookup = {
        tuple(int(value) for value in row): balanced[index]
        for index, row in enumerate(labels)
    }
    carrier_connected = np.zeros(active_workload.mask.shape, dtype=np.complex128)
    for row in range(len(active)):
        for slot in range(int(active_workload.widths[row])):
            key = tuple(
                int(value) for value in active_workload.connected_labels[row, slot]
            )
            carrier_connected[row, slot] = lookup[key]
    residual_params = init_many_cell_schur_vit(jax.random.PRNGKey(0), config)
    local = evaluate_external_carrier_additive_local_energy(
        balanced[active],
        carrier_connected,
        residual_params,
        0.0,
        active_workload,
        config,
    )
    active_weights = np.abs(balanced[active]) ** 2 / norm
    ratio_energy = float(np.sum(active_weights * local.local_energies).real)

    metrics: dict[str, object] = {
        "n_sites": 16,
        "n_cells": 4,
        "singlet_dimension": len(labels),
        "physical_fixed_m_dimension": len(physical),
        "physical_amplitude_max_abs_error": float(
            np.max(np.abs(difference), initial=0.0)
        ),
        "physical_amplitude_relative_l2_error": float(
            np.linalg.norm(difference) / np.linalg.norm(physical)
        ),
        "physical_fidelity": float(
            abs(np.vdot(physical, reconstructed)) ** 2
            / (np.vdot(reconstructed, reconstructed).real * np.vdot(physical, physical).real)
        ),
        "balanced_coefficient_norm": norm,
        "matrix_free_energy_total": energy,
        "matrix_free_variance_total": variance,
        "ratio_energy_total": ratio_energy,
        "ratio_vs_scatter_energy_abs_error": abs(ratio_energy - energy),
        "sealed_energy_total": None if sealed_energy is None else float(sealed_energy),
        "matrix_free_vs_sealed_energy_abs_error": (
            None if sealed_energy is None else abs(energy - float(sealed_energy))
        ),
        "matrix_free_connectivity_maximum_width": workload.maximum_width,
        "matrix_free_connectivity_entries": workload.total_entries,
        "matrix_free_connectivity_seconds": workload.connectivity_seconds,
        "active_ratio_state_count": len(active),
        "additive_zero_scale_replays_carrier": bool(
            abs(ratio_energy - energy) < 2.0e-11
        ),
        "repeated_ed_performed": False,
    }
    metrics["amplitude_gate_passed"] = bool(
        metrics["physical_amplitude_max_abs_error"] < 2.0e-12
        and metrics["physical_amplitude_relative_l2_error"] < 2.0e-11
        and 1.0 - metrics["physical_fidelity"] < 2.0e-12
    )
    metrics["matrix_free_gate_passed"] = bool(
        variance < 5.0e-18
        and abs(ratio_energy - energy) < 2.0e-11
        and (
            sealed_energy is None
            or abs(energy - float(sealed_energy)) < 2.0e-11
        )
    )
    arrays = {
        "labels": labels,
        "balanced_coefficients": balanced,
        "reconstructed_physical_state": reconstructed,
        "matrix_free_action": action,
    }
    return metrics, arrays


__all__ = [
    "TRIANGULAR_N16_SUPPORTS",
    "balanced_tree_to_rotated_coefficients",
    "distill_many_cell_vit",
    "replay_sealed_triangular_n16_state",
    "rotated_to_balanced_tree_coefficients",
]
