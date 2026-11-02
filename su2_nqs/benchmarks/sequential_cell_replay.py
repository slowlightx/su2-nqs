"""N16 replay and neural-capacity gate for cell-sequential Schur coordinates."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.sequential_cell_schur_vit import (
    SequentialCellSchurViTConfig,
    init_sequential_cell_schur_vit,
    sequential_cell_schur_log_amplitudes,
    sequential_cell_schur_vit_parameter_count,
)
from ..models.carrier_anchored_schur import carrier_anchored_additive_state
from ..su2.many_cell_schur import (
    rectangular_four_spin_layout,
    rectangular_sequential_four_spin_layout,
)
from ..su2.tree_rotations import recoupling_matrix_sparse
from ..vmc.adam import train_exact_adam
from .four_cell_schur import four_cell_schur_basis_matrix
from .many_cell_replay import (
    TRIANGULAR_N16_SUPPORTS,
    balanced_tree_to_rotated_coefficients,
    rotated_to_balanced_tree_coefficients,
)


def _normalize(vector: Sequence[complex]) -> np.ndarray:
    result = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have finite nonzero norm")
    return result / norm


def _matrix_free_action(workload, coefficients: np.ndarray) -> np.ndarray:
    positions = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(workload.center_labels)
    }
    action = np.zeros_like(coefficients, dtype=np.complex128)
    for old, coefficient in enumerate(coefficients):
        for slot in range(int(workload.widths[old])):
            new = tuple(int(value) for value in workload.connected_labels[old, slot])
            action[positions[new]] += workload.coefficients[old, slot] * coefficient
    return action


def balanced_to_sequential_coefficients(
    balanced_coefficients: Sequence[complex],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Apply only exact F-moves between balanced and sequential cell trees."""

    source_layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    target_layout = rectangular_sequential_four_spin_layout(
        4, 4, triangular_coordinates=True
    )
    source = _normalize(balanced_coefficients)
    source_states = source_layout.tree.enumerate_states(0)
    target_states = target_layout.tree.enumerate_states(0)
    if source.shape != (len(source_states),) or len(target_states) != len(source_states):
        raise ValueError("balanced coefficients do not span the N16 singlet sector")
    rotation, moves = recoupling_matrix_sparse(
        source_layout.tree, target_layout.tree, target_q=0
    )
    target = np.asarray(rotation @ source, dtype=np.complex128)
    recovered = np.asarray(rotation.T @ target, dtype=np.complex128)
    identity = rotation.T @ rotation
    labels = np.asarray([state.labels for state in target_states], dtype=np.int32)
    return target, labels, {
        "singlet_dimension": len(target_states),
        "f_move_count": len(moves),
        "rotation_orthogonality_max_abs_error": float(
            np.max(np.abs(identity.toarray() - np.eye(len(source))), initial=0.0)
        ),
        "coefficient_roundtrip_max_abs_error": float(
            np.max(np.abs(recovered - source), initial=0.0)
        ),
        "coefficient_norm": float(np.vdot(target, target).real),
    }


def sequential_to_rotated_coefficients(
    sequential_coefficients: Sequence[complex],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Invert the N16 cell recoupling into the D4-labelled Schur basis."""

    balanced_layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=True
    )
    sequential_layout = rectangular_sequential_four_spin_layout(
        4, 4, triangular_coordinates=True
    )
    source = _normalize(sequential_coefficients)
    balanced_states = balanced_layout.tree.enumerate_states(0)
    sequential_states = sequential_layout.tree.enumerate_states(0)
    if source.shape != (len(sequential_states),):
        raise ValueError("sequential coefficients do not span the N16 singlet sector")
    rotation, moves = recoupling_matrix_sparse(
        balanced_layout.tree, sequential_layout.tree, target_q=0
    )
    balanced = np.asarray(rotation.T @ source, dtype=np.complex128)
    rotated = balanced_tree_to_rotated_coefficients(balanced, balanced_layout)
    recovered_balanced, _ = rotated_to_balanced_tree_coefficients(
        rotated, balanced_layout
    )
    recovered = np.asarray(rotation @ recovered_balanced, dtype=np.complex128)
    return rotated, {
        "singlet_dimension": len(sequential_states),
        "f_move_count": len(moves),
        "coefficient_roundtrip_max_abs_error": float(
            np.max(np.abs(recovered - source), initial=0.0)
        ),
        "coefficient_norm": float(np.vdot(rotated, rotated).real),
    }


def a1_reduced_to_sequential_coefficients(
    reduced_coefficients: Sequence[complex],
    coefficient_a1_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Embed Gamma-A1 coefficients and recouple them to sequential labels."""

    basis = np.asarray(coefficient_a1_basis, dtype=np.complex128)
    reduced = _normalize(reduced_coefficients)
    if basis.ndim != 2 or basis.shape[1] != len(reduced):
        raise ValueError("coefficient_a1_basis and reduced coefficients disagree")
    gram_error = float(
        np.max(
            np.abs(basis.conj().T @ basis - np.eye(basis.shape[1])), initial=0.0
        )
    )
    if gram_error > 1.0e-10:
        raise ValueError("coefficient_a1_basis must have orthonormal columns")
    balanced_layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=True
    )
    rotated = np.asarray(basis @ reduced, dtype=np.complex128)
    balanced, _ = rotated_to_balanced_tree_coefficients(rotated, balanced_layout)
    sequential, labels, diagnostics = balanced_to_sequential_coefficients(balanced)
    return sequential, labels, {
        **diagnostics,
        "a1_dimension": int(basis.shape[1]),
        "a1_basis_orthonormality_max_abs_error": gram_error,
    }


def computational_to_sequential_coefficients(
    amplitudes: Sequence[complex],
    configurations: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Project one complete N16 fixed-M vector into sequential singlet labels."""

    basis, basis_configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_SUPPORTS
    )
    supplied = tuple(tuple(int(bit) for bit in row) for row in configurations)
    positions = {state: index for index, state in enumerate(supplied)}
    if len(positions) != len(supplied):
        raise ValueError("configurations must not contain duplicates")
    try:
        order = np.fromiter(
            (positions[state] for state in basis_configurations),
            dtype=np.int64,
            count=len(basis_configurations),
        )
    except KeyError as error:
        raise ValueError("configurations do not span the required fixed-M basis") from error
    physical = _normalize(amplitudes)
    if physical.shape != (len(supplied),):
        raise ValueError("amplitudes and configurations disagree")
    ordered = physical[order]
    rotated = np.asarray(basis.T.conj() @ ordered, dtype=np.complex128)
    balanced_layout = rectangular_four_spin_layout(4, 4)
    balanced, _ = rotated_to_balanced_tree_coefficients(rotated, balanced_layout)
    sequential, labels, diagnostics = balanced_to_sequential_coefficients(balanced)
    reconstructed = np.asarray(basis @ rotated, dtype=np.complex128)
    diagnostics = {
        **diagnostics,
        "input_fixed_m_norm": float(np.vdot(ordered, ordered).real),
        "singlet_projection_norm": float(np.vdot(rotated, rotated).real),
        "singlet_reconstruction_relative_l2_error": float(
            np.linalg.norm(reconstructed - ordered) / np.linalg.norm(ordered)
        ),
    }
    return sequential, labels, diagnostics


def distill_sequential_cell_vit(
    labels: np.ndarray,
    target_coefficients: np.ndarray,
    workload,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    embedding_dim: int = 32,
    head_count: int = 4,
    depth: int = 2,
    mlp_ratio: int = 2,
    objective: str = "born_phase",
    phase_weight: float = 1.0,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    """Fit one size-transferable cell Spatial-ViT on exhaustive N16 support."""

    if steps < 1 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    if objective not in {"fidelity", "born_phase"}:
        raise ValueError("objective must be 'fidelity' or 'born_phase'")
    if phase_weight <= 0.0:
        raise ValueError("phase_weight must be positive")
    layout = rectangular_sequential_four_spin_layout(4, 4)
    config = SequentialCellSchurViTConfig(
        layout,
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
        mlp_ratio=int(mlp_ratio),
    )
    target = _normalize(target_coefficients)
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.shape != (len(target), 15):
        raise ValueError("N16 labels and target coefficients disagree")
    labels_jax = jnp.asarray(label_array)
    target_jax = jnp.asarray(target)
    target_probability = jnp.abs(target_jax) ** 2
    target_phase = jnp.angle(target_jax)
    initial = init_sequential_cell_schur_vit(jax.random.PRNGKey(int(seed)), config)

    def loss(params):
        logs = sequential_cell_schur_log_amplitudes(params, labels_jax, config)
        candidate = jnp.exp(logs)
        fidelity = jnp.abs(jnp.vdot(target_jax, candidate)) ** 2 / jnp.vdot(
            candidate, candidate
        ).real
        if objective == "fidelity":
            return -jnp.log(jnp.maximum(fidelity, jnp.asarray(1.0e-15)))
        # The exact Born KL gives every autoregressive conditional a direct
        # learning signal.  The second term then learns only the SU(2)-scalar
        # nodal/phase structure, avoiding destructive cancellation of the two
        # gradients early in training.
        cross_entropy = -jnp.sum(target_probability * (2.0 * jnp.real(logs)))
        phase_coherence = jnp.abs(
            jnp.sum(target_probability * jnp.exp(1.0j * (jnp.imag(logs) - target_phase)))
        ) ** 2
        return cross_entropy - float(phase_weight) * jnp.log(
            jnp.maximum(phase_coherence, jnp.asarray(1.0e-15))
        )

    trained, optimizer = train_exact_adam(
        initial,
        loss_fn=loss,
        steps=int(steps),
        learning_rate=float(learning_rate),
        record_every=max(1, int(steps) // 8),
    )
    candidate = np.exp(
        np.asarray(sequential_cell_schur_log_amplitudes(trained, labels_jax, config))
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
        "mlp_ratio": int(mlp_ratio),
        "objective": objective,
        "phase_weight": float(phase_weight),
        "parameter_count": sequential_cell_schur_vit_parameter_count(trained),
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "coefficient_max_abs_error": float(np.max(np.abs(candidate - target), initial=0.0)),
        "coefficient_relative_l2_error": float(np.linalg.norm(candidate - target)),
        "magnitude_fidelity": float(np.sum(np.abs(candidate) * np.abs(target)) ** 2),
        "phase_coherence_given_magnitudes": float(
            fidelity / max(np.sum(np.abs(candidate) * np.abs(target)) ** 2, 1.0e-300)
        ),
        "energy_total": energy,
        "variance_total": float(np.vdot(residual, residual).real),
        "optimizer": optimizer,
        "hamiltonian_calls_during_distillation": 0,
    }
    return report, candidate, trained


def distill_carrier_anchored_sequential_cell_vit(
    labels: np.ndarray,
    target_coefficients: np.ndarray,
    carrier_coefficients: np.ndarray,
    workload,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    embedding_dim: int = 32,
    head_count: int = 4,
    depth: int = 2,
    mlp_ratio: int = 2,
    initial_residual_scale: float = 0.01,
    q_conditioned_attention: bool = True,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    """Distill a target as an additive SU(2)-scalar dressing of a carrier.

    The autoregressive residual spans the complete sequential singlet basis.
    Orthogonalization only removes the carrier's redundant normalization
    direction; because the channel is additive it can alter carrier nodes.
    """

    if steps < 1 or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    if not np.isfinite(initial_residual_scale) or initial_residual_scale == 0.0:
        raise ValueError("initial_residual_scale must be finite and nonzero")
    layout = rectangular_sequential_four_spin_layout(4, 4)
    config = SequentialCellSchurViTConfig(
        layout,
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
        mlp_ratio=int(mlp_ratio),
        q_conditioned_attention=bool(q_conditioned_attention),
    )
    target = _normalize(target_coefficients)
    carrier = _normalize(carrier_coefficients)
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.shape != (len(target), 15) or carrier.shape != target.shape:
        raise ValueError("N16 labels, carrier, and target coefficients disagree")
    labels_jax = jnp.asarray(label_array)
    target_jax = jnp.asarray(target)
    carrier_jax = jnp.asarray(carrier)
    initial = {
        "model": init_sequential_cell_schur_vit(
            jax.random.PRNGKey(int(seed)), config
        ),
        "residual_scale": jnp.asarray(
            float(initial_residual_scale), dtype=jnp.float64
        ),
    }

    def state(parameters):
        residual = jnp.exp(
            sequential_cell_schur_log_amplitudes(
                parameters["model"], labels_jax, config
            )
        )
        return carrier_anchored_additive_state(
            carrier_jax, residual, parameters["residual_scale"]
        )

    def fidelity(parameters):
        candidate = state(parameters)
        return jnp.abs(jnp.vdot(target_jax, candidate)) ** 2 / jnp.vdot(
            candidate, candidate
        ).real

    initial_fidelity = float(fidelity(initial))
    trained, optimizer = train_exact_adam(
        initial,
        loss_fn=lambda parameters: -jnp.log(
            jnp.maximum(fidelity(parameters), jnp.asarray(1.0e-15))
        ),
        steps=int(steps),
        learning_rate=float(learning_rate),
        record_every=max(1, int(steps) // 8),
    )
    candidate = _normalize(np.asarray(state(trained)))
    overlap = np.vdot(target, candidate)
    if abs(overlap) > 0.0:
        candidate *= np.exp(-1.0j * np.angle(overlap))
    action = _matrix_free_action(workload, candidate)
    energy = float(np.vdot(candidate, action).real)
    residual = action - energy * candidate
    final_fidelity = float(abs(np.vdot(target, candidate)) ** 2)
    magnitude_fidelity = float(np.sum(np.abs(candidate) * np.abs(target)) ** 2)
    report: dict[str, object] = {
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "embedding_dim": int(embedding_dim),
        "head_count": int(head_count),
        "depth": int(depth),
        "mlp_ratio": int(mlp_ratio),
        "q_conditioned_attention": bool(q_conditioned_attention),
        "initial_residual_scale": float(initial_residual_scale),
        "trained_residual_scale": float(trained["residual_scale"]),
        "parameter_count": sequential_cell_schur_vit_parameter_count(trained),
        "carrier_target_fidelity": float(abs(np.vdot(target, carrier)) ** 2),
        "initial_fidelity": initial_fidelity,
        "fidelity": final_fidelity,
        "infidelity": 1.0 - final_fidelity,
        "fidelity_gain_over_carrier": final_fidelity
        - float(abs(np.vdot(target, carrier)) ** 2),
        "coefficient_max_abs_error": float(
            np.max(np.abs(candidate - target), initial=0.0)
        ),
        "coefficient_relative_l2_error": float(np.linalg.norm(candidate - target)),
        "magnitude_fidelity": magnitude_fidelity,
        "phase_coherence_given_magnitudes": float(
            final_fidelity / max(magnitude_fidelity, 1.0e-300)
        ),
        "energy_total": energy,
        "variance_total": float(np.vdot(residual, residual).real),
        "optimizer": optimizer,
        "hamiltonian_calls_during_distillation": 0,
    }
    return report, candidate, trained


__all__ = [
    "a1_reduced_to_sequential_coefficients",
    "balanced_to_sequential_coefficients",
    "computational_to_sequential_coefficients",
    "distill_carrier_anchored_sequential_cell_vit",
    "distill_sequential_cell_vit",
    "sequential_to_rotated_coefficients",
]
