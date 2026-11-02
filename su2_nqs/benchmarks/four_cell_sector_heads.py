"""Exact N=12 validation of shared SU(2)/space-group sector heads."""

from __future__ import annotations

from fractions import Fraction
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.su2_four_cell_sector_vit import (
    exhaustive_four_cell_sector_actions,
    init_su2_four_cell_sector_vit,
    su2_four_cell_sector_log_amplitudes,
    su2_four_cell_sector_vit_parameter_count,
    triangular_sector_config,
)
from ..operators.permutation import heisenberg_graph_fusion_matrix
from ..vmc.adam import train_exact_adam
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    TRIANGULAR_N12_SUPERCELL,
    TRIANGULAR_N12_SUPPORTS,
    _relative_panel_errors,
    _triangular_response_panel,
)
from .sector_spectroscopy import fixed_m_coupled_basis
from .triangular_hnf_space_group import (
    d6_character_projectors,
    momentum_projector_matrix,
    reciprocal_little_group,
)
from .triangular_monopole_branch import little_group_character_projector


jax.config.update("jax_enable_x64", True)


HEADS = ("A1", "E2", "A2", "K", "M")


def _projector_basis(projector: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (projector + projector.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    return np.asarray(vectors[:, values > 0.5], dtype=np.complex128)


def _head_data() -> dict[str, dict[str, object]]:
    bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.125
    )
    hamiltonians = {
        target_q: heisenberg_graph_fusion_matrix(
            12, bonds, target_q=target_q
        ).toarray()
        for target_q in (0, 2)
    }
    transforms = {}
    coupled = {}
    for target_q in (0, 2):
        physical, configurations = four_cell_schur_sector_basis_matrix(
            "triangular_triangle",
            target_q,
            supports=TRIANGULAR_N12_SUPPORTS,
        )
        fusion, fusion_configurations = fixed_m_coupled_basis(12, target_q)
        if configurations != fusion_configurations:
            raise RuntimeError("four-cell and left-fusion fixed-M bases disagree")
        transforms[target_q] = fusion.conj().T @ physical
        coupled[target_q] = fusion
    gamma = momentum_projector_matrix(
        TRIANGULAR_N12_SUPERCELL, (0, 0), target_q=0
    )
    point = d6_character_projectors(TRIANGULAR_N12_SUPERCELL, target_q=0)
    projectors = {
        "A1": gamma @ point["A1"],
        "E2": gamma @ point["E2"],
        "A2": gamma @ point["A2"],
    }
    for name, momentum in (
        ("K", (Fraction(1, 3), Fraction(1, 3))),
        ("M", (Fraction(1, 2), Fraction(0))),
    ):
        characters = {
            element: 1
            for element in reciprocal_little_group(
                TRIANGULAR_N12_SUPERCELL, momentum
            )
        }
        projectors[name] = little_group_character_projector(
            TRIANGULAR_N12_SUPERCELL,
            momentum,
            characters,
            target_q=2,
        )
    output = {}
    for sector_index, name in enumerate(HEADS):
        target_q = 0 if name in {"A1", "E2", "A2"} else 2
        projector = np.asarray(projectors[name], dtype=np.complex128)
        basis = _projector_basis(projector)
        reduced_hamiltonian = basis.conj().T @ hamiltonians[target_q] @ basis
        values, vectors = np.linalg.eigh(
            0.5 * (reduced_hamiltonian + reduced_hamiltonian.conj().T)
        )
        exact = basis @ vectors[:, 0]
        exact /= np.linalg.norm(exact)
        ground_selection = np.abs(values - values[0]) < 1.0e-10
        ground_basis = basis @ vectors[:, ground_selection]
        config = triangular_sector_config(
            target_q,
            sector_index,
            embedding_dim=12,
            head_count=3,
            depth=1,
        )
        output[name] = {
            "target_q": target_q,
            "config": config,
            "actions": exhaustive_four_cell_sector_actions(config),
            "transform": jnp.asarray(transforms[target_q]),
            "projector": jnp.asarray(projector),
            "hamiltonian": jnp.asarray(hamiltonians[target_q]),
            "coupled": coupled[target_q],
            "rank": int(basis.shape[1]),
            "exact_energy_total": float(values[0]),
            "next_energy_total": float(values[1]) if len(values) > 1 else None,
            "exact_state": exact,
            "exact_ground_basis": ground_basis,
            "projector_idempotency_error": float(
                np.linalg.norm(projector @ projector - projector)
            ),
            "local_to_fusion_isometry_error": float(
                np.linalg.norm(
                    transforms[target_q].conj().T @ transforms[target_q]
                    - np.eye(transforms[target_q].shape[1])
                )
            ),
        }
    return output


def _state(params: object, row: dict[str, object]) -> jax.Array:
    coefficients = jnp.exp(
        su2_four_cell_sector_log_amplitudes(
            params, row["actions"], row["config"]
        )
    )
    fusion = row["projector"] @ (row["transform"] @ coefficients)
    return fusion / jnp.linalg.norm(fusion)


def _metrics(state: np.ndarray, row: dict[str, object]) -> dict[str, float]:
    vector = np.array(state, dtype=np.complex128, copy=True)
    vector /= np.linalg.norm(vector)
    hamiltonian = np.asarray(row["hamiltonian"])
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    exact_basis = np.asarray(row["exact_ground_basis"])
    return {
        "energy_total": energy,
        "energy_error_total": energy - float(row["exact_energy_total"]),
        "variance_total": float(np.vdot(residual, residual).real),
        "ground_subspace_fidelity": float(
            np.linalg.norm(exact_basis.conj().T @ vector) ** 2
        ),
        "projector_closure": float(
            np.linalg.norm(np.asarray(row["projector"]) @ vector - vector)
        ),
    }


def shared_four_cell_sector_head_gate(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 1500,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Jointly train five strict heads on one shared reduced backbone."""

    data = _head_data()
    initial_config = data["A1"]["config"]

    def loss(params):
        energies = []
        for name in HEADS:
            row = data[name]
            vector = _state(params, row)
            energies.append(
                jnp.vdot(vector, row["hamiltonian"] @ vector).real
            )
        return jnp.mean(jnp.stack(energies))

    runs = []
    best_params = None
    best_loss = np.inf
    for seed in seeds:
        params = init_su2_four_cell_sector_vit(
            jax.random.PRNGKey(seed), initial_config
        )
        initial = float(loss(params))
        started = time.perf_counter()
        trained, optimizer = train_exact_adam(
            params,
            loss_fn=loss,
            steps=steps,
            learning_rate=learning_rate,
            record_every=max(1, steps // 5),
        )
        final = float(loss(trained))
        runs.append(
            {
                "seed": int(seed),
                "initial_mean_energy_total": initial,
                "final_mean_energy_total": final,
                "wall_seconds": time.perf_counter() - started,
                "optimizer": optimizer,
            }
        )
        if final < best_loss:
            best_loss = final
            best_params = trained
    assert best_params is not None
    direction = jax.tree_util.tree_map(jnp.ones_like, best_params)
    reports = {}
    for name in HEADS:
        row = data[name]
        state, tangent = jax.jvp(
            lambda params: _state(params, row),
            (best_params,),
            (direction,),
        )
        state_np = np.asarray(state)
        tangent_np = np.asarray(tangent)
        report = {
            "target_q": int(row["target_q"]),
            "sector_rank": int(row["rank"]),
            "exact_energy_total": float(row["exact_energy_total"]),
            "next_energy_total": row["next_energy_total"],
            "projector_idempotency_error": row["projector_idempotency_error"],
            "local_to_fusion_isometry_error": row[
                "local_to_fusion_isometry_error"
            ],
            **_metrics(state_np, row),
            "tangent_projector_closure": float(
                np.linalg.norm(
                    np.asarray(row["projector"]) @ tangent_np - tangent_np
                )
                / np.linalg.norm(tangent_np)
            ),
            "tangent_norm": float(np.linalg.norm(tangent_np)),
        }
        reports[name] = report
    a1_fusion = np.asarray(_state(best_params, data["A1"]))
    a1_physical = np.asarray(data["A1"]["coupled"] @ a1_fusion)
    exact_fusion = np.asarray(data["A1"]["exact_state"])
    exact_physical = np.asarray(data["A1"]["coupled"] @ exact_fusion)
    response = _triangular_response_panel(a1_physical)
    exact_response = _triangular_response_panel(exact_physical)
    response_errors = _relative_panel_errors(response, exact_response)
    exact_gates = {
        "all_projectors_idempotent": max(
            float(reports[name]["projector_idempotency_error"])
            for name in HEADS
        )
        < 3.0e-13,
        "all_local_bases_isometric": max(
            float(reports[name]["local_to_fusion_isometry_error"])
            for name in HEADS
        )
        < 5.0e-13,
        "all_state_and_tangent_sectors_closed": max(
            max(
                float(reports[name]["projector_closure"]),
                float(reports[name]["tangent_projector_closure"]),
            )
            for name in HEADS
        )
        < 5.0e-12,
    }
    trainability = {
        "all_energy_errors_below_1e-6": max(
            abs(float(reports[name]["energy_error_total"])) for name in HEADS
        )
        < 1.0e-6,
        "all_variances_below_1e-5": max(
            float(reports[name]["variance_total"]) for name in HEADS
        )
        < 1.0e-5,
        "a1_response_median_below_1e-4": float(
            np.median(tuple(response_errors.values()))
        )
        < 1.0e-4,
    }
    return {
        "system": "triangular_N12_full_D6_J2_0.125",
        "shared_parameter_count": su2_four_cell_sector_vit_parameter_count(
            best_params
        ),
        "runs": runs,
        "heads": reports,
        "A1_response": response,
        "A1_exact_response": exact_response,
        "A1_response_relative_errors": response_errors,
        "A1_median_response_relative_error": float(
            np.median(tuple(response_errors.values()))
        ),
        "exact_gates": exact_gates,
        "exact_gates_passed": bool(all(exact_gates.values())),
        "trainability_gates": trainability,
        "trainability_gates_passed": bool(all(trainability.values())),
        "claim_boundary": (
            "Exact-small joint trainability of five strict sector heads. The "
            "sector dimensions are tiny after projection, so this is not an "
            "energy-efficiency or scaling claim."
        ),
    }


__all__ = ["HEADS", "shared_four_cell_sector_head_gate"]
