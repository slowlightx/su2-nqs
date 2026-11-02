"""Coherent multi-Pfaffian resonance after single-basin backflow failure."""

from __future__ import annotations

from typing import Mapping

import numpy as np


N_SITES = 16


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state must have finite nonzero norm")
    return values / norm


def lowest_ritz_bank(columns: np.ndarray, hamiltonian: np.ndarray):
    """Return the lowest state and diagnostics in a possibly redundant bank."""

    matrix = np.asarray(columns, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("columns must be a nonempty rank-two matrix")
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1.0e-11))
    basis = left[:, :rank]
    projected = basis.conj().T @ hamiltonian @ basis
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    state = _normalize(basis @ vectors[:, 0])
    action = hamiltonian @ state
    energy = float(np.vdot(state, action).real)
    residual = action - energy * state
    variance = float(np.vdot(residual, residual).real)
    return state, {
        "input_column_count": matrix.shape[1],
        "rank": rank,
        "singular_values": singular.tolist(),
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": variance,
        "variance_per_site": variance / N_SITES,
    }


def _krylov(anchor: np.ndarray, hamiltonian: np.ndarray, dimension: int):
    columns = []
    current = _normalize(anchor)
    for _ in range(dimension):
        columns.append(current)
        current = hamiltonian @ current
    return lowest_ritz_bank(np.column_stack(columns), hamiltonian)


def resonating_neural_pfaffian_frontier(
    *,
    hamiltonian_a1: np.ndarray,
    exact_a1: np.ndarray,
    anchor_a1: np.ndarray,
    direct_states: Mapping[str, np.ndarray],
    residual_states: Mapping[str, np.ndarray],
    operator_vit_energy_per_site: float,
    minimum_resolved_energy_gain_per_site: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare physical, scrambled, and rank-matched coherent banks."""

    hamiltonian = np.asarray(hamiltonian_a1, dtype=np.complex128)
    exact = _normalize(exact_a1)
    anchor = _normalize(anchor_a1)
    required = {
        "compact_physical",
        "compact_scrambled",
        "extended_physical",
        "extended_scrambled",
    }
    if set(direct_states) != required or set(residual_states) != required:
        raise ValueError("direct and residual state banks must use registered labels")
    definitions = {
        "direct_only": (
            (anchor, direct_states["compact_physical"], direct_states["extended_physical"]),
            (anchor, direct_states["compact_scrambled"], direct_states["extended_scrambled"]),
        ),
        "residual_only": (
            (anchor, residual_states["compact_physical"], residual_states["extended_physical"]),
            (anchor, residual_states["compact_scrambled"], residual_states["extended_scrambled"]),
        ),
        "combined": (
            (
                anchor,
                direct_states["compact_physical"],
                direct_states["extended_physical"],
                residual_states["compact_physical"],
                residual_states["extended_physical"],
            ),
            (
                anchor,
                direct_states["compact_scrambled"],
                direct_states["extended_scrambled"],
                residual_states["compact_scrambled"],
                residual_states["extended_scrambled"],
            ),
        ),
    }
    rows = {}
    arrays: dict[str, np.ndarray] = {"anchor_a1": anchor, "exact_a1": exact}
    for label, (physical_columns, scrambled_columns) in definitions.items():
        physical_state, physical = lowest_ritz_bank(
            np.column_stack(physical_columns), hamiltonian
        )
        scrambled_state, scrambled = lowest_ritz_bank(
            np.column_stack(scrambled_columns), hamiltonian
        )
        krylov_state, krylov = _krylov(anchor, hamiltonian, physical["rank"])
        physical["exact_fidelity"] = float(abs(np.vdot(exact, physical_state)) ** 2)
        scrambled["exact_fidelity"] = float(abs(np.vdot(exact, scrambled_state)) ** 2)
        krylov["exact_fidelity"] = float(abs(np.vdot(exact, krylov_state)) ** 2)
        gates = {
            "physical_and_scrambled_match_rank": physical["rank"] == scrambled["rank"],
            "physical_beats_scrambled_energy_and_variance": (
                physical["energy_per_site"] < scrambled["energy_per_site"]
                and physical["variance_per_site"] < scrambled["variance_per_site"]
            ),
            "physical_beats_frozen_operator_vit_energy": (
                physical["energy_per_site"] < operator_vit_energy_per_site
            ),
            "physical_beats_rank_matched_krylov_energy_and_variance": (
                physical["energy_per_site"]
                < krylov["energy_per_site"] - minimum_resolved_energy_gain_per_site
                and physical["variance_per_site"] < krylov["variance_per_site"]
            ),
            "bank_is_nonsaturating": physical["rank"] < len(exact),
        }
        gates["variational_win"] = bool(all(gates.values()))
        rows[label] = {
            "physical": physical,
            "scrambled": scrambled,
            "same_anchor_rank_matched_krylov": krylov,
            "gates": gates,
        }
        arrays[f"{label}_physical_a1"] = physical_state
        arrays[f"{label}_scrambled_a1"] = scrambled_state
        arrays[f"{label}_krylov_a1"] = krylov_state
    win = bool(rows["combined"]["gates"]["variational_win"])
    return {
        "case": "triangular_4x4_j2_0.125_resonating_neural_pfaffian",
        "exact_s0_gamma_a1_dimension": len(exact),
        "bank_contract": (
            "coherent Ritz coefficients use energy only; no exact-state overlap "
            "enters selection or fitting"
        ),
        "frozen_operator_vit_energy_per_site": float(operator_vit_energy_per_site),
        "banks": rows,
        "decision": {
            "combined_variational_win": win,
            "triangular_n36_reopened": win,
            "branch": (
                "reopen_only_bounded_n36_multi_pfaffian_preflight"
                if win
                else "freeze_resonating_neural_pfaffian_route"
            ),
        },
        "claim_boundary": (
            "Deterministic exact-support multi-Pfaffian state-manifold test; "
            "no repeated ED, scale, efficiency, phase, or QSL claim."
        ),
    }, arrays


__all__ = ["lowest_ritz_bank", "resonating_neural_pfaffian_frontier"]
