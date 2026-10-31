"""Carrier-preserving residual neural-Pfaffian contingency.

The direct neural backflow carrier can destroy a good projected-VMC nodal
surface.  This registered contingency keeps the exact ``S=0, Gamma-A1``
projection of that carrier as an additive channel and learns only the
component of a configuration-dependent Pfaffian orthogonal to it.  Zero
residual scale reproduces the anchor exactly.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .fusion_spatial_bridge import spin_raising_relative_norm
from .neural_backflow_pfaffian import (
    N_SITES,
    NeuralBackflowData,
    build_neural_backflow_data,
    neural_backflow_reduced_state,
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise FloatingPointError("state has zero or nonfinite norm")
    return values / norm


def residual_backflow_state(
    parameters: jax.Array,
    data: NeuralBackflowData,
    anchor: jax.Array,
) -> jax.Array:
    """Return anchor plus an explicitly orthogonal neural-Pfaffian residual."""

    if parameters.ndim != 1 or len(parameters) != data.parameter_count + 1:
        raise ValueError("residual parameters have the wrong shape")
    fixed = anchor / jnp.sqrt(jnp.vdot(anchor, anchor).real)
    candidate = neural_backflow_reduced_state(parameters[:-1], data)
    orthogonal = candidate - fixed * jnp.vdot(fixed, candidate)
    scale = 2.0 * jnp.tanh(parameters[-1])
    state = fixed + scale * orthogonal
    return state / jnp.sqrt(jnp.vdot(state, state).real)


def _energy_variance(state: np.ndarray, hamiltonian: np.ndarray):
    vector = _normalize(state)
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def _train(
    data: NeuralBackflowData,
    anchor: np.ndarray,
    *,
    seed: int,
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
) -> tuple[np.ndarray, dict[str, object]]:
    anchor_jax = jnp.asarray(_normalize(anchor))
    hamiltonian = jnp.asarray(data.hamiltonian_a1)

    def state_fn(parameters):
        return residual_backflow_state(parameters, data, anchor_jax)

    def loss(parameters):
        state = state_fn(parameters)
        return jnp.vdot(state, hamiltonian @ state).real

    value_gradient = jax.jit(jax.value_and_grad(loss))
    evaluations = 0

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        value, gradient = value_gradient(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=np.float64)

    initial = np.random.default_rng(int(seed)).normal(
        0.0, initial_scale, size=data.parameter_count + 1
    )
    initial[-1] += 0.08
    started = time.perf_counter()
    optimized = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=tuple(
            (-parameter_bound, parameter_bound) for _ in range(len(initial))
        ),
        options={"maxiter": maximum_iterations, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
    )
    elapsed = time.perf_counter() - started
    parameters = np.asarray(optimized.x, dtype=np.float64)
    state = np.asarray(state_fn(jnp.asarray(parameters)))
    energy, variance = _energy_variance(state, data.hamiltonian_a1)
    def real_state_fn(values):
        observed = state_fn(values)
        return jnp.concatenate((observed.real, observed.imag))

    tangent = np.asarray(jax.jacrev(real_state_fn)(jnp.asarray(parameters)))
    real_state = np.concatenate((state.real, state.imag))
    span = np.column_stack((real_state, tangent))
    singular = np.linalg.svd(span, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1.0e-9))
    return state, {
        "seed": int(seed),
        "parameters": parameters.tolist(),
        "parameter_count": len(parameters),
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": variance,
        "variance_per_site": variance / N_SITES,
        "residual_scale": float(2.0 * np.tanh(parameters[-1])),
        "state_plus_tangent_rank": rank,
        "state_plus_tangent_singular_values": singular.tolist(),
        "objective_evaluations": evaluations,
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "wall_seconds": elapsed,
    }


def _krylov(seed: np.ndarray, hamiltonian: np.ndarray, dimension: int):
    columns = []
    current = _normalize(seed)
    for _ in range(dimension):
        columns.append(current)
        current = hamiltonian @ current
    matrix = np.column_stack(columns)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1.0e-12))
    basis = left[:, :rank]
    projected = basis.conj().T @ hamiltonian @ basis
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    state = _normalize(basis @ vectors[:, 0])
    energy, variance = _energy_variance(state, hamiltonian)
    return state, {
        "requested_dimension": dimension,
        "rank": rank,
        "energy_per_site": energy / N_SITES,
        "variance_per_site": variance / N_SITES,
        "energy_total": energy,
        "variance_total": variance,
    }


def _summary(rows):
    energies = np.asarray([row["energy_per_site"] for row in rows])
    variances = np.asarray([row["variance_per_site"] for row in rows])
    return {
        "median_energy_per_site": float(np.median(energies)),
        "best_energy_per_site": float(np.min(energies)),
        "median_variance_per_site": float(np.median(variances)),
        "best_variance_per_site": float(np.min(variances)),
        "all_energy_per_site": energies.tolist(),
        "all_variance_per_site": variances.tolist(),
    }


def residual_neural_backflow_frontier(
    *,
    configurations: np.ndarray,
    a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    exact_state_physical: np.ndarray,
    anchor_state_physical: np.ndarray,
    frozen_baselines: Mapping[str, Mapping[str, float]],
    seeds: Sequence[int],
    scramble_seed: int,
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
    gates: Mapping[str, float | int],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run carrier-preserving compact and extended residual contingencies."""

    checked_seeds = tuple(int(value) for value in seeds)
    basis = np.asarray(a1_basis, dtype=np.float64)
    hamiltonian = np.asarray(hamiltonian_a1, dtype=np.float64)
    exact = _normalize(basis.T @ exact_state_physical)
    anchor = _normalize(basis.T @ anchor_state_physical)
    anchor_energy, anchor_variance = _energy_variance(anchor, hamiltonian)
    permutation = np.random.default_rng(int(scramble_seed)).permutation(N_SITES)
    models = {}
    arrays = {"exact_a1": exact, "anchor_a1": anchor}
    for kind in ("compact_j1", "extended_j1j2"):
        physical_data = build_neural_backflow_data(
            configurations, basis, hamiltonian, kind=kind
        )
        scrambled_data = build_neural_backflow_data(
            configurations,
            basis,
            hamiltonian,
            kind=kind,
            relation_site_permutation=permutation,
        )
        physical_rows = []
        scrambled_rows = []
        best_physical = None
        best_scrambled = None
        for seed in checked_seeds:
            state, row = _train(
                physical_data,
                anchor,
                seed=seed,
                initial_scale=initial_scale,
                maximum_iterations=maximum_iterations,
                parameter_bound=parameter_bound,
            )
            physical = basis @ state
            row.update(
                {
                    "exact_fidelity": float(abs(np.vdot(exact, state)) ** 2),
                    "anchor_fidelity": float(abs(np.vdot(anchor, state)) ** 2),
                    "spin_raising_relative_norm": float(
                        spin_raising_relative_norm(physical, configurations)
                    ),
                }
            )
            physical_rows.append(row)
            if best_physical is None or row["energy_total"] < best_physical[0]:
                best_physical = (row["energy_total"], state)
            control_state, control = _train(
                scrambled_data,
                anchor,
                seed=seed,
                initial_scale=initial_scale,
                maximum_iterations=maximum_iterations,
                parameter_bound=parameter_bound,
            )
            control_physical = basis @ control_state
            control.update(
                {
                    "exact_fidelity": float(abs(np.vdot(exact, control_state)) ** 2),
                    "anchor_fidelity": float(abs(np.vdot(anchor, control_state)) ** 2),
                    "spin_raising_relative_norm": float(
                        spin_raising_relative_norm(control_physical, configurations)
                    ),
                }
            )
            scrambled_rows.append(control)
            if best_scrambled is None or control["energy_total"] < best_scrambled[0]:
                best_scrambled = (control["energy_total"], control_state)
        assert best_physical is not None and best_scrambled is not None
        physical_summary = _summary(physical_rows)
        scrambled_summary = _summary(scrambled_rows)
        maximum_dimension = physical_data.parameter_count + 2
        krylov_state, krylov = _krylov(anchor, hamiltonian, maximum_dimension)
        majority = len(checked_seeds) // 2 + 1
        model_gates = {
            "zero_residual_exactly_contains_anchor": True,
            "state_tangent_manifold_is_nonsaturating": max(
                row["state_plus_tangent_rank"] for row in physical_rows
            )
            <= maximum_dimension
            and maximum_dimension < basis.shape[1],
            "median_improves_anchor_energy_and_variance": (
                physical_summary["median_energy_per_site"] < anchor_energy / N_SITES
                and physical_summary["median_variance_per_site"] < anchor_variance / N_SITES
            ),
            "majority_beats_frozen_operator_vit_energy": sum(
                row["energy_per_site"]
                < frozen_baselines["exact_su2_operator_vit"]["energy_per_site"]
                for row in physical_rows
            )
            >= majority,
            "physical_beats_scrambled_median_energy_and_variance": (
                physical_summary["median_energy_per_site"]
                < scrambled_summary["median_energy_per_site"]
                and physical_summary["median_variance_per_site"]
                < scrambled_summary["median_variance_per_site"]
            ),
            "best_beats_same_anchor_matched_krylov_energy_and_variance": (
                physical_summary["best_energy_per_site"]
                < krylov["energy_per_site"]
                - float(gates["minimum_resolved_energy_gain_per_site"])
                and physical_summary["best_variance_per_site"]
                < krylov["variance_per_site"]
            ),
            "all_states_exact_singlets": max(
                row["spin_raising_relative_norm"]
                for row in (*physical_rows, *scrambled_rows)
            )
            <= float(gates["maximum_spin_raising_relative_norm"]),
        }
        model_gates["variational_win"] = bool(all(model_gates.values()))
        models[kind] = {
            "parameter_count": physical_data.parameter_count + 1,
            "maximum_state_plus_tangent_dimension": maximum_dimension,
            "physical_runs": physical_rows,
            "scrambled_runs": scrambled_rows,
            "physical_summary": physical_summary,
            "scrambled_summary": scrambled_summary,
            "same_anchor_dimension_matched_krylov": krylov,
            "gates": model_gates,
        }
        arrays[f"{kind}_best_physical_a1"] = best_physical[1]
        arrays[f"{kind}_best_scrambled_a1"] = best_scrambled[1]
        arrays[f"{kind}_matched_krylov_a1"] = krylov_state

    compact = bool(models["compact_j1"]["gates"]["variational_win"])
    extended = bool(models["extended_j1j2"]["gates"]["variational_win"])
    selected = "compact_j1" if compact else "extended_j1j2" if extended else None
    return {
        "case": "triangular_4x4_periodic_j2_0.125_residual_backflow",
        "n_sites": N_SITES,
        "exact_s0_gamma_a1_dimension": basis.shape[1],
        "anchor": {
            "identity": "exact-S0/A1 projection of optimized projected-VMC/two-Lanczos",
            "energy_per_site": anchor_energy / N_SITES,
            "variance_per_site": anchor_variance / N_SITES,
            "retained_norm": float(
                np.linalg.norm(basis.T @ anchor_state_physical)
                / np.linalg.norm(anchor_state_physical)
            ),
        },
        "frozen_baselines": {key: dict(value) for key, value in frozen_baselines.items()},
        "scrambled_relation_site_permutation": permutation.tolist(),
        "models": models,
        "decision": {
            "compact_variational_win": compact,
            "extended_variational_win": extended,
            "selected_model": selected,
            "triangular_n36_reopened": selected is not None,
            "branch": (
                "reopen_only_bounded_n36_matrix_free_preflight"
                if selected is not None
                else "freeze_all_current_neural_pfaffian_backflow_routes"
            ),
        },
        "claim_boundary": (
            "Exact-support carrier-preserving contingency; no repeated ED, "
            "sampled scaling, efficiency, phase, or QSL claim."
        ),
    }, arrays


__all__ = ["residual_backflow_state", "residual_neural_backflow_frontier"]
