"""Exact-small gate for symmetry-projected neural backflow singlet carriers.

The raw wavefunction is a number-projected singlet Pfaffian (a determinant in
the spin-singlet block form) whose pair kernel depends on the complete spin
configuration through a relational one-hidden-channel backflow map.  The raw
map is not assumed to be a spin scalar.  Instead, the benchmark projects it
onto a sealed complete ``S=0, Gamma-A1`` basis before normalization.  Thus
state and parameter tangents are exactly in the target sector on ``N=16``.

Two deliberately small manifolds are registered.  The compact J1 model has
four real parameters and at most five state-plus-tangent directions.  Its
pre-registered contingency adds J2 context/output for six parameters and at
most seven directions.  Each is compared with a Krylov space of the same
maximum dimension from the identical projected pi-flux carrier and with a
site-scrambled relation control of exactly the same architecture.

This exact projector is an oracle, not yet a scalable N=36 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal, Mapping, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from .fusion_spatial_bridge import spin_raising_relative_norm
from .triangular_induced_parton import (
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)


ModelKind = Literal["compact_j1", "extended_j1j2"]
SUPERCELL = ((4, 0), (0, 4))
N_SITES = 16


@dataclass(frozen=True)
class NeuralBackflowData:
    """Static exact-support tensors for one relational backflow model."""

    configurations: np.ndarray
    up_sites: np.ndarray
    down_sites: np.ndarray
    fermion_signs: np.ndarray
    base_pairing: np.ndarray
    j1_relation: np.ndarray
    j2_relation: np.ndarray
    a1_basis: np.ndarray
    hamiltonian_a1: np.ndarray
    kind: ModelKind

    @property
    def parameter_count(self) -> int:
        return 4 if self.kind == "compact_j1" else 6


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise FloatingPointError("state has zero or nonfinite norm")
    return values / norm


def _energy_variance(
    state: np.ndarray, hamiltonian: np.ndarray
) -> tuple[float, float]:
    vector = _normalized(state)
    action = hamiltonian @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    return energy, float(np.vdot(residual, residual).real)


def _pairing_matrix(boundary_x: str = "A", boundary_y: str = "P") -> np.ndarray:
    hopping = triangular_pi_zero_flux_hopping(
        4, boundary_x=boundary_x, boundary_y=boundary_y
    )
    packed, gap = closed_shell_pairing_from_hopping(hopping)
    if gap <= 1.0e-10:
        raise RuntimeError("the registered N16 backflow carrier is open shell")
    first, second = np.triu_indices(N_SITES, k=1)
    pairing = np.zeros((N_SITES, N_SITES), dtype=np.float64)
    pairing[first, second] = packed
    pairing[second, first] = packed
    pairing /= np.linalg.norm(pairing)
    return pairing


def _relation_matrix(*, j1: float, j2: float) -> np.ndarray:
    matrix = np.zeros((N_SITES, N_SITES), dtype=np.float64)
    for first, second, value in triangular_hnf_j1j2_bonds(
        SUPERCELL, j1=j1, j2=j2
    ):
        matrix[int(first), int(second)] += float(value)
        matrix[int(second), int(first)] += float(value)
    np.fill_diagonal(matrix, 0.0)
    degree = np.sum(np.abs(matrix), axis=1)
    if np.max(np.abs(degree - degree[0])) > 1.0e-12 or degree[0] <= 0.0:
        raise RuntimeError("registered triangular relation is not regular")
    return matrix / degree[0]


def _checked_half_filled(configurations: np.ndarray) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != N_SITES or len(states) < 1:
        raise ValueError("configurations must have shape (batch, 16)")
    if np.any((states != 0) & (states != 1)):
        raise ValueError("configurations must be binary")
    if np.any(np.sum(states, axis=1) != N_SITES // 2):
        raise ValueError("every configuration must have Sz=0")
    return states


def build_neural_backflow_data(
    configurations: np.ndarray,
    a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    *,
    kind: ModelKind,
    relation_site_permutation: Sequence[int] | None = None,
) -> NeuralBackflowData:
    """Build physical or parameter-matched site-scrambled relation tensors."""

    states = _checked_half_filled(configurations)
    basis = np.asarray(a1_basis, dtype=np.float64)
    hamiltonian = np.asarray(hamiltonian_a1, dtype=np.float64)
    if basis.shape[0] != len(states) or basis.ndim != 2:
        raise ValueError("a1_basis must have one row per configuration")
    if hamiltonian.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("hamiltonian_a1 does not match the A1 basis")
    if kind not in ("compact_j1", "extended_j1j2"):
        raise ValueError("unknown neural-backflow model kind")
    j1_relation = _relation_matrix(j1=1.0, j2=0.0)
    j2_relation = _relation_matrix(j1=0.0, j2=1.0)
    if relation_site_permutation is not None:
        image = np.asarray(tuple(int(value) for value in relation_site_permutation))
        if image.shape != (N_SITES,) or tuple(sorted(image.tolist())) != tuple(
            range(N_SITES)
        ):
            raise ValueError("relation_site_permutation must permute all 16 sites")
        j1_relation = j1_relation[np.ix_(image, image)]
        j2_relation = j2_relation[np.ix_(image, image)]

    order = np.argsort(states, axis=1, kind="stable")
    down = order[:, : N_SITES // 2]
    up = order[:, N_SITES // 2 :]
    left_down = np.cumsum(1 - states, axis=1) - (1 - states)
    inversions = np.sum(states * left_down, axis=1)
    parity = (inversions + 28) % 2
    signs = np.where(parity == 0, 1.0, -1.0)
    return NeuralBackflowData(
        configurations=states,
        up_sites=np.asarray(up, dtype=np.int32),
        down_sites=np.asarray(down, dtype=np.int32),
        fermion_signs=np.asarray(signs, dtype=np.float64),
        base_pairing=_pairing_matrix(),
        j1_relation=j1_relation,
        j2_relation=j2_relation,
        a1_basis=basis,
        hamiltonian_a1=0.5 * (hamiltonian + hamiltonian.T),
        kind=kind,
    )


def neural_backflow_pairing_batch(
    parameters: jax.Array, data: NeuralBackflowData
) -> jax.Array:
    """Return the configuration-dependent symmetric pair kernels."""

    if parameters.ndim != 1 or parameters.shape[0] != data.parameter_count:
        raise ValueError("parameters have the wrong shape")
    spins = 2.0 * jnp.asarray(data.configurations) - 1.0
    j1 = jnp.asarray(data.j1_relation)
    j2 = jnp.asarray(data.j2_relation)
    bounded = jnp.tanh(parameters)
    if data.kind == "compact_j1":
        self_weight, j1_weight, j1_output, dense_output = bounded
        context = jnp.tanh(
            self_weight * spins + j1_weight * (spins @ j1.T)
        )
        pair = (
            jnp.asarray(data.base_pairing)[None, :, :]
            + 0.5
            * j1_output
            * j1[None, :, :]
            * (context[:, :, None] + context[:, None, :])
            + 0.5
            * dense_output
            * jnp.asarray(data.base_pairing)[None, :, :]
            * context[:, :, None]
            * context[:, None, :]
        )
    else:
        (
            self_weight,
            j1_weight,
            j2_weight,
            j1_output,
            j2_output,
            dense_output,
        ) = bounded
        context = jnp.tanh(
            self_weight * spins
            + j1_weight * (spins @ j1.T)
            + j2_weight * (spins @ j2.T)
        )
        pair = (
            jnp.asarray(data.base_pairing)[None, :, :]
            + 0.5
            * j1_output
            * j1[None, :, :]
            * (context[:, :, None] + context[:, None, :])
            + 0.5
            * j2_output
            * j2[None, :, :]
            * (context[:, :, None] + context[:, None, :])
            + 0.5
            * dense_output
            * jnp.asarray(data.base_pairing)[None, :, :]
            * context[:, :, None]
            * context[:, None, :]
        )
    diagonal = jnp.arange(N_SITES)
    return pair.at[:, diagonal, diagonal].set(0.0)


def neural_backflow_reduced_state(
    parameters: jax.Array, data: NeuralBackflowData
) -> jax.Array:
    """Evaluate and exactly project the neural Pfaffian into ``S=0,A1``."""

    pair = neural_backflow_pairing_batch(parameters, data)
    rows = jnp.arange(len(data.configurations))[:, None, None]
    up = jnp.asarray(data.up_sites)[:, :, None]
    down = jnp.asarray(data.down_sites)[:, None, :]
    blocks = pair[rows, up, down]
    raw = jnp.asarray(data.fermion_signs) * jnp.linalg.det(blocks)
    coefficients = jnp.asarray(data.a1_basis).T @ raw
    norm = jnp.sqrt(jnp.vdot(coefficients, coefficients).real)
    return coefficients / norm


def _train_one(
    data: NeuralBackflowData,
    *,
    seed: int,
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
) -> tuple[np.ndarray, dict[str, object]]:
    hamiltonian = jnp.asarray(data.hamiltonian_a1)

    def loss(parameters: jax.Array) -> jax.Array:
        state = neural_backflow_reduced_state(parameters, data)
        return jnp.vdot(state, hamiltonian @ state).real

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    evaluations = 0

    def objective(parameters: np.ndarray):
        nonlocal evaluations
        evaluations += 1
        value, gradient = value_and_gradient(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=np.float64)

    initial = np.random.default_rng(int(seed)).normal(
        0.0, float(initial_scale), size=data.parameter_count
    )
    # Avoid the bilinear zero-gradient point while keeping every seed inside
    # the same small initialization ball.
    initial += np.linspace(0.03, -0.03, data.parameter_count)
    started = time.perf_counter()
    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=tuple(
            (-float(parameter_bound), float(parameter_bound))
            for _ in range(data.parameter_count)
        ),
        options={
            "maxiter": int(maximum_iterations),
            "ftol": 1.0e-14,
            "gtol": 1.0e-10,
            "maxls": 40,
        },
    )
    elapsed = time.perf_counter() - started
    parameters = np.asarray(result.x, dtype=np.float64)
    state = np.asarray(neural_backflow_reduced_state(jnp.asarray(parameters), data))
    energy, variance = _energy_variance(state, data.hamiltonian_a1)
    tangent = np.asarray(
        jax.jacrev(lambda value: neural_backflow_reduced_state(value, data))(
            jnp.asarray(parameters)
        )
    )
    span = np.column_stack((state, tangent))
    singular_values = np.linalg.svd(span, compute_uv=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1.0e-9))
    return state, {
        "seed": int(seed),
        "parameters": parameters.tolist(),
        "parameter_count": data.parameter_count,
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": variance,
        "variance_per_site": variance / N_SITES,
        "state_plus_tangent_rank": rank,
        "state_plus_tangent_singular_values": singular_values.tolist(),
        "objective_evaluations": evaluations,
        "maximum_iterations": int(maximum_iterations),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "wall_seconds": elapsed,
    }


def _krylov_state(
    seed: np.ndarray, hamiltonian: np.ndarray, dimension: int
) -> tuple[np.ndarray, dict[str, object]]:
    columns = []
    current = _normalized(seed)
    for _ in range(int(dimension)):
        columns.append(current)
        current = hamiltonian @ current
    matrix = np.column_stack(columns)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = singular_values[0] * 1.0e-12
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = left[:, :rank]
    projected = basis.conj().T @ hamiltonian @ basis
    energies, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    state = _normalized(basis @ vectors[:, 0])
    energy, variance = _energy_variance(state, hamiltonian)
    return state, {
        "requested_dimension": int(dimension),
        "rank": rank,
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": variance,
        "variance_per_site": variance / N_SITES,
        "singular_values": singular_values.tolist(),
    }


def _state_audit(
    state_a1: np.ndarray,
    *,
    exact_a1: np.ndarray,
    reference_a1: np.ndarray,
    a1_basis: np.ndarray,
    configurations: np.ndarray,
) -> dict[str, object]:
    state = _normalized(state_a1)
    exact = _normalized(exact_a1)
    reference = _normalized(reference_a1)
    physical = np.asarray(a1_basis) @ state
    overlap = np.vdot(reference, state)
    aligned = state * np.exp(-1.0j * np.angle(overlap))
    changed = np.real(aligned) * np.real(reference) < 0.0
    return {
        "exact_fidelity": float(abs(np.vdot(exact, state)) ** 2),
        "reference_fidelity": float(abs(overlap) ** 2),
        "optimized_probability_on_reduced_sign_changed_coordinates": float(
            np.sum(np.abs(state[changed]) ** 2)
        ),
        "spin_raising_relative_norm": float(
            spin_raising_relative_norm(physical, configurations)
        ),
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    energies = np.asarray([float(row["energy_per_site"]) for row in rows])
    variances = np.asarray([float(row["variance_per_site"]) for row in rows])
    return {
        "seed_count": len(rows),
        "median_energy_per_site": float(np.median(energies)),
        "best_energy_per_site": float(np.min(energies)),
        "worst_energy_per_site": float(np.max(energies)),
        "median_variance_per_site": float(np.median(variances)),
        "best_variance_per_site": float(np.min(variances)),
        "all_energy_per_site": energies.tolist(),
        "all_variance_per_site": variances.tolist(),
    }


def neural_backflow_pfaffian_frontier(
    *,
    configurations: np.ndarray,
    a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    exact_state_physical: np.ndarray,
    base_carrier_a1: np.ndarray,
    projected_vmc_physical: np.ndarray,
    frozen_baselines: Mapping[str, Mapping[str, float]],
    seeds: Sequence[int],
    scramble_seed: int,
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
    gates: Mapping[str, float | int],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Train compact/extended physical and scrambled backflow manifolds."""

    checked_seeds = tuple(int(seed) for seed in seeds)
    if len(checked_seeds) < 3 or len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("at least three distinct training seeds are required")
    basis = np.asarray(a1_basis, dtype=np.float64)
    hamiltonian = np.asarray(hamiltonian_a1, dtype=np.float64)
    exact_a1 = _normalized(basis.T @ np.asarray(exact_state_physical))
    base_a1 = _normalized(base_carrier_a1)
    projected_vmc_a1 = _normalized(basis.T @ np.asarray(projected_vmc_physical))
    reconstruction = {
        "a1_basis_gram_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]))
        ),
        "exact_state_projection_residual": float(
            np.linalg.norm(basis @ (basis.T @ exact_state_physical) - exact_state_physical)
        ),
        "base_carrier_projection_norm": float(np.linalg.norm(base_a1)),
        "projected_vmc_s0_a1_retained_norm": float(
            np.linalg.norm(basis.T @ projected_vmc_physical)
            / np.linalg.norm(projected_vmc_physical)
        ),
    }
    permutation = np.random.default_rng(int(scramble_seed)).permutation(N_SITES)
    model_results = {}
    arrays: dict[str, np.ndarray] = {
        "exact_a1": exact_a1,
        "base_carrier_a1": base_a1,
        "projected_vmc_a1": projected_vmc_a1,
    }
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
        raw_zero = np.asarray(
            neural_backflow_reduced_state(
                jnp.zeros(physical_data.parameter_count), physical_data
            )
        )
        base_fidelity = float(abs(np.vdot(base_a1, raw_zero)) ** 2)
        physical_rows = []
        scrambled_rows = []
        best_physical: tuple[float, np.ndarray] | None = None
        best_scrambled: tuple[float, np.ndarray] | None = None
        for seed in checked_seeds:
            state, row = _train_one(
                physical_data,
                seed=seed,
                initial_scale=initial_scale,
                maximum_iterations=maximum_iterations,
                parameter_bound=parameter_bound,
            )
            row.update(
                _state_audit(
                    state,
                    exact_a1=exact_a1,
                    reference_a1=raw_zero,
                    a1_basis=basis,
                    configurations=configurations,
                )
            )
            physical_rows.append(row)
            if best_physical is None or float(row["energy_total"]) < best_physical[0]:
                best_physical = (float(row["energy_total"]), state)

            control_state, control_row = _train_one(
                scrambled_data,
                seed=seed,
                initial_scale=initial_scale,
                maximum_iterations=maximum_iterations,
                parameter_bound=parameter_bound,
            )
            control_row.update(
                _state_audit(
                    control_state,
                    exact_a1=exact_a1,
                    reference_a1=raw_zero,
                    a1_basis=basis,
                    configurations=configurations,
                )
            )
            scrambled_rows.append(control_row)
            if best_scrambled is None or float(control_row["energy_total"]) < best_scrambled[0]:
                best_scrambled = (float(control_row["energy_total"]), control_state)
        assert best_physical is not None and best_scrambled is not None
        matched_dimension = physical_data.parameter_count + 1
        krylov_state, krylov = _krylov_state(base_a1, hamiltonian, matched_dimension)
        stronger_state, stronger_krylov = _krylov_state(
            projected_vmc_a1, hamiltonian, matched_dimension
        )
        physical_summary = _summary(physical_rows)
        scrambled_summary = _summary(scrambled_rows)
        majority_count = len(checked_seeds) // 2 + 1
        gate = {
            "base_limit_reproduces_projected_pi_flux": base_fidelity
            >= float(gates["minimum_base_carrier_fidelity"]),
            "state_tangent_manifold_is_nonsaturating": max(
                int(row["state_plus_tangent_rank"]) for row in physical_rows
            )
            <= matched_dimension
            and matched_dimension < basis.shape[1],
            "majority_beats_frozen_operator_vit_energy": sum(
                float(row["energy_per_site"])
                < float(frozen_baselines["exact_su2_operator_vit"]["energy_per_site"])
                for row in physical_rows
            )
            >= majority_count,
            "median_beats_projected_vmc_energy_and_variance": (
                float(physical_summary["median_energy_per_site"])
                < float(frozen_baselines["projected_vmc_two_lanczos"]["energy_per_site"])
                and float(physical_summary["median_variance_per_site"])
                < float(frozen_baselines["projected_vmc_two_lanczos"]["variance_per_site"])
            ),
            "median_beats_fixed_sz_vit_energy_and_variance": (
                float(physical_summary["median_energy_per_site"])
                < float(frozen_baselines["fixed_sz_spatial_vit"]["energy_per_site"])
                and float(physical_summary["median_variance_per_site"])
                < float(frozen_baselines["fixed_sz_spatial_vit"]["variance_per_site"])
            ),
            "physical_beats_scrambled_median_energy_and_variance": (
                float(physical_summary["median_energy_per_site"])
                < float(scrambled_summary["median_energy_per_site"])
                and float(physical_summary["median_variance_per_site"])
                < float(scrambled_summary["median_variance_per_site"])
            ),
            "best_beats_same_carrier_matched_krylov_energy_and_variance": (
                float(physical_summary["best_energy_per_site"])
                < float(krylov["energy_per_site"])
                - float(gates["minimum_resolved_energy_gain_per_site"])
                and min(float(row["variance_per_site"]) for row in physical_rows)
                < float(krylov["variance_per_site"])
            ),
            "all_states_exact_singlets": max(
                float(row["spin_raising_relative_norm"])
                for row in (*physical_rows, *scrambled_rows)
            )
            <= float(gates["maximum_spin_raising_relative_norm"]),
        }
        gate["variational_win"] = bool(all(gate.values()))
        model_results[kind] = {
            "parameter_count": physical_data.parameter_count,
            "maximum_state_plus_tangent_dimension": matched_dimension,
            "base_limit_fidelity_with_sealed_projected_pi_flux": base_fidelity,
            "physical_runs": physical_rows,
            "scrambled_runs": scrambled_rows,
            "physical_summary": physical_summary,
            "scrambled_summary": scrambled_summary,
            "same_carrier_dimension_matched_krylov": krylov,
            "stronger_projected_vmc_seeded_krylov_ceiling": stronger_krylov,
            "gates": gate,
        }
        arrays[f"{kind}_best_physical_a1"] = best_physical[1]
        arrays[f"{kind}_best_scrambled_a1"] = best_scrambled[1]
        arrays[f"{kind}_matched_krylov_a1"] = krylov_state
        arrays[f"{kind}_stronger_krylov_a1"] = stronger_state

    compact_passed = bool(model_results["compact_j1"]["gates"]["variational_win"])
    extended_passed = bool(
        model_results["extended_j1j2"]["gates"]["variational_win"]
    )
    selected = (
        "compact_j1"
        if compact_passed
        else "extended_j1j2"
        if extended_passed
        else None
    )
    result = {
        "case": "triangular_4x4_periodic_j2_0.125",
        "n_sites": N_SITES,
        "fixed_sz_dimension": len(configurations),
        "exact_s0_gamma_a1_dimension": basis.shape[1],
        "architecture": {
            "raw_carrier": "configuration-dependent real singlet-Pfaffian determinant",
            "configuration_map": "relation-aware one-hidden-channel tanh backflow",
            "spin_constraint": "exact projection into sealed complete S=0 basis",
            "space_constraint": "same sealed complete Gamma-A1 basis",
            "compact_parameters": 4,
            "extended_parameters": 6,
            "magnetic_indices_in_final_model": False,
            "n36_implementation_status": "not_implemented_or_licensed",
        },
        "reconstruction_audit": reconstruction,
        "scrambled_relation_site_permutation": permutation.tolist(),
        "frozen_baselines": {
            key: dict(value) for key, value in frozen_baselines.items()
        },
        "models": model_results,
        "decision": {
            "compact_variational_win": compact_passed,
            "extended_contingency_variational_win": extended_passed,
            "selected_model": selected,
            "triangular_n36_reopened": selected is not None,
            "branch": (
                "reopen_only_bounded_n36_matrix_free_preflight"
                if selected is not None
                else "freeze_projected_neural_backflow_pfaffian_route"
            ),
        },
        "claim_boundary": (
            "Exact-support architecture test with reused sealed references. "
            "No repeated ED, sampled scaling, efficiency, phase, or QSL claim."
        ),
    }
    return result, arrays


__all__ = [
    "NeuralBackflowData",
    "build_neural_backflow_data",
    "neural_backflow_pairing_batch",
    "neural_backflow_pfaffian_frontier",
    "neural_backflow_reduced_state",
]
