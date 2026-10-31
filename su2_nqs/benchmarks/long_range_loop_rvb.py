"""Exact-support gate for a phase-aware long-range loop-RVB carrier.

The carrier is a coherent sum of oriented valence-bond coverings in the
orthogonal fixed-``Sz`` spin basis.  Coverings contain nearest-neighbour bonds
and at most one J2 bond.  Their fixed signs come from projected pi/0-flux or
paired-Z2 kernels, while a shared exponential loop-Jastrow depends on the
number of long bonds and on four- and six-site alternating resonance loops.
The latter features are non-factorizable over individual bonds.

Every covering is an exact spin singlet.  For this N=16 falsifier each orbit
is represented directly in a sealed complete ``S=0, Gamma-A1`` basis, so the
state and all registered tangent directions are exactly in the target sector.
This projector is an exact-small oracle, not a scalable N=36 evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from .clebsch_tree_exact import physical_permutation_index_table
from .fusion_spatial_bridge import spin_raising_relative_norm
from .paired_z2_carrier import paired_z2_kernel_bank
from .rvb_covering_oracle import (
    DimerMatching,
    dimer_matching_orbits,
    enumerate_perfect_matchings,
    fixed_m_dimer_amplitudes,
    lowest_ritz_state,
)
from .triangular_induced_parton import (
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)


SUPERCELL = ((4, 0), (0, 4))
N_SITES = 16


@dataclass(frozen=True)
class LongRangeLoopRVBData:
    """Static exact-support data for the two registered loop-RVB models."""

    configurations: np.ndarray
    a1_basis: np.ndarray
    hamiltonian_a1: np.ndarray
    anchor_a1: np.ndarray
    exact_a1: np.ndarray
    pi_flux_orbit_columns: np.ndarray
    paired_z2_orbit_columns: np.ndarray
    standardized_loop_features: np.ndarray
    loop_parity: np.ndarray
    raw_loop_features: np.ndarray
    permutation_indices: np.ndarray
    matching_count: int
    orbit_count: int
    nonzero_pi_flux_orbit_count: int
    nonzero_paired_z2_orbit_count: int


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise FloatingPointError("state has zero or nonfinite norm")
    return values / norm


def _pairing_matrix() -> np.ndarray:
    hopping = triangular_pi_zero_flux_hopping(
        4, boundary_x="A", boundary_y="P"
    )
    packed, gap = closed_shell_pairing_from_hopping(hopping)
    if gap <= 1.0e-10:
        raise RuntimeError("the registered N16 pi/0-flux kernel is open shell")
    first, second = np.triu_indices(N_SITES, k=1)
    pairing = np.zeros((N_SITES, N_SITES), dtype=np.float64)
    pairing[first, second] = packed
    pairing[second, first] = packed
    return pairing


def _alternating_resonance_present(
    selected_edges: Sequence[tuple[int, int]],
    nearest_edges: frozenset[tuple[int, int]],
) -> bool:
    """Return whether selected dimers have a connected disjoint alternative."""

    current = frozenset(tuple(sorted(edge)) for edge in selected_edges)
    vertices = tuple(sorted({site for edge in current for site in edge}))
    if len(vertices) != 2 * len(current) or len(current) not in (2, 3):
        raise ValueError("selected_edges must contain two or three disjoint dimers")
    found = False

    def visit(
        remaining: frozenset[int], chosen: tuple[tuple[int, int], ...]
    ) -> None:
        nonlocal found
        if found:
            return
        if not remaining:
            alternative = frozenset(chosen)
            if alternative == current or not alternative.isdisjoint(current):
                return
            adjacency = {site: set() for site in vertices}
            for first, second in current | alternative:
                adjacency[first].add(second)
                adjacency[second].add(first)
            seen = {vertices[0]}
            stack = [vertices[0]]
            while stack:
                site = stack.pop()
                for candidate in adjacency[site]:
                    if candidate not in seen:
                        seen.add(candidate)
                        stack.append(candidate)
            found = len(seen) == len(vertices)
            return
        first = min(remaining)
        for second in sorted(remaining - {first}):
            edge = tuple(sorted((first, second)))
            if edge in nearest_edges:
                visit(remaining - {first, second}, chosen + (edge,))

    visit(frozenset(vertices), ())
    return found


def _loop_features(
    matching: DimerMatching,
    nearest_edges: frozenset[tuple[int, int]],
) -> tuple[int, int, int]:
    """Return long-bond, four-loop, and six-loop covering features."""

    long_bonds = sum(edge not in nearest_edges for edge in matching)
    four_loops = sum(
        _alternating_resonance_present(edges, nearest_edges)
        for edges in combinations(matching, 2)
    )
    six_loops = sum(
        _alternating_resonance_present(edges, nearest_edges)
        for edges in combinations(matching, 3)
    )
    return int(long_bonds), int(four_loops), int(six_loops)


def _orbit_kernel_coefficient(orbit, pairing: np.ndarray) -> float:
    amplitudes = []
    for matching, sign in zip(
        orbit.members, orbit.member_signs, strict=True
    ):
        value = float(sign)
        for first, second in matching:
            value *= float(pairing[first, second])
        amplitudes.append(value)
    return float(np.mean(amplitudes))


def build_long_range_loop_rvb_data(
    configurations: np.ndarray,
    a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    anchor_physical: np.ndarray,
    exact_physical: np.ndarray,
) -> LongRangeLoopRVBData:
    """Compile the registered J1-plus-one-J2 covering manifold."""

    states = np.asarray(configurations, dtype=np.int8)
    basis = np.asarray(a1_basis, dtype=np.float64)
    hamiltonian = np.asarray(hamiltonian_a1, dtype=np.float64)
    if states.shape != (12870, N_SITES):
        raise ValueError("the registered support must contain all 12870 Sz=0 states")
    if basis.shape != (len(states), 14):
        raise ValueError("the complete S=0,Gamma-A1 basis must have dimension 14")
    if hamiltonian.shape != (14, 14):
        raise ValueError("hamiltonian_a1 must have shape (14,14)")

    anchor = _normalized(basis.conj().T @ np.asarray(anchor_physical))
    exact = _normalized(basis.conj().T @ np.asarray(exact_physical))
    j1_bonds = triangular_hnf_j1j2_bonds(SUPERCELL, j1=1.0, j2=0.0)
    j1j2_bonds = triangular_hnf_j1j2_bonds(SUPERCELL, j1=1.0, j2=1.0)
    nearest_edges = frozenset(
        tuple(sorted((int(first), int(second))))
        for first, second, _value in j1_bonds
    )
    all_matchings = enumerate_perfect_matchings(N_SITES, j1j2_bonds)
    matchings = tuple(
        matching
        for matching in all_matchings
        if sum(edge not in nearest_edges for edge in matching) <= 1
    )
    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    orbits = dimer_matching_orbits(matchings, group)
    if any(not orbit.has_a1_vector for orbit in orbits):
        raise RuntimeError("the registered covering support has a vanishing A1 orbit")

    pi_pairing = _pairing_matrix()
    paired_bank = {
        kernel.label: kernel for kernel in paired_z2_kernel_bank("triangular_n16")
    }
    paired_pairing = paired_bank["z2_j2_e2_mixed"].pairing
    orbit_coordinates = []
    raw_features = []
    pi_coefficients = []
    paired_coefficients = []
    for orbit in orbits:
        representative = orbit.representative
        physical = fixed_m_dimer_amplitudes(states, representative)
        # Orbit sum = orbit_size * P_A1 |representative>.  Since every basis
        # column is already A1, its coefficient is obtained without explicitly
        # replaying the full group orbit.
        orbit_coordinates.append(
            orbit.orbit_size * (basis.conj().T @ physical)
        )
        raw_features.append(_loop_features(representative, nearest_edges))
        pi_coefficients.append(_orbit_kernel_coefficient(orbit, pi_pairing))
        paired_coefficients.append(
            _orbit_kernel_coefficient(orbit, paired_pairing)
        )

    coordinates = np.asarray(orbit_coordinates, dtype=np.float64).T
    features = np.asarray(raw_features, dtype=np.float64)
    scale = np.std(features, axis=0)
    if np.any(scale <= 1.0e-12):
        raise RuntimeError("registered loop features must all fluctuate")
    standardized = (features - np.mean(features, axis=0)) / scale
    pi_coefficients = np.asarray(pi_coefficients, dtype=np.float64)
    paired_coefficients = np.asarray(paired_coefficients, dtype=np.float64)
    pi_columns = coordinates * pi_coefficients[None, :]
    paired_columns = coordinates * paired_coefficients[None, :]
    if np.linalg.norm(np.sum(pi_columns, axis=1)) <= 1.0e-14:
        raise RuntimeError("projected pi/0-flux covering carrier vanished")
    if np.linalg.norm(np.sum(paired_columns, axis=1)) <= 1.0e-14:
        raise RuntimeError("projected paired-Z2 covering carrier vanished")

    indices = physical_permutation_index_table(tuple(map(tuple, states)), group)
    return LongRangeLoopRVBData(
        configurations=states,
        a1_basis=basis,
        hamiltonian_a1=0.5 * (hamiltonian + hamiltonian.T),
        anchor_a1=anchor,
        exact_a1=exact,
        pi_flux_orbit_columns=pi_columns,
        paired_z2_orbit_columns=paired_columns,
        standardized_loop_features=standardized,
        loop_parity=(-1.0) ** features[:, 1].astype(np.int64),
        raw_loop_features=features.astype(np.int64),
        permutation_indices=indices,
        matching_count=len(matchings),
        orbit_count=len(orbits),
        nonzero_pi_flux_orbit_count=int(
            np.count_nonzero(np.abs(pi_coefficients) > 1.0e-14)
        ),
        nonzero_paired_z2_orbit_count=int(
            np.count_nonzero(np.abs(paired_coefficients) > 1.0e-14)
        ),
    )


def _primary_columns(
    parameters: np.ndarray,
    data: LongRangeLoopRVBData,
    features: np.ndarray,
) -> np.ndarray:
    weights = np.exp(np.asarray(features) @ np.asarray(parameters))
    rvb = data.pi_flux_orbit_columns @ weights
    return np.column_stack((data.anchor_a1, rvb))


def _contingency_columns(
    parameters: np.ndarray,
    data: LongRangeLoopRVBData,
    features: np.ndarray,
    parity: np.ndarray,
) -> np.ndarray:
    weights = np.exp(np.asarray(features) @ np.asarray(parameters))
    return np.column_stack(
        (
            data.anchor_a1,
            data.pi_flux_orbit_columns @ weights,
            data.pi_flux_orbit_columns @ (weights * parity),
            data.paired_z2_orbit_columns @ weights,
        )
    )


def _jax_primary_energy(data: LongRangeLoopRVBData):
    anchor = jnp.asarray(data.anchor_a1)
    columns = jnp.asarray(data.pi_flux_orbit_columns)
    hamiltonian = jnp.asarray(data.hamiltonian_a1)

    def energy(parameters: jax.Array, features: jax.Array) -> jax.Array:
        rvb = columns @ jnp.exp(features @ parameters)
        rvb = rvb - anchor * jnp.vdot(anchor, rvb)
        rvb = rvb / jnp.linalg.norm(rvb)
        matrix = jnp.asarray(
            (
                (jnp.vdot(anchor, hamiltonian @ anchor), jnp.vdot(anchor, hamiltonian @ rvb)),
                (jnp.vdot(rvb, hamiltonian @ anchor), jnp.vdot(rvb, hamiltonian @ rvb)),
            )
        )
        matrix = 0.5 * (matrix + matrix.conj().T)
        return jnp.linalg.eigvalsh(matrix)[0].real

    return jax.jit(jax.value_and_grad(energy))


def _jax_contingency_energy(data: LongRangeLoopRVBData):
    anchor = jnp.asarray(data.anchor_a1)
    pi_columns = jnp.asarray(data.pi_flux_orbit_columns)
    paired_columns = jnp.asarray(data.paired_z2_orbit_columns)
    hamiltonian = jnp.asarray(data.hamiltonian_a1)

    def energy(
        parameters: jax.Array, features: jax.Array, parity: jax.Array
    ) -> jax.Array:
        weights = jnp.exp(features @ parameters)
        columns = jnp.column_stack(
            (
                anchor,
                pi_columns @ weights,
                pi_columns @ (weights * parity),
                paired_columns @ weights,
            )
        )
        orthonormal, _triangular = jnp.linalg.qr(columns, mode="reduced")
        matrix = orthonormal.conj().T @ hamiltonian @ orthonormal
        matrix = 0.5 * (matrix + matrix.conj().T)
        return jnp.linalg.eigvalsh(matrix)[0].real

    return jax.jit(jax.value_and_grad(energy))


def _optimize(
    value_and_gradient,
    initial: np.ndarray,
    arguments: tuple[np.ndarray, ...],
    *,
    maximum_iterations: int,
    parameter_bound: float,
) -> tuple[np.ndarray, dict[str, object]]:
    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_gradient(
            jnp.asarray(parameters), *(jnp.asarray(argument) for argument in arguments)
        )
        return float(value), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-parameter_bound, parameter_bound)] * len(initial),
        options={
            "maxiter": int(maximum_iterations),
            "ftol": 1.0e-14,
            "gtol": 1.0e-10,
        },
    )
    return np.asarray(result.x, dtype=np.float64), {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "message": str(result.message),
    }


def _metrics(state: np.ndarray, data: LongRangeLoopRVBData) -> dict[str, float]:
    vector = _normalized(state)
    action = data.hamiltonian_a1 @ vector
    energy = float(np.vdot(vector, action).real)
    residual = action - energy * vector
    physical = data.a1_basis @ vector
    projected = np.mean(physical[data.permutation_indices], axis=0)
    return {
        "energy_total": energy,
        "energy_per_site": energy / N_SITES,
        "variance_total": float(np.vdot(residual, residual).real),
        "variance_per_site": float(np.vdot(residual, residual).real) / N_SITES,
        "exact_fidelity": float(abs(np.vdot(data.exact_a1, vector)) ** 2),
        "spin_raising_relative_norm": spin_raising_relative_norm(
            physical, data.configurations
        ),
        "gamma_a1_relative_residual": float(
            np.linalg.norm(projected - physical) / np.linalg.norm(physical)
        ),
    }


def _krylov_state(
    anchor: np.ndarray, hamiltonian: np.ndarray, dimension: int
):
    columns = []
    vector = _normalized(anchor)
    for _index in range(int(dimension)):
        columns.append(vector)
        vector = hamiltonian @ vector
    return lowest_ritz_state(np.column_stack(columns), hamiltonian)


def _summary(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    energies = np.asarray(
        [run["metrics"]["energy_per_site"] for run in runs], dtype=np.float64
    )
    variances = np.asarray(
        [run["metrics"]["variance_per_site"] for run in runs], dtype=np.float64
    )
    return {
        "seed_count": len(runs),
        "median_energy_per_site": float(np.median(energies)),
        "best_energy_per_site": float(np.min(energies)),
        "median_variance_per_site": float(np.median(variances)),
        "best_variance_per_site": float(np.min(variances)),
        "all_energy_per_site": energies.tolist(),
        "all_variance_per_site": variances.tolist(),
    }


def _model_runs(
    data: LongRangeLoopRVBData,
    *,
    model: str,
    seeds: Sequence[int],
    scramble_seeds: Sequence[int],
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    if len(seeds) != len(scramble_seeds):
        raise ValueError("seeds and scramble_seeds must have equal length")
    physical_runs = []
    scrambled_runs = []
    arrays: dict[str, np.ndarray] = {}
    if model == "single_psg_loop_jastrow":
        value_and_gradient = _jax_primary_energy(data)
        build_columns = lambda parameters, features, parity: _primary_columns(
            parameters, data, features
        )
        parameter_count = 3
    elif model == "two_psg_loop_resonance_bank":
        value_and_gradient = _jax_contingency_energy(data)
        build_columns = lambda parameters, features, parity: _contingency_columns(
            parameters, data, features, parity
        )
        parameter_count = 3
    else:
        raise ValueError(f"unknown loop-RVB model: {model!r}")

    for index, (seed, scramble_seed) in enumerate(
        zip(seeds, scramble_seeds, strict=True)
    ):
        initial = np.random.default_rng(int(seed)).normal(
            scale=float(initial_scale), size=parameter_count
        )
        for control, feature_permutation in (
            ("physical", np.arange(data.orbit_count, dtype=np.int64)),
            (
                "scrambled",
                np.random.default_rng(int(scramble_seed)).permutation(
                    data.orbit_count
                ),
            ),
        ):
            features = data.standardized_loop_features[feature_permutation]
            parity = data.loop_parity[feature_permutation]
            arguments = (
                (features,)
                if model == "single_psg_loop_jastrow"
                else (features, parity)
            )
            parameters, optimizer = _optimize(
                value_and_gradient,
                initial,
                arguments,
                maximum_iterations=maximum_iterations,
                parameter_bound=parameter_bound,
            )
            ritz = lowest_ritz_state(
                build_columns(parameters, features, parity),
                data.hamiltonian_a1,
            )
            row = {
                "seed": int(seed),
                "scramble_seed": int(scramble_seed),
                "parameters": parameters.tolist(),
                "input_column_rank": int(ritz.rank),
                "optimizer": optimizer,
                "metrics": _metrics(ritz.state, data),
            }
            if control == "physical":
                physical_runs.append(row)
            else:
                scrambled_runs.append(row)
            arrays[f"{model}_{control}_seed{seed}_a1"] = ritz.state
            arrays[f"{model}_{control}_seed{seed}_parameters"] = parameters
    return physical_runs, scrambled_runs, arrays


def long_range_loop_rvb_frontier(
    *,
    configurations: np.ndarray,
    a1_basis: np.ndarray,
    hamiltonian_a1: np.ndarray,
    anchor_physical: np.ndarray,
    exact_physical: np.ndarray,
    frozen_baselines: Mapping[str, Mapping[str, float]],
    seeds: Sequence[int],
    scramble_seeds: Sequence[int],
    initial_scale: float,
    maximum_iterations: int,
    parameter_bound: float,
    minimum_resolved_energy_gain_per_site: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the preregistered primary and two-PSG contingency gates."""

    data = build_long_range_loop_rvb_data(
        configurations,
        a1_basis,
        hamiltonian_a1,
        anchor_physical,
        exact_physical,
    )
    anchor_metrics = _metrics(data.anchor_a1, data)
    zero_loop = lowest_ritz_state(
        _primary_columns(
            np.zeros(3), data, data.standardized_loop_features
        ),
        data.hamiltonian_a1,
    )
    arrays: dict[str, np.ndarray] = {
        "anchor_a1": data.anchor_a1,
        "exact_a1": data.exact_a1,
        "raw_loop_features": data.raw_loop_features,
        "standardized_loop_features": data.standardized_loop_features,
        "loop_parity": data.loop_parity,
        "pi_flux_orbit_columns": data.pi_flux_orbit_columns,
        "paired_z2_orbit_columns": data.paired_z2_orbit_columns,
        "zero_loop_a1": zero_loop.state,
    }
    model_specs = (
        ("single_psg_loop_jastrow", 5),
        ("two_psg_loop_resonance_bank", 7),
    )
    models = {}
    tolerance = float(minimum_resolved_energy_gain_per_site)
    for model, matched_dimension in model_specs:
        physical_runs, scrambled_runs, model_arrays = _model_runs(
            data,
            model=model,
            seeds=seeds,
            scramble_seeds=scramble_seeds,
            initial_scale=initial_scale,
            maximum_iterations=maximum_iterations,
            parameter_bound=parameter_bound,
        )
        arrays.update(model_arrays)
        physical_summary = _summary(physical_runs)
        scrambled_summary = _summary(scrambled_runs)
        best_index = int(
            np.argmin(
                [run["metrics"]["energy_per_site"] for run in physical_runs]
            )
        )
        best_state = arrays[f"{model}_physical_seed{seeds[best_index]}_a1"]
        best_metrics = _metrics(best_state, data)
        krylov = _krylov_state(
            data.anchor_a1, data.hamiltonian_a1, matched_dimension
        )
        krylov_metrics = _metrics(krylov.state, data)
        paired_joint_wins = sum(
            physical["metrics"]["energy_per_site"]
            < scrambled["metrics"]["energy_per_site"] - tolerance
            and physical["metrics"]["variance_per_site"]
            < scrambled["metrics"]["variance_per_site"] - tolerance
            for physical, scrambled in zip(
                physical_runs, scrambled_runs, strict=True
            )
        )
        gates = {
            "state_manifold_is_nonsaturating": matched_dimension < 14,
            "physical_beats_scrambled_median_energy_and_variance": bool(
                physical_summary["median_energy_per_site"]
                < scrambled_summary["median_energy_per_site"] - tolerance
                and physical_summary["median_variance_per_site"]
                < scrambled_summary["median_variance_per_site"] - tolerance
            ),
            "physical_joint_paired_wins": int(paired_joint_wins),
            "best_beats_operator_vit_energy_and_variance": bool(
                best_metrics["energy_per_site"]
                < frozen_baselines["exact_su2_operator_vit"]["energy_per_site"]
                - tolerance
                and best_metrics["variance_per_site"]
                < frozen_baselines["exact_su2_operator_vit"]["variance_per_site"]
                - tolerance
            ),
            "best_beats_matched_krylov_energy_and_variance": bool(
                best_metrics["energy_per_site"]
                < krylov_metrics["energy_per_site"] - tolerance
                and best_metrics["variance_per_site"]
                < krylov_metrics["variance_per_site"] - tolerance
            ),
            "all_states_exact_s0_gamma_a1": bool(
                max(
                    run["metrics"]["spin_raising_relative_norm"]
                    for run in physical_runs + scrambled_runs
                )
                < 1.0e-11
                and max(
                    run["metrics"]["gamma_a1_relative_residual"]
                    for run in physical_runs + scrambled_runs
                )
                < 1.0e-11
            ),
        }
        gates["variational_win"] = bool(
            gates["state_manifold_is_nonsaturating"]
            and gates["physical_beats_scrambled_median_energy_and_variance"]
            and paired_joint_wins >= 3
            and gates["best_beats_operator_vit_energy_and_variance"]
            and gates["best_beats_matched_krylov_energy_and_variance"]
            and gates["all_states_exact_s0_gamma_a1"]
        )
        models[model] = {
            "loop_parameter_count": 3,
            "maximum_state_plus_tangent_dimension": matched_dimension,
            "physical_runs": physical_runs,
            "scrambled_runs": scrambled_runs,
            "physical_summary": physical_summary,
            "scrambled_summary": scrambled_summary,
            "best_physical_metrics": best_metrics,
            "matched_dimension_krylov": {
                "requested_dimension": matched_dimension,
                "rank": int(krylov.rank),
                **krylov_metrics,
            },
            "gates": gates,
        }
        arrays[f"{model}_best_physical_a1"] = best_state
        arrays[f"{model}_matched_krylov_a1"] = krylov.state

    selected = (
        "single_psg_loop_jastrow"
        if models["single_psg_loop_jastrow"]["gates"]["variational_win"]
        else (
            "two_psg_loop_resonance_bank"
            if models["two_psg_loop_resonance_bank"]["gates"]["variational_win"]
            else None
        )
    )
    result = {
        "case": "triangular_n16_j2_0.125",
        "n_sites": N_SITES,
        "fixed_sz_dimension": len(data.configurations),
        "exact_s0_gamma_a1_dimension": data.a1_basis.shape[1],
        "matching_support": {
            "definition": "J1 coverings plus coverings with exactly one J2 bond",
            "matching_count": data.matching_count,
            "orbit_count": data.orbit_count,
            "nonzero_pi_flux_orbit_count": data.nonzero_pi_flux_orbit_count,
            "nonzero_paired_z2_orbit_count": data.nonzero_paired_z2_orbit_count,
        },
        "feature_ranges": {
            "long_bond_count": [
                int(np.min(data.raw_loop_features[:, 0])),
                int(np.max(data.raw_loop_features[:, 0])),
            ],
            "four_site_resonance_count": [
                int(np.min(data.raw_loop_features[:, 1])),
                int(np.max(data.raw_loop_features[:, 1])),
            ],
            "six_site_resonance_count": [
                int(np.min(data.raw_loop_features[:, 2])),
                int(np.max(data.raw_loop_features[:, 2])),
            ],
        },
        "anchor": anchor_metrics,
        "factorized_psg_zero_loop": _metrics(zero_loop.state, data),
        "frozen_baselines": {
            key: dict(value) for key, value in frozen_baselines.items()
        },
        "models": models,
        "decision": {
            "selected_model": selected,
            "triangular_n36_reopened": selected is not None,
            "branch": (
                "license_bounded_n36_loop_rvb_design_only"
                if selected is not None
                else "freeze_current_long_range_loop_rvb_route"
            ),
        },
        "claim_boundary": (
            "This is an exact-support non-saturating state-manifold gate. "
            "The exact A1 projection is an oracle and cannot by itself support "
            "an N=36 efficiency, scaling, or QSL claim."
        ),
    }
    return result, arrays


__all__ = [
    "LongRangeLoopRVBData",
    "build_long_range_loop_rvb_data",
    "long_range_loop_rvb_frontier",
]
