"""Matched exact-support frontier for SU(2), fixed-Sz, and generic ViTs."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import eigsh

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.fixed_sz_generic_vit import (
    FixedSzGenericViTConfig,
    fixed_sz_generic_vit_log_amplitude,
    init_fixed_sz_generic_vit,
)
from ..models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)
from ..models.su2_four_cell_sector_vit import (
    exhaustive_four_cell_sector_actions,
    init_su2_four_cell_sector_vit,
    square_sector_config,
    su2_four_cell_sector_log_amplitudes,
    su2_four_cell_sector_vit_parameter_count,
    triangular_sector_config,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    coo_matvec,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPERCELL,
    TRIANGULAR_N12_SUPPORTS,
    _coo_sparse,
    _energy_variance,
    _square_response_panel,
    _triangular_response_panel,
    _two_lanczos,
)
from .xdiag_physical_seed_krylov import SMALL_CASES, physical_seed_amplitudes


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "exact_su2_a1_schur_vit",
    "fixed_sz_a1_spatial_vit",
    "symmetry_unaware_fixed_sz_vit",
)


def _parameter_count(params: object) -> int:
    return int(sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(params)))


def _stable_generic_amplitudes(
    params: object,
    configurations: jax.Array,
    config: FixedSzGenericViTConfig,
) -> jax.Array:
    logs = jax.vmap(
        lambda row: fixed_sz_generic_vit_log_amplitude(params, row, config)
    )(configurations)
    finite = jnp.isfinite(jnp.real(logs))
    maximum = jnp.max(jnp.where(finite, jnp.real(logs), -jnp.inf))
    shifted = jnp.where(finite, logs - jax.lax.stop_gradient(maximum), -jnp.inf)
    return jnp.where(finite, jnp.exp(shifted), 0.0 + 0.0j)


def _metrics(
    state: np.ndarray,
    hamiltonian,
    exact_state: np.ndarray,
    singlet_basis: np.ndarray,
    permutation_indices: np.ndarray,
) -> dict[str, float]:
    vector = np.array(state, dtype=np.complex128, copy=True)
    vector /= np.linalg.norm(vector)
    energy, variance = _energy_variance(vector, hamiltonian)
    singlet_coefficients = singlet_basis.conj().T @ vector
    singlet_weight = float(np.vdot(singlet_coefficients, singlet_coefficients).real)
    projected = singlet_basis @ singlet_coefficients
    projected = np.array(
        reynolds_project_state(
            jnp.asarray(projected), jnp.asarray(permutation_indices)
        ),
        copy=True,
    )
    projected_weight = float(np.vdot(projected, projected).real)
    if projected_weight > 1.0e-20:
        projected /= np.sqrt(projected_weight)
        projected_energy, projected_variance = _energy_variance(projected, hamiltonian)
    else:
        projected_energy = float("nan")
        projected_variance = float("nan")
    return {
        "energy_total": energy,
        "variance_total": variance,
        "ground_fidelity": float(abs(np.vdot(exact_state, vector)) ** 2),
        "singlet_weight": singlet_weight,
        "common_su2_a1_projection_weight": projected_weight,
        "common_su2_a1_projected_energy_total": projected_energy,
        "common_su2_a1_projected_variance_total": projected_variance,
    }


def _train_arm(
    *,
    arm_id: str,
    params: object,
    state_fn,
    h_rows: np.ndarray,
    h_columns: np.ndarray,
    h_values: np.ndarray,
    hamiltonian,
    exact_state: np.ndarray,
    singlet_basis: np.ndarray,
    permutation_indices: np.ndarray,
    steps: int,
    learning_rate: float,
) -> tuple[dict[str, object], np.ndarray]:
    rows = jnp.asarray(h_rows)
    columns = jnp.asarray(h_columns)
    values = jnp.asarray(h_values)

    def loss(candidate):
        vector = state_fn(candidate)
        action = coo_matvec(rows, columns, values, vector)
        return jnp.vdot(vector, action).real / jnp.vdot(vector, vector).real

    initial = float(loss(params))
    started = time.perf_counter()
    trained, optimizer = train_exact_adam(
        params,
        loss_fn=loss,
        steps=steps,
        learning_rate=learning_rate,
        record_every=max(1, steps // 5),
    )
    wall = time.perf_counter() - started
    state = np.asarray(state_fn(trained))
    state = state / np.linalg.norm(state)
    report = {
        "arm_id": arm_id,
        "parameter_count": _parameter_count(trained),
        "initial_energy_total": initial,
        "optimizer": optimizer,
        "wall_seconds": wall,
        "full_h_calls": int(steps + 2),
        "metrics": _metrics(
            state,
            hamiltonian,
            exact_state,
            singlet_basis,
            permutation_indices,
        ),
    }
    return report, state


def _bond_features(
    configurations: tuple[tuple[int, ...], ...],
    j1_bonds,
    j2_bonds,
) -> np.ndarray:
    spins = np.asarray(configurations, dtype=np.float64) - 0.5
    features = []
    for bonds in (j1_bonds, j2_bonds):
        value = np.zeros(len(spins), dtype=np.float64)
        for first, second, coupling in bonds:
            value += float(coupling) * spins[:, first] * spins[:, second]
        features.append(value)
    result = np.column_stack(features)
    result -= np.mean(result, axis=0, keepdims=True)
    scale = np.std(result, axis=0, keepdims=True)
    return result / np.where(scale > 1.0e-12, scale, 1.0)


def _optimized_jastrow_lanczos(
    seeds: np.ndarray,
    features: np.ndarray,
    permutation_indices: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    hamiltonian,
    exact_state: np.ndarray,
    singlet_basis: np.ndarray,
    *,
    steps: int,
) -> dict[str, object]:
    row = jnp.asarray(rows)
    column = jnp.asarray(columns)
    elements = jnp.asarray(values)
    feature_array = jnp.asarray(features)
    indices = jnp.asarray(permutation_indices)
    candidates = []
    for seed_index in range(seeds.shape[1]):
        seed = jnp.asarray(seeds[:, seed_index])
        params = jnp.zeros((2,), dtype=jnp.float64)

        def state(theta):
            raw = seed * jnp.exp(feature_array @ theta)
            return reynolds_project_state(raw, indices)

        def loss(theta):
            vector = state(theta)
            action = coo_matvec(row, column, elements, vector)
            return jnp.vdot(vector, action).real / jnp.vdot(vector, vector).real

        started = time.perf_counter()
        trained, optimizer = train_exact_adam(
            params,
            loss_fn=loss,
            steps=steps,
            learning_rate=2.0e-2,
            record_every=max(1, steps // 5),
        )
        jastrow = np.array(state(trained), copy=True)
        jastrow /= np.linalg.norm(jastrow)
        lanczos = _two_lanczos(jastrow, hamiltonian)
        energy, variance = _energy_variance(lanczos, hamiltonian)
        candidates.append(
            {
                "seed_index": seed_index,
                "jastrow_parameters": np.asarray(trained).tolist(),
                "optimizer": optimizer,
                "wall_seconds": time.perf_counter() - started,
                "state": lanczos,
                "energy_total": energy,
                "variance_total": variance,
            }
        )
    best = min(candidates, key=lambda row: row["energy_total"])
    state = np.asarray(best["state"])
    for candidate in candidates:
        candidate.pop("state")
    report = {
        "arm_id": "optimized_physical_jastrow_two_lanczos",
        "parameter_count": 5,
        "full_h_calls": int(steps + 4),
        "candidates": candidates,
        "metrics": _metrics(
            state,
            hamiltonian,
            exact_state,
            singlet_basis,
            permutation_indices,
        ),
    }
    return report


def _case(
    name: str,
    *,
    seeds: tuple[int, ...],
    steps: int,
) -> dict[str, object]:
    if name == "triangular_n12_j2_0125":
        n_sites = 12
        kind = "triangular_triangle"
        supports = TRIANGULAR_N12_SUPPORTS
        bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.125
        )
        j1_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.0
        )
        j2_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=0.0, j2=1.0
        )
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_N12_SUPERCELL)
        su2_config = triangular_sector_config(
            0, 0, embedding_dim=12, head_count=3, depth=1
        )
        spatial_config = FixedSzSpatialViTConfig(
            "triangular", 2, 6, 1, 3, 14, 2, 1, target_m2=0
        )
        physical_case = SMALL_CASES["hnf6x2t4"]
    elif name == "square_n16_j2_05":
        n_sites = 16
        kind = "square_plaquette"
        supports = SQUARE_N16_SUPPORTS
        bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=True)
        j1_bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=True)
        j2_bonds = square_j1j2_bonds(4, 4, j1=0.0, j2=1.0, periodic=True)
        group = square_gamma_a1_group(4)
        su2_config = square_sector_config(
            0, 0, embedding_dim=12, head_count=3, depth=1
        )
        spatial_config = FixedSzSpatialViTConfig(
            "square", 4, 4, 2, 2, 14, 2, 1, target_m2=0
        )
        physical_case = SMALL_CASES["square4"]
    else:
        raise ValueError(f"unknown matched-frontier case: {name}")
    h_rows, h_columns, h_values, configurations, hamiltonian = _coo_sparse(
        n_sites, bonds
    )
    exact_values, exact_vectors = eigsh(
        hamiltonian, k=1, which="SA", tol=1.0e-12
    )
    exact_state = np.asarray(exact_vectors[:, 0])
    exact_state /= np.linalg.norm(exact_state)
    singlet_basis, checked = four_cell_schur_sector_basis_matrix(
        kind, 0, target_m2=0, supports=supports
    )
    if checked != configurations:
        raise RuntimeError("matched fixed-M bases disagree")
    permutation_indices = physical_permutation_index_table(configurations, group)
    basis_array = jnp.asarray(np.asarray(configurations), dtype=jnp.int32)
    singlet_array = jnp.asarray(singlet_basis)
    indices = jnp.asarray(permutation_indices)
    actions = exhaustive_four_cell_sector_actions(su2_config)
    generic_config = FixedSzGenericViTConfig(
        n_sites, embedding_dim=12, head_count=3, depth=1, target_m2=0
    )
    runs = {arm_id: [] for arm_id in ARM_IDS}
    best_states = {}
    for seed in seeds:
        su2_params = init_su2_four_cell_sector_vit(
            jax.random.PRNGKey(seed), su2_config
        )

        def su2_state(params):
            coefficients = jnp.exp(
                su2_four_cell_sector_log_amplitudes(
                    params, actions, su2_config
                )
            )
            return reynolds_project_state(singlet_array @ coefficients, indices)

        spatial_params = init_fixed_sz_spatial_vit(
            jax.random.PRNGKey(seed), spatial_config
        )

        def spatial_state(params):
            raw = fixed_sz_spatial_vit_amplitudes(
                params, basis_array, spatial_config
            )
            return reynolds_project_state(raw, indices)

        generic_params = init_fixed_sz_generic_vit(
            jax.random.PRNGKey(seed), generic_config
        )

        def generic_state(params):
            return _stable_generic_amplitudes(
                params, basis_array, generic_config
            )

        specifications = (
            (
                ARM_IDS[0],
                su2_params,
                su2_state,
                1.0e-2,
                su2_four_cell_sector_vit_parameter_count(su2_params),
            ),
            (
                ARM_IDS[1],
                spatial_params,
                spatial_state,
                5.0e-3,
                fixed_sz_spatial_vit_parameter_count(spatial_params),
            ),
            (
                ARM_IDS[2],
                generic_params,
                generic_state,
                5.0e-3,
                _parameter_count(generic_params),
            ),
        )
        for arm_id, params, state_fn, learning_rate, expected_parameters in specifications:
            report, state = _train_arm(
                arm_id=arm_id,
                params=params,
                state_fn=state_fn,
                h_rows=h_rows,
                h_columns=h_columns,
                h_values=h_values,
                hamiltonian=hamiltonian,
                exact_state=exact_state,
                singlet_basis=singlet_basis,
                permutation_indices=permutation_indices,
                steps=steps,
                learning_rate=learning_rate,
            )
            if report["parameter_count"] != expected_parameters:
                raise RuntimeError("model parameter count changed during training")
            report["seed"] = int(seed)
            runs[arm_id].append(report)
            previous = best_states.get(arm_id)
            if previous is None or report["metrics"]["energy_total"] < previous[0]:
                best_states[arm_id] = (report["metrics"]["energy_total"], state)
    physical_seeds = physical_seed_amplitudes(
        np.asarray(configurations, dtype=np.int8), physical_case
    )
    jastrow = _optimized_jastrow_lanczos(
        physical_seeds,
        _bond_features(configurations, j1_bonds, j2_bonds),
        permutation_indices,
        h_rows,
        h_columns,
        h_values,
        hamiltonian,
        exact_state,
        singlet_basis,
        steps=steps,
    )
    exact_energy = float(exact_values[0])
    summary = {}
    for arm_id, reports in runs.items():
        errors = np.asarray(
            [row["metrics"]["energy_total"] - exact_energy for row in reports]
        )
        summary[arm_id] = {
            "median_energy_error_total": float(np.median(errors)),
            "best_energy_error_total": float(np.min(errors)),
            "median_variance_total": float(
                np.median([row["metrics"]["variance_total"] for row in reports])
            ),
            "median_wall_seconds": float(
                np.median([row["wall_seconds"] for row in reports])
            ),
            "all_seed_energy_errors_total": errors.tolist(),
        }
    summary[jastrow["arm_id"]] = {
        "median_energy_error_total": jastrow["metrics"]["energy_total"]
        - exact_energy,
        "best_energy_error_total": jastrow["metrics"]["energy_total"]
        - exact_energy,
        "median_variance_total": jastrow["metrics"]["variance_total"],
    }
    response = {}
    for arm_id, (_energy, state) in best_states.items():
        projected = singlet_basis @ (singlet_basis.conj().T @ state)
        projected = np.array(
            reynolds_project_state(jnp.asarray(projected), indices), copy=True
        )
        if np.linalg.norm(projected) <= 1.0e-14:
            continue
        projected /= np.linalg.norm(projected)
        if name == "triangular_n12_j2_0125":
            response[arm_id] = _triangular_response_panel(projected)
        else:
            response[arm_id] = _square_response_panel(
                projected, hamiltonian, configurations
            )
    return {
        "case_id": name,
        "n_sites": n_sites,
        "basis_size": len(configurations),
        "group_order": len(group),
        "steps_per_neural_arm": steps,
        "exact_energy_total": exact_energy,
        "runs": runs,
        "optimized_physical_jastrow_two_lanczos": jastrow,
        "summary": summary,
        "common_projected_response": response,
    }


def four_cell_matched_frontier(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 1000,
) -> dict[str, object]:
    """Run matched local neural and optimized physical-control comparisons."""

    cases = {
        case: _case(case, seeds=seeds, steps=steps)
        for case in ("triangular_n12_j2_0125", "square_n16_j2_05")
    }
    gates = {}
    for name, case in cases.items():
        summary = case["summary"]
        su2 = summary[ARM_IDS[0]]
        competitors = [summary[arm] for arm in (*ARM_IDS[1:], "optimized_physical_jastrow_two_lanczos")]
        gates[name] = {
            "su2_median_energy_beats_every_control": all(
                su2["median_energy_error_total"] < row["median_energy_error_total"]
                for row in competitors
            ),
            "su2_median_variance_beats_every_control": all(
                su2["median_variance_total"] < row["median_variance_total"]
                for row in competitors
            ),
            "su2_all_seeds_below_1e-5_total_error": max(
                su2["all_seed_energy_errors_total"]
            )
            < 1.0e-5,
        }
    return {
        "cases": cases,
        "promotion_gates": gates,
        "promotion_gates_passed": bool(
            all(all(row.values()) for row in gates.values())
        ),
        "claim_boundary": (
            "Matched exact-enumeration local comparison. It tests optimization "
            "and symmetry efficiency at fixed small size, not sampled scaling, "
            "literature-frontier energy, or QSL identification."
        ),
    }


__all__ = ["ARM_IDS", "four_cell_matched_frontier"]
