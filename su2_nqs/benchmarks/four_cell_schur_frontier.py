"""Local N=12/N=16 physics gate for the four-cell Schur Spatial-ViT."""

from __future__ import annotations

from fractions import Fraction
from math import comb, sqrt
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    hnf_sites,
    triangular_hnf_j1j2_bonds,
)
from ..models.su2_four_cell_schur_vit import (
    SU2FourCellSchurViTConfig,
    exhaustive_four_cell_actions,
    init_su2_four_cell_schur_vit,
    square_four_cell_config,
    su2_four_cell_schur_log_amplitudes,
    su2_four_cell_schur_vit_parameter_count,
)
from ..operators.permutation import heisenberg_graph_fusion_matrix
from ..su2.clebsch_gordan import clebsch_gordan
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    coo_matvec,
    graph_hamiltonian_coo,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .four_cell_schur import four_cell_schur_basis_matrix
from .fusion_spatial_bridge import spin_raising_relative_norm
from .low_energy_subspace import matrix_free_chirality_action
from .physical_interpretation import hnf_oriented_triangles
from .sector_spectroscopy import fixed_m_coupled_basis
from .spectral_physics import compile_sector_spin_vector
from .triangular_hnf_space_group import (
    d6_character_projectors,
    momentum_projector_matrix,
    reciprocal_little_group,
)
from .triangular_monopole_branch import (
    bond_center_orientation_operators,
    little_group_character_projector,
)
from .xdiag_physical_seed_krylov import (
    SMALL_CASES,
    physical_seed_amplitudes,
)


jax.config.update("jax_enable_x64", True)


TRIANGULAR_N12_SUPERCELL = ((6, 4), (0, 2))
TRIANGULAR_N12_SUPPORTS = (
    (0, 1, 7),
    (2, 3, 6),
    (4, 10, 11),
    (5, 8, 9),
)
SQUARE_N16_SUPPORTS = (
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (8, 9, 12, 13),
    (10, 11, 14, 15),
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.complex128)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-14 or not np.isfinite(norm):
        raise ValueError("state must have nonzero finite norm")
    return result / norm


def _energy_variance(vector: np.ndarray, hamiltonian) -> tuple[float, float]:
    state = _normalize(vector)
    action = np.asarray(hamiltonian @ state)
    energy = float(np.vdot(state, action).real)
    residual = action - energy * state
    return energy, float(np.vdot(residual, residual).real)


def _two_lanczos(seed: np.ndarray, hamiltonian) -> np.ndarray:
    state = _normalize(seed)
    columns = np.column_stack((state, hamiltonian @ state, hamiltonian @ (hamiltonian @ state)))
    basis, singular, _ = np.linalg.svd(columns, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1.0e-12))
    basis = basis[:, :rank]
    projected = basis.conj().T @ (hamiltonian @ basis)
    values, vectors = np.linalg.eigh(0.5 * (projected + projected.conj().T))
    return _normalize(basis @ vectors[:, 0])


def _train_model(
    basis_matrix: np.ndarray,
    configurations: tuple[tuple[int, ...], ...],
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    group,
    config: SU2FourCellSchurViTConfig,
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
) -> tuple[list[dict[str, object]], np.ndarray, object]:
    matrix = jnp.asarray(np.asarray(basis_matrix))
    permutation_indices = jnp.asarray(
        physical_permutation_index_table(configurations, group)
    )
    row = jnp.asarray(rows)
    column = jnp.asarray(columns)
    elements = jnp.asarray(values)
    actions = exhaustive_four_cell_actions(config)

    def state(params):
        coefficients = jnp.exp(
            su2_four_cell_schur_log_amplitudes(params, actions, config)
        )
        return reynolds_project_state(matrix @ coefficients, permutation_indices)

    def loss(params):
        vector = state(params)
        action = coo_matvec(row, column, elements, vector)
        return jnp.vdot(vector, action).real / jnp.vdot(vector, vector).real

    reports = []
    best_state = None
    best_params = None
    for seed in seeds:
        params = init_su2_four_cell_schur_vit(jax.random.PRNGKey(seed), config)
        initial = float(loss(params))
        started = time.perf_counter()
        trained, optimizer = train_exact_adam(
            params,
            loss_fn=loss,
            steps=steps,
            learning_rate=learning_rate,
            record_every=max(1, steps // 4),
        )
        vector = _normalize(np.array(state(trained), copy=True))
        reports.append(
            {
                "seed": int(seed),
                "initial_energy_total": initial,
                "wall_seconds": time.perf_counter() - started,
                "optimizer": optimizer,
                "parameter_count": su2_four_cell_schur_vit_parameter_count(trained),
            }
        )
        if best_state is None:
            best_state, best_params = vector, trained
        else:
            old_action = np.zeros_like(best_state)
            new_action = np.zeros_like(vector)
            np.add.at(old_action, np.asarray(rows), np.asarray(values) * best_state[np.asarray(columns)])
            np.add.at(new_action, np.asarray(rows), np.asarray(values) * vector[np.asarray(columns)])
            if np.vdot(vector, new_action).real < np.vdot(best_state, old_action).real:
                best_state, best_params = vector, trained
    assert best_state is not None and best_params is not None
    return reports, best_state, best_params


def _coo_sparse(
    n_sites: int,
    bonds,
):
    rows, columns, values, basis = graph_hamiltonian_coo(n_sites, 0, bonds)
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()
    return rows, columns, values, basis, matrix


def _collective_neel_singlet(
    configurations: tuple[tuple[int, ...], ...],
    linear_size: int,
) -> np.ndarray:
    labels = np.asarray(
        [
            (row + column) % 2
            for row in range(linear_size)
            for column in range(linear_size)
        ]
    )
    count = int(np.count_nonzero(labels == 0))
    result = np.zeros(len(configurations), dtype=np.float64)
    for index, state in enumerate(configurations):
        bits = np.asarray(state)
        up = tuple(int(np.sum(bits[labels == label])) for label in (0, 1))
        m2 = tuple(2 * value - count for value in up)
        result[index] = clebsch_gordan(count, m2[0], count, m2[1], 0, 0) / sqrt(
            comb(count, up[0]) * comb(count, up[1])
        )
    return _normalize(result)


def _moment(source: np.ndarray, hamiltonian, energy: float) -> dict[str, float]:
    vector = np.asarray(source, dtype=np.complex128)
    m0 = float(np.vdot(vector, vector).real)
    m1 = float(np.vdot(vector, hamiltonian @ vector - energy * vector).real)
    return {"m0": m0, "m1": m1, "sma": m1 / m0 if m0 > 1.0e-20 else float("nan")}


def _triangular_response_panel(state_physical: np.ndarray) -> dict[str, object]:
    supercell = TRIANGULAR_N12_SUPERCELL
    coupled, configurations = fixed_m_coupled_basis(12, 0)
    if len(configurations) != len(state_physical):
        raise RuntimeError("triangular fixed-M basis changed")
    state = _normalize(coupled.conj().T @ state_physical)
    h0 = heisenberg_graph_fusion_matrix(
        12,
        triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.125),
        target_q=0,
    ).toarray()
    h2 = heisenberg_graph_fusion_matrix(
        12,
        triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.125),
        target_q=2,
    ).toarray()
    energy = float(np.vdot(state, h0 @ state).real)
    gamma = (Fraction(0), Fraction(0))
    p_gamma = momentum_projector_matrix(supercell, (0, 0), target_q=0)
    d6 = d6_character_projectors(supercell, target_q=0)
    p_e2 = 0.5 * (p_gamma @ d6["E2"] + (p_gamma @ d6["E2"]).conj().T)
    p_a2 = 0.5 * (p_gamma @ d6["A2"] + (p_gamma @ d6["A2"]).conj().T)
    orientation = np.asarray(
        (
            (1.0 / sqrt(2.0), 1.0 / sqrt(6.0)),
            (-1.0 / sqrt(2.0), 1.0 / sqrt(6.0)),
            (0.0, -2.0 / sqrt(6.0)),
        )
    )
    bonds = bond_center_orientation_operators(
        supercell, gamma, J1_POSITIVE, target_q=0
    )
    e2_sources = p_e2 @ (np.column_stack([operator @ state for operator in bonds]) @ orientation)
    e2 = _moment(e2_sources[:, 0], h0, energy)
    second = _moment(e2_sources[:, 1], h0, energy)
    e2 = {key: e2[key] + second[key] for key in ("m0", "m1")}
    e2["sma"] = e2["m1"] / e2["m0"]

    chirality = matrix_free_chirality_action(
        state, 12, hnf_oriented_triangles(supercell), target_q=0
    )
    a2 = _moment(p_a2 @ chirality, h0, energy)

    sites = hnf_sites(supercell)
    spin = {}
    for name, momentum in (
        ("K", (Fraction(1, 3), Fraction(1, 3))),
        ("M", (Fraction(1, 2), Fraction(0))),
    ):
        weights = np.asarray(
            [
                np.exp(
                    2.0j
                    * np.pi
                    * float(momentum[0] * site[0] + momentum[1] * site[1])
                )
                for site in sites
            ]
        )
        operator = compile_sector_spin_vector(weights, q_new=2, q_old=0).matrix / sqrt(3.0)
        source = np.asarray(operator @ state)
        characters = {
            element: 1
            for element in reciprocal_little_group(supercell, momentum)
        }
        projector = little_group_character_projector(
            supercell, momentum, characters, target_q=2
        )
        spin[name] = _moment(projector @ source, h2, energy)
    return {"energy_total": energy, "E2": e2, "A2": a2, "spin": spin}


def _square_response_panel(
    state: np.ndarray,
    hamiltonian,
    configurations: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    vector = _normalize(state)
    energy = float(np.vdot(vector, hamiltonian @ vector).real)
    spins = np.asarray(configurations, dtype=np.float64) - 0.5
    neel_weights = np.asarray(
        [(-1.0) ** (row + column) for row in range(4) for column in range(4)]
    )
    neel_source = (spins @ neel_weights) * vector
    neel = _moment(neel_source, hamiltonian, energy)
    neel["isotropic_m0_per_site"] = 3.0 * neel["m0"] / 16.0

    horizontal = []
    vertical = []
    for row in range(4):
        for column in range(4):
            site = row * 4 + column
            horizontal.append((site, row * 4 + (column + 1) % 4, (-1.0) ** column))
            vertical.append((site, ((row + 1) % 4) * 4 + column, (-1.0) ** row))
    sources = []
    for bonds in (horizontal, vertical):
        _r, _c, _v, _basis = graph_hamiltonian_coo(16, 0, bonds)
        operator = coo_matrix((_v, (_r, _c)), shape=hamiltonian.shape).tocsr()
        raw = operator @ vector
        raw = raw - np.vdot(vector, raw) * vector
        sources.append(raw)
    dimer_x = _moment(sources[0], hamiltonian, energy)
    dimer_y = _moment(sources[1], hamiltonian, energy)
    dimer = {
        "m0": dimer_x["m0"] + dimer_y["m0"],
        "m1": dimer_x["m1"] + dimer_y["m1"],
    }
    dimer["sma"] = dimer["m1"] / dimer["m0"]
    return {"energy_total": energy, "Neel": neel, "columnar_dimer": dimer}


def _relative_panel_errors(candidate: dict, exact: dict) -> dict[str, float]:
    output = {}

    def visit(prefix: str, left, right) -> None:
        if isinstance(right, dict):
            for key in right:
                if key in left:
                    visit(f"{prefix}.{key}" if prefix else key, left[key], right[key])
        elif isinstance(right, (float, int)) and np.isfinite(float(right)):
            scale = max(abs(float(right)), 1.0e-10)
            output[prefix] = abs(float(left) - float(right)) / scale

    visit("", candidate, exact)
    output.pop("energy_total", None)
    return output


def _tangent_sector_audit(
    params: object,
    config: SU2FourCellSchurViTConfig,
    basis_matrix: np.ndarray,
    configurations: tuple[tuple[int, ...], ...],
    group,
) -> dict[str, float]:
    actions = exhaustive_four_cell_actions(config)
    direction = jax.tree_util.tree_map(jnp.ones_like, params)
    coefficients, tangent = jax.jvp(
        lambda candidate: jnp.exp(
            su2_four_cell_schur_log_amplitudes(candidate, actions, config)
        ),
        (params,),
        (direction,),
    )
    indices = physical_permutation_index_table(configurations, group)
    state = np.asarray(
        reynolds_project_state(
            jnp.asarray(basis_matrix) @ coefficients, jnp.asarray(indices)
        )
    )
    tangent_state = np.asarray(
        reynolds_project_state(
            jnp.asarray(basis_matrix) @ tangent, jnp.asarray(indices)
        )
    )
    projected_again = np.asarray(
        reynolds_project_state(jnp.asarray(state), jnp.asarray(indices))
    )
    tangent_again = np.asarray(
        reynolds_project_state(jnp.asarray(tangent_state), jnp.asarray(indices))
    )
    return {
        "state_spin_raising_relative_norm": spin_raising_relative_norm(
            state, configurations
        ),
        "tangent_spin_raising_relative_norm": spin_raising_relative_norm(
            tangent_state, configurations
        ),
        "state_spatial_projector_closure": float(
            np.linalg.norm(projected_again - state) / np.linalg.norm(state)
        ),
        "tangent_spatial_projector_closure": float(
            np.linalg.norm(tangent_again - tangent_state)
            / np.linalg.norm(tangent_state)
        ),
        "tangent_norm": float(np.linalg.norm(tangent_state)),
    }


def four_cell_schur_frontier(
    *,
    triangular_seeds: tuple[int, ...] = (0, 1, 2),
    square_seeds: tuple[int, ...] = (0, 1),
    triangular_steps: int = 1200,
    square_steps: int = 1000,
    learning_rate: float = 1.0e-2,
    embedding_dim: int = 12,
    head_count: int = 3,
    depth: int = 1,
) -> dict[str, object]:
    """Run the combined N=12 triangular and N=16 square local gate."""

    # Triangular N=12 full-D6 torus.
    tri_basis, tri_configs = four_cell_schur_basis_matrix(
        "triangular_triangle", supports=TRIANGULAR_N12_SUPPORTS
    )
    tri_bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.125
    )
    tr, tc, tv, checked_tri_configs, tri_h = _coo_sparse(12, tri_bonds)
    if tri_configs != checked_tri_configs:
        raise RuntimeError("triangular fixed-M bases disagree")
    tri_values, tri_vectors = eigsh(tri_h, k=2, which="SA", tol=1.0e-12)
    tri_order = np.argsort(tri_values)
    tri_values = tri_values[tri_order]
    tri_vectors = tri_vectors[:, tri_order]
    tri_exact = _normalize(tri_vectors[:, 0])
    tri_seed = physical_seed_amplitudes(
        np.asarray(tri_configs, dtype=np.int8), SMALL_CASES["hnf6x2t4"]
    )[:, 0]
    tri_control = _two_lanczos(tri_seed, tri_h)
    tri_config = SU2FourCellSchurViTConfig(
        "triangular_triangle",
        ((0.5, 0.25), (2.5, 0.25), (4.5, 0.75), (5.5, 0.75)),
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
    )
    tri_runs, tri_model, tri_params = _train_model(
        tri_basis,
        tri_configs,
        tr,
        tc,
        tv,
        triangular_hnf_gamma_a1_group(TRIANGULAR_N12_SUPERCELL),
        tri_config,
        seeds=triangular_seeds,
        steps=triangular_steps,
        learning_rate=learning_rate,
    )

    # Square N=16 4x4 torus.
    square_basis, square_configs = four_cell_schur_basis_matrix(
        "square_plaquette", supports=SQUARE_N16_SUPPORTS
    )
    square_bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=True)
    sr, sc, sv, checked_square_configs, square_h = _coo_sparse(16, square_bonds)
    if square_configs != checked_square_configs:
        raise RuntimeError("square fixed-M bases disagree")
    square_values, square_vectors = eigsh(square_h, k=2, which="SA", tol=1.0e-11)
    square_order = np.argsort(square_values)
    square_values = square_values[square_order]
    square_vectors = square_vectors[:, square_order]
    square_exact = _normalize(square_vectors[:, 0])
    square_seed = _collective_neel_singlet(square_configs, 4)
    square_control = _two_lanczos(square_seed, square_h)
    square_config = square_four_cell_config(
        embedding_dim=embedding_dim, head_count=head_count, depth=depth
    )
    square_runs, square_model, square_params = _train_model(
        square_basis,
        square_configs,
        sr,
        sc,
        sv,
        square_gamma_a1_group(4),
        square_config,
        seeds=square_seeds,
        steps=square_steps,
        learning_rate=learning_rate,
    )

    def summarize(
        exact_state, control_state, model_state, hamiltonian, configs, response_fn
    ):
        exact_energy, exact_variance = _energy_variance(exact_state, hamiltonian)
        control_energy, control_variance = _energy_variance(control_state, hamiltonian)
        model_energy, model_variance = _energy_variance(model_state, hamiltonian)
        exact_panel = response_fn(exact_state)
        control_panel = response_fn(control_state)
        model_panel = response_fn(model_state)
        control_errors = _relative_panel_errors(control_panel, exact_panel)
        model_errors = _relative_panel_errors(model_panel, exact_panel)
        return {
            "exact": {"energy_total": exact_energy, "variance_total": exact_variance},
            "two_lanczos_control": {
                "energy_total": control_energy,
                "variance_total": control_variance,
                "response": control_panel,
                "response_relative_errors": control_errors,
                "median_response_relative_error": float(np.median(tuple(control_errors.values()))),
            },
            "schur_spatial_vit": {
                "energy_total": model_energy,
                "variance_total": model_variance,
                "energy_error_total": model_energy - exact_energy,
                "ground_fidelity": float(abs(np.vdot(exact_state, model_state)) ** 2),
                "spin_raising_relative_norm": spin_raising_relative_norm(model_state, configs),
                "response": model_panel,
                "response_relative_errors": model_errors,
                "median_response_relative_error": float(np.median(tuple(model_errors.values()))),
            },
            "exact_response": exact_panel,
        }

    tri_summary = summarize(
        tri_exact,
        tri_control,
        tri_model,
        tri_h,
        tri_configs,
        _triangular_response_panel,
    )
    square_summary = summarize(
        square_exact,
        square_control,
        square_model,
        square_h,
        square_configs,
        lambda state: _square_response_panel(state, square_h, square_configs),
    )
    tri_summary["runs"] = tri_runs
    square_summary["runs"] = square_runs
    tri_summary["sector_tangent_audit"] = _tangent_sector_audit(
        tri_params,
        tri_config,
        tri_basis,
        tri_configs,
        triangular_hnf_gamma_a1_group(TRIANGULAR_N12_SUPERCELL),
    )
    square_summary["sector_tangent_audit"] = _tangent_sector_audit(
        square_params,
        square_config,
        square_basis,
        square_configs,
        square_gamma_a1_group(4),
    )
    tri_summary["singlet_dimension"] = 132
    tri_summary["first_excitation_gap_total"] = float(
        tri_values[1] - tri_values[0]
    )
    square_summary["singlet_dimension"] = 1430
    square_summary["first_excitation_gap_total"] = float(
        square_values[1] - square_values[0]
    )
    tri_summary["basis_gram_error"] = float(
        np.linalg.norm(tri_basis.T @ tri_basis - np.eye(132))
    )
    # A full 1430x1430 Gram product is unnecessary; state/tangent singlet
    # checks and analytic CG orthogonality are the registered N=16 gate.
    square_summary["basis_column_norm_max_error"] = float(
        np.max(np.abs(np.sum(square_basis * square_basis, axis=0) - 1.0))
    )
    square_summary["full_basis_gram_materialized"] = False
    exact_invariants = {
        "triangular_state_and_tangent_singlets": max(
            tri_summary["sector_tangent_audit"]["state_spin_raising_relative_norm"],
            tri_summary["sector_tangent_audit"]["tangent_spin_raising_relative_norm"],
        ) < 1.0e-11,
        "square_state_and_tangent_singlets": max(
            square_summary["sector_tangent_audit"]["state_spin_raising_relative_norm"],
            square_summary["sector_tangent_audit"]["tangent_spin_raising_relative_norm"],
        ) < 1.0e-11,
        "triangular_spatial_projector_closed": max(
            tri_summary["sector_tangent_audit"]["state_spatial_projector_closure"],
            tri_summary["sector_tangent_audit"]["tangent_spatial_projector_closure"],
        ) < 1.0e-11,
        "square_spatial_projector_closed": max(
            square_summary["sector_tangent_audit"]["state_spatial_projector_closure"],
            square_summary["sector_tangent_audit"]["tangent_spatial_projector_closure"],
        ) < 1.0e-11,
        "triangular_complete_basis": tri_summary["basis_gram_error"] < 1.0e-10,
        "square_basis_columns_normalized": square_summary[
            "basis_column_norm_max_error"
        ] < 1.0e-10,
    }
    trainability_diagnostics = {
        "triangular_energy_error_below_1e-10": abs(
            tri_summary["schur_spatial_vit"]["energy_error_total"]
        ) < 1.0e-10,
        "triangular_response_median_below_1e-8": tri_summary[
            "schur_spatial_vit"
        ]["median_response_relative_error"] < 1.0e-8,
        "square_energy_error_below_1e-5": abs(
            square_summary["schur_spatial_vit"]["energy_error_total"]
        ) < 1.0e-5,
        "square_variance_below_1e-4": square_summary["schur_spatial_vit"][
            "variance_total"
        ] < 1.0e-4,
        "square_response_median_below_1e-3": square_summary[
            "schur_spatial_vit"
        ]["median_response_relative_error"] < 1.0e-3,
    }
    return {
        "triangular_n12_j2_0125": tri_summary,
        "square_n16_j2_05": square_summary,
        "exact_invariants": exact_invariants,
        "exact_invariants_passed": bool(all(exact_invariants.values())),
        "trainability_diagnostics": trainability_diagnostics,
        "trainability_diagnostics_passed": bool(
            all(trainability_diagnostics.values())
        ),
        "claim_boundary": (
            "Exact-small complete-space architecture/response gate. The controls are "
            "physical two-Lanczos references but not the strongest published ViT/VMC "
            "baselines; no scalable variational-win claim is licensed."
        ),
        "next_gate": (
            "freeze a sampled four-cell continuation and compare against fixed-Sz and "
            "symmetry-unaware ViTs at matched samples, updates, full-H calls, and time"
        ),
    }


__all__ = [
    "SQUARE_N16_SUPPORTS",
    "TRIANGULAR_N12_SUPERCELL",
    "TRIANGULAR_N12_SUPPORTS",
    "four_cell_schur_frontier",
]
