"""Additive physical-carrier test for the four-cell Schur Spatial-ViT."""

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
from ..models.su2_four_cell_sector_vit import (
    exhaustive_four_cell_sector_actions,
    init_su2_four_cell_sector_vit,
    square_sector_config,
    su2_four_cell_sector_log_amplitudes,
    triangular_sector_config,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    coo_matvec,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPERCELL,
    TRIANGULAR_N12_SUPPORTS,
    _coo_sparse,
)
from .xdiag_physical_seed_krylov import SMALL_CASES, physical_seed_amplitudes


jax.config.update("jax_enable_x64", True)


ARM_IDS = ("no_carrier", "physical_additive_carrier", "scrambled_additive_carrier")


def _case_data(name: str) -> dict[str, object]:
    if name == "triangular_n12_j2_0125":
        n_sites = 12
        kind = "triangular_triangle"
        supports = TRIANGULAR_N12_SUPPORTS
        bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.125
        )
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_N12_SUPERCELL)
        config = triangular_sector_config(
            0, 0, embedding_dim=12, head_count=3, depth=1
        )
        physical_case = SMALL_CASES["hnf6x2t4"]
        labels = ("120_tower_A1",)
    elif name == "square_n16_j2_05":
        n_sites = 16
        kind = "square_plaquette"
        supports = SQUARE_N16_SUPPORTS
        bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=True)
        group = square_gamma_a1_group(4)
        config = square_sector_config(
            0, 0, embedding_dim=12, head_count=3, depth=1
        )
        physical_case = SMALL_CASES["square4"]
        labels = ("pi0_DSL_A1", "stripe_tower_A1")
    else:
        raise ValueError(f"unknown additive-carrier case: {name}")
    rows, columns, values, configurations, hamiltonian = _coo_sparse(
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
    if configurations != checked:
        raise RuntimeError("four-cell and Hamiltonian bases disagree")
    indices = physical_permutation_index_table(configurations, group)
    physical = physical_seed_amplitudes(
        np.asarray(configurations, dtype=np.int8), physical_case
    )
    carrier = singlet_basis.conj().T @ physical
    carrier_norms = np.linalg.norm(carrier, axis=0)
    if np.any(carrier_norms <= 1.0e-14):
        raise RuntimeError("a physical carrier has zero singlet component")
    carrier /= carrier_norms[None, :]
    permutation = np.random.default_rng(20260921).permutation(carrier.shape[0])
    scrambled = carrier[permutation]
    return {
        "n_sites": n_sites,
        "config": config,
        "actions": exhaustive_four_cell_sector_actions(config),
        "basis": jnp.asarray(singlet_basis),
        "physical_carriers": jnp.asarray(carrier),
        "scrambled_carriers": jnp.asarray(scrambled),
        "carrier_labels": labels,
        "indices": jnp.asarray(indices),
        "indices_numpy": indices,
        "rows": jnp.asarray(rows),
        "columns": jnp.asarray(columns),
        "values": jnp.asarray(values),
        "hamiltonian": hamiltonian,
        "exact_energy_total": float(exact_values[0]),
        "exact_state": exact_state,
        "singlet_basis": singlet_basis,
    }


def _init_params(
    key: jax.Array,
    config,
    carrier_count: int,
    *,
    additive: bool,
) -> dict[str, object]:
    params: dict[str, object] = {
        "backbone": init_su2_four_cell_sector_vit(key, config)
    }
    if additive:
        params.update(
            {
                "carrier_logits": jnp.zeros((carrier_count,), dtype=jnp.float64),
                "carrier_phases": jnp.zeros((carrier_count,), dtype=jnp.float64),
                "residual_log_scale": jnp.asarray(np.log(0.1), dtype=jnp.float64),
                "residual_phase": jnp.asarray(0.0, dtype=jnp.float64),
            }
        )
    return params


def _state(params: object, data: dict[str, object], carriers) -> jax.Array:
    residual = jnp.exp(
        su2_four_cell_sector_log_amplitudes(
            params["backbone"], data["actions"], data["config"]
        )
    )
    if carriers is None:
        coefficients = residual
    else:
        weights = jax.nn.softmax(params["carrier_logits"])
        weights = weights * jnp.exp(1.0j * params["carrier_phases"])
        carrier = carriers @ weights
        residual_scale = jnp.exp(
            params["residual_log_scale"] + 1.0j * params["residual_phase"]
        )
        coefficients = carrier + residual_scale * residual
    physical = data["basis"] @ coefficients
    return reynolds_project_state(physical, data["indices"])


def additive_carrier_frontier(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 1000,
) -> dict[str, object]:
    """Compare no carrier, physical additive carriers, and a scrambled bank."""

    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            for arm_id, carriers in (
                (ARM_IDS[0], None),
                (ARM_IDS[1], data["physical_carriers"]),
                (ARM_IDS[2], data["scrambled_carriers"]),
            ):
                params = _init_params(
                    jax.random.PRNGKey(seed),
                    data["config"],
                    len(data["carrier_labels"]),
                    additive=carriers is not None,
                )

                def state_fn(candidate):
                    return _state(candidate, data, carriers)

                def loss(candidate):
                    state = state_fn(candidate)
                    action = coo_matvec(
                        data["rows"], data["columns"], data["values"], state
                    )
                    return jnp.vdot(state, action).real / jnp.vdot(state, state).real

                initial = float(loss(params))
                started = time.perf_counter()
                trained, optimizer = train_exact_adam(
                    params,
                    loss_fn=loss,
                    steps=steps,
                    learning_rate=1.0e-2,
                    record_every=max(1, steps // 5),
                )
                state = np.asarray(state_fn(trained))
                state = state / np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial,
                    "wall_seconds": time.perf_counter() - started,
                    "full_h_calls": int(steps + 2),
                    "optimizer": optimizer,
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
                if carriers is not None:
                    row["carrier_weights"] = np.asarray(
                        jax.nn.softmax(trained["carrier_logits"])
                    ).tolist()
                    row["residual_scale_magnitude"] = float(
                        np.exp(np.asarray(trained["residual_log_scale"]))
                    )
                runs[arm_id].append(row)
        exact = float(data["exact_energy_total"])
        summary = {}
        for arm_id, rows in runs.items():
            errors = np.asarray(
                [row["metrics"]["energy_total"] - exact for row in rows]
            )
            summary[arm_id] = {
                "median_energy_error_total": float(np.median(errors)),
                "maximum_energy_error_total": float(np.max(errors)),
                "all_seed_energy_errors_total": errors.tolist(),
                "median_variance_total": float(
                    np.median([row["metrics"]["variance_total"] for row in rows])
                ),
                "median_wall_seconds": float(
                    np.median([row["wall_seconds"] for row in rows])
                ),
            }
        physical = summary[ARM_IDS[1]]
        raw = summary[ARM_IDS[0]]
        scrambled = summary[ARM_IDS[2]]
        gates = {
            "physical_improves_median_energy_over_no_carrier": physical[
                "median_energy_error_total"
            ]
            < raw["median_energy_error_total"],
            "physical_improves_worst_seed_over_no_carrier": physical[
                "maximum_energy_error_total"
            ]
            < raw["maximum_energy_error_total"],
            "physical_beats_scrambled_median_energy": physical[
                "median_energy_error_total"
            ]
            < scrambled["median_energy_error_total"],
            "physical_all_seeds_below_1e-5_total_error": physical[
                "maximum_energy_error_total"
            ]
            < 1.0e-5,
        }
        cases[case_name] = {
            "n_sites": data["n_sites"],
            "carrier_labels": list(data["carrier_labels"]),
            "exact_energy_total": exact,
            "steps": steps,
            "runs": runs,
            "summary": summary,
            "gates": gates,
            "gates_passed": bool(all(gates.values())),
        }
    return {
        "cases": cases,
        "all_cases_passed": bool(
            all(case["gates_passed"] for case in cases.values())
        ),
        "claim_boundary": (
            "The triangular carrier is a 120-degree collective tower and the "
            "square bank contains pi/0-flux DSL and stripe towers. This tests "
            "physical versus scrambled additive initialization, not the full "
            "planned Dirac/paired-Z2 carrier bank."
        ),
    }


__all__ = ["ARM_IDS", "additive_carrier_frontier"]
