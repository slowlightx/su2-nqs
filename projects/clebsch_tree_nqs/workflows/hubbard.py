"""Exact-small HubbardCG symmetry ladder on physical Fock configurations.

The two backbones differ only in their internal symmetry labels:

- ``ne_sz`` fixes physical charge and doubled magnetization;
- ``su2`` fixes physical charge and exact total spin.

Signed fermionic lattice projection is then applied as an independent rung.
This file imports no historical project runner and performs no eigensolve.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import coo_matvec
from su2_nqs.benchmarks.hubbard_cg_exact import hubbard_spin_squared_sector_sparse
from su2_nqs.benchmarks.hubbard_space_group import (
    fermionic_permutation_index_sign_tables,
    fermionic_reynolds_project_state,
)
from su2_nqs.frontier.energy_targets import observed_energy_to_target
from su2_nqs.frontier.artifacts import (
    array_sha256 as _array_sha256,
    canonical_sha256 as _canonical_sha256,
    coo_sha256 as _coo_sha256,
    pytree_sha256 as _pytree_sha256,
)
from su2_nqs.frontier.route_policy import require_active_production_method
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_gamma_a1_group,
    rectangular_translation_group,
)
from su2_nqs.lattices.square import square_bonds
from su2_nqs.models.physical_tree import (
    MatchedHubbardAbelianTreeConfig,
    PhysicalHubbardTreeConfig,
    init_matched_hubbard_abelian_tree,
    init_physical_hubbard_tree,
    matched_hubbard_abelian_tree_amplitude,
    matched_hubbard_abelian_tree_parameter_count,
    physical_hubbard_tree_amplitude,
    physical_hubbard_tree_parameter_count,
)
from su2_nqs.operators.hubbard_fock import hubbard_fock_sector_sparse
from su2_nqs.su2.hubbard_cg_irrep import DOUBLON, token_quantum_number_tables
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.common import ExactWorkflow


jax.config.update("jax_enable_x64", True)

ALLOWED_BACKBONES = {"ne_sz", "su2"}
ALLOWED_PROJECTIONS = {"none", "translation", "gamma_a1"}


def _groups(rows: int, columns: int, periodic: bool):
    identity = (tuple(range(rows * columns)),)
    if not periodic:
        return {"none": identity}
    return {
        "none": identity,
        "translation": rectangular_translation_group(rows, columns),
        "gamma_a1": rectangular_gamma_a1_group(rows, columns),
    }


def _space_group_metadata(
    groups: dict[str, tuple[tuple[int, ...], ...]],
    *,
    rows: int,
    columns: int,
    periodic: bool,
) -> dict[str, object]:
    """Name the actually retained finite-graph group without implying C4v."""

    if not periodic:
        return {
            "space_group_label": "open-boundary identity only",
            "translation_order": 1,
            "full_retained_group_order": 1,
            "retained_point_quotient_order": 1,
            "character_label": "trivial",
        }
    translation_order = len(groups["translation"])
    full_order = len(groups["gamma_a1"])
    if full_order % translation_order:
        raise RuntimeError("retained space-group order is not divisible by translations")
    return {
        "space_group_label": f"signed rectangular-G{full_order} on {rows}x{columns}",
        "translation_order": translation_order,
        "full_retained_group_order": full_order,
        "retained_point_quotient_order": full_order // translation_order,
        "character_label": "trivial signed fermionic representation at k=0",
    }


def _prepare_case(case: dict[str, object]) -> dict[str, object]:
    if "exact_energy_anchor" not in case:
        raise ValueError("a stored scalar exact_energy_anchor is required")
    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    periodic = bool(case["periodic"])
    n_electrons = int(case["n_electrons"])
    target_m2 = int(case["target_m2"])
    interaction = float(case["interaction"])
    bonds = square_bonds(rows, columns, periodic=periodic)
    hamiltonian, basis = hubbard_fock_sector_sparse(
        n_sites,
        n_electrons,
        target_m2,
        bonds,
        interaction=interaction,
    )
    hamiltonian = hamiltonian.tocsr()
    coo = hamiltonian.tocoo()
    spin_squared, spin_basis = hubbard_spin_squared_sector_sparse(
        n_sites, n_electrons, total_m2=target_m2
    )
    if tuple(basis) != tuple(spin_basis):
        raise RuntimeError("Hamiltonian and S2 basis orders differ")
    groups = _groups(rows, columns, periodic)
    tables = {
        name: fermionic_permutation_index_sign_tables(basis, group)
        for name, group in groups.items()
    }
    return {
        "rows": rows,
        "columns": columns,
        "n_sites": n_sites,
        "periodic": periodic,
        "n_electrons": n_electrons,
        "target_q": int(case["target_q"]),
        "target_m2": target_m2,
        "interaction": interaction,
        "bonds": bonds,
        "basis": tuple(tuple(int(token) for token in row) for row in basis),
        "basis_array": np.asarray(basis, dtype=np.int32),
        "hamiltonian": hamiltonian,
        "spin_squared": spin_squared,
        "h_rows": np.asarray(coo.row, dtype=np.int32),
        "h_columns": np.asarray(coo.col, dtype=np.int32),
        "h_values": np.asarray(coo.data, dtype=np.float64),
        "groups": groups,
        "space_group_metadata": _space_group_metadata(
            groups, rows=rows, columns=columns, periodic=periodic
        ),
        "tables": tables,
        "exact_energy": float(case["exact_energy_anchor"]),
        "anchor_provenance": str(case.get("anchor_provenance", "unspecified")),
        "basis_sha256": _array_sha256(np.asarray(basis, dtype=np.int8)),
        "bonds_sha256": _canonical_sha256(bonds),
        "hamiltonian_coo_sha256": _coo_sha256(
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
            np.asarray(coo.data, dtype=np.float64),
        ),
        "group_sha256": {
            name: _canonical_sha256(group) for name, group in groups.items()
        },
        "anchor_sha256": _canonical_sha256(
            {
                "energy": float(case["exact_energy_anchor"]),
                "provenance": str(case.get("anchor_provenance", "unspecified")),
            }
        ),
    }


def _initialize(
    backbone: str,
    *,
    seed: int,
    model: dict[str, object],
    data: dict[str, object],
) -> tuple[object, object, Callable, int]:
    require_active_production_method(backbone)
    if backbone not in ALLOWED_BACKBONES:
        raise ValueError(f"unknown Hubbard backbone: {backbone!r}")
    width = int(model["feature_dim"])
    scale = float(model["initialization_scale"])
    key = jax.random.PRNGKey(seed)
    if backbone == "su2":
        config = PhysicalHubbardTreeConfig(
            int(data["n_sites"]),
            int(data["n_electrons"]),
            int(data["target_q"]),
            int(data["target_m2"]),
            width,
            scale,
        )
        params = init_physical_hubbard_tree(key, config)
        return config, params, physical_hubbard_tree_amplitude, physical_hubbard_tree_parameter_count(params)
    config = MatchedHubbardAbelianTreeConfig(
        int(data["n_sites"]),
        int(data["n_electrons"]),
        int(data["target_m2"]),
        width,
        scale,
    )
    params = init_matched_hubbard_abelian_tree(key, config)
    return config, params, matched_hubbard_abelian_tree_amplitude, matched_hubbard_abelian_tree_parameter_count(params)


def _project(raw: np.ndarray, data: dict[str, object], projection: str) -> np.ndarray:
    indices, signs = data["tables"][projection]
    return np.mean(signs * raw[indices], axis=0)


def _density_structure_factors(
    state: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    columns: int,
) -> dict[str, object]:
    charge_table, _, m2_table, _ = token_quantum_number_tables()
    tokens = np.asarray(basis, dtype=np.int32)
    charge = np.asarray(charge_table, dtype=np.float64)[tokens]
    sz = 0.5 * np.asarray(m2_table, dtype=np.float64)[tokens]
    probabilities = np.abs(state) ** 2
    # Use the conserved filling, not a site-resolved variational mean.  The
    # latter would subtract away a static charge-density pattern in a state
    # that has not yet been translation projected.
    filling = float(np.sum(probabilities[:, None] * charge) / charge.shape[1])
    charge_centered = charge - filling
    charge_corr = np.einsum("b,bi,bj->ij", probabilities, charge_centered, charge_centered)
    sz_corr = np.einsum("b,bi,bj->ij", probabilities, sz, sz)
    coordinates = np.asarray(
        [(row, column) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    charge_grid = np.empty((rows, columns), dtype=np.float64)
    spin_zz_grid = np.empty((rows, columns), dtype=np.float64)
    for ky in range(rows):
        for kx in range(columns):
            phases = np.exp(
                2.0j * np.pi * (
                    ky * coordinates[:, 0] / rows
                    + kx * coordinates[:, 1] / columns
                )
            )
            charge_grid[ky, kx] = float(
                np.einsum("i,ij,j->", phases.conj(), charge_corr, phases).real
                / (rows * columns)
            )
            spin_zz_grid[ky, kx] = float(
                np.einsum("i,ij,j->", phases.conj(), sz_corr, phases).real
                / (rows * columns)
            )
    result: dict[str, object] = {
        "charge_structure_factor_grid": charge_grid.tolist(),
        "longitudinal_spin_structure_factor_grid": spin_zz_grid.tolist(),
        "density_per_site": np.sum(probabilities[:, None] * charge, axis=0).tolist(),
    }
    if columns % 2 == 0:
        result["charge_pi_along_chain"] = float(charge_grid[0, columns // 2])
        result["spin_zz_pi_along_chain"] = float(spin_zz_grid[0, columns // 2])
    if rows % 2 == 0 and columns % 2 == 0:
        result["charge_pi_pi"] = float(charge_grid[rows // 2, columns // 2])
        result["spin_zz_pi_pi"] = float(spin_zz_grid[rows // 2, columns // 2])
    return result


def _metrics(raw: np.ndarray, data: dict[str, object], *, projection: str) -> dict[str, object]:
    selected = _project(raw, data, projection)
    raw_norm = float(np.vdot(raw, raw).real)
    norm = float(np.vdot(selected, selected).real)
    if norm <= 1.0e-28:
        raise ValueError("projected state has zero numerical norm")
    state = selected / np.sqrt(norm)
    h_state = np.asarray(data["hamiltonian"] @ state)
    energy = float(np.vdot(state, h_state).real)
    residual = h_state - energy * state
    s2_state = np.asarray(data["spin_squared"] @ state)
    doublons = np.asarray(
        [sum(token == DOUBLON for token in configuration) for configuration in data["basis"]],
        dtype=np.float64,
    )
    doublon_total = float(np.abs(state) ** 2 @ doublons)
    potential = float(data["interaction"]) * doublon_total
    return {
        "energy": energy,
        "energy_per_site": energy / int(data["n_sites"]),
        "energy_error_per_site": (energy - float(data["exact_energy"])) / int(data["n_sites"]),
        "variance_per_site": float(np.vdot(residual, residual).real) / int(data["n_sites"]),
        "spin_squared_expectation": float(np.vdot(state, s2_state).real),
        "singlet_residual_norm": float(np.linalg.norm(s2_state)),
        "double_occupancy_per_site": doublon_total / int(data["n_sites"]),
        "potential_energy_per_site": potential / int(data["n_sites"]),
        "kinetic_energy_per_site": (energy - potential) / int(data["n_sites"]),
        "selected_weight_of_raw": norm / raw_norm,
        "projection_closure_error": float(np.linalg.norm(_project(state, data, projection) - state)),
        "physical_observables": _density_structure_factors(
            state,
            data["basis"],
            rows=int(data["rows"]),
            columns=int(data["columns"]),
        ),
    }


def _run_trial(
    arm: dict[str, object],
    *,
    seed: int,
    spec: dict[str, object],
    data: dict[str, object],
) -> dict[str, object]:
    trial_started = time.perf_counter()
    backbone, projection = str(arm["backbone"]), str(arm["projection"])
    if projection not in ALLOWED_PROJECTIONS or projection not in data["groups"]:
        raise ValueError(f"projection {projection!r} is not valid for this case")
    model_spec = dict(spec["model"])
    model_spec.update(dict(arm.get("model", {})))
    config, params, amplitude_fn, parameter_count = _initialize(
        backbone, seed=seed, model=model_spec, data=data
    )
    basis = jnp.asarray(data["basis_array"], dtype=jnp.int32)
    indices, signs = data["tables"][projection]
    indices = jnp.asarray(indices, dtype=jnp.int32)
    signs = jnp.asarray(signs, dtype=jnp.float64)
    h_rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    h_columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    h_values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def raw_amplitudes(candidate):
        return jax.vmap(lambda row: amplitude_fn(candidate, row, config))(basis)

    def amplitudes(candidate):
        return fermionic_reynolds_project_state(raw_amplitudes(candidate), indices, signs)

    def loss(candidate):
        state = amplitudes(candidate)
        action = coo_matvec(h_rows, h_columns, h_values, state)
        return jnp.vdot(state, action).real / jnp.vdot(state, state).real

    optimization = spec["optimization"]
    initial_params_sha256 = _pytree_sha256(params)
    initial_raw = np.asarray(raw_amplitudes(params), dtype=np.float64)
    initial_raw_sha256 = _array_sha256(initial_raw)
    initialization_pair_id = _canonical_sha256(
        {
            "backbone": backbone,
            "seed": int(seed),
            "model": model_spec,
            "basis_sha256": str(data["basis_sha256"]),
            "n_electrons": int(data["n_electrons"]),
            "target_q": int(data["target_q"]),
            "target_m2": int(data["target_m2"]),
        }
    )
    optimizer_started = time.perf_counter()
    params, training = train_exact_adam(
        params,
        loss_fn=loss,
        steps=int(optimization["steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    compile_plus_optimizer_seconds = time.perf_counter() - optimizer_started
    observables_started = time.perf_counter()
    raw = np.asarray(raw_amplitudes(params), dtype=np.float64)
    metrics = _metrics(raw, data, projection=projection)
    final_observable_seconds = time.perf_counter() - observables_started
    trial_wall_seconds = time.perf_counter() - trial_started
    group_order = len(data["groups"][projection])
    return {
        "arm_id": str(arm["arm_id"]),
        "backbone": backbone,
        "projection": projection,
        "seed": seed,
        "feature_dim": int(model_spec["feature_dim"]),
        "parameter_count": parameter_count,
        "group_order": group_order,
        "initial_params_sha256": initial_params_sha256,
        "initial_raw_amplitudes_sha256": initial_raw_sha256,
        "initialization_pair_id": initialization_pair_id,
        "training": training,
        "compile_plus_optimizer_seconds": compile_plus_optimizer_seconds,
        "final_observable_seconds": final_observable_seconds,
        "trial_wall_seconds_excluding_case_preparation": trial_wall_seconds,
        "metrics": metrics,
        "work_ledger_per_loss": {
            "raw_amplitude_configurations": len(data["basis"]),
            "projector_indexed_gathers": group_order * len(data["basis"]),
            "hamiltonian_coo_entries": len(data["h_values"]),
        },
        "energy_to_target": observed_energy_to_target(
            training,
            final_energy=float(metrics["energy"]),
            exact_energy=float(data["exact_energy"]),
            n_sites=int(data["n_sites"]),
            total_steps=int(optimization["steps"]),
            targets_per_site=optimization["energy_error_per_site_targets"],
        ),
    }


def _trial_identity_sha256(
    *,
    config_sha256: str,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
    case_id: str,
    arm: dict[str, object],
    seed: int,
    data: dict[str, object],
) -> str:
    """Bind a checkpoint to its exact protocol and physical sparse problem."""

    projection = str(arm["projection"])
    return _canonical_sha256(
        {
            "config_sha256": config_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "runtime_identity_sha256": runtime_identity_sha256,
            "case_id": case_id,
            "arm_id": str(arm["arm_id"]),
            "backbone": str(arm["backbone"]),
            "projection": projection,
            "seed": int(seed),
            "basis_sha256": str(data["basis_sha256"]),
            "bonds_sha256": str(data["bonds_sha256"]),
            "hamiltonian_coo_sha256": str(data["hamiltonian_coo_sha256"]),
            "group_sha256": str(data["group_sha256"][projection]),
            "anchor_sha256": str(data["anchor_sha256"]),
        }
    )


def _prepare_case_for_campaign(
    case_spec: dict[str, object], campaign_spec: dict[str, object]
) -> dict[str, object]:
    del campaign_spec
    return _prepare_case(case_spec)


def _case_record(
    case_spec: dict[str, object],
    data: dict[str, object],
    preparation_seconds: float,
    campaign_spec: dict[str, object],
) -> dict[str, object]:
    del campaign_spec
    return {
        "case_id": str(case_spec["case_id"]),
        "rows": data["rows"],
        "columns": data["columns"],
        "n_sites": data["n_sites"],
        "periodic": data["periodic"],
        "n_electrons": data["n_electrons"],
        "interaction": data["interaction"],
        "basis_size": len(data["basis"]),
        "hamiltonian_nnz": int(data["hamiltonian"].nnz),
        "bond_count": len(data["bonds"]),
        "group_orders": {
            name: len(group) for name, group in data["groups"].items()
        },
        "space_group_metadata": data["space_group_metadata"],
        "exact_energy_anchor": data["exact_energy"],
        "anchor_provenance": data["anchor_provenance"],
        "basis_sha256": data["basis_sha256"],
        "bonds_sha256": data["bonds_sha256"],
        "hamiltonian_coo_sha256": data["hamiltonian_coo_sha256"],
        "group_sha256": data["group_sha256"],
        "anchor_sha256": data["anchor_sha256"],
        "case_preparation_seconds": preparation_seconds,
    }


TIMING_SCOPE = (
    "compile_plus_optimizer_seconds includes cold JIT and optimizer updates; "
    "trial_wall_seconds_excluding_case_preparation additionally includes "
    "initial-state hashing and final exact observables. Case preparation is "
    "reported separately. Signed orbit work is a full-vector gather and is "
    "separately counted; timings are not scalable-VMC claims."
)


WORKFLOW = ExactWorkflow(
    domain="hubbard",
    source_entry_paths=(Path(__file__).resolve(),),
    prepare_case=_prepare_case_for_campaign,
    case_record=_case_record,
    run_trial=_run_trial,
    trial_identity=_trial_identity_sha256,
    timing_scope=TIMING_SCOPE,
    route_metadata={"charge_only_control_available": False},
)
