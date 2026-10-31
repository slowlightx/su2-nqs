"""Exact-small same-backbone spin symmetry ladder.

This production runner is intentionally self-contained at the project layer:
it imports exact-tested ``su2_nqs`` core modules, never another project.  Two
supports are distinguished in the output:

``fixed_m0``
    A common Hamiltonian support for the unrestricted parametrization, exact
    U(1), exact SU(2), and SU(2)+lattice rungs.  The unrestricted rung is *not*
    called symmetry-free because the objective is externally restricted.

``true_full``
    A genuinely unrestricted tree optimized over every ``2**N`` bit string.
    This is kept as a small calibration rung and has no lattice projection.

Stored scalar energy anchors are required.  The runner never diagonalizes a
Hamiltonian and never imports a direct fusion-path implementation.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    coo_matvec,
    physical_permutation_index_table,
    reynolds_project_state,
)
from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.frontier.energy_targets import observed_energy_to_target
from su2_nqs.frontier.artifacts import (
    array_sha256 as _array_sha256,
    canonical_sha256 as _canonical_sha256,
    coo_sha256 as _coo_sha256,
    pytree_sha256 as _pytree_sha256,
)
from su2_nqs.frontier.route_policy import require_active_production_method
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_gamma_a1_group,
    rectangular_translation_group,
    triangular_hnf_gamma_a1_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.physical_tree import (
    PhysicalSpinTreeConfig,
    init_physical_spin_tree,
    physical_spin_tree_amplitudes,
    physical_spin_tree_parameter_count,
)
from su2_nqs.models.spin_control_tree import (
    SpinFullTreeConfig,
    SpinU1TreeConfig,
    init_spin_full_tree,
    init_spin_u1_tree,
    spin_control_parameter_budget_bracket,
    spin_control_tree_parameter_count,
    spin_full_tree_amplitudes,
    spin_u1_tree_amplitudes,
)
from su2_nqs.operators.computational_graph_sparse import (
    heisenberg_computational_graph_sparse,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.adam import train_exact_adam

from projects.clebsch_tree_nqs.workflows.common import ExactWorkflow


jax.config.update("jax_enable_x64", True)

ALLOWED_BACKBONES = {"unconstrained", "u1", "su2"}
ALLOWED_PROJECTIONS = {"none", "translation", "gamma_a1"}


def _groups(geometry: str, rows: int, columns: int):
    identity = (tuple(range(rows * columns)),)
    if geometry == "square":
        return {
            "none": identity,
            "translation": rectangular_translation_group(rows, columns),
            "gamma_a1": rectangular_gamma_a1_group(rows, columns),
        }
    if geometry == "triangular":
        # HNF coordinates enumerate the column primitive direction first.
        supercell = ((columns, 0), (0, rows))
        result = {
            "none": identity,
            "translation": triangular_hnf_translation_group(supercell),
        }
        # A rectangular HNF generally retains only a subgroup of D6.  Until a
        # character-aware retained-little-group projector is registered, fail
        # closed instead of mislabelling a nonsquare torus as full Gamma.A1.
        if rows == columns:
            result["gamma_a1"] = triangular_hnf_gamma_a1_group(supercell)
        return result
    raise ValueError(f"unknown geometry: {geometry!r}")


def _bonds(geometry: str, rows: int, columns: int, ratio: float):
    if geometry == "square":
        return square_j1j2_bonds(rows, columns, j1=1.0, j2=ratio, periodic=True)
    if geometry == "triangular":
        return triangular_j1j2_bonds(rows, columns, j1=1.0, j2=ratio, periodic=True)
    raise ValueError(f"unknown geometry: {geometry!r}")


def _coordinates(rows: int, columns: int) -> np.ndarray:
    return np.asarray(
        [(row, column) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )


def _spin_correlations(
    state: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Return exact ``<S_i.S_j>`` from physical exchange matrix elements."""

    n_sites = len(basis[0])
    positions = {configuration: index for index, configuration in enumerate(basis)}
    result = np.empty((n_sites, n_sites), dtype=np.float64)
    probabilities = np.abs(state) ** 2
    for first in range(n_sites):
        result[first, first] = 0.75
        for second in range(first + 1, n_sites):
            swapped = []
            for configuration in basis:
                candidate = list(configuration)
                candidate[first], candidate[second] = candidate[second], candidate[first]
                swapped.append(positions[tuple(candidate)])
            permutation_expectation = np.vdot(state, state[np.asarray(swapped)]).real
            value = 0.5 * permutation_expectation - 0.25 * probabilities.sum()
            result[first, second] = result[second, first] = float(value)
    return result


def _structure_factor_grid(
    correlations: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    coordinates = _coordinates(rows, columns)
    grid = np.empty((rows, columns), dtype=np.float64)
    for ky in range(rows):
        for kx in range(columns):
            phases = np.exp(
                2.0j
                * np.pi
                * (
                    ky * coordinates[:, 0] / rows
                    + kx * coordinates[:, 1] / columns
                )
            )
            grid[ky, kx] = float(
                np.einsum("i,ij,j->", phases.conj(), correlations, phases).real
                / (rows * columns)
            )
    return grid


def _physical_spin_observables(
    state: np.ndarray,
    basis: tuple[tuple[int, ...], ...],
    *,
    geometry: str,
    rows: int,
    columns: int,
    bonds: tuple[tuple[int, int, float], ...],
) -> dict[str, object]:
    correlations = _spin_correlations(state, basis)
    structure = _structure_factor_grid(correlations, rows, columns)
    nonzero = structure.copy()
    nonzero[0, 0] = -np.inf
    peak = np.unravel_index(int(np.argmax(nonzero)), nonzero.shape)
    nearest = [correlations[first, second] for first, second, coupling in bonds if abs(coupling - 1.0) < 1.0e-12]
    result: dict[str, object] = {
        "spin_structure_factor_grid": structure.tolist(),
        "peak_nonzero_momentum_index": [int(peak[0]), int(peak[1])],
        "peak_nonzero_structure_factor": float(structure[peak]),
        "uniform_structure_factor": float(structure[0, 0]),
        "mean_unit_coupling_bond_correlation": None if not nearest else float(np.mean(nearest)),
    }
    if geometry == "square" and rows % 2 == 0 and columns % 2 == 0:
        result.update(
            {
                "neel_structure_factor": float(structure[rows // 2, columns // 2]),
                "stripe_structure_factor_max": float(
                    max(structure[rows // 2, 0], structure[0, columns // 2])
                ),
            }
        )
    if geometry == "triangular":
        m_values = []
        if rows % 2 == 0:
            m_values.append(structure[rows // 2, 0])
        if columns % 2 == 0:
            m_values.append(structure[0, columns // 2])
        if rows % 2 == 0 and columns % 2 == 0:
            m_values.append(structure[rows // 2, columns // 2])
        k_compatible = rows % 3 == 0 and columns % 3 == 0
        result["k_120_commensurate"] = k_compatible
        result["stripe_M_structure_factor_max"] = (
            None if not m_values else float(max(m_values))
        )
        if k_compatible:
            k_values = (
                structure[rows // 3, columns // 3],
                structure[2 * rows // 3, 2 * columns // 3],
            )
            result["order_120_K_structure_factor_max"] = float(max(k_values))
        else:
            result["order_120_K_structure_factor_max"] = None
    return result


def _prepare_case(spec: dict[str, object], support: str) -> dict[str, object]:
    if "exact_energy_anchor" not in spec:
        raise ValueError("a stored scalar exact_energy_anchor is required")
    geometry = str(spec["geometry"])
    rows, columns = int(spec["rows"]), int(spec["columns"])
    n_sites = rows * columns
    ratio = float(spec["j2_over_j1"])
    bonds = _bonds(geometry, rows, columns, ratio)
    groups = _groups(geometry, rows, columns)
    if support == "fixed_m0":
        hamiltonian, basis = heisenberg_sz_graph_sparse(
            n_sites, bonds, n_up=n_sites // 2
        )
        spin_squared, spin_basis = spin_squared_sz_sparse(
            n_sites, n_up=n_sites // 2
        )
        if tuple(basis) != tuple(spin_basis):
            raise RuntimeError("Hamiltonian and S2 basis orders differ")
    elif support == "true_full":
        hamiltonian, basis = heisenberg_computational_graph_sparse(n_sites, bonds)
        spin_squared = None
    else:
        raise ValueError(f"unknown support: {support!r}")
    hamiltonian = hamiltonian.tocsr()
    coo = hamiltonian.tocoo()
    tables = {
        name: physical_permutation_index_table(basis, group)
        for name, group in groups.items()
    }
    if rows > 1 and columns > 1:
        order = tuple(morton_order(rows, columns))
    else:
        order = tuple(range(n_sites))
    ordered_basis = np.asarray(basis, dtype=np.int32)[:, np.asarray(order)]
    return {
        "geometry": geometry,
        "rows": rows,
        "columns": columns,
        "n_sites": n_sites,
        "j2_over_j1": ratio,
        "bonds": bonds,
        "groups": groups,
        "tables": tables,
        "basis": tuple(tuple(int(bit) for bit in row) for row in basis),
        "ordered_basis": ordered_basis,
        "site_order": order,
        "hamiltonian": hamiltonian,
        "spin_squared": spin_squared,
        "h_rows": np.asarray(coo.row, dtype=np.int32),
        "h_columns": np.asarray(coo.col, dtype=np.int32),
        "h_values": np.asarray(coo.data, dtype=np.float64),
        "exact_energy": float(spec["exact_energy_anchor"]),
        "anchor_provenance": str(spec.get("anchor_provenance", "unspecified")),
        "support": support,
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
                "energy": float(spec["exact_energy_anchor"]),
                "provenance": str(spec.get("anchor_provenance", "unspecified")),
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
        raise ValueError(f"unknown backbone: {backbone!r}")
    n_sites = int(data["n_sites"])
    tree = FusionTree.balanced(n_sites)
    width = int(model["feature_dim"])
    scale = float(model["initialization_scale"])
    key = jax.random.PRNGKey(seed)
    if backbone == "su2":
        config = PhysicalSpinTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=width,
            maximum_intermediate_q=int(model["maximum_intermediate_q"]),
            initialization_scale=scale,
        )
        params = init_physical_spin_tree(key, config)
        return config, params, physical_spin_tree_amplitudes, physical_spin_tree_parameter_count(params)
    if backbone == "u1":
        config = SpinU1TreeConfig(
            tree,
            tuple(range(n_sites)),
            target_m2=0,
            feature_dim=width,
            maximum_intermediate_abs_m2=int(model["maximum_intermediate_q"]),
            initialization_scale=scale,
        )
        params = init_spin_u1_tree(key, config)
        return config, params, spin_u1_tree_amplitudes, spin_control_tree_parameter_count(params)
    config = SpinFullTreeConfig(
        tree,
        tuple(range(n_sites)),
        feature_dim=width,
        initialization_scale=scale,
    )
    params = init_spin_full_tree(key, config)
    return config, params, spin_full_tree_amplitudes, spin_control_tree_parameter_count(params)


def _project(raw: np.ndarray, data: dict[str, object], projection: str) -> np.ndarray:
    return np.mean(raw[np.asarray(data["tables"][projection], dtype=np.int32)], axis=0)


def _metrics(
    raw: np.ndarray,
    data: dict[str, object],
    *,
    projection: str,
) -> dict[str, object]:
    selected = _project(raw, data, projection)
    raw_norm = float(np.vdot(raw, raw).real)
    norm = float(np.vdot(selected, selected).real)
    if norm <= 1.0e-28:
        raise ValueError("projected state has zero numerical norm")
    state = selected / np.sqrt(norm)
    h_state = np.asarray(data["hamiltonian"] @ state)
    energy = float(np.vdot(state, h_state).real)
    residual = h_state - energy * state
    result: dict[str, object] = {
        "energy": energy,
        "energy_per_site": energy / int(data["n_sites"]),
        "energy_error": energy - float(data["exact_energy"]),
        "energy_error_per_site": (energy - float(data["exact_energy"])) / int(data["n_sites"]),
        "variance_per_site": float(np.vdot(residual, residual).real) / int(data["n_sites"]),
        "selected_weight_of_raw": norm / raw_norm,
        "projection_closure_error": float(np.linalg.norm(_project(state, data, projection) - state)),
    }
    if data["spin_squared"] is not None:
        s2_state = np.asarray(data["spin_squared"] @ state)
        result["spin_squared_expectation"] = float(np.vdot(state, s2_state).real)
        result["spin_squared_residual"] = float(np.linalg.norm(s2_state))
    else:
        m2 = np.asarray([sum(2 * bit - 1 for bit in row) for row in data["basis"]])
        probabilities = np.abs(state) ** 2
        sector_weights = {
            str(value): float(probabilities[m2 == value].sum())
            for value in sorted(set(int(value) for value in m2))
        }
        result["m2_sector_weights"] = sector_weights
        result["m0_weight"] = sector_weights.get("0", 0.0)
        result["total_sz_squared"] = float(np.sum(probabilities * (m2 / 2.0) ** 2))
    result["physical_observables"] = _physical_spin_observables(
        state,
        data["basis"],
        geometry=str(data["geometry"]),
        rows=int(data["rows"]),
        columns=int(data["columns"]),
        bonds=data["bonds"],
    )
    return result


def _run_trial(
    arm: dict[str, object],
    *,
    seed: int,
    spec: dict[str, object],
    data: dict[str, object],
) -> dict[str, object]:
    trial_started = time.perf_counter()
    backbone = str(arm["backbone"])
    projection = str(arm["projection"])
    if projection not in ALLOWED_PROJECTIONS:
        raise ValueError(f"unknown projection: {projection!r}")
    if data["support"] == "true_full" and (backbone != "unconstrained" or projection != "none"):
        raise ValueError("true_full support is reserved for the unconstrained/no-projector rung")
    model_spec = dict(spec["model"])
    model_spec.update(dict(arm.get("model", {})))
    config, params, amplitude_fn, parameter_count = _initialize(
        backbone, seed=seed, model=model_spec, data=data
    )
    basis = jnp.asarray(data["ordered_basis"], dtype=jnp.int32)
    indices = jnp.asarray(data["tables"][projection], dtype=jnp.int32)
    rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def raw_amplitudes(candidate):
        return amplitude_fn(candidate, basis, config)

    def amplitudes(candidate):
        return reynolds_project_state(raw_amplitudes(candidate), indices)

    def loss(candidate):
        state = amplitudes(candidate)
        action = coo_matvec(rows, columns, values, state)
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
            "support": str(data["support"]),
            "basis_sha256": str(data["basis_sha256"]),
            "site_order": list(data["site_order"]),
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
        "support": data["support"],
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


def _parameter_brackets(spec: dict[str, object], data: dict[str, object]) -> dict[str, object]:
    if data["support"] != "fixed_m0":
        return {}
    config, params, _, target = _initialize(
        "su2", seed=0, model=spec["model"], data=data
    )
    del config, params
    tree = FusionTree.balanced(int(data["n_sites"]))
    order = tuple(range(int(data["n_sites"])))
    result = {}
    for control in ("u1", "full"):
        bracket = spin_control_parameter_budget_bracket(
            control=control,
            tree=tree,
            site_order=order,
            target_parameter_count=target,
            target_m2=0,
            maximum_intermediate_abs_m2=int(spec["model"]["maximum_intermediate_q"]),
            maximum_feature_dim=int(spec["parameter_bracket_maximum_width"]),
        )
        result[control] = {
            "target_su2_parameter_count": target,
            "lower": None if bracket.lower is None else bracket.lower.__dict__,
            "upper": None if bracket.upper is None else bracket.upper.__dict__,
        }
    return result


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
            "support": str(data["support"]),
            "basis_sha256": str(data["basis_sha256"]),
            "bonds_sha256": str(data["bonds_sha256"]),
            "hamiltonian_coo_sha256": str(data["hamiltonian_coo_sha256"]),
            "group_sha256": str(data["group_sha256"][projection]),
            "anchor_sha256": str(data["anchor_sha256"]),
        }
    )


def _prepare_case_from_spec(case_spec: dict[str, object]) -> dict[str, object]:
    return _prepare_case(case_spec, str(case_spec["support"]))


def _case_record(
    case_spec: dict[str, object],
    data: dict[str, object],
    preparation_seconds: float,
    campaign_spec: dict[str, object],
) -> dict[str, object]:
    return {
        "case_id": str(case_spec["case_id"]),
        "geometry": data["geometry"],
        "rows": data["rows"],
        "columns": data["columns"],
        "n_sites": data["n_sites"],
        "j2_over_j1": data["j2_over_j1"],
        "support": data["support"],
        "basis_size": len(data["basis"]),
        "hamiltonian_nnz": int(data["hamiltonian"].nnz),
        "bond_count": len(data["bonds"]),
        "site_order": list(data["site_order"]),
        "group_orders": {
            name: len(group) for name, group in data["groups"].items()
        },
        "exact_energy_anchor": data["exact_energy"],
        "anchor_provenance": data["anchor_provenance"],
        "basis_sha256": data["basis_sha256"],
        "bonds_sha256": data["bonds_sha256"],
        "hamiltonian_coo_sha256": data["hamiltonian_coo_sha256"],
        "group_sha256": data["group_sha256"],
        "anchor_sha256": data["anchor_sha256"],
        "case_preparation_seconds": preparation_seconds,
        "k_120_commensurate": bool(
            int(data["rows"]) % 3 == 0 and int(data["columns"]) % 3 == 0
        ),
        "parameter_brackets": _parameter_brackets(campaign_spec, data),
    }


def _prepare_case_for_campaign(
    case_spec: dict[str, object], campaign_spec: dict[str, object]
) -> dict[str, object]:
    """Prepare a spin case through the shared campaign adapter."""

    del campaign_spec
    return _prepare_case_from_spec(case_spec)


TIMING_SCOPE = (
    "compile_plus_optimizer_seconds includes cold JIT and optimizer updates; "
    "trial_wall_seconds_excluding_case_preparation additionally includes "
    "initial-state hashing and final exact observables. Case preparation is "
    "reported separately. The lattice projector is a full-vector gather, so "
    "orbit work is explicit and timings are not scalable-VMC claims."
)


WORKFLOW = ExactWorkflow(
    domain="spin",
    source_entry_paths=(Path(__file__).resolve(),),
    prepare_case=_prepare_case_for_campaign,
    case_record=_case_record,
    run_trial=_run_trial,
    trial_identity=_trial_identity_sha256,
    timing_scope=TIMING_SCOPE,
)
