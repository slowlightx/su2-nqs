"""Physical diagnosis of the Hamiltonian residual in the N=16 Schur tree.

The diagnostic separates the exact local-energy residual by physical bond
class and by the autoregressive fusion-tree filtration.  It asks *where* the
missing imaginary-time direction first becomes visible, rather than trying a
new optimizer or another unconstrained neural head.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.causal_residual import (
    compact_tree_action_sequences,
    martingale_residual_decomposition,
    residual_concentration_summary,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.models.tree_gru import tree_decisions
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_imaginary_time_compression import (
    _bonds,
    _dense_matrix,
    _source_artifact,
)
from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate as _validate_preflight


jax.config.update("jax_enable_x64", True)

TASK = "spin_n16_hamiltonian_filtration"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 Hamiltonian-filtration identity")
    controls = spec.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    batch = controls.get("audit_batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("controls.audit_batch_size must be a positive integer")


def _bond_groups(layout, bonds, j2: float) -> dict[str, tuple[tuple[int, int, float], ...]]:
    site_cell = {
        int(site): cell
        for cell, support in enumerate(layout.physical_cells)
        for site in support
    }
    groups: dict[str, list[tuple[int, int, float]]] = {}
    for first, second, coupling in bonds:
        locality = "intra_cell" if site_cell[int(first)] == site_cell[int(second)] else "inter_cell"
        shell = "j2" if abs(float(coupling) - j2) < 1.0e-12 and abs(j2 - 1.0) > 1.0e-12 else "j1"
        groups.setdefault(f"{locality}_{shell}", []).append(
            (int(first), int(second), float(coupling))
        )
    return {name: tuple(rows) for name, rows in sorted(groups.items()) if rows}


def _decision_category(layout, node_index: int) -> str:
    if node_index in layout.intercell_indices:
        return "intercell_q"
    if node_index in layout.cell_root_indices:
        return "cell_q"
    return "cell_multiplicity_pair"


def _decorate_decomposition(decomposition: dict[str, object], layout) -> dict[str, object]:
    rows = []
    totals = {
        "intercell_q": 0.0,
        "cell_q": 0.0,
        "cell_multiplicity_pair": 0.0,
    }
    decisions = tree_decisions(layout.tree)
    for source, decision in zip(decomposition["decisions"], decisions, strict=True):
        row = dict(source)
        category = _decision_category(layout, int(decision.node_index))
        row.update(
            {
                "node_index": int(decision.node_index),
                "tree_depth": int(decision.depth),
                "left_leaves": int(decision.left_leaves),
                "right_leaves": int(decision.right_leaves),
                "physical_category": category,
            }
        )
        totals[category] += float(row["variance_contribution"])
        rows.append(row)
    variance = float(decomposition["variance"])
    payload = {key: value for key, value in decomposition.items() if key != "decisions"}
    payload["decisions"] = rows
    payload["physical_category_variance"] = totals
    payload["physical_category_fraction"] = {
        name: value / variance if variance > 0.0 else 0.0
        for name, value in totals.items()
    }
    return payload


def _run_case(
    case: dict[str, object],
    *,
    shared: dict[str, object],
    source: dict[str, object],
    source_root: Path,
    audit_batch_size: int,
    case_index: int,
) -> dict[str, object]:
    geometry = str(case["geometry"])
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    model = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **shared
    )
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092530 + case_index), model
    )
    artifact = _source_artifact(source, geometry, source_root)
    params = load_parameter_artifact(artifact, template)
    logs = evaluate_log_amplitudes_in_batches(
        params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=audit_batch_size,
    )
    state = np.exp(logs)
    state /= np.linalg.norm(state)
    probabilities = np.abs(state) ** 2
    actions = compact_tree_action_sequences(layout.tree, 0, labels)
    bonds = _bonds(geometry, j2)
    total_matrix = _dense_matrix(
        build_many_cell_tree_connectivity(labels, model, bonds)
    )
    local_total = (total_matrix @ state) / state
    total_energy = np.sum(probabilities * local_total)
    total_residual = local_total - total_energy
    total_decomposition = _decorate_decomposition(
        martingale_residual_decomposition(
            probabilities, local_total, actions
        ),
        layout,
    )
    concentration = residual_concentration_summary(
        probabilities, total_residual
    )

    groups = _bond_groups(layout, bonds, j2)
    group_rows = []
    residuals = []
    for name, group_bonds in groups.items():
        matrix = _dense_matrix(
            build_many_cell_tree_connectivity(labels, model, group_bonds)
        )
        local = (matrix @ state) / state
        energy = np.sum(probabilities * local)
        residual = local - energy
        residuals.append(residual)
        decomposition = _decorate_decomposition(
            martingale_residual_decomposition(probabilities, local, actions),
            layout,
        )
        group_rows.append(
            {
                "name": name,
                "bond_count": len(group_bonds),
                "energy_total_real": float(np.real(energy)),
                "energy_total_imag": float(np.imag(energy)),
                "residual_variance": float(
                    np.sum(probabilities * np.abs(residual) ** 2)
                ),
                "filtration": decomposition,
            }
        )
    residual_matrix = np.stack(residuals, axis=1)
    covariance = np.real(
        residual_matrix.conj().T @ (probabilities[:, None] * residual_matrix)
    )
    total_variance = float(np.sum(probabilities * np.abs(total_residual) ** 2))
    covariance_closure = float(abs(np.sum(covariance) - total_variance))
    if covariance_closure > 2.0e-10:
        raise RuntimeError("bond-class covariance failed to reconstruct variance")
    category = total_decomposition["physical_category_fraction"]
    leading_category = max(category, key=category.get)
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "source_parameter_artifact": str(artifact),
        "energy_total": float(np.real(total_energy)),
        "energy_per_site": float(np.real(total_energy) / 16.0),
        "variance_total": total_variance,
        "variance_per_site": total_variance / 16.0,
        "total_filtration": total_decomposition,
        "residual_concentration": concentration,
        "bond_class_order": [row["name"] for row in group_rows],
        "bond_classes": group_rows,
        "bond_class_residual_covariance": covariance.tolist(),
        "bond_class_covariance_closure_error": covariance_closure,
        "leading_filtration_category": leading_category,
        "architecture_implication": (
            "change_intercell_order_or_multiframe_partition_before_adding_capacity"
            if leading_category == "intercell_q"
            else "add_causal_local_recoupling_features_before_global_attention"
        ),
    }


def run_spin_n16_hamiltonian_filtration(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the exact bond-class and fusion-filtration diagnosis."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    if source.get("task") != "spin_scale_first_carrier_distillation":
        raise ValueError("invalid source initialization result")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    shared_raw = dict(preflight["shared_spatial_backbone"])
    shared_raw["direct_phase_initialization_scale"] = 0.0
    shared = _shared_kwargs(shared_raw)
    cases = [
        _run_case(
            case,
            shared=shared,
            source=source,
            source_root=source_path.parent,
            audit_batch_size=int(spec["controls"]["audit_batch_size"]),
            case_index=index,
        )
        for index, case in enumerate(preflight["systems"])
    ]
    implications = {case["architecture_implication"] for case in cases}
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_preflight_sha256": canonical_sha256(preflight),
        "source_initialization_result_sha256": source["result_sha256"],
        "cases": cases,
        "common_architecture_implication": (
            next(iter(implications)) if len(implications) == 1 else None
        ),
        "remote_training_allowed": False,
        "claim_boundary": (
            "Exact N=16 physical diagnosis of the sealed initialization. It "
            "selects an architecture mechanism but contains no trained-energy win."
        ),
        "runtime_identity": numerical_runtime_identity(),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_hamiltonian_filtration"]
