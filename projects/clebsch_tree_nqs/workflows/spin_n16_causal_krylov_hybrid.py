"""Rank-matched hybrid causal/Krylov complementarity gate at N=16."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.benchmarks.rvb_covering_oracle import lowest_ritz_state, two_lanczos_state
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.vmc.carrier_distillation import evaluate_log_amplitudes_in_batches
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity

from .spin_n16_causal_operator_oracle import _causal_features, _normalized
from .spin_n16_imaginary_time_compression import (
    _bonds,
    _dense_matrix,
    _source_artifact,
    _state_metrics,
)
from .spin_n16_krylov_causal_oracle import _state_logs
from .spin_scale_first_accelerator_slices import _shared_kwargs
from .spin_scale_first_three_arm_preflight import _validate as _validate_preflight


jax.config.update("jax_enable_x64", True)

TASK = "spin_n16_causal_krylov_hybrid"


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 causal/Krylov hybrid identity")
    comparison = spec.get("comparison")
    if not isinstance(comparison, dict):
        raise TypeError("comparison must be an object")
    required = {
        "shared_columns_through_h3": True,
        "physical_extra_column": "trained_two_lanczos_causal_state",
        "control_extra_column": "h4_psi",
        "require_equal_numerical_rank": True,
    }
    if any(comparison.get(key) != value for key, value in required.items()):
        raise ValueError("the rank-matched comparison contract changed")


def _orthogonal_fraction(column: np.ndarray, shared: np.ndarray) -> float:
    basis, singular, _ = np.linalg.svd(shared, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1.0e-12))
    projection = basis[:, :rank] @ (basis[:, :rank].conj().T @ column)
    return float(np.linalg.norm(column - projection) ** 2 / np.linalg.norm(column) ** 2)


def _run_case(case, source_case, *, shared, source, source_root, batch_size, case_index):
    geometry = str(case["geometry"])
    if source_case["geometry"] != geometry:
        raise ValueError("source Krylov-causal case order changed")
    j2 = float(case["j2_over_j1"])
    layout = rectangular_four_spin_layout(
        4, 4, triangular_coordinates=geometry == "triangular"
    )
    model = ManyCellSchurViTConfig(layout, target_q=0, geometry=geometry, **shared)
    labels = np.asarray(
        [state.labels for state in layout.tree.enumerate_states(0)], dtype=np.int32
    )
    template = init_many_cell_schur_vit(
        jax.random.PRNGKey(2026092780 + case_index), model
    )
    source_artifact = _source_artifact(source, geometry, source_root)
    base_params = load_parameter_artifact(source_artifact, template)
    raw_logs = evaluate_log_amplitudes_in_batches(
        base_params,
        labels,
        lambda candidate, batch: many_cell_schur_log_amplitudes(
            candidate, batch, model
        ),
        batch_size=batch_size,
    )
    raw = np.exp(raw_logs)
    raw /= np.linalg.norm(raw)
    physical_bonds = _bonds(geometry, j2)
    matrix = _dense_matrix(
        build_many_cell_tree_connectivity(labels, model, physical_bonds)
    )
    anchor = two_lanczos_state(raw, matrix).state
    anchor_logs = _state_logs(anchor)
    probabilities = np.abs(anchor) ** 2
    actions = compact_tree_action_sequences(layout.tree, 0, labels)
    local = (matrix @ anchor) / anchor
    physical_features, _, _, _ = _causal_features(
        layout, actions, probabilities, local
    )
    physical_theta = np.asarray(
        source_case["physical_oracle"]["parameters"], dtype=np.float64
    )
    physical_column = _normalized(anchor_logs + physical_features @ physical_theta)

    powers = [raw]
    for _ in range(4):
        powers.append(matrix @ powers[-1])
    shared_columns = np.column_stack(powers[:4])
    physical_hybrid = lowest_ritz_state(
        np.column_stack((shared_columns, physical_column)), matrix
    )
    matched_krylov = lowest_ritz_state(np.column_stack(powers), matrix)

    scramble_rows = []
    for source_scramble in source_case["scrambled_oracles"]:
        seed = int(source_scramble["scramble_seed"])
        permutation = np.random.default_rng(seed).permutation(16)
        scrambled_bonds = tuple(
            (int(permutation[first]), int(permutation[second]), float(coupling))
            for first, second, coupling in physical_bonds
        )
        scrambled_matrix = _dense_matrix(
            build_many_cell_tree_connectivity(labels, model, scrambled_bonds)
        )
        scrambled_local = (scrambled_matrix @ anchor) / anchor
        features, _, _, _ = _causal_features(
            layout, actions, probabilities, scrambled_local
        )
        theta = np.asarray(source_scramble["parameters"], dtype=np.float64)
        column = _normalized(anchor_logs + features @ theta)
        hybrid = lowest_ritz_state(
            np.column_stack((shared_columns, column)), matrix
        )
        scramble_rows.append(
            {
                "scramble_seed": seed,
                "extra_column_orthogonal_fraction": _orthogonal_fraction(
                    column, shared_columns
                ),
                "rank": int(hybrid.rank),
                **_state_metrics(hybrid.state, matrix),
            }
        )
    physical_metrics = _state_metrics(physical_hybrid.state, matrix)
    krylov_metrics = _state_metrics(matched_krylov.state, matrix)
    scramble_energies = [row["energy_per_site"] for row in scramble_rows]
    ranks = [physical_hybrid.rank, matched_krylov.rank] + [
        row["rank"] for row in scramble_rows
    ]
    tolerance = 1.0e-10
    passed = bool(
        len(set(ranks)) == 1
        and physical_metrics["energy_per_site"]
        < krylov_metrics["energy_per_site"] - tolerance
        and physical_metrics["variance_per_site"]
        < krylov_metrics["variance_per_site"] - tolerance
        and physical_metrics["energy_per_site"]
        < float(np.median(scramble_energies)) - tolerance
    )
    return {
        "case_id": str(case["case_id"]),
        "geometry": geometry,
        "j2_over_j1": j2,
        "singlet_dimension": int(len(labels)),
        "shared_krylov_rank": int(
            np.linalg.matrix_rank(shared_columns, tol=1.0e-10)
        ),
        "physical_extra_column_orthogonal_fraction": _orthogonal_fraction(
            physical_column, shared_columns
        ),
        "physical_hybrid": {
            "rank": int(physical_hybrid.rank),
            **physical_metrics,
        },
        "matched_degree_four_krylov": {
            "rank": int(matched_krylov.rank),
            **krylov_metrics,
        },
        "scrambled_hybrids": scramble_rows,
        "physical_minus_matched_krylov_energy_per_site": float(
            physical_metrics["energy_per_site"] - krylov_metrics["energy_per_site"]
        ),
        "physical_minus_matched_krylov_variance_per_site": float(
            physical_metrics["variance_per_site"] - krylov_metrics["variance_per_site"]
        ),
        "physical_below_median_scramble_per_site": float(
            np.median(scramble_energies) - physical_metrics["energy_per_site"]
        ),
        "all_compared_ranks": [int(rank) for rank in ranks],
        "passed": passed,
    }


def run_spin_n16_causal_krylov_hybrid(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Compare a causal extra column with the next matched Krylov column."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    preflight_path = _resolve(config_path, spec["source_preflight"])
    preflight = json.loads(preflight_path.read_text())
    _validate_preflight(preflight)
    source_path = _resolve(config_path, spec["source_initialization_result"])
    source = json.loads(source_path.read_text())
    source_result_path = _resolve(config_path, spec["source_krylov_causal_result"])
    source_result = json.loads(source_result_path.read_text())
    if source_result.get("task") != "spin_n16_krylov_causal_oracle":
        raise ValueError("invalid source Krylov-causal result")
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
            source_case,
            shared=shared,
            source=source,
            source_root=source_path.parent,
            batch_size=int(spec["controls"]["audit_batch_size"]),
            case_index=index,
        )
        for index, (case, source_case) in enumerate(
            zip(preflight["systems"], source_result["cases"], strict=True)
        )
    ]
    passed = bool(all(case["passed"] for case in cases))
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_krylov_causal_result_sha256": source_result["result_sha256"],
        "cases": cases,
        "all_hybrid_gates_passed": passed,
        "causal_large_size_channel_allowed": passed,
        "n36_training_allowed": False,
        "decision": (
            "retain_causal_column_and_design_sampled_additive_channel"
            if passed
            else "freeze_causal_energy_channel_keep_only_as_diagnostic"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": (
            "Exact N=16 rank-matched subspace test. It measures complementarity "
            "but is not a trainable, sampled, or large-size ansatz."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n16_causal_krylov_hybrid"]
