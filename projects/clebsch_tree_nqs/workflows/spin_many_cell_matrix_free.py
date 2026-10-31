"""Seal the N16 replay and fail-closed N36 many-cell production gate."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.benchmarks.many_cell_replay import (
    TRIANGULAR_N16_SUPERCELL,
    distill_many_cell_vit,
    replay_sealed_triangular_n16_state,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    sample_many_cell_schur_vit,
)
from su2_nqs.su2.many_cell_schur import (
    rectangular_four_spin_layout,
    validate_layout_state,
)
from su2_nqs.vmc.many_cell_tree import build_many_cell_tree_connectivity


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_many_cell_matrix_free"


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid many-cell matrix-free config identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the replay and distillation gates must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("the replay gate is fixed to triangular 4x4")
    for name in (
        "frozen_state_source",
        "many_cell_architecture",
        "distillation",
        "gates",
        "n36_preflight",
    ):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    architecture = spec["many_cell_architecture"]
    if architecture.get("cell") != "balanced_four_spin_complete_Schur":
        raise ValueError("the physical balanced four-spin Schur cell is required")
    if architecture.get("magnetic_indices_visible_to_vit") is not False:
        raise ValueError("the ViT may not see magnetic indices")
    if architecture.get("additive_carrier") is not True:
        raise ValueError("the additive carrier channel is required")
    if spec["n36_preflight"].get("run_training_only_if_all_gates_pass") is not True:
        raise ValueError("N36 production must fail closed")


def _parameter_shapes(params: object) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in leaf.shape) for leaf in jax.tree_util.tree_leaves(params))


def run_spin_many_cell_matrix_free(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the replay gate; do not launch N36 energy optimization."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")

    source = spec["frozen_state_source"]
    source_result_path = ROOT / source["result"]
    source_bundle_path = ROOT / source["bundle"]
    source_result = json.loads(source_result_path.read_text())
    if source_result.get("result_sha256") != source["result_sha256"]:
        raise RuntimeError("frozen result seal changed")
    if canonical_sha256(
        {key: value for key, value in source_result.items() if key != "result_sha256"}
    ) != source["result_sha256"]:
        raise RuntimeError("frozen result self-seal is invalid")
    if file_sha256(source_bundle_path) != source["bundle_file_sha256"]:
        raise RuntimeError("frozen state bundle file seal changed")
    with np.load(source_bundle_path) as bundle:
        sealed = np.asarray(bundle[source["state_key"]], dtype=np.complex128)
    sealed_energy = float(
        source_result["suite"]["summary"]
        ["operator_anchored_complete_multiplicity_vit"]["best_energy_total"]
    )
    replay, arrays = replay_sealed_triangular_n16_state(
        sealed,
        j2_over_j1=float(spec["j2_over_j1"]),
        sealed_energy=sealed_energy,
    )

    architecture = spec["many_cell_architecture"]
    small_layout = rectangular_four_spin_layout(4, 4)
    small_config = ManyCellSchurViTConfig(
        small_layout,
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
    )
    bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N16_SUPERCELL,
        j1=1.0,
        j2=float(spec["j2_over_j1"]),
    )
    workload = build_many_cell_tree_connectivity(
        arrays["labels"],
        small_config,
        bonds,
        maximum_width=int(replay["matrix_free_connectivity_maximum_width"]),
    )
    distillation = spec["distillation"]
    distilled, distilled_state = distill_many_cell_vit(
        arrays["labels"],
        arrays["balanced_coefficients"],
        workload,
        seed=int(distillation["seed"]),
        steps=int(distillation["steps"]),
        learning_rate=float(distillation["learning_rate"]),
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
    )
    arrays["distilled_coefficients"] = distilled_state

    large_layout = rectangular_four_spin_layout(6, 6)
    large_config = ManyCellSchurViTConfig(
        large_layout,
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
    )
    small_params = init_many_cell_schur_vit(jax.random.PRNGKey(0), small_config)
    large_params = init_many_cell_schur_vit(jax.random.PRNGKey(0), large_config)
    shape_transfer = _parameter_shapes(small_params) == _parameter_shapes(large_params)
    sample_count = int(spec["n36_preflight"]["sample_count"])
    n36_samples = np.asarray(
        sample_many_cell_schur_vit(
            large_params, jax.random.PRNGKey(1), sample_count, large_config
        )
    )
    for row in n36_samples:
        validate_layout_state(large_layout, row, target_q=0)
    decoded = large_layout.decode(n36_samples)
    arrays["n36_sampler_smoke_labels"] = n36_samples

    gates = spec["gates"]
    gate_results = {
        "n16_physical_amplitude_replay": bool(
            replay["physical_amplitude_max_abs_error"]
            <= float(gates["physical_amplitude_max_abs_error"])
            and replay["physical_amplitude_relative_l2_error"]
            <= float(gates["physical_amplitude_relative_l2_error"])
        ),
        "n16_matrix_free_energy": bool(
            replay["matrix_free_vs_sealed_energy_abs_error"]
            <= float(gates["matrix_free_energy_abs_error"])
            and replay["matrix_free_variance_total"]
            <= float(gates["matrix_free_variance"])
        ),
        "transferable_vit_distillation": bool(
            distilled["fidelity"] >= float(gates["distilled_fidelity"])
            and distilled["coefficient_max_abs_error"]
            <= float(gates["distilled_coefficient_max_abs_error"])
        ),
        "n16_n36_parameter_shape_transfer": bool(shape_transfer),
        "n36_autoregressive_sampler_smoke": bool(
            n36_samples.shape == (sample_count, 35)
            and np.all(n36_samples[:, -1] == 0)
            and decoded.cell_q.shape == (sample_count, 9)
            and decoded.intercell_q.shape == (sample_count, 8)
        ),
        "scalable_full_gamma_a1_projector": False,
    }
    production_authorized = all(gate_results.values())
    _atomic_npz(state_path, arrays)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "n16_exact_replay": replay,
        "transferable_vit_distillation": distilled,
        "n36_preflight": {
            "n_sites": 36,
            "n_cells": 9,
            "local_label_shape": list(decoded.cell_q.shape),
            "intercell_label_shape": list(decoded.intercell_q.shape),
            "parameter_shapes_match_n16": shape_transfer,
            "energy_training_started": False,
        },
        "gate_results": gate_results,
        "n36_gpu_production_authorized": production_authorized,
        "blocking_reasons": [
            name for name, passed in gate_results.items() if not passed
        ],
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "repeated_ed_performed": False,
        "runtime_identity": numerical_runtime_identity(),
        "interpretation": (
            "Exact many-cell representation and local recoupling are distinct "
            "from neural capacity. N36 production remains fail-closed when "
            "either transferable distillation or full Gamma-A1 is absent."
        ),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_many_cell_matrix_free"]
