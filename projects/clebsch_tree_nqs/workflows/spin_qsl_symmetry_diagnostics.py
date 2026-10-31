"""Train the local QSL frontier and audit symmetry-resolved observables."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.qsl_symmetry_diagnostics import (
    triangular_4x4_exact_ground,
    triangular_qsl_symmetry_diagnostics,
)
from su2_nqs.benchmarks.qsl_three_arm import (
    TRIANGULAR_4X4_J2_0125_EXACT_ENERGY,
    triangular_qsl_three_arm_state_panel,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_qsl_symmetry_diagnostics"


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
        raise ValueError("invalid QSL symmetry-diagnostic config identity")
    if abs(float(spec.get("j2_over_j1", -1.0)) - 0.125) > 1.0e-14:
        raise ValueError("the preregistered local diagnostic is J2/J1=0.125")
    optimization = spec.get("optimization")
    if not isinstance(optimization, dict):
        raise TypeError("optimization must be an object")
    for key in ("jastrow_seeds", "vit_seeds", "operator_seeds"):
        values = optimization.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"optimization.{key} must be a nonempty list")
        seeds = tuple(int(value) for value in values)
        if any(value < 0 for value in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"optimization.{key} must contain distinct seeds")
    for key in ("vit_steps", "operator_steps"):
        if int(optimization.get(key, 0)) < 1:
            raise ValueError(f"optimization.{key} must be positive")


def run_spin_qsl_symmetry_diagnostics(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the exact-small symmetry/physics corroboration gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    optimization = spec["optimization"]
    data, states, metadata = triangular_qsl_three_arm_state_panel(
        float(spec["j2_over_j1"]),
        jastrow_seeds=tuple(int(value) for value in optimization["jastrow_seeds"]),
        vit_seeds=tuple(int(value) for value in optimization["vit_seeds"]),
        operator_seeds=tuple(
            int(value) for value in optimization["operator_seeds"]
        ),
        vit_steps=int(optimization["vit_steps"]),
        operator_steps=int(optimization["operator_steps"]),
    )
    exact_energy, exact_state, residual = triangular_4x4_exact_ground(data)
    if abs(exact_energy - TRIANGULAR_4X4_J2_0125_EXACT_ENERGY) > 2.0e-10:
        raise RuntimeError("recomputed exact energy disagrees with the stored anchor")
    diagnostic = triangular_qsl_symmetry_diagnostics(data, states, exact_state)
    arrays = {
        "basis": np.asarray(data.configurations, dtype=np.int8),
        "exact_ground": exact_state,
        **{name: np.asarray(state) for name, state in states.items()},
    }
    _atomic_npz(state_path, arrays)
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "exact_ground": {
            "energy_total": exact_energy,
            "residual_norm": residual,
            "stored_anchor_total": TRIANGULAR_4X4_J2_0125_EXACT_ENERGY,
        },
        "selected_arm_metadata": metadata,
        "diagnostics": diagnostic,
        "state_bundle": str(state_path.resolve().relative_to(ROOT.resolve())),
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(
        payload["source_manifest"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_qsl_symmetry_diagnostics"]
