"""Run the sealed triangular-N16 intertwiner-loop Spatial-ViT gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.intertwiner_loop_n16 import (
    intertwiner_loop_n16_comparison,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n16_intertwiner_loop_vit"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N16 intertwiner-loop identity")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the architecture comparison must be preregistered")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("the first exact gate is triangular 4x4")
    if float(spec.get("j2_over_j1", np.nan)) != 0.125:
        raise ValueError("the registered coupling is J2/J1=0.125")
    for name in ("source", "architecture", "optimization"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    architecture = spec["architecture"]
    if architecture.get("representation") != (
        "coherent_pair_singlet_intertwiner_frames_with_shared_reduced_spatial_vit"
    ):
        raise ValueError("architecture representation changed")
    for name in ("embedding_dim", "head_count", "depth", "mlp_ratio"):
        if isinstance(architecture.get(name), bool) or int(
            architecture.get(name, 0)
        ) < 1:
            raise ValueError(f"architecture.{name} must be positive")
    if int(architecture["embedding_dim"]) % int(architecture["head_count"]):
        raise ValueError("embedding_dim must be divisible by head_count")
    optimization = spec["optimization"]
    seeds = optimization.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("optimization.seeds must be nonempty and distinct")
    if any(isinstance(seed, bool) or int(seed) < 0 for seed in seeds):
        raise ValueError("optimization seeds must be nonnegative integers")
    for name in ("steps", "scramble_seed"):
        if isinstance(optimization.get(name), bool) or int(
            optimization.get(name, -1)
        ) < (1 if name == "steps" else 0):
            raise ValueError(f"optimization.{name} is invalid")
    for name in ("learning_rate", "initial_residual_scale"):
        if float(optimization.get(name, 0.0)) <= 0.0:
            raise ValueError(f"optimization.{name} must be positive")


def _load_source(spec: dict[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(spec["result"])
    bundle_path = ROOT / str(spec["bundle"])
    result = json.loads(result_path.read_text())
    seal = str(spec["result_sha256"])
    if result.get("result_sha256") != seal or canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError("source result seal changed")
    if file_sha256(bundle_path) != str(spec["bundle_sha256"]):
        raise RuntimeError("source bundle seal changed")
    return result, bundle_path


def _load_audit_parent(spec: object) -> dict[str, object] | None:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError("audit_of must be an object")
    path = ROOT / str(spec["result"])
    result = json.loads(path.read_text())
    seal = str(spec["result_sha256"])
    if result.get("result_sha256") != seal or canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError("audit-parent result seal changed")
    return result


def run_spin_n16_intertwiner_loop_vit(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the exact-small state-manifold comparison."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    source_result, bundle_path = _load_source(spec["source"])
    audit_parent = _load_audit_parent(spec.get("audit_of"))
    with np.load(bundle_path, allow_pickle=False) as arrays:
        report = intertwiner_loop_n16_comparison(
            coefficient_a1_basis=np.asarray(arrays["coefficient_a1_basis"]),
            physical_a1_basis=np.asarray(arrays["physical_a1_basis"]),
            carrier=np.asarray(arrays["coupling_0.125_carrier_a1"]),
            two_lanczos=np.asarray(arrays["coupling_0.125_two_lanczos_a1"]),
            hamiltonian=np.asarray(arrays["coupling_0.125_h_a1"]),
            seeds=tuple(int(value) for value in spec["optimization"]["seeds"]),
            steps=int(spec["optimization"]["steps"]),
            learning_rate=float(spec["optimization"]["learning_rate"]),
            initial_residual_scale=float(
                spec["optimization"]["initial_residual_scale"]
            ),
            scramble_seed=int(spec["optimization"]["scramble_seed"]),
            embedding_dim=int(spec["architecture"]["embedding_dim"]),
            head_count=int(spec["architecture"]["head_count"]),
            depth=int(spec["architecture"]["depth"]),
            mlp_ratio=int(spec["architecture"]["mlp_ratio"]),
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_result_sha256": source_result["result_sha256"],
        "audit_parent_result_sha256": (
            None if audit_parent is None else audit_parent["result_sha256"]
        ),
        **report,
        "repeated_many_body_ed_performed": False,
        "runtime_identity": numerical_runtime_identity(),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    payload = run_spin_n16_intertwiner_loop_vit(
        arguments.config, arguments.output, overwrite=arguments.overwrite
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["TASK", "_validate", "run_spin_n16_intertwiner_loop_vit"]
