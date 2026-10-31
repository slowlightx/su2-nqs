"""Immutable exact-small work--variance benchmark for singlet forces."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.singlet_force_efficiency import (
    singlet_force_efficiency_benchmark,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_sampled_adam_center_only import sampled_vmc_value_and_grad
from su2_nqs.vmc.singlet_diagonal_force import (
    sampled_singlet_diagonal_value_and_grad,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime_workflow
from projects.clebsch_tree_nqs.workflows.spin_sampled_training_v2 import (
    _compiled_kernels,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_born_singlet_force_efficiency"


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return f"external/{resolved.name}"


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("singlet-force benchmark requires schema_version=2 and spin")
    if spec.get("task") != TASK:
        raise ValueError(f"config task must be {TASK}")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        if case.get("geometry") not in {"square", "triangular"}:
            raise ValueError("geometry must be square or triangular")
        if (case.get("rows"), case.get("columns")) != (2, 4):
            raise ValueError("the current exact-small benchmark is fixed to 2x4")
        if case.get("periodic") is not False:
            raise ValueError("the N=8 estimator benchmark uses an open 2x4 graph")
    sampling = spec.get("sampling")
    if not isinstance(sampling, dict):
        raise TypeError("sampling must be an object")
    profiles = spec.get("runtime_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("runtime_profiles must be a nonempty list")
    for profile in profiles:
        if not isinstance(profile, dict):
            raise TypeError("every runtime profile must be an object")
        if profile.get("geometry") not in {"square", "triangular"}:
            raise ValueError("runtime profile geometry must be square or triangular")
        if (profile.get("rows"), profile.get("columns")) != (6, 6):
            raise ValueError("the local runtime profiles are fixed to 6x6")


def _block(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _runtime_profile(profile: dict[str, object]) -> dict[str, object]:
    """Measure one random-initialization 6x6 SU(2) force kernel pair."""

    geometry = str(profile["geometry"])
    rows = columns = 6
    ratio = float(profile["j2_over_j1"])
    if geometry == "square":
        bonds = square_j1j2_bonds(rows, columns, j1=1.0, j2=ratio, periodic=True)
        group = rectangular_translation_group(rows, columns)
    else:
        bonds = triangular_j1j2_bonds(
            rows, columns, j1=1.0, j2=ratio, periodic=True
        )
        group = triangular_hnf_translation_group(((columns, 0), (0, rows)))
    projector = PhysicalSpatialProjector(
        group,
        orbit_chunk_size=int(profile["orbit_chunk_size"]),
        rematerialize_orbit=bool(profile["rematerialize_orbit"]),
    )
    arm = {
        "arm_id": f"{geometry}_su2_D{int(profile['feature_dim'])}_translation",
        "backbone": "su2",
        "seed": int(profile["model_seed"]),
        "model": {
            "feature_dim": int(profile["feature_dim"]),
            "maximum_intermediate_q": int(profile["maximum_intermediate_q"]),
            "initialization_scale": float(profile["initialization_scale"]),
        },
    }
    runtime = runtime_workflow._initialize_arm(
        arm, rows=rows, columns=columns, projector=projector
    )
    batch_size = int(profile["batch_size"])
    states = initialize_spin_exchange_chains(
        jax.random.PRNGKey(int(profile["sample_seed"])),
        n_chains=batch_size,
        n_sites=rows * columns,
        n_up=rows * columns // 2,
    )
    padded_width = len(bonds) + 1
    connected, elements, valid, widths = padded_spin_connectivity(
        np.asarray(states), bonds, padded_width=padded_width
    )
    batch = (
        jnp.asarray(states),
        jnp.asarray(connected),
        jnp.asarray(elements),
        jnp.asarray(valid),
    )
    _, _, hpsi_fn, center_fn = _compiled_kernels(runtime, bonds)

    def full_call():
        return sampled_vmc_value_and_grad(
            runtime.params, (batch,), hpsi_fn=hpsi_fn, center_fn=center_fn
        )

    def diagonal_call():
        return sampled_singlet_diagonal_value_and_grad(
            runtime.params,
            (batch[0],),
            center_fn=lambda candidate, samples: center_fn(
                candidate,
                (
                    samples,
                    batch[1],
                    batch[2],
                    batch[3],
                ),
            ),
            bonds=bonds,
        )

    timings: dict[str, list[float]] = {"full": [], "diagonal": []}
    results: dict[str, object] = {}
    for name, call in (("full", full_call), ("diagonal", diagonal_call)):
        started = time.perf_counter()
        value = call()
        _block(value)
        cold = time.perf_counter() - started
        warm = []
        for _ in range(int(profile["warm_repeats"])):
            started = time.perf_counter()
            value = call()
            _block(value)
            warm.append(time.perf_counter() - started)
        timings[name] = warm
        results[name] = {
            "cold_compile_and_evaluate_seconds": cold,
            "warm_seconds": warm,
            "warm_median_seconds": float(np.median(warm)),
            "energy": float(value.energy),
            "estimator_variance": float(value.variance),
            "gradient_norm": float(value.gradient_norm),
        }
    full_median = float(results["full"]["warm_median_seconds"])
    diagonal_median = float(results["diagonal"]["warm_median_seconds"])
    return {
        "profile_id": str(profile["profile_id"]),
        "geometry": geometry,
        "rows": rows,
        "columns": columns,
        "j2_over_j1": ratio,
        "batch_size": batch_size,
        "bond_count": len(bonds),
        "padded_width": padded_width,
        "actual_width_minimum": int(np.min(widths)),
        "actual_width_maximum": int(np.max(widths)),
        "translation_group_order": len(group),
        "parameter_count": runtime.parameter_count,
        "full_amplitude_work_per_sample": float(np.mean(widths) + 1.0),
        "diagonal_amplitude_work_per_sample": 2.0,
        "full": results["full"],
        "diagonal": results["diagonal"],
        "measured_diagonal_speedup": full_median / diagonal_median,
        "energy_difference_is_not_a_bias_test": True,
        "timing_claim_eligible": False,
        "timing_boundary": (
            "single-process local CPU random-state kernel profile; compile excluded "
            "from warm median, machine load not isolated"
        ),
    }


def run_spin_singlet_force_efficiency(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the deterministic N=8 estimator comparison and seal its output."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    runtime = numerical_runtime_identity()
    sampling = dict(spec["sampling"])
    rows: list[dict[str, object]] = []
    for case in spec["cases"]:
        geometry = str(case["geometry"])
        ratio = float(case["j2_over_j1"])
        if geometry == "square":
            bonds = square_j1j2_bonds(2, 4, j1=1.0, j2=ratio, periodic=False)
        else:
            bonds = triangular_j1j2_bonds(
                2, 4, j1=1.0, j2=ratio, periodic=False
            )
        report = singlet_force_efficiency_benchmark(
            bonds,
            overlaps=tuple(float(value) for value in sampling["ground_state_overlaps"]),
            sample_counts=tuple(int(value) for value in sampling["sample_counts"]),
            replicates=int(sampling["replicates"]),
            tangent_count=int(sampling["tangent_count"]),
            seed=int(sampling["seed"]) + len(rows),
        )
        anchor_error = abs(
            float(report["ground_energy"]) - float(case["exact_energy_anchor"])
        )
        if anchor_error > 2.0e-11:
            raise RuntimeError("exact-small ground energy disagrees with stored anchor")
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "geometry": geometry,
                "rows": 2,
                "columns": 4,
                "periodic": False,
                "j2_over_j1": ratio,
                "exact_energy_anchor": float(case["exact_energy_anchor"]),
                "anchor_provenance": str(case["anchor_provenance"]),
                "anchor_error": anchor_error,
                "report": report,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": _portable(config_path),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_sha256": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "jax_backend": jax.default_backend(),
        "claim_boundary": (
            "N=8 iid exact-Born work--variance decision only; amplitude-work is "
            "a call-count proxy, not wall time, convergence, or a ground-state win"
        ),
        "cases": rows,
        "runtime_profiles": [
            _runtime_profile(dict(profile)) for profile in spec["runtime_profiles"]
        ],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_singlet_force_efficiency"]
