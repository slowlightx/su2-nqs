"""Local N=36 support gate for coherent plaquette-frame singlets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.coherent_plaquette_frames import distinct_partition_orbit
from su2_nqs.benchmarks.coherent_plaquette_sampling import (
    importance_diagnostics,
    intercell_casimir_support,
    sample_coherent_plaquette_frames,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


TASK = "spin_n36_coherent_frame_support"
TRIANGULAR_6X6 = ((6, 0), (0, 6))


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1) / math.sqrt(len(values)))


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 coherent-frame support identity")
    controls = spec.get("controls")
    gates = spec.get("gates")
    if not isinstance(controls, dict) or not isinstance(gates, dict):
        raise TypeError("controls and gates must be objects")
    seeds = controls.get("sample_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 4
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("sample_seeds must contain at least four unique integers")
    if int(controls.get("sample_count_per_seed", 0)) < 128:
        raise ValueError("sample_count_per_seed must be at least 128")
    for name in (
        "minimum_importance_effective_sample_fraction",
        "maximum_zero_target_fraction",
        "minimum_mean_intercell_casimir",
        "minimum_paired_casimir_z_score",
    ):
        value = float(gates.get(name, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"gates.{name} must be finite and nonnegative")
    if spec.get("remote_training_on_pass") is not False:
        raise ValueError("support passage cannot directly authorize remote training")


def _case(case_id: str):
    if case_id == "square_6x6_j2_0.5":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=False)
        group = square_gamma_a1_group(6)
    elif case_id == "triangular_6x6_j2_0.125":
        layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=True)
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_6X6)
    else:
        raise ValueError(f"unknown N36 coherent-frame case: {case_id}")
    frames = distinct_partition_orbit(layout.physical_cells, group)
    return layout, frames


def _run_case(case_id: str, spec: dict[str, object]) -> dict[str, object]:
    layout, frames = _case(case_id)
    seeds = tuple(int(seed) for seed in spec["controls"]["sample_seeds"])
    count = int(spec["controls"]["sample_count_per_seed"])
    rows = []
    for seed in seeds:
        arm_rows = {}
        for arm_id, arm_frames in (
            ("aligned_single_frame", frames[:1]),
            ("coherent_all_frames", frames),
        ):
            batch = sample_coherent_plaquette_frames(
                arm_frames, sample_count=count, seed=seed
            )
            row = {
                "arm_id": arm_id,
                "sample_seed": seed,
                "sample_count": count,
                "frame_count": len(arm_frames),
                **importance_diagnostics(batch),
                **intercell_casimir_support(batch, arm_frames, layout),
            }
            arm_rows[arm_id] = row
            rows.append(row)
        rows.append(
            {
                "arm_id": "paired_coherent_minus_aligned",
                "sample_seed": seed,
                "mean_intercell_casimir_difference": float(
                    arm_rows["coherent_all_frames"]["mean_intercell_casimir"]
                    - arm_rows["aligned_single_frame"]["mean_intercell_casimir"]
                ),
            }
        )
    coherent = [row for row in rows if row["arm_id"] == "coherent_all_frames"]
    aligned = [row for row in rows if row["arm_id"] == "aligned_single_frame"]
    paired = [
        row["mean_intercell_casimir_difference"]
        for row in rows
        if row["arm_id"] == "paired_coherent_minus_aligned"
    ]
    paired_mean = float(np.mean(paired))
    paired_se = _standard_error(paired)
    z_score = paired_mean / max(paired_se, np.finfo(float).tiny)
    gates = spec["gates"]
    minimum_ess = min(
        row["importance_effective_sample_fraction"] for row in coherent
    )
    maximum_zero = max(row["zero_target_fraction"] for row in coherent)
    coherent_mean = float(
        np.mean([row["mean_intercell_casimir"] for row in coherent])
    )
    aligned_mean = float(
        np.mean([row["mean_intercell_casimir"] for row in aligned])
    )
    passed = bool(
        minimum_ess
        >= float(gates["minimum_importance_effective_sample_fraction"])
        and maximum_zero <= float(gates["maximum_zero_target_fraction"])
        and coherent_mean >= float(gates["minimum_mean_intercell_casimir"])
        and z_score >= float(gates["minimum_paired_casimir_z_score"])
    )
    return {
        "case_id": case_id,
        "distinct_frame_count": len(frames),
        "rows": rows,
        "summary": {
            "aligned_mean_intercell_casimir": aligned_mean,
            "coherent_mean_intercell_casimir": coherent_mean,
            "paired_casimir_increase": paired_mean,
            "paired_casimir_standard_error": paired_se,
            "paired_casimir_z_score": z_score,
            "minimum_coherent_importance_effective_sample_fraction": minimum_ess,
            "maximum_coherent_zero_target_fraction": maximum_zero,
        },
        "passed": passed,
    }


def run_spin_n36_coherent_frame_support(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the bounded square/triangular N36 virtual-spin support gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    source_path = Path(str(spec["source_exact_n16_gate"]))
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    source = json.loads(source_path.read_text())
    if (
        source.get("task") != "spin_coherent_plaquette_frame_gate"
        or source.get("all_exact_n16_gates_passed") is not True
        or source.get("n36_support_preflight_allowed") is not True
    ):
        raise RuntimeError("the sealed exact-N16 coherent-frame gate did not pass")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    cases = [
        _run_case(case_id, spec)
        for case_id in ("square_6x6_j2_0.5", "triangular_6x6_j2_0.125")
    ]
    passed = bool(all(case["passed"] for case in cases))
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_exact_n16_result_sha256": source["result_sha256"],
        "cases": cases,
        "all_support_gates_passed": passed,
        "bounded_su2_canary_allowed": passed,
        "remote_training_allowed": False,
        "training_performed": False,
        "many_body_diagonalization_performed": False,
        "decision": (
            "run_local_physical_distance_bounded_su2_canary"
            if passed
            else "stop_coherent_frame_route_before_training"
        ),
        "runtime_identity": numerical_runtime_identity(),
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_validate", "run_spin_n36_coherent_frame_support"]
