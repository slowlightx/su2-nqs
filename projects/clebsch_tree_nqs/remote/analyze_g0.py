#!/usr/bin/env python3
"""Compare sealed CPU and GPU G0 kernel profiles and emit a go/no-go record.

The decision is intentionally mechanical.  A faster isolated kernel is not a
physics result, but it is the prerequisite for spending GPU hours on the
matched variational frontier.  The script therefore checks provenance and
numerical agreement before considering throughput.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ARM_IDS = ("fixed_sz_translation", "exact_su2_translation")
MINIMUM_SPEEDUP = 5.0


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete G0 result: {path}")
    if payload.get("task") != "physical_mcmc_runtime_smoke":
        raise ValueError(f"wrong task in {path}")
    return payload


def _trial_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trials = payload["cases"][0]["trials"]
    result = {str(row["arm_id"]): row for row in trials}
    if tuple(sorted(result)) != tuple(sorted(ARM_IDS)):
        raise ValueError(f"G0 arms must be exactly {ARM_IDS}, got {tuple(result)}")
    return result


def _profile_map(trial: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["batch_size"]): row for row in trial["performance_profile"]}


def _numerically_close(cpu: float, gpu: float) -> bool:
    return math.isclose(cpu, gpu, rel_tol=1.0e-7, abs_tol=1.0e-9)


def _compare_case(cpu_path: Path, gpu_path: Path) -> dict[str, Any]:
    cpu = _load(cpu_path)
    gpu = _load(gpu_path)
    if cpu["config_sha256"] != gpu["config_sha256"]:
        raise ValueError(f"config hash mismatch: {cpu_path} versus {gpu_path}")
    if cpu["cases"][0]["case_id"] != gpu["cases"][0]["case_id"]:
        raise ValueError("case identity mismatch")
    cpu_trials = _trial_map(cpu)
    gpu_trials = _trial_map(gpu)
    arm_rows: list[dict[str, Any]] = []
    for arm_id in ARM_IDS:
        cpu_trial, gpu_trial = cpu_trials[arm_id], gpu_trials[arm_id]
        identity_checks = {
            key: cpu_trial[key] == gpu_trial[key]
            for key in (
                "initial_params_sha256",
                "initial_chain_states_sha256",
                "common_profile_states_sha256",
                "parameter_count",
                "translation_group_order",
                "padded_connectivity_width",
            )
        }
        cpu_profiles = _profile_map(cpu_trial)
        gpu_profiles = _profile_map(gpu_trial)
        if set(cpu_profiles) != set(gpu_profiles):
            raise ValueError(f"profile batch mismatch for {arm_id}")
        batch_rows: list[dict[str, Any]] = []
        for batch_size in sorted(cpu_profiles):
            cpu_row, gpu_row = cpu_profiles[batch_size], gpu_profiles[batch_size]
            numerical_checks = {
                key: _numerically_close(float(cpu_row[key]), float(gpu_row[key]))
                for key in (
                    "profile_energy",
                    "gradient_norm",
                    "profile_minimum_center_magnitude",
                    "profile_maximum_local_energy_magnitude",
                )
            }
            numerical_checks["profile_nonfinite_local_count"] = (
                int(cpu_row["profile_nonfinite_local_count"])
                == int(gpu_row["profile_nonfinite_local_count"])
            )
            hpsi_speedup = float(
                gpu_row["hpsi_warm_throughput_states_per_second"]
            ) / float(cpu_row["hpsi_warm_throughput_states_per_second"])
            gradient_speedup = float(
                gpu_row["gradient_warm_throughput_states_per_second"]
            ) / float(cpu_row["gradient_warm_throughput_states_per_second"])
            batch_rows.append(
                {
                    "batch_size": batch_size,
                    "numerical_checks": numerical_checks,
                    "numerical_agreement": all(numerical_checks.values()),
                    "hpsi_warm_speedup": hpsi_speedup,
                    "gradient_warm_speedup": gradient_speedup,
                    "passes_joint_5x_gate": (
                        hpsi_speedup >= MINIMUM_SPEEDUP
                        and gradient_speedup >= MINIMUM_SPEEDUP
                    ),
                    "cpu_hpsi_states_per_second": float(
                        cpu_row["hpsi_warm_throughput_states_per_second"]
                    ),
                    "gpu_hpsi_states_per_second": float(
                        gpu_row["hpsi_warm_throughput_states_per_second"]
                    ),
                    "cpu_gradient_states_per_second": float(
                        cpu_row["gradient_warm_throughput_states_per_second"]
                    ),
                    "gpu_gradient_states_per_second": float(
                        gpu_row["gradient_warm_throughput_states_per_second"]
                    ),
                }
            )
        numerical_agreement = all(identity_checks.values()) and all(
            row["numerical_agreement"] for row in batch_rows
        )
        arm_rows.append(
            {
                "arm_id": arm_id,
                "identity_checks": identity_checks,
                "numerical_agreement": numerical_agreement,
                "batch_profiles": batch_rows,
                "arm_promoted": numerical_agreement
                and any(row["passes_joint_5x_gate"] for row in batch_rows),
            }
        )
    case_numerical = all(row["numerical_agreement"] for row in arm_rows)
    return {
        "case_id": cpu["cases"][0]["case_id"],
        "cpu_result": str(cpu_path),
        "gpu_result": str(gpu_path),
        "config_sha256": cpu["config_sha256"],
        "gpu_backend_verified": gpu.get("jax_backend") == "gpu",
        "gpu_devices": gpu.get("devices", []),
        "numerical_agreement": case_numerical,
        "arms": arm_rows,
        "case_promoted": (
            gpu.get("jax_backend") == "gpu"
            and case_numerical
            and all(row["arm_promoted"] for row in arm_rows)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-root", required=True, type=Path)
    parser.add_argument("--gpu-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("triangular_n36", "square_n64"),
        help="result-directory names to compare under both roots",
    )
    args = parser.parse_args()
    cases = [
        _compare_case(
            args.cpu_root / name / "results.json",
            args.gpu_root / name / "results.json",
        )
        for name in args.cases
    ]
    su2_promoted = all(
        next(row for row in case["arms"] if row["arm_id"] == "exact_su2_translation")[
            "arm_promoted"
        ]
        and case["gpu_backend_verified"]
        for case in cases
    )
    matched_promoted = all(case["case_promoted"] for case in cases)
    payload = {
        "schema_version": 1,
        "task": "g0_cpu_gpu_promotion_decision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_joint_warm_speedup": MINIMUM_SPEEDUP,
        "cases": cases,
        "decision": {
            "exact_su2_kernel_promoted": su2_promoted,
            "matched_fixed_sz_and_su2_kernel_promoted": matched_promoted,
            "g1_matched_training_licensed": matched_promoted,
        },
        "claim_boundary": (
            "A pass licenses a bounded matched training canary only. It is not "
            "an energy, phase, or variational-efficiency result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
