from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.remote.analyze_g0 import _compare_case


def _result(backend: str, speedup: float = 1.0) -> dict[str, object]:
    trials = []
    for arm_id in ("fixed_sz_translation", "exact_su2_translation"):
        trials.append(
            {
                "arm_id": arm_id,
                "initial_params_sha256": f"params-{arm_id}",
                "initial_chain_states_sha256": "chains",
                "common_profile_states_sha256": "profile",
                "parameter_count": 10,
                "translation_group_order": 36,
                "padded_connectivity_width": 217,
                "performance_profile": [
                    {
                        "batch_size": 8,
                        "profile_energy": -1.25,
                        "gradient_norm": 3.5,
                        "profile_minimum_center_magnitude": 0.1,
                        "profile_maximum_local_energy_magnitude": 5.0,
                        "profile_nonfinite_local_count": 0,
                        "hpsi_warm_throughput_states_per_second": 10.0
                        * speedup,
                        "gradient_warm_throughput_states_per_second": 8.0
                        * speedup,
                    }
                ],
            }
        )
    return {
        "status": "complete",
        "task": "physical_mcmc_runtime_smoke",
        "config_sha256": "same-config",
        "jax_backend": backend,
        "devices": [f"{backend}:0"],
        "cases": [{"case_id": "case", "trials": trials}],
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))


def test_g0_analyzer_promotes_only_verified_joint_fivefold_speedup(
    tmp_path: Path,
) -> None:
    cpu_path, gpu_path = tmp_path / "cpu.json", tmp_path / "gpu.json"
    _write(cpu_path, _result("cpu"))
    _write(gpu_path, _result("gpu", speedup=5.1))

    comparison = _compare_case(cpu_path, gpu_path)

    assert comparison["gpu_backend_verified"] is True
    assert comparison["numerical_agreement"] is True
    assert comparison["case_promoted"] is True
    assert all(row["arm_promoted"] for row in comparison["arms"])


def test_g0_analyzer_fails_closed_on_numerical_or_gradient_mismatch(
    tmp_path: Path,
) -> None:
    cpu_path, gpu_path = tmp_path / "cpu.json", tmp_path / "gpu.json"
    cpu, gpu = _result("cpu"), _result("gpu", speedup=6.0)
    gpu["cases"][0]["trials"][0]["performance_profile"][0][
        "profile_energy"
    ] = -1.20
    gpu["cases"][0]["trials"][1]["performance_profile"][0][
        "gradient_warm_throughput_states_per_second"
    ] = 8.0 * 4.9
    _write(cpu_path, cpu)
    _write(gpu_path, gpu)

    comparison = _compare_case(cpu_path, gpu_path)

    assert comparison["case_promoted"] is False
    assert comparison["arms"][0]["numerical_agreement"] is False
    assert comparison["arms"][1]["batch_profiles"][0][
        "passes_joint_5x_gate"
    ] is False
