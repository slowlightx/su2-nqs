from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
import os
import shutil
import subprocess
import sys

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_three_arm_preflight import (
    ARM_IDS,
    _validate,
)
from projects.clebsch_tree_nqs.workflows.spin_scale_first_accelerator_slices import (
    TASK as SLICE_TASK,
    _selection,
    run_spin_scale_first_accelerator_merge,
)
from su2_nqs.frontier.artifacts import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_scale_first_three_arm_preflight.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results"
    / "spin_scale_first_three_arm_preflight_v1/results.json"
)
SYNC_MANIFEST = (
    ROOT
    / "projects/clebsch_tree_nqs/remote/scale_first_sync_manifest.txt"
)
REMOTE_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote"
    / "spin_scale_first_gpu_preflight.json"
)
REMOTE_SCRIPT = (
    ROOT
    / "projects/clebsch_tree_nqs/remote"
    / "scale_first_three_arm_preflight_a100.sbatch"
)
TILE_SWEEP_SCRIPT = (
    ROOT
    / "projects/clebsch_tree_nqs/remote"
    / "scale_first_su2_tile_sweep_a100.sbatch"
)
MICROBATCH_SWEEP_SCRIPT = (
    ROOT
    / "projects/clebsch_tree_nqs/remote"
    / "scale_first_su2_microbatch_sweep_a100.sbatch"
)
TRAINING_CALIBRATION_SCRIPT = (
    ROOT
    / "projects/clebsch_tree_nqs/remote"
    / "scale_first_training_calibration_a100.sbatch"
)


def test_scale_first_three_arm_contract_and_cli_are_registered() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    parsed = _build_parser().parse_args(
        ["spin-scale-first-three-arm-preflight", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-scale-first-three-arm-preflight"
    assert tuple(row["arm_id"] for row in spec["arms"]) == ARM_IDS
    assert spec["local_canary"]["sample_count_per_arm"] == 4
    assert spec["local_canary"]["su2_connected_capacity"] == 4096
    assert spec["local_canary"]["offline_capacity_audit_sample_count"] == 32
    assert spec["optimizer_canary"]["name"] == (
        "matrix_free_sampled_SR_direction_only"
    )
    assert spec["production_contract"]["seeds"] == [
        2026092501,
        2026092502,
        2026092503,
    ]


def test_scale_first_remote_manifest_is_minimal_and_complete() -> None:
    entries = [line for line in SYNC_MANIFEST.read_text().splitlines() if line]
    assert len(entries) == len(set(entries))
    assert all((ROOT / entry).is_file() for entry in entries)
    assert not any("notebooks" in entry or "results/" in entry for entry in entries)
    assert (
        "projects/clebsch_tree_nqs/remote/scale_first_three_arm_preflight_a100.sbatch"
        in entries
    )
    assert "su2_nqs/vmc/local_energy.py" in entries
    assert (
        "projects/clebsch_tree_nqs/workflows/"
        "spin_scale_first_accelerator_slices.py"
    ) in entries


def test_scale_first_remote_manifest_imports_from_an_isolated_copy(tmp_path) -> None:
    entries = [line for line in SYNC_MANIFEST.read_text().splitlines() if line]
    for relative in entries:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    command = (
        "import json; "
        "from projects.clebsch_tree_nqs.workflows."
        "spin_scale_first_three_arm_preflight import _validate; "
        "from projects.clebsch_tree_nqs.workflows."
        "spin_scale_first_accelerator_slices import _selection; "
        "from projects.clebsch_tree_nqs.workflows."
        "spin_scale_first_training_calibration import TASK; "
        "p=json.load(open('projects/clebsch_tree_nqs/configs/remote/"
        "spin_scale_first_gpu_preflight.json')); "
        "_validate(p); _selection(p, p['systems'][0]['case_id'], "
        "p['arms'][0]['arm_id'], p['accelerator_benchmark']['batch_sizes'][0])"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_target_gpu_preflight_registers_a_real_warm_batch_sweep() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    _validate(spec)
    benchmark = spec["accelerator_benchmark"]
    assert benchmark == {
        "enabled": True,
        "measurement": "end_to_end_full_h_action",
        "batch_sizes": [4, 8, 16],
        "device_microbatch_size": 4,
        "su2_device_microbatch_sizes": [4, 8, 16],
        "su2_amplitude_tile_sizes": [16, 32, 64, 128, 256],
        "warm_repetitions": 5,
        "replay_absolute_tolerance": 1e-11,
        "replay_relative_tolerance": 1e-11,
    }
    script = REMOTE_SCRIPT.read_text()
    assert str(REMOTE_CONFIG.relative_to(ROOT)) in script
    assert "--loop=1" in script
    assert "utilization.memory" in script
    assert "power.draw" in script
    assert script.count("spin-scale-first-accelerator-slice") == 1
    assert "for CASE_ID in" in script
    assert "for ARM_ID in" in script
    assert "for BATCH_SIZE in 4 8 16" in script
    assert "spin-scale-first-accelerator-merge" in script
    tile_script = TILE_SWEEP_SCRIPT.read_text()
    assert "for TILE_SIZE in 16 32 64 128 256" in tile_script
    assert "--amplitude-tile-size" in tile_script
    microbatch_script = MICROBATCH_SWEEP_SCRIPT.read_text()
    assert "for MICROBATCH_SIZE in 4 8 16" in microbatch_script
    assert "--device-microbatch-size" in microbatch_script
    calibration_script = TRAINING_CALIBRATION_SCRIPT.read_text()
    assert "spin-scale-first-training-calibration" in calibration_script
    assert "exact_su2_balanced_schur_spatial_vit" in calibration_script


def test_accelerator_slice_cli_and_registered_selection() -> None:
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-accelerator-slice",
            "--config",
            str(REMOTE_CONFIG),
            "--case-id",
            "triangular_6x6_j2_0.125",
            "--arm-id",
            ARM_IDS[0],
            "--batch-size",
            "8",
            "--output",
            "slice",
        ]
    )
    assert parsed.command == "spin-scale-first-accelerator-slice"
    spec = json.loads(REMOTE_CONFIG.read_text())
    case, case_index, batch_index = _selection(
        spec, parsed.case_id, parsed.arm_id, parsed.batch_size
    )
    assert case["geometry"] == "triangular"
    assert (case_index, batch_index) == (0, 1)
    with pytest.raises(ValueError, match="unregistered batch_size"):
        _selection(spec, parsed.case_id, parsed.arm_id, 32)


def test_accelerator_slice_merge_requires_complete_gpu_coverage(tmp_path) -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    input_root = tmp_path / "slices"
    for case in spec["systems"]:
        for arm_id in ARM_IDS:
            for batch_size in spec["accelerator_benchmark"]["batch_sizes"]:
                row = {
                    "schema_version": 1,
                    "domain": "spin",
                    "task": SLICE_TASK,
                    "config_sha256": canonical_sha256(spec),
                    "case_id": case["case_id"],
                    "arm_id": arm_id,
                    "batch_size": batch_size,
                    "fresh_process_required": True,
                    "optimization_started": False,
                    "passed": True,
                    "runtime_identity": {"jax_backend": "gpu"},
                }
                row["result_sha256"] = canonical_sha256(row)
                destination = (
                    input_root
                    / case["case_id"]
                    / arm_id
                    / f"batch{batch_size}"
                    / "results.json"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(row))
    merged = run_spin_scale_first_accelerator_merge(
        REMOTE_CONFIG, input_root, tmp_path / "merged"
    )
    assert merged["slice_count"] == 18
    assert merged["accelerator_admission_passed"]

    missing = next(input_root.glob("*/*/batch16/results.json"))
    missing.unlink()
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        run_spin_scale_first_accelerator_merge(
            REMOTE_CONFIG,
            input_root,
            tmp_path / "incomplete",
        )


def test_target_gpu_preflight_rejects_under_repeated_timings() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["accelerator_benchmark"]["warm_repetitions"] = 2
    with pytest.raises(ValueError, match="at least three"):
        _validate(modified)


def test_target_gpu_preflight_rejects_incommensurate_microbatch() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["accelerator_benchmark"]["device_microbatch_size"] = 3
    with pytest.raises(ValueError, match="divisible"):
        _validate(modified)


def test_target_gpu_preflight_rejects_invalid_su2_tile_sweep() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["accelerator_benchmark"]["su2_amplitude_tile_sizes"] = [16, 31, 64]
    with pytest.raises(ValueError, match="tile sizes"):
        _validate(modified)


def test_target_gpu_preflight_rejects_invalid_su2_microbatch_sweep() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["accelerator_benchmark"]["su2_device_microbatch_sizes"] = [3, 4, 8]
    with pytest.raises(ValueError, match="microbatch sizes"):
        _validate(modified)


def test_training_calibration_has_compile_plus_two_warm_steps() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    _validate(spec)
    calibration = spec["training_calibration"]
    assert calibration["total_steps"] == 3
    assert calibration["sample_count"] == 16
    assert calibration["su2_amplitude_tile_size"] == 256
    assert calibration["su2_connected_capacity"] == 8192
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-training-calibration",
            "--config",
            str(REMOTE_CONFIG),
            "--case-id",
            "triangular_6x6_j2_0.125",
            "--arm-id",
            ARM_IDS[0],
            "--output",
            "calibration",
        ]
    )
    assert parsed.command == "spin-scale-first-training-calibration"


def test_training_calibration_rejects_unregistered_tile() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["training_calibration"]["su2_amplitude_tile_size"] = 512
    with pytest.raises(ValueError, match="tile size is not registered"):
        _validate(modified)


def test_target_gpu_preflight_rejects_loose_replay_tolerance() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    modified = deepcopy(spec)
    modified["accelerator_benchmark"]["replay_absolute_tolerance"] = 1.0e-6
    with pytest.raises(ValueError, match=r"must lie in \[0,1e-10\]"):
        _validate(modified)


def test_sealed_local_preflight_never_claims_trained_energy() -> None:
    if not RESULT.is_file():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["local_preflight_passed"]
    assert not payload["optimization_started"]
    assert not payload["production_training_ready"]
    assert payload["remaining_training_blockers"]
    artifact_arm_ids = tuple(
        row["arm_id"] for row in payload["config"]["arms"]
    )
    if artifact_arm_ids != ARM_IDS:
        return
    artifact_canary = payload["config"]["local_canary"]
    for row in payload["cases"]:
        assert row["shared_parameter_pytree_identical"]
        assert row["shared_initial_parameter_arrays_identical"]
        assert row["su2_device_h"]["overflow_count"] == 0
        sample_counts = {
            row["su2_device_h"]["sample_count"],
            row["fixed_sz_shared_device_h"]["sample_count"],
            row["symmetry_unaware_device_h"]["sample_count"],
        }
        assert sample_counts == {artifact_canary["sample_count_per_arm"]}
        audit = row["su2_device_h"].get("offline_reference_capacity_audit")
        if "offline_capacity_audit_sample_count" in artifact_canary:
            assert audit is not None
            assert audit["sample_count"] == artifact_canary[
                "offline_capacity_audit_sample_count"
            ]
            assert audit["capacity_headroom"] > 0
        sampler = row["symmetry_unaware_device_h"].get(
            "native_autoregressive_sampler"
        )
        if sampler is not None:
            assert sampler["exactly_normalized"]
        if "optimizer_canaries" in row:
            assert set(row["optimizer_canaries"]) == set(ARM_IDS)
            assert all(
                canary["all_finite"]
                and canary["direction_only_no_update_accepted"]
                for canary in row["optimizer_canaries"].values()
            )
