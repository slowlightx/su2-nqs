from __future__ import annotations

from pathlib import Path
import json

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_checkpoint_audit import (
    _checkpoint_path,
    _load_contract,
    _resolved_descent,
    _run_identity,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote"
    / "spin_scale_first_carrier_checkpoint_audit.json"
)
PILOT_ROOT = (
    ROOT
    / "projects/clebsch_tree_nqs/results"
    / "remote_scale_first_carrier_pilot_482522"
)


def test_carrier_checkpoint_audit_contract_and_cli() -> None:
    spec, training, base, initialization, _artifact_root = _load_contract(CONFIG)
    assert spec["controls"]["checkpoint_steps"] == [0, 16, 32, 48, 64]
    assert spec["controls"]["sample_count"] == 512
    assert set(spec["controls"]["audit_seeds"]).isdisjoint(
        base["production_contract"]["seeds"]
    )
    assert training["controls"]["total_steps"] == 64
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-carrier-checkpoint-audit",
            "--config",
            str(CONFIG),
            "--checkpoint-root",
            "source",
            "--case-id",
            "case",
            "--arm-id",
            "arm",
            "--output",
            "output",
        ]
    )
    assert parsed.command == "spin-scale-first-carrier-checkpoint-audit"
    merged = _build_parser().parse_args(
        [
            "spin-scale-first-carrier-checkpoint-audit-merge",
            "--config",
            str(CONFIG),
            "--input-root",
            "input",
            "--output",
            "output",
        ]
    )
    assert merged.command == "spin-scale-first-carrier-checkpoint-audit-merge"
    identity = _run_identity(
        training=training,
        base=base,
        initialization=initialization,
        case_index=0,
        case_id="triangular_6x6_j2_0.125",
        arm_id="fixed_sz_shared_autoregressive_spatial_vit",
        initial_params_sha256="abc",
    )
    assert identity["config_sha256"] == canonical_sha256(training)
    assert identity["physical_phase_convention"] == "marshall_checkerboard"
    assert identity["controls"]["cg_max_iterations"] == 512


def test_resolved_descent_requires_two_standard_errors() -> None:
    assert _resolved_descent(
        {
            "paired_energy_change_from_step_0": -3.1,
            "paired_energy_change_standard_error": 1.5,
        },
        "energy",
    )
    assert not _resolved_descent(
        {
            "paired_variance_change_from_step_0": -2.9,
            "paired_variance_change_standard_error": 1.5,
        },
        "variance",
    )


def test_carrier_checkpoint_layout_matches_process_isolated_output() -> None:
    if not PILOT_ROOT.is_dir():
        return
    path = _checkpoint_path(
        PILOT_ROOT,
        "triangular_6x6_j2_0.125",
        "exact_su2_balanced_schur_spatial_vit",
        1,
    )
    assert path.name.startswith("step_00000001_")
    result_path = (
        PILOT_ROOT
        / "triangular_6x6_j2_0.125"
        / "exact_su2_balanced_schur_spatial_vit"
        / "results.json"
    )
    result = json.loads(result_path.read_text())
    spec, training, base, initialization, _artifact_root = _load_contract(CONFIG)
    del spec
    expected = _run_identity(
        training=training,
        base=base,
        initialization=initialization,
        case_index=0,
        case_id="triangular_6x6_j2_0.125",
        arm_id="exact_su2_balanced_schur_spatial_vit",
        initial_params_sha256=result["initial_params_sha256"],
    )
    checkpoint = load_resumable_checkpoint(path, expected_run_identity=expected)
    assert checkpoint["completed_steps"] == 1
