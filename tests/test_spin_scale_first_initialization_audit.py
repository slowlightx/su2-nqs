from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser, main
from projects.clebsch_tree_nqs.workflows import spin_scale_first_initialization_pilot
from projects.clebsch_tree_nqs.workflows.spin_scale_first_initialization_audit import (
    TASK,
    _aggregate,
    _load_contract,
)
from projects.clebsch_tree_nqs.workflows.spin_scale_first_initialization_pilot import (
    _load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_scale_first_initialization_audit.json"
)
PILOT_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_scale_first_initialization_pilot.json"
)


def test_initialization_audit_contract_and_cli_are_registered() -> None:
    audit, base = _load_contract(CONFIG)
    assert audit["task"] == TASK
    assert audit["controls"]["selected_direct_phase_initialization_scale"] == 16.0
    assert set(audit["controls"]["development_seeds"]).isdisjoint(
        base["production_contract"]["seeds"]
    )
    parsed = _build_parser().parse_args(
        ["spin-scale-first-initialization-audit", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-scale-first-initialization-audit"
    pilot = _build_parser().parse_args(
        ["spin-scale-first-initialization-pilot", "--config", str(PILOT_CONFIG)]
    )
    assert pilot.command == "spin-scale-first-initialization-pilot"
    pilot_spec, pilot_audit, pilot_base = _load_pilot_contract(PILOT_CONFIG)
    assert pilot_spec["controls"]["total_steps"] == 2
    assert pilot_audit == audit
    assert pilot_base == base


def test_initialization_audit_aggregation_is_deterministic() -> None:
    rows = []
    for seed, energy, variance in ((2, 3.0, 5.0), (1, 1.0, 3.0)):
        rows.append(
            {
                "case_id": "case",
                "arm_id": "arm",
                "seed": seed,
                "direct_phase_initialization_scale": 4.0,
                "energy": energy,
                "variance": variance,
                "gradient_norm": 2.0,
                "overflow_count": 0,
                "parameter_count": 7,
            }
        )
    summary = _aggregate(rows)
    assert len(summary) == 1
    assert summary[0]["energy_mean"] == 2.0
    assert summary[0]["variance_mean"] == 4.0
    assert summary[0]["parameter_counts"] == [7]
    assert json.loads(json.dumps(summary)) == summary


def test_initialization_pilot_cli_forwards_process_isolation_filters(
    monkeypatch, tmp_path
) -> None:
    observed = {}

    def fake_run(config, output, **kwargs):
        observed.update(config=config, output=output, **kwargs)

    monkeypatch.setattr(
        spin_scale_first_initialization_pilot,
        "run_spin_scale_first_initialization_pilot",
        fake_run,
    )
    arm_id = "exact_su2_balanced_schur_spatial_vit"
    case_id = "triangular_6x6_j2_0.125"
    destination = tmp_path / "slice"
    assert main(
        [
            "spin-scale-first-initialization-pilot",
            "--config",
            str(PILOT_CONFIG),
            "--case-id",
            case_id,
            "--arm-id",
            arm_id,
            "--output",
            str(destination),
        ]
    ) == 0
    assert observed["case_id"] == case_id
    assert observed["arm_id"] == arm_id
    assert observed["output"] == destination
