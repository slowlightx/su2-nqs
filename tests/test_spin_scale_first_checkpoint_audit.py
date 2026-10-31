from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_checkpoint_audit import (
    _load_contract,
    _paired_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote"
    / "spin_scale_first_checkpoint_audit.json"
)


def test_checkpoint_audit_contract_and_cli() -> None:
    spec, pilot, _audit, base = _load_contract(CONFIG)
    assert spec["controls"]["checkpoint_steps"] == [0, 16, 32, 48, 64]
    assert spec["controls"]["sample_count"] == 512
    assert spec["controls"]["device_microbatch_size"] == 16
    assert set(spec["controls"]["audit_seeds"]).isdisjoint(
        base["production_contract"]["seeds"]
    )
    assert pilot["controls"]["cg_max_iterations"] == 256
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-checkpoint-audit",
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
    assert parsed.command == "spin-scale-first-checkpoint-audit"
    merged = _build_parser().parse_args(
        [
            "spin-scale-first-checkpoint-audit-merge",
            "--config",
            str(CONFIG),
            "--input-root",
            "input",
            "--output",
            "output",
        ]
    )
    assert merged.command == "spin-scale-first-checkpoint-audit-merge"


def test_paired_checkpoint_summary_resolves_common_seed_change() -> None:
    rows = []
    for step, variance_offset in ((0, 0.0), (16, 2.0)):
        for seed, noise in ((11, -0.1), (12, 0.0), (13, 0.1), (14, 0.2)):
            rows.append(
                {
                    "case_id": "case",
                    "arm_id": "arm",
                    "checkpoint_step": step,
                    "audit_seed": seed,
                    "energy": 4.0 - step / 16.0 + noise,
                    "variance": 3.0 + variance_offset + noise,
                }
            )
    summary = _paired_summary(rows)
    assert len(summary) == 2
    final = summary[-1]
    assert np.isclose(final["paired_energy_change_from_step_0"], -1.0)
    assert np.isclose(final["paired_variance_change_from_step_0"], 2.0)
    assert final["variance_change_classification"] == "resolved_increase"
