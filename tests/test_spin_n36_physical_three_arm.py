from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n36_physical_three_arm import (
    ARM_IDS,
    TASK,
    _teacher_born_fidelity,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n36_physical_three_arm_pilot.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_n36_physical_three_arm_pilot_v1/results.json"
)


def test_three_arm_config_uses_independent_wavefunctions_and_replicates() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert spec["sampling"]["replicates"] >= 2
    assert ARM_IDS == (
        "from_scratch_fixed_sz_spatial_vit",
        "pi_flux_spatial_vit_correlator",
        "pi_flux_distilled_su2_multiplicity_spatial",
    )


def test_teacher_born_ratio_fidelity_is_scale_invariant() -> None:
    teacher = np.asarray([1.0, -2.0, 3.0, -4.0])
    proportional = (2.0 - 3.0j) * teacher
    assert np.isclose(float(_teacher_born_fidelity(proportional, teacher)), 1.0)
    alternating = teacher * np.asarray([1.0, -1.0, 1.0, -1.0])
    assert float(_teacher_born_fidelity(alternating, teacher)) < 1.0e-14


def test_three_arm_result_records_ess_and_no_historical_residual_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["hamiltonian_diagonalization_performed"] is False
    assert payload["arm_contract"]["historical_six_scalar_residual_used"] is False
    assert set(payload["aggregate"]) == set(ARM_IDS)
    assert len(payload["replicates"]) >= 2
    for replicate in payload["replicates"]:
        assert {row["arm_id"] for row in replicate["arms"]} == set(ARM_IDS)
        for row in replicate["arms"]:
            assert 0.0 < row["final_audit"]["importance_ess_fraction"] <= 1.0
            assert "paired_final_minus_initial" in row
