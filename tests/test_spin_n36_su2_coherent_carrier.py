from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n36_su2_coherent_carrier import (
    ARM_ID,
    TASK,
    _validate,
    rms_match_residual_scale,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_n36_su2_coherent_carrier_pilot.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/spin_n36_su2_coherent_carrier_pilot_v2/results.json"
)


def test_coherent_carrier_config_preregisters_joint_activation() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert spec["sampling"]["replicates"] >= 2
    assert spec["training"]["steps"] >= 2
    assert ARM_ID == "pi_flux_plus_coherent_su2_multiplicity_attention"


def test_rms_calibration_matches_channel_scales_without_energy() -> None:
    carrier = np.asarray([1.0, -2.0, 3.0])
    residual = np.asarray([0.5, -1.0, 1.5])
    scale = rms_match_residual_scale(carrier, residual)
    assert np.isclose(scale, 2.0)
    assert np.isclose(
        np.sqrt(np.mean(np.abs(scale * residual / carrier) ** 2)),
        1.0,
    )


def test_coherent_carrier_result_records_symmetry_boundary_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["hamiltonian_diagonalization_performed"] is False
    contract = payload["architecture_contract"]
    assert contract["zero_head_exactly_equals_bare_carrier"] is True
    assert contract["configuration_wise_product_used"] is False
    assert contract["residual_total_spin_sector"] == "exact S=0, M=0"
    assert len(payload["replicates"]) >= 2
    assert payload["aggregate"]["ess_gate_passed"] is False
