from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "clebsch_tree_nqs"


def test_scan_config_uses_registered_finite_4x4_anchors() -> None:
    reference = json.loads(
        (PROJECT / "references/spin_j1j2_coupling_scan.json").read_text()
    )
    config = json.loads(
        (
            PROJECT
            / "configs/validation/spin_su2_route_vit_coupling_scan.json"
        ).read_text()
    )
    lookup = {
        (geometry, float(row["j2_over_j1"])): float(row["total_energy"])
        for geometry in ("square", "triangular")
        for row in reference["finite_4x4_pbc_exact_anchors"][geometry]
    }
    assert len(config["cases"]) == 13
    for case in config["cases"]:
        key = (case["geometry"], float(case["j2_over_j1"]))
        assert float(case["exact_energy_anchor"]) == lookup[key]


def test_user_ipeps_curve_is_thermodynamic_context_not_exact_target() -> None:
    reference = json.loads(
        (PROJECT / "references/spin_j1j2_coupling_scan.json").read_text()
    )
    curve = reference["triangular_user_ipeps_d5"]
    points = curve["points"]
    np.testing.assert_allclose(
        [row["j2_over_j1"] for row in points],
        [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
        atol=0.0,
        rtol=0.0,
    )
    assert "thermodynamic-limit" in curve["scope"]
    exact = reference["finite_4x4_pbc_exact_anchors"]["triangular"]
    assert not np.allclose(
        [row["total_energy"] / 16.0 for row in exact],
        [
            row["energy_per_site"]
            for row in points
            if row["j2_over_j1"] != 0.025
        ],
    )
