from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_schur_channel_audit import (
    _load_contract,
    _metrics,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_scale_first_carrier_schur_channel_audit.json"
)


def test_schur_channel_audit_contract_and_cli() -> None:
    spec, audit, training, base, _initialization, _artifact_root = _load_contract(CONFIG)
    assert len(spec["controls"]["sample_seeds"]) == 4
    assert spec["controls"]["sample_count_per_seed"] == 1024
    assert set(spec["controls"]["sample_seeds"]).isdisjoint(
        audit["controls"]["audit_seeds"]
    )
    assert training["controls"]["total_steps"] == 64
    assert len(base["systems"]) == 2
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-carrier-schur-channel-audit",
            "--config",
            str(CONFIG),
            "--checkpoint-root",
            "checkpoints",
        ]
    )
    assert parsed.command == "spin-scale-first-carrier-schur-channel-audit"


def test_schur_channel_metrics_identify_plaquette_singlet_support() -> None:
    layout = rectangular_four_spin_layout(2, 4)
    labels = np.zeros((5, 7), dtype=np.int32)
    result = _metrics(labels, layout)
    assert result["cell_q0_fraction"] == 1.0
    assert result["cell_q2_fraction"] == 0.0
    assert result["cell_q4_fraction"] == 0.0
    assert result["cell_channel_entropy"] == 0.0
    assert result["intercell_nonzero_fraction"] == 0.0
