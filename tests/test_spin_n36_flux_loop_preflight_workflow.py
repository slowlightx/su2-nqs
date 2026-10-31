from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n36_flux_loop_preflight import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs"
    / "configs/validation/spin_n36_flux_loop_preflight.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs"
    / "results/spin_n36_flux_loop_preflight_v1/results.json"
)


def test_registered_n36_flux_loop_preflight_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n36-flux-loop-preflight", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n36-flux-loop-preflight"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["sampler"]["measurement_blocks"] = 3
    with pytest.raises(ValueError, match="sampler protocol"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official N36 preflight not run")
def test_sealed_n36_preflight_never_authorizes_gpu_production() -> None:
    payload = json.loads(RESULT.read_text())
    gates = payload["preflight"]["gates"]
    assert payload["status"] == "complete"
    assert gates["gpu_production_licensed"] is False
    assert payload["promotion_decision"] in {
        "license_small_matched_energy_pilot",
        "hold_and_revise_phase_aware_carrier_or_action",
    }
