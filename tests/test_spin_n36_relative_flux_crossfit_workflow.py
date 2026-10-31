from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n36_relative_flux_crossfit import (
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs"
    / "configs/validation/spin_n36_relative_flux_crossfit.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs"
    / "results/spin_n36_relative_flux_crossfit_v1/results.json"
)


def test_registered_relative_flux_crossfit_is_fail_closed() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n36-relative-flux-crossfit", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n36-relative-flux-crossfit"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    mutated = json.loads(CONFIG.read_text())
    mutated["spectral_protocol"]["relative_carrier_eigenvalue_cutoff"] = 0.001
    with pytest.raises(ValueError, match="spectral protocol"):
        _validate(mutated)


@pytest.mark.skipif(not RESULT.is_file(), reason="official crossfit not run")
def test_sealed_relative_flux_crossfit_never_authorizes_gpu_production() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["crossfit"]["gates"]["gpu_production_licensed"] is False
    assert payload["promotion_decision"] in {
        "license_one_small_matched_energy_pilot",
        "freeze_current_flux_carrier_bank",
    }
