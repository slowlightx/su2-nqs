from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.spin_n36_su2_scalar_dressing import (
    TASK,
    _validate,
    run_spin_n36_su2_scalar_dressing,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_n36_su2_scalar_dressing_preflight.json"
)
REGULARIZED_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote/"
    "spin_n36_su2_scalar_dressing_regularized.json"
)


def test_preflight_reproduces_sealed_two_lanczos_on_common_untouched_chains(
    tmp_path: Path,
) -> None:
    payload = run_spin_n36_su2_scalar_dressing(CONFIG, tmp_path / "result")
    contract = payload["sealed_common_sample_contract"]
    assert contract["train_count"] == 512
    assert contract["audit_count"] == 512
    assert contract["chain_count"] == 64
    assert contract["two_lanczos_reproduced"]
    assert contract["two_lanczos"]["energy_per_site"] == pytest.approx(
        -0.515077745477902, abs=2.0e-14
    )
    assert payload["decision"]["remote_n36_pilot_licensed"]
    assert json.loads((tmp_path / "result/results.json").read_text())[
        "result_sha256"
    ] == payload["result_sha256"]


def test_config_rejects_zero_training_steps() -> None:
    spec = json.loads(CONFIG.read_text())
    assert spec["task"] == TASK
    spec["training"]["su2_steps"] = 0
    with pytest.raises(ValueError, match="su2_steps"):
        _validate(spec)


def test_regularized_refit_contract_is_preregistered() -> None:
    spec = json.loads(REGULARIZED_CONFIG.read_text())
    _validate(spec)
    assert spec["execution_tier"] == "remote_regularized_refit"
    assert spec["training"]["secondary_audit_start_block"] == 1
    assert spec["training"]["crossfit_folds"] == 8
    assert spec["training"]["ridge_grid"][0] == 0.0
