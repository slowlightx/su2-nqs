from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_n36_pi_flux_benchmark import (
    DSL_ONE_LANCZOS_ENERGY_PER_SITE,
    DSL_ZERO_LANCZOS_ENERGY_PER_SITE,
    EXACT_ENERGY_PER_SITE,
    TASK,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_pi_flux_benchmark.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_n36_pi_flux_benchmark_v1/results.json"


def test_pi_flux_config_is_repeated_n36_benchmark() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK
    assert spec["sampler"]["replicates"] == 3


def test_pi_flux_result_is_complete_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["model_case"]["j2_over_j1"] == 0.125
    assert payload["model_case"]["target_sector"] == "exact S=0, M=0, Gamma-A1"
    assert payload["literature_targets"]["exact_36_site_energy_per_site"] == EXACT_ENERGY_PER_SITE
    assert DSL_ZERO_LANCZOS_ENERGY_PER_SITE == -0.501788
    assert DSL_ONE_LANCZOS_ENERGY_PER_SITE == -0.510344
    assert len(payload["replicates"]) == 3
    assert payload["hamiltonian_diagonalization_performed"] is False
