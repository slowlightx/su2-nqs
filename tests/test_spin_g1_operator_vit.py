from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (
    _paired_chain_difference,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_g1_operator_vit_local_preflight.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_g1_operator_vit_local_preflight_v1/results.json"


def test_g1_local_preflight_is_exact_su2_a1_and_licenses_reduced_remote_canary() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["execution_tier"] == "local_exact_connectivity_preflight"
    assert spec["resonance_token"] == "rhombus_ring_exchange"
    assert spec["training"]["steps"] == 20

    result = json.loads(RESULT.read_text())
    exact = result["exact_small"]
    assert exact["final"]["energy_total"] < exact["initial"]["energy_total"]
    assert (
        exact["full_effective_space_oracle"]["energy_total"]
        < exact["two_lanczos_control"]["energy_total"]
    )
    profile = result["n36_connectivity_profile"]
    assert profile["maximum_word_depth"] == 3
    assert profile["word_count"] == 31
    assert profile["unique_terminal_amplitude_evaluations"] > 500_000
    assert result["decision"] == {
        "remote_g1_canary_licensed": True,
        "full_five_token_depth_three_rejected": True,
        "selected_effective_tokens": ["H_J1J2", "A1_resonance"],
    }


def test_paired_g1_uncertainty_uses_self_normalized_ratio_estimators() -> None:
    first_state = np.asarray([1.0, 10.0, 2.0, 20.0], dtype=np.complex128)
    first_local_energy = np.asarray([1.0, 4.0, 2.0, 8.0])
    first_action = first_state * first_local_energy
    second_state = np.ones(4, dtype=np.complex128)
    second_local_energy = np.asarray([0.5, 1.0, 1.5, 2.0])
    second_action = second_state * second_local_energy

    result = _paired_chain_difference(
        first_state,
        first_action,
        second_state,
        second_action,
        chain_count=2,
    )
    expected_first = np.sum(np.abs(first_state) ** 2 * first_local_energy) / np.sum(
        np.abs(first_state) ** 2
    )
    expected_second = np.mean(second_local_energy)
    assert np.isclose(
        result["difference_per_site"],
        (expected_first - expected_second) / 36.0,
    )
    naive_local_difference = np.mean(first_local_energy - second_local_energy) / 36.0
    assert not np.isclose(result["difference_per_site"], naive_local_difference)
    assert result["chain_standard_error_per_site"] > 0.0
    assert result["uncertainty_method"].startswith("paired_delete_one_chain")
