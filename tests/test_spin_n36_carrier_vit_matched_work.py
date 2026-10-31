from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
    ARM_IDS,
    TASK,
    _known_mixture_samples,
    _normalized_gradient_step,
    _validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_carrier_vit_matched_work.json"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/spin_n36_carrier_vit_matched_work_v7/results.json"


def test_registered_config() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["task"] == TASK


def test_known_common_sample_mixture_has_positive_exact_density() -> None:
    states, density, ledger = _known_mixture_samples(
        seed=9,
        count=16,
        reference_dimer_probability=0.80,
        exploratory_dimer_probability=0.15,
    )
    assert states.shape == (16, 36)
    assert (states.sum(axis=1) == 18).all()
    assert (density > 0.0).all()
    assert (
        ledger["reference_dimer_origin_count"]
        + ledger["exploratory_dimer_origin_count"]
        + ledger["uniform_origin_count"]
        == 16
    )


def test_head_optimizer_uses_registered_step_norm_even_for_small_gradient() -> None:
    import numpy as np

    params = {"coefficients": np.zeros(3)}
    gradient = {"coefficients": np.asarray((3.0e-6, 4.0e-6, 0.0))}
    candidate, norm, scale = _normalized_gradient_step(params, gradient, 0.04)
    assert abs(norm - 5.0e-6) < 1.0e-15
    assert abs(np.linalg.norm(candidate["coefficients"]) - 0.04) < 1.0e-14
    assert abs(scale - 8000.0) < 1.0e-9
    unchanged, zero_norm, zero_scale = _normalized_gradient_step(
        params, {"coefficients": np.zeros(3)}, 0.04
    )
    assert zero_norm == zero_scale == 0.0
    np.testing.assert_array_equal(unchanged["coefficients"], 0.0)


def test_sealed_result_has_strict_matched_work_when_present() -> None:
    if not RESULT.exists():
        return
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert tuple(row["arm_id"] for row in payload["arms"]) == ARM_IDS
    ledgers = [row["work_ledger"] for row in payload["arms"]]
    assert len({row["training_steps"] for row in ledgers}) == 1
    assert len({row["shared_samples_per_step"] for row in ledgers}) == 1
    assert len({row["full_h_configuration_calls_total"] for row in ledgers}) == 1
    assert payload["common_samples"]["exact_same_training_configurations_every_arm"]
    assert payload["carrier_registration"]["generic_orthogonal_to_physical_span"]
    assert payload["gates"]["all_comparison_gates_passed"]
    assert payload["gates"]["real_local_energy_gate_passed"]
    assert payload["gates"]["maximum_final_imaginary_local_energy"] == 0.0
    assert payload["vit_residual_controls"]["real_ground_state_correlators"]
    assert payload["vit_residual_controls"]["fixed_sz_spatial_vit"]["tangent_rank"] == 6
