from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_g1_regularized_operator import (
    _ratio_summary,
)
from projects.clebsch_tree_nqs.workflows.spin_n36_projected_multiplicity_preflight import (
    N_SITES,
    _has_committed_checkpoint,
    _matched_two_lanczos,
    _preflight_decision,
    _sealed_common_data,
    _su2_work_cursor,
    _validate,
)
from su2_nqs.frontier.artifacts import array_sha256
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REMOTE_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote/"
    "spin_n36_projected_multiplicity_a100_pilot.json"
)


def _remote_inputs():
    spec = json.loads(REMOTE_CONFIG.read_text())
    _validate(spec)
    development, _audit, train_states, train_raw, audit_states, audit_raw = (
        _sealed_common_data(spec)
    )
    samples = spec["samples"]
    train_slice = slice(
        samples["training_start"],
        samples["training_start"] + samples["training_count"],
    )
    audit_slice = slice(
        samples["audit_start"],
        samples["audit_start"] + samples["audit_count"],
    )
    return (
        spec,
        development,
        train_states,
        train_raw,
        audit_states,
        audit_raw,
        train_slice,
        audit_slice,
    )


def test_remote_selected_complex_action_arrays_match_preregistered_hashes() -> None:
    spec, _development, _ts, train_raw, _aus, audit_raw, train_slice, audit_slice = (
        _remote_inputs()
    )
    assert (
        array_sha256(train_raw[train_slice])
        == spec["samples"]["training_raw_actions_sha256"]
    )
    assert (
        array_sha256(audit_raw[audit_slice])
        == spec["samples"]["audit_raw_actions_sha256"]
    )


def test_remote_two_lanczos_is_refit_on_the_registered_training_block() -> None:
    (
        _spec,
        development,
        _ts,
        train_raw,
        _aus,
        audit_raw,
        train_slice,
        audit_slice,
    ) = _remote_inputs()
    fit = _matched_two_lanczos(
        development,
        train_raw,
        audit_raw,
        train_slice=train_slice,
        audit_slice=audit_slice,
    )
    frozen = np.asarray(
        development["frozen_two_lanczos"]["coefficients"], dtype=np.float64
    )
    assert fit["gram_rank"] == 3
    assert np.linalg.norm(fit["coefficients"] - frozen) / np.linalg.norm(frozen) > 0.1
    summary = _ratio_summary(
        fit["audit_state"],
        fit["audit_action"],
        chain_count=64,
        n_sites=N_SITES,
    )
    np.testing.assert_allclose(
        summary["energy_per_site"], -0.5149413870822656, atol=2.0e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        summary["variance_total"], 0.13565375557466602, atol=2.0e-12, rtol=0.0
    )


def test_remote_work_ledger_has_equal_logical_full_h_calls() -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    train_count = spec["samples"]["training_count"]
    audit_count = spec["samples"]["audit_count"]
    baseline_calls = 3 * (train_count + audit_count)
    neural_calls = spec["training"]["steps"] * train_count + audit_count
    su2_calls = train_count + (spec["training"]["steps"] - 1) * train_count + audit_count
    assert baseline_calls == neural_calls == su2_calls == 384


def _finite_history(count: int) -> list[dict[str, float | int]]:
    return [
        {
            "step": step,
            "training_energy_per_site": -0.5,
            "training_variance_per_site": 0.1,
            "training_ess_fraction": 0.8,
            "gradient_norm": 1.0,
            "applied_gradient_scale": 0.01,
        }
        for step in range(1, count + 1)
    ]


def _passing_decision_inputs() -> dict[str, object]:
    return {
        "execution_tier": "remote_gpu_pilot",
        "baseline_summary": {
            "energy_per_site": -0.52,
            "variance_total": 1.0,
            "importance_ess_fraction": 0.2,
        },
        "fixed_summary": {
            "energy_per_site": -0.520,
            "variance_total": 1.1,
            "importance_ess_fraction": 0.3,
        },
        "su2_summary": {
            "energy_per_site": -0.519,
            "variance_total": 0.9,
            "importance_ess_fraction": 0.4,
        },
        # Both upper two-SE bounds equal the registered margin exactly.
        "su2_minus_baseline": {
            "difference_per_site": 0.001,
            "chain_standard_error_per_site": 0.0005,
        },
        "su2_minus_fixed": {
            "difference_per_site": 0.001,
            "chain_standard_error_per_site": 0.0005,
        },
        "su2_history": _finite_history(4),
        "fixed_history": _finite_history(5),
        "training_steps": 5,
        "carrier_relative_error": 1.0e-10,
        "maximum_carrier_relative_error": 1.0e-10,
        "minimum_ess": 0.2,
        "energy_tolerance": 0.002,
    }


def test_fresh_repeat_gate_accepts_registered_boundaries() -> None:
    inputs = _passing_decision_inputs()
    inputs["su2_summary"]["importance_ess_fraction"] = 1.0 + 2.0e-16
    decision = _preflight_decision(**inputs)
    assert decision["fresh_independent_repeat_licensed"] is True
    assert decision["su2_noninferior_to_two_lanczos_at_registered_margin"] is True
    assert decision["su2_noninferior_to_fixed_sz_at_registered_margin"] is True
    assert decision["variance_gate_is_point_estimate_only"] is True
    assert decision["variance_claim_licensed"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (
            lambda row: row.update(execution_tier="local_sampled_preflight"),
            "remote_gpu_tier",
        ),
        (
            lambda row: row["baseline_summary"].update(energy_per_site=np.nan),
            "summaries_finite",
        ),
        (
            lambda row: row["fixed_summary"].update(importance_ess_fraction=0.19),
            "all_arms_ess_passed",
        ),
        (
            lambda row: row["su2_summary"].update(variance_total=1.0),
            "variance_below_two_lanczos_point_estimate",
        ),
        (
            lambda row: row["su2_summary"].update(variance_total=1.1),
            "variance_below_fixed_sz_point_estimate",
        ),
        (
            lambda row: row["su2_summary"].update(variance_total=-0.1),
            "summary_domains_valid",
        ),
        (
            lambda row: row["su2_summary"].update(importance_ess_fraction=1.1),
            "summary_domains_valid",
        ),
        (
            lambda row: row["su2_minus_baseline"].update(difference_per_site=0.00101),
            "noninferior_to_two_lanczos_at_registered_margin",
        ),
        (
            lambda row: row["su2_minus_fixed"].update(difference_per_site=0.00101),
            "noninferior_to_fixed_sz_at_registered_margin",
        ),
        (
            lambda row: row["su2_minus_baseline"].update(
                chain_standard_error_per_site=-1.0
            ),
            "paired_standard_errors_nonnegative",
        ),
        (
            lambda row: row.update(carrier_relative_error=1.01e-10),
            "carrier_certified",
        ),
        (
            lambda row: row.update(carrier_relative_error=-1.0),
            "carrier_certified",
        ),
        (
            lambda row: row.update(energy_tolerance=np.inf),
            "registered_thresholds_valid",
        ),
        (
            lambda row: row["su2_history"][0].update(training_energy_per_site=np.inf),
            "su2_history_complete_and_finite",
        ),
        (
            lambda row: row["fixed_history"][0].update(training_variance_per_site=np.nan),
            "fixed_sz_history_complete_and_finite",
        ),
        (
            lambda row: row["su2_history"][0].update(step=2),
            "su2_history_complete_and_finite",
        ),
        (
            lambda row: row["fixed_history"].pop(),
            "fixed_sz_history_complete_and_finite",
        ),
    ),
)
def test_fresh_repeat_gate_fails_closed(mutation, failed_check: str) -> None:
    inputs = copy.deepcopy(_passing_decision_inputs())
    mutation(inputs)
    decision = _preflight_decision(**inputs)
    assert decision["fresh_independent_repeat_licensed"] is False
    assert decision["fresh_independent_repeat_gate_checks"][failed_check] is False


def test_local_pilot_license_checks_both_histories_and_all_arms() -> None:
    inputs = _passing_decision_inputs()
    inputs["execution_tier"] = "local_sampled_preflight"
    decision = _preflight_decision(**inputs)
    assert decision["remote_gpu_pilot_licensed"] is True
    inputs["fixed_history"][0]["gradient_norm"] = np.nan
    decision = _preflight_decision(**inputs)
    assert decision["remote_gpu_pilot_licensed"] is False


def test_checkpoint_detection_ignores_only_uncommitted_temporary_files(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "arm"
    assert _has_committed_checkpoint(checkpoint) is False
    checkpoint.mkdir()
    (checkpoint / ".step_00000000.partial.tmp").write_bytes(b"partial")
    assert _has_committed_checkpoint(checkpoint) is False

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "step_00000000_formal.pkl").write_bytes(b"corrupt")
    assert _has_committed_checkpoint(corrupt) is True
    with pytest.raises(ValueError, match="no valid resumable checkpoint generation"):
        load_latest_resumable_checkpoint(corrupt)


def test_su2_work_cursor_includes_registered_calibration_pass() -> None:
    assert _su2_work_cursor(0, 64) == 64
    assert _su2_work_cursor(1, 64) == 128
    assert _su2_work_cursor(4, 64) == 320
    with pytest.raises(ValueError, match=r"invalid SU\(2\) work cursor"):
        _su2_work_cursor(-1, 64)


@pytest.mark.parametrize(
    ("gate", "value"),
    (
        ("minimum_ess_fraction", 0.0),
        ("minimum_ess_fraction", 1.01),
        ("minimum_ess_fraction", np.nan),
        ("maximum_carrier_relative_error", -1.0),
        ("maximum_carrier_relative_error", np.inf),
        ("competitive_energy_tolerance_per_site", -1.0),
        ("competitive_energy_tolerance_per_site", np.inf),
    ),
)
def test_validate_rejects_nonphysical_gate_values(gate: str, value: float) -> None:
    spec = json.loads(REMOTE_CONFIG.read_text())
    spec["gates"][gate] = value
    with pytest.raises(ValueError, match="gates"):
        _validate(spec)
