from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs import analysis


PROJECT = ROOT / "projects/clebsch_tree_nqs"

U1_PARAMETERS = {2: 354, 3: 1176, 4: 2772, 5: 5200, 6: 8500}
SU2_PARAMETERS = {2: 246, 3: 819, 4: 1932, 5: 4000, 6: 7000}


def _logical_arm(arm: dict[str, object], config: dict[str, object]) -> tuple[str, str, int]:
    width = int(arm.get("model", {}).get("feature_dim", config["model"]["feature_dim"]))
    return str(arm["backbone"]), str(arm["projection"]), width


def _curve_error(
    *,
    backbone: str,
    projection: str,
    width: int,
    seed: int,
    coupling: float,
    completed_updates: int,
) -> float:
    seed_factor = 1.0 + 0.006 * (seed - 24.5)
    coupling_factor = 1.0 + 0.2 * coupling
    if projection == "gamma_a1":
        asymptote = 2.0e-7 if backbone == "u1" else 1.0e-13
        return asymptote + 0.05 * math.exp(-0.12 * completed_updates)
    if projection == "none":
        asymptote = 0.008 if backbone == "u1" else 0.006
        return seed_factor * coupling_factor * (
            asymptote + 0.05 * math.exp(-0.006 * completed_updates)
        )
    asymptote = (1.2e-4 if backbone == "u1" else 4.0e-5) * (7 - width)
    rate = (0.012 if backbone == "u1" else 0.020) + 0.0005 * width
    return seed_factor * coupling_factor * (
        asymptote + 0.05 * math.exp(-rate * completed_updates)
    )


def _physical_observables(
    *, geometry: str, rows: int, columns: int, spin_squared: float
) -> dict[str, object]:
    n_sites = rows * columns
    grid = np.full((rows, columns), (0.75 * n_sites - spin_squared / n_sites) / (n_sites - 1))
    grid[0, 0] = spin_squared / n_sites
    result: dict[str, object] = {
        "spin_structure_factor_grid": grid.tolist(),
        "peak_nonzero_momentum_index": [0, 1],
        "peak_nonzero_structure_factor": float(grid[0, 1]),
        "uniform_structure_factor": float(grid[0, 0]),
        "mean_unit_coupling_bond_correlation": -0.2,
    }
    if geometry == "square":
        result.update(
            {
                "neel_structure_factor": float(grid[rows // 2, columns // 2]),
                "stripe_structure_factor_max": float(
                    max(grid[rows // 2, 0], grid[0, columns // 2])
                ),
            }
        )
    else:
        result.update(
            {
                "k_120_commensurate": False,
                "stripe_M_structure_factor_max": float(
                    max(
                        grid[rows // 2, 0],
                        grid[0, columns // 2],
                        grid[rows // 2, columns // 2],
                    )
                ),
                "order_120_K_structure_factor_max": None,
            }
        )
    return result


def _make_trial(
    *,
    case_spec: dict[str, object],
    arm: dict[str, object],
    config: dict[str, object],
    seed: int,
    config_sha256: str,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    backbone, projection, width = _logical_arm(arm, config)
    steps = int(config["optimization"]["steps"])
    record_every = int(config["optimization"]["record_every"])
    anchor = float(case_spec["exact_energy_anchor"])
    n_sites = int(case_spec["rows"]) * int(case_spec["columns"])
    coupling = float(case_spec["j2_over_j1"])
    parameter_count = (U1_PARAMETERS if backbone == "u1" else SU2_PARAMETERS)[width]
    per_step = (0.0028 if backbone == "u1" else 0.0045) * (1.0 + 0.08 * width)
    training_seconds = steps * per_step
    compile_seconds = (0.12 if backbone == "u1" else 0.2) * width
    setup_seconds = 0.03 * width
    compile_plus = setup_seconds + compile_seconds + training_seconds
    history = []
    for step in range(record_every, steps + 1, record_every):
        error = _curve_error(
            backbone=backbone,
            projection=projection,
            width=width,
            seed=seed,
            coupling=coupling,
            completed_updates=step - 1,
        )
        history.append(
            {
                "step": float(step),
                "loss_before_update": anchor + n_sites * error,
                "gradient_norm": 0.1 + 1.0 / step,
                "cumulative_training_seconds": step * per_step,
            }
        )
    final_error = _curve_error(
        backbone=backbone,
        projection=projection,
        width=width,
        seed=seed,
        coupling=coupling,
        completed_updates=steps,
    )
    initial_error = _curve_error(
        backbone=backbone,
        projection=projection,
        width=width,
        seed=seed,
        coupling=coupling,
        completed_updates=0,
    )
    energy = anchor + n_sites * final_error
    spin_squared = 0.0 if backbone == "su2" else 0.16
    pair_id = f"pair-{backbone}-D{width}-seed{seed}"
    params_hash = f"params-{backbone}-D{width}-seed{seed}"
    raw_hash = f"raw-{backbone}-D{width}-seed{seed}"
    trial: dict[str, object] = {
        "arm_id": str(arm["arm_id"]),
        "backbone": backbone,
        "projection": projection,
        "support": "fixed_m0",
        "seed": seed,
        "feature_dim": width,
        "parameter_count": parameter_count,
        "group_order": {"none": 1, "translation": 16, "gamma_a1": 128}[projection],
        "initial_params_sha256": params_hash,
        "initial_raw_amplitudes_sha256": raw_hash,
        "initialization_pair_id": pair_id,
        "training": {
            "initial_loss": anchor + n_sites * initial_error,
            "final_loss_before_last_update": history[-1]["loss_before_update"],
            "compile_seconds": compile_seconds,
            "training_seconds": training_seconds,
            "history": history,
        },
        "compile_plus_optimizer_seconds": compile_plus,
        "final_observable_seconds": 100.0,
        "trial_wall_seconds_excluding_case_preparation": 999.0,
        "metrics": {
            "energy": energy,
            "energy_per_site": energy / n_sites,
            "energy_error": energy - anchor,
            "energy_error_per_site": final_error,
            "variance_per_site": max(2.0 * final_error, 1.0e-14),
            "selected_weight_of_raw": 0.5,
            "projection_closure_error": 0.0,
            "spin_squared_expectation": spin_squared,
            "spin_squared_residual": 0.0 if backbone == "su2" else 0.4,
            "physical_observables": _physical_observables(
                geometry=str(case_spec["geometry"]),
                rows=int(case_spec["rows"]),
                columns=int(case_spec["columns"]),
                spin_squared=spin_squared,
            ),
        },
        "work_ledger_per_loss": {},
        "energy_to_target": [],
        "config_sha256": config_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "trial_identity_sha256": analysis._canonical_sha256(
            [case_spec["case_id"], arm["arm_id"], seed]
        ),
    }
    trial["trial_payload_sha256"] = analysis._trial_payload_sha256(trial)
    return trial


def _make_campaign(config_name: str) -> dict[str, object]:
    config = json.loads((PROJECT / "configs" / config_name).read_text())
    config_sha256 = analysis._canonical_sha256(config)
    source = {"synthetic_fixture.py": "source-hash"}
    source_manifest_sha256 = analysis._canonical_sha256(source)
    runtime = {"python": "synthetic", "numpy": "synthetic"}
    runtime_identity_sha256 = analysis._canonical_sha256(runtime)
    arms = {str(arm["arm_id"]): arm for arm in config["arms"]}
    cases = []
    for case_spec in config["cases"]:
        physical_tag = (
            f"{case_spec['geometry']}-{case_spec['rows']}x{case_spec['columns']}-"
            f"{case_spec['j2_over_j1']}"
        )
        case = {
            "case_id": str(case_spec["case_id"]),
            "geometry": str(case_spec["geometry"]),
            "rows": int(case_spec["rows"]),
            "columns": int(case_spec["columns"]),
            "n_sites": int(case_spec["rows"]) * int(case_spec["columns"]),
            "j2_over_j1": float(case_spec["j2_over_j1"]),
            "support": str(case_spec["support"]),
            "basis_size": 12870,
            "hamiltonian_nnz": 123,
            "bond_count": 64,
            "site_order": list(range(16)),
            "group_orders": {"none": 1, "translation": 16, "gamma_a1": 128},
            "exact_energy_anchor": float(case_spec["exact_energy_anchor"]),
            "anchor_provenance": str(case_spec["anchor_provenance"]),
            "basis_sha256": f"basis-{physical_tag}",
            "bonds_sha256": f"bonds-{physical_tag}",
            "hamiltonian_coo_sha256": f"hamiltonian-{physical_tag}",
            "group_sha256": {},
            "anchor_sha256": f"anchor-{physical_tag}",
            "case_preparation_seconds": 0.0,
            "k_120_commensurate": False,
            "parameter_brackets": {},
            "paired_initialization_audit": [],
            "trials": [],
        }
        for arm_id in case_spec["arm_ids"]:
            for seed in config["optimization"]["seeds"]:
                case["trials"].append(
                    _make_trial(
                        case_spec=case_spec,
                        arm=arms[str(arm_id)],
                        config=config,
                        seed=int(seed),
                        config_sha256=config_sha256,
                        source_manifest_sha256=source_manifest_sha256,
                        runtime_identity_sha256=runtime_identity_sha256,
                    )
                )
        cases.append(case)
    payload: dict[str, object] = {
        "schema_version": 4,
        "timing_schema_version": 2,
        "domain": "spin",
        "status": "complete",
        "generated_at_utc": "synthetic",
        "config_path": f"projects/clebsch_tree_nqs/configs/{config_name}",
        "config_sha256": config_sha256,
        "config": config,
        "source_sha256": source,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_identity_sha256,
        "jax_backend": "cpu",
        "devices": ["synthetic-cpu"],
        "route_boundary": {},
        "timing_scope": "synthetic",
        "cases": cases,
    }
    _reseal(payload)
    return payload


def _reseal(payload: dict[str, object]) -> None:
    manifest = {}
    for case in payload["cases"]:
        for trial in case["trials"]:
            trial["trial_payload_sha256"] = analysis._trial_payload_sha256(trial)
            manifest[
                f"{case['case_id']}/{trial['arm_id']}/seed{trial['seed']}"
            ] = trial["trial_payload_sha256"]
    manifest = dict(sorted(manifest.items()))
    payload["trial_manifest"] = manifest
    payload["trial_manifest_sha256"] = analysis._canonical_sha256(manifest)


@pytest.fixture(scope="module")
def synthetic_campaigns() -> tuple[dict[str, object], dict[str, object]]:
    return (
        _make_campaign("spin_prl_local_pareto.json"),
        _make_campaign("spin_prl_local_u1_long.json"),
    )


def test_accepts_384_plus_32_and_returns_json_summary(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, long = synthetic_campaigns
    validation = analysis.validate_prl_results(base, long)
    assert validation["base"]["trial_count"] == 384
    assert validation["long"]["trial_count"] == 32
    assert validation["physics"]["checked_trials"] == 416
    assert validation["initialization"]["base_long_matched_initializations"] == 32
    assert validation["initialization"]["u1_su2_initial_functions_are_paired"] is False

    summary = analysis.analyze_prl_results(
        base,
        long,
        bootstrap_resamples=100,
        bootstrap_seed=1234,
        time_grid_size=11,
    )
    assert len(summary["cases"]) == 4
    assert len(summary["a1_floor_diagnostics"]) == 2
    assert all(
        case["parameter_pareto"]["average_log10_advantage"] > 0.0
        for case in summary["cases"].values()
    )
    assert all(
        case["matched_time_pareto"]["warm"]["u1_D6_source"].startswith("sealed")
        for case in summary["cases"].values()
    )
    assert set(summary["promotion_gates"]["families"]) == {
        "parameter",
        "warm",
        "cold",
    }
    assert summary["promotion_gates"]["a1_excluded"] is True
    for family in summary["promotion_gates"]["families"].values():
        assert len(family["cases"]) == 4
        assert family["passed_case_count"] == sum(
            case["passed"] for case in family["cases"].values()
        )
    json.dumps(summary, allow_nan=False)


def test_optional_su2_long_horizon_is_sealed_and_replaces_d6_curve(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, long = synthetic_campaigns
    su2_long = _make_campaign("spin_prl_local_su2_long.json")
    summary = analysis.analyze_prl_results(
        base,
        long,
        su2_long,
        bootstrap_resamples=100,
        bootstrap_seed=4321,
        time_grid_size=11,
    )
    assert summary["validation"]["physics"]["checked_trials"] == 448
    assert summary["validation"]["su2_long"]["trial_count"] == 32
    assert (
        summary["validation"]["initialization"][
            "base_su2_long_matched_initializations"
        ]
        == 32
    )
    drifts = summary["validation"]["initialization"][
        "su2_long_anchor_metadata_drifts"
    ]
    assert drifts == []
    assert all(row["numerical_anchor_and_hamiltonian_match"] for row in drifts)
    for case in summary["cases"].values():
        warm = case["matched_time_pareto"]["warm"]
        assert warm["su2_D6_source"].startswith("sealed 600-update")
        assert "su2_D6:long" in warm["su2_arm_labels"]
    json.dumps(summary, allow_nan=False)


def test_rejects_partial_duplicate_and_unregistered_campaigns(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, long = synthetic_campaigns
    running = deepcopy(long)
    running["status"] = "running"
    with pytest.raises(analysis.AnalysisValidationError, match="not sealed complete"):
        analysis.validate_prl_results(base, running)

    missing = deepcopy(base)
    missing["cases"][0]["trials"].pop()
    _reseal(missing)
    with pytest.raises(analysis.AnalysisValidationError, match="trial count"):
        analysis.validate_prl_results(missing, long)

    duplicate = deepcopy(base)
    duplicate["cases"][0]["trials"][-1] = deepcopy(
        duplicate["cases"][0]["trials"][0]
    )
    _reseal(duplicate)
    with pytest.raises(analysis.AnalysisValidationError, match="duplicate trial identity"):
        analysis.validate_prl_results(duplicate, long)


def test_long_control_requires_all_initialization_hashes_and_prefix(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, long = synthetic_campaigns
    changed_hash = deepcopy(long)
    changed_hash["cases"][0]["trials"][0]["initial_raw_amplitudes_sha256"] = "changed"
    _reseal(changed_hash)
    with pytest.raises(analysis.AnalysisValidationError, match="initialization mismatch"):
        analysis.validate_prl_results(base, changed_hash)

    changed_prefix = deepcopy(long)
    changed_prefix["cases"][0]["trials"][0]["training"]["history"][0][
        "loss_before_update"
    ] += 1.0e-5
    _reseal(changed_prefix)
    with pytest.raises(analysis.AnalysisValidationError, match="does not reproduce base prefix"):
        analysis.validate_prl_results(base, changed_prefix)


def test_physical_sum_rule_is_fail_closed(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, long = synthetic_campaigns
    broken = deepcopy(base)
    broken["cases"][0]["trials"][0]["metrics"]["physical_observables"][
        "spin_structure_factor_grid"
    ][0][1] += 0.01
    _reseal(broken)
    with pytest.raises(analysis.AnalysisValidationError, match="physical identity"):
        analysis.validate_prl_results(broken, long)


def test_warm_and_cold_clocks_ignore_observer_and_trial_wall(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, _ = synthetic_campaigns
    case = deepcopy(base["cases"][0])
    trial = deepcopy(case["trials"][0])
    trial["training"]["training_seconds"] = 10.0
    trial["training"]["compile_seconds"] = 7.0
    trial["compile_plus_optimizer_seconds"] = 18.0
    trial["training"]["history"] = [
        {
            "step": float(step),
            "loss_before_update": case["exact_energy_anchor"]
            + case["n_sites"] * (0.04 / (index + 1)),
            "gradient_norm": 1.0,
            "cumulative_training_seconds": timestamp,
        }
        for index, (step, timestamp) in enumerate(((10, 2.0), (20, 5.0), (30, 9.5)), 1)
    ]
    trial["final_observable_seconds"] = 1.0e9
    trial["trial_wall_seconds_excluding_case_preparation"] = 2.0e9
    warm_times, _ = analysis._trial_curve(case, trial, clock="warm")
    cold_times, _ = analysis._trial_curve(case, trial, clock="cold")
    np.testing.assert_allclose(warm_times, [0.0, 2.0, 5.0, 9.5, 10.0])
    # The complete cold clock assigns the measured setup+compile remainder
    # before the optimizer trajectory, so its endpoint is the authoritative
    # compile_plus_optimizer_seconds value.
    np.testing.assert_allclose(cold_times, [8.0, 10.0, 13.0, 17.5, 18.0])


def test_exact_8_plus_8_permutation_extremes_boundary_and_ties() -> None:
    separated = analysis.exact_two_sample_permutation(np.arange(9, 17), np.arange(1, 9))
    assert separated["u_statistic"] == 64.0
    assert separated["enumerated_allocations"] == math.comb(16, 8)
    assert separated["one_sided_p_value"] == pytest.approx(1.0 / math.comb(16, 8))

    ranks = np.arange(1, 17, dtype=float)
    for selected_ranks, expected_u, expected_p in (
        ([1, 3, 11, 12, 13, 14, 15, 16], 49.0, 0.04149184149184149),
        ([1, 2, 11, 12, 13, 14, 15, 16], 48.0, 0.05244755244755245),
    ):
        selected = np.asarray(selected_ranks) - 1
        mask = np.ones(16, dtype=bool)
        mask[selected] = False
        result = analysis.exact_two_sample_permutation(ranks[selected], ranks[mask])
        assert result["u_statistic"] == expected_u
        assert result["one_sided_p_value"] == pytest.approx(expected_p)

    tied = analysis.exact_two_sample_permutation(np.ones(8), np.ones(8))
    assert tied["u_statistic"] == 32.0
    assert tied["one_sided_p_value"] == 1.0
    with pytest.raises(ValueError, match="length-eight"):
        analysis.exact_two_sample_permutation(np.ones(7), np.ones(8))


def test_a1_diagnosis_reports_structural_and_floor_saturation(
    synthetic_campaigns: tuple[dict[str, object], dict[str, object]],
) -> None:
    base, _ = synthetic_campaigns
    square = next(
        case for case in base["cases"] if case["case_id"] == "square_4x4_j2_0p5_prl_pareto"
    )
    triangular = next(
        case
        for case in base["cases"]
        if case["case_id"] == "triangular_4x4_j2_0p125_prl_pareto"
    )
    square_result = analysis.a1_floor_diagnosis(square)
    triangular_result = analysis.a1_floor_diagnosis(triangular)
    assert square_result["sector_ranks"]["singlet"]["gamma_a1"] == 31
    assert triangular_result["sector_ranks"]["singlet"]["gamma_a1"] == 14
    assert square_result["small_sector_saturated"] is True
    assert square_result["floor_dominated"] is True


def test_bootstrap_recomputes_width_selection() -> None:
    # Shapes are width, seed, resource point.  Which width supplies the median
    # frontier changes when the independently resampled seed clusters change.
    u1 = np.asarray(
        [
            [[0.0], [0.0], [0.0], [0.0], [3.0], [3.0], [3.0], [3.0]],
            [[3.0], [3.0], [3.0], [3.0], [0.0], [0.0], [0.0], [0.0]],
        ]
    )
    su2 = np.full((1, 8, 1), 1.0)
    result = analysis._bootstrap_resource(
        u1,
        su2,
        np.ones(1),
        resamples=200,
        seed=17,
    )
    low, high = result["average_log10_advantage"]["percentile_95"]
    assert low < high
    assert result["method"].startswith("independent_backbone")


def test_official_prl_artifacts_and_frozen_decision_are_consistent() -> None:
    """Keep the paper-facing local decision tied to its three sealed inputs."""

    result_root = PROJECT / "results"
    base_path = result_root / "spin_prl_local_pareto_v1/results.json"
    u1_long_path = result_root / "spin_prl_local_u1_long_v1/results.json"
    su2_long_path = result_root / "spin_prl_local_su2_long_v1/results.json"
    summary_path = result_root / "spin_prl_local_analysis_v1/summary.json"

    base = json.loads(base_path.read_text())
    u1_long = json.loads(u1_long_path.read_text())
    su2_long = json.loads(su2_long_path.read_text())
    summary = json.loads(summary_path.read_text())
    validation = analysis.validate_prl_results(base, u1_long, su2_long)

    stored_summary_hash = summary.pop("summary_payload_sha256")
    assert stored_summary_hash == analysis._canonical_sha256(summary)
    summary["summary_payload_sha256"] = stored_summary_hash
    source_manifest = summary["analysis_provenance"]["source_sha256"]
    assert summary["analysis_provenance"][
        "source_manifest_sha256"
    ] == analysis._canonical_sha256(source_manifest)
    assert source_manifest == {
        "projects/clebsch_tree_nqs/analysis.py": hashlib.sha256(
            Path(analysis.__file__).read_bytes()
        ).hexdigest()
    }
    assert validation["passed"] is True
    assert validation["physics"]["checked_trials"] == 448
    assert len(
        validation["initialization"]["su2_long_anchor_metadata_drifts"]
    ) == 2
    assert all(
        drift["numerical_anchor_and_hamiltonian_match"]
        for drift in validation["initialization"][
            "su2_long_anchor_metadata_drifts"
        ]
    )
    assert summary["inputs"]["base_trial_manifest_sha256"] == base[
        "trial_manifest_sha256"
    ]
    assert summary["inputs"]["long_trial_manifest_sha256"] == u1_long[
        "trial_manifest_sha256"
    ]
    assert summary["inputs"]["su2_long_trial_manifest_sha256"] == su2_long[
        "trial_manifest_sha256"
    ]
    families = summary["promotion_gates"]["families"]
    assert families["parameter"]["passed_case_count"] == 4
    assert families["parameter"]["all_four_cases_pass"] is True
    assert families["parameter"]["claim_eligible"] is True
    assert families["warm"]["passed_case_count"] == 0
    assert families["cold"]["passed_case_count"] == 0
    assert families["warm"]["claim_eligible"] is False
    assert families["cold"]["claim_eligible"] is False
