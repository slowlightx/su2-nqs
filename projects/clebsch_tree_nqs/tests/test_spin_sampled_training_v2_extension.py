from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2 as scan_v2
from projects.clebsch_tree_nqs.workflows import (
    spin_sampled_training_v2_extension as extension,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    canonical_sha256,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_sampled_adam_center_only import (
    SampledAdamState,
    initialize_sampled_adam,
)
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    "spin_6x6_center_only_confirmatory_extension.json"
)


def _tiny_case_arm_runtime() -> tuple[dict[str, object], dict[str, object], runtime._ArmRuntime]:
    case = {
        "case_id": "square_2x2_extension_test",
        "geometry": "square",
        "rows": 2,
        "columns": 2,
        "periodic": True,
        "j2_over_j1": 0.0,
        "support": "fixed_m0",
        "ed": False,
        "coordinate_convention": "row_major_cartesian_column_row",
    }
    arm = {
        "arm_id": "u1_D1_translation_test",
        "backbone": "u1",
        "projection": "translation",
        "seed": 17,
        "model": {
            "feature_dim": 1,
            "maximum_intermediate_q": 4,
            "initialization_scale": 0.2,
        },
    }
    projector = PhysicalSpatialProjector(
        (tuple(range(4)),), orbit_chunk_size=1, rematerialize_orbit=True
    )
    return case, arm, runtime._initialize_arm(
        arm, rows=2, columns=2, projector=projector
    )


def _mini_protocol() -> dict[str, object]:
    protocol = deepcopy(extension._expected_protocol())
    protocol["fixed_measurement"].update(
        {
            "chains": 8,
            "parent_lineage_chains": 4,
            "independent_fresh_m0_chains": 4,
            "burn_sweeps": 2,
            "production_sweeps": 4,
            "raw_local_energy_shape": [4, 8],
        }
    )
    protocol["fixed_measurement_mixing_gates"].update(
        {
            "runtime_initial_positive_window_ess_minimum": 1.0,
            "production_sweeps_minimum_tau_multiple": 0.0,
            "split_rhat_strict_maximum": 100.0,
            "checkpoint_vs_fresh_each_observable_maximum_combined_mcse": 1.0e9,
            "energy_first_vs_second_half_maximum_combined_mcse": 1.0e9,
        }
    )
    return protocol


def test_official_config_and_all_four_parent_bindings_preflight() -> None:
    spec = json.loads(CONFIG.read_text())
    extension._validate_spec(spec)
    result = extension.preflight_spin_sampled_training_v2_extension(CONFIG)
    assert result["status"] == "preflight_complete"
    assert result["long_training_or_measurement_started"] is False
    assert result["selector_is_training_authority"] is False
    assert len(result["arms"]) == 4
    assert {
        (row["case_id"], row["backbone"], row["learning_rate"])
        for row in result["arms"]
    } == {
        ("square_6x6_j2_0p5", "u1", 2.0e-3),
        ("square_6x6_j2_0p5", "su2", 2.0e-3),
        ("triangular_6x6_j2_0p125", "u1", 1.0e-3),
        ("triangular_6x6_j2_0p125", "su2", 1.0e-3),
    }
    assert all(row["parent_params_adam_chains_key_verified"] for row in result["arms"])


def test_selector_binding_and_policy_mutations_fail_closed() -> None:
    spec = json.loads(CONFIG.read_text())
    bad = deepcopy(spec)
    bad["selector_is_training_authority"] = True
    with pytest.raises(ValueError, match="training authority"):
        extension._validate_spec(bad)

    bad = deepcopy(spec)
    bad["protocol"]["fixed_measurement"]["production_sweeps"] = 255
    with pytest.raises(ValueError, match="preregistration"):
        extension._validate_spec(bad)

    bad = deepcopy(spec)
    bad["selector_binding"]["result_file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="result file seal"):
        extension._audit_selector_and_parents(bad)


def test_long_measurement_stores_raw_arrays_slow_modes_and_attempt_ledger() -> None:
    case, _arm, arm_runtime = _tiny_case_arm_runtime()
    bonds = (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    )
    advance_uncached, advance_cached, hpsi, _center = scan_v2._compiled_kernels(
        arm_runtime, bonds
    )
    lineage = np.asarray(
        initialize_spin_exchange_chains(
            jax.random.PRNGKey(91), n_chains=4, n_sites=4, n_up=2
        ),
        dtype=np.int32,
    )
    protocol = _mini_protocol()
    measured, ledger = extension._fixed_confirmatory_measurement(
        runtime=arm_runtime,
        params=arm_runtime.params,
        lineage_states=lineage,
        case=case,
        bonds=bonds,
        protocol=protocol,
        measurement_seed=812,
        campaign_id="tiny-measurement",
        update=0,
        advance_uncached=advance_uncached,
        advance_cached=advance_cached,
        hpsi_fn=hpsi,
        microbatch_size=4,
        proposals_per_sweep=4,
        bond_proposal_probability=0.5,
    )
    real = np.asarray(measured["real_local_energy_raw"], dtype=np.float64)
    imaginary = np.asarray(measured["imaginary_local_energy_raw"], dtype=np.float64)
    assert real.shape == imaginary.shape == (4, 8)
    assert measured["real_local_energy_raw_sha256"] == array_sha256(real)
    assert measured["imaginary_local_energy_raw_sha256"] == array_sha256(imaginary)
    assert set(measured["diagonal_slow_mode_raw"]) == {
        "diagonal_Szz_Neel_Q",
        "diagonal_Szz_stripe_star",
    }
    for item in measured["diagonal_slow_mode_raw"].values():
        values = np.asarray(item["values"], dtype=np.float64)
        assert values.shape == (4, 8)
        assert item["sha256"] == array_sha256(values)
    assert measured["group_label_counts"] == {
        "parent_lineage": 4,
        "fresh_random_m0": 4,
    }
    diagnostics = measured["sampler_diagnostics_attempt_weighted"]
    assert diagnostics["aggregation"].startswith("proposal_attempt_count_weighted")
    assert diagnostics["proposal_attempts"] == 8 * 4 * (2 + 4)
    assert ledger["fixed_parameter_energy_samples"] == 4 * 8
    assert measured["full_spin_structure_factor_computed"] is False
    assert all(
        row["rank_normalized_bulk_ess"] is None
        for row in measured["autocorrelation"].values()
    )


def test_paired_endpoint_gates_use_all_64_fixed_chain_indices() -> None:
    protocol = extension._expected_protocol()
    pattern = np.linspace(-0.002, 0.002, 64)
    energy0 = -0.30 + pattern
    energy100 = energy0 - 0.06 - 0.0001 * pattern
    variance0 = 0.40 + pattern
    variance100 = variance0 - 0.12 - 0.0001 * pattern
    baseline = {
        "energy_per_site_chain_means": energy0.tolist(),
        "variance_h_per_site_chain_contributions": variance0.tolist(),
    }
    final = {
        "energy_per_site_chain_means": energy100.tolist(),
        "variance_h_per_site_chain_contributions": variance100.tolist(),
    }
    result = extension._paired_final_optimization_gates(
        baseline, final, protocol
    )
    assert result["energy_lowering_per_site"] > 0.05
    assert result["energy_lowering_paired_z"] > 5.0
    assert result["variance_fractional_decrease"] > 0.25
    assert result["variance_decrease_paired_z"] > 3.0
    assert result["all_final_optimization_gates_passed"] is True


def test_four_arm_route_hierarchy_is_fail_closed() -> None:
    passed = [
        {
            "status": "complete",
            "all_mixing_and_training_stability_gates_passed": True,
            "all_final_optimization_gates_passed": True,
        }
        for _ in range(4)
    ]
    assert extension._route_decision(passed)["next_route"] == "heldout"
    optimization_failed = deepcopy(passed)
    optimization_failed[2]["all_final_optimization_gates_passed"] = False
    assert (
        extension._route_decision(optimization_failed)["next_route"]
        == "matrix_free_SR"
    )
    mixing_failed = deepcopy(passed)
    mixing_failed[0]["all_mixing_and_training_stability_gates_passed"] = False
    assert extension._route_decision(mixing_failed)["next_route"] == "sampler_audit"
    assert extension._route_decision(passed[:3])["next_route"] == "sampler_audit"


def test_top_level_four_arm_manifest_and_complete_result_seal() -> None:
    cases = []
    for case_id in ("square", "triangular"):
        trials = []
        for arm_id in ("u1", "su2"):
            trial = {
                "status": "complete",
                "arm_id": arm_id,
                "seed": 11,
                "all_mixing_and_training_stability_gates_passed": True,
                "all_final_optimization_gates_passed": True,
            }
            trial["trial_payload_sha256"] = trial_payload_sha256(trial)
            trials.append(trial)
        cases.append({"case_id": case_id, "trials": trials})
    payload = {
        "schema_version": 1,
        "task": extension.TASK,
        "status": "running",
        "cases": cases,
    }
    finalized = extension._finalize_extension_payload(payload)
    assert finalized["status"] == "complete"
    assert finalized["route_decision"]["next_route"] == "heldout"
    assert len(finalized["trial_manifest"]) == 4
    assert finalized["trial_manifest_sha256"] == canonical_sha256(
        finalized["trial_manifest"]
    )
    assert finalized["result_payload_sha256"] == extension._result_payload_sha256(
        finalized
    )


def _fake_fixed_measurement(*, update: int, **_kwargs):
    pattern = np.linspace(-1.0e-3, 1.0e-3, 64)
    energy = -0.30 - 0.001 * update + pattern
    variance = 0.40 - 0.0015 * update + 0.1 * pattern
    return (
        {
            "update": update,
            "energy_per_site_chain_means": energy.tolist(),
            "energy_per_site_chain_means_sha256": array_sha256(energy),
            "variance_h_per_site_chain_contributions": variance.tolist(),
            "variance_h_per_site_chain_contributions_sha256": array_sha256(variance),
            "all_mixing_gates_passed": True,
        },
        {"fixed_parameter_energy_samples": 64},
    )


def _tiny_parent(tmp_path: Path):
    case, arm, arm_runtime = _tiny_case_arm_runtime()
    params = arm_runtime.params
    zero_state = initialize_sampled_adam(params)
    adam50 = SampledAdamState(
        step=50,
        first_moment=jax.tree_util.tree_map(jnp.zeros_like, params),
        second_moment=jax.tree_util.tree_map(jnp.zeros_like, params),
    )
    states = np.asarray(
        initialize_spin_exchange_chains(
            jax.random.PRNGKey(33), n_chains=32, n_sites=4, n_up=2
        ),
        dtype=np.int32,
    )
    key = np.asarray(jax.random.PRNGKey(44), dtype=np.uint32)
    identity = {"synthetic_parent": True}
    metadata = {
        "chain_states": states,
        "chain_states_sha256": array_sha256(states),
        "sampling_key": key,
        "sampling_key_sha256": array_sha256(key),
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(adam50),
    }
    step50 = {
        "params": params,
        "optimizer_state": adam50,
        "completed_steps": 50,
        "next_sample_index": 1600,
        "run_identity": identity,
        "history": [{"update": index + 1} for index in range(50)],
        "metadata": metadata,
    }
    metadata0 = {
        **metadata,
        "optimizer_state_sha256": pytree_sha256(zero_state),
    }
    step0 = {
        "params": params,
        "optimizer_state": zero_state,
        "completed_steps": 0,
        "next_sample_index": 0,
        "run_identity": identity,
        "history": [],
        "metadata": metadata0,
    }
    binding = {
        "result_file_sha256": "parent-result",
        "config_sha256": "parent-config",
        "source_manifest_sha256": "parent-source",
        "runtime_identity_sha256": "parent-runtime",
        "trial_payload_sha256": "parent-trial",
        "checkpoint_generation_sha256": "parent-step50",
        "checkpoint_run_identity_sha256": canonical_sha256(identity),
        "params_sha256": pytree_sha256(params),
        "optimizer_state_sha256": pytree_sha256(adam50),
        "chain_states_sha256": array_sha256(states),
        "sampling_key_sha256": array_sha256(key),
    }
    parent_config = {
        "projection": {
            "kind": "translation",
            "orbit_chunk_size": 1,
            "rematerialize_orbit": True,
        },
        "sampler": {
            "chains": 32,
            "decorrelation_sweeps_per_update": 2,
            "proposals_per_sweep": 4,
            "bond_proposal_probability": 0.5,
        },
        "optimizer": {
            "learning_rate": 1.0e-5,
            "maximum_relative_update_norm": 0.05,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
        },
    }
    step0_path = tmp_path / "synthetic_step0.pkl"
    step0_path.write_bytes(b"synthetic test binding only")
    parent = {
        "binding": binding,
        "arm_spec": {"seed": 17},
        "parent_config": parent_config,
        "parent_trial": {
            "fixed_parameter_measurements": [{"update": 50, "synthetic": True}]
        },
        "step50_payload": step50,
        "step0_payload": step0,
        "step0_path": step0_path,
        "step0_path_sha256": "synthetic-step0",
    }
    return case, arm_runtime, parent


def _tiny_extension_spec() -> dict[str, object]:
    return {
        "campaign_id": "tiny-extension-resume",
        "measurement_seed": 12,
        "protocol": extension._expected_protocol(),
    }


def _run_tiny_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    directory_name: str,
    overwrite: bool,
    stop_after_update: int,
):
    monkeypatch.setattr(
        extension, "_fixed_confirmatory_measurement", _fake_fixed_measurement
    )
    case, arm_runtime, parent = _tiny_parent(tmp_path)
    bonds = (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    )
    return extension._run_extension_arm(
        runtime=arm_runtime,
        parent=parent,
        case=case,
        bonds=bonds,
        spec=_tiny_extension_spec(),
        selector_decision_sha256="tiny-selector-decision",
        checkpoint_directory=tmp_path / directory_name,
        config_hash="tiny-config",
        source_hash="tiny-source",
        runtime_hash="tiny-runtime",
        overwrite=overwrite,
        stop_after_update=stop_after_update,
    )


def test_generation_resume_matches_uninterrupted_params_adam_chains_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uninterrupted = _run_tiny_extension(
        tmp_path,
        monkeypatch,
        directory_name="uninterrupted",
        overwrite=True,
        stop_after_update=100,
    )
    first = _run_tiny_extension(
        tmp_path,
        monkeypatch,
        directory_name="resumed",
        overwrite=True,
        stop_after_update=75,
    )
    assert first["status"] == "running"
    resumed = _run_tiny_extension(
        tmp_path,
        monkeypatch,
        directory_name="resumed",
        overwrite=False,
        stop_after_update=100,
    )
    assert uninterrupted["status"] == resumed["status"] == "complete"
    full_checkpoint = load_latest_resumable_checkpoint(tmp_path / "uninterrupted")
    resumed_checkpoint = load_latest_resumable_checkpoint(tmp_path / "resumed")
    assert pytree_sha256(full_checkpoint["params"]) == pytree_sha256(
        resumed_checkpoint["params"]
    )
    assert pytree_sha256(full_checkpoint["optimizer_state"]) == pytree_sha256(
        resumed_checkpoint["optimizer_state"]
    )
    assert array_sha256(full_checkpoint["metadata"]["chain_states"]) == array_sha256(
        resumed_checkpoint["metadata"]["chain_states"]
    )
    assert array_sha256(full_checkpoint["metadata"]["sampling_key"]) == array_sha256(
        resumed_checkpoint["metadata"]["sampling_key"]
    )
    assert full_checkpoint["history"] == resumed_checkpoint["history"]
    assert len(full_checkpoint["history"]) == 100
    assert full_checkpoint["completed_steps"] == resumed_checkpoint["completed_steps"] == 100
