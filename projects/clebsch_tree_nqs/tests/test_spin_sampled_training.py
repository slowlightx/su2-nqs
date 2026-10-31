from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime
from projects.clebsch_tree_nqs.workflows import spin_sampled_training as training
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.artifacts import embedded_trial_manifest
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    / "spin_6x6_sampled_training_preflight.json"
)
PILOT_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    / "spin_6x6_sampled_training_pilot.json"
)
OFFICIAL_RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    / "spin_6x6_sampled_training_preflight_v1/results.json"
)


def _registered_spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def _tiny_spec() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    case = {
        "case_id": "square_2x2_sampled_training_test",
        "geometry": "square",
        "rows": 2,
        "columns": 2,
        "periodic": True,
        "j2_over_j1": 0.0,
        "support": "fixed_m0",
        "ed": False,
        "coordinate_convention": "row_major_cartesian_column_row",
        "modes": [
            {"name": "q0", "wavevector": [0.0, 0.0]},
            {"name": "neel", "wavevector": [np.pi, np.pi]},
            {"name": "stripe_x", "wavevector": [np.pi, 0.0]},
            {"name": "stripe_y", "wavevector": [0.0, np.pi]},
        ],
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
    spec = {
        "schema_version": 1,
        "domain": "spin",
        "task": "physical_sampled_training",
        "campaign_stage": "readiness_preflight",
        "claim_boundary": training.CLAIM_BOUNDARY,
        "cases": [case],
        "projection": {
            "kind": "translation",
            "orbit_chunk_size": 1,
            "rematerialize_orbit": True,
        },
        "arms": [arm],
        "sampler": {
            "chains": 8,
            "initial_burn_sweeps": 1,
            "decorrelation_sweeps_per_update": 1,
            "proposals_per_sweep": 4,
            "bond_proposal_probability": 0.5,
            "seed": 101,
        },
        "optimizer": {
            "name": "local_adam",
            "updates": 2,
            "learning_rate": 1.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "microbatch_size": 4,
            "maximum_relative_update_norm": 0.1,
            "checkpoint_every_updates": 1,
        },
        "measurement": {
            "every_updates": 1,
            "burn_sweeps": 1,
            "measurement_sweeps": 2,
            "observable_batch_size": 4,
            "seed": 202,
            "su2_q0_local_tolerance": 1.0e-8,
        },
    }
    return spec, case, arm


def _tiny_runtime(arm: dict[str, object]) -> runtime._ArmRuntime:
    identity = tuple(range(4))
    projector = PhysicalSpatialProjector(
        (identity,), orbit_chunk_size=1, rematerialize_orbit=True
    )
    return runtime._initialize_arm(arm, rows=2, columns=2, projector=projector)


def _run_tiny(
    directory: Path,
    *,
    overwrite: bool,
    maximum_updates_this_invocation: int | None = None,
) -> dict[str, object]:
    spec, case, arm = _tiny_spec()
    model = _tiny_runtime(arm)
    bonds = (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    )
    group = (tuple(range(4)),)
    return training._run_training_arm(
        runtime=model,
        arm=arm,
        case=case,
        bonds=bonds,
        group=group,
        spec=spec,
        checkpoint_directory=directory,
        config_hash="tiny-config",
        source_hash="tiny-source",
        runtime_hash="tiny-runtime",
        overwrite=overwrite,
        maximum_updates_this_invocation=maximum_updates_this_invocation,
    )


def test_registered_training_config_is_two_geometry_two_update_preflight() -> None:
    spec = _registered_spec()
    training._validate_spec(spec)
    assert {case["geometry"] for case in spec["cases"]} == {
        "square",
        "triangular",
    }
    assert all(case["rows"] == case["columns"] == 6 for case in spec["cases"])
    assert {arm["backbone"] for arm in spec["arms"]} == {"u1", "su2"}
    assert all(arm["model"]["feature_dim"] == 4 for arm in spec["arms"])
    assert spec["optimizer"]["updates"] == 2
    assert spec["optimizer"]["microbatch_size"] == 8
    assert spec["sampler"]["chains"] == 16
    assert spec["projection"]["kind"] == "translation"

    broken = deepcopy(spec)
    broken["optimizer"]["microbatch_size"] = 16
    with pytest.raises(ValueError, match="at most eight"):
        training._validate_spec(broken)


def test_registered_exploratory_pilot_keeps_the_predeclared_nonclaim_protocol() -> None:
    spec = json.loads(PILOT_CONFIG.read_text())
    training._validate_spec(spec)
    assert spec["campaign_stage"] == "pilot"
    assert spec["claim_boundary"] == training.CLAIM_BOUNDARY
    assert spec["sampler"] == {
        "chains": 32,
        "initial_burn_sweeps": 16,
        "decorrelation_sweeps_per_update": 2,
        "proposals_per_sweep": 36,
        "bond_proposal_probability": 0.5,
        "seed": 20260918,
    }
    assert spec["optimizer"]["updates"] == 20
    assert spec["optimizer"]["learning_rate"] == 5.0e-4
    assert spec["optimizer"]["microbatch_size"] == 8
    assert spec["optimizer"]["maximum_relative_update_norm"] == 0.05
    assert spec["optimizer"]["checkpoint_every_updates"] == 5
    assert spec["measurement"]["every_updates"] == 5
    assert spec["measurement"]["burn_sweeps"] == 4
    assert spec["measurement"]["measurement_sweeps"] == 8
    assert spec["measurement"]["observable_batch_size"] == 4


def test_training_source_closure_uses_active_physical_route_only() -> None:
    manifest = training._source_manifest()
    for required in (
        "projects/clebsch_tree_nqs/workflows/spin_sampled_training.py",
        "projects/clebsch_tree_nqs/workflows/spin_sampled_runtime.py",
        "su2_nqs/vmc/physical_sampled_adam.py",
        "su2_nqs/vmc/resumable_checkpoint.py",
        "su2_nqs/vmc/spin_observables.py",
    ):
        assert required in manifest
    forbidden_paths = {
        f"{prefix.replace('.', '/')}.py"
        for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
    }
    assert forbidden_paths.isdisjoint(manifest)


def test_cli_registers_sampled_training_command() -> None:
    source = (ROOT / "projects/clebsch_tree_nqs/run.py").read_text()
    assert '"sampled-training"' in source
    assert "run_spin_sampled_training" in source


def test_controlled_resume_matches_uninterrupted_state_and_never_caches_amplitudes(
    tmp_path: Path,
) -> None:
    resumed_directory = tmp_path / "resumed"
    first = _run_tiny(
        resumed_directory,
        overwrite=True,
        maximum_updates_this_invocation=1,
    )
    assert first["status"] == "running"
    assert first["completed_updates"] == 1
    assert first["resume_count"] == 0

    resumed = _run_tiny(resumed_directory, overwrite=False)
    uninterrupted = _run_tiny(tmp_path / "uninterrupted", overwrite=True)
    assert resumed["final_params_sha256"] == uninterrupted["final_params_sha256"]
    assert resumed["final_chain_states_sha256"] == uninterrupted[
        "final_chain_states_sha256"
    ]
    assert resumed["checkpoint"]["optimizer_state_sha256"] == uninterrupted[
        "checkpoint"
    ]["optimizer_state_sha256"]
    assert resumed["sampler"]["resume_count"] == 1
    assert resumed["timing"]["timing_claim_eligible"] is False
    assert uninterrupted["sampler"]["resume_count"] == 0
    assert uninterrupted["timing"]["timing_claim_eligible"] is True
    for resumed_row, direct_row in zip(
        resumed["history"], uninterrupted["history"], strict=True
    ):
        for key in (
            "params_before_sha256",
            "params_after_sha256",
            "chain_states_sha256",
            "pre_update_sampled_energy",
            "pre_update_sampled_variance",
        ):
            assert resumed_row[key] == direct_row[key]
        assert resumed_row["post_update_amplitude_cache"] == (
            "discarded_and_not_checkpointed"
        )

    identity = training._run_identity(
        config_hash="tiny-config",
        source_hash="tiny-source",
        runtime_hash="tiny-runtime",
        case=_tiny_spec()[1],
        arm=_tiny_spec()[2],
        runtime=_tiny_runtime(_tiny_spec()[2]),
        bonds=(
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ),
        group=(tuple(range(4)),),
        initial_chain_states_sha256=resumed["initial_chain_states_sha256"],
        spec=_tiny_spec()[0],
    )
    checkpoint = load_latest_resumable_checkpoint(
        resumed_directory, expected_run_identity=identity
    )
    metadata = checkpoint["metadata"]
    assert metadata["amplitude_cache_present"] is False
    assert "amplitudes" not in metadata
    assert metadata["resume_count"] == 1
    assert checkpoint["completed_steps"] == 2
    assert canonical_sha256(identity) != ""


def test_fixed_parameter_observables_use_post_update_params_without_pooled_ess(
    tmp_path: Path,
) -> None:
    """Audit the physics/statistics boundary of each measurement block."""

    trial = _run_tiny(tmp_path / "fixed_parameter_audit", overwrite=True)
    spec, _case, _arm = _tiny_spec()
    expected_samples = (
        int(spec["sampler"]["chains"])
        * int(spec["measurement"]["measurement_sweeps"])
    )
    assert trial["claim_boundary"] == training.CLAIM_BOUNDARY
    assert "no convergence" in trial["claim_boundary"]
    assert "phase" in trial["claim_boundary"]
    assert "ESS is not combined" in trial["observable_claim_boundary"]

    for row in trial["history"]:
        measurement = row["fixed_parameter_measurement"]
        assert measurement is not None
        assert measurement["fixed_params_sha256"] == row["params_after_sha256"]
        assert measurement["params_frozen_throughout_block"] is True
        assert measurement["training_chains_copied_and_not_mutated"] is True
        assert measurement["independent_measurement_random_stream"] is True
        assert measurement["samples_are_autocorrelated"] is True
        assert measurement["ess_combined_across_optimizer_steps"] is False
        structure = measurement["full_spin_structure_factor"]
        assert structure["operator"] == "full_SU2_invariant_spin_structure_factor"
        assert structure["normalization"] == "N^-1_sum_ij"
        assert all(
            summary["sample_count"] == expected_samples
            for summary in structure["mode_summaries"].values()
        )
        assert row["post_update_amplitude_cache"] == (
            "discarded_and_not_checkpointed"
        )

    assert trial["final_fixed_parameter_measurement"]["update"] == len(
        trial["history"]
    )
    assert trial["work_ledger"][
        "parameter_count_is_not_used_as_a_work_substitute"
    ] is True


def test_official_6x6_training_preflight_is_current_sealed_and_readiness_only() -> None:
    payload = json.loads(OFFICIAL_RESULT.read_text())
    result_directory = OFFICIAL_RESULT.parent
    assert payload["schema_version"] == training.TRAINING_RESULT_SCHEMA_VERSION
    assert payload["status"] == "complete"
    assert payload["task"] == "physical_sampled_training"
    assert payload["campaign_stage"] == "readiness_preflight"
    assert payload["claim_boundary"] == training.CLAIM_BOUNDARY
    assert payload["config_sha256"] == canonical_sha256(payload["config"])
    current_source = training._source_manifest()
    assert payload["source_sha256"] == current_source
    assert payload["source_manifest_sha256"] == canonical_sha256(current_source)
    assert payload["runtime_identity_sha256"] == canonical_sha256(
        payload["runtime_identity"]
    )
    manifest = embedded_trial_manifest(payload)
    assert payload["trial_manifest"] == manifest
    assert payload["trial_manifest_sha256"] == canonical_sha256(manifest)
    assert payload["all_trials_training_ready"] is True
    assert payload["route_boundary"]["direct_fusion_path_used"] is False
    assert payload["route_boundary"]["energy_or_physics_win_claimed"] is False
    assert len(payload["cases"]) == 2
    assert {case["geometry"] for case in payload["cases"]} == {
        "square",
        "triangular",
    }

    count_keys = (
        "training_samples",
        "sampler_proposal_attempts",
        "sampler_projected_amplitude_evaluations",
        "hpsi_two_pass_projected_amplitude_evaluations",
        "gradient_backward_microbatch_invocations",
        "base_tree_orbit_image_work_proxy",
        "measurement_sampler_projected_amplitude_evaluations",
        "measurement_hpsi_projected_amplitude_evaluations",
        "full_spin_observable_projected_amplitude_evaluations",
        "measurement_base_tree_orbit_image_work_proxy",
    )
    for case in payload["cases"]:
        trials = case["trials"]
        assert {trial["backbone"] for trial in trials} == {"u1", "su2"}
        assert len(trials) == 2
        reference_work = {key: trials[0]["work_ledger"][key] for key in count_keys}
        for trial in trials:
            assert trial["all_training_readiness_gates_passed"] is True
            assert all(trial["readiness_gates"].values())
            assert trial["optimizer"]["completed_updates"] == 2
            assert trial["sampler"]["resume_count"] == 0
            assert trial["timing"]["timing_claim_eligible"] is True
            assert trial["claim_boundary"] == training.CLAIM_BOUNDARY
            assert len(trial["history"]) == 2
            assert "energy_decreased" not in trial["readiness_gates"]
            for row in trial["history"]:
                measurement = row["fixed_parameter_measurement"]
                assert measurement["fixed_params_sha256"] == row[
                    "params_after_sha256"
                ]
                assert measurement["ess_combined_across_optimizer_steps"] is False
                assert measurement["full_spin_structure_factor"]["normalization"] == (
                    "N^-1_sum_ij"
                )
                assert row["post_update_amplitude_cache"] == (
                    "discarded_and_not_checkpointed"
                )
            if trial["backbone"] == "su2":
                assert trial["final_fixed_parameter_measurement"][
                    "full_spin_structure_factor"
                ]["q0_maximum_abs_local"] <= payload["config"]["measurement"][
                    "su2_q0_local_tolerance"
                ]
            observed_work = {
                key: trial["work_ledger"][key] for key in count_keys
            }
            assert observed_work == reference_work

            checkpoint_directory = result_directory / trial["checkpoint"][
                "series_directory"
            ]
            checkpoint = load_latest_resumable_checkpoint(checkpoint_directory)
            metadata = checkpoint["metadata"]
            assert checkpoint["completed_steps"] == 2
            assert checkpoint["run_identity_sha256"] == trial["checkpoint"][
                "run_identity_sha256"
            ]
            assert metadata["params_sha256"] == trial["final_params_sha256"]
            assert metadata["chain_states_sha256"] == trial[
                "final_chain_states_sha256"
            ]
            assert metadata["amplitude_cache_present"] is False
            assert "amplitudes" not in metadata
            generation = Path(checkpoint["checkpoint_generation_path"])
            assert generation.name == trial["checkpoint"]["latest_generation"]
            assert file_sha256(generation) == trial["checkpoint"][
                "latest_generation_sha256"
            ]
