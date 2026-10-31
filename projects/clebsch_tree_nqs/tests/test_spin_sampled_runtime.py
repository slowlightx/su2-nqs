from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime
from su2_nqs.frontier.artifacts import (
    array_sha256,
    canonical_sha256,
    embedded_trial_manifest,
    trial_payload_sha256,
)
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES


TRIANGULAR_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    / "spin_triangular_6x6_runtime_smoke.json"
)
SQUARE_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/"
    / "spin_square_6x6_runtime_smoke.json"
)
RESULT_ROOT = ROOT / "projects/clebsch_tree_nqs/results"


def test_official_runtime_config_is_fail_closed_and_has_requested_protocol() -> None:
    spec = json.loads(TRIANGULAR_CONFIG.read_text())
    runtime._validate_spec(spec)
    assert spec["case"] == {
        "case_id": "triangular_6x6_j2_0p125_translation_runtime_smoke",
        "geometry": "triangular",
        "rows": 6,
        "columns": 6,
        "periodic": True,
        "j2_over_j1": 0.125,
        "support": "fixed_m0",
        "ed": False,
    }
    assert {arm["backbone"] for arm in spec["arms"]} == {"u1", "su2"}
    assert all(arm["projection"] == "translation" for arm in spec["arms"])
    assert all(arm["model"]["feature_dim"] == 4 for arm in spec["arms"])
    assert all(
        arm["model"]["maximum_intermediate_q"] == 8 for arm in spec["arms"]
    )
    assert spec["sampler"]["chains"] == 16
    assert spec["sampler"]["burn_in_sweeps"] == 16
    assert spec["sampler"]["measurement_sweeps"] == 64
    assert spec["sampler"]["proposals_per_sweep"] == 36
    assert spec["sampler"]["bond_proposal_probability"] == 0.5
    assert spec["profile"]["batch_sizes"] == [1, 2, 4, 8]
    assert spec["observable_contract"] == {
        "basis_estimator": "diagonal_Szz_from_sampled_computational_basis_configurations",
        "reported_estimators": ["diagonal_Szz_K_star", "diagonal_Szz_M_star"],
        "su2_invariant_structure_factor": False,
        "interpretation": (
            "Autocorrelation slow-mode diagnostic only; not a full spin "
            "structure factor or phase estimate."
        ),
    }
    square = json.loads(SQUARE_CONFIG.read_text())
    runtime._validate_spec(square)
    assert square["case"]["geometry"] == "square"
    assert square["case"]["j2_over_j1"] == 0.5
    assert square["case"]["ed"] is False
    assert square["observable_contract"]["reported_estimators"] == [
        "diagonal_Szz_Neel_Q",
        "diagonal_Szz_stripe_star",
    ]
    assert square["sampler"] == spec["sampler"]
    assert square["profile"] == spec["profile"]


def test_runtime_source_closure_uses_no_frozen_direct_fusion_route() -> None:
    manifest = runtime._source_manifest()
    assert "projects/clebsch_tree_nqs/workflows/spin_sampled_runtime.py" in manifest
    for required in (
        "su2_nqs/models/spatial_projector.py",
        "su2_nqs/vmc/physical_exchange.py",
        "su2_nqs/vmc/physical_vmc.py",
        "su2_nqs/vmc/spin_physical.py",
    ):
        assert required in manifest
    forbidden = {
        f"{prefix.replace('.', '/')}.py"
        for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
    }
    assert forbidden.isdisjoint(manifest)


def test_k_m_diagnostics_and_autocorrelation_contract() -> None:
    stripe = np.asarray(
        [[(row % 2) for row in range(6) for _column in range(6)]],
        dtype=np.int32,
    )
    triangular = runtime._ordering_structure_factor_series(
        stripe, geometry="triangular", rows=6, columns=6
    )
    np.testing.assert_allclose(
        triangular["diagonal_Szz_K_star"], 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        triangular["diagonal_Szz_M_star"], 9.0, atol=1.0e-14
    )
    square = runtime._ordering_structure_factor_series(
        stripe, geometry="square", rows=6, columns=6
    )
    np.testing.assert_allclose(
        square["diagonal_Szz_Neel_Q"], 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        square["diagonal_Szz_stripe_star"], 9.0, atol=1.0e-14
    )

    constant = np.ones((8, 4), dtype=np.float64)
    independent = runtime._autocorrelation_summary(constant)
    assert independent["tau_int"] == 1.0
    assert independent["ess"] == 32.0
    slow = np.repeat(np.arange(8, dtype=np.float64), 4)[:, None]
    slow = np.repeat(slow, 4, axis=1)
    correlated = runtime._autocorrelation_summary(slow)
    assert correlated["tau_int"] > 1.0
    assert correlated["ess"] < slow.size


def test_atomic_checkpoint_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    identity = {
        "config_sha256": "config",
        "source_manifest_sha256": "source",
        "runtime_identity_sha256": "runtime",
        "case_id": "case",
        "arm_id": "arm",
        "backbone": "su2",
        "initial_params_sha256": "params",
    }
    written = runtime._write_checkpoint(
        path,
        identity=identity,
        initial_chain_states_sha256="initial-states",
        common_profile_states_sha256="profile-states",
        status="running",
        states=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        amplitudes=np.asarray([1.0 + 2.0j, -3.0j]),
        key=np.asarray([3, 7], dtype=np.uint32),
        burn_completed=2,
        measurement_completed=1,
        burn_diagnostics=[{"accepted": 0.5}],
        measurement_diagnostics=[{"accepted": 0.25}],
        energy_series=[[-1.0, -2.0]],
        ordering_first_series=[[0.1, 0.2]],
        ordering_second_series=[[0.3, 0.4]],
        snapshot_changed=[[True, False]],
        performance_profile=[{"batch_size": 1}],
        resume_count=0,
    )
    loaded = runtime._load_checkpoint(
        path,
        identity,
        initial_chain_states_sha256="initial-states",
        common_profile_states_sha256="profile-states",
    )
    assert loaded == written
    assert loaded["schema_version"] == runtime.RUNTIME_CHECKPOINT_SCHEMA_VERSION
    assert loaded["initial_chain_states_sha256"] == "initial-states"
    assert loaded["common_profile_states_sha256"] == "profile-states"
    np.testing.assert_array_equal(loaded["states"], [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        runtime._pairs_complex(loaded["amplitudes"]), [1.0 + 2.0j, -3.0j]
    )
    with pytest.raises(RuntimeError, match="initial-chain identity mismatch"):
        runtime._load_checkpoint(
            path,
            identity,
            initial_chain_states_sha256="wrong-initial-states",
            common_profile_states_sha256="profile-states",
        )
    with pytest.raises(RuntimeError, match="profile-state identity mismatch"):
        runtime._load_checkpoint(
            path,
            identity,
            initial_chain_states_sha256="initial-states",
            common_profile_states_sha256="wrong-profile-states",
        )

    legacy = dict(written)
    legacy.pop("checkpoint_sha256")
    legacy["schema_version"] = 1
    legacy["checkpoint_sha256"] = canonical_sha256(legacy)
    path.write_text(json.dumps(legacy))
    with pytest.raises(RuntimeError, match="checkpoint schema mismatch"):
        runtime._load_checkpoint(
            path,
            identity,
            initial_chain_states_sha256="initial-states",
            common_profile_states_sha256="profile-states",
        )

    path.write_text(json.dumps(written))
    tampered = json.loads(path.read_text())
    tampered["burn_completed"] = 3
    path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="checkpoint seal failed"):
        runtime._load_checkpoint(
            path,
            identity,
            initial_chain_states_sha256="initial-states",
            common_profile_states_sha256="profile-states",
        )


def test_controlled_runtime_resume_preserves_origin_hashes_and_moves_chains(
    tmp_path: Path,
) -> None:
    """Exercise the real resume branch without a production-size calculation."""

    rows = columns = 2
    # The identity projector keeps this test focused on checkpoint mechanics;
    # the official artifacts separately cover the full translation projector.
    group = (tuple(range(rows * columns)),)
    projector = runtime.PhysicalSpatialProjector(
        group,
        orbit_chunk_size=1,
        rematerialize_orbit=True,
    )
    arm = {
        "arm_id": "u1_D1_translation_resume_test",
        "backbone": "u1",
        "projection": "translation",
        "seed": 11,
        "model": {
            "feature_dim": 1,
            "maximum_intermediate_q": 4,
            "initialization_scale": 0.2,
        },
    }
    arm_runtime = runtime._initialize_arm(
        arm,
        rows=rows,
        columns=columns,
        projector=projector,
    )
    spec = {
        "arms": [arm],
        "sampler": {
            "chains": 16,
            "burn_in_sweeps": 2,
            "measurement_sweeps": 2,
            "proposals_per_sweep": 4,
            "bond_proposal_probability": 0.5,
            "seed": 20260916,
            "checkpoint_every_sweeps": 1,
        },
        "profile": {"batch_sizes": [1, 2, 4, 8], "warm_repeats": 1},
        "observable_contract": {
            "basis_estimator": (
                "diagonal_Szz_from_sampled_computational_basis_configurations"
            ),
            "reported_estimators": [
                "diagonal_Szz_Neel_Q",
                "diagonal_Szz_stripe_star",
            ],
            "su2_invariant_structure_factor": False,
            "interpretation": "resume-test slow modes only",
        },
    }
    case = {
        "case_id": "square_2x2_resume_test",
        "geometry": "square",
        "rows": rows,
        "columns": columns,
        "j2_over_j1": 0.0,
    }
    bonds = (
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    )
    identity = {
        "config_sha256": "controlled-resume-config",
        "source_manifest_sha256": "controlled-resume-source",
        "runtime_identity_sha256": "controlled-resume-runtime",
        "case_id": case["case_id"],
        "arm_id": arm["arm_id"],
        "backbone": arm["backbone"],
        "initial_params_sha256": runtime.pytree_sha256(arm_runtime.params),
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    fresh_trial = runtime._run_arm(
        arm_runtime,
        spec=spec,
        case=case,
        bonds=bonds,
        checkpoint_path=checkpoint_path,
        identity=identity,
        overwrite=True,
    )
    fresh_checkpoint = json.loads(checkpoint_path.read_text())
    initial_key = runtime.jax.random.PRNGKey(spec["sampler"]["seed"])
    chain_key, _ = runtime.jax.random.split(initial_key)
    original_states = np.asarray(
        runtime.initialize_spin_exchange_chains(
            chain_key,
            n_chains=spec["sampler"]["chains"],
            n_sites=rows * columns,
            n_up=rows * columns // 2,
        )
    )
    assert np.any(np.asarray(fresh_checkpoint["states"]) != original_states)

    initial_hash = fresh_trial["initial_chain_states_sha256"]
    profile_hash = fresh_trial["common_profile_states_sha256"]
    assert initial_hash == array_sha256(original_states)
    assert initial_hash != profile_hash

    # Turn the complete checkpoint into a sealed start-of-run checkpoint.  The
    # state/amplitude/key tuple is internally consistent, so the real resume
    # branch deterministically replays the configured burn and measurements.
    partial = dict(fresh_checkpoint)
    partial.pop("checkpoint_sha256")
    partial["status"] = "running"
    partial["states"] = original_states.tolist()
    initial_amplitudes = runtime.jax.vmap(
        lambda state: arm_runtime.projected_amplitude(arm_runtime.params, state)
    )(runtime.jnp.asarray(original_states))
    partial["amplitudes"] = runtime._complex_pairs(
        np.asarray(initial_amplitudes)
    )
    _, sampling_key = runtime.jax.random.split(initial_key)
    partial["key"] = np.asarray(sampling_key, dtype=np.uint32).tolist()
    partial["burn_completed"] = 0
    partial["burn_diagnostics"] = []
    partial["measurement_completed"] = 0
    for name in (
        "measurement_diagnostics",
        "energy_series",
        "ordering_first_series",
        "ordering_second_series",
        "snapshot_changed",
    ):
        partial[name] = []
    partial["resume_count"] = 0
    partial["checkpoint_sha256"] = canonical_sha256(partial)
    checkpoint_path.write_text(json.dumps(partial))
    pre_resume_states = np.asarray(partial["states"])

    resumed_trial = runtime._run_arm(
        arm_runtime,
        spec=spec,
        case=case,
        bonds=bonds,
        checkpoint_path=checkpoint_path,
        identity=identity,
        overwrite=False,
    )
    resumed_checkpoint = json.loads(checkpoint_path.read_text())
    assert resumed_trial["sampler"]["resume_count"] == 1
    assert resumed_checkpoint["resume_count"] == 1
    assert np.any(np.asarray(resumed_checkpoint["states"]) != pre_resume_states)
    assert resumed_trial["initial_chain_states_sha256"] == initial_hash
    assert resumed_trial["common_profile_states_sha256"] == profile_hash
    assert resumed_checkpoint["initial_chain_states_sha256"] == initial_hash
    assert resumed_checkpoint["common_profile_states_sha256"] == profile_hash

    # Runtime timing fields are scoped to the final process attempt, not the
    # entire resumed history.  A resumed trial is therefore ineligible for a
    # timing-advantage claim even though its physics/sampler state is valid.
    timing_claim_eligible = resumed_trial["sampler"]["resume_count"] == 0
    assert timing_claim_eligible is False


def test_profile_finite_gate_allows_zero_single_sample_gradient() -> None:
    row = {
        "hpsi_cold_compile_and_call_seconds": 1.0,
        "hpsi_warm_seconds": [0.1, 0.09],
        "hpsi_warm_median_seconds": 0.095,
        "hpsi_warm_throughput_states_per_second": 10.0,
        "gradient_cold_compile_and_call_seconds": 2.0,
        "gradient_warm_seconds": [0.2, 0.18],
        "gradient_warm_median_seconds": 0.19,
        "gradient_warm_throughput_states_per_second": 5.0,
        "gradient_norm": 0.0,
        "profile_energy": -1.0,
        "profile_energy_imaginary_abs": 0.0,
        "profile_minimum_center_magnitude": 1.0e-9,
        "profile_maximum_local_energy_magnitude": 3.0,
        "profile_nonfinite_local_count": 0,
    }
    assert runtime._profile_rows_are_finite([row])
    broken = dict(row)
    broken["gradient_norm"] = np.inf
    assert not runtime._profile_rows_are_finite([broken])
    broken = dict(row)
    broken["profile_nonfinite_local_count"] = 1
    assert not runtime._profile_rows_are_finite([broken])

    source = Path(runtime.__file__).read_text()
    assert "local_energy_imaginary_abs_below_1e_minus_10" in source
    assert "local_energy_imaginary_abs_below_1e10" not in source


def test_official_6x6_runtime_results_are_current_sealed_and_readiness_only() -> None:
    current_source = runtime._source_manifest()
    current_source_hash = canonical_sha256(current_source)
    for directory, geometry in (
        ("spin_square_6x6_runtime_smoke_v2", "square"),
        ("spin_triangular_6x6_runtime_smoke_v2", "triangular"),
    ):
        result_directory = RESULT_ROOT / directory
        path = result_directory / "results.json"
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == runtime.RUNTIME_RESULT_SCHEMA_VERSION
        assert payload["status"] == "complete"
        assert payload["config_sha256"] == canonical_sha256(payload["config"])
        assert payload["source_sha256"] == current_source
        assert payload["source_manifest_sha256"] == current_source_hash
        assert payload["runtime_identity_sha256"] == canonical_sha256(
            payload["runtime_identity"]
        )
        assert payload["route_boundary"]["hilbert_space_enumerated"] is False
        assert payload["route_boundary"]["hamiltonian_diagonalization_performed"] is False
        assert payload["route_boundary"]["energy_or_physics_win_claimed"] is False
        assert payload["cases"][0]["geometry"] == geometry
        assert payload["cases"][0]["n_sites"] == 36
        trials = payload["cases"][0]["trials"]
        assert {trial["backbone"] for trial in trials} == {"u1", "su2"}
        sampler_seed = payload["config"]["sampler"]["seed"]
        initial_key = runtime.jax.random.PRNGKey(sampler_seed)
        chain_key, _ = runtime.jax.random.split(initial_key)
        expected_initial_states = runtime.initialize_spin_exchange_chains(
            chain_key,
            n_chains=payload["config"]["sampler"]["chains"],
            n_sites=36,
            n_up=18,
        )
        expected_profile_states = runtime.initialize_spin_exchange_chains(
            initial_key,
            n_chains=payload["config"]["sampler"]["chains"],
            n_sites=36,
            n_up=18,
        )
        expected_initial_hash = array_sha256(np.asarray(expected_initial_states))
        expected_profile_hash = array_sha256(np.asarray(expected_profile_states))
        assert expected_initial_hash != expected_profile_hash
        for trial in trials:
            assert trial["trial_payload_sha256"] == trial_payload_sha256(trial)
            assert trial["initial_chain_states_sha256"] == expected_initial_hash
            assert trial["common_profile_states_sha256"] == expected_profile_hash
            assert trial["all_readiness_gates_passed"] is True
            assert all(trial["readiness_gates"].values())
            assert trial["observables"]["estimator_contract"][
                "su2_invariant_structure_factor"
            ] is False
            checkpoint_path = result_directory / trial["checkpoint"]["path"]
            checkpoint = json.loads(checkpoint_path.read_text())
            assert (
                checkpoint["schema_version"]
                == runtime.RUNTIME_CHECKPOINT_SCHEMA_VERSION
            )
            expected_identity = {
                "config_sha256": payload["config_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "runtime_identity_sha256": payload["runtime_identity_sha256"],
                "case_id": payload["cases"][0]["case_id"],
                "arm_id": trial["arm_id"],
                "backbone": trial["backbone"],
                "initial_params_sha256": trial["initial_params_sha256"],
            }
            assert checkpoint["identity"] == expected_identity
            assert checkpoint["initial_chain_states_sha256"] == expected_initial_hash
            assert (
                checkpoint["common_profile_states_sha256"]
                == expected_profile_hash
            )
            assert checkpoint["burn_completed"] == 16
            assert checkpoint["measurement_completed"] == 64
            assert trial["sampler"]["resume_count"] == 0
            timing_claim_eligible = trial["sampler"]["resume_count"] == 0
            assert timing_claim_eligible is True
            loaded = runtime._load_checkpoint(
                checkpoint_path,
                expected_identity,
                initial_chain_states_sha256=expected_initial_hash,
                common_profile_states_sha256=expected_profile_hash,
            )
            assert loaded["status"] == "complete"
            assert loaded["checkpoint_sha256"] == trial["checkpoint"][
                "checkpoint_sha256"
            ]
        manifest = embedded_trial_manifest(payload)
        assert payload["trial_manifest"] == manifest
        assert payload["trial_manifest_sha256"] == canonical_sha256(manifest)
