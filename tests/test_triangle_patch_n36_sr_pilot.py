from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest
from su2_nqs.frontier.triangle_patch_physical_init import (
    sample_triangle_patch_cluster_product_chains,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin_triangle_patch_n36_sr_pilot as pilot
from projects.clebsch_tree_nqs import run as project_cli


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/spin_triangle_patch_n36_sr_pilot.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results"
    / "spin_triangle_patch_n36_sr_pilot_v1/results.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_sr_pilot_config_freezes_physical_initialization_and_work_budget() -> None:
    spec = _spec()
    pilot._validate(spec)
    assert spec["initialization"] == {
        "kind": "cluster_eigenvector",
        "pre_projection_background_coefficient": 0.05,
    }
    assert spec["projection"]["applied_from_update"] == 0
    assert spec["sampler"]["chains"] == 16
    assert (
        spec["sampler"]["initial_chain_kind"]
        == "cluster_product_born_random_translation"
    )
    assert spec["sampler"]["random_control_kind"] == "uniform_fixed_sz0"
    assert spec["sampler"]["initial_chain_probe_proposals_per_chain"] == 36
    assert spec["sampler"]["snapshots_per_update"] == 4
    assert spec["sampler"]["proposals_between_snapshots"] == 36
    assert spec["optimizer"]["updates"] == 2
    assert spec["optimizer"]["damping"] == 1.0e-2
    assert spec["optimizer"]["cg_tolerance"] == 1.0e-3
    assert spec["optimizer"]["cg_max_iterations"] == 20
    assert spec["optimizer"]["qgt_center_microbatch_size"] == 2
    assert spec["optimizer"]["qgt_rematerialize_center"] is True
    assert spec["audit"]["updates"] == [0, 2]
    assert all(case["ed"] is False for case in spec["cases"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda x: x["initialization"].__setitem__("kind", "random"),
        lambda x: x["initialization"].__setitem__(
            "pre_projection_background_coefficient", 0.02
        ),
        lambda x: x["projection"].__setitem__("applied_from_update", 1),
        lambda x: x["sampler"].__setitem__("initial_chain_kind", "uniform_fixed_sz0"),
        lambda x: x["sampler"].__setitem__(
            "initial_chain_probe_proposals_per_chain", 0
        ),
        lambda x: x["sampler"].__setitem__("chains", 8),
        lambda x: x["sampler"].__setitem__("snapshots_per_update", 3),
        lambda x: x["optimizer"].__setitem__("updates", 3),
        lambda x: x["optimizer"].__setitem__("qgt_center_microbatch_size", 4),
        lambda x: x["audit"].__setitem__("updates", [0]),
        lambda x: x["cases"][0].__setitem__("ed", True),
    ),
)
def test_sr_pilot_config_mutations_fail_closed(mutation) -> None:
    spec = copy.deepcopy(_spec())
    mutation(spec)
    with pytest.raises((TypeError, ValueError)):
        pilot._validate(spec)


def test_rematerialized_center_microbatches_preserve_values_and_jvp() -> None:
    params = jnp.asarray([0.3, -0.2, 0.5], dtype=jnp.float64)
    direction = jnp.asarray([-0.4, 0.1, 0.2], dtype=jnp.float64)
    states = jnp.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    def centers(candidate, batch):
        return jnp.exp(batch @ candidate) * jnp.asarray([1.0, -1.0])[: len(batch)]

    wrapped = pilot._rematerialized_center_microbatch_fn(
        centers, microbatch_size=2, n_sites=3
    )
    expected_value, expected_tangent = jax.jvp(
        lambda candidate: jnp.concatenate(
            (centers(candidate, states[:2]), centers(candidate, states[2:]))
        ),
        (params,),
        (direction,),
    )
    actual_value, actual_tangent = jax.jvp(
        lambda candidate: wrapped(candidate, states),
        (params,),
        (direction,),
    )
    np.testing.assert_allclose(actual_value, expected_value, atol=1.0e-13)
    np.testing.assert_allclose(actual_tangent, expected_tangent, atol=1.0e-13)


def test_initial_chain_ensembles_use_physical_starts_and_separate_random_control() -> None:
    spec = _spec()
    case = spec["cases"][0]
    plan = triangle_patch_6x6_canary_plan("square")
    runtime = type("Runtime", (), {"plan": plan})()
    key = jax.random.PRNGKey(int(case["sampler_seed"]))
    physical, random_control, physics_probe, random_probe, sampling = (
        pilot._initial_chain_ensembles(key, runtime, case, spec)
    )

    split = jax.random.split(key, 5)
    expected_physical = sample_triangle_patch_cluster_product_chains(
        split[0],
        plan,
        j2_over_j1=float(case["j2_over_j1"]),
        n_chains=16,
        translate=True,
    )
    expected_random = initialize_spin_exchange_chains(
        split[1], n_chains=16, n_sites=36, n_up=18
    )
    np.testing.assert_array_equal(physical, expected_physical)
    np.testing.assert_array_equal(random_control, expected_random)
    np.testing.assert_array_equal(physics_probe, split[2])
    np.testing.assert_array_equal(random_probe, split[3])
    np.testing.assert_array_equal(sampling, split[4])
    np.testing.assert_array_equal(np.asarray(physical).sum(axis=1), np.full(16, 18))
    np.testing.assert_array_equal(
        np.asarray(random_control).sum(axis=1), np.full(16, 18)
    )
    assert not np.array_equal(np.asarray(physical), np.asarray(random_control))


def test_projected_center_support_summary_preserves_exact_zero_diagnostic() -> None:
    summary = pilot._center_support_summary(
        jnp.asarray([0.0, -2.0e-16, 3.0e-12, -4.0])
    )
    assert summary["exact_zero_count"] == 1
    assert summary["exact_nonzero_fraction"] == pytest.approx(0.75)
    assert summary["above_absolute_1e-14_fraction"] == pytest.approx(0.5)
    assert summary["magnitude_minimum"] == 0.0
    assert summary["magnitude_maximum"] == 4.0
    assert np.isfinite(summary["log_magnitude_mean_over_nonzero"])


def test_sr_checkpoint_metadata_records_optimizer_step_and_chain_integrity() -> None:
    states = jnp.tile(
        jnp.concatenate((jnp.ones(18, dtype=jnp.int32), jnp.zeros(18, dtype=jnp.int32))),
        (16, 1),
    )
    key = jax.random.PRNGKey(44)
    metadata = pilot._checkpoint_metadata(
        states,
        key,
        audits=[{"update": 0}],
        initial={"kind": "test"},
        work={"sampler_proposals": 0},
        resume_count=1,
        completed_steps=2,
    )
    assert metadata["optimizer"] == "matrix_free_sampled_physical_sr"
    assert metadata["optimizer_completed_steps"] == 2
    assert metadata["samples_per_update"] == 64
    assert metadata["amplitude_cache_present"] is False
    assert metadata["chain_states_sha256"] == pilot.array_sha256(np.asarray(states))
    assert metadata["sampling_key_sha256"] == pilot.array_sha256(np.asarray(key))


def test_short_chain_ess_distinguishes_persistent_chains_from_four_samples() -> None:
    persistent = np.tile(np.arange(16, dtype=np.float64), (4, 1))
    result = pilot._chain_block_ess(persistent)
    assert result["sample_count"] == 64
    assert result["effective_sample_size"] == pytest.approx(16.0)
    assert result["short_trajectory_not_publication_ess"] is True
    assert 0.0 < result["effective_sample_size_fraction"] <= 1.0


def test_sr_pilot_writes_a_sealed_honest_artifact_without_running_kernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_case(
        case,
        spec,
        output,
        *,
        overwrite,
        source_manifest_sha256,
        runtime_identity_sha256,
    ):
        del (
            spec,
            output,
            overwrite,
            source_manifest_sha256,
            runtime_identity_sha256,
        )
        return {
            "case_id": case["case_id"],
            "geometry": case["geometry"],
            "status": "complete",
            "completed_updates": 2,
            "requested_updates": 2,
            "samples_per_update": 64,
            "history": [
                {
                    "update": 1,
                    "sr": {"linear_solve_converged": True},
                },
                {
                    "update": 2,
                    "sr": {"linear_solve_converged": True},
                },
            ],
        }

    monkeypatch.setattr(pilot, "_run_case", fake_case)
    monkeypatch.chdir(tmp_path)
    output = Path("artifact")
    result = pilot.run_triangle_patch_n36_sr_pilot(CONFIG, output)
    stored = json.loads((tmp_path / output / "results.json").read_text())
    unsealed = copy.deepcopy(stored)
    seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    assert result == stored
    assert stored["status"] == "complete"
    assert stored["hamiltonian_diagonalization_performed"] is False
    assert stored["translation_projection_applied_from_update_zero"] is True
    assert stored["training_force"] == "exact_singlet_3hz_center_only"
    assert stored["optimizer"] == "matrix_free_sampled_physical_sr"
    assert stored["uniform_fixed_sz0_initial_control_is_diagnostic_only"] is True
    assert stored["either_initial_ensemble_claimed_equilibrated"] is False
    assert "combined Monte Carlo uncertainty" in stored[
        "equilibration_promotion_requirement"
    ]
    assert stored["source_manifest_sha256"] == canonical_sha256(
        stored["source_manifest"]
    )
    assert (
        "su2_nqs/frontier/triangle_patch_physical_init.py"
        in stored["source_manifest"]
    )
    assert "su2_nqs/vmc/physical_sampled_sr.py" in stored["source_manifest"]
    assert stored["runtime_identity_sha256"] == canonical_sha256(
        stored["runtime_identity"]
    )
    assert "no converged energy" in stored["claim_boundary"]
    assert {case["geometry"] for case in stored["cases"]} == {
        "square",
        "triangular",
    }


def test_project_cli_exposes_the_sr_pilot() -> None:
    args = project_cli._build_parser().parse_args(
        ["triangle-patch-n36-sr-pilot", "--config", str(CONFIG)]
    )
    assert args.command == "triangle-patch-n36-sr-pilot"
    assert args.config == CONFIG


def test_completed_sr_pilot_artifact_records_the_negative_energy_gate() -> None:
    result = json.loads(RESULT.read_text())
    unsealed = copy.deepcopy(result)
    seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    assert result["status"] == "complete"
    assert result["config"] == _spec()
    manifest = transitive_local_source_manifest(
        (Path(pilot.__file__).resolve(),),
        root=pilot.ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    assert result["source_manifest"] == manifest
    assert result["source_manifest_sha256"] == canonical_sha256(manifest)
    assert result["hamiltonian_diagonalization_performed"] is False
    assert result["uniform_fixed_sz0_initial_control_is_diagnostic_only"] is True
    assert result["either_initial_ensemble_claimed_equilibrated"] is False

    cases = {case["geometry"]: case for case in result["cases"]}
    assert set(cases) == {"square", "triangular"}
    for case in cases.values():
        assert case["status"] == "complete"
        assert case["completed_updates"] == 2
        assert len(case["history"]) == 2
        assert len(case["full_h_paired_audits"]) == 2
        assert all(row["sr"]["linear_solve_converged"] for row in case["history"])
        assert all(row["sr"]["fisher_norm"] <= 0.05 + 1.0e-13 for row in case["history"])

    square_initial, square_final = cases["square"]["full_h_paired_audits"]
    assert square_final["energy_per_site"] > square_initial["energy_per_site"]
    assert square_final["variance_h_per_site"] > 10.0 * square_initial["variance_h_per_site"]

    triangular_initial, triangular_final = cases["triangular"]["full_h_paired_audits"]
    change = triangular_final["energy_per_site"] - triangular_initial["energy_per_site"]
    combined_mcse = np.hypot(
        triangular_initial["energy_per_site_chain_mcse"],
        triangular_final["energy_per_site_chain_mcse"],
    )
    assert change < 0.0
    assert abs(change) < combined_mcse
    assert triangular_final["variance_h_per_site"] > triangular_initial["variance_h_per_site"]
