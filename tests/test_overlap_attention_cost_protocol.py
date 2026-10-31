"""Cost-only registration guardrails; these tests never evaluate N36 states."""
import json

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_overlap_attention_cost as workflow
from su2_nqs.models.shared_resonance_attention import (
    ResonanceAttentionConfig, shared_resonance_parameter_count,
)
from su2_nqs.frontier.provenance import file_sha256


RESULTS = workflow.ROOT / "projects/clebsch_tree_nqs/results"
WORKFLOW_PATH = "projects/clebsch_tree_nqs/workflows/spin_overlap_attention_cost.py"


def _config():
    return json.loads(workflow.CONFIG.read_text())


def _translate_cell(cell, dx, dy, length=6):
    return [((site // length + dy) % length) * length + (site % length + dx) % length
            for site in cell]


def test_registered_probe_has_no_training_or_physics_estimator_claim():
    config = _config()
    assert workflow.validate_cost_config(config) is None
    assert config["task"] == "spin_overlap_attention_cost"
    assert (config["geometry"], config["length"], config["j2_over_j1"]) == ("triangular", 6, .125)
    assert config["training_steps"] == 0
    assert config["remote_allowed"] is False
    assert len(config["center_seeds"]) == len(set(config["center_seeds"])) == 2
    assert config["arms"] == ["cascade", "no_cascade", "diagonal_conditioned", "factorized"]
    assert "not Monte Carlo or energy estimation" in config["scope"]
    assert "NOT optimized VMC/two-Lanczos" in config["carrier"]
    assert "free46" in config["strong_controls"]
    assert "not a proof" in config["strong_controls"]
    assert config["max_unique_queries"] <= 4_000_000
    assert config["max_terminal_requests"] <= 25_000_000
    assert config["max_plan_bytes"] <= 805306368
    assert config["peak_rss_gib_cap"] <= 4.
    assert config["wall_seconds_cap"] <= 1200
    assert shared_resonance_parameter_count(ResonanceAttentionConfig(
        config["width"], config["hidden_width"])) == 730


@pytest.mark.parametrize("field,value", [
    ("identity_relative_tolerance", float("nan")),
    ("jvp_relative_tolerance", 1.),
    ("checkpoint_relative_tolerance", 0.),
    ("H_output_chunk_size", 0),
    ("carrier_batch_size", False),
])
def test_validation_tolerances_and_chunk_budget_fail_closed(field, value):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        workflow.validate_cost_config(config)


def test_both_physical_supports_preserve_the_registered_three_cell_translation_chain():
    config = _config()
    assert config["physical_cells"] != config["scrambled_cells"]
    assert config["displacements"] == [[2., 0.], [2., 0.]]
    for name in ("physical_cells", "scrambled_cells"):
        cells = config[name]
        assert len(cells) == 3 and all(len(cell) == 4 for cell in cells)
        assert len(set(site for cell in cells for site in cell)) == 12
        assert all(0 <= site < 36 for cell in cells for site in cell)
        for old, new in zip(cells[:-1], cells[1:], strict=True):
            assert _translate_cell(old, 2, 0) == new


@pytest.mark.parametrize("field,value", [
    ("training_steps", 1),
    ("remote_allowed", True),
    ("task", "different_experiment"),
    ("geometry", "square"),
    ("length", 8),
    ("arms", ["cascade"]),
    ("center_seeds", [2026101201, 2026101202, 2026101205]),
])
def test_mutated_protocol_rejected_before_any_resource_or_wavefunction_work(tmp_path, monkeypatch, field, value):
    config = _config()
    config[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    def expensive_work_was_reached():
        raise AssertionError("invalid protocol reached numerical setup")
    monkeypatch.setattr(workflow, "numerical_runtime_identity", expensive_work_was_reached)
    with pytest.raises(ValueError):
        workflow.run(tmp_path / "must_not_run", path)


def test_registered_output_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "registration.json"
    marker.write_text('{"immutable": true}')
    def expensive_work_was_reached():
        raise AssertionError("overwrite reached numerical setup")
    monkeypatch.setattr(workflow, "numerical_runtime_identity", expensive_work_was_reached)
    with pytest.raises(FileExistsError):
        workflow.run(output)
    assert marker.read_text() == '{"immutable": true}'


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_archived_workflow_matches_original_registration_after_intentional_active_fix(version):
    """Only the active runner changed; its original registered source survives.

    v1 stopped at its unique-query cap. v2 ran once successfully. Correcting
    the active timing formula later neither changes those outcomes nor turns
    their archived sources into the updated runner.
    """
    folder = RESULTS / f"spin_overlap_attention_cost_{version}"
    registration = json.loads((folder / "registration.json").read_text())
    result = json.loads((folder / "results.json").read_text())
    manifest = registration["source_manifest"]
    assert file_sha256(folder / "workflow_snapshot.py") == manifest[WORKFLOW_PATH]
    changed = {path for path, digest in manifest.items()
               if file_sha256(workflow.ROOT / path) != digest}
    assert changed == {WORKFLOW_PATH}
    assert file_sha256(folder / "registration.json") == result["registration_sha256"]
    assert file_sha256(folder / "checkpoint.npz") == result["checkpoint_sha256"]
    if version == "v1":
        assert file_sha256(folder / "config_snapshot.json") == registration["config_sha256"]
        assert result["status"] == "resource_cap_reached"
        assert result["geometries"] == {}
        assert not (folder / "full_h_replay.npz").exists()
    else:
        assert file_sha256(workflow.CONFIG) == registration["config_sha256"]
        assert registration["config"] == _config() == result["config"]
        assert result["status"] == "completed"
        assert file_sha256(folder / "full_h_replay.npz") == result["replay_sha256"]
    assert result["training_steps"] == 0
    assert result["energy_estimate"] is None and result["variance_estimate"] is None
    assert result["remote_released"] is False


def _complex_log(values):
    return np.asarray(values["real"]) + 1j * np.asarray(values["imag"])


def test_saved_full_h_action_and_jvp_replay_without_new_amplitude_calls():
    """Linear NumPy reconstruction only: no carrier, model, sampler, or ED."""
    folder = RESULTS / "spin_overlap_attention_cost_v2"
    result = json.loads((folder / "results.json").read_text())
    with np.load(folder / "full_h_replay.npz", allow_pickle=False) as saved:
        assert saved["h_configurations"].shape == (232, 36)
        assert saved["centers"].shape == (2, 36)
        indices, weights, valid = saved["h_indices"], saved["h_elements"], saved["h_valid"]
        def action(values):
            return values[saved["h_centers"]], np.sum(
                np.where(valid, values[indices] * weights, 0.), axis=1)
        for geometry, profile in result["geometries"].items():
            for arm, record in profile["arms"].items():
                values = saved[f"{geometry}_{arm}_all_H_outputs"]
                psi, hpsi = action(values)
                np.testing.assert_allclose(psi, _complex_log(record["psi"]), rtol=2e-13, atol=1e-24)
                np.testing.assert_allclose(hpsi, _complex_log(record["hpsi"]), rtol=2e-13, atol=1e-24)
            tangent = action(saved[f"{geometry}_derivative_all_H_outputs"])
            finite_difference = action(saved[f"{geometry}_finite_difference_all_H_outputs"])
            relative_errors = [np.linalg.norm(a-b) / np.linalg.norm(b)
                               for a, b in zip(tangent, finite_difference, strict=True)]
            assert max(relative_errors) < result["config"]["jvp_relative_tolerance"]
            np.testing.assert_allclose(relative_errors, profile["jvp_finite_difference_relative_errors"],
                                       rtol=2e-5, atol=1e-15)
            assert max(c["checkpoint_relative_error"] for c in profile["chunks"]) < 1e-12
            assert all(value > 0 and np.isfinite(value) for value in profile["gradient_leaf_norms"].values())


def test_v2_cost_reconstruction_includes_compilation_and_keeps_old_logs_immutable():
    """The v2 fresh_* names exclude JIT; recompute their documented correction."""
    result = json.loads((RESULTS / "spin_overlap_attention_cost_v2/results.json").read_text())
    current_seconds = {}
    total_determinants = 0
    batch = result["config"]["carrier_batch_size"]
    for geometry, profile in result["geometries"].items():
        chunks = profile["chunks"]
        assert len(chunks) == 10
        assert sum(c["plan"]["output_count"] for c in chunks) == 232
        cold = sum(c["planning_seconds"] + c["carrier_query_inclusive_seconds"]
                   + c["forward_compile_inclusive_seconds"] for c in chunks)
        cold_ad = cold + sum(c["gradient_compile_inclusive_seconds"] for c in chunks)
        old_formula = sum(c["planning_seconds"] + c["carrier_query_inclusive_seconds"]
                          + float(np.median(c["arms"]["cascade"]["warm_seconds"])) for c in chunks)
        np.testing.assert_allclose(profile["fresh_full_H_inclusive_seconds"], old_formula, atol=1e-12)
        assert cold > old_formula
        assert cold_ad > profile["fresh_full_H_plus_reverse_AD_inclusive_seconds"]
        current_seconds[geometry] = cold, cold_ad
        queries = sum(c["plan"]["unique_carrier_queries"] for c in chunks)
        assert queries == profile["carrier_queries_summed_over_chunks"]  # Not globally unique.
        determinants = sum(3 * batch * ((c["plan"]["unique_carrier_queries"] + batch - 1) // batch)
                           for c in chunks)
        assert determinants == profile["determinants_including_padding"]
        total_determinants += determinants
    np.testing.assert_allclose(current_seconds["physical"], (41.36348499817541, 49.69928187620826), atol=1e-10)
    np.testing.assert_allclose(current_seconds["scrambled"], (39.550769626162946, 48.07625183527125), atol=1e-10)
    assert total_determinants == 49_775_616
    assert total_determinants + 2 * 3 * batch == 49_777_152  # Setup padded warmup/reference.
