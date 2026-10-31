"""Bounded saved-probe protocol and scheduling checks; no N36 amplitudes.

Artifact checks skip in a clean checkout without research outputs. The live
tests use small synthetic arrays or the 36-context coefficient table only;
they do not launch the N36 workflow, sample states, or diagonalize H.
"""

import hashlib
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_resonance_reuse import _stream_kernel
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.multicontext_resonance_attention import (
    MulticontextResonanceAttentionConfig, complete_environment_labels,
    free_multicontext_generators, init_multicontext_resonance_attention,
    make_multicontext_reference, multicontext_resonance_generators,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_resonance_reuse.json"
RESULTS = ROOT / "projects/clebsch_tree_nqs/results/spin_resonance_reuse_v1"


def _config():
    return json.loads(CONFIG.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_protocol_is_same_probe_cost_only_and_bounded_minor_batches():
    config = _config()
    assert config["task"] == "spin_resonance_reuse"
    assert config["geometry"] == "triangular_6x6_j2_0125"
    assert config["training_steps"] == 0 and config["remote_allowed"] is False
    assert config["H_centers"] == 2 and config["full_H_outputs"] == 232
    assert config["minor_max_rank"] == 7
    assert config["minor_batch_size"] == 4096
    assert config["H_output_chunk_size"] == 24
    assert config["supports"] == ["physical", "scrambled"]
    assert config["multicontext_controls"] == [
        "full", "one_environment", "additive", "free_table", "position_scrambled"]
    for key in ("max_unique_queries", "max_terminal_requests", "max_plan_bytes",
                "wall_seconds_cap", "carrier_batch_size"):
        assert isinstance(config[key], int) and config[key] > 0
    assert 0 < config["peak_rss_gib_cap"] <= 4.
    for key in ("amplitude_relative_tolerance", "H_relative_tolerance", "gradient_relative_tolerance"):
        assert 0 < config[key] <= 1e-8


def test_registered_old_source_seals_and_centers_are_unchanged():
    config = _config()
    source = ROOT / config["source_run"]
    required = [source / name for name in ("results.json", "checkpoint.npz", "full_h_replay.npz")]
    if not all(path.exists() for path in required):
        pytest.skip("sealed research outputs are not included in this checkout")
    assert _sha(source / "results.json") == config["source_results_sha256"]
    old = json.loads((source / "results.json").read_text())
    assert old["status"] == "completed"
    assert _sha(source / "checkpoint.npz") == old["checkpoint_sha256"]
    assert _sha(source / "full_h_replay.npz") == old["replay_sha256"]
    with np.load(source / "checkpoint.npz", allow_pickle=False) as checkpoint:
        centers = checkpoint["centers"].copy()
        assert sum(checkpoint[k].size for k in checkpoint.files if k.startswith("parameter_")) == 730
        assert checkpoint["reference"].shape == (2, 6, 10)
    with np.load(source / "full_h_replay.npz", allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay["centers"], centers)
        assert replay["h_configurations"].shape == (config["full_H_outputs"], 36)
        np.testing.assert_array_equal(replay["h_configurations"][replay["h_centers"]], centers)
        assert len(np.unique(replay["h_configurations"], axis=0)) == config["full_H_outputs"]
    assert centers.shape == (config["H_centers"], 36)
    np.testing.assert_array_equal(centers.sum(axis=1), 18)
    for geometry in config["supports"]:
        chunks = old["geometries"][geometry]["chunks"]
        assert sum(chunk["plan"]["output_count"] for chunk in chunks) == config["full_H_outputs"]
        assert all(chunk["plan"]["group_order"] == 432 for chunk in chunks)


def test_target_spatial_group_is_complete_not_a_selected_image_subset():
    # Group enumeration only: no N36 wavefunction, H, or query compilation.
    group = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), dtype=int)
    assert group.shape == (432, 36)
    assert len(np.unique(group, axis=0)) == 432
    np.testing.assert_array_equal(np.sort(group, axis=1), np.tile(np.arange(36), (432, 1)))
    keys = {tuple(row) for row in group}
    assert tuple(range(36)) in keys
    assert all(tuple(np.argsort(row)) in keys for row in group)


def test_static_stream_retains_every_output_and_gradient_with_partial_last_chunk():
    rng = np.random.default_rng(6130101)
    rows = rng.integers(0, 9, size=(7, 5), dtype=np.int32)
    values = rng.normal(size=(7, 5, 3)) + 1j * rng.normal(size=(7, 5, 3))
    weights = rng.normal(size=7)
    params = {"scale": jnp.array([.2, -.3])}
    traces = []

    def forward(p, r, v):
        traces.append((r.shape, v.shape))
        return jnp.sum((p["scale"][0] + p["scale"][1] * r)[..., None] * v, axis=(1, 2))

    kernel = jax.jit(forward)
    gradient = jax.jit(jax.grad(lambda p, r, v, w: jnp.real(jnp.sum(w * forward(p, r, v)))))
    amplitude, observed_grad, records = _stream_kernel(
        kernel, gradient, params, rows, values, weights, chunk_size=3, repeats=1)
    expected = np.sum((.2 - .3 * rows)[..., None] * values, axis=(1, 2))
    expected_grad = np.array([
        np.sum(weights[:, None, None] * values).real,
        np.sum(weights[:, None, None] * rows[..., None] * values).real,
    ])
    np.testing.assert_allclose(amplitude, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(observed_grad["scale"], expected_grad, atol=2e-14, rtol=2e-14)
    assert len(amplitude) == 7 and len(records) == 3
    # Forward and reverse each trace once, even though the last block is partial.
    assert traces == [((3, 5), (3, 5, 3)), ((3, 5), (3, 5, 3))]


def test_two_environment_neural_table_is_exactly_represented_by_free360():
    config = _config()
    model = MulticontextResonanceAttentionConfig()
    positions = np.asarray(config["multicontext_positions"], dtype=float)
    assert positions.shape == (2, 2)
    initial = init_multicontext_resonance_attention(
        jax.random.PRNGKey(config["multicontext_initialization_seed"]), model)
    reference = make_multicontext_reference(initial, positions, model)
    rng = np.random.default_rng(config["multicontext_perturbation_seed"])
    direction = {name: rng.normal(size=value.shape) for name, value in initial.items()}
    norm = np.sqrt(sum(np.sum(value**2) for value in direction.values()))
    params = {name: value + config["multicontext_parameter_distance"] * direction[name] / norm
              for name, value in initial.items()}
    table = multicontext_resonance_generators(params, reference, positions, model)
    assert table.shape == (6, 6, 10) and table.size == 360
    np.testing.assert_array_equal(free_multicontext_generators(table.reshape(-1), model), table)
    contexts = complete_environment_labels(model)
    assert contexts.shape == (36, 2, 2)
    assert len(np.unique(contexts.reshape(36, 4), axis=0)) == 36
    assert np.isfinite(np.asarray(table)).all()


def test_position_only_scramble_is_context_reassignment_not_new_free_table_capacity():
    config = _config()
    model = MulticontextResonanceAttentionConfig()
    positions = np.asarray(config["multicontext_positions"], dtype=float)
    initial = init_multicontext_resonance_attention(jax.random.PRNGKey(6130102), model)
    params = dict(initial)
    params["generator_kernel"] = initial["generator_kernel"] + .01
    reference = make_multicontext_reference(initial, positions, model)
    scrambled_reference = make_multicontext_reference(initial, positions[::-1], model)
    table = multicontext_resonance_generators(params, reference, positions, model)
    reassigned = multicontext_resonance_generators(params, scrambled_reference, positions[::-1], model)
    # Token-order covariance: changing only positions with fixed physical
    # labels swaps the table axes. The free360 control contains both exactly.
    np.testing.assert_allclose(reassigned, np.swapaxes(table, 0, 1), atol=2e-15, rtol=2e-13)
    np.testing.assert_array_equal(free_multicontext_generators(reassigned.reshape(-1), model), reassigned)


def _completed_artifact():
    """Read the sealed fixed-probe experiment, never evaluate a model."""
    required = ("results.json", "registration.json", "replay.npz", "multicontext_checkpoint.npz")
    if not all((RESULTS / name).exists() for name in required):
        pytest.skip("completed reuse artifacts are not included in this checkout")
    result = json.loads((RESULTS / "results.json").read_text())
    registration = json.loads((RESULTS / "registration.json").read_text())
    assert result["status"] == "completed"
    return result, registration


def _relative(actual, expected):
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / max(
        np.linalg.norm(expected), 1e-300)


def test_completed_artifact_seals_include_exact_cost_sources_and_no_training_claim():
    result, registration = _completed_artifact()
    assert _sha(RESULTS / "registration.json") == result["registration_sha256"]
    assert _sha(RESULTS / "replay.npz") == result["replay_sha256"]
    assert _sha(RESULTS / "multicontext_checkpoint.npz") == result["checkpoint_sha256"]
    assert _sha(CONFIG) == registration["config_sha256"]
    assert registration["config"] == _config()
    manifest = registration["source_manifest"]
    for required in (
        "projects/clebsch_tree_nqs/workflows/spin_resonance_reuse.py",
        "projects/clebsch_tree_nqs/workflows/spin_overlap_attention_cost.py",
        "su2_nqs/benchmarks/a1_carrier_queries.py",
        "su2_nqs/models/slater_minor_updates.py",
        "su2_nqs/models/multicontext_resonance_attention.py",
        "su2_nqs/vmc/physical_h_queries.py",
    ):
        assert required in manifest
    # A later intentional numerical revision must preserve an appropriate
    # registered snapshot/update this provenance test, not silently attribute
    # this fixed result to different cost or amplitude code.
    assert all(_sha(ROOT / path) == digest for path, digest in manifest.items())
    assert result["training_steps"] == 0 and result["remote_released"] is False
    assert result["energy_estimate"] is None and result["variance_estimate"] is None
    assert result["total_seconds"] <= registration["config"]["wall_seconds_cap"]
    assert result["peak_rss_gib"] <= registration["config"]["peak_rss_gib_cap"]
    assert result["h_stencil"]["bond_count"] == 216
    assert result["h_stencil"]["full_H_centers"] == 2
    assert result["carrier_all_images_relative_error"] < 1e-12


def test_completed_checkpoint_and_saved_support_match_registered_parent_without_model_calls():
    _, registration = _completed_artifact()
    source = ROOT / registration["config"]["source_run"]
    with np.load(RESULTS / "replay.npz", allow_pickle=False) as replay, np.load(
        source / "full_h_replay.npz", allow_pickle=False
    ) as old:
        for key in ("centers", "h_configurations", "h_indices", "h_elements", "h_valid"):
            np.testing.assert_array_equal(replay[key], old[key])
        assert replay["h_configurations"].shape == (232, 36)
        assert replay["centers"].shape == (2, 36)
        for support in registration["config"]["supports"]:
            for arm in ("cascade", "no_cascade", "diagonal_conditioned", "factorized"):
                assert _relative(replay[f"{support}_baseline_{arm}"],
                                 old[f"{support}_{arm}_all_H_outputs"]) < 1e-12
    with np.load(RESULTS / "multicontext_checkpoint.npz", allow_pickle=False) as checkpoint:
        names = [key for key in checkpoint.files if key.startswith("parameter_")]
        assert len(names) == 18 and sum(checkpoint[key].size for key in names) == 730
        assert checkpoint["reference"].shape == (6, 6, 10)
        np.testing.assert_array_equal(checkpoint["positions"], registration["config"]["multicontext_positions"])
        distance = np.sqrt(sum(np.sum((checkpoint[key] - checkpoint["initial_" + key[10:]])**2)
                               for key in names))
        assert distance == pytest.approx(registration["config"]["multicontext_parameter_distance"], abs=2e-15)


def test_saved_complete_h_stencil_matches_independent_pauli_coefficients():
    """Only construct two saved rows of H; no new psi, sampling, or ED."""
    _completed_artifact()
    with np.load(RESULTS / "replay.npz", allow_pickle=False) as replay:
        centers, states = replay["centers"], replay["h_configurations"]
        lookup = {tuple(bits): index for index, bits in enumerate(states)}
        expected = np.zeros((2, len(states)))
        bonds = []
        for coupling, displacements in (
            (1., ((1, 0), (0, 1), (1, 1))),
            (.125, ((1, -1), (1, 2), (2, 1))),
        ):
            for y in range(6):
                for x in range(6):
                    for dx, dy in displacements:
                        bonds.append((6*y+x, 6*((y+dy) % 6)+(x+dx) % 6, coupling))
        assert len(bonds) == len({tuple(sorted((i, j))) for i, j, _ in bonds}) == 216
        for row, bits in enumerate(centers):
            for i, j, coupling in bonds:
                expected[row, lookup[tuple(bits)]] += coupling * (.25 if bits[i] == bits[j] else -.25)
                if bits[i] != bits[j]:
                    target = bits.copy()
                    target[i], target[j] = bits[j], bits[i]
                    expected[row, lookup[tuple(target)]] += .5 * coupling
        actual = np.zeros_like(expected)
        for row in range(2):
            valid = replay["h_valid"][row]
            np.add.at(actual[row], replay["h_indices"][row, valid], replay["h_elements"][row, valid])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("support", ("physical", "scrambled"))
def test_saved_reused_amplitudes_full_h_and_free_table_replay(support):
    result, _ = _completed_artifact()
    profile = result["geometries"][support]
    with np.load(RESULTS / "replay.npz", allow_pickle=False) as replay:
        indices, elements, valid = replay["h_indices"], replay["h_elements"], replay["h_valid"]
        def action(values):
            return np.sum(np.where(valid, elements * values[indices], 0.), axis=1)
        base = replay[support + "_baseline_cascade"]
        fast = replay[support + "_fast_all_H_outputs"]
        assert base.shape == fast.shape == (232,)
        observed = _relative(fast, base)
        h_observed = _relative(action(fast), action(base))
        assert observed < 1e-12 and h_observed < 1e-12
        assert observed == pytest.approx(profile["amplitude_relative_error"], abs=2e-16)
        assert h_observed == pytest.approx(profile["full_H_relative_error"], abs=5e-16)
        full = replay[support + "_multicontext_full"]
        free = replay[support + "_multicontext_free_table"]
        assert _relative(full, free) < 1e-12
        assert _relative(action(full), action(free)) < 1e-12
        assert _relative(full, free) == pytest.approx(profile["multicontext_free_table_replay_error"], abs=2e-16)
        for control in ("full", "one_environment", "additive", "free_table", "position_scrambled"):
            values = replay[f"{support}_multicontext_{control}"]
            assert values.shape == (232,) and np.isfinite(values).all()
        assert _relative(full, replay[support + "_multicontext_additive"]) == pytest.approx(
            profile["multicontext_vs_additive_point_difference"], abs=2e-16)
        gradient_keys = [key for key in replay.files if key.startswith(support + "_gradient_")]
        assert len(gradient_keys) == 18 and sum(replay[key].size for key in gradient_keys) == 730
        assert all(np.isfinite(replay[key]).all() for key in gradient_keys)
        # Only the fast reverse vector was saved. Its reported baseline error
        # is not an independently replayed second reverse-AD computation.
        assert profile["reverse_gradient_relative_error"] < 1e-12


@pytest.mark.parametrize("support", ("physical", "scrambled"))
def test_cost_replay_includes_planning_transfers_first_compile_and_all_queries(support):
    result, registration = _completed_artifact()
    config = registration["config"]
    profile = result["geometries"][support]
    chunks, reused, batches = (profile[key] for key in ("baseline_chunks", "reused_kernel_chunks", "minor_batches"))
    assert sum(chunk["plan"]["output_count"] for chunk in chunks) == 232
    assert len(chunks) == len(reused) == 10
    assert all(chunk["plan"]["group_order"] == 432 for chunk in chunks)
    plan = profile["pullback_plan"]
    assert plan["output_count"] == 232 and plan["group_order"] == 432
    assert plan["maximum_local_connections"] == 216
    assert plan["complete_group"] and plan["projector_only_after_both_gates"]
    assert plan["intermediate_shared_cell_summed_coherently"] and not plan["magnetic_channels_truncated"]
    assert plan["carrier_sector_precondition"] == "Gamma-A1"
    assert plan["carrier_symmetry_used_for_deduplication"]
    assert plan["raw_total_requests"] - plan["unique_carrier_queries"] == plan["duplicate_requests_removed"]
    assert plan["padded_terminal_requests"] == 232 * 432 * 216
    assert profile["minor_fixed_batch_size"] == config["minor_batch_size"] == 4096
    assert sum(batch["valid_queries"] for batch in batches) == plan["unique_carrier_queries"] == profile["minor_valid_queries"]
    assert sum(batch["padded_queries"] for batch in batches) == len(batches) * 4096 == profile["minor_padded_queries"]
    assert sum(profile["minor_rank_counts"].values()) == profile["minor_valid_queries"]
    for index, batch in enumerate(batches):
        assert batch["query_start"] == 4096 * index
        assert 0 < batch["valid_queries"] <= batch["padded_queries"] == 4096
    for field, recorded in (
        ("planning_seconds", "minor_planning_seconds"),
        ("first_execution_seconds", "minor_first_seconds"),
        ("output_transfer_seconds", "minor_output_transfer_seconds"),
    ):
        assert sum(batch[field] for batch in batches) == pytest.approx(profile[recorded], abs=1e-12)
    # The occupied-orbital tensor is transferred once before the per-query
    # batches. Its time is correctly in the aggregate, not in a batch record.
    assert profile["minor_input_transfer_seconds"] >= sum(batch["input_transfer_seconds"] for batch in batches)
    minor = sum(profile[key] for key in ("minor_planning_seconds", "minor_input_transfer_seconds",
                                        "minor_first_seconds", "minor_output_transfer_seconds"))
    assert minor == pytest.approx(profile["minor_fresh_carrier_seconds"], abs=1e-12)
    selected = "row_minor" if minor < profile["pullback_dense_carrier_seconds"] else "dense"
    assert selected == profile["selected_carrier_evaluator"]
    carrier = min(minor, profile["pullback_dense_carrier_seconds"])
    base_h = sum(chunk["planning_seconds"] + chunk["carrier_query_inclusive_seconds"]
                 + chunk["forward_compile_inclusive_seconds"] for chunk in chunks)
    base_ad = base_h + sum(chunk["gradient_compile_inclusive_seconds"] for chunk in chunks)
    fast_h = (profile["pullback_planning_seconds"] + carrier + profile["terminal_gather_seconds"]
              + sum(chunk["host_staging_seconds"] + chunk["forward_first_seconds"] for chunk in reused))
    fast_ad = fast_h + sum(chunk["reverse_first_seconds"] for chunk in reused)
    for key, expected in (("baseline_fresh_H_seconds", base_h), ("baseline_fresh_H_AD_seconds", base_ad),
                          ("fast_fresh_H_seconds", fast_h), ("fast_fresh_H_AD_seconds", fast_ad),
                          ("H_speedup", base_h / fast_h), ("H_AD_speedup", base_ad / fast_ad)):
        assert profile[key] == pytest.approx(expected, abs=1e-12)
    # These selected-path costs omit comparison/control/warm-repeat overhead;
    # report the separately measured total experiment wall time as well.
    assert result["total_seconds"] > base_ad + fast_ad
