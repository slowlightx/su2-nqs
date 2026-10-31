"""Cheap provenance/configuration checks; no PEPS compilation or optimization."""

from copy import deepcopy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from peps_vmc.pilot import Experiment, digest, source_manifest, validate_config, write_json
from peps_vmc.proposals import CoveringBridgeProposal
from su2_nqs.vmc.checkpoint import load_checkpoint


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "triangular_6x6_j2_0125_local.json"


def test_registered_local_config_is_bounded_and_roles_have_distinct_seeds():
    config = json.loads(CONFIG_PATH.read_text())
    validate_config(config)
    assert (config["rows"], config["columns"], config["j2"], config["steps"]) == (6, 6, .125, 2)
    assert config["audit_records"] == 256
    assert config["wall_limit_seconds"] <= 1800
    assert config["maximum_RSS_GiB"] <= 6
    seeds = [config[key] for key in ("initial_seed", "covering_seed", "training_seed", "ward_seed", "audit_seed")]
    assert len(set(seeds)) == 5


@pytest.mark.parametrize("change", [
    {"steps": 5}, {"fit_records": 1}, {"geometry_records": 3},
    {"audit_groups": 3}, {"audit_records": 255}, {"damping": np.nan},
    {"maximum_finite_FS": 0}, {"candidate_fractions": [0.5, 1.0]},
    {"candidate_fractions": [1.0, 1.0]}, {"j2": -0.1},
    {"ward_seed": 29003}, {"training_seed": True},
])
def test_invalid_or_nonindependent_pilot_configs_are_rejected(change):
    config = json.loads(CONFIG_PATH.read_text())
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_source_seal_covers_isolated_core_and_direct_math_reuse_only():
    manifest = source_manifest()
    assert "peps-vmc/peps_vmc/pilot.py" in manifest
    assert "peps-vmc/peps_vmc/model.py" in manifest
    assert "peps-vmc/peps_vmc/proposals.py" in manifest
    assert "peps-vmc/peps_vmc/contraction.py" in manifest
    assert "su2_nqs/vmc/importance_amplitude_sr.py" in manifest
    assert "su2_nqs/su2/clebsch_gordan.py" in manifest
    assert all("clebsch_tree" not in path for path in manifest)
    assert all(len(sha) == 64 for sha in manifest.values())
    assert manifest == source_manifest()


def test_digest_is_stable_for_numeric_pytree_and_dictionary_order():
    first = {"w": np.array([[1.0, -2.0], [0.0, 3.0]]), "next": {"a": 1, "b": 2}}
    second = {"next": {"b": 2, "a": 1}, "w": [[1.0, -2.0], [0.0, 3.0]]}
    assert digest(first) == digest(second)
    with pytest.raises(ValueError):
        digest({"invalid": np.nan})


def test_atomic_json_refuses_invalid_numeric_results(tmp_path):
    target = tmp_path / "result.json"
    write_json(target, {"valid": np.array([1.0, 2.0])})
    assert json.loads(target.read_text()) == {"valid": [1.0, 2.0]}
    with pytest.raises(ValueError):
        write_json(target, {"invalid": np.nan})
    assert json.loads(target.read_text()) == {"valid": [1.0, 2.0]}


def test_checkpoint_preserves_parameters_optimizer_and_exact_next_rng(tmp_path):
    # Construct only the metadata owner, not a numerical PEPS runtime.
    experiment = Experiment.__new__(Experiment)
    experiment.c = json.loads(CONFIG_PATH.read_text())
    experiment.proposal = CoveringBridgeProposal(
        np.array([[[0, 1], [2, 3]]], dtype=np.int32),
        np.array([[0, 1], [1, 2]], dtype=np.int32), np.array([.3, .7]))
    params = {"w": np.array([[1.2, -.3], [.04, .7]])}
    moments = (np.array(7, dtype=np.int64), {"w": np.ones((2, 2))}, {"w": np.full((2, 2), .2)})
    rng = np.random.default_rng(2026103008)
    rng.random(13)
    expected_rng = deepcopy(rng.bit_generator.state)
    path = tmp_path / "checkpoint.pkl"
    experiment.checkpoint(path, params, moments, rng, 7)
    state = load_checkpoint(path)
    for actual, expected in zip(jax.tree.leaves((state["params"], state["optimizer_state"])),
                                jax.tree.leaves((params, moments))):
        np.testing.assert_array_equal(actual, expected)
    assert state["metadata"]["step"] == 7
    assert state["metadata"]["params_sha256"] == digest(params)
    assert state["metadata"]["config_sha256"] == digest(experiment.c)
    assert state["metadata"]["source_sha256"] == digest(source_manifest())
    assert state["metadata"]["next_rng"] == expected_rng
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = state["metadata"]["next_rng"]
    np.testing.assert_array_equal(restored_rng.integers(0, 2**32, 16, dtype=np.uint32),
                                  rng.integers(0, 2**32, 16, dtype=np.uint32))
