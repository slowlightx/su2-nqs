from pathlib import Path

import jax
import numpy as np
import pytest

from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig, init_compact_tree_gru
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint
from su2_nqs.vmc.resumable_tree_adam import run_resumable_compact_tree_adam


def _problem():
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, embedding_dim=2, hidden_dim=4)
    params = init_compact_tree_gru(jax.random.PRNGKey(9201), config)
    bonds = (
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 0, 1.0),
        (0, 3, 0.125),
    )
    return config, params, bonds


def _run(path: Path, *, resume: bool, maximum: int | None):
    config, params, bonds = _problem()
    return run_resumable_compact_tree_adam(
        params,
        config=config,
        bonds=bonds,
        total_steps=3,
        batch_size=5,
        learning_rate=2.0e-3,
        sample_key=jax.random.PRNGKey(9202),
        run_identity={"test": "split_resume"},
        checkpoint_path=path,
        resume=resume,
        checkpoint_every=1,
        max_steps_this_invocation=maximum,
        execution_mode="adaptive",
        registered_widths=(4, 8, 16, 32, 64),
        microbatch_size=2,
        devices=(jax.local_devices()[0],),
    )


def _assert_pytrees_equal(first, second):
    first_leaves = jax.tree_util.tree_leaves(first)
    second_leaves = jax.tree_util.tree_leaves(second)
    assert len(first_leaves) == len(second_leaves)
    for left, right in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(left, right)


def _deterministic_history(history):
    timing = {
        "sampling_seconds",
        "connectivity_seconds",
        "first_pass_seconds",
        "second_pass_seconds",
    }
    return [{key: value for key, value in row.items() if key not in timing} for row in history]


def test_uninterrupted_and_split_resume_are_trajectory_identical(tmp_path: Path):
    uninterrupted = _run(tmp_path / "full", resume=False, maximum=None)
    partial = _run(tmp_path / "split", resume=False, maximum=1)
    assert partial.completed_steps == 1
    assert partial.stopped_early
    resumed = _run(tmp_path / "split", resume=True, maximum=None)
    assert resumed.completed_steps == 3
    assert resumed.next_sample_index == 15
    assert resumed.resumed
    assert not resumed.stopped_early
    _assert_pytrees_equal(uninterrupted.params, resumed.params)
    _assert_pytrees_equal(uninterrupted.optimizer_state, resumed.optimizer_state)
    assert _deterministic_history(uninterrupted.history) == _deterministic_history(
        resumed.history
    )
    assert [row["sample_index_start"] for row in resumed.history] == [0, 5, 10]
    latest = load_latest_resumable_checkpoint(
        tmp_path / "split", expected_run_identity=resumed.run_identity
    )
    assert latest["completed_steps"] == 3


def test_resume_rejects_trajectory_change(tmp_path: Path):
    partial = _run(tmp_path / "split", resume=False, maximum=1)
    assert partial.completed_steps == 1
    config, params, bonds = _problem()
    with pytest.raises(ValueError, match="identity does not match"):
        run_resumable_compact_tree_adam(
            params,
            config=config,
            bonds=bonds,
            total_steps=3,
            batch_size=6,
            learning_rate=2.0e-3,
            sample_key=jax.random.PRNGKey(9202),
            run_identity={"test": "split_resume"},
            checkpoint_path=tmp_path / "split",
            resume=True,
            execution_mode="adaptive",
            registered_widths=(4, 8, 16, 32, 64),
            microbatch_size=2,
            devices=(jax.local_devices()[0],),
        )


def test_resume_rejects_allocator_environment_change(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    _run(tmp_path / "split", resume=False, maximum=1)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with pytest.raises(ValueError, match="identity does not match"):
        _run(tmp_path / "split", resume=True, maximum=None)
