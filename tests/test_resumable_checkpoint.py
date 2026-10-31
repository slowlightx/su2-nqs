import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    load_latest_resumable_checkpoint,
    load_resumable_checkpoint,
    save_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


def _state(step: int):
    return {
        "params": {"x": jnp.asarray([1.0 + step, -2.0])},
        "optimizer_state": (step, jnp.asarray([0.1 * step])),
        "completed_steps": step,
        "next_sample_index": 7 * step,
        "run_identity": {"system": "test", "batch_size": 7},
        "history": [{"step": index + 1} for index in range(step)],
    }


def test_resumable_checkpoint_round_trip_and_identity_gate(tmp_path: Path):
    state = _state(2)
    destination = tmp_path / "state.pkl"
    save_resumable_checkpoint(destination, **state)
    restored = load_resumable_checkpoint(
        destination, expected_run_identity=state["run_identity"]
    )
    assert restored["completed_steps"] == 2
    assert restored["next_sample_index"] == 14
    np.testing.assert_array_equal(restored["params"]["x"], state["params"]["x"])
    with pytest.raises(ValueError, match="identity does not match"):
        load_resumable_checkpoint(
            destination,
            expected_run_identity={"system": "changed", "batch_size": 7},
        )


def test_truncated_resumable_checkpoint_is_rejected(tmp_path: Path):
    state = _state(1)
    destination = tmp_path / "state.pkl"
    save_resumable_checkpoint(destination, **state)
    destination.write_bytes(destination.read_bytes()[:31])
    with pytest.raises(ValueError, match="cannot decode"):
        load_resumable_checkpoint(destination)


def test_generation_series_falls_back_and_recovers_manifest_orphan(tmp_path: Path):
    identity = _state(0)["run_identity"]
    first = save_resumable_checkpoint_generation(tmp_path, **_state(0))
    latest = save_resumable_checkpoint_generation(tmp_path, **_state(1))
    latest.write_bytes(latest.read_bytes()[:47])
    restored = load_latest_resumable_checkpoint(
        tmp_path, expected_run_identity=identity
    )
    assert restored["completed_steps"] == 0
    assert Path(restored["checkpoint_generation_path"]) == first
    assert restored["newer_generation_failures"]

    # A valid generation committed after a stale/corrupt manifest is still
    # discovered by scanning immutable generation names.
    latest.unlink()
    manifest = tmp_path / "LATEST.json"
    manifest.write_text("not-json")
    orphan = tmp_path / f"step_00000001_{canonical_json_sha256(identity)[:12]}.pkl"
    save_resumable_checkpoint(orphan, **_state(1))
    restored_orphan = load_latest_resumable_checkpoint(
        tmp_path, expected_run_identity=identity
    )
    assert restored_orphan["completed_steps"] == 1


def test_series_manifest_records_immutable_generations(tmp_path: Path):
    save_resumable_checkpoint_generation(tmp_path, **_state(0))
    save_resumable_checkpoint_generation(tmp_path, **_state(1))
    manifest = json.loads((tmp_path / "LATEST.json").read_text())
    assert manifest["latest_completed_steps"] == 1
    assert [row["completed_steps"] for row in manifest["generations"]] == [0, 1]
    assert all("commit_sequence" in row for row in manifest["generations"])
    assert len(list(tmp_path.glob("step_*.pkl"))) == 2


def test_corrupt_latest_can_fall_back_then_recompute_same_step(tmp_path: Path):
    identity = _state(0)["run_identity"]
    first = save_resumable_checkpoint_generation(tmp_path, **_state(0))
    corrupt = save_resumable_checkpoint_generation(tmp_path, **_state(1))
    damaged_bytes = corrupt.read_bytes()[:53]
    corrupt.write_bytes(damaged_bytes)

    fallback = load_latest_resumable_checkpoint(
        tmp_path, expected_run_identity=identity
    )
    assert Path(fallback["checkpoint_generation_path"]) == first
    assert fallback["newer_generation_failures"]

    replacement = save_resumable_checkpoint_generation(tmp_path, **_state(1))
    assert replacement != corrupt
    assert corrupt.read_bytes() == damaged_bytes
    restored = load_latest_resumable_checkpoint(
        tmp_path, expected_run_identity=identity
    )
    assert restored["completed_steps"] == 1
    assert Path(restored["checkpoint_generation_path"]) == replacement
    manifest = json.loads((tmp_path / "LATEST.json").read_text())
    assert [row["completed_steps"] for row in manifest["generations"]] == [0, 1, 1]
    step_one = [
        row for row in manifest["generations"] if row["completed_steps"] == 1
    ]
    assert step_one[0]["commit_sequence"] < step_one[1]["commit_sequence"]
