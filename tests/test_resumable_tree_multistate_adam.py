from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint
from su2_nqs.vmc.resumable_tree_adam import pytree_sha256


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests/_resumable_three_root_worker.py"


def _worker(
    checkpoint: Path,
    *,
    resume: bool = False,
    maximum: int | None = None,
    samples_per_root: int = 4,
    overlap_diagonal_policy: str = "legacy",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        sys.executable,
        str(WORKER),
        "--checkpoint",
        str(checkpoint),
        "--samples-per-root",
        str(samples_per_root),
        "--overlap-diagonal-policy",
        overlap_diagonal_policy,
    ]
    if resume:
        arguments.append("--resume")
    if maximum is not None:
        arguments.extend(("--maximum", str(maximum)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["JAX_PLATFORM_NAME"] = "cpu"
    environment["JAX_ENABLE_X64"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment.pop("XLA_CLIENT_MEM_FRACTION", None)
    environment.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if check:
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _payload(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _trajectory_signature(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "params_sha256",
        "root_params_sha256",
        "optimizer_states_sha256",
        "root_optimizer_state_sha256",
        "completed_steps",
        "next_root_sample_indices",
        "next_sample_index",
        "history",
    )
    return {key: payload[key] for key in keys}


def test_fresh_process_split_is_exact_and_checkpoint_owns_all_roots(tmp_path: Path):
    full = _payload(_worker(tmp_path / "full"))
    partial = _payload(_worker(tmp_path / "split", maximum=1))
    assert partial["completed_steps"] == 1
    assert partial["next_root_sample_indices"] == [4, 4, 4]
    assert partial["next_sample_index"] == 12
    assert partial["stopped_early"]

    resumed = _payload(_worker(tmp_path / "split", resume=True))
    assert resumed["resumed"]
    assert not resumed["stopped_early"]
    assert resumed["next_root_sample_indices"] == [8, 8, 8]
    assert resumed["next_sample_index"] == 24
    assert _trajectory_signature(resumed) == _trajectory_signature(full)
    history = resumed["history"]
    assert history[0]["root_sample_index_start"] == [0, 0, 0]
    assert history[0]["root_sample_index_stop"] == [4, 4, 4]
    assert history[1]["root_sample_index_start"] == [4, 4, 4]
    assert history[1]["root_sample_index_stop"] == [8, 8, 8]
    assert history[0]["params_after_sha256"] == history[1]["params_before_sha256"]

    latest = load_latest_resumable_checkpoint(tmp_path / "split")
    assert isinstance(latest["params"], tuple) and len(latest["params"]) == 3
    assert (
        isinstance(latest["optimizer_state"], tuple)
        and len(latest["optimizer_state"]) == 3
    )
    assert latest["completed_steps"] == 2
    assert latest["next_sample_index"] == 24
    assert latest["metadata"]["next_root_sample_indices"] == [8, 8, 8]
    assert latest["metadata"]["root_params_sha256"] == [
        pytree_sha256(root) for root in latest["params"]
    ]
    assert latest["metadata"]["root_optimizer_state_sha256"] == [
        pytree_sha256(state) for state in latest["optimizer_state"]
    ]
    assert (
        latest["run_identity"]["sampled_ritz"]["overlap_diagonal_policy"]
        == "sampled_legacy"
    )


def test_identity_mismatch_fails_before_resumed_update(tmp_path: Path):
    _worker(tmp_path / "split", maximum=1)
    rejected = _worker(
        tmp_path / "split",
        resume=True,
        samples_per_root=5,
        check=False,
    )
    assert rejected.returncode != 0
    assert "identity does not match" in rejected.stderr
    latest = load_latest_resumable_checkpoint(tmp_path / "split")
    assert latest["completed_steps"] == 1
    assert latest["metadata"]["next_root_sample_indices"] == [4, 4, 4]

    corrected_rejected = _worker(
        tmp_path / "split",
        resume=True,
        overlap_diagonal_policy="exact_normalized",
        check=False,
    )
    assert corrected_rejected.returncode != 0
    assert "Ritz-only" in corrected_rejected.stderr


def test_corrupt_newest_falls_back_and_recomputes_unique_generation_exactly(
    tmp_path: Path,
):
    full = _payload(_worker(tmp_path / "full"))
    _worker(tmp_path / "recover")
    newest = tuple((tmp_path / "recover").glob("step_00000002_*.pkl"))
    assert len(newest) == 1
    original = newest[0].read_bytes()
    newest[0].write_bytes(original[: max(1, len(original) // 7)])

    recovered = _payload(_worker(tmp_path / "recover", resume=True))
    assert _trajectory_signature(recovered) == _trajectory_signature(full)
    failures = recovered["recovered_checkpoint_failures"]
    assert any("file checksum mismatch" in failure for failure in failures)
    step_two_generations = tuple(
        (tmp_path / "recover").glob("step_00000002_*.pkl")
    )
    assert len(step_two_generations) == 2
    assert newest[0] in step_two_generations
    manifest = json.loads((tmp_path / "recover/LATEST.json").read_text())
    assert [
        row["completed_steps"] for row in manifest["generations"]
    ].count(2) == 2
    latest = load_latest_resumable_checkpoint(tmp_path / "recover")
    assert latest["completed_steps"] == 2
    assert latest["newer_generation_failures"] == []
