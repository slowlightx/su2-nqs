"""Checksummed, generational checkpoints for preemptible training jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import secrets
import tempfile
import time


RESUMABLE_CHECKPOINT_SCHEMA_VERSION = 1


def canonical_json_sha256(value: object) -> str:
    """Return a stable SHA256 digest for a JSON-compatible object."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("run identity must be finite and JSON serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_bytes(destination: Path, data: bytes) -> None:
    """Write bytes through a same-directory temporary file and fsync it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def save_resumable_checkpoint(
    path: str | Path,
    params: object,
    *,
    optimizer_state: object,
    completed_steps: int,
    next_sample_index: int,
    run_identity: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Atomically save all state needed for deterministic indexed resumption.

    ``run_identity`` should contain every choice that can change the trajectory
    (Hamiltonian, model, optimizer, random keys, batch/microbatch schedule, and
    relevant source hashes).  Loading with an expected identity fails closed
    on any mismatch.  Checkpoints are pickle files and therefore must only be
    loaded from trusted sources.
    """

    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise ValueError("completed_steps must be a non-negative integer")
    if (
        isinstance(next_sample_index, bool)
        or not isinstance(next_sample_index, int)
        or next_sample_index < 0
    ):
        raise ValueError("next_sample_index must be a non-negative integer")
    identity = dict(run_identity)
    identity_sha256 = canonical_json_sha256(identity)
    payload = {
        "schema_version": RESUMABLE_CHECKPOINT_SCHEMA_VERSION,
        "kind": "su2_nqs_resumable_training",
        "params": params,
        "optimizer_state": optimizer_state,
        "completed_steps": int(completed_steps),
        "next_sample_index": int(next_sample_index),
        "run_identity": identity,
        "run_identity_sha256": identity_sha256,
        "history": [dict(row) for row in history],
        "metadata": {} if metadata is None else dict(metadata),
    }
    serialized_payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema_version": RESUMABLE_CHECKPOINT_SCHEMA_VERSION,
        "kind": "su2_nqs_resumable_checkpoint_envelope",
        "payload_sha256": hashlib.sha256(serialized_payload).hexdigest(),
        "payload": serialized_payload,
    }
    _atomic_write_bytes(
        Path(path), pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )


def load_resumable_checkpoint(
    path: str | Path,
    *,
    expected_run_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load and verify a trusted resumable checkpoint.

    Truncation, checksum failure, schema mismatch, inconsistent counters, and a
    changed run identity are all rejected before parameters are returned.
    """

    try:
        with Path(path).open("rb") as handle:
            envelope = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("cannot decode resumable su2-nqs checkpoint") from error
    if not isinstance(envelope, dict):
        raise ValueError("invalid resumable checkpoint envelope")
    if (
        envelope.get("schema_version") != RESUMABLE_CHECKPOINT_SCHEMA_VERSION
        or envelope.get("kind") != "su2_nqs_resumable_checkpoint_envelope"
        or not isinstance(envelope.get("payload"), bytes)
        or not isinstance(envelope.get("payload_sha256"), str)
    ):
        raise ValueError("invalid resumable checkpoint envelope")
    serialized_payload = envelope["payload"]
    observed_digest = hashlib.sha256(serialized_payload).hexdigest()
    if observed_digest != envelope["payload_sha256"]:
        raise ValueError("resumable checkpoint checksum mismatch")
    try:
        payload = pickle.loads(serialized_payload)
    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError("cannot decode resumable checkpoint payload") from error
    required = {
        "schema_version",
        "kind",
        "params",
        "optimizer_state",
        "completed_steps",
        "next_sample_index",
        "run_identity",
        "run_identity_sha256",
        "history",
        "metadata",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("invalid resumable checkpoint payload")
    if (
        payload["schema_version"] != RESUMABLE_CHECKPOINT_SCHEMA_VERSION
        or payload["kind"] != "su2_nqs_resumable_training"
    ):
        raise ValueError("unsupported resumable checkpoint schema")
    if not isinstance(payload["run_identity"], dict):
        raise ValueError("invalid checkpoint run identity")
    identity_digest = canonical_json_sha256(payload["run_identity"])
    if identity_digest != payload["run_identity_sha256"]:
        raise ValueError("checkpoint run identity checksum mismatch")
    if expected_run_identity is not None:
        expected_digest = canonical_json_sha256(dict(expected_run_identity))
        if identity_digest != expected_digest:
            raise ValueError("checkpoint run identity does not match this run")
    completed_steps = payload["completed_steps"]
    next_sample_index = payload["next_sample_index"]
    history = payload["history"]
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or isinstance(next_sample_index, bool)
        or not isinstance(next_sample_index, int)
        or next_sample_index < 0
        or not isinstance(history, list)
        or len(history) != completed_steps
    ):
        raise ValueError("inconsistent resumable checkpoint counters")
    return payload


def save_resumable_checkpoint_generation(
    directory: str | Path,
    params: object,
    *,
    optimizer_state: object,
    completed_steps: int,
    next_sample_index: int,
    run_identity: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Commit an immutable checkpoint generation and then its latest manifest.

    Generation files are never overwritten or deleted.  This gives an NFS job
    a valid earlier generation if the newest file is corrupted, and also makes
    an interruption between generation and manifest commits recoverable by a
    directory scan.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identity_sha256 = canonical_json_sha256(dict(run_identity))
    manifest_path = root / "LATEST.json"
    entries: list[dict[str, object]] = []
    if manifest_path.is_file():
        try:
            current = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            current = None
        if (
            isinstance(current, dict)
            and current.get("kind") == "su2_nqs_checkpoint_series"
            and current.get("run_identity_sha256") == identity_sha256
            and isinstance(current.get("generations"), list)
        ):
            for position, value in enumerate(current["generations"]):
                if not isinstance(value, dict):
                    continue
                entry = dict(value)
                step = entry.get("completed_steps")
                filename = entry.get("file")
                digest = entry.get("sha256")
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or step < 0
                    or not isinstance(filename, str)
                    or not isinstance(digest, str)
                ):
                    continue
                sequence = entry.get("commit_sequence")
                if (
                    isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence < 0
                ):
                    # Old v1 manifests had no explicit commit order.  Their
                    # list order was the commit order, so retain it exactly.
                    entry["commit_sequence"] = position
                entries.append(entry)
    largest_sequence = max(
        (int(entry["commit_sequence"]) for entry in entries), default=-1
    )
    commit_sequence = max(time.time_ns(), largest_sequence + 1)
    # The sequence preserves commit order; the nonce makes concurrent or
    # clock-collision writes fail by identity-safe uniqueness instead of
    # overwriting an immutable generation.
    generation = root / (
        f"step_{completed_steps:08d}_{identity_sha256[:12]}_"
        f"{commit_sequence:020d}_{secrets.token_hex(6)}.pkl"
    )
    while generation.exists():
        commit_sequence += 1
        generation = root / (
            f"step_{completed_steps:08d}_{identity_sha256[:12]}_"
            f"{commit_sequence:020d}_{secrets.token_hex(6)}.pkl"
        )
    save_resumable_checkpoint(
        generation,
        params,
        optimizer_state=optimizer_state,
        completed_steps=completed_steps,
        next_sample_index=next_sample_index,
        run_identity=run_identity,
        history=history,
        metadata=metadata,
    )
    generation_sha256 = hashlib.sha256(generation.read_bytes()).hexdigest()
    entries.append(
        {
            "completed_steps": completed_steps,
            "file": generation.name,
            "sha256": generation_sha256,
            "commit_sequence": commit_sequence,
        }
    )
    entries.sort(
        key=lambda entry: (
            int(entry["completed_steps"]),
            int(entry["commit_sequence"]),
            str(entry["file"]),
        )
    )
    manifest = {
        "schema_version": RESUMABLE_CHECKPOINT_SCHEMA_VERSION,
        "kind": "su2_nqs_checkpoint_series",
        "run_identity_sha256": identity_sha256,
        "latest_completed_steps": max(
            int(entry["completed_steps"]) for entry in entries
        ),
        "generations": entries,
    }
    _atomic_write_bytes(
        manifest_path,
        (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return generation


def load_latest_resumable_checkpoint(
    directory: str | Path,
    *,
    expected_run_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load the newest valid generation, falling back after corruption.

    The manifest is advisory: valid orphan generations left by a preemption
    between the data and manifest commits are discovered as well.  An expected
    run identity is strongly recommended and is mandatory for campaign code.
    """

    root = Path(directory)
    if not root.is_dir():
        raise ValueError("resumable checkpoint series directory is missing")
    expected_digest = (
        None
        if expected_run_identity is None
        else canonical_json_sha256(dict(expected_run_identity))
    )
    manifest_hashes: dict[str, str] = {}
    manifest_sequences: dict[str, int] = {}
    manifest_path = root / "LATEST.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            manifest_identity = manifest.get("run_identity_sha256")
            if expected_digest is not None and manifest_identity != expected_digest:
                raise ValueError("checkpoint series run identity does not match this run")
            generations = manifest.get("generations")
            if isinstance(generations, list):
                for position, entry in enumerate(generations):
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("file"), str)
                        and isinstance(entry.get("sha256"), str)
                    ):
                        manifest_hashes[entry["file"]] = entry["sha256"]
                        sequence = entry.get("commit_sequence")
                        manifest_sequences[entry["file"]] = (
                            int(sequence)
                            if isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            and sequence >= 0
                            else position
                        )

    step_pattern = re.compile(r"^step_([0-9]{8})_.*\.pkl$")
    sequence_pattern = re.compile(r"_([0-9]{20})_[0-9a-f]+\.pkl$")

    def candidate_order(candidate: Path) -> tuple[int, int, str]:
        step_match = step_pattern.match(candidate.name)
        if step_match is None:
            return (-1, -1, candidate.name)
        step = int(step_match.group(1))
        if candidate.name in manifest_sequences:
            sequence = manifest_sequences[candidate.name]
        else:
            sequence_match = sequence_pattern.search(candidate.name)
            if sequence_match is not None:
                sequence = int(sequence_match.group(1))
            else:
                # Backward-compatible priority for an old-style orphan.
                try:
                    sequence = candidate.stat().st_mtime_ns
                except OSError:
                    sequence = -1
        return (step, sequence, candidate.name)

    candidates = sorted(
        root.glob(
            "step_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*.pkl"
        ),
        key=candidate_order,
        reverse=True,
    )
    failures: list[str] = []
    for candidate in candidates:
        registered_digest = manifest_hashes.get(candidate.name)
        if registered_digest is not None:
            observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if observed != registered_digest:
                failures.append(f"{candidate.name}: file checksum mismatch")
                continue
        try:
            payload = load_resumable_checkpoint(
                candidate, expected_run_identity=expected_run_identity
            )
        except ValueError as error:
            failures.append(f"{candidate.name}: {error}")
            continue
        payload = dict(payload)
        payload["checkpoint_generation_path"] = str(candidate)
        payload["newer_generation_failures"] = failures
        return payload
    detail = "; ".join(failures) if failures else "no generation files found"
    raise ValueError(f"no valid resumable checkpoint generation: {detail}")


__all__ = [
    "RESUMABLE_CHECKPOINT_SCHEMA_VERSION",
    "canonical_json_sha256",
    "load_latest_resumable_checkpoint",
    "load_resumable_checkpoint",
    "save_resumable_checkpoint_generation",
    "save_resumable_checkpoint",
]
