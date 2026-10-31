"""Fail-closed, content-addressed storage for exact reference arrays.

The cache deliberately separates *loading* from *generation*.  A missing or
invalid entry never invokes an expensive exact solver implicitly; callers must
set ``recompute=True`` explicitly.  Each entry consists of a JSON manifest and
an NPZ payload named by a SHA-256 digest of the complete problem identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Literal
import uuid

import numpy as np


SCHEMA_VERSION = 1
_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ExactReferenceCacheError(RuntimeError):
    """Base class for missing or invalid exact-reference cache entries."""


class ExactReferenceCacheMiss(ExactReferenceCacheError):
    """Raised when no complete entry exists and recomputation was not allowed."""


@dataclass(frozen=True)
class ExactReferenceRecord:
    """One verified exact-reference cache entry."""

    cache_key: str
    status: Literal["hit", "recomputed"]
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, object]
    manifest_path: Path
    payload_path: Path

    @property
    def cache_hit(self) -> bool:
        """Whether this record was loaded without invoking the solver."""

        return self.status == "hit"


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("cache identity must be finite and JSON serializable") from error
    return encoded.encode("utf-8")


def _normalized_identity(identity: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("cache identity must be a nonempty mapping")
    return json.loads(_canonical_json(dict(identity)))


def reference_cache_key(identity: Mapping[str, object]) -> str:
    """Return the schema-qualified content key for ``identity``."""

    normalized = _normalized_identity(identity)
    digest = hashlib.sha256()
    digest.update(f"su2-nqs-exact-reference-v{SCHEMA_VERSION}\0".encode("ascii"))
    digest.update(_canonical_json(normalized))
    return digest.hexdigest()


def ndarray_sha256(value: np.ndarray | object) -> str:
    """Hash an array including its dtype and shape, in canonical C order."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are forbidden in exact-reference caches")
    contiguous = array.copy(order="C") if array.ndim == 0 else np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"su2-nqs-ndarray-v1\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(contiguous.shape)))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _entry_lock(path: Path):
    """Serialize readers and explicit writers on Unix research hosts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - supported targets are Unix
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _checked_arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("exact solver must return a nonempty array mapping")
    arrays: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid exact-reference array name: {name!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("object arrays are forbidden in exact-reference caches")
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise ValueError("exact-reference arrays must be numeric or boolean")
        arrays[name] = (
            array.copy(order="C") if array.ndim == 0 else np.ascontiguousarray(array)
        )
    return arrays


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": ndarray_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }


def _payload_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _load_verified(
    *,
    manifest_path: Path,
    payload_path: Path,
    cache_key: str,
    identity: Mapping[str, object],
) -> ExactReferenceRecord:
    if not manifest_path.is_file() and not payload_path.is_file():
        raise ExactReferenceCacheMiss(
            f"exact reference {cache_key} is absent; pass --recompute to generate it"
        )
    if not manifest_path.is_file() or not payload_path.is_file():
        raise ExactReferenceCacheError(
            f"exact reference {cache_key} is incomplete; explicit recomputation is required"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExactReferenceCacheError("exact-reference manifest is unreadable") from error
    if not isinstance(manifest, dict):
        raise ExactReferenceCacheError("exact-reference manifest must be a JSON object")
    normalized = _normalized_identity(identity)
    identity_sha256 = _sha256_bytes(_canonical_json(normalized))
    required = {
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "identity_sha256": identity_sha256,
        "identity": normalized,
    }
    for name, expected in required.items():
        if manifest.get(name) != expected:
            raise ExactReferenceCacheError(
                f"exact-reference manifest field {name!r} does not match"
            )
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ExactReferenceCacheError("exact-reference payload manifest is missing")
    if payload.get("filename") != payload_path.name:
        raise ExactReferenceCacheError("exact-reference payload filename does not match")
    try:
        payload_bytes = payload_path.read_bytes()
    except OSError as error:
        raise ExactReferenceCacheError("exact-reference payload is unreadable") from error
    if payload.get("sha256") != _sha256_bytes(payload_bytes):
        raise ExactReferenceCacheError("exact-reference payload SHA-256 mismatch")
    descriptors = payload.get("arrays")
    if not isinstance(descriptors, dict) or not descriptors:
        raise ExactReferenceCacheError("exact-reference array manifest is missing")
    try:
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
            if set(archive.files) != set(descriptors):
                raise ExactReferenceCacheError(
                    "exact-reference payload arrays disagree with the manifest"
                )
            arrays = {}
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = (
                    array.copy(order="C")
                    if array.ndim == 0
                    else np.ascontiguousarray(array)
                )
    except ExactReferenceCacheError:
        raise
    except Exception as error:
        raise ExactReferenceCacheError("exact-reference NPZ payload is invalid") from error
    for name, array in arrays.items():
        descriptor = descriptors.get(name)
        expected = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": ndarray_sha256(array),
        }
        if descriptor != expected:
            raise ExactReferenceCacheError(
                f"exact-reference array {name!r} failed manifest validation"
            )
        array.setflags(write=False)
    return ExactReferenceRecord(
        cache_key=cache_key,
        status="hit",
        arrays=arrays,
        manifest=manifest,
        manifest_path=manifest_path,
        payload_path=payload_path,
    )


def load_or_recompute_exact_reference(
    cache_root: str | Path,
    identity: Mapping[str, object],
    *,
    recompute: bool = False,
    compute: Callable[[], Mapping[str, object]] | None = None,
) -> ExactReferenceRecord:
    """Load a verified entry, or explicitly recompute and atomically replace it.

    A cache miss raises :class:`ExactReferenceCacheMiss`.  Supplying ``compute``
    alone does not authorize its execution; ``recompute=True`` is mandatory.
    """

    root = Path(cache_root)
    normalized = _normalized_identity(identity)
    cache_key = reference_cache_key(normalized)
    manifest_path = root / f"{cache_key}.manifest.json"
    payload_path = root / f"{cache_key}.npz"
    lock_path = root / f"{cache_key}.lock"
    if (
        not recompute
        and not manifest_path.exists()
        and not payload_path.exists()
    ):
        raise ExactReferenceCacheMiss(
            f"exact reference {cache_key} is absent; pass --recompute to generate it"
        )
    if recompute and compute is None:
        raise ValueError("compute callback is required when recompute=True")
    with _entry_lock(lock_path):
        if not recompute:
            return _load_verified(
                manifest_path=manifest_path,
                payload_path=payload_path,
                cache_key=cache_key,
                identity=normalized,
            )
        assert compute is not None
        arrays = _checked_arrays(compute())
        payload_bytes = _payload_bytes(arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cache_key": cache_key,
            "identity_sha256": _sha256_bytes(_canonical_json(normalized)),
            "identity": normalized,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "filename": payload_path.name,
                "sha256": _sha256_bytes(payload_bytes),
                "arrays": _array_manifest(arrays),
            },
        }
        _atomic_write(payload_path, payload_bytes)
        _atomic_write(
            manifest_path,
            json.dumps(
                manifest, indent=2, ensure_ascii=True, sort_keys=True
            ).encode("utf-8")
            + b"\n",
        )
        verified = _load_verified(
            manifest_path=manifest_path,
            payload_path=payload_path,
            cache_key=cache_key,
            identity=normalized,
        )
        return ExactReferenceRecord(
            cache_key=verified.cache_key,
            status="recomputed",
            arrays=verified.arrays,
            manifest=verified.manifest,
            manifest_path=verified.manifest_path,
            payload_path=verified.payload_path,
        )


__all__ = [
    "ExactReferenceCacheError",
    "ExactReferenceCacheMiss",
    "ExactReferenceRecord",
    "load_or_recompute_exact_reference",
    "ndarray_sha256",
    "reference_cache_key",
]
