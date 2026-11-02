"""Portable, checksummed initial-parameter artifacts for accelerator runs.

The artifact deliberately stores only host NumPy leaves.  A caller supplies a
fresh model template when loading; its pytree structure, leaf shapes, and
dtypes must match the manifest before any values are moved to a JAX device.
This avoids backend-dependent random initialization while keeping model
construction and validation in the owning model module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import array_sha256, atomic_json, canonical_sha256, pytree_sha256


PARAMETER_ARTIFACT_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_parameter_artifact(
    manifest_path: str | Path,
    params: object,
    *,
    identity: dict[str, object],
    overwrite: bool = False,
) -> dict[str, object]:
    """Save one immutable host-array pytree next to a JSON manifest."""

    destination = Path(manifest_path)
    if destination.suffix != ".json":
        raise ValueError("parameter artifact manifest must end in .json")
    data_path = destination.with_suffix(".npz")
    if not overwrite and (destination.exists() or data_path.exists()):
        raise FileExistsError(f"parameter artifact already exists: {destination}")
    # Fail early if the identity is not canonical JSON.
    canonical_sha256(identity)
    leaves, structure = jax.tree_util.tree_flatten(params)
    arrays = [np.asarray(leaf) for leaf in leaves]
    if not arrays:
        raise ValueError("parameter artifact cannot contain an empty pytree")
    if any(array.dtype.kind not in "biufc" for array in arrays):
        raise TypeError("parameter leaves must be numeric arrays")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{data_path.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                **{f"leaf_{index:04d}": array for index, array in enumerate(arrays)},
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, data_path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise

    manifest: dict[str, object] = {
        "schema_version": PARAMETER_ARTIFACT_SCHEMA_VERSION,
        "kind": "su2_nqs_initial_parameter_artifact",
        "identity": dict(identity),
        "identity_sha256": canonical_sha256(identity),
        "pytree_structure": repr(structure),
        "pytree_sha256": pytree_sha256(params),
        "data_file": data_path.name,
        "data_sha256": _file_sha256(data_path),
        "leaves": [
            {
                "name": f"leaf_{index:04d}",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "array_sha256": array_sha256(array),
            }
            for index, array in enumerate(arrays)
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_json(destination, manifest)
    return manifest


def load_parameter_artifact(
    manifest_path: str | Path,
    template: object,
    *,
    expected_identity: dict[str, object] | None = None,
) -> object:
    """Load an artifact after validating it against a model template."""

    source = Path(manifest_path)
    manifest = json.loads(source.read_text())
    if (
        manifest.get("schema_version") != PARAMETER_ARTIFACT_SCHEMA_VERSION
        or manifest.get("kind") != "su2_nqs_initial_parameter_artifact"
    ):
        raise ValueError("unsupported initial-parameter artifact schema")
    stored_manifest_hash = manifest.pop("manifest_sha256", None)
    if stored_manifest_hash != canonical_sha256(manifest):
        raise ValueError("initial-parameter manifest checksum mismatch")
    manifest["manifest_sha256"] = stored_manifest_hash
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("identity_sha256") != canonical_sha256(identity):
        raise ValueError("initial-parameter identity checksum mismatch")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("initial-parameter identity mismatch")

    template_leaves, structure = jax.tree_util.tree_flatten(template)
    if manifest.get("pytree_structure") != repr(structure):
        raise ValueError("initial-parameter pytree structure mismatch")
    records = manifest.get("leaves")
    if not isinstance(records, list) or len(records) != len(template_leaves):
        raise ValueError("initial-parameter leaf count mismatch")
    data_path = source.parent / str(manifest.get("data_file"))
    if not data_path.is_file() or manifest.get("data_sha256") != _file_sha256(data_path):
        raise ValueError("initial-parameter data checksum mismatch")

    restored = []
    with np.load(data_path, allow_pickle=False) as archive:
        expected_names = {str(record.get("name")) for record in records}
        if set(archive.files) != expected_names:
            raise ValueError("initial-parameter archive members mismatch")
        for record, template_leaf in zip(records, template_leaves, strict=True):
            name = str(record.get("name"))
            array = np.asarray(archive[name])
            expected_shape = tuple(int(value) for value in record.get("shape", ()))
            template_array = np.asarray(template_leaf)
            if (
                array.shape != expected_shape
                or array.shape != template_array.shape
                or str(array.dtype) != str(record.get("dtype"))
                or array.dtype != template_array.dtype
            ):
                raise ValueError(f"initial-parameter leaf metadata mismatch: {name}")
            if record.get("array_sha256") != array_sha256(array):
                raise ValueError(f"initial-parameter leaf checksum mismatch: {name}")
            restored.append(jnp.asarray(array))
    result = jax.tree_util.tree_unflatten(structure, restored)
    if manifest.get("pytree_sha256") != pytree_sha256(result):
        raise ValueError("loaded initial-parameter pytree checksum mismatch")
    return result


__all__ = [
    "PARAMETER_ARTIFACT_SCHEMA_VERSION",
    "load_parameter_artifact",
    "save_parameter_artifact",
]
