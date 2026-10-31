"""Small, deterministic helpers for immutable experiment artifacts.

The functions in this module intentionally know nothing about a Hamiltonian,
model, or optimizer.  They provide the byte-level conventions shared by
production campaign runners so that checkpoints and final JSON artifacts can
be compared without duplicating subtly different hashing code.

These hashes are reproducibility seals, not a serialization format.  In
particular, :func:`array_sha256` includes NumPy's dtype string, shape
representation, and native contiguous bytes.  Artifacts should therefore also
record their numerical runtime identity when portability across platforms is
required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import cast

import jax
import numpy as np


def atomic_json(path: Path, payload: object) -> None:
    """Write an indented JSON payload via a sibling temporary file.

    The final ``Path.replace`` is atomic when the temporary and destination
    files reside on the same filesystem.  The emitted representation matches
    the existing symmetry-ladder artifact convention: two-space indentation,
    insertion-order keys, and one trailing newline.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-serializable value using the campaign canonical encoding."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-contiguous host bytes.

    This deliberately preserves the established artifact convention rather
    than normalizing byte order or dtype.  Callers comparing artifacts across
    runtimes should verify a separately stored runtime identity first.
    """

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(contiguous.dtype).encode())
    digest.update(repr(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def pytree_sha256(tree: object) -> str:
    """Hash a JAX pytree's structure and array leaves in flattening order."""

    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(repr(structure).encode())
    for leaf in leaves:
        digest.update(array_sha256(np.asarray(leaf)).encode())
    return digest.hexdigest()


def coo_sha256(
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
) -> str:
    """Hash an ordered sparse COO representation without creating JSON lists.

    COO entry order is part of the seal.  Canonicalize or sort a sparse matrix
    before calling this function if order-independent identity is desired.
    """

    digest = hashlib.sha256()
    for array in (rows, columns, values):
        digest.update(array_sha256(np.asarray(array)).encode())
    return digest.hexdigest()


def repo_relative_path(path: Path, *, root: Path) -> str:
    """Return ``path`` resolved relative to the resolved repository ``root``.

    A path outside ``root`` raises :class:`ValueError`; this prevents absolute
    machine-specific paths from silently entering portable artifact records.
    """

    return str(path.resolve().relative_to(root.resolve()))


def paired_initialization_audit(
    trials: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Verify projection-only arms share parameters and raw initial states.

    Trials with fewer than two distinct projection labels are omitted because
    they do not form a projection comparison.  A mismatched parameter or raw
    amplitude seal raises :class:`RuntimeError` immediately.
    """

    groups: dict[tuple[str], list[Mapping[str, object]]] = {}
    for trial in trials:
        key = (str(trial["initialization_pair_id"]),)
        groups.setdefault(key, []).append(trial)
    records: list[dict[str, object]] = []
    for (pair_id,), members in sorted(groups.items()):
        first = members[0]
        backbone = str(first["backbone"])
        feature_dim = int(first["feature_dim"])
        seed = int(first["seed"])
        projections = sorted({str(member["projection"]) for member in members})
        if len(projections) < 2:
            continue
        parameter_hashes = sorted(
            {str(member["initial_params_sha256"]) for member in members}
        )
        raw_hashes = sorted(
            {str(member["initial_raw_amplitudes_sha256"]) for member in members}
        )
        passed = len(parameter_hashes) == 1 and len(raw_hashes) == 1
        record: dict[str, object] = {
            "backbone": backbone,
            "feature_dim": feature_dim,
            "seed": seed,
            "initialization_pair_id": pair_id,
            "projections": projections,
            "initial_params_sha256": parameter_hashes,
            "initial_raw_amplitudes_sha256": raw_hashes,
            "passed": passed,
        }
        records.append(record)
        if not passed:
            raise RuntimeError(
                f"projection arms did not share initialization: {record}"
            )
    return records


def trial_payload_sha256(trial: Mapping[str, object]) -> str:
    """Seal a trial payload while excluding its self-referential seal field."""

    return canonical_sha256(
        {key: value for key, value in trial.items() if key != "trial_payload_sha256"}
    )


def embedded_trial_manifest(payload: Mapping[str, object]) -> dict[str, str]:
    """Validate embedded trial seals and return their sorted manifest.

    Manifest keys use ``case_id/arm_id/seedN`` and therefore retain the exact
    naming convention of the existing spin and Hubbard campaign artifacts.
    """

    manifest: dict[str, str] = {}
    for case in payload.get("cases", []):
        case_mapping = cast(Mapping[str, object], case)
        for trial in case_mapping.get("trials", []):
            trial_mapping = cast(Mapping[str, object], trial)
            stored = str(trial_mapping["trial_payload_sha256"])
            if stored != trial_payload_sha256(trial_mapping):
                raise RuntimeError(
                    f"embedded trial payload seal failed for "
                    f"{case_mapping['case_id']}/{trial_mapping['arm_id']}/"
                    f"seed{trial_mapping['seed']}"
                )
            key = (
                f"{case_mapping['case_id']}/{trial_mapping['arm_id']}/"
                f"seed{trial_mapping['seed']}"
            )
            manifest[key] = stored
    return dict(sorted(manifest.items()))


__all__ = [
    "array_sha256",
    "atomic_json",
    "canonical_sha256",
    "coo_sha256",
    "embedded_trial_manifest",
    "paired_initialization_audit",
    "pytree_sha256",
    "repo_relative_path",
    "trial_payload_sha256",
]
