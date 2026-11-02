"""Fail-closed contract/provenance helpers for prospective D1-T C0 v8.

This module never constructs the N=36 basis or Hamiltonian.  It validates the
immutable authorization and checkpoint objects consumed by the two pinned
XDiag executables.  Registration alone does not authorize execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from typing import Any

import numpy as np


CONFIG_PATH = Path(
    "configs/physics_win/"
    "p9v2_d1_n36_seeded_locality_teacher_c0_production_v8.json"
)
CONFIG_SHA256 = "4cc274699563c6867cef84f11e746b503af5d74c9d2c59ad9add19b620762ab8"
ACCEPTANCE_PATH = Path(
    "sims/p9v2_d1_c0_acceptance_audit_v2/V8_REQUIRED_FIELDS_v1.json"
)
ACCEPTANCE_SHA256 = "e5ada547c903bad9b4125a37b9b72bba905fc6628ab0b462e477e9b4ea201120"
FORMAL_SEED_SCOPE = "FORMAL_N36_C0_V8_SEED_FILL_ONLY"
FORMAL_KRYLOV_SCOPE = "FORMAL_N36_C0_V8"
EXACT_SMALL_SCOPE = "EXACT_SMALL_C0_PREFLIGHT_V8"
SYNTHETIC_SCOPE = "SYNTHETIC_LOCAL_TEST_V8"
FULL_DIMENSION = 21_029_820
CHUNK_SIZE = 100_000
CHUNK_COUNT = 211
LAST_CHUNK_COUNT = 29_820
COMMITTED_DETERMINANT_CALLS = 63_089_460
EXPECTED_ALLOCATION_SHAPE = (
    "kim/kim/normal:nodes=1:tasks=1:cpus_per_task=40:"
    "memory_gib=128:wall=06:00:00:gpus=0"
)


def sha256_file(path: Path) -> str:
    """Hash one regular file without trusting metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key rejected: {key}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> Any:
        raise ValueError(f"nonfinite JSON scalar rejected: {token}")

    payload = json.loads(
        Path(path).read_text(),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _utc(value: str) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            value,
        )
        is None
    ):
        raise ValueError("a canonical whole-second UTC timestamp is required")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nested(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"missing required field {dotted}")
        value = value[component]
    return value


def load_frozen_config(root: Path) -> dict[str, Any]:
    """Load v8 only after its bytes and every static dependency validate."""

    root = Path(root)
    path = root / CONFIG_PATH
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("active C0 v8 config hash mismatch")
    acceptance = root / ACCEPTANCE_PATH
    if sha256_file(acceptance) != ACCEPTANCE_SHA256:
        raise ValueError("v8 acceptance table hash mismatch")
    config = _load_json(path)
    required = _load_json(acceptance)
    if list(config) != required["required_top_level_keys_exact"]:
        raise ValueError("v8 top-level schema differs from its frozen table")
    if config.get("schema_version") != 8 or config.get("stage") != (
        "P9v2-D1-T-C0-N36-seeded-locality-teacher-production-v8"
    ):
        raise ValueError("wrong active C0 identity")
    registered = _utc(config["registered_utc"])
    if not _utc(required["generated_utc"]) < registered:
        raise ValueError("v8 registration does not follow its requirement table")
    cell = config["cell"]
    scale = 0.75 * (
        int(cell["j1_edge_count"]) * abs(float(cell["j1"]))
        + int(cell["j2_edge_count"]) * abs(float(cell["j2_over_j1"]))
    )
    if scale != 91.125 or float(cell["scale_exact"]) != scale:
        raise ValueError("C0 scale mismatch rejected before any action")
    if sorted(cell["site_map_xdiag_to_repository"]) != list(range(36)):
        raise ValueError("registered N36 site map is not a permutation")
    authoritative = {
        record["role"]: {"path": record["path"], "sha256": record["sha256"]}
        for record in required["authoritative_parent_dependencies"]
    }
    expected_dependencies = dict(authoritative)
    expected_dependencies["acceptance_field_table"] = {
        "path": str(ACCEPTANCE_PATH), "sha256": ACCEPTANCE_SHA256
    }
    if set(config["bound_dependencies"]) != set(expected_dependencies):
        raise ValueError("v8 bound-dependency roles differ from frozen requirements")
    root_resolved = root.resolve(strict=True)
    for record in config["bound_dependencies"].values():
        if not isinstance(record, Mapping) or "path" not in record or "sha256" not in record:
            raise ValueError("dependency records require canonical path and SHA")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dependency path is not repository-relative")
        dependency = root / relative
        resolved = dependency.resolve(strict=True)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError("dependency symlink escapes repository root") from error
        if sha256_file(dependency) != record["sha256"]:
            raise ValueError(f"dependency hash mismatch: {record['path']}")
    for role, expected in expected_dependencies.items():
        observed = config["bound_dependencies"][role]
        if observed["path"] != expected["path"] or observed["sha256"] != expected["sha256"]:
            raise ValueError(f"v8 dependency identity mismatch: {role}")
    void_record = config["void_history"][-1]
    void_marker = _load_json(root / void_record["void_marker"]["path"])
    if (
        void_record["path"] != "configs/physics_win/p9v2_d1_n36_seeded_locality_teacher_c0_production_v7.json"
        or void_record["sha256"] != "11bb510e47bb110fab6bc4c8e80b218f56e9703d87d4412cbf82c5ef53144f90"
        or void_record["void_marker"]["sha256"] != "c7aa7c4a2f67ccf3487fde8dcf2b6eabf43027df49089b221f9d3ec8ad6c7f86"
        or void_marker.get("status") != "VOID_BEFORE_ANY_V7_OUTPUT_CONFIG_MATH_NO_GO"
        or any(void_marker.get(key) is not False for key in (
            "v7_exact_small_output_created", "v7_synthetic_output_created",
            "v7_n36_basis_constructed", "v7_n36_seed_fill_started",
            "v7_n36_hamiltonian_action_performed", "v7_energy_observed",
            "v7_checkpoint_created", "v7_implementation_frozen",
            "silent_in_place_repair_allowed", "wip_code_is_evidence",
        ))
    ):
        raise ValueError("v7 VOID chronology/evidence facts are invalid")
    return config


def validate_review_chronology(
    registered_utc: str,
    review_created_utcs: Sequence[str],
    prepared_utc: str | None = None,
    first_output_utc: str | None = None,
) -> None:
    """Validate unordered dual-review chronology."""

    if len(review_created_utcs) != 2:
        raise ValueError("exactly two independent static reviews are required")
    registered = _utc(registered_utc)
    reviews = [_utc(value) for value in review_created_utcs]
    observed_now = _utc(_utc_now())
    if any(review <= registered for review in reviews):
        raise ValueError("a static review does not strictly follow v8 registration")
    if any(review > observed_now for review in reviews):
        raise ValueError("a static review timestamp is in the future")
    if prepared_utc is None:
        return
    prepared = _utc(prepared_utc)
    if prepared < max(reviews):
        raise ValueError("PREPARED predates the later static review")
    if prepared > observed_now:
        raise ValueError("PREPARED timestamp is in the future")
    if first_output_utc is not None and _utc(first_output_utc) < prepared:
        raise ValueError("first output predates PREPARED")
    if first_output_utc is not None and _utc(first_output_utc) > observed_now:
        raise ValueError("first output timestamp is in the future")


@dataclass(frozen=True)
class PreparedValidation:
    path: Path
    sha256: str
    payload: dict[str, Any]


_PREPARED_ROLES = {
    "active_config",
    "acceptance_table",
    "P2_evidence",
    "exact_anchor_root_and_eight_cell_evidence",
    "total_splus_primary_and_independent",
    "transitive_local_sources",
    "seed_fill_executable",
    "block_krylov_executable",
    "build_manifest",
    "official_toml",
    "pinned_xdiag_source_patch_commit_manifest",
    "runtime_environment_manifest",
    "sbatch_and_runner",
    "operator_approval",
    "stable_run_identity",
}


def _canonical_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError("a nonempty repository-relative canonical path is required")
    path = Path(value)
    if ".." in path.parts or str(path) != value:
        raise ValueError("record path is not canonical repository-relative text")
    return value


def _full_sha(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("a complete lowercase SHA256 is required")
    return value


def _validate_record_file(
    record: Mapping[str, Any],
    *,
    repository_root: Path | None,
    exact_keys: set[str] = {"path", "sha256"},
) -> tuple[str, str]:
    if not isinstance(record, Mapping) or set(record) != exact_keys:
        raise ValueError("record has extra or missing fields")
    path = _canonical_relative_path(record["path"])
    digest = _full_sha(record["sha256"])
    if repository_root is not None:
        root_resolved = Path(repository_root).resolve(strict=True)
        artifact = Path(repository_root) / path
        cursor = root_resolved
        for component in Path(path).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError(f"record path traverses a symlink: {path}")
        resolved = artifact.resolve(strict=True)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(
                f"record symlink escapes repository root: {path}"
            ) from error
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"record path/SHA mismatch: {path}")
    return path, digest


def _validate_manifest_record(
    record: Mapping[str, Any],
    *,
    repository_root: Path | None,
    cardinality: int | None,
    expected_role: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "entries"}:
        raise ValueError("aggregate record has extra or missing fields")
    _canonical_relative_path(record["path"])
    _full_sha(record["sha256"])
    entries = record["entries"]
    if not isinstance(entries, list) or (cardinality is not None and len(entries) != cardinality):
        raise ValueError("aggregate record cardinality mismatch")
    roles: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"role", "path", "sha256"}:
            raise ValueError("aggregate entry has extra or missing fields")
        role = entry["role"]
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("aggregate entry roles must be unique")
        roles.add(role)
        _validate_record_file(
            {"path": entry["path"], "sha256": entry["sha256"]},
            repository_root=repository_root,
        )
    # The aggregate manifest itself is immutable and must exactly reproduce
    # the embedded entry list.
    if repository_root is not None:
        manifest = repository_root / str(record["path"])
        if not manifest.is_file() or sha256_file(manifest) != record["sha256"]:
            raise ValueError("aggregate manifest path/SHA mismatch")
        manifest_payload = _load_json(manifest)
        if manifest_payload != {
            "schema_version": 8,
            "stage": "P9v2-D1-T-C0-evidence-manifest-v8",
            "role": expected_role,
            "entries": entries,
        }:
            raise ValueError("aggregate manifest entries differ from PREPARED")
    return entries


def validate_prepared(
    path: Path,
    expected_sha256: str,
    *,
    scope: str,
    run_id: str,
    executable_role: str,
    executable_path: str,
    executable_sha256: str,
    official_input_path: str,
    official_input_sha256: str,
    repository_root: Path | None = None,
) -> PreparedValidation:
    """Open and validate the immutable formal handoff; env tokens are ignored."""

    path = Path(path)
    if repository_root is None:
        raise ValueError("formal PREPARED validation requires a repository root")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("malformed expected PREPARED SHA")
    if sha256_file(path) != expected_sha256:
        raise ValueError("PREPARED hash mismatch")
    payload = _load_json(path)
    required_top = {
        "schema_version", "stage", "created_utc", "authorized_scopes",
        "run_id", "active_config", "records", "review_markers",
    }
    if set(payload) != required_top:
        raise ValueError("PREPARED has extra or missing top-level fields")
    if payload.get("schema_version") != 8 or payload.get("stage") != "P9v2-D1-T-C0-v8-PREPARED":
        raise ValueError("wrong PREPARED schema/stage")
    authorized_scopes = payload.get("authorized_scopes")
    if (
        not isinstance(authorized_scopes, list)
        or set(authorized_scopes) != {FORMAL_SEED_SCOPE, FORMAL_KRYLOV_SCOPE}
        or scope not in authorized_scopes
    ):
        raise ValueError("PREPARED does not authorize the requested scope")
    if payload.get("run_id") != run_id or not run_id:
        raise ValueError("PREPARED run identity mismatch")
    active = payload.get("active_config")
    if active != {"path": str(CONFIG_PATH), "sha256": CONFIG_SHA256}:
        raise ValueError("PREPARED active-config identity mismatch")
    records = payload.get("records")
    if not isinstance(records, Mapping) or set(records) != _PREPARED_ROLES:
        raise ValueError("PREPARED roles are missing or unexpected")
    root = Path(repository_root)
    active_path, active_sha = _validate_record_file(
        records["active_config"], repository_root=root
    )
    if {"path": active_path, "sha256": active_sha} != active:
        raise ValueError("active-config top-level and record identities differ")
    acceptance_path, acceptance_sha = _validate_record_file(
        records["acceptance_table"], repository_root=root
    )
    if (acceptance_path, acceptance_sha) != (str(ACCEPTANCE_PATH), ACCEPTANCE_SHA256):
        raise ValueError("PREPARED acceptance-table identity mismatch")
    simple_roles = {
        "seed_fill_executable", "block_krylov_executable", "build_manifest",
        "official_toml", "pinned_xdiag_source_patch_commit_manifest",
        "runtime_environment_manifest", "operator_approval",
    }
    for role in simple_roles:
        extra = {"path", "sha256", "approved"} if role == "operator_approval" else {"path", "sha256"}
        _validate_record_file(records[role], repository_root=root, exact_keys=extra)
    if records[executable_role]["path"] != _canonical_relative_path(executable_path) or records[executable_role]["sha256"] != executable_sha256:
        raise ValueError("PREPARED executable hash mismatch")
    if records["official_toml"]["path"] != _canonical_relative_path(official_input_path) or records["official_toml"]["sha256"] != official_input_sha256:
        raise ValueError("PREPARED official input hash mismatch")
    approval = records["operator_approval"]
    if approval.get("approved") is not True:
        raise ValueError("operator approval is absent")
    p2_entries = _validate_manifest_record(
        records["P2_evidence"], repository_root=root, cardinality=3,
        expected_role="P2_evidence",
    )
    if {entry["role"] for entry in p2_entries} != {
        "P2_RESULT", "P2_RESOURCE_REVIEW", "P2_FINAL_RESULT"
    }:
        raise ValueError("P2 evidence manifest roles differ from the exact product")
    if root is not None:
        validate_p2_manifest_semantics(root, p2_entries)
    splus_entries = _validate_manifest_record(
        records["total_splus_primary_and_independent"],
        repository_root=root,
        cardinality=2,
        expected_role="total_splus_primary_and_independent",
    )
    if root is not None:
        validate_total_splus_manifest_semantics(root, splus_entries)
    source_entries = _validate_manifest_record(
        records["transitive_local_sources"], repository_root=root, cardinality=14,
        expected_role="transitive_local_sources",
    )
    config = load_frozen_config(root) if root is not None else None
    if config is not None:
        expected_role_paths = {
            entry["role"]: entry["path"]
            for entry in config["implementation_allowlist"]
        }
        observed_role_paths = {
            entry["role"]: entry["path"] for entry in source_entries
        }
        if observed_role_paths != expected_role_paths:
            raise ValueError(
                "transitive source manifest role/path mapping differs from "
                "implementation allowlist"
            )
    anchor_entries = _validate_manifest_record(
        records["exact_anchor_root_and_eight_cell_evidence"],
        repository_root=root,
        cardinality=None,
        expected_role="exact_anchor_root_and_eight_cell_evidence",
    )
    anchor_roles = {entry["role"] for entry in anchor_entries}
    required_anchor_roles = {
        "ANCHOR_RESULT", "CANARY_RESOURCE_REVIEW", "PAIR_CLASSIFICATION_PASS"
    } | {f"{task}_{kind}" for task in (
        "g0p09_nup18_seed42017", "g0p09_nup18_seed78193",
        "g0p09_nup19_seed42017", "g0p09_nup19_seed78193",
        "g0p125_nup18_seed42017", "g0p125_nup18_seed78193",
        "g0p125_nup19_seed42017", "g0p125_nup19_seed78193",
    ) for kind in ("RESULT", "DONE")}
    if not required_anchor_roles <= anchor_roles:
        raise ValueError("exact-anchor aggregate lacks the eight DONE/RESULT pairs")
    if root is not None:
        validate_exact_anchor_manifest_semantics(root, anchor_entries)
    sbatch_entries = _validate_manifest_record(
        records["sbatch_and_runner"], repository_root=root, cardinality=2,
        expected_role="sbatch_and_runner",
    )
    if {entry["role"] for entry in sbatch_entries} != {"sbatch", "runner"}:
        raise ValueError("sbatch/runner aggregate roles differ from frozen pair")
    stable_run = records["stable_run_identity"]
    if not isinstance(stable_run, Mapping) or set(stable_run) != {
        "run_id", "config_sha256", "input_sha256", "seed_fill_binary_sha256",
        "block_krylov_binary_sha256", "xdiag_commit",
    }:
        raise ValueError("stable run identity schema mismatch")
    if stable_run["run_id"] != run_id or stable_run["config_sha256"] != CONFIG_SHA256:
        raise ValueError("stable run identity differs from PREPARED")
    for key, value in stable_run.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"stable run identity {key} is empty")
    if (
        stable_run["input_sha256"] != records["official_toml"]["sha256"]
        or stable_run["seed_fill_binary_sha256"] != records["seed_fill_executable"]["sha256"]
        or stable_run["block_krylov_binary_sha256"] != records["block_krylov_executable"]["sha256"]
    ):
        raise ValueError("stable run input/binary hashes differ from role records")
    reviews = payload["review_markers"]
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise ValueError("exactly two unordered static review records are required")
    review_times: list[str] = []
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {"role", "path", "sha256", "created_utc", "verdict"}:
            raise ValueError("static review marker schema mismatch")
        if review["role"] not in {"acceptance_static_review", "math_static_review"}:
            raise ValueError("unknown static review role")
        review_path, _ = _validate_record_file(
            {"path": review["path"], "sha256": review["sha256"]},
            repository_root=root,
        )
        review_artifact = _load_json(root / review_path)
        if set(review_artifact) != {
            "schema_version", "stage", "role", "created_utc", "verdict",
            "active_config", "implementation_manifest_sha256",
            "confirms_no_v8_output",
        }:
            raise ValueError("static review artifact schema mismatch")
        if (
            review_artifact["schema_version"] != 8
            or review_artifact["stage"] != "P9v2-D1-T-C0-v8-static-review"
            or review_artifact["role"] != review["role"]
            or review_artifact["created_utc"] != review["created_utc"]
            or review_artifact["verdict"] != review["verdict"]
            or review_artifact["active_config"]
            != {"path": str(CONFIG_PATH), "sha256": CONFIG_SHA256}
            or review_artifact["implementation_manifest_sha256"]
            != records["transitive_local_sources"]["sha256"]
            or review_artifact["confirms_no_v8_output"] is not True
        ):
            raise ValueError("static review artifact content mismatch")
        if review["verdict"] != "STATIC_PASS_EXACT_SMALL_UNRUN":
            raise ValueError("static review did not pass before exact-small")
        review_times.append(review["created_utc"])
    if len({review["role"] for review in reviews}) != 2:
        raise ValueError("the two static review roles must be distinct")
    validate_review_chronology(
        "2026-09-14T02:12:46Z", review_times, payload["created_utc"]
    )
    return PreparedValidation(path=path, sha256=expected_sha256, payload=payload)


def validate_p2_evidence(root: Path) -> dict[str, Any]:
    """Validate the actual three-file P2 aggregate schema and hash links."""

    root = Path(root)
    result_path = root / "RESULT.json"
    review_path = root / "RESOURCE_REVIEW.json"
    final_path = root / "FINAL_RESULT.json"
    result, review, final = map(_load_json, (result_path, review_path, final_path))
    if result.get("status") != "P2_NUMERICAL_PASS_RESOURCE_REVIEW_REQUIRED":
        raise ValueError("P2 numerical aggregate is not complete")
    if review.get("status") != "P2_PASS_AMPLITUDE_ONLY" or final.get("status") != "P2_PASS_AMPLITUDE_ONLY":
        raise ValueError("P2 resource/final status is not PASS")
    numerical_gates = result.get("numerical_gates")
    resource_gates = review.get("gates")
    if (
        not isinstance(numerical_gates, Mapping)
        or not numerical_gates
        or not isinstance(resource_gates, Mapping)
        or not resource_gates
        or any(type(value) is not bool or value is not True for value in numerical_gates.values())
        or any(type(value) is not bool or value is not True for value in resource_gates.values())
    ):
        raise ValueError("P2 boolean gates are incomplete")
    result_sha = sha256_file(result_path)
    review_sha = sha256_file(review_path)
    if review.get("aggregated_result_sha256") != result_sha:
        raise ValueError("P2 review-to-result hash mismatch")
    if final.get("result_sha256") != result_sha or final.get("resource_review_sha256") != review_sha:
        raise ValueError("P2 final reciprocal hashes mismatch")
    return final


def _entry_paths(entries: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    return {str(entry["role"]): Path(str(entry["path"])) for entry in entries}


def validate_p2_manifest_semantics(
    repository_root: Path, entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate the exact three P2 records already hash-bound by PREPARED."""

    paths = _entry_paths(entries)
    result = _load_json(repository_root / paths["P2_RESULT"])
    review = _load_json(repository_root / paths["P2_RESOURCE_REVIEW"])
    final = _load_json(repository_root / paths["P2_FINAL_RESULT"])
    if result.get("status") != "P2_NUMERICAL_PASS_RESOURCE_REVIEW_REQUIRED":
        raise ValueError("P2 numerical status is not PASS-pending-review")
    if review.get("status") != "P2_PASS_AMPLITUDE_ONLY" or final.get("status") != "P2_PASS_AMPLITUDE_ONLY":
        raise ValueError("P2 final/resource status is not PASS")
    for mapping, label in ((result.get("numerical_gates"), "numerical"), (review.get("gates"), "resource")):
        if not isinstance(mapping, Mapping) or not mapping or any(
            type(value) is not bool or value is not True for value in mapping.values()
        ):
            raise ValueError(f"P2 {label} boolean map is empty or not all true")
    result_path = repository_root / paths["P2_RESULT"]
    review_path = repository_root / paths["P2_RESOURCE_REVIEW"]
    if (
        review.get("aggregated_result_sha256") != sha256_file(result_path)
        or final.get("result_sha256") != sha256_file(result_path)
        or final.get("resource_review_sha256") != sha256_file(review_path)
    ):
        raise ValueError("P2 reciprocal hashes are broken")
    return final


def validate_total_splus_manifest_semantics(
    repository_root: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    paths = _entry_paths(entries)
    if set(paths) != {"total_splus_primary", "total_splus_independent"}:
        raise ValueError("total-Splus manifest roles differ from frozen pair")
    if paths != {
        "total_splus_primary": Path("sims/p9v2_d1_xdiag_total_splus_bridge_preflight_v1/RESULT.json"),
        "total_splus_independent": Path("sims/p9v2_d1_xdiag_total_splus_bridge_independent_audit/AUDIT.json"),
    }:
        raise ValueError("total-Splus manifest paths differ from frozen artifacts")
    primary_path = repository_root / paths["total_splus_primary"]
    independent_path = repository_root / paths["total_splus_independent"]
    primary, independent = _load_json(primary_path), _load_json(independent_path)
    required_primary = {
        "schema_version": 1,
        "passed": True,
        "decision.status": "EXACT_SMALL_TOTAL_SPLUS_BRIDGE_PASS_D1_C0_STILL_REQUIRES_OTHER_GATES",
        "decision.n36_constructed_or_run": False,
        "decision.many_body_energy_or_eigensolve_performed": False,
    }
    required_independent = {
        "schema_version": 1,
        "verdict": "ACCEPT_EXACT_SMALL_TOTAL_SPLUS_BRIDGE_PREREQUISITE",
        "closes_c0_exact_small_splus_prerequisite": True,
        "does_not_unlock_c0_by_itself": True,
        "n36_certified": False,
        "audited_result_sha256": sha256_file(primary_path),
    }
    for dotted, expected in required_primary.items():
        if _nested(primary, dotted) != expected:
            raise ValueError(f"primary total-Splus field mismatch: {dotted}")
    for dotted, expected in required_independent.items():
        if _nested(independent, dotted) != expected:
            raise ValueError(f"independent total-Splus field mismatch: {dotted}")


def _walk_named_values(value: Any, key_fragment: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key_fragment in str(key).lower():
                found.append(child)
            found.extend(_walk_named_values(child, key_fragment))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_named_values(child, key_fragment))
    return found


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def validate_exact_anchor_manifest_semantics(
    repository_root: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    """Validate root markers and the exact 2x2x2 DONE/RESULT product."""

    paths = _entry_paths(entries)
    tasks = (
        "g0p09_nup18_seed42017", "g0p09_nup18_seed78193",
        "g0p09_nup19_seed42017", "g0p09_nup19_seed78193",
        "g0p125_nup18_seed42017", "g0p125_nup18_seed78193",
        "g0p125_nup19_seed42017", "g0p125_nup19_seed78193",
    )
    required_roles = {
        "ANCHOR_RESULT", "CANARY_RESOURCE_REVIEW", "PAIR_CLASSIFICATION_PASS"
    } | {f"{task}_{kind}" for task in tasks for kind in ("RESULT", "DONE")}
    required_roles |= {"CANARY_RESOURCE_EVIDENCE", "CANARY_SACCT"}
    if set(paths) != required_roles or len(paths) != 21:
        raise ValueError("exact-anchor manifest is not the exact 21-artifact product")
    anchor = _load_json(repository_root / paths["ANCHOR_RESULT"])
    resource = _load_json(repository_root / paths["CANARY_RESOURCE_REVIEW"])
    pair = _load_json(repository_root / paths["PAIR_CLASSIFICATION_PASS"])
    resource_evidence_path = repository_root / paths["CANARY_RESOURCE_EVIDENCE"]
    sacct_path = repository_root / paths["CANARY_SACCT"]
    if anchor.get("verdict") != "ED_ANCHOR_PASS" or anchor.get("passed") is not True or anchor.get("all_tasks_passed") is not True:
        raise ValueError("exact-anchor aggregate is not a full PASS")
    if resource.get("passed") is not True or pair.get("passed") is not True:
        raise ValueError("exact-anchor resource/pair-classification marker failed")
    task0_result = repository_root / paths[f"{tasks[0]}_RESULT"]
    task0_done = repository_root / paths[f"{tasks[0]}_DONE"]
    if (
        resource.get("task_result_sha256") != sha256_file(task0_result)
        or resource.get("task_done_sha256") != sha256_file(task0_done)
        or resource.get("resource_evidence_sha256") != sha256_file(resource_evidence_path)
        or resource.get("raw_sacct_sha256") != sha256_file(sacct_path)
        or resource.get("scheduler_evidence") != _load_json(resource_evidence_path)
    ):
        raise ValueError("exact-anchor canary resource reciprocal hashes/content fail")
    campaign_identity: Mapping[str, Any] | None = None
    observed_task_ids: list[str] = []
    for task in tasks:
        result_path = repository_root / paths[f"{task}_RESULT"]
        done = _load_json(repository_root / paths[f"{task}_DONE"])
        result = _load_json(result_path)
        if done.get("result_sha256") != sha256_file(result_path):
            raise ValueError(f"exact-anchor DONE/RESULT hash mismatch: {task}")
        if result.get("passed") is not True:
            raise ValueError(f"exact-anchor task did not pass: {task}")
        task_record = result.get("task")
        if not isinstance(task_record, Mapping) or task_record.get("task_id") != task:
            raise ValueError(f"exact-anchor task identity mismatch: {task}")
        observed_task_ids.append(str(task_record["task_id"]))
        current_identity = result.get("campaign_identity")
        if not isinstance(current_identity, Mapping):
            raise ValueError("exact-anchor campaign identity is missing")
        if campaign_identity is None:
            campaign_identity = current_identity
        if current_identity != campaign_identity:
            raise ValueError("exact-anchor campaign identity changed across tasks")
        if done.get("campaign_identity_sha256") != _canonical_sha256(current_identity):
            raise ValueError("exact-anchor DONE campaign hash mismatch")
        gates = result.get("gate_passes")
        if not isinstance(gates, Mapping) or not gates or any(
            type(value) is not bool or value is not True for value in gates.values()
        ):
            raise ValueError("exact-anchor gate_passes map is absent/empty/not all true")
        direct = np.asarray(result.get("direct_residual_norms"), dtype=np.float64)
        analysis_prefix = 16
        limits = result.get("limits")
        observed = result.get("observed")
        if not isinstance(limits, Mapping) or not isinstance(observed, Mapping):
            raise ValueError("exact-anchor observed/limits maps are missing")
        limit = float(limits.get("maximum_direct_residual", np.nan))
        if (
            analysis_prefix < 16
            or direct.ndim != 1
            or direct.size < analysis_prefix
            or np.any(~np.isfinite(direct))
            or np.any(direct < 0.0)
            or not np.isfinite(limit)
            or limit <= 0.0
            or np.any(direct[:analysis_prefix] > limit)
            or not np.isclose(
                float(observed.get("maximum_direct_residual", np.nan)),
                float(np.max(direct)),
                rtol=1e-12,
                atol=1e-15,
            )
        ):
            raise ValueError("exact-anchor direct residual diagnostics are invalid")
    aggregate_tasks = anchor.get("tasks")
    if not isinstance(aggregate_tasks, list) or [row.get("task_id") for row in aggregate_tasks] != list(tasks):
        raise ValueError("exact-anchor aggregate task table is not the exact frozen product")
    pair_hashes = pair.get("task_hashes")
    if not isinstance(pair_hashes, Mapping) or set(pair_hashes) != {tasks[0], tasks[2]}:
        raise ValueError("paired-classification task hashes are incomplete")
    for task in (tasks[0], tasks[2]):
        if pair_hashes[task] != {
            "result_sha256": sha256_file(repository_root / paths[f"{task}_RESULT"]),
            "done_sha256": sha256_file(repository_root / paths[f"{task}_DONE"]),
        }:
            raise ValueError("paired-classification reciprocal task hash mismatch")
    classification = pair.get("classification")
    if (
        not isinstance(classification, Mapping)
        or classification.get("classification_passed") is not True
        or pair.get("minimum_singlets_passed") is not True
    ):
        raise ValueError("paired exact-anchor classification gate failed")
    assert campaign_identity is not None
    campaign_sha = _canonical_sha256(campaign_identity)
    if (
        resource.get("campaign_identity_sha256") != campaign_sha
        or pair.get("campaign_identity_sha256") != campaign_sha
        or anchor.get("campaign_identity") != campaign_identity
    ):
        raise ValueError("exact-anchor root-marker campaign identity mismatch")
    classifications = _walk_named_values(anchor, "classification_passed")
    if not classifications or any(value is not True for value in classifications):
        raise ValueError("exact-anchor classification gates are incomplete")


def construct_c0_ownership_context(
    repository_root: Path, prepared_payload: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """Rebuild the conditional first-four-singlet interval context.

    Centers only fix the registered order.  Interval half-widths are rebuilt
    from both M=0 seed calculations, their direct residuals, reseed spread,
    and the registered M=0/M=1 matching tolerance.  The returned digest is a
    canonical JSON digest, not a substitute for the PREPARED evidence hashes.
    """

    records = prepared_payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("PREPARED records are absent")
    aggregate = records.get("exact_anchor_root_and_eight_cell_evidence")
    if not isinstance(aggregate, Mapping) or not isinstance(
        aggregate.get("entries"), list
    ):
        raise ValueError("PREPARED exact-anchor aggregate is absent")
    paths = _entry_paths(aggregate["entries"])
    validate_exact_anchor_manifest_semantics(repository_root, aggregate["entries"])
    anchor = _load_json(repository_root / paths["ANCHOR_RESULT"])
    coupling_rows = anchor.get("couplings")
    if not isinstance(coupling_rows, list):
        raise ValueError("exact-anchor coupling rows are absent")
    selected = [
        row
        for row in coupling_rows
        if isinstance(row, Mapping)
        and float(row.get("coupling", np.nan)) == 0.125
    ]
    if len(selected) != 1:
        raise ValueError("exact anchor lacks one g=0.125 row")
    row = selected[0]
    consensus = row.get("consensus_classification")
    seed_classifications = row.get("seed_classifications")
    if not isinstance(consensus, Mapping) or not isinstance(
        seed_classifications, list
    ):
        raise ValueError("exact-anchor classifications are absent")
    singlets = consensus.get("certified_singlets")
    if not isinstance(singlets, list) or len(singlets) < 4:
        raise ValueError("exact anchor does not certify four singlets")
    first_four = singlets[:4]
    indices = np.asarray([item["m0_index"] for item in first_four], dtype=np.int64)
    centers = np.asarray([item["energy"] for item in first_four], dtype=np.float64)
    if (
        indices.shape != (4,)
        or np.any(indices < 0)
        or np.any(np.diff(indices) <= 0)
        or np.any(~np.isfinite(centers))
        or np.any(np.diff(centers) < 0.0)
    ):
        raise ValueError("exact-anchor first-four order is malformed")
    seed_energies: list[np.ndarray] = []
    seed_residuals: list[np.ndarray] = []
    for seed in (42017, 78193):
        task = f"g0p125_nup18_seed{seed}_RESULT"
        result = _load_json(repository_root / paths[task])
        energies = np.asarray(result.get("energies"), dtype=np.float64)
        residuals = np.asarray(
            result.get("direct_residual_norms"), dtype=np.float64
        )
        if (
            energies.ndim != 1
            or residuals.shape != energies.shape
            or int(indices[-1]) >= energies.size
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(residuals))
            or np.any(residuals < 0.0)
        ):
            raise ValueError("exact-anchor raw seed diagnostics are invalid")
        seed_energies.append(energies[indices])
        seed_residuals.append(residuals[indices])
    for classified in seed_classifications:
        certified = (
            classified.get("certified_singlets")
            if isinstance(classified, Mapping)
            else None
        )
        if (
            not isinstance(certified, list)
            or len(certified) < 4
            or [int(item["m0_index"]) for item in certified[:4]]
            != indices.tolist()
            or classified.get("classification_passed") is not True
            or classified.get("boundary_guard_safe") is not True
        ):
            raise ValueError("exact-anchor seeds disagree on singlet identity")
    matching_tolerance = _strict_finite_number(
        consensus.get("comparison_tolerance"),
        "exact-anchor matching tolerance",
        nonnegative=True,
    )
    complete = bool(
        anchor.get("passed") is True
        and row.get("passed") is True
        and row.get("minimum_singlets_passed") is True
        and row.get("seed_singlet_agreement_passed") is True
        and consensus.get("classification_passed") is True
        and consensus.get("boundary_guard_safe") is True
    )
    from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
        construct_exact_anchor_intervals,
    )

    rebuilt = construct_exact_anchor_intervals(
        np.stack(seed_energies),
        np.stack(seed_residuals),
        centers,
        matching_tolerance=matching_tolerance,
        degeneracy_tolerance=1.0e-10,
        complete_through_fourth=complete,
    )
    context = {
        "exact_singlet_intervals": rebuilt.intervals.tolist(),
        "exact_cluster_ids": rebuilt.cluster_ids.tolist(),
        "anchor_complete_through_fourth": rebuilt.complete_through_fourth,
    }
    return context, _canonical_sha256(context)


def expected_chunk_range(ordinal: int) -> tuple[int, int]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < CHUNK_COUNT:
        raise ValueError("seed chunk ordinal is outside [0,211)")
    start = ordinal * CHUNK_SIZE
    return start, min(start + CHUNK_SIZE, FULL_DIMENSION)


def _recognized_partial(name: str, prefix: str) -> bool:
    return bool(re.fullmatch(rf"\.{re.escape(prefix)}-\d{{4}}\.[A-Za-z0-9_.-]+\.partial", name))


@dataclass(frozen=True)
class SeedChainSummary:
    chunks: int
    rows: int
    determinant_calls: int
    last_marker_sha256: str
    basis_initializations: int
    finite_counts: tuple[int, int, int]
    exact_zero_counts: tuple[int, int, int]
    near_zero_counts: tuple[int, int, int]
    representative_uint_xor: int
    representative_uint_sum_mod_2_64: int
    ordinal_weighted_uint_sum_mod_2_64: int


_CHUNK_METADATA_KEYS = {
    "schema_version", "stage", "scope", "config_sha256", "prepared_sha256",
    "run_id", "producer_invocation_id", "attempt_identity", "ordinal", "start",
    "stop", "count", "stored_rows", "shape", "dtype", "row_order",
    "determinant_calls", "forbidden_H_constructions", "forbidden_H_actions",
    "forbidden_energy_calculations", "forbidden_energy_logs",
    "finite_counts", "exact_zero_counts", "near_zero_counts",
    "representative_uint_xor", "representative_uint_sum_mod_2_64",
    "ordinal_weighted_uint_sum_mod_2_64", "basis_initialized_once_in_invocation",
    "basis_initialization_seconds", "chunk_fill_seconds", "process_elapsed_seconds",
    "process_max_rss_gib", "observed_openmp_team_size",
}
_CHUNK_MARKER_KEYS = {
    "schema_version", "stage", "ordinal", "parent_marker_sha256",
    "prepared_sha256", "run_id", "config_sha256", "science_identity",
    "producer_invocation_id", "attempt_identity", "payload_sha256",
    "metadata_sha256",
}


def _seed_entries(root: Path) -> tuple[set[int], set[int]]:
    chunks_root, markers_root = root / "chunks", root / "markers"
    if (
        not chunks_root.is_dir()
        or chunks_root.is_symlink()
        or not markers_root.is_dir()
        or markers_root.is_symlink()
    ):
        raise FileNotFoundError("seed chunks/markers directories are missing")
    chunk_indices: set[int] = set()
    for path in chunks_root.iterdir():
        if path.is_symlink():
            raise ValueError("seed chunk entry may not be a symlink")
        if path.is_dir() and _recognized_partial(path.name, "chunk"):
            continue
        match = re.fullmatch(r"chunk-(\d{4})", path.name)
        if not path.is_dir() or match is None:
            raise ValueError("unexpected seed chunk entry")
        chunk_indices.add(int(match.group(1)))
    marker_indices: set[int] = set()
    for path in markers_root.iterdir():
        if path.is_symlink():
            raise ValueError("seed marker entry may not be a symlink")
        if path.is_file() and (
            re.fullmatch(
                r"\.chunk-\d{4}\.json\.[A-Za-z0-9_.:-]+\.partial",
                path.name,
            )
            or re.fullmatch(r"\.LATEST\.[A-Za-z0-9_.:-]+\.partial", path.name)
        ):
            continue
        match = re.fullmatch(r"chunk-(\d{4})\.json", path.name)
        if not path.is_file() or match is None:
            raise ValueError("unexpected seed marker entry")
        marker_indices.add(int(match.group(1)))
    return chunk_indices, marker_indices


def quarantine_unmarked_seed_tail(root: Path, *, recovery_id: str) -> Path | None:
    """Rename one crash-window payload orphan to a recognized partial.

    No bytes are trusted or deleted.  Only the unique next ordinal with a
    final-name payload directory and no published marker is recoverable; every
    other mismatch remains a hard fork/orphan error.
    """

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", recovery_id):
        raise ValueError("recovery_id contains unsafe characters")
    root = Path(root)
    chunks_root, markers_root = root / "chunks", root / "markers"
    if not chunks_root.is_dir() or not markers_root.is_dir():
        raise FileNotFoundError("seed chunks/markers directories are missing")
    chunks: set[int] = set()
    markers: set[int] = set()
    for path in chunks_root.iterdir():
        if path.is_symlink():
            raise ValueError("seed recovery refuses a symlink chunk entry")
        if path.is_dir() and _recognized_partial(path.name, "chunk"):
            continue
        match = re.fullmatch(r"chunk-(\d{4})", path.name)
        if not path.is_dir() or match is None:
            raise ValueError("unexpected seed chunk entry")
        chunks.add(int(match.group(1)))
    for path in markers_root.iterdir():
        if path.is_symlink():
            raise ValueError("seed recovery refuses a symlink marker entry")
        if path.is_file() and (
            re.fullmatch(
                r"\.chunk-\d{4}\.json\.[A-Za-z0-9_.:-]+\.partial",
                path.name,
            )
            or re.fullmatch(r"\.LATEST\.[A-Za-z0-9_.:-]+\.partial", path.name)
        ):
            continue
        match = re.fullmatch(r"chunk-(\d{4})\.json", path.name)
        if not path.is_file() or match is None:
            raise ValueError("unexpected seed marker entry")
        markers.add(int(match.group(1)))
    if markers - chunks:
        raise ValueError("published seed marker lacks its payload")
    orphans = chunks - markers
    if not orphans:
        return None
    expected = len(markers)
    if markers != set(range(expected)) or orphans != {expected}:
        raise ValueError("seed payload mismatch is not one recoverable tail")
    source = chunks_root / f"chunk-{expected:04d}"
    target = chunks_root / f".chunk-{expected:04d}.{recovery_id}.partial"
    if target.exists():
        raise FileExistsError("seed recovery partial already exists")
    os.replace(source, target)
    directory_fd = os.open(chunks_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _validate_seed_prefix(
    root: Path,
    *,
    prepared_sha256: str,
    run_id: str,
    synthetic_payloads: bool,
) -> SeedChainSummary:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("seed checkpoint root must be a real directory")
    allowed_root = {"chunks", "markers"}
    if (root / "LATEST.json").exists():
        allowed_root.add("LATEST.json")
    if {entry.name for entry in root.iterdir()} != allowed_root:
        raise ValueError("seed checkpoint root allowlist mismatch")
    chunk_indices, marker_indices = _seed_entries(root)
    if chunk_indices != marker_indices:
        raise ValueError("seed chunk/marker fork or orphan")
    if not chunk_indices:
        return SeedChainSummary(
            0, 0, 0, "", 0, (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 0, 0
        )
    maximum = max(chunk_indices)
    if maximum >= CHUNK_COUNT or chunk_indices != set(range(maximum + 1)):
        raise ValueError("seed chain contains a gap or out-of-contract ordinal")

    parent_sha: str | None = None
    rows = determinant_calls = 0
    finite_totals = [0, 0, 0]
    zero_totals = [0, 0, 0]
    near_zero_totals = [0, 0, 0]
    representative_xor = 0
    representative_sum = 0
    ordinal_weighted_sum = 0
    invocation_ids: set[str] = set()
    science_identity: Mapping[str, Any] | None = None
    for ordinal in range(maximum + 1):
        start, stop = expected_chunk_range(ordinal)
        count = stop - start
        directory = root / "chunks" / f"chunk-{ordinal:04d}"
        children = list(directory.iterdir())
        names = {path.name for path in children}
        if (
            directory.is_symlink()
            or names != {"PAYLOAD.bin", "METADATA.json"}
            or any(path.is_symlink() or not path.is_file() for path in children)
        ):
            raise ValueError("seed chunk payload allowlist mismatch")
        payload_path = directory / "PAYLOAD.bin"
        metadata_path = directory / "METADATA.json"
        marker_path = root / "markers" / f"chunk-{ordinal:04d}.json"
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError("seed marker must be a real regular file")
        metadata, marker = _load_json(metadata_path), _load_json(marker_path)
        if set(metadata) != _CHUNK_METADATA_KEYS or set(marker) != _CHUNK_MARKER_KEYS:
            raise ValueError("seed metadata/marker has extra or missing fields")
        required = {
            "schema_version": 8,
            "stage": "P9v2-D1-T-C0-full-seed-fill-chunk-v8",
            "ordinal": ordinal,
            "start": start,
            "stop": stop,
            "count": count,
            "shape": [count, 3],
            "dtype": "little-endian-float64",
            "row_order": "C-row-major-XDiag-BasisSublattice64<3>-ordinal",
            "config_sha256": CONFIG_SHA256,
            "prepared_sha256": prepared_sha256,
            "run_id": run_id,
            "observed_openmp_team_size": 40,
            "basis_initialized_once_in_invocation": True,
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise ValueError("seed metadata identity/range/shape mismatch")
        if metadata["scope"] not in ({SYNTHETIC_SCOPE} if synthetic_payloads else {FORMAL_SEED_SCOPE}):
            raise ValueError("seed chunk scope mismatch")
        attempt = metadata["attempt_identity"]
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "attempt_id", "allocation_id", "slurm_job_id", "node_list",
            "runtime_manifest_sha256", "allocation_shape",
        } or any(not isinstance(value, str) or not value for value in attempt.values()):
            raise ValueError("seed per-attempt allocation/runtime identity is malformed")
        invocation = metadata["producer_invocation_id"]
        if not isinstance(invocation, str) or not invocation:
            raise ValueError("seed producer invocation identity is empty")
        invocation_ids.add(invocation)
        stored_rows = metadata["stored_rows"]
        if isinstance(stored_rows, bool) or not isinstance(stored_rows, int):
            raise ValueError("stored_rows must be an integer")
        if synthetic_payloads:
            if not 1 <= stored_rows <= count:
                raise ValueError("invalid synthetic abbreviated payload")
        elif stored_rows != count:
            raise ValueError("formal seed payload cannot be abbreviated")
        if payload_path.stat().st_size != stored_rows * 3 * 8:
            raise ValueError("seed payload byte count mismatch")
        if (
            metadata["determinant_calls"] != 3 * count
            or any(metadata[key] != 0 for key in (
                "forbidden_H_constructions", "forbidden_H_actions",
                "forbidden_energy_calculations", "forbidden_energy_logs",
            ))
        ):
            raise ValueError("seed work ledger includes forbidden H/energy work")
        for key in ("finite_counts", "exact_zero_counts", "near_zero_counts"):
            values = metadata[key]
            if not isinstance(values, list) or len(values) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise ValueError("seed per-column counter is malformed")
        if any(value != count for value in metadata["finite_counts"]):
            raise ValueError("each logical seed row must be finite in all columns")
        for key in (
            "representative_uint_xor", "representative_uint_sum_mod_2_64",
            "ordinal_weighted_uint_sum_mod_2_64",
        ):
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 64):
                raise ValueError("representative checksum must be uint64")
        for key in (
            "basis_initialization_seconds", "chunk_fill_seconds",
            "process_elapsed_seconds", "process_max_rss_gib",
        ):
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError("seed resource measurement is invalid")
        if float(metadata["process_max_rss_gib"]) > 89.6:
            raise ValueError("seed process RSS exceeds the frozen resource gate")
        for column in range(3):
            finite_totals[column] += metadata["finite_counts"][column]
            zero_totals[column] += metadata["exact_zero_counts"][column]
            near_zero_totals[column] += metadata["near_zero_counts"][column]
        representative_xor ^= int(metadata["representative_uint_xor"])
        representative_sum = (
            representative_sum + int(metadata["representative_uint_sum_mod_2_64"])
        ) % (1 << 64)
        ordinal_weighted_sum = (
            ordinal_weighted_sum
            + int(metadata["ordinal_weighted_uint_sum_mod_2_64"])
        ) % (1 << 64)
        current_science = marker["science_identity"]
        if not isinstance(current_science, Mapping) or set(current_science) != {
            "config_sha256", "input_sha256", "seed_fill_binary_sha256",
            "xdiag_commit", "run_id",
        }:
            raise ValueError("seed stable science identity schema mismatch")
        if (
            current_science["config_sha256"] != CONFIG_SHA256
            or current_science["run_id"] != run_id
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(current_science[key])) is None
                for key in ("config_sha256", "input_sha256", "seed_fill_binary_sha256")
            )
            or not isinstance(current_science["xdiag_commit"], str)
            or not current_science["xdiag_commit"]
        ):
            raise ValueError("seed stable science identity values are not bound")
        if science_identity is None:
            science_identity = current_science
        if current_science != science_identity:
            raise ValueError("seed stable science identity changed across chunks")
        expected_marker = {
            "schema_version": 8,
            "stage": "P9v2-D1-T-C0-full-seed-fill-marker-v8",
            "ordinal": ordinal,
            "parent_marker_sha256": parent_sha,
            "prepared_sha256": prepared_sha256,
            "run_id": run_id,
            "config_sha256": CONFIG_SHA256,
            "science_identity": current_science,
            "producer_invocation_id": invocation,
            "attempt_identity": attempt,
            "payload_sha256": sha256_file(payload_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
        if marker != expected_marker:
            raise ValueError("seed marker path/hash/identity/parent mismatch")
        parent_sha = sha256_file(marker_path)
        rows += count
        determinant_calls += int(metadata["determinant_calls"])
    return SeedChainSummary(
        maximum + 1,
        rows,
        determinant_calls,
        str(parent_sha),
        len(invocation_ids),
        tuple(finite_totals),
        tuple(zero_totals),
        tuple(near_zero_totals),
        representative_xor,
        representative_sum,
        ordinal_weighted_sum,
    )


def validate_seed_chunk_chain(
    root: Path,
    *,
    prepared_sha256: str,
    run_id: str,
    synthetic_payloads: bool = False,
) -> SeedChainSummary:
    """Recursively validate all 211 payloads, metadata objects and parents."""

    root = Path(root)
    summary = _validate_seed_prefix(
        root,
        prepared_sha256=prepared_sha256,
        run_id=run_id,
        synthetic_payloads=synthetic_payloads,
    )
    if summary.chunks != CHUNK_COUNT:
        raise ValueError("seed chain must contain exactly 211 committed chunks")
    latest_path = root / "LATEST.json"
    if latest_path.is_symlink() or not latest_path.is_file():
        raise ValueError("seed LATEST must be a real regular file")
    latest = _load_json(latest_path)
    if latest != {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-full-seed-fill-latest-v8",
        "ordinal": 210,
        "marker_sha256": summary.last_marker_sha256,
    }:
        raise ValueError("seed LATEST is missing, lagging, ahead or hash-invalid")
    if summary.rows != FULL_DIMENSION or summary.determinant_calls != COMMITTED_DETERMINANT_CALLS:
        raise ValueError("full seed aggregate ledger mismatch")
    return summary


def validate_seed_chunk_prefix(
    root: Path,
    *,
    prepared_sha256: str,
    run_id: str,
    synthetic_payloads: bool = False,
) -> SeedChainSummary:
    """Pure validation of a possibly empty, contiguous committed seed prefix."""

    return _validate_seed_prefix(
        Path(root),
        prepared_sha256=prepared_sha256,
        run_id=run_id,
        synthetic_payloads=synthetic_payloads,
    )


def promote_seed_lagging_latest(
    root: Path,
    *,
    prepared_sha256: str,
    run_id: str,
    synthetic_payloads: bool = False,
) -> dict[str, Any] | None:
    """Repair only a valid lagging pointer; never accept ahead/forked state."""

    root = Path(root)
    summary = _validate_seed_prefix(
        root,
        prepared_sha256=prepared_sha256,
        run_id=run_id,
        synthetic_payloads=synthetic_payloads,
    )
    latest_path = root / "LATEST.json"
    if summary.chunks == 0:
        if latest_path.exists():
            raise ValueError("seed LATEST exists ahead of an empty chain")
        return None
    if not latest_path.is_file():
        current = -1
    else:
        latest = _load_json(latest_path)
        if set(latest) != {"schema_version", "stage", "ordinal", "marker_sha256"}:
            raise ValueError("seed LATEST schema mismatch")
        current = int(latest["ordinal"])
        if current < 0 or current >= summary.chunks:
            raise ValueError("seed LATEST points ahead/outside valid chain")
        current_marker = root / "markers" / f"chunk-{current:04d}.json"
        if latest["marker_sha256"] != sha256_file(current_marker):
            raise ValueError("seed LATEST marker hash mismatch")
        if current == summary.chunks - 1:
            return latest
    final_ordinal = summary.chunks - 1
    promoted = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-full-seed-fill-latest-v8",
        "ordinal": final_ordinal,
        "marker_sha256": summary.last_marker_sha256,
    }
    atomic_json(latest_path, promoted)
    return promoted


def validate_full_seed_fill_done(
    path: Path,
    expected_sha256: str,
    *,
    seed_root: Path,
    prepared_sha256: str,
    run_id: str,
) -> SeedChainSummary:
    """Validate the second-stage marker and independently replay all 211 chunks."""

    path = Path(path)
    if sha256_file(path) != _full_sha(expected_sha256):
        raise ValueError("FULL_SEED_FILL_DONE SHA mismatch")
    marker = _load_json(path)
    required_keys = {
        "schema_version", "stage", "created_utc", "prepared_sha256", "run_id",
        "seed_chain_root", "chunk_count", "committed_rows",
        "committed_determinant_calls", "aggregate_checksums",
        "last_marker_sha256",
    }
    if set(marker) != required_keys or marker.get("schema_version") != 8 or marker.get("stage") != "P9v2-D1-T-C0-v8-FULL_SEED_FILL_DONE":
        raise ValueError("FULL_SEED_FILL_DONE schema/stage mismatch")
    if marker["prepared_sha256"] != prepared_sha256 or marker["run_id"] != run_id:
        raise ValueError("FULL_SEED_FILL_DONE parent/run mismatch")
    if Path(marker["seed_chain_root"]) != Path(seed_root):
        raise ValueError("FULL_SEED_FILL_DONE seed-root mismatch")
    summary = validate_seed_chunk_chain(
        seed_root, prepared_sha256=prepared_sha256, run_id=run_id
    )
    if (
        marker["chunk_count"] != summary.chunks
        or marker["committed_rows"] != summary.rows
        or marker["committed_determinant_calls"] != summary.determinant_calls
        or marker["last_marker_sha256"] != summary.last_marker_sha256
    ):
        raise ValueError("FULL_SEED_FILL_DONE aggregate differs from replay")
    aggregate = marker["aggregate_checksums"]
    expected_aggregate = {
        "finite_counts": list(summary.finite_counts),
        "exact_zero_counts": list(summary.exact_zero_counts),
        "near_zero_counts": list(summary.near_zero_counts),
        "representative_uint_xor": summary.representative_uint_xor,
        "representative_uint_sum_mod_2_64": summary.representative_uint_sum_mod_2_64,
        "ordinal_weighted_uint_sum_mod_2_64": summary.ordinal_weighted_uint_sum_mod_2_64,
        "basis_initializations": summary.basis_initializations,
    }
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        "finite_counts", "exact_zero_counts", "near_zero_counts",
        "representative_uint_xor", "representative_uint_sum_mod_2_64",
        "ordinal_weighted_uint_sum_mod_2_64", "basis_initializations",
    } or dict(aggregate) != expected_aggregate:
        raise ValueError("FULL_SEED_FILL_DONE checksum/basis ledger malformed")
    return summary


def read_row_major_seed_payloads(
    payloads: Sequence[Path], counts: Sequence[int]
) -> np.ndarray:
    """Directly map C-row-major chunks into a dimension-by-three matrix."""

    if len(payloads) != len(counts) or not payloads:
        raise ValueError("payload/count lists differ or are empty")
    result = np.empty((sum(counts), 3), dtype=np.float64)
    offset = 0
    for path, count in zip(payloads, counts, strict=True):
        values = np.fromfile(path, dtype="<f8")
        if values.size != count * 3:
            raise ValueError("row-major payload has wrong byte count")
        result[offset : offset + count, :] = values.reshape(count, 3, order="C")
        offset += count
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("seed payload contains nonfinite amplitudes")
    return result


_NUMERICAL_RESULT_KEYS = {
    "schema_version",
    "stage",
    "created_utc",
    "status",
    "numerical_verdict",
    "run_id",
    "parent_records",
    "ownership_context_sha256",
    "depth_summaries",
    "numerical_gates",
    "ownership_endpoint",
    "committed_science",
    "c1_unlock",
    "claim_boundary",
}
_NUMERICAL_GATE_KEYS = {
    "authorization_revalidated",
    "full_seed_fill_replayed",
    "checkpoint_chain_replayed_from_increment_only_real64",
    "depth_dimensions_equal_3_6_9",
    "frontier_rank_three_each_depth",
    "physical_null_same_span_gates_pass",
    "cluster_invariants_pass",
    "ownership_endpoint_schema_preserved_separate_from_verdict",
    "committed_science_exact",
    "nonfinite_absent",
}
_NUMERICAL_PARENT_KEYS = {
    "prepared",
    "full_seed_fill_done",
    "generation_markers",
    "event_chain",
}
_DEPTH_SUMMARY_KEYS = {
    "depth",
    "accepted_dimension",
    "physical_frontier_projector_replay_error",
    "null_frontier_projector_replay_error",
    "full_span_projector_frobenius_error",
    "ritz_subspace_projector_frobenius_error",
    "common_projected_clusters",
    "ownership_status",
    "total_committed_scalar_MVM",
}


def _validated_prepared_parent(
    repository_root: Path, record: Mapping[str, Any]
) -> PreparedValidation:
    prepared_path, prepared_sha = _validate_record_file(
        record, repository_root=repository_root
    )
    prepared_payload = _load_json(repository_root / prepared_path)
    records = prepared_payload.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("numerical RESULT parent PREPARED lacks records")
    executable = records.get("block_krylov_executable")
    official = records.get("official_toml")
    if not isinstance(executable, Mapping) or not isinstance(official, Mapping):
        raise ValueError("numerical RESULT PREPARED lacks executable/input")
    return validate_prepared(
        repository_root / prepared_path,
        prepared_sha,
        scope=FORMAL_KRYLOV_SCOPE,
        run_id=str(prepared_payload.get("run_id", "")),
        executable_role="block_krylov_executable",
        executable_path=str(executable.get("path", "")),
        executable_sha256=str(executable.get("sha256", "")),
        official_input_path=str(official.get("path", "")),
        official_input_sha256=str(official.get("sha256", "")),
        repository_root=repository_root,
    )


def _checkpoint_stable_identity(
    prepared: PreparedValidation, full_seed_sha256: str
) -> dict[str, str]:
    records = prepared.payload["records"]
    stable = records["stable_run_identity"]
    return {
        "config_sha256": str(stable["config_sha256"]),
        "source_manifest_sha256": str(
            records["transitive_local_sources"]["sha256"]
        ),
        "input_sha256": str(stable["input_sha256"]),
        "seed_fill_binary_sha256": str(stable["seed_fill_binary_sha256"]),
        "block_krylov_binary_sha256": str(
            stable["block_krylov_binary_sha256"]
        ),
        "xdiag_commit": str(stable["xdiag_commit"]),
        "run_id": str(stable["run_id"]),
        "prepared_sha256": prepared.sha256,
        "full_seed_fill_done_sha256": full_seed_sha256,
    }


def _recompute_c0_numerical_result(
    *,
    repository_root: Path,
    run_root: Path,
    created_utc: str,
    prepared_record: Mapping[str, Any],
    full_seed_fill_done_record: Mapping[str, Any],
    generation_marker_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild a numerical RESULT solely from immutable parent bytes."""

    repository_root = Path(repository_root)
    root_resolved = repository_root.resolve(strict=True)
    created = _utc(created_utc)
    prepared = _validated_prepared_parent(repository_root, prepared_record)
    if created < _utc(prepared.payload["created_utc"]):
        raise ValueError("numerical RESULT predates PREPARED")
    full_path, full_sha = _validate_record_file(
        full_seed_fill_done_record, repository_root=repository_root
    )
    full = _load_json(repository_root / full_path)
    seed_root_text = full.get("seed_chain_root")
    if not isinstance(seed_root_text, str) or not seed_root_text:
        raise ValueError("FULL seed parent lacks its chain root")
    seed_root = Path(seed_root_text).resolve(strict=True)
    try:
        seed_root.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("FULL seed chain root escapes repository root") from error
    validate_full_seed_fill_done(
        repository_root / full_path,
        full_sha,
        seed_root=seed_root,
        prepared_sha256=prepared.sha256,
        run_id=str(prepared.payload["run_id"]),
    )
    if created < _utc(full["created_utc"]):
        raise ValueError("numerical RESULT predates FULL seed completion")
    if len(generation_marker_records) != 3:
        raise ValueError("numerical RESULT requires exactly three generation markers")
    normalized_markers: list[dict[str, str]] = []
    marker_paths: list[Path] = []
    for depth, record in enumerate(generation_marker_records):
        path, digest = _validate_record_file(
            record, repository_root=repository_root
        )
        expected_suffix = Path("markers") / f"generation-{depth:04d}.json"
        if not Path(path).as_posix().endswith(expected_suffix.as_posix()):
            raise ValueError("generation marker paths are not in depth order")
        normalized_markers.append({"path": path, "sha256": digest})
        marker_paths.append(repository_root / path)
    checkpoint_roots = {path.parent.parent.resolve(strict=True) for path in marker_paths}
    if len(checkpoint_roots) != 1:
        raise ValueError("generation markers do not share one checkpoint root")
    checkpoint_root = checkpoint_roots.pop()
    try:
        checkpoint_root.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("checkpoint root escapes repository root") from error
    ownership_context, ownership_context_sha = construct_c0_ownership_context(
        repository_root, prepared.payload
    )
    stable = _checkpoint_stable_identity(prepared, full_sha)
    prepared_runtime_sha256 = _full_sha(
        prepared.payload["records"]["runtime_environment_manifest"]["sha256"]
    )
    replayed_reports: tuple[dict[str, Any], ...] | None = None

    def semantic_replay(
        states: tuple[dict[str, np.ndarray], ...],
        markers: tuple[dict[str, Any], ...],
    ) -> None:
        nonlocal replayed_reports
        from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
            replay_and_compare_checkpoint_semantics,
        )

        replayed_reports = replay_and_compare_checkpoint_semantics(
            states,
            markers,
            hamiltonian_scale=91.125,
            nsites=36,
            nup=18,
            coupling=0.125,
            expected_scope=FORMAL_KRYLOV_SCOPE,
            expected_geometry="triangular_J1J2_6x6_periodic_axial_torus",
            expected_ambient_dimension=FULL_DIMENSION,
            ownership_context=ownership_context,
            tolerance=1.0e-12,
        )

    from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
        validate_checkpoint_chain,
        validate_partitioned_ownership_payload,
    )

    markers = validate_checkpoint_chain(
        checkpoint_root,
        through_generation=2,
        expected_stable_identity=stable,
        semantic_replay=semantic_replay,
        require_semantic_replay=True,
    )
    if replayed_reports is None or len(replayed_reports) != 3:
        raise RuntimeError("checkpoint semantic replay did not return three depths")
    for depth, marker_path in enumerate(marker_paths):
        if sha256_file(marker_path) != normalized_markers[depth]["sha256"]:
            raise ValueError("generation marker changed during semantic replay")
        if markers[depth]["generation"] != depth:
            raise ValueError("generation marker order changed")
    run = Path(run_root).resolve(strict=True)
    try:
        run.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("numerical RESULT run root escapes repository") from error
    event_root = run / "events"
    if event_root.is_symlink() or not event_root.is_dir():
        raise ValueError("numerical RESULT event root is absent/unsafe")
    indexed_events: dict[int, Path] = {}
    for entry in event_root.iterdir():
        match = re.fullmatch(r"event-(\d{6})\.json", entry.name)
        if entry.is_symlink() or not entry.is_file() or match is None:
            raise ValueError("numerical RESULT refuses event partial/unknown entry")
        indexed_events[int(match.group(1))] = entry
    if not indexed_events or set(indexed_events) != set(range(max(indexed_events) + 1)):
        raise ValueError("numerical RESULT event chain is empty or gapped")
    allowed_after = {
        None: {"STARTED"},
        "STARTED": {"COMPLETED", "ABANDONED", "TERMINATED"},
        "COMPLETED": {"DONE", "ABANDONED"},
        "TERMINATED": {"REQUEUE"},
        "REQUEUE": {"STARTED"},
        "ABANDONED": {"STARTED"},
        "DONE": {"STARTED"},
    }
    previous_sha: str | None = None
    previous_type: str | None = None
    previous_time: datetime | None = None
    open_attempt: str | None = None
    attempts: dict[str, dict[str, Any]] = {}
    event_records: list[dict[str, str]] = []
    depth2_done: dict[str, Any] | None = None
    depth_done: dict[int, dict[str, Any]] = {}
    seed_done: dict[str, Any] | None = None
    full_identity_seen = False
    for index in sorted(indexed_events):
        path = indexed_events[index]
        event = _load_json(path)
        event_type = event.get("event_type")
        event_time = _utc(event.get("created_utc"))
        science = event.get("science_identity")
        attempt = event.get("attempt_identity")
        phase = event.get("phase")
        depth = event.get("depth")
        if (
            set(event)
            != {
                "schema_version", "stage", "event_index", "event_type",
                "created_utc", "phase", "depth", "science_identity",
                "attempt_identity", "parent_event_sha256", "details",
            }
            or event.get("schema_version") != 8
            or event.get("stage") != "P9v2-D1-T-C0-v8-event"
            or event.get("event_index") != index
            or event.get("parent_event_sha256") != previous_sha
            or event_type not in allowed_after[previous_type]
            or not isinstance(event.get("details"), Mapping)
            or not isinstance(science, Mapping)
            or set(science) != set(stable)
            or any(
                science[key] != value
                for key, value in stable.items()
                if key != "full_seed_fill_done_sha256"
            )
            or science["full_seed_fill_done_sha256"]
            not in {"NOT_CREATED_YET_BY_DESIGN", full_sha}
            or (
                isinstance(phase, str)
                and re.fullmatch(r"depth-[0-2]", phase) is not None
                and science["full_seed_fill_done_sha256"] != full_sha
            )
            or not isinstance(attempt, Mapping)
            or set(attempt)
            != {
                "attempt_id", "slurm_job_id", "node_list", "allocation_id",
                "runtime_manifest_sha256", "allocation_shape",
            }
            or any(not isinstance(value, str) or not value for value in attempt.values())
            or re.fullmatch(r"[1-9][0-9]*", attempt["slurm_job_id"]) is None
            or attempt["runtime_manifest_sha256"] != prepared_runtime_sha256
            or attempt["allocation_shape"] != EXPECTED_ALLOCATION_SHAPE
            or re.fullmatch(
                rf"slurm-{re.escape(attempt['slurm_job_id'])}-restart-(?:0|[1-9][0-9]*)",
                attempt["allocation_id"],
            )
            is None
            or not attempt["attempt_id"].endswith(
                f"-{attempt['allocation_id']}"
            )
            or not (
                (phase == "seed-fill" and depth is None)
                or (
                    isinstance(phase, str)
                    and re.fullmatch(r"depth-[0-2]", phase) is not None
                    and depth == int(phase[-1])
                )
                or (phase == "commit-recovery-only" and depth in (None, 0, 1, 2))
            )
            or event_time > created
            or (previous_time is not None and event_time < previous_time)
        ):
            raise ValueError("numerical RESULT event-chain semantic replay failed")
        observed_full = science["full_seed_fill_done_sha256"]
        if observed_full == full_sha:
            full_identity_seen = True
        elif full_identity_seen:
            raise ValueError("numerical RESULT event identity regresses before FULL")
        attempt_id = str(attempt["attempt_id"])
        if event_type == "STARTED":
            if open_attempt is not None or attempt_id in attempts:
                raise ValueError("numerical RESULT event lifecycles overlap/reuse")
            open_attempt = attempt_id
            attempts[attempt_id] = {
                "phase": phase,
                "attempt_identity": dict(attempt),
                "science_identity": dict(science),
                "started_utc": event["created_utc"],
                "terminal": None,
            }
        elif open_attempt != attempt_id:
            raise ValueError("numerical RESULT terminal event changes attempt")
        if event_type in {"DONE", "ABANDONED", "REQUEUE"}:
            attempts[attempt_id]["terminal"] = event_type
            open_attempt = None
        if event_type == "DONE" and phase == "depth-2":
            if depth2_done is not None:
                raise ValueError("more than one depth-two DONE claims the commit")
            depth2_done = event
        if event_type == "DONE" and isinstance(phase, str) and re.fullmatch(
            r"depth-[0-2]", phase
        ):
            if int(phase[-1]) in depth_done:
                raise ValueError("more than one DONE claims one committed depth")
            depth_done[int(phase[-1])] = event
        if event_type == "DONE" and phase == "seed-fill":
            if seed_done is not None:
                raise ValueError("more than one seed DONE claims FULL")
            seed_done = event
        digest = sha256_file(path)
        event_records.append(
            {
                "path": path.resolve(strict=True).relative_to(root_resolved).as_posix(),
                "sha256": digest,
            }
        )
        previous_sha = digest
        previous_type = str(event_type)
        previous_time = event_time
    if (
        open_attempt is not None
        or previous_type != "DONE"
        or any(item["terminal"] is None for item in attempts.values())
        or depth2_done is None
        or seed_done is None
        or set(depth_done) != {0, 1, 2}
    ):
        raise ValueError("numerical RESULT requires a closed DONE-tipped event chain")
    scanned_attempts = _scan_c0_attempt_artifacts(repository_root, run, attempts)
    for attempt_id, item in attempts.items():
        identity = scanned_attempts[attempt_id]["identity_payload"]
        if (
            set(identity)
            != {
                "schema_version", "stage", "created_utc", "phase",
                "science_identity", "attempt_identity",
            }
            or identity.get("schema_version") != 8
            or identity.get("stage") != "P9v2-D1-T-C0-v8-attempt-identity"
            or identity.get("phase") != item["phase"]
            or identity.get("science_identity") != item["science_identity"]
            or identity.get("attempt_identity") != item["attempt_identity"]
            or _utc(identity.get("created_utc")) > _utc(item["started_utc"])
        ):
            raise ValueError("numerical RESULT attempt IDENTITY replay failed")
    if (
        seed_done["details"].get("full_seed_fill_done_sha256") != full_sha
        or seed_done["details"].get("committed_rows") != FULL_DIMENSION
        or seed_done["details"].get("committed_determinant_calls")
        != COMMITTED_DETERMINANT_CALLS
    ):
        raise ValueError("seed DONE does not bind the FULL seed marker")
    for depth, done in depth_done.items():
        if (
            done["attempt_identity"] != markers[depth]["attempt_identity"]
            or done["details"].get("generation_marker_sha256")
            != normalized_markers[depth]["sha256"]
            or done["details"].get("paired_depth_committed") is not True
            or done["details"].get(
                "committed_scalar_equivalent_h_mvm_this_depth"
            )
            != 6
        ):
            raise ValueError("depth DONE does not bind its replayed commit")
    if (
        depth2_done["attempt_identity"] != markers[2]["attempt_identity"]
        or depth2_done["details"].get("generation_marker_sha256")
        != normalized_markers[2]["sha256"]
        or depth2_done["details"].get("paired_depth_committed") is not True
        or depth2_done["details"].get(
            "committed_scalar_equivalent_h_mvm_this_depth"
        )
        != 6
    ):
        raise ValueError("depth-two DONE does not bind the replayed commit")
    depth_summaries: list[dict[str, Any]] = []
    for depth, report in enumerate(replayed_reports):
        if set(report) != _DEPTH_SUMMARY_KEYS or report["depth"] != depth:
            raise ValueError("checkpoint replay report schema/depth mismatch")
        normalized = dict(report)
        normalized["common_projected_clusters"] = [
            list(cluster) for cluster in report["common_projected_clusters"]
        ]
        depth_summaries.append(normalized)
    ownership = validate_partitioned_ownership_payload(
        markers[2]["diagnostics"]["ownership"]
    )
    committed = {
        "seed_rows": int(full["committed_rows"]),
        "determinant_calls": int(full["committed_determinant_calls"]),
        "physical_H_batched_actions": int(
            markers[2]["diagnostics"]["physical_batched_h_actions"]
        ),
        "physical_H_scalar_MVM": int(
            markers[2]["diagnostics"]["physical_scalar_equivalent_h_mvm"]
        ),
        "null_H_batched_actions": int(
            markers[2]["diagnostics"]["null_batched_h_actions"]
        ),
        "null_H_scalar_MVM": int(
            markers[2]["diagnostics"]["null_scalar_equivalent_h_mvm"]
        ),
        "total_H_batched_actions": int(
            markers[2]["diagnostics"]["total_batched_h_actions"]
        ),
        "total_committed_scalar_MVM": int(
            markers[2]["diagnostics"]["total_committed_scalar_MVM"]
        ),
    }
    expected_committed = {
        "seed_rows": FULL_DIMENSION,
        "determinant_calls": COMMITTED_DETERMINANT_CALLS,
        "physical_H_batched_actions": 3,
        "physical_H_scalar_MVM": 9,
        "null_H_batched_actions": 3,
        "null_H_scalar_MVM": 9,
        "total_H_batched_actions": 6,
        "total_committed_scalar_MVM": 18,
    }
    if committed != expected_committed:
        raise ValueError("replayed committed-science ledger is not the frozen work")
    gates = {key: True for key in sorted(_NUMERICAL_GATE_KEYS)}
    config = load_frozen_config(repository_root)
    payload = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-v8-RESULT",
        "created_utc": created_utc,
        "status": "C0_NUMERICAL_PASS_RESOURCE_REVIEW_REQUIRED",
        "numerical_verdict": "C0_NUMERICAL_PASS",
        "run_id": str(prepared.payload["run_id"]),
        "parent_records": {
            "prepared": dict(prepared_record),
            "full_seed_fill_done": dict(full_seed_fill_done_record),
            "generation_markers": normalized_markers,
            "event_chain": {
                "record_count": len(event_records),
                "tip": event_records[-1],
                "records": event_records,
            },
        },
        "ownership_context_sha256": ownership_context_sha,
        "depth_summaries": depth_summaries,
        "numerical_gates": gates,
        "ownership_endpoint": ownership,
        "committed_science": committed,
        "c1_unlock": {
            "criterion": "C0_NUMERICAL_PASS",
            "unlocked": True,
            "ownership_status_does_not_lock_c1": True,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload


def build_and_validate_c0_numerical_result(
    *,
    repository_root: Path,
    run_root: Path,
    created_utc: str,
    prepared_record: Mapping[str, Any],
    full_seed_fill_done_record: Mapping[str, Any],
    generation_marker_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the pre-resource numerical PASS after full fresh replay."""

    payload = _recompute_c0_numerical_result(
        repository_root=repository_root,
        run_root=run_root,
        created_utc=created_utc,
        prepared_record=prepared_record,
        full_seed_fill_done_record=full_seed_fill_done_record,
        generation_marker_records=generation_marker_records,
    )
    return validate_c0_numerical_result_payload(payload)


def validate_c0_numerical_result_payload(
    payload: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate, and optionally replay, an immutable numerical RESULT."""

    if not isinstance(payload, Mapping) or set(payload) != _NUMERICAL_RESULT_KEYS:
        raise ValueError("C0 numerical RESULT has missing or extra fields")
    if (
        payload["schema_version"] != 8
        or payload["stage"] != "P9v2-D1-T-C0-v8-RESULT"
        or payload["status"] != "C0_NUMERICAL_PASS_RESOURCE_REVIEW_REQUIRED"
        or payload["numerical_verdict"] != "C0_NUMERICAL_PASS"
        or not isinstance(payload["run_id"], str)
        or not payload["run_id"]
    ):
        raise ValueError("C0 numerical RESULT identity/verdict is invalid")
    _utc(payload["created_utc"])
    _full_sha(payload["ownership_context_sha256"])
    parents = payload["parent_records"]
    if not isinstance(parents, Mapping) or set(parents) != _NUMERICAL_PARENT_KEYS:
        raise ValueError("C0 numerical RESULT parent schema mismatch")
    _validate_record_file(parents["prepared"], repository_root=repository_root)
    _validate_record_file(
        parents["full_seed_fill_done"], repository_root=repository_root
    )
    markers = parents["generation_markers"]
    if not isinstance(markers, list) or len(markers) != 3:
        raise ValueError("C0 numerical RESULT requires three marker records")
    for marker in markers:
        _validate_record_file(marker, repository_root=repository_root)
    event_chain = parents["event_chain"]
    if not isinstance(event_chain, Mapping) or set(event_chain) != {
        "record_count", "tip", "records"
    }:
        raise ValueError("C0 numerical RESULT event-chain manifest is malformed")
    if (
        isinstance(event_chain["record_count"], bool)
        or not isinstance(event_chain["record_count"], int)
        or event_chain["record_count"] < 1
        or not isinstance(event_chain["records"], list)
        or len(event_chain["records"]) != event_chain["record_count"]
    ):
        raise ValueError("C0 numerical RESULT event-chain cardinality is invalid")
    for record in event_chain["records"]:
        _validate_record_file(record, repository_root=repository_root)
    _validate_record_file(event_chain["tip"], repository_root=repository_root)
    if event_chain["tip"] != event_chain["records"][-1]:
        raise ValueError("C0 numerical RESULT event-chain tip differs from records")
    gates = payload["numerical_gates"]
    if (
        not isinstance(gates, Mapping)
        or set(gates) != _NUMERICAL_GATE_KEYS
        or any(type(value) is not bool or value is not True for value in gates.values())
    ):
        raise ValueError("C0 numerical RESULT gates are not exact all-true booleans")
    summaries = payload["depth_summaries"]
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise ValueError("C0 numerical RESULT depth summaries are incomplete")
    for depth, summary in enumerate(summaries):
        if (
            not isinstance(summary, Mapping)
            or set(summary) != _DEPTH_SUMMARY_KEYS
            or summary["depth"] != depth
            or summary["accepted_dimension"] != 3 * (depth + 1)
            or summary["total_committed_scalar_MVM"] != 6 * (depth + 1)
            or (
                depth < 2
                and summary["ownership_status"]
                != "OWNERSHIP_NOT_EVALUATED_BY_DESIGN"
            )
        ):
            raise ValueError("C0 numerical depth summary is malformed")
        for key in (
            "physical_frontier_projector_replay_error",
            "null_frontier_projector_replay_error",
            "full_span_projector_frobenius_error",
            "ritz_subspace_projector_frobenius_error",
        ):
            if _strict_finite_number(summary[key], key, nonnegative=True) > 1.0e-10:
                raise ValueError("C0 numerical replay discrepancy exceeds tolerance")
    from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
        validate_partitioned_ownership_payload,
    )

    ownership = validate_partitioned_ownership_payload(payload["ownership_endpoint"])
    if ownership["evaluation_depth"] != 2 or ownership["status"] != summaries[2]["ownership_status"]:
        raise ValueError("C0 ownership endpoint is not bound to depth two")
    expected_committed = {
        "seed_rows": FULL_DIMENSION,
        "determinant_calls": COMMITTED_DETERMINANT_CALLS,
        "physical_H_batched_actions": 3,
        "physical_H_scalar_MVM": 9,
        "null_H_batched_actions": 3,
        "null_H_scalar_MVM": 9,
        "total_H_batched_actions": 6,
        "total_committed_scalar_MVM": 18,
    }
    if payload["committed_science"] != expected_committed or any(
        type(value) is not int for value in payload["committed_science"].values()
    ):
        raise ValueError("C0 numerical committed-science ledger changed")
    if payload["c1_unlock"] != {
        "criterion": "C0_NUMERICAL_PASS",
        "unlocked": True,
        "ownership_status_does_not_lock_c1": True,
    }:
        raise ValueError("C1 unlock is not numerical-pass alone")
    expected_claim = {
        "is_nqs": False,
        "is_fusion_nqs": False,
        "phase_claim": False,
        "qsl_claim": False,
        "strict_locality_claim": False,
        "efficiency_claim": False,
        "null_is_physical_control": False,
        "allowed_label": "exact-Hilbert-space shallow block-Krylov teacher",
        "variational_nonclaim": "Lower Ritz energy under nested Krylov depth is a variational identity and cannot be called an NQS, phase, QSL, locality, carrier, accuracy, or efficiency win.",
    }
    if payload["claim_boundary"] != expected_claim:
        raise ValueError("C0 numerical RESULT claim boundary changed")
    json.dumps(dict(payload), allow_nan=False, sort_keys=True)
    if repository_root is not None:
        replayed = _recompute_c0_numerical_result(
            repository_root=repository_root,
            run_root=(Path(repository_root) / event_chain["tip"]["path"]).parent.parent,
            created_utc=str(payload["created_utc"]),
            prepared_record=parents["prepared"],
            full_seed_fill_done_record=parents["full_seed_fill_done"],
            generation_marker_records=markers,
        )
        if replayed != dict(payload):
            raise ValueError("C0 numerical RESULT differs from parent-byte replay")
    return dict(payload)


_SUBMISSION_INTENT_STAGE = "P9v2-D1-T-C0-v8-SUBMISSION_INTENT"
_SUBMISSION_STAGE = "P9v2-D1-T-C0-v8-SUBMISSION"
_HOLD_OBSERVATION_STAGE = "P9v2-D1-T-C0-v8-HOLD_OBSERVED"
_RELEASE_OBSERVATION_STAGE = "P9v2-D1-T-C0-v8-RELEASE_OBSERVED"
_ALLOCATION_STAGE = "P9v2-D1-T-C0-v8-ALLOCATION"
_SLURM_CLUSTER = "unicorn"
_SLURM_CURRENT_CLIENT_PATHS = {
    role: f"/usr/local/slurm/current/bin/{role}"
    for role in ("sbatch", "scontrol", "squeue", "sacct", "srun")
}
_SUBMISSION_RESOURCE_REQUEST = {
    "cluster": _SLURM_CLUSTER,
    "partition": "kim",
    "account": "kim",
    "qos": "normal",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 40,
    "memory_gib": 128,
    "wall_time": "06:00:00",
    "signal_seconds_before_end": 900,
    "requeue": True,
    "gpus": 0,
    "allocation_shape": EXPECTED_ALLOCATION_SHAPE,
}


def _ledger_file_record(repository_root: Path, path: Path) -> dict[str, str]:
    """Return a repository-relative record without following a symlink leaf."""

    root = Path(repository_root).resolve(strict=True)
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("ledger artifact is not a real regular file")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("ledger artifact escapes repository root") from error
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("ledger artifact traverses a symlink")
    return {"path": relative.as_posix(), "sha256": sha256_file(candidate)}


def _validate_submission_slurm_clients(
    value: Any,
) -> dict[str, dict[str, Any]]:
    """Validate one immutable, version-consistent Slurm client suite.

    The historical ``/usr/local/slurm/current`` spelling is disclosure only.
    Every later action uses and rehashes ``resolved_path``.  This is essential:
    ``current`` changed from 24.11.5 to 25.05.8 during the surrounding audit.
    """

    if not isinstance(value, Mapping) or set(value) != set(
        _SLURM_CURRENT_CLIENT_PATHS
    ):
        raise ValueError("submission Slurm client suite is incomplete")
    result: dict[str, dict[str, Any]] = {}
    parents: set[Path] = set()
    versions: set[str] = set()
    for role, current_path in _SLURM_CURRENT_CLIENT_PATHS.items():
        record = value[role]
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "path", "resolved_path", "sha256", "version_stdout",
                "version_stderr", "version_returncode",
            }
            or record.get("path") != current_path
            or record.get("version_returncode") != 0
            or record.get("version_stderr") != ""
        ):
            raise ValueError("submission Slurm client record is malformed")
        digest = _full_sha(record.get("sha256"))
        version_stdout = record.get("version_stdout")
        match = (
            re.fullmatch(r"slurm ([0-9]+(?:\.[0-9]+){2})\n?", version_stdout,
                         flags=re.IGNORECASE)
            if isinstance(version_stdout, str)
            else None
        )
        if match is None:
            raise ValueError("submission Slurm client version is not canonical")
        version = match.group(1)
        resolved = Path(str(record.get("resolved_path", "")))
        if (
            not resolved.is_absolute()
            or resolved.name != role
            or resolved.parent.name != "bin"
            or resolved.parent.parent.name != f"slurm-{version}"
            or resolved.is_symlink()
            or not resolved.is_file()
            or sha256_file(resolved) != digest
        ):
            raise ValueError("submission versioned Slurm client changed or is absent")
        parents.add(resolved.parent)
        versions.add(version)
        result[role] = dict(record)
    if len(parents) != 1 or len(versions) != 1:
        raise ValueError("submission Slurm client suite is version-inconsistent")
    return result


def _validate_submission_intent_payload(
    payload: Any,
    *,
    repository_root: Path,
    run_root: Path,
    run_id: str,
    index: int,
    parent_intent_sha256: str | None,
    latest_allowed_utc: datetime,
) -> dict[str, Any]:
    exact = {
        "schema_version", "stage", "submission_index", "created_utc",
        "run_id", "requested_phase", "depth", "client_nonce", "job_name",
        "comment", "prepared_record", "runtime_manifest_sha256",
        "full_seed_fill_done_sha256", "paths", "sbatch_script_record",
        "resource_request", "slurm_clients", "exported_environment",
        "sbatch_argv", "parent_intent_sha256",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != exact
        or payload.get("schema_version") != 8
        or payload.get("stage") != _SUBMISSION_INTENT_STAGE
        or payload.get("submission_index") != index
        or payload.get("run_id") != run_id
        or payload.get("parent_intent_sha256") != parent_intent_sha256
        or re.fullmatch(r"[0-9a-f]{32}", str(payload.get("client_nonce", "")))
        is None
    ):
        raise ValueError("submission intent exact schema/identity is invalid")
    created = _utc(payload.get("created_utc"))
    if created > latest_allowed_utc:
        raise ValueError("submission intent postdates its consumer")
    phase = payload.get("requested_phase")
    depth = payload.get("depth")
    if phase == "seed-fill":
        if depth is not None:
            raise ValueError("seed submission has a depth")
    elif isinstance(phase, str) and re.fullmatch(r"depth-[0-2]", phase):
        if depth != int(phase[-1]):
            raise ValueError("submission depth differs from phase")
    elif phase == "resume":
        if depth is not None:
            raise ValueError("resume submission preselects a depth")
    else:
        raise ValueError("submission phase is outside the formal allowlist")
    nonce = str(payload["client_nonce"])
    if (
        payload.get("job_name") != f"nqs-c0v8-{index:06d}-{nonce[:12]}"
        or payload.get("comment") != f"nqs-c0-v8:{nonce}"
        or payload.get("resource_request") != _SUBMISSION_RESOURCE_REQUEST
    ):
        raise ValueError("submission scheduler/resource identity changed")
    root = Path(repository_root).resolve(strict=True)
    run = Path(run_root).resolve(strict=True)
    expected_paths = {
        "run_root": run.relative_to(root).as_posix(),
        "seed_root": (run / "seed_fill").relative_to(root).as_posix(),
        "full_seed_done": (run / "FULL_SEED_FILL_DONE.json").relative_to(root).as_posix(),
        "checkpoint_root": (run / "checkpoint").relative_to(root).as_posix(),
    }
    if payload.get("paths") != expected_paths:
        raise ValueError("submission formal paths differ from the run layout")
    prepared = payload.get("prepared_record")
    prepared_path, prepared_sha = _validate_record_file(
        prepared, repository_root=root
    )
    prepared_payload = _load_json(root / prepared_path)
    if (
        prepared_payload.get("run_id") != run_id
        or prepared_payload.get("stage") != "P9v2-D1-T-C0-v8-PREPARED"
    ):
        raise ValueError("submission PREPARED record has another run identity")
    runtime_sha = _full_sha(payload.get("runtime_manifest_sha256"))
    records = prepared_payload.get("records")
    if (
        not isinstance(records, Mapping)
        or not isinstance(records.get("runtime_environment_manifest"), Mapping)
        or records["runtime_environment_manifest"].get("sha256") != runtime_sha
    ):
        raise ValueError("submission runtime SHA differs from PREPARED")
    full = payload.get("full_seed_fill_done_sha256")
    if phase == "seed-fill":
        if full != "NOT_CREATED_YET_BY_DESIGN":
            raise ValueError("seed submission predeclares a FULL marker")
        exported_full = ""
    else:
        exported_full = _full_sha(full)
    script_path, script_sha = _validate_record_file(
        payload.get("sbatch_script_record"), repository_root=root
    )
    if script_path != (
        "cluster/unicorn/p9v2_d1_seeded_block_krylov_c0_v8_cpu.sbatch"
    ):
        raise ValueError("submission sbatch path is not the v8 wrapper")
    clients = _validate_submission_slurm_clients(payload.get("slurm_clients"))
    runtime_record = records["runtime_environment_manifest"]
    runtime_path, _ = _validate_record_file(runtime_record, repository_root=root)
    runtime_payload = _load_json(root / runtime_path)
    python_path = runtime_payload.get("python")
    if not isinstance(python_path, str) or not python_path.startswith("/"):
        raise ValueError("submission runtime Python is not absolute")
    intent_path = run / "submissions" / f"intent-{index:06d}.json"
    expected_exports = {
        "NQS_REMOTE_ROOT": str(root),
        "NQS_PYTHON": python_path,
        "NQS_RUN_ROOT": str(run.parent),
        "NQS_RUN_ID": run_id,
        "NQS_C0_PHASE": str(phase),
        "NQS_PREPARED": str(root / prepared_path),
        "NQS_PREPARED_SHA256": prepared_sha,
        "NQS_RUNTIME_MANIFEST_SHA256": runtime_sha,
        "NQS_FULL_SEED_DONE_SHA256": exported_full,
        "NQS_SEED_ROOT": str(run / "seed_fill"),
        "NQS_FULL_SEED_DONE": str(run / "FULL_SEED_FILL_DONE.json"),
        "NQS_CHECKPOINT_ROOT": str(run / "checkpoint"),
        "NQS_SUBMISSION_INDEX": f"{index:06d}",
        "NQS_SUBMISSION_INTENT": str(intent_path),
        "NQS_SLURM_BIN_DIR": str(
            Path(clients["srun"]["resolved_path"]).parent
        ),
    }
    if payload.get("exported_environment") != expected_exports:
        raise ValueError("submission exported environment is not exactly derived")
    export_argument = "--export=" + ",".join(
        f"{key}={value}" for key, value in sorted(expected_exports.items())
    )
    expected_argv = [
        str(clients["sbatch"]["resolved_path"]),
        "--clusters=unicorn", "--hold", "--parsable",
        f"--job-name={payload['job_name']}",
        f"--comment={payload['comment']}", f"--chdir={root}",
        export_argument, str(root / script_path),
    ]
    if payload.get("sbatch_argv") != expected_argv:
        raise ValueError("submission sbatch argv is not exactly derived")
    return dict(payload)


def _validate_submission_record_payload(
    payload: Any,
    *,
    record_path: Path,
    intent: Mapping[str, Any],
    intent_sha256: str,
    latest_allowed_utc: datetime,
) -> dict[str, Any]:
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema_version", "stage", "submission_index", "created_utc",
            "intent_record", "slurm_job_id", "cluster", "sbatch_execution",
        }
        or payload.get("schema_version") != 8
        or payload.get("stage") != _SUBMISSION_STAGE
        or payload.get("submission_index") != intent["submission_index"]
        or payload.get("intent_record")
        != {
            "path": str(record_path.parent / f"intent-{intent['submission_index']:06d}.json"),
            "sha256": intent_sha256,
        }
        or re.fullmatch(r"[1-9][0-9]*", str(payload.get("slurm_job_id", "")))
        is None
        or payload.get("cluster") != _SLURM_CLUSTER
    ):
        raise ValueError("submission record exact schema/identity is invalid")
    created = _utc(payload.get("created_utc"))
    if not _utc(intent["created_utc"]) <= created <= latest_allowed_utc:
        raise ValueError("submission record chronology is invalid")
    execution = payload.get("sbatch_execution")
    if (
        not isinstance(execution, Mapping)
        or set(execution)
        != {
            "mode", "argv", "started_utc", "finished_utc", "returncode",
            "stdout", "stderr",
        }
        or execution.get("mode")
        not in {"DIRECT_HELD_SBATCH", "RECOVERED_HELD_JOB_FROM_INTENT_QUERY"}
        or execution.get("argv") != intent.get("sbatch_argv")
        or execution.get("returncode") != 0
        or not isinstance(execution.get("stdout"), str)
        or not isinstance(execution.get("stderr"), str)
    ):
        raise ValueError("submission sbatch execution record is malformed")
    started = _utc(execution["started_utc"])
    finished = _utc(execution["finished_utc"])
    if not _utc(intent["created_utc"]) <= started <= finished <= created:
        raise ValueError("submission execution chronology is invalid")
    if execution["mode"] == "DIRECT_HELD_SBATCH":
        match = re.fullmatch(
            r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?",
            execution["stdout"].strip(),
        )
        if (
            match is None
            or match.group(1) != payload["slurm_job_id"]
            or (match.group(2) or _SLURM_CLUSTER) != _SLURM_CLUSTER
        ):
            raise ValueError("submission stdout does not bind job and cluster")
    return dict(payload)


def validate_c0_submission_allocation_ledger(
    *,
    repository_root: Path,
    run_root: Path,
    run_id: str,
    latest_allowed_utc: str,
) -> dict[str, Any]:
    """Replay the four-stage submission chain and every wrapper allocation.

    Missing lifecycle edges are retained as typed gate failures so a submitted
    but pre-wrapper-killed allocation cannot disappear.  Unsafe names,
    symlinks, duplicate records, broken hashes, or malformed existing bytes are
    structural attacks and fail immediately.
    """

    root = Path(repository_root).resolve(strict=True)
    run = Path(run_root).resolve(strict=True)
    try:
        run.relative_to(root)
    except ValueError as error:
        raise ValueError("formal run root escapes repository root") from error
    latest = _utc(latest_allowed_utc)
    submission_root = run / "submissions"
    if submission_root.is_symlink() or not submission_root.is_dir():
        raise ValueError("submission ledger root is absent or unsafe")
    maps: dict[str, dict[int, Path]] = {
        "intent": {}, "submission": {}, "hold": {}, "release": {}
    }
    for entry in submission_root.iterdir():
        if entry.name == "SUBMISSION.lock":
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("submission lock is unsafe")
            continue
        match = re.fullmatch(
            r"(intent|submission|hold|release)-([0-9]{6})\.json", entry.name
        )
        if match is None or entry.is_symlink() or not entry.is_file():
            raise ValueError("submission root contains an unknown/unsafe entry")
        kind, raw_index = match.groups()
        index = int(raw_index)
        if index in maps[kind]:
            raise ValueError("duplicate submission ledger index")
        maps[kind][index] = entry
    intents = maps["intent"]
    if not intents or set(intents) != set(range(max(intents) + 1)):
        raise ValueError("submission intent ledger is empty or has a gap")
    if (
        not set(maps["submission"]) <= set(intents)
        or not set(maps["hold"]) <= set(maps["submission"])
        or not set(maps["release"]) <= set(maps["hold"])
    ):
        raise ValueError("submission child exists without its parent")
    entries: list[dict[str, Any]] = []
    jobs: set[str] = set()
    prior_intent_sha: str | None = None
    prior_created: datetime | None = None
    for index in range(max(intents) + 1):
        intent_path = intents[index]
        intent_sha = sha256_file(intent_path)
        intent = _validate_submission_intent_payload(
            _load_json(intent_path), repository_root=root, run_root=run,
            run_id=run_id, index=index,
            parent_intent_sha256=prior_intent_sha,
            latest_allowed_utc=latest,
        )
        intent_created = _utc(intent["created_utc"])
        if prior_created is not None and intent_created < prior_created:
            raise ValueError("submission intent chronology moves backwards")
        prior_created = intent_created
        prior_intent_sha = intent_sha
        issues: list[str] = []
        submission_record = None
        hold_record = None
        release_record = None
        job_id = None
        if index not in maps["submission"]:
            issues.append("MISSING_DURABLE_SUBMISSION_JOB_HANDLE")
        else:
            submission_path = maps["submission"][index]
            submission = _validate_submission_record_payload(
                _load_json(submission_path), record_path=submission_path,
                intent=intent, intent_sha256=intent_sha,
                latest_allowed_utc=latest,
            )
            job_id = str(submission["slurm_job_id"])
            if job_id in jobs:
                raise ValueError("one Slurm job is reused by multiple submissions")
            jobs.add(job_id)
            submission_record = _ledger_file_record(root, submission_path)
            if index not in maps["hold"]:
                issues.append("MISSING_HOLD_OBSERVATION")
            else:
                hold_path = maps["hold"][index]
                hold = _load_json(hold_path)
                expected_submission = {
                    "path": str(submission_path),
                    "sha256": submission_record["sha256"],
                }
                if (
                    set(hold)
                    != {
                        "schema_version", "stage", "submission_index",
                        "created_utc", "submission_record", "job_record",
                        "hold_classification",
                    }
                    or hold.get("schema_version") != 8
                    or hold.get("stage") != _HOLD_OBSERVATION_STAGE
                    or hold.get("submission_index") != index
                    or hold.get("submission_record") != expected_submission
                    or not isinstance(hold.get("job_record"), str)
                    or hold.get("hold_classification")
                    not in {"EXACT_HELD_JOB_OBSERVED", "NOT_HELD_RESOURCE_NO_GO"}
                    or not _utc(submission["created_utc"])
                    <= _utc(hold.get("created_utc")) <= latest
                ):
                    raise ValueError("hold observation is malformed")
                if hold["hold_classification"] != "EXACT_HELD_JOB_OBSERVED":
                    issues.append("HELD_JOB_NOT_EXACTLY_OBSERVED")
                hold_record = _ledger_file_record(root, hold_path)
                if index not in maps["release"]:
                    issues.append("MISSING_RELEASE_OBSERVATION")
                else:
                    release_path = maps["release"][index]
                    release = _load_json(release_path)
                    expected_hold = {
                        "path": str(hold_path), "sha256": hold_record["sha256"]
                    }
                    if (
                        set(release)
                        != {
                            "schema_version", "stage", "submission_index",
                            "created_utc", "submission_record", "hold_observation",
                            "release_command", "job_record_after",
                            "release_classification",
                        }
                        or release.get("schema_version") != 8
                        or release.get("stage") != _RELEASE_OBSERVATION_STAGE
                        or release.get("submission_index") != index
                        or release.get("submission_record") != expected_submission
                        or release.get("hold_observation") != expected_hold
                        or release.get("release_classification")
                        not in {
                            "HELD_JOB_RELEASED_AFTER_DURABLE_REGISTRATION",
                            "ALREADY_NOT_HELD_OBSERVED_IDEMPOTENTLY",
                        }
                        or not isinstance(release.get("release_command"), Mapping)
                        or not isinstance(release.get("job_record_after"), str)
                        or not _utc(hold["created_utc"])
                        <= _utc(release.get("created_utc")) <= latest
                    ):
                        raise ValueError("release observation is malformed")
                    release_record = _ledger_file_record(root, release_path)
        entries.append(
            {
                "submission_index": index,
                "requested_phase": intent["requested_phase"],
                "job_name": intent["job_name"],
                "comment": intent["comment"],
                "job_id": job_id,
                "intent_record": _ledger_file_record(root, intent_path),
                "submission_record": submission_record,
                "hold_record": hold_record,
                "release_record": release_record,
                "sacct_client": dict(intent["slurm_clients"]["sacct"]),
                "issues": issues,
            }
        )

    allocation_root = run / "allocations"
    allocations: list[dict[str, Any]] = []
    allocation_residuals: list[dict[str, str]] = []
    if allocation_root.exists() or allocation_root.is_symlink():
        if allocation_root.is_symlink() or not allocation_root.is_dir():
            raise ValueError("allocation ledger root is unsafe")
        submission_by_index = {entry["submission_index"]: entry for entry in entries}
        seen_allocation_ids: set[str] = set()
        for path in sorted(allocation_root.iterdir(), key=lambda item: item.name):
            partial = re.fullmatch(
                r"\.allocation-([1-9][0-9]*)-restart-(0|[1-9][0-9]*)\.json\.([1-9][0-9]*)\.partial",
                path.name,
            )
            canonical = re.fullmatch(
                r"allocation-([1-9][0-9]*)-restart-(0|[1-9][0-9]*)\.json",
                path.name,
            )
            if path.is_symlink() or not path.is_file():
                raise ValueError("allocation ledger entry is not a real file")
            if partial is not None:
                allocation_residuals.append(_ledger_file_record(root, path))
                continue
            if canonical is None:
                raise ValueError("allocation ledger contains an unknown entry")
            job_id, restart = canonical.groups()
            payload = _load_json(path)
            if (
                set(payload)
                != {
                    "schema_version", "stage", "created_utc", "run_id",
                    "requested_phase", "submission_index", "submission_intent",
                    "allocation_identity", "raw_slurm_environment",
                }
                or payload.get("schema_version") != 8
                or payload.get("stage") != _ALLOCATION_STAGE
                or payload.get("run_id") != run_id
                or re.fullmatch(r"[0-9]{6}", str(payload.get("submission_index", "")))
                is None
                or _utc(payload.get("created_utc")) > latest
            ):
                raise ValueError("allocation ledger exact schema/run is invalid")
            index = int(payload["submission_index"])
            parent = submission_by_index.get(index)
            if parent is None or parent["job_id"] != job_id:
                raise ValueError("allocation ledger does not bind a submission job")
            intent_path = submission_root / f"intent-{index:06d}.json"
            if payload.get("submission_intent") != {
                "path": str(intent_path), "sha256": sha256_file(intent_path)
            }:
                raise ValueError("allocation ledger intent record changed")
            identity = payload.get("allocation_identity")
            allocation_id = f"slurm-{job_id}-restart-{restart}"
            if (
                not isinstance(identity, Mapping)
                or set(identity)
                != {
                    "slurm_job_id", "restart_count", "node_list",
                    "allocation_id", "runtime_manifest_sha256",
                    "allocation_shape",
                }
                or identity.get("slurm_job_id") != job_id
                or identity.get("restart_count") != restart
                or identity.get("allocation_id") != allocation_id
                or identity.get("allocation_shape") != EXPECTED_ALLOCATION_SHAPE
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(identity.get("node_list", ""))) is None
                or _full_sha(identity.get("runtime_manifest_sha256"))
                != _load_json(intent_path)["runtime_manifest_sha256"]
            ):
                raise ValueError("allocation ledger identity is malformed")
            raw_environment = payload.get("raw_slurm_environment")
            if (
                not isinstance(raw_environment, Mapping)
                or set(raw_environment)
                != {
                    "cpus_per_task", "ntasks", "nodes", "mem_per_node",
                    "mem_per_cpu", "job_gpus", "cuda_visible_devices",
                }
            ):
                raise ValueError("allocation raw environment schema changed")
            environment_exact = (
                raw_environment["cpus_per_task"] == "40"
                and raw_environment["ntasks"] == "1"
                and raw_environment["nodes"] == "1"
                and raw_environment["mem_per_node"] == "131072"
                and raw_environment["mem_per_cpu"] == "0"
                and raw_environment["job_gpus"] in {"", "0"}
                and raw_environment["cuda_visible_devices"]
                in {"", "-1", "NoDevFiles"}
            )
            if allocation_id in seen_allocation_ids:
                raise ValueError("duplicate allocation identity")
            seen_allocation_ids.add(allocation_id)
            allocations.append(
                {
                    "allocation_id": allocation_id,
                    "slurm_job_id": job_id,
                    "restart_count": int(restart),
                    "submission_index": index,
                    "requested_phase": payload["requested_phase"],
                    "node_list": identity["node_list"],
                    "runtime_manifest_sha256": identity["runtime_manifest_sha256"],
                    "allocation_shape": identity["allocation_shape"],
                    "environment_exact": environment_exact,
                    "record": _ledger_file_record(root, path),
                }
            )
    entries_complete = all(not entry["issues"] for entry in entries)
    return {
        "submission_entries": entries,
        "allocation_entries": allocations,
        "allocation_publication_residuals": allocation_residuals,
        "gates": {
            "four_stage_submission_chain_complete": entries_complete,
            "allocation_publication_residuals_absent": not allocation_residuals,
            "allocation_environment_exact": all(
                item["environment_exact"] for item in allocations
            ),
        },
    }


_SACCT_RAW_COLUMNS = (
    "Cluster", "JobID", "JobIDRaw", "JobName", "Submit", "Start", "End",
    "State", "ExitCode", "ElapsedRaw", "AllocCPUS", "NNodes", "NTasks",
    "ReqMem", "AllocTRES", "MaxRSS", "Partition", "Account", "QOS",
    "NodeList", "Restarts",
)
_RAW_SACCT_KEYS = {
    "schema_version",
    "stage",
    "captured_utc",
    "run_id",
    "query",
    "parser",
    "raw_stdout_utf8",
    "raw_stdout_sha256",
    "rows",
}
_RAW_SACCT_QUERY_KEYS = {
    "argv",
    "environment",
    "job_ids_in_first_attempt_order",
    "capture_started_utc",
    "capture_completed_utc",
    "returncode",
    "stderr_utf8",
    "hostname",
    "executable_sha256",
    "version_stdout",
}


def _sacct_raw_utc(value: Any, label: str) -> str:
    """Normalize one Slurm whole-second timestamp captured under ``TZ=UTC``."""

    if not isinstance(value, str):
        raise ValueError(f"raw sacct {label} timestamp is not text")
    rendered = value if value.endswith("Z") else value + "Z"
    _utc(rendered)
    return rendered


def _sacct_gib(value: Any, label: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"raw sacct {label} is not text")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT])?", value)
    if match is None:
        raise ValueError(f"raw sacct {label} is not a finite memory value")
    scalar = float(match.group(1))
    factor = {None: 1.0, "K": 1.0 / (1024.0**2), "M": 1.0 / 1024.0,
              "G": 1.0, "T": 1024.0}[match.group(2)]
    result = scalar * factor
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"raw sacct {label} is invalid")
    return result


def _sacct_requested_memory_gib(value: Any, nodes: int, cpus: int) -> float:
    if not isinstance(value, str):
        raise ValueError("raw sacct ReqMem is not text")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT])([cn]?)", value)
    if match is None:
        raise ValueError("raw sacct ReqMem has an unsupported representation")
    base = _sacct_gib(match.group(1) + match.group(2), "ReqMem")
    suffix = match.group(3)
    return base * (cpus if suffix == "c" else nodes if suffix == "n" else 1)


def _sacct_gpu_count(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("raw sacct AllocTRES is not text")
    generic: int | None = None
    typed: list[int] = []
    for component in value.split(","):
        if not component:
            continue
        if "=" not in component:
            raise ValueError("raw sacct AllocTRES component lacks '='")
        key, raw_count = component.rsplit("=", 1)
        if not key or not raw_count:
            raise ValueError("raw sacct AllocTRES component is empty")
        if key in {"gres/gpu", "gpu"}:
            if not raw_count.isdigit():
                raise ValueError("raw sacct GPU count is not integral")
            count = int(raw_count)
            generic = count
        elif key.startswith("gres/gpu:") or key.startswith("gpu:"):
            if not raw_count.isdigit():
                raise ValueError("raw sacct typed GPU count is not integral")
            count = int(raw_count)
            typed.append(count)
    if generic is not None and typed and generic != sum(typed):
        raise ValueError("raw sacct generic/typed GPU counts disagree")
    return generic if generic is not None else sum(typed)


def _raw_sacct_command(job_ids: Sequence[str]) -> list[str]:
    return [
        _SACCT_EXECUTABLE,
        "-j",
        ",".join(job_ids),
        "--clusters=unicorn",
        "--units=K",
        "--parsable2",
        "--duplicates",
        "--format=" + ",".join(_SACCT_RAW_COLUMNS),
    ]


def _sacct_terminal_state(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("RAW_SACCT state is empty")
    state = value.split()[0].removesuffix("+")
    allowed = {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
        "NODE_FAIL", "PREEMPTED", "REQUEUED", "REVOKED", "BOOT_FAIL",
        "DEADLINE",
    }
    if state not in allowed:
        raise ValueError("RAW_SACCT state is not terminal")
    return state


def _sacct_row_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in _SACCT_RAW_COLUMNS)


def _parse_raw_sacct_snapshot(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    expected_attempts: Sequence[Mapping[str, Any]],
    latest_allowed_utc: datetime,
    require_embedded_rows: bool = True,
) -> list[dict[str, Any]]:
    """Reparse literal ``sacct --duplicates --parsable2`` allocation/step bytes."""

    if not isinstance(payload, Mapping) or set(payload) != _RAW_SACCT_KEYS:
        raise ValueError("RAW_SACCT snapshot has missing or extra fields")
    if (
        payload.get("schema_version") != 8
        or payload.get("stage") != "P9v2-D1-T-C0-v8-RAW_SACCT"
        or payload.get("run_id") != run_id
    ):
        raise ValueError("RAW_SACCT snapshot identity is invalid")
    captured = _utc(payload.get("captured_utc"))
    if captured > latest_allowed_utc:
        raise ValueError("RAW_SACCT was captured after its consumer")
    query = payload.get("query")
    if not isinstance(query, Mapping) or set(query) != _RAW_SACCT_QUERY_KEYS:
        raise ValueError("RAW_SACCT query provenance is malformed")
    capture_started = _utc(query.get("capture_started_utc"))
    capture_completed = _utc(query.get("capture_completed_utc"))
    if capture_started > capture_completed or capture_completed != captured:
        raise ValueError("RAW_SACCT capture chronology is invalid")
    if query.get("environment") != {"LC_ALL": "C", "TZ": "UTC"}:
        raise ValueError("RAW_SACCT locale/timezone environment is not frozen")
    if (
        query.get("returncode") != 0
        or query.get("stderr_utf8") != ""
        or not isinstance(query.get("hostname"), str)
        or not query["hostname"]
        or re.fullmatch(r"[0-9a-f]{64}", str(query.get("executable_sha256")))
        is None
        or not isinstance(query.get("version_stdout"), str)
        or re.search(r"\bslurm\s+[0-9]+(?:\.[0-9]+)+\b", query["version_stdout"], re.I)
        is None
    ):
        raise ValueError("RAW_SACCT command execution provenance is incomplete")
    expected_job_ids: list[str] = []
    expected_allocation_ids: set[str] = set()
    for attempt in expected_attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("RAW_SACCT expected attempt is not an object")
        identity = attempt.get("attempt_identity")
        if (
            not isinstance(identity, Mapping)
            or set(identity)
            != {
                "attempt_id", "slurm_job_id", "node_list", "allocation_id",
                "runtime_manifest_sha256", "allocation_shape",
            }
            or any(not isinstance(value, str) or not value for value in identity.values())
            or attempt.get("terminal") not in {"DONE", "ABANDONED", "REQUEUE"}
        ):
            raise ValueError("RAW_SACCT expected attempt identity/lifecycle is malformed")
        job_id = str(identity["slurm_job_id"])
        allocation_id = str(identity["allocation_id"])
        if not allocation_id or allocation_id in expected_allocation_ids:
            raise ValueError("each attempt requires one unique allocation identity")
        expected_allocation_ids.add(allocation_id)
        if job_id not in expected_job_ids:
            expected_job_ids.append(job_id)
    if query.get("job_ids_in_first_attempt_order") != expected_job_ids:
        raise ValueError("RAW_SACCT job selection omits/reorders an allocation")
    if query.get("argv") != _raw_sacct_command(expected_job_ids):
        raise ValueError("RAW_SACCT command differs from the registered raw query")
    if payload.get("parser") != {
        "name": "STRICT_SACCT_PIPE_V8",
        "columns": list(_SACCT_RAW_COLUMNS),
        "delimiter": "|",
        "header": True,
        "units": "K",
        "selection_rule": (
            "exact Unicorn root plus every same-start step; unique batch/extern; "
            "Restarts binds allocation_id"
        ),
    }:
        raise ValueError("RAW_SACCT parser provenance is not exact")
    raw = payload.get("raw_stdout_utf8")
    if not isinstance(raw, str) or not raw.endswith("\n"):
        raise ValueError("RAW_SACCT stdout must be complete newline-terminated UTF-8")
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if payload.get("raw_stdout_sha256") != raw_sha:
        raise ValueError("RAW_SACCT stdout SHA does not match its literal bytes")
    lines = raw.splitlines()
    if not lines:
        raise ValueError("RAW_SACCT stdout is empty")
    header = lines[0].split("|")
    if header and header[-1] == "":
        header.pop()
    if tuple(header) != _SACCT_RAW_COLUMNS:
        raise ValueError("RAW_SACCT header differs from the registered fields")
    parsed_raw: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) != len(_SACCT_RAW_COLUMNS) + 1 or fields[-1] != "":
            raise ValueError("RAW_SACCT line has wrong parsable2 cardinality")
        record = dict(zip(_SACCT_RAW_COLUMNS, fields[:-1], strict=True))
        if any("|" in value or "\n" in value or "\r" in value for value in record.values()):
            raise ValueError("RAW_SACCT field contains a delimiter/control character")
        parsed_raw.append(record)
    if not parsed_raw:
        raise ValueError("RAW_SACCT contains no accounting rows")
    restarts_by_job: dict[str, list[int]] = {}
    for attempt in expected_attempts:
        identity = attempt["attempt_identity"]
        job_id = str(identity["slurm_job_id"])
        match = re.fullmatch(
            rf"slurm-{re.escape(job_id)}-restart-(0|[1-9][0-9]*)",
            str(identity["allocation_id"]),
        )
        if match is None:
            raise ValueError("event allocation ID does not bind Slurm restart count")
        restarts_by_job.setdefault(job_id, []).append(int(match.group(1)))
    for job_id, restarts in restarts_by_job.items():
        if sorted(restarts) != list(range(max(restarts) + 1)):
            raise ValueError(f"event restart lineage is incomplete for job {job_id}")
    normalized: list[dict[str, Any]] = []
    selected_rows: set[tuple[str, ...]] = set()
    for attempt in expected_attempts:
        identity = attempt["attempt_identity"]
        job_id = str(identity["slurm_job_id"])
        restart = int(str(identity["allocation_id"]).rsplit("-", 1)[1])
        roots = [
            row for row in parsed_raw
            if row["Cluster"] == "unicorn"
            and row["JobIDRaw"] == job_id
            and re.fullmatch(rf"{re.escape(job_id)}(?:_[0-9]+)?", row["JobID"])
            and row["JobName"] == "nqs-d1-c0-v8"
            and row["Restarts"] == str(restart)
        ]
        if len(roots) != 1:
            raise ValueError("RAW_SACCT lacks one exact root for an attempt/restart")
        root = roots[0]
        submit = _sacct_raw_utc(root["Submit"], "Submit")
        start = _sacct_raw_utc(root["Start"], "Start")
        end = _sacct_raw_utc(root["End"], "End")
        if not _utc(submit) <= _utc(start) <= _utc(end) <= captured:
            raise ValueError("RAW_SACCT root chronology is invalid")
        elapsed = int(root["ElapsedRaw"])
        if elapsed <= 0 or abs((_utc(end) - _utc(start)).total_seconds() - elapsed) > 1.0:
            raise ValueError("RAW_SACCT root elapsed does not match Start/End")
        if not root["AllocCPUS"].isdigit() or int(root["AllocCPUS"]) != 40:
            raise ValueError("RAW_SACCT root CPU allocation is not 40")
        if not root["NNodes"].isdigit() or int(root["NNodes"]) != 1:
            raise ValueError("RAW_SACCT root node allocation is not one")
        if root["NTasks"] not in {"", "1"}:
            raise ValueError("RAW_SACCT root task count is incompatible with one")
        requested_memory = _sacct_requested_memory_gib(root["ReqMem"], 1, 40)
        if requested_memory != 128.0:
            raise ValueError("RAW_SACCT requested memory differs from 128 GiB")
        if _sacct_gpu_count(root["AllocTRES"]) != 0:
            raise ValueError("RAW_SACCT root allocation contains a GPU")
        state = _sacct_terminal_state(root["State"])
        if re.fullmatch(r"[0-9]+:[0-9]+", root["ExitCode"]) is None:
            raise ValueError("RAW_SACCT root ExitCode is malformed")
        terminal = attempt.get("terminal")
        if terminal == "DONE":
            if state != "COMPLETED" or root["ExitCode"] != "0:0":
                raise ValueError("DONE lifecycle is not COMPLETED/0:0 in sacct")
        elif terminal == "REQUEUE":
            if state not in {"REQUEUED", "PREEMPTED", "CANCELLED", "TIMEOUT"}:
                raise ValueError("REQUEUE lifecycle has no requeue-like sacct state")
        elif terminal == "ABANDONED":
            if state == "COMPLETED" and root["ExitCode"] == "0:0":
                raise ValueError("ABANDONED lifecycle is falsely COMPLETED/0:0")
        else:
            raise ValueError("RAW_SACCT expected lifecycle lacks a terminal state")
        steps = [
            row for row in parsed_raw
            if row["Cluster"] == "unicorn"
            and row["JobIDRaw"].startswith(job_id + ".")
            and row["Start"] == root["Start"]
            and row["Restarts"] in {"", str(restart)}
        ]
        by_suffix = {row["JobIDRaw"].removeprefix(job_id + "."): row for row in steps}
        if len(by_suffix) != len(steps) or not {"batch", "extern"} <= set(by_suffix):
            raise ValueError("RAW_SACCT lacks unique batch/extern steps")
        rss_values: list[float] = []
        for suffix, step in by_suffix.items():
            if step["JobName"] != (suffix if suffix in {"batch", "extern"} else step["JobName"]):
                raise ValueError("RAW_SACCT batch/extern job name is inconsistent")
            if (
                _sacct_raw_utc(step["Submit"], "step Submit") != start
                or not step["ElapsedRaw"].isdigit()
                or int(step["ElapsedRaw"]) <= 0
                or not step["AllocCPUS"].isdigit()
                or int(step["AllocCPUS"]) != 40
                or step["NNodes"] not in {"", "1"}
                or step["NTasks"] not in {"", "1"}
                or re.fullmatch(r"[0-9]+:[0-9]+", step["ExitCode"]) is None
            ):
                raise ValueError("RAW_SACCT step accounting is malformed")
            step_end = _utc(_sacct_raw_utc(step["End"], "step End"))
            if (
                step_end < _utc(start)
                or step_end > _utc(end) + timedelta(seconds=1)
                or step_end > captured
            ):
                raise ValueError("RAW_SACCT step End exceeds root by more than one second")
            _sacct_terminal_state(step["State"])
            if step["MaxRSS"]:
                rss_values.append(_sacct_gib(step["MaxRSS"], "step MaxRSS"))
        if not rss_values or max(rss_values) <= 0.0:
            raise ValueError("RAW_SACCT lacks a positive step MaxRSS")
        group_keys = {_sacct_row_key(root)} | {_sacct_row_key(step) for step in steps}
        if selected_rows.intersection(group_keys):
            raise ValueError("RAW_SACCT accounting row is reused across attempts")
        selected_rows.update(group_keys)
        normalized.append({
            "attempt_id": identity["attempt_id"],
            "slurm_job_id": job_id,
            "cluster": root["Cluster"],
            "job_name": root["JobName"],
            "submit_utc": submit,
            "start_utc": start,
            "end_utc": end,
            "elapsed_seconds": elapsed,
            "max_rss_gib": max(rss_values),
            "partition": root["Partition"],
            "account": root["Account"],
            "qos": root["QOS"],
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": 40,
            "memory_gib": 128,
            "gpu_count": 0,
            "allocation_id": identity["allocation_id"],
            "node_list": root["NodeList"],
            "allocation_shape": identity["allocation_shape"],
            "state": state,
            "exit_code": root["ExitCode"],
        })
    if selected_rows != {_sacct_row_key(row) for row in parsed_raw}:
        raise ValueError("RAW_SACCT contains unbound or cherry-picked accounting rows")
    if require_embedded_rows and payload.get("rows") != normalized:
        raise ValueError("RAW_SACCT normalized rows differ from literal stdout replay")
    return normalized


def _build_c0_raw_sacct_snapshot(
    *,
    captured_utc: str,
    capture_started_utc: str,
    run_id: str,
    expected_attempts: Sequence[Mapping[str, Any]],
    raw_stdout_utf8: str,
    hostname: str,
    executable_sha256: str,
    version_stdout: str,
    stderr_utf8: str = "",
    returncode: int = 0,
) -> dict[str, Any]:
    """Build a strict operator snapshot while retaining literal sacct stdout."""

    expected_job_ids: list[str] = []
    for attempt in expected_attempts:
        identity = attempt.get("attempt_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("RAW_SACCT builder attempt lacks its identity")
        job_id = str(identity.get("slurm_job_id", ""))
        if not job_id:
            raise ValueError("RAW_SACCT builder attempt lacks a Slurm job ID")
        if job_id not in expected_job_ids:
            expected_job_ids.append(job_id)
    payload: dict[str, Any] = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-v8-RAW_SACCT",
        "captured_utc": captured_utc,
        "run_id": run_id,
        "query": {
            "argv": _raw_sacct_command(expected_job_ids),
            "environment": {
                "LC_ALL": "C",
                "TZ": "UTC",
            },
            "job_ids_in_first_attempt_order": expected_job_ids,
            "capture_started_utc": capture_started_utc,
            "capture_completed_utc": captured_utc,
            "returncode": returncode,
            "stderr_utf8": stderr_utf8,
            "hostname": hostname,
            "executable_sha256": executable_sha256,
            "version_stdout": version_stdout,
        },
        "parser": {
            "name": "STRICT_SACCT_PIPE_V8",
            "columns": list(_SACCT_RAW_COLUMNS),
            "delimiter": "|",
            "header": True,
            "units": "K",
            "selection_rule": (
                "exact Unicorn root plus every same-start step; unique "
                "batch/extern; Restarts binds allocation_id"
            ),
        },
        "raw_stdout_utf8": raw_stdout_utf8,
        "raw_stdout_sha256": hashlib.sha256(
            raw_stdout_utf8.encode("utf-8")
        ).hexdigest(),
        "rows": [],
    }
    payload["rows"] = _parse_raw_sacct_snapshot(
        payload,
        run_id=run_id,
        expected_attempts=expected_attempts,
        latest_allowed_utc=_utc(captured_utc),
        require_embedded_rows=False,
    )
    _parse_raw_sacct_snapshot(
        payload,
        run_id=run_id,
        expected_attempts=expected_attempts,
        latest_allowed_utc=_utc(captured_utc),
    )
    return payload


def capture_and_publish_c0_raw_sacct_snapshot(
    *,
    repository_root: Path,
    run_root: Path,
    numerical_result_record: Mapping[str, Any],
    output_path: Path,
) -> dict[str, str]:
    """Execute the sole registered sacct query and immutably publish raw bytes.

    There is deliberately no raw-input argument.  Attempt/job/restart selection
    is rebuilt from the already immutable numerical RESULT and full event chain.
    """

    root = Path(repository_root).resolve(strict=True)
    result_path, _ = _validate_record_file(
        numerical_result_record, repository_root=root
    )
    result = validate_c0_numerical_result_payload(
        _load_json(root / result_path), repository_root=root
    )
    expected_run = (
        root / result["parent_records"]["event_chain"]["tip"]["path"]
    ).resolve(strict=True).parent.parent
    if Path(run_root).resolve(strict=True) != expected_run:
        raise ValueError("RAW_SACCT capture run root differs from RESULT")
    expected_attempts: list[dict[str, Any]] = []
    by_attempt: dict[str, dict[str, Any]] = {}
    for record in result["parent_records"]["event_chain"]["records"]:
        event = _load_json(root / record["path"])
        attempt_id = str(event["attempt_identity"]["attempt_id"])
        if event["event_type"] == "STARTED":
            item = {
                "attempt_identity": dict(event["attempt_identity"]),
                "terminal": None,
            }
            expected_attempts.append(item)
            by_attempt[attempt_id] = item
        elif event["event_type"] in {"DONE", "ABANDONED", "REQUEUE"}:
            if attempt_id not in by_attempt or by_attempt[attempt_id]["terminal"] is not None:
                raise ValueError("RAW_SACCT event lifecycle cannot be normalized")
            by_attempt[attempt_id]["terminal"] = event["event_type"]
    if not expected_attempts or any(item["terminal"] is None for item in expected_attempts):
        raise ValueError("RAW_SACCT capture requires closed event lifecycles")
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    parent = destination.parent.resolve(strict=True)
    try:
        destination = parent / destination.name
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("RAW_SACCT destination escapes repository root") from error
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("RAW_SACCT destination parent is unsafe")
    partials = sorted(parent.glob(f".{destination.name}.*.partial"))
    if len(partials) > 1:
        raise ValueError("RAW_SACCT publication has ambiguous partials")

    def validate_snapshot(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError("RAW_SACCT snapshot/partial is not a real file")
        candidate = _load_json(path)
        _parse_raw_sacct_snapshot(
            candidate,
            run_id=result["run_id"],
            expected_attempts=expected_attempts,
            latest_allowed_utc=_utc(_utc_now()),
        )

    # Recover both publication crash windows without re-querying mutable
    # scheduler state.  A complete pre-link partial is promoted; a post-link
    # hard-link alias must be the very same inode.  Torn or ambiguous bytes are
    # retained and fail closed for operator inspection.
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("RAW_SACCT canonical path is unsafe")
        validate_snapshot(destination)
        if partials:
            validate_snapshot(partials[0])
            if not os.path.samefile(destination, partials[0]):
                raise ValueError("RAW_SACCT canonical/partial files diverge")
            partials[0].unlink()
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return {
            "path": destination.relative_to(root).as_posix(),
            "sha256": sha256_file(destination),
        }
    if partials:
        validate_snapshot(partials[0])
        os.link(partials[0], destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        partials[0].unlink()
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "path": destination.relative_to(root).as_posix(),
            "sha256": sha256_file(destination),
        }
    executable = Path(_SACCT_EXECUTABLE)
    if executable.is_symlink() or not executable.is_file():
        raise ValueError("registered absolute sacct executable is absent/unsafe")
    environment = {**os.environ, "LC_ALL": "C", "TZ": "UTC"}
    version = subprocess.run(
        [_SACCT_EXECUTABLE, "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    if version.returncode != 0 or version.stderr:
        raise RuntimeError("registered sacct --version failed")
    job_ids: list[str] = []
    for item in expected_attempts:
        job_id = str(item["attempt_identity"]["slurm_job_id"])
        if job_id not in job_ids:
            job_ids.append(job_id)
    command = _raw_sacct_command(job_ids)
    started = _utc_now()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    captured = _utc_now()
    payload = _build_c0_raw_sacct_snapshot(
        captured_utc=captured,
        capture_started_utc=started,
        run_id=result["run_id"],
        expected_attempts=expected_attempts,
        raw_stdout_utf8=completed.stdout,
        hostname=socket.gethostname(),
        executable_sha256=sha256_file(executable),
        version_stdout=version.stdout,
        stderr_utf8=completed.stderr,
        returncode=completed.returncode,
    )
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "path": destination.relative_to(root).as_posix(),
        "sha256": sha256_file(destination),
    }


_SACCT_ROW_KEYS = {
    "attempt_id",
    "slurm_job_id",
    "cluster",
    "job_name",
    "submit_utc",
    "start_utc",
    "end_utc",
    "elapsed_seconds",
    "max_rss_gib",
    "partition",
    "account",
    "qos",
    "nodes",
    "tasks",
    "cpus_per_task",
    "memory_gib",
    "gpu_count",
    "allocation_id",
    "node_list",
    "allocation_shape",
    "state",
    "exit_code",
}
_RESOURCE_ATTEMPT_KEYS = {
    "attempt_id",
    "phase",
    "attempt_class",
    "terminal_event_type",
    "event_records",
    "attempt_identity_record",
    "stdout_record",
    "stderr_record",
    "residual_staging_records",
    "residual_staging_directory_present",
    "source_duplicate_record",
    "cpp_telemetry_archived",
    "slurm_job_id",
    "cluster",
    "job_name",
    "partition",
    "account",
    "qos",
    "submit_utc",
    "start_utc",
    "end_utc",
    "allocation_id",
    "node_list",
    "runtime_manifest_sha256",
    "allocation_shape",
    "nodes",
    "tasks",
    "cpus_per_task",
    "memory_gib",
    "gpu_count",
    "runner_elapsed_seconds",
    "runner_elapsed_measurement",
    "slurm_elapsed_seconds",
    "conservative_elapsed_seconds",
    "cpp_max_rss_gib",
    "slurm_max_rss_gib",
    "conservative_max_rss_gib",
    "attempted_h_batched_actions",
    "attempted_actual_scalar_MVM",
    "attempted_splus_started_batched_actions",
    "attempted_splus_completed_batched_actions",
    "attempted_splus_abandoned_batched_actions",
    "basis_initializations_started",
    "basis_initializations_completed",
    "basis_initializations_abandoned",
    "seed_rows_started",
    "seed_rows_completed",
    "seed_rows_abandoned",
    "determinant_calls_started",
    "determinant_calls_completed",
    "determinant_calls_abandoned",
    "retry_ordinal_for_phase",
    "cpp_telemetry_stage",
    "cpp_telemetry_validation",
    "cpp_telemetry_record",
    "sacct_state",
    "sacct_exit_code",
}
_RESOURCE_REVIEW_KEYS = {
    "schema_version",
    "stage",
    "created_utc",
    "status",
    "run_id",
    "parent_records",
    "raw_sacct",
    "thresholds",
    "attempt_records",
    "attempted_totals",
    "committed_science",
    "operational_residuals",
    "gates",
    "interpretation",
}

_OPERATIONAL_RESIDUAL_KEYS = {
    "kind", "path", "is_directory", "file_records"
}


def _residual_tree_record(
    repository_root: Path, path: Path, *, kind: str
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    if path.is_symlink() or not path.exists():
        raise ValueError("operational residual is absent or a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("operational residual escapes repository root") from error
    files: list[dict[str, Any]] = []
    if path.is_file():
        record = _relative_hashed_file_record(root, path)
        files.append({**record, "byte_count": path.stat().st_size})
        is_directory = False
    elif path.is_dir():
        is_directory = True
        for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            if candidate.is_symlink() or not candidate.is_file():
                if candidate.is_dir() and not candidate.is_symlink():
                    continue
                raise ValueError("operational residual tree contains an unsafe entry")
            record = _relative_hashed_file_record(root, candidate)
            files.append({**record, "byte_count": candidate.stat().st_size})
    else:
        raise ValueError("operational residual has an unsupported file type")
    return {
        "kind": kind,
        "path": relative,
        "is_directory": is_directory,
        "file_records": files,
    }


def _scan_c0_operational_residuals(
    repository_root: Path,
    *,
    run_root: Path,
    seed_root: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    """Inventory every recognized uncommitted seed/checkpoint crash artifact."""

    root = Path(repository_root).resolve(strict=True)
    locations = (
        (
            Path(seed_root) / "chunks",
            "UNCOMMITTED_SEED_CHUNK_PARTIAL",
            r"\.chunk-\d{4}\.[A-Za-z0-9_.-]+\.partial",
        ),
        (
            Path(seed_root) / "markers",
            "UNCOMMITTED_SEED_MARKER_OR_LATEST_PARTIAL",
            r"(?:\.chunk-\d{4}\.json|\.LATEST)\.[A-Za-z0-9_.:-]+\.partial",
        ),
        (
            Path(checkpoint_root) / "generations",
            "UNCOMMITTED_CHECKPOINT_GENERATION_PARTIAL",
            r"\.generation-\d{4}\.[A-Za-z0-9_.-]+\.partial",
        ),
        (
            Path(checkpoint_root) / "markers",
            "UNCOMMITTED_CHECKPOINT_MARKER_OR_LATEST_PARTIAL",
            r"(?:\.generation-\d{4}\.json|\.LATEST\.json)\."
            r"[A-Za-z0-9_.-]+\.partial",
        ),
    )
    residuals: list[dict[str, Any]] = []
    for directory, kind, pattern in locations:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("operational residual scan root is absent/unsafe")
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if re.fullmatch(pattern, entry.name) is None:
                continue
            residuals.append(_residual_tree_record(root, entry, kind=kind))
    run = Path(run_root)
    if run.is_symlink() or not run.is_dir():
        raise ValueError("operational run root is absent/unsafe")
    residual_root = run / "operational_residuals"
    residual_categories = {
        "event_publication": "RUNNER_EVENT_PUBLICATION_RESIDUAL",
        "attempt_reservation": "RUNNER_ATTEMPT_RESERVATION_RESIDUAL",
        "full_seed_publication": "RUNNER_FULL_SEED_PUBLICATION_RESIDUAL",
        "checkpoint_staging": "RUNNER_CHECKPOINT_STAGING_RESIDUAL",
    }
    if residual_root.is_symlink() or not residual_root.is_dir():
        raise ValueError("runner operational-residual root is absent/unsafe")
    observed_categories = {entry.name for entry in residual_root.iterdir()}
    if observed_categories != set(residual_categories):
        raise ValueError("runner operational-residual categories changed")
    for category, kind in residual_categories.items():
        directory = residual_root / category
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("runner operational-residual category is unsafe")
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            residuals.append(_residual_tree_record(root, entry, kind=kind))
    for entry in sorted(run.iterdir(), key=lambda item: item.name):
        if re.fullmatch(
            r"\.(?:FULL_SEED_FILL_DONE|PREPARED|RESULT|RESOURCE_REVIEW|"
            r"FINAL_RESULT|RAW_SACCT)(?:\.json)?\.[A-Za-z0-9_.-]+\.partial",
            entry.name,
        ):
            residuals.append(
                _residual_tree_record(
                    root, entry, kind="UNCOMMITTED_RUN_MARKER_PARTIAL"
                )
            )
    return residuals
_COMMITTED_SCIENCE_KEYS = {
    "seed_rows",
    "determinant_calls",
    "physical_H_batched_actions",
    "physical_H_scalar_MVM",
    "null_H_batched_actions",
    "null_H_scalar_MVM",
    "total_H_batched_actions",
    "total_committed_scalar_MVM",
}
_RESOURCE_GATE_KEYS = {
    "attempt_event_cpp_sacct_bijection",
    "scheduler_identity_exact",
    "allocation_shape_exact",
    "scheduler_chronology_valid",
    "cpp_telemetry_complete",
    "attempt_artifacts_complete",
    "attempt_work_reconciled_to_commits",
    "committed_science_exact",
    "cumulative_elapsed_within_limit",
    "peak_rss_within_limit",
    "gpu_forbidden",
}


def _relative_hashed_file_record(
    repository_root: Path, path: Path
) -> dict[str, str]:
    root = Path(repository_root).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt artifact is not a real regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("attempt artifact escapes repository root") from error
    return {"path": relative.as_posix(), "sha256": sha256_file(path)}


def _scan_c0_attempt_artifacts(
    repository_root: Path,
    run_root: Path,
    attempts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Enumerate every canonical attempt directory without caller selection."""

    root = Path(repository_root).resolve(strict=True)
    run = Path(run_root).resolve(strict=True)
    try:
        run.relative_to(root)
    except ValueError as error:
        raise ValueError("formal run root escapes repository root") from error
    attempts_root = run / "attempts"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("formal attempts root is absent or unsafe")
    observed: dict[str, Path] = {}
    for entry in attempts_root.iterdir():
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", entry.name
            ) is None
            or entry.name in observed
        ):
            raise ValueError("attempt root contains partial/unknown/duplicate entry")
        observed[entry.name] = entry
    if set(observed) != set(attempts):
        raise ValueError("event chain and canonical attempt directories differ")
    result: dict[str, dict[str, Any]] = {}
    for attempt_id, directory in observed.items():
        phase = str(attempts[attempt_id]["phase"])
        identity = directory / "IDENTITY.json"
        stdout = directory / f"{phase}.stdout.log"
        stderr = directory / f"{phase}.stderr.log"
        archive = directory / f"{phase}.cpp-telemetry.json"
        seed_source = directory / "CPP_ATTEMPT_TELEMETRY.json"
        staging = directory / "cpp_staging"
        allowed_names = {
            "IDENTITY.json",
            stdout.name,
            stderr.name,
            archive.name,
            "CPP_ATTEMPT_TELEMETRY.json",
            "cpp_staging",
        }
        top_level_partials: list[Path] = []
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise ValueError("attempt directory may not contain a symlink")
            if entry.name in allowed_names:
                continue
            if (
                not entry.is_file()
                or re.fullmatch(
                    rf"\.(?:{re.escape(phase)}\.(?:stdout|stderr)\.log|"
                    rf"CPP_ATTEMPT_TELEMETRY\.json)\.[A-Za-z0-9_.-]+\.partial|"
                    rf"CPP_ATTEMPT_TELEMETRY\.json\.update\.partial",
                    entry.name,
                )
                is None
            ):
                raise ValueError("attempt directory contains an unknown artifact")
            top_level_partials.append(entry)
        if not identity.is_file() or identity.is_symlink():
            raise ValueError("attempt directory lacks its real IDENTITY")
        for candidate in (stdout, stderr, archive, seed_source, staging):
            if os.path.lexists(candidate) and candidate.is_symlink():
                raise ValueError("attempt file artifact is a symlink")
        for candidate in (stdout, stderr, archive, seed_source):
            if os.path.lexists(candidate) and not candidate.is_file():
                raise ValueError("attempt file artifact is unsafe")
        if seed_source.exists() and phase != "seed-fill":
            raise ValueError("seed-only telemetry source appears in another phase")
        residual_records: list[dict[str, Any]] = [
            {
                **_relative_hashed_file_record(root, partial),
                "byte_count": partial.stat().st_size,
            }
            for partial in sorted(top_level_partials, key=lambda item: item.name)
        ]
        telemetry_sources: list[Path] = [seed_source] if seed_source.exists() else []
        if os.path.lexists(staging):
            if not staging.is_dir():
                raise ValueError("CPP staging artifact is unsafe")
            for child in sorted(staging.iterdir(), key=lambda item: item.name):
                directory_match = re.fullmatch(
                    r"\.generation-(000[0-2])\.[A-Za-z0-9_.-]+\.partial",
                    child.name,
                )
                diagnostic_match = re.fullmatch(
                    r"\.generation-(000[0-2])\.diagnostics\.[A-Za-z0-9_.-]+"
                    r"\.partial(?:\.(?:normal|attempt|update)\.partial)?",
                    child.name,
                )
                if diagnostic_match is not None:
                    record = _relative_hashed_file_record(root, child)
                    residual_records.append(
                        {**record, "byte_count": child.stat().st_size}
                    )
                    if re.fullmatch(
                        r"\.generation-(000[0-2])\.diagnostics\."
                        r"[A-Za-z0-9_.-]+\.partial",
                        child.name,
                    ) is not None:
                        telemetry_sources.append(child)
                elif directory_match is not None:
                    if child.is_symlink() or not child.is_dir():
                        raise ValueError("state staging partial is unsafe")
                    depth = int(directory_match.group(1))
                    expected_names = {
                        f"PHYSICAL_F_{depth}.bin",
                        f"PHYSICAL_HF_{depth}.bin",
                        f"NULL_F_{depth}.bin",
                        f"NULL_HF_{depth}.bin",
                    } | ({"PHYSICAL_CARRIERS.bin"} if depth == 0 else set())
                    for payload in sorted(child.iterdir(), key=lambda item: item.name):
                        if payload.name not in expected_names:
                            raise ValueError("state staging partial has unknown payload")
                        record = _relative_hashed_file_record(root, payload)
                        residual_records.append(
                            {**record, "byte_count": payload.stat().st_size}
                        )
                else:
                    raise ValueError("CPP staging directory contains unknown entry")
        if len(telemetry_sources) > 1:
            raise ValueError("attempt has multiple candidate C++ telemetry sources")
        telemetry_source = telemetry_sources[0] if telemetry_sources else None
        source_record = (
            _relative_hashed_file_record(root, telemetry_source)
            if telemetry_source is not None
            else None
        )
        if archive.exists() and telemetry_source is not None:
            if (
                not os.path.samefile(telemetry_source, archive)
                or sha256_file(telemetry_source) != sha256_file(archive)
            ):
                raise ValueError("source/archive telemetry is not one inode/hash")
        cpp_path = archive if archive.exists() else telemetry_source
        result[attempt_id] = {
            "identity_payload": _load_json(identity),
            "identity_record": _relative_hashed_file_record(root, identity),
            "stdout_record": (
                _relative_hashed_file_record(root, stdout) if stdout.exists() else None
            ),
            "stderr_record": (
                _relative_hashed_file_record(root, stderr) if stderr.exists() else None
            ),
            "cpp_payload": _load_json(cpp_path) if cpp_path is not None else None,
            "cpp_record": (
                _relative_hashed_file_record(root, cpp_path)
                if cpp_path is not None
                else None
            ),
            "residual_staging_records": residual_records,
            "residual_staging_directory_present": staging.exists(),
            "has_source_duplicate": telemetry_source is not None,
            "source_duplicate_record": source_record,
            "cpp_telemetry_archived": archive.exists(),
        }
    return result


def _strict_finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is boolean, not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} is nonfinite or has the wrong sign")
    return result


def validate_c0_attempt_telemetry(
    payload: Mapping[str, Any],
    phase: str,
    *,
    expected_attempt_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Strictly validate one CPP attempt artifact before science commit."""

    if not isinstance(payload, Mapping):
        raise ValueError("CPP attempt telemetry must be one JSON object")
    attempt_keys = {
        "attempt_id", "allocation_id", "slurm_job_id", "node_list",
        "runtime_manifest_sha256", "allocation_shape",
    }
    if (
        not isinstance(expected_attempt_identity, Mapping)
        or set(expected_attempt_identity) != attempt_keys
        or any(
            not isinstance(value, str) or not value
            for value in expected_attempt_identity.values()
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            expected_attempt_identity["runtime_manifest_sha256"],
        )
        is None
    ):
        raise ValueError("expected attempt identity is malformed")
    stage = payload.get("stage")
    if phase == "seed-fill":
        exact = {
            "schema_version", "stage", "producer_invocation_id", "attempt_id",
            "allocation_id", "slurm_job_id", "node_list",
            "runtime_manifest_sha256", "allocation_shape",
            "status", "signal_stop_requested", "basis_initializations_started",
            "basis_initializations_completed", "chunks_started", "chunks_completed",
            "chunks_abandoned", "logical_rows_started", "logical_rows_completed",
            "logical_rows_abandoned", "determinant_calls_started",
            "determinant_calls_completed", "determinant_calls_abandoned",
            "forbidden_H_constructions", "forbidden_H_actions",
            "forbidden_energy_calculations", "forbidden_energy_logs",
            "process_max_rss_gib",
        }
        if set(payload) != exact or stage != "P9v2-D1-T-C0-v8-seed-attempt-telemetry":
            raise ValueError("seed CPP attempt telemetry schema/stage mismatch")
        if (
            payload["schema_version"] != 8
            or not isinstance(payload["producer_invocation_id"], str)
            or not payload["producer_invocation_id"]
            or not isinstance(payload["attempt_id"], str)
            or not payload["attempt_id"]
            or not isinstance(payload["status"], str)
            or payload["status"] not in {
                "AUTHORIZED_BEFORE_BASIS",
                "BASIS_INITIALIZATION_STARTED",
                "BASIS_INITIALIZATION_COMPLETED",
                "CHUNK_STARTED",
                "CHUNK_COMMITTED",
                "COMPLETE",
                "CHECKPOINT_STOP",
                "FAILED_OR_INTERRUPTED",
            }
            or type(payload["signal_stop_requested"]) is not bool
        ):
            raise ValueError("seed CPP attempt telemetry identity is invalid")
        observed_attempt = {
            key: payload[key]
            for key in (
                "attempt_id", "allocation_id", "slurm_job_id", "node_list",
                "runtime_manifest_sha256", "allocation_shape",
            )
        }
        if observed_attempt != dict(expected_attempt_identity):
            raise ValueError("seed CPP telemetry belongs to another attempt")
        counters = {
            key: payload[key]
            for key in exact
            if key.endswith(("_started", "_completed", "_abandoned"))
            or key.startswith("forbidden_")
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters.values()):
            raise ValueError("seed CPP attempt counters must be nonnegative integers")
        if (
            payload["basis_initializations_started"] not in (0, 1)
            or payload["basis_initializations_completed"]
            > payload["basis_initializations_started"]
            or
            payload["chunks_abandoned"] != payload["chunks_started"] - payload["chunks_completed"]
            or payload["logical_rows_abandoned"] != payload["logical_rows_started"] - payload["logical_rows_completed"]
            or payload["determinant_calls_abandoned"] != payload["determinant_calls_started"] - payload["determinant_calls_completed"]
            or payload["determinant_calls_started"] != 3 * payload["logical_rows_started"]
            or payload["determinant_calls_completed"] != 3 * payload["logical_rows_completed"]
            or payload["chunks_started"] > CHUNK_COUNT
            or payload["chunks_completed"] > CHUNK_COUNT
            or payload["logical_rows_started"] > FULL_DIMENSION
            or payload["logical_rows_completed"] > FULL_DIMENSION
            or any(payload[key] != 0 for key in (
                "forbidden_H_constructions", "forbidden_H_actions",
                "forbidden_energy_calculations", "forbidden_energy_logs",
            ))
        ):
            raise ValueError("seed CPP attempt ledger is arithmetically inconsistent")
        if payload["status"] == "COMPLETE" and (
            payload["basis_initializations_started"] != 1
            or payload["basis_initializations_completed"] != 1
            or payload["chunks_started"] != payload["chunks_completed"]
            or payload["logical_rows_started"] != payload["logical_rows_completed"]
        ):
            raise ValueError("complete seed telemetry does not end on a durable boundary")
        if payload["status"] == "CHECKPOINT_STOP" and (
            payload["signal_stop_requested"] is not True
            or payload["chunks_started"] != payload["chunks_completed"]
            or payload["logical_rows_started"] != payload["logical_rows_completed"]
        ):
            raise ValueError("seed checkpoint-stop is not a durable boundary")
        expected_intermediate_shapes = {
            "AUTHORIZED_BEFORE_BASIS": (0, 0, 0),
            "BASIS_INITIALIZATION_STARTED": (1, 0, 0),
            "BASIS_INITIALIZATION_COMPLETED": (1, 1, 0),
        }
        if payload["status"] in expected_intermediate_shapes:
            started, completed, chunks_started = expected_intermediate_shapes[
                payload["status"]
            ]
            if (
                payload["basis_initializations_started"] != started
                or payload["basis_initializations_completed"] != completed
                or payload["chunks_started"] != chunks_started
                or payload["chunks_completed"] != 0
                or payload["logical_rows_started"] != 0
                or payload["logical_rows_completed"] != 0
                or payload["determinant_calls_started"] != 0
                or payload["determinant_calls_completed"] != 0
            ):
                raise ValueError("seed pre-chunk crash snapshot counters changed")
        if payload["status"] == "CHUNK_STARTED" and not (
            payload["basis_initializations_started"] == 1
            and payload["basis_initializations_completed"] == 1
            and payload["chunks_started"] == payload["chunks_completed"] + 1
            and payload["logical_rows_started"]
            > payload["logical_rows_completed"]
        ):
            raise ValueError("seed in-flight chunk snapshot is inconsistent")
        if payload["status"] == "CHUNK_COMMITTED" and not (
            payload["basis_initializations_started"] == 1
            and payload["basis_initializations_completed"] == 1
            and payload["chunks_started"] == payload["chunks_completed"]
            and payload["logical_rows_started"]
            == payload["logical_rows_completed"]
            and payload["determinant_calls_started"]
            == payload["determinant_calls_completed"]
        ):
            raise ValueError("seed committed-chunk snapshot is inconsistent")
        return {
            "stage": stage,
            "attempted_h_batched_actions": 0,
            "attempted_actual_scalar_MVM": 0,
            "attempted_splus_started_batched_actions": 0,
            "attempted_splus_completed_batched_actions": 0,
            "attempted_splus_abandoned_batched_actions": 0,
            "basis_initializations_started": payload["basis_initializations_started"],
            "basis_initializations_completed": payload["basis_initializations_completed"],
            "basis_initializations_abandoned": (
                payload["basis_initializations_started"]
                - payload["basis_initializations_completed"]
            ),
            "seed_rows_started": payload["logical_rows_started"],
            "seed_rows_completed": payload["logical_rows_completed"],
            "seed_rows_abandoned": payload["logical_rows_abandoned"],
            "determinant_calls_started": payload["determinant_calls_started"],
            "determinant_calls_completed": payload["determinant_calls_completed"],
            "determinant_calls_abandoned": payload["determinant_calls_abandoned"],
            "cpp_max_rss_gib": _strict_finite_number(
                payload["process_max_rss_gib"],
                "seed CPP MaxRSS",
                nonnegative=True,
            ),
        }
    if not re.fullmatch(r"depth-[0-2]", phase):
        raise ValueError("formal attempt phase is not seed-fill or depth 0/1/2")
    if stage == "P9v2-D1-T-C0-v8-attempt-only":
        exact = {
            "schema_version", "stage", "status", "signal_stop_requested",
            "attempt_identity",
            "physical_started_batched_actions", "physical_completed_batched_actions",
            "null_started_batched_actions", "null_completed_batched_actions",
            "started_batched_actions", "completed_batched_actions",
            "abandoned_batched_actions", "attempted_actual_scalar_MVM",
            "splus_started_batched_actions", "splus_completed_batched_actions",
            "splus_abandoned_batched_actions", "process_max_rss_gib",
        }
        if set(payload) != exact:
            raise ValueError("block attempt-only telemetry schema mismatch")
        if (
            payload["schema_version"] != 8
            or payload["status"] not in {
                "FAILED_AFTER_AUTHORIZATION_BEFORE_COMMIT",
                "CHECKPOINT_STOP_BEFORE_PHYSICAL_ACTION",
                "CHECKPOINT_STOP_BEFORE_SPLUS_ACTION",
                "CHECKPOINT_STOP_AFTER_SPLUS_ACTION",
                "FAILED_DURING_SPLUS_ACTION",
                "FAILED_DURING_PHYSICAL_ACTION",
                "CHECKPOINT_STOP_AFTER_PHYSICAL_BEFORE_NULL_ACTION",
                "FAILED_DURING_NULL_ACTION",
                "FAILED_AFTER_PAIRED_ACTIONS_BEFORE_COMMIT",
            }
            or type(payload["signal_stop_requested"]) is not bool
        ):
            raise ValueError("block attempt-only telemetry identity is invalid")
        observed_attempt = payload["attempt_identity"]
        if (
            not isinstance(observed_attempt, Mapping)
            or set(observed_attempt) != attempt_keys
            or dict(observed_attempt) != dict(expected_attempt_identity)
        ):
            raise ValueError("block attempt-only telemetry belongs to another attempt")
        for key in exact - {
            "stage", "status", "signal_stop_requested", "attempt_identity",
            "process_max_rss_gib",
        }:
            if isinstance(payload[key], bool) or not isinstance(payload[key], int) or payload[key] < 0:
                raise ValueError("block attempt-only counters must be nonnegative integers")
        if (
            payload["started_batched_actions"]
            != payload["physical_started_batched_actions"] + payload["null_started_batched_actions"]
            or payload["completed_batched_actions"]
            != payload["physical_completed_batched_actions"] + payload["null_completed_batched_actions"]
            or payload["abandoned_batched_actions"]
            != payload["started_batched_actions"] - payload["completed_batched_actions"]
            or payload["attempted_actual_scalar_MVM"] != 3 * payload["started_batched_actions"]
            or payload["splus_abandoned_batched_actions"]
            != payload["splus_started_batched_actions"]
            - payload["splus_completed_batched_actions"]
            or payload["physical_started_batched_actions"] > 1
            or payload["physical_completed_batched_actions"]
            > payload["physical_started_batched_actions"]
            or payload["null_started_batched_actions"] > 1
            or payload["null_completed_batched_actions"]
            > payload["null_started_batched_actions"]
            or (
                payload["null_started_batched_actions"]
                and payload["physical_completed_batched_actions"] != 1
            )
            or payload["splus_started_batched_actions"]
            > (2 if phase in {"depth-0", "depth-1"} else 4)
        ):
            raise ValueError("block attempt-only work ledger is inconsistent")
        checkpoint_status = payload["status"].startswith("CHECKPOINT_STOP_")
        if checkpoint_status and payload["signal_stop_requested"] is not True:
            raise ValueError("block checkpoint-stop lacks its signal flag")
        status = payload["status"]
        h_tuple = (
            payload["physical_started_batched_actions"],
            payload["physical_completed_batched_actions"],
            payload["null_started_batched_actions"],
            payload["null_completed_batched_actions"],
        )
        splus_started = payload["splus_started_batched_actions"]
        splus_completed = payload["splus_completed_batched_actions"]
        paired_or_unstarted = h_tuple in {(0, 0, 0, 0), (1, 1, 1, 1)}
        if (
            (status == "FAILED_AFTER_AUTHORIZATION_BEFORE_COMMIT"
             and h_tuple != (0, 0, 0, 0))
            or (status == "CHECKPOINT_STOP_BEFORE_PHYSICAL_ACTION"
             and h_tuple != (0, 0, 0, 0))
            or (status == "FAILED_DURING_PHYSICAL_ACTION"
                and h_tuple != (1, 0, 0, 0))
            or (status == "CHECKPOINT_STOP_AFTER_PHYSICAL_BEFORE_NULL_ACTION"
                and h_tuple != (1, 1, 0, 0))
            or (status == "FAILED_DURING_NULL_ACTION"
                and h_tuple != (1, 1, 1, 0))
            or (status == "FAILED_AFTER_PAIRED_ACTIONS_BEFORE_COMMIT"
                and h_tuple != (1, 1, 1, 1))
            or (status in {
                    "CHECKPOINT_STOP_BEFORE_SPLUS_ACTION",
                    "CHECKPOINT_STOP_AFTER_SPLUS_ACTION",
                    "FAILED_DURING_SPLUS_ACTION",
                } and not paired_or_unstarted)
            or (status == "FAILED_DURING_SPLUS_ACTION"
                and splus_started - splus_completed != 1)
            or (status != "FAILED_DURING_SPLUS_ACTION"
                and splus_started != splus_completed)
            or (h_tuple != (0, 0, 0, 0) and splus_completed < 2)
            or (phase in {"depth-0", "depth-1"} and splus_started > 2)
            or (phase == "depth-2" and splus_started > 4)
        ):
            raise ValueError("block status/work tuple is not a reachable v8 state")
        rss: float | None = _strict_finite_number(
            payload["process_max_rss_gib"],
            "block attempt-only MaxRSS",
            nonnegative=True,
        )
    elif stage == "P9v2-D1-T-C0-production-v8":
        from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
            validate_c0_normal_attempt_telemetry,
        )
        return validate_c0_normal_attempt_telemetry(
            payload, expected_attempt_identity=expected_attempt_identity
        )
    else:
        raise ValueError("unknown block CPP telemetry stage")
    return {
        "stage": stage,
        "attempted_h_batched_actions": payload["started_batched_actions"] if stage.endswith("attempt-only") else payload["attempted_started_batched_actions"],
        "attempted_actual_scalar_MVM": payload["attempted_actual_scalar_MVM"],
        "attempted_splus_started_batched_actions": payload[
            "splus_started_batched_actions"
        ],
        "attempted_splus_completed_batched_actions": payload[
            "splus_completed_batched_actions"
        ],
        "attempted_splus_abandoned_batched_actions": payload[
            "splus_abandoned_batched_actions"
        ],
        "basis_initializations_started": 0,
        "basis_initializations_completed": 0,
        "basis_initializations_abandoned": 0,
        "seed_rows_started": 0,
        "seed_rows_completed": 0,
        "seed_rows_abandoned": 0,
        "determinant_calls_started": 0,
        "determinant_calls_completed": 0,
        "determinant_calls_abandoned": 0,
        "cpp_max_rss_gib": rss,
    }


def aggregate_and_validate_c0_resource_review(
    *,
    repository_root: Path,
    run_root: Path,
    created_utc: str,
    numerical_result_record: Mapping[str, Any],
    raw_sacct_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the strict post-depth-two manual C0 resource review payload.

    The cumulative elapsed gate is ``max(sum(runner), sum(sacct))`` because
    those are two measurements of the same allocations.  Runner time comes
    only from final-event monotonic timers, never cross-allocation UTC gaps;
    the scheduler sum counts each allocation/restart once.  Peak RSS is the
    maximum over every available CPP and sacct measurement.  Retry/abandoned
    work is retained in the attempted totals.
    """

    created = _utc(created_utc)
    repository_root = Path(repository_root)
    result_path, result_sha = _validate_record_file(
        numerical_result_record, repository_root=repository_root
    )
    result = validate_c0_numerical_result_payload(
        _load_json(repository_root / result_path),
        repository_root=repository_root,
    )
    run_id = result["run_id"]
    result_parents = result["parent_records"]
    event_chain_records = result_parents["event_chain"]["records"]
    expected_run_root = (
        repository_root / result_parents["event_chain"]["tip"]["path"]
    ).resolve(strict=True).parent.parent
    if Path(run_root).resolve(strict=True) != expected_run_root:
        raise ValueError("resource review run root differs from numerical RESULT")
    prepared_sha256 = result_parents["prepared"]["sha256"]
    full_seed_fill_done_sha256 = result_parents["full_seed_fill_done"]["sha256"]
    depth2_marker_sha256 = result_parents["generation_markers"][2]["sha256"]
    committed_science = result["committed_science"]
    parent_records = {
        "result": {"path": result_path, "sha256": result_sha},
        "prepared_sha256": _full_sha(prepared_sha256),
        "full_seed_fill_done_sha256": _full_sha(full_seed_fill_done_sha256),
        "depth2_marker_sha256": _full_sha(depth2_marker_sha256),
        "event_chain_tip_sha256": "",
        "event_chain_record_count": 0,
    }
    raw_path, raw_sha = _validate_record_file(
        raw_sacct_record, repository_root=repository_root
    )
    raw_sacct = _load_json(repository_root / raw_path)
    if not isinstance(raw_sacct, Mapping):
        raise ValueError("raw sacct snapshot is not a JSON object")
    if set(committed_science) != _COMMITTED_SCIENCE_KEYS:
        raise ValueError("committed-science ledger schema mismatch")
    expected_committed = {
        "seed_rows": FULL_DIMENSION,
        "determinant_calls": COMMITTED_DETERMINANT_CALLS,
        "physical_H_batched_actions": 3,
        "physical_H_scalar_MVM": 9,
        "null_H_batched_actions": 3,
        "null_H_scalar_MVM": 9,
        "total_H_batched_actions": 6,
        "total_committed_scalar_MVM": 18,
    }
    committed_gate = dict(committed_science) == expected_committed and all(
        type(value) is int for value in committed_science.values()
    )
    if not event_chain_records:
        raise ValueError("resource review requires a nonempty immutable event chain")
    event_chain: list[dict[str, Any]] = []
    event_shas: list[str] = []
    previous_event_sha: str | None = None
    previous_event_time: datetime | None = None
    previous_event_type: str | None = None
    open_attempt_id: str | None = None
    stable_science_baseline: dict[str, str] | None = None
    observed_full_seed_sha: str | None = None
    allowed_after = {
        None: {"STARTED"},
        "STARTED": {"COMPLETED", "ABANDONED", "TERMINATED"},
        "COMPLETED": {"DONE", "ABANDONED"},
        "TERMINATED": {"REQUEUE"},
        "REQUEUE": {"STARTED"},
        "ABANDONED": {"STARTED"},
        "DONE": {"STARTED"},
    }
    for index, record in enumerate(event_chain_records):
        event_path, event_sha = _validate_record_file(
            record, repository_root=repository_root
        )
        event = _load_json(repository_root / event_path)
        if (
            set(event)
            != {
                "schema_version", "stage", "event_index", "event_type",
                "created_utc", "phase", "depth", "science_identity",
                "attempt_identity", "parent_event_sha256", "details",
            }
            or event.get("schema_version") != 8
            or event.get("stage") != "P9v2-D1-T-C0-v8-event"
            or event.get("event_index") != index
            or event.get("parent_event_sha256") != previous_event_sha
            or event.get("event_type") not in allowed_after[previous_event_type]
        ):
            raise ValueError("resource event record schema/hash chain mismatch")
        event_time = _utc(event["created_utc"])
        if (
            event_time > created
            or (previous_event_time is not None and event_time < previous_event_time)
        ):
            raise ValueError("resource event chronology moves backwards")
        science = event["science_identity"]
        attempt = event["attempt_identity"]
        if not isinstance(science, Mapping) or not isinstance(attempt, Mapping):
            raise ValueError("resource event identities are not objects")
        stable_science = {
            key: value
            for key, value in science.items()
            if key != "full_seed_fill_done_sha256"
        }
        if stable_science_baseline is None:
            stable_science_baseline = stable_science
        elif stable_science != stable_science_baseline:
            raise ValueError("stable science identity changes across events")
        observed_full = science.get("full_seed_fill_done_sha256")
        if observed_full == "NOT_CREATED_YET_BY_DESIGN":
            if observed_full_seed_sha is not None:
                raise ValueError("event science identity regresses before FULL")
        else:
            _full_sha(observed_full)
            if observed_full != full_seed_fill_done_sha256:
                raise ValueError("event science identity binds another FULL marker")
            observed_full_seed_sha = str(observed_full)
        event_attempt_id = attempt.get("attempt_id")
        if event["event_type"] == "STARTED":
            if open_attempt_id is not None or not isinstance(event_attempt_id, str):
                raise ValueError("resource event starts an overlapping lifecycle")
            open_attempt_id = event_attempt_id
        elif event_attempt_id != open_attempt_id:
            raise ValueError("resource event closes another lifecycle")
        if event["event_type"] in {"DONE", "ABANDONED", "REQUEUE"}:
            open_attempt_id = None
        event_chain.append(event)
        event_shas.append(event_sha)
        previous_event_sha = event_sha
        previous_event_time = event_time
        previous_event_type = str(event["event_type"])
    if open_attempt_id is not None:
        raise ValueError("resource review cannot consume an open event lifecycle")
    parent_records["event_chain_tip_sha256"] = event_shas[-1]
    parent_records["event_chain_record_count"] = len(event_shas)
    attempts: dict[str, dict[str, Any]] = {}
    recovery_disclosures: dict[str, dict[str, Any]] = {}
    event_times: dict[str, list[datetime]] = {}
    event_records_by_attempt: dict[str, list[dict[str, str]]] = {}
    event_details_by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for event, event_sha in zip(event_chain, event_shas, strict=True):
        if not isinstance(event, Mapping):
            raise ValueError("event chain entry is not an object")
        attempt = event.get("attempt_identity")
        attempt_identity_keys = {
            "attempt_id", "slurm_job_id", "node_list", "allocation_id",
            "runtime_manifest_sha256", "allocation_shape",
        }
        science_identity = event.get("science_identity")
        science_identity_keys = {
            "config_sha256", "source_manifest_sha256", "input_sha256",
            "seed_fill_binary_sha256", "block_krylov_binary_sha256",
            "xdiag_commit", "run_id", "prepared_sha256",
            "full_seed_fill_done_sha256",
        }
        if (
            not isinstance(attempt, Mapping)
            or set(attempt) != attempt_identity_keys
            or any(not isinstance(value, str) or not value for value in attempt.values())
            or not isinstance(science_identity, Mapping)
            or set(science_identity) != science_identity_keys
            or any(
                not isinstance(value, str) or not value
                for value in science_identity.values()
            )
            or science_identity.get("run_id") != run_id
            or science_identity.get("prepared_sha256") != prepared_sha256
        ):
            raise ValueError("event attempt identity is missing")
        attempt_id = attempt["attempt_id"]
        event_records_by_attempt.setdefault(attempt_id, []).append(
            {
                "path": str(event_chain_records[event["event_index"]]["path"]),
                "sha256": event_sha,
            }
        )
        phase = event.get("phase")
        event_type = event.get("event_type")
        when = _utc(event.get("created_utc"))
        event_times.setdefault(attempt_id, []).append(when)
        details = event.get("details")
        if not isinstance(details, Mapping):
            raise ValueError("event details are missing")
        event_details_by_attempt.setdefault(attempt_id, []).append(details)
        disclosed = details.get("recovery_attempt_identity")
        if disclosed is not None:
            if (
                not isinstance(disclosed, Mapping)
                or set(disclosed) != {
                    "attempt_id", "slurm_job_id", "node_list", "allocation_id",
                    "runtime_manifest_sha256", "allocation_shape",
                }
                or any(not isinstance(item, str) or not item for item in disclosed.values())
            ):
                raise ValueError("recovery allocation disclosure is malformed")
            disclosed_id = disclosed["attempt_id"]
            prior = recovery_disclosures.get(disclosed_id)
            if prior is not None and prior != dict(disclosed):
                raise ValueError("one recovery attempt identity changes across disclosures")
            recovery_disclosures[disclosed_id] = dict(disclosed)
        if event_type == "STARTED":
            if attempt_id in attempts:
                raise ValueError("an attempt has more than one STARTED event")
            attempts[attempt_id] = {
                "phase": phase,
                "attempt_identity": dict(attempt),
                "science_identity": dict(event["science_identity"]),
                "terminal": None,
                "terminal_details": None,
            }
        elif attempt_id not in attempts:
            raise ValueError("terminal event precedes its attempt STARTED event")
        elif (
            attempts[attempt_id]["phase"] != phase
            or attempts[attempt_id]["attempt_identity"] != dict(attempt)
            or attempts[attempt_id]["science_identity"]
            != dict(event["science_identity"])
        ):
            raise ValueError("attempt/science identity changes within a lifecycle")
        elif event_type in {"DONE", "ABANDONED", "REQUEUE"}:
            if attempts[attempt_id]["terminal"] is not None:
                raise ValueError("an attempt has multiple closing terminal events")
            attempts[attempt_id]["terminal"] = event_type
            attempts[attempt_id]["terminal_details"] = dict(details)
    if not attempts or any(item["terminal"] is None for item in attempts.values()):
        raise ValueError("every resource-reviewed attempt must have a closed lifecycle")
    scanned_attempts = _scan_c0_attempt_artifacts(
        repository_root, run_root, attempts
    )
    identity_payloads = {
        attempt_id: artifact["identity_payload"]
        for attempt_id, artifact in scanned_attempts.items()
    }
    normalized_identity_records = {
        attempt_id: artifact["identity_record"]
        for attempt_id, artifact in scanned_attempts.items()
    }
    identity_sha256s = {
        attempt_id: record["sha256"]
        for attempt_id, record in normalized_identity_records.items()
    }
    cpp_attempt_telemetry = {
        attempt_id: artifact["cpp_payload"]
        for attempt_id, artifact in scanned_attempts.items()
        if artifact["cpp_payload"] is not None
    }
    normalized_cpp_records = {
        attempt_id: artifact["cpp_record"]
        for attempt_id, artifact in scanned_attempts.items()
        if artifact["cpp_record"] is not None
    }
    cpp_attempt_telemetry_sha256 = {
        attempt_id: record["sha256"]
        for attempt_id, record in normalized_cpp_records.items()
    }
    for disclosed_id, disclosed in recovery_disclosures.items():
        if (
            disclosed_id not in attempts
            or attempts[disclosed_id]["attempt_identity"] != disclosed
        ):
            raise ValueError(
                "recovery allocation disclosure lacks its own bound lifecycle"
            )
    if set(identity_payloads) != set(attempts) or set(identity_sha256s) != set(attempts):
        raise ValueError("event/attempt-identity record sets are not bijective")
    for attempt_id, item in attempts.items():
        identity = identity_payloads[attempt_id]
        if (
            set(identity)
            != {
                "schema_version", "stage", "created_utc", "phase",
                "science_identity", "attempt_identity",
            }
            or identity.get("schema_version") != 8
            or identity.get("stage") != "P9v2-D1-T-C0-v8-attempt-identity"
            or identity.get("phase") != item["phase"]
            or identity.get("science_identity") != item["science_identity"]
            or identity.get("attempt_identity") != item["attempt_identity"]
            or _utc(identity["created_utc"]) > event_times[attempt_id][0]
        ):
            raise ValueError("attempt IDENTITY does not bind its STARTED event")
    cpp_required_attempts: set[str] = set()
    attempt_classes: dict[str, str] = {}
    attempt_artifacts_complete = True
    for attempt_id, item in attempts.items():
        phase = item["phase"]
        terminal_details = item["terminal_details"]
        if (
            not isinstance(terminal_details, Mapping)
            or type(terminal_details.get("child_process_started")) is not bool
        ):
            raise ValueError("terminal event lacks a strict child-process disclosure")
        child_started = terminal_details["child_process_started"]
        if phase in {"seed-fill", "depth-0", "depth-1", "depth-2"}:
            if child_started:
                cpp_required_attempts.add(attempt_id)
                attempt_classes[attempt_id] = "SCIENCE_CHILD"
            else:
                attempt_classes[attempt_id] = "SCIENCE_PRELAUNCH_ZERO"
        elif phase == "commit-recovery-only":
            if child_started:
                raise ValueError("recovery-only lifecycle claims a science child")
            attempt_classes[attempt_id] = "RECOVERY_ONLY_ZERO"
        else:
            raise ValueError("resource event phase is outside the v8 lifecycle")
        artifact = scanned_attempts[attempt_id]
        has_logs = (
            artifact["stdout_record"] is not None
            and artifact["stderr_record"] is not None
        )
        has_cpp = artifact["cpp_record"] is not None
        cpp_archived = artifact["cpp_telemetry_archived"]
        has_residual = bool(artifact["residual_staging_records"])
        has_staging = artifact["residual_staging_directory_present"]
        has_source = bool(artifact["has_source_duplicate"])
        if child_started:
            attempt_artifacts_complete &= has_logs and has_cpp and cpp_archived
            if item["terminal"] == "DONE":
                attempt_artifacts_complete &= (
                    not has_staging and not has_residual and not has_source
                )
        elif has_logs or has_cpp or has_staging or has_residual or has_source:
            raise ValueError("zero-child lifecycle contains child artifacts")
    if not set(cpp_attempt_telemetry).issubset(cpp_required_attempts):
        raise ValueError("zero-child lifecycle has a CPP telemetry artifact")
    sacct_rows = _parse_raw_sacct_snapshot(
        raw_sacct,
        run_id=run_id,
        expected_attempts=list(attempts.values()),
        latest_allowed_utc=created,
    )
    captured = _utc(raw_sacct["captured_utc"])
    sacct_by_attempt: dict[str, Mapping[str, Any]] = {}
    for row in sacct_rows:
        if not isinstance(row, Mapping) or set(row) != _SACCT_ROW_KEYS:
            raise ValueError("normalized sacct row has missing or extra fields")
        attempt_id = row["attempt_id"]
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in sacct_by_attempt:
            raise ValueError("sacct attempt IDs are empty or duplicated")
        sacct_by_attempt[attempt_id] = row
    if set(sacct_by_attempt) != set(attempts):
        raise ValueError("event/sacct attempt sets are not bijective")
    if list(sacct_by_attempt) != list(attempts):
        raise ValueError("raw sacct rows are not in immutable attempt order")
    if any(_utc(row["end_utc"]) > captured for row in sacct_rows):
        raise ValueError("raw sacct was captured before an allocation ended")

    phase_counts: dict[str, int] = {}
    attempt_records: list[dict[str, Any]] = []
    scheduler_identity_exact = True
    allocation_shape_exact = True
    chronology_valid = True
    telemetry_complete = True
    for attempt_id, event_record in attempts.items():
        phase = str(event_record["phase"])
        identity = event_record["attempt_identity"]
        row = sacct_by_attempt[attempt_id]
        submit = _utc(row["submit_utc"])
        start = _utc(row["start_utc"])
        end = _utc(row["end_utc"])
        chronology_valid &= submit <= start <= end <= created
        slurm_elapsed = _strict_finite_number(
            row["elapsed_seconds"], "sacct elapsed", nonnegative=True
        )
        chronology_valid &= abs((end - start).total_seconds() - slurm_elapsed) <= 1.0
        terminal_details = event_record["terminal_details"]
        measured_runner_elapsed = terminal_details.get(
            "runner_monotonic_elapsed_seconds_this_attempt"
        )
        if measured_runner_elapsed is None:
            runner_elapsed = 0.0
            runner_elapsed_measurement = (
                "UNAVAILABLE_HISTORICAL_CRASH_RAW_SACCT_COVERS"
            )
        else:
            runner_elapsed = _strict_finite_number(
                measured_runner_elapsed,
                "runner monotonic allocation elapsed",
                nonnegative=True,
            )
            runner_elapsed_measurement = "FINAL_EVENT_MONOTONIC"
        if attempt_id in cpp_attempt_telemetry:
            cpp = validate_c0_attempt_telemetry(
                cpp_attempt_telemetry[attempt_id],
                phase,
                expected_attempt_identity=identity,
            )
            cpp_sha: str | None = _full_sha(cpp_attempt_telemetry_sha256[attempt_id])
            if cpp_attempt_telemetry[attempt_id].get("attempt_id", attempt_id) != attempt_id:
                telemetry_complete = False
            cpp_validation = "VALID_EXACT_SCHEMA"
        else:
            launched_without_cpp = attempt_id in cpp_required_attempts
            cpp = {
                "stage": (
                    "MISSING_CPP_TELEMETRY_RESOURCE_NO_GO"
                    if launched_without_cpp
                    else
                    "PRELAUNCH_ZERO_NO_CPP_BY_DESIGN"
                    if attempt_classes[attempt_id] == "SCIENCE_PRELAUNCH_ZERO"
                    else "RECOVERY_ONLY_NO_CPP_BY_DESIGN"
                ),
                "attempted_h_batched_actions": 0,
                "attempted_actual_scalar_MVM": 0,
                "attempted_splus_started_batched_actions": 0,
                "attempted_splus_completed_batched_actions": 0,
                "attempted_splus_abandoned_batched_actions": 0,
                "basis_initializations_started": 0,
                "basis_initializations_completed": 0,
                "basis_initializations_abandoned": 0,
                "seed_rows_started": 0,
                "seed_rows_completed": 0,
                "seed_rows_abandoned": 0,
                "determinant_calls_started": 0,
                "determinant_calls_completed": 0,
                "determinant_calls_abandoned": 0,
                "cpp_max_rss_gib": None,
            }
            cpp_sha = None
            cpp_validation = (
                "MISSING_REQUIRED_CPP_TELEMETRY"
                if launched_without_cpp
                else "NO_CPP_BY_DESIGN"
            )
            if launched_without_cpp:
                telemetry_complete = False
        scheduler_identity_exact &= (
            row["slurm_job_id"] == identity.get("slurm_job_id")
            and row["allocation_id"] == identity.get("allocation_id")
            and row["node_list"] == identity.get("node_list")
            and row["cluster"] == "unicorn"
            and row["job_name"] == "nqs-d1-c0-v8"
            and row["partition"] == "kim"
            and row["account"] == "kim"
            and row["qos"] == "normal"
        )
        allocation_shape_exact &= (
            row["allocation_shape"] == identity.get("allocation_shape")
            and row["nodes"] == 1
            and row["tasks"] == 1
            and row["cpus_per_task"] == 40
            and row["memory_gib"] == 128
            and row["gpu_count"] == 0
        )
        slurm_rss = _strict_finite_number(
            row["max_rss_gib"], "sacct MaxRSS", nonnegative=True
        )
        cpp_rss = cpp["cpp_max_rss_gib"]
        conservative_rss = max(
            slurm_rss, 0.0 if cpp_rss is None else float(cpp_rss)
        )
        retry = phase_counts.get(phase, 0)
        phase_counts[phase] = retry + 1
        attempt_records.append(
            {
                "attempt_id": attempt_id,
                "phase": phase,
                "attempt_class": attempt_classes[attempt_id],
                "terminal_event_type": event_record["terminal"],
                "event_records": event_records_by_attempt[attempt_id],
                "attempt_identity_record": normalized_identity_records[attempt_id],
                "stdout_record": scanned_attempts[attempt_id]["stdout_record"],
                "stderr_record": scanned_attempts[attempt_id]["stderr_record"],
                "residual_staging_records": scanned_attempts[attempt_id][
                    "residual_staging_records"
                ],
                "residual_staging_directory_present": scanned_attempts[
                    attempt_id
                ]["residual_staging_directory_present"],
                "source_duplicate_record": scanned_attempts[attempt_id][
                    "source_duplicate_record"
                ],
                "cpp_telemetry_archived": scanned_attempts[attempt_id][
                    "cpp_telemetry_archived"
                ],
                "slurm_job_id": row["slurm_job_id"],
                "cluster": row["cluster"],
                "job_name": row["job_name"],
                "partition": row["partition"],
                "account": row["account"],
                "qos": row["qos"],
                "submit_utc": row["submit_utc"],
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "allocation_id": row["allocation_id"],
                "node_list": row["node_list"],
                "runtime_manifest_sha256": identity["runtime_manifest_sha256"],
                "allocation_shape": row["allocation_shape"],
                "nodes": row["nodes"],
                "tasks": row["tasks"],
                "cpus_per_task": row["cpus_per_task"],
                "memory_gib": row["memory_gib"],
                "gpu_count": row["gpu_count"],
                "runner_elapsed_seconds": runner_elapsed,
                "runner_elapsed_measurement": runner_elapsed_measurement,
                "slurm_elapsed_seconds": slurm_elapsed,
                "conservative_elapsed_seconds": max(runner_elapsed, slurm_elapsed),
                "cpp_max_rss_gib": cpp_rss,
                "slurm_max_rss_gib": slurm_rss,
                "conservative_max_rss_gib": conservative_rss,
                "attempted_h_batched_actions": cpp["attempted_h_batched_actions"],
                "attempted_actual_scalar_MVM": cpp["attempted_actual_scalar_MVM"],
                "attempted_splus_started_batched_actions": cpp[
                    "attempted_splus_started_batched_actions"
                ],
                "attempted_splus_completed_batched_actions": cpp[
                    "attempted_splus_completed_batched_actions"
                ],
                "attempted_splus_abandoned_batched_actions": cpp[
                    "attempted_splus_abandoned_batched_actions"
                ],
                "basis_initializations_started": cpp[
                    "basis_initializations_started"
                ],
                "basis_initializations_completed": cpp[
                    "basis_initializations_completed"
                ],
                "basis_initializations_abandoned": cpp[
                    "basis_initializations_abandoned"
                ],
                "seed_rows_started": cpp["seed_rows_started"],
                "seed_rows_completed": cpp["seed_rows_completed"],
                "seed_rows_abandoned": cpp["seed_rows_abandoned"],
                "determinant_calls_started": cpp["determinant_calls_started"],
                "determinant_calls_completed": cpp["determinant_calls_completed"],
                "determinant_calls_abandoned": cpp["determinant_calls_abandoned"],
                "retry_ordinal_for_phase": retry,
                "cpp_telemetry_stage": cpp["stage"],
                "cpp_telemetry_validation": cpp_validation,
                "cpp_telemetry_record": (
                    None
                    if cpp_sha is None
                    else normalized_cpp_records[attempt_id]
                ),
                "sacct_state": row["state"],
                "sacct_exit_code": row["exit_code"],
            }
        )
    committed_seed_rows_by_attempt: dict[str, int] = {}
    committed_seed_invocations_by_attempt: dict[str, set[str]] = {}
    committed_seed_identities: dict[str, dict[str, str]] = {}
    full_parent_path = repository_root / result_parents["full_seed_fill_done"]["path"]
    full_parent = _load_json(full_parent_path)
    seed_chain_root = Path(full_parent["seed_chain_root"]).resolve(strict=True)
    for ordinal in range(CHUNK_COUNT):
        metadata = _load_json(
            seed_chain_root / "chunks" / f"chunk-{ordinal:04d}" / "METADATA.json"
        )
        seed_identity = metadata["attempt_identity"]
        if not isinstance(seed_identity, Mapping):
            raise ValueError("committed seed metadata lacks attempt identity")
        attempt_id = seed_identity["attempt_id"]
        if (
            attempt_id not in attempts
            or attempts[attempt_id]["phase"] != "seed-fill"
            or attempt_classes[attempt_id] != "SCIENCE_CHILD"
            or dict(seed_identity) != attempts[attempt_id]["attempt_identity"]
        ):
            raise ValueError(
                "committed seed chunk is not bound to its launched seed attempt"
            )
        previous_seed_identity = committed_seed_identities.get(attempt_id)
        if previous_seed_identity is not None and previous_seed_identity != dict(
            seed_identity
        ):
            raise ValueError("one committed seed attempt changes allocation identity")
        committed_seed_identities[attempt_id] = dict(seed_identity)
        committed_seed_rows_by_attempt[attempt_id] = (
            committed_seed_rows_by_attempt.get(attempt_id, 0)
            + int(metadata["count"])
        )
        committed_seed_invocations_by_attempt.setdefault(attempt_id, set()).add(
            str(metadata["producer_invocation_id"])
        )
    committed_depth_by_attempt: dict[str, int] = {}
    checkpoint_roots: set[Path] = set()
    for depth, marker_record in enumerate(
        result_parents["generation_markers"]
    ):
        marker_path = (repository_root / marker_record["path"]).resolve(strict=True)
        checkpoint_roots.add(marker_path.parent.parent)
        marker = _load_json(marker_path)
        marker_identity = marker["attempt_identity"]
        if not isinstance(marker_identity, Mapping):
            raise ValueError("committed depth marker lacks attempt identity")
        attempt_id = marker_identity["attempt_id"]
        if (
            attempt_id not in attempts
            or attempts[attempt_id]["phase"] != f"depth-{depth}"
            or attempt_classes[attempt_id] != "SCIENCE_CHILD"
            or dict(marker_identity) != attempts[attempt_id]["attempt_identity"]
        ):
            raise ValueError(
                "committed depth marker is not bound to its launched depth attempt"
            )
        if attempt_id in committed_depth_by_attempt:
            raise ValueError("one attempt ID claims multiple committed depths")
        committed_depth_by_attempt[attempt_id] = depth
    if set(committed_seed_rows_by_attempt).intersection(committed_depth_by_attempt):
        raise ValueError("seed and depth commits reuse an attempt identity")
    if len(checkpoint_roots) != 1:
        raise ValueError("resource review generation markers have multiple roots")
    operational_residuals = _scan_c0_operational_residuals(
        repository_root,
        run_root=Path(run_root),
        seed_root=seed_chain_root,
        checkpoint_root=checkpoint_roots.pop(),
    )
    attempt_artifacts_complete &= not operational_residuals
    work_reconciled = True
    for attempt in attempt_records:
        attempt_id = attempt["attempt_id"]
        phase = attempt["phase"]
        committed_rows_for_attempt = committed_seed_rows_by_attempt.get(
            attempt_id, 0
        )
        if phase == "seed-fill":
            work_reconciled &= (
                attempt["seed_rows_completed"] == committed_rows_for_attempt
                and attempt["determinant_calls_completed"]
                == 3 * committed_rows_for_attempt
            )
            if attempt_id in cpp_attempt_telemetry:
                committed_invocations = committed_seed_invocations_by_attempt.get(
                    attempt_id, set()
                )
                work_reconciled &= (
                    not committed_invocations
                    or committed_invocations
                    == {
                        str(
                            cpp_attempt_telemetry[attempt_id][
                                "producer_invocation_id"
                            ]
                        )
                    }
                )
        elif phase in {"depth-0", "depth-1", "depth-2"}:
            expected_depth = int(phase[-1])
            is_committed = committed_depth_by_attempt.get(attempt_id) == expected_depth
            if is_committed:
                work_reconciled &= (
                    attempt["terminal_event_type"] == "DONE"
                    and attempt["cpp_telemetry_stage"]
                    == "P9v2-D1-T-C0-production-v8"
                    and attempt["attempted_h_batched_actions"] == 2
                )
            elif attempt["terminal_event_type"] == "DONE":
                work_reconciled = False
        elif phase == "commit-recovery-only":
            work_reconciled &= all(
                attempt[key] == 0
                for key in (
                    "attempted_h_batched_actions",
                    "attempted_actual_scalar_MVM",
                    "seed_rows_started",
                    "determinant_calls_started",
                )
            )
    work_reconciled &= (
        set(committed_seed_rows_by_attempt).issubset(
            {item["attempt_id"] for item in attempt_records}
        )
        and set(committed_depth_by_attempt).issubset(
            {item["attempt_id"] for item in attempt_records}
        )
        and sum(committed_seed_rows_by_attempt.values()) == FULL_DIMENSION
        and set(committed_depth_by_attempt.values()) == {0, 1, 2}
    )
    runner_total = float(sum(item["runner_elapsed_seconds"] for item in attempt_records))
    slurm_by_allocation: dict[str, float] = {}
    slurm_allocation_fingerprints: dict[str, tuple[Any, ...]] = {}
    for item in attempt_records:
        allocation = item["allocation_id"]
        elapsed = float(item["slurm_elapsed_seconds"])
        fingerprint = tuple(
            item[key]
            for key in (
                "cluster", "job_name", "partition", "account", "qos",
                "submit_utc", "start_utc", "end_utc", "slurm_elapsed_seconds",
                "slurm_max_rss_gib", "nodes", "tasks", "cpus_per_task",
                "memory_gib", "gpu_count", "slurm_job_id", "node_list",
                "allocation_shape", "sacct_state", "sacct_exit_code",
            )
        )
        if (
            allocation in slurm_allocation_fingerprints
            and slurm_allocation_fingerprints[allocation] != fingerprint
        ):
            raise ValueError("duplicated sacct allocation rows disagree")
        slurm_allocation_fingerprints[allocation] = fingerprint
        slurm_by_allocation[allocation] = elapsed
    slurm_total = float(sum(slurm_by_allocation.values()))
    conservative_elapsed = max(runner_total, slurm_total)
    peak_rss = float(max(item["conservative_max_rss_gib"] for item in attempt_records))
    attempted_totals = {
        "attempt_count": len(attempt_records),
        "retry_count": sum(item["retry_ordinal_for_phase"] > 0 for item in attempt_records),
        "runner_elapsed_seconds_sum": runner_total,
        "slurm_elapsed_seconds_sum": slurm_total,
        "conservative_elapsed_seconds_max_of_sums": conservative_elapsed,
        "maximum_of_cpp_and_slurm_peak_rss_gib": peak_rss,
        "H_batched_actions_started": sum(item["attempted_h_batched_actions"] for item in attempt_records),
        "attempted_actual_scalar_MVM": sum(item["attempted_actual_scalar_MVM"] for item in attempt_records),
        "Splus_batched_actions_started": sum(
            item["attempted_splus_started_batched_actions"]
            for item in attempt_records
        ),
        "Splus_batched_actions_completed": sum(
            item["attempted_splus_completed_batched_actions"]
            for item in attempt_records
        ),
        "Splus_batched_actions_abandoned": sum(
            item["attempted_splus_abandoned_batched_actions"]
            for item in attempt_records
        ),
        "basis_initializations_started": sum(
            item["basis_initializations_started"] for item in attempt_records
        ),
        "basis_initializations_completed": sum(
            item["basis_initializations_completed"] for item in attempt_records
        ),
        "basis_initializations_abandoned": sum(
            item["basis_initializations_abandoned"] for item in attempt_records
        ),
        "seed_rows_started": sum(item["seed_rows_started"] for item in attempt_records),
        "seed_rows_completed": sum(item["seed_rows_completed"] for item in attempt_records),
        "seed_rows_abandoned": sum(item["seed_rows_abandoned"] for item in attempt_records),
        "determinant_calls_started": sum(item["determinant_calls_started"] for item in attempt_records),
        "determinant_calls_completed": sum(item["determinant_calls_completed"] for item in attempt_records),
        "determinant_calls_abandoned": sum(item["determinant_calls_abandoned"] for item in attempt_records),
    }
    gates = {
        "attempt_event_cpp_sacct_bijection": (
            set(cpp_attempt_telemetry) == cpp_required_attempts
            and set(identity_payloads) == set(attempts)
            and set(sacct_by_attempt) == set(attempts)
        ),
        "scheduler_identity_exact": bool(scheduler_identity_exact),
        "allocation_shape_exact": bool(allocation_shape_exact),
        "scheduler_chronology_valid": bool(chronology_valid),
        "cpp_telemetry_complete": bool(telemetry_complete),
        "attempt_artifacts_complete": bool(attempt_artifacts_complete),
        "attempt_work_reconciled_to_commits": bool(work_reconciled),
        "committed_science_exact": bool(committed_gate),
        "cumulative_elapsed_within_limit": conservative_elapsed <= 18000.0,
        "peak_rss_within_limit": peak_rss <= 89.6,
        "gpu_forbidden": all(row["gpu_count"] == 0 for row in sacct_rows),
    }
    payload = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-v8-RESOURCE_REVIEW",
        "created_utc": created_utc,
        "status": "C0_RESOURCE_PASS" if all(gates.values()) else "C0_RESOURCE_NO_GO",
        "run_id": run_id,
        "parent_records": parent_records,
        "raw_sacct": {"path": raw_path, "sha256": raw_sha},
        "thresholds": {
            "maximum_cumulative_attempted_runner_and_slurm_elapsed_seconds": 18000.0,
            "maximum_of_process_and_slurm_peak_rss_gib": 89.6,
            "elapsed_aggregation_rule": "max(sum(runner_elapsed_each_lifecycle),sum(unique_slurm_allocation_elapsed))",
            "rss_aggregation_rule": "max(all_CPP_process_MaxRSS,all_slurm_MaxRSS)",
        },
        "attempt_records": attempt_records,
        "attempted_totals": attempted_totals,
        "committed_science": dict(committed_science),
        "operational_residuals": operational_residuals,
        "gates": gates,
        "interpretation": "null is an implementation invariant only; resource review contains no energy or physical/null timing comparison",
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return validate_c0_resource_review_payload(
        payload, repository_root=repository_root
    )


def validate_c0_resource_review_payload(
    payload: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Recompute every aggregate and status in a v8 C0 resource review."""

    if not isinstance(payload, Mapping) or set(payload) != _RESOURCE_REVIEW_KEYS:
        raise ValueError("C0 resource review has missing or extra fields")
    if (
        payload["schema_version"] != 8
        or payload["stage"] != "P9v2-D1-T-C0-v8-RESOURCE_REVIEW"
        or payload["status"] not in {"C0_RESOURCE_PASS", "C0_RESOURCE_NO_GO"}
        or not isinstance(payload["run_id"], str)
        or not payload["run_id"]
        or not isinstance(payload["attempt_records"], list)
        or not payload["attempt_records"]
    ):
        raise ValueError("C0 resource review identity is invalid")
    review_created = _utc(payload["created_utc"])
    prepared_runtime_sha256: str | None = None
    parents = payload["parent_records"]
    if not isinstance(parents, Mapping) or set(parents) != {
        "result", "prepared_sha256", "full_seed_fill_done_sha256",
        "depth2_marker_sha256", "event_chain_tip_sha256",
        "event_chain_record_count",
    }:
        raise ValueError("resource review parent schema mismatch")
    for key, digest in parents.items():
        if key == "event_chain_record_count":
            if isinstance(digest, bool) or not isinstance(digest, int) or digest < 1:
                raise ValueError("resource event count is invalid")
        elif key == "result":
            _validate_record_file(digest, repository_root=repository_root)
        else:
            _full_sha(digest)
    if repository_root is not None:
        result_path, _ = _validate_record_file(
            parents["result"], repository_root=repository_root
        )
        parent_result = validate_c0_numerical_result_payload(
            _load_json(Path(repository_root) / result_path),
            repository_root=repository_root,
        )
        prepared_path = (
            Path(repository_root)
            / parent_result["parent_records"]["prepared"]["path"]
        )
        prepared_payload = _load_json(prepared_path)
        prepared_runtime_sha256 = _full_sha(
            prepared_payload["records"]["runtime_environment_manifest"]["sha256"]
        )
        result_parents = parent_result["parent_records"]
        if (
            parent_result["run_id"] != payload["run_id"]
            or _utc(parent_result["created_utc"]) > review_created
            or parents["prepared_sha256"]
            != result_parents["prepared"]["sha256"]
            or parents["full_seed_fill_done_sha256"]
            != result_parents["full_seed_fill_done"]["sha256"]
            or parents["depth2_marker_sha256"]
            != result_parents["generation_markers"][2]["sha256"]
            or parent_result["committed_science"] != payload["committed_science"]
        ):
            raise ValueError("resource review is detached from numerical RESULT")
    _validate_record_file(
        payload["raw_sacct"], repository_root=repository_root
    )
    thresholds = payload["thresholds"]
    if not isinstance(thresholds, Mapping) or thresholds != {
        "maximum_cumulative_attempted_runner_and_slurm_elapsed_seconds": 18000.0,
        "maximum_of_process_and_slurm_peak_rss_gib": 89.6,
        "elapsed_aggregation_rule": "max(sum(runner_elapsed_each_lifecycle),sum(unique_slurm_allocation_elapsed))",
        "rss_aggregation_rule": "max(all_CPP_process_MaxRSS,all_slurm_MaxRSS)",
    }:
        raise ValueError("resource review threshold/aggregation rules changed")
    operational_residuals = payload["operational_residuals"]
    if not isinstance(operational_residuals, list):
        raise ValueError("resource operational residual inventory is not a list")
    for residual in operational_residuals:
        if (
            not isinstance(residual, Mapping)
            or set(residual) != _OPERATIONAL_RESIDUAL_KEYS
            or not isinstance(residual["kind"], str)
            or not residual["kind"]
            or not isinstance(residual["path"], str)
            or not residual["path"]
            or type(residual["is_directory"]) is not bool
            or not isinstance(residual["file_records"], list)
        ):
            raise ValueError("resource operational residual record is malformed")
        for file_record in residual["file_records"]:
            if not isinstance(file_record, Mapping) or set(file_record) != {
                "path", "sha256", "byte_count"
            }:
                raise ValueError("resource residual file record is malformed")
            path, _ = _validate_record_file(
                {"path": file_record["path"], "sha256": file_record["sha256"]},
                repository_root=repository_root,
            )
            if (
                isinstance(file_record["byte_count"], bool)
                or not isinstance(file_record["byte_count"], int)
                or file_record["byte_count"] < 0
                or (
                    repository_root is not None
                    and (Path(repository_root) / path).stat().st_size
                    != file_record["byte_count"]
                )
            ):
                raise ValueError("resource residual file byte count is invalid")
    attempts = payload["attempt_records"]
    seen_attempt_ids: set[str] = set()
    phase_counts: dict[str, int] = {}
    scheduler_identity_exact = True
    allocation_shape_exact = True
    chronology_valid = True
    telemetry_complete = True
    attempt_artifacts_complete = not operational_residuals
    work_reconciled: bool | None = None
    gpu_forbidden = True
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or set(attempt) != _RESOURCE_ATTEMPT_KEYS:
            raise ValueError("resource attempt record schema mismatch")
        for key in (
            "runner_elapsed_seconds", "slurm_elapsed_seconds",
            "conservative_elapsed_seconds", "slurm_max_rss_gib",
            "conservative_max_rss_gib",
        ):
            _strict_finite_number(attempt[key], key, nonnegative=True)
        if attempt["cpp_max_rss_gib"] is not None:
            _strict_finite_number(attempt["cpp_max_rss_gib"], "CPP MaxRSS", nonnegative=True)
        for key in (
            "attempted_h_batched_actions", "attempted_actual_scalar_MVM",
            "attempted_splus_started_batched_actions",
            "attempted_splus_completed_batched_actions",
            "attempted_splus_abandoned_batched_actions",
            "basis_initializations_started",
            "basis_initializations_completed",
            "basis_initializations_abandoned",
            "seed_rows_started", "seed_rows_completed", "seed_rows_abandoned",
            "determinant_calls_started", "determinant_calls_completed",
            "determinant_calls_abandoned", "retry_ordinal_for_phase",
        ):
            if isinstance(attempt[key], bool) or not isinstance(attempt[key], int) or attempt[key] < 0:
                raise ValueError("resource attempt work counter is invalid")
        if attempt["runner_elapsed_measurement"] not in {
            "FINAL_EVENT_MONOTONIC",
            "UNAVAILABLE_HISTORICAL_CRASH_RAW_SACCT_COVERS",
        } or (
            attempt["runner_elapsed_measurement"]
            == "UNAVAILABLE_HISTORICAL_CRASH_RAW_SACCT_COVERS"
            and float(attempt["runner_elapsed_seconds"]) != 0.0
        ):
            raise ValueError("resource runner elapsed provenance is invalid")
        attempt_id = attempt["attempt_id"]
        phase = attempt["phase"]
        attempt_class = attempt["attempt_class"]
        science_phase = phase in {"seed-fill", "depth-0", "depth-1", "depth-2"}
        launched_science = attempt_class == "SCIENCE_CHILD"
        if launched_science and attempt["cpp_telemetry_validation"] == "VALID_EXACT_SCHEMA":
            if attempt["cpp_max_rss_gib"] is None:
                raise ValueError("validated science telemetry lacks CPP MaxRSS")
        elif launched_science and attempt["cpp_telemetry_validation"] == "MISSING_REQUIRED_CPP_TELEMETRY":
            if attempt["cpp_max_rss_gib"] is not None:
                raise ValueError("missing science telemetry claims CPP MaxRSS")
        elif launched_science:
            raise ValueError("launched science CPP validation label is invalid")
        elif attempt["cpp_telemetry_record"] is not None:
            raise ValueError("zero-child attempt cannot claim CPP telemetry")
        elif attempt["cpp_max_rss_gib"] is not None:
            raise ValueError("zero-child attempt cannot claim CPP MaxRSS")
        identity_path, identity_sha = _validate_record_file(
            attempt["attempt_identity_record"], repository_root=repository_root
        )
        event_records = attempt["event_records"]
        if (
            not isinstance(event_records, list)
            or not event_records
        ):
            raise ValueError("resource attempt event records are malformed")
        normalized_event_records = [
            _validate_record_file(record, repository_root=repository_root)
            for record in event_records
        ]
        stdout_record = attempt["stdout_record"]
        stderr_record = attempt["stderr_record"]
        if stdout_record is not None:
            _validate_record_file(stdout_record, repository_root=repository_root)
        if stderr_record is not None:
            _validate_record_file(stderr_record, repository_root=repository_root)
        residual_records = attempt["residual_staging_records"]
        if (
            not isinstance(residual_records, list)
            or type(attempt["residual_staging_directory_present"]) is not bool
            or type(attempt["cpp_telemetry_archived"]) is not bool
        ):
            raise ValueError("resource residual-staging manifest is not a list")
        for residual in residual_records:
            if not isinstance(residual, Mapping) or set(residual) != {
                "path", "sha256", "byte_count"
            }:
                raise ValueError("resource residual-staging record is malformed")
            path, _ = _validate_record_file(
                {"path": residual["path"], "sha256": residual["sha256"]},
                repository_root=repository_root,
            )
            if (
                isinstance(residual["byte_count"], bool)
                or not isinstance(residual["byte_count"], int)
                or residual["byte_count"] < 0
                or (
                    repository_root is not None
                    and (Path(repository_root) / path).stat().st_size
                    != residual["byte_count"]
                )
            ):
                raise ValueError("resource residual-staging byte count is invalid")
        source_record = attempt["source_duplicate_record"]
        if source_record is not None:
            source_path, source_sha = _validate_record_file(
                source_record, repository_root=repository_root
            )
            if attempt["cpp_telemetry_record"] is None:
                raise ValueError("telemetry source lacks its CPP record")
            cpp_path_for_source, cpp_sha_for_source = _validate_record_file(
                attempt["cpp_telemetry_record"], repository_root=repository_root
            )
            if source_sha != cpp_sha_for_source:
                raise ValueError("telemetry source and CPP record hashes differ")
            if attempt["cpp_telemetry_archived"]:
                if (
                    repository_root is not None
                    and not os.path.samefile(
                        Path(repository_root) / source_path,
                        Path(repository_root) / cpp_path_for_source,
                    )
                ):
                    raise ValueError("source/archive duplicate is not one inode")
            elif source_path != cpp_path_for_source:
                raise ValueError("unarchived telemetry must point to its source")
        elif attempt["cpp_telemetry_record"] is not None and not attempt[
            "cpp_telemetry_archived"
        ]:
            raise ValueError("unarchived CPP telemetry lacks its source record")
        if launched_science and attempt["cpp_telemetry_validation"] == "VALID_EXACT_SCHEMA":
            if attempt["cpp_telemetry_record"] is None:
                raise ValueError("valid CPP telemetry lacks its file record")
            _validate_record_file(
                attempt["cpp_telemetry_record"], repository_root=repository_root
            )
        elif launched_science and attempt["cpp_telemetry_validation"] == "MISSING_REQUIRED_CPP_TELEMETRY":
            if attempt["cpp_telemetry_record"] is not None or attempt["cpp_max_rss_gib"] is not None:
                raise ValueError("missing CPP telemetry record claims CPP data")
        elif not launched_science and attempt["cpp_telemetry_validation"] != "NO_CPP_BY_DESIGN":
            raise ValueError("zero-child CPP validation label is invalid")
        expected_attempt_identity = {
            "attempt_id": attempt["attempt_id"],
            "allocation_id": attempt["allocation_id"],
            "slurm_job_id": attempt["slurm_job_id"],
            "node_list": attempt["node_list"],
            "runtime_manifest_sha256": attempt["runtime_manifest_sha256"],
            "allocation_shape": attempt["allocation_shape"],
        }
        if repository_root is not None:
            identity_payload = _load_json(Path(repository_root) / identity_path)
            if (
                set(identity_payload)
                != {
                    "schema_version", "stage", "created_utc", "phase",
                    "science_identity", "attempt_identity",
                }
                or identity_payload.get("schema_version") != 8
                or identity_payload.get("stage")
                != "P9v2-D1-T-C0-v8-attempt-identity"
                or identity_payload.get("phase") != phase
                or identity_payload.get("attempt_identity")
                != expected_attempt_identity
                or not isinstance(identity_payload.get("science_identity"), Mapping)
                or identity_payload["science_identity"].get("run_id")
                != payload["run_id"]
                or identity_payload["science_identity"].get("prepared_sha256")
                != parents["prepared_sha256"]
            ):
                raise ValueError("resource attempt IDENTITY semantic replay failed")
            _utc(identity_payload["created_utc"])
            if attempt["cpp_telemetry_record"] is not None:
                cpp_path, _ = _validate_record_file(
                    attempt["cpp_telemetry_record"],
                    repository_root=repository_root,
                )
                replayed_cpp = validate_c0_attempt_telemetry(
                    _load_json(Path(repository_root) / cpp_path),
                    phase,
                    expected_attempt_identity=expected_attempt_identity,
                )
                compared_cpp_fields = {
                    "attempted_h_batched_actions": "attempted_h_batched_actions",
                    "attempted_actual_scalar_MVM": "attempted_actual_scalar_MVM",
                    "attempted_splus_started_batched_actions": "attempted_splus_started_batched_actions",
                    "attempted_splus_completed_batched_actions": "attempted_splus_completed_batched_actions",
                    "attempted_splus_abandoned_batched_actions": "attempted_splus_abandoned_batched_actions",
                    "basis_initializations_started": "basis_initializations_started",
                    "basis_initializations_completed": "basis_initializations_completed",
                    "basis_initializations_abandoned": "basis_initializations_abandoned",
                    "seed_rows_started": "seed_rows_started",
                    "seed_rows_completed": "seed_rows_completed",
                    "seed_rows_abandoned": "seed_rows_abandoned",
                    "determinant_calls_started": "determinant_calls_started",
                    "determinant_calls_completed": "determinant_calls_completed",
                    "determinant_calls_abandoned": "determinant_calls_abandoned",
                    "cpp_max_rss_gib": "cpp_max_rss_gib",
                    "stage": "cpp_telemetry_stage",
                }
                if any(
                    replayed_cpp[source] != attempt[target]
                    for source, target in compared_cpp_fields.items()
                ):
                    raise ValueError("resource CPP telemetry normalization changed")
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in seen_attempt_ids
            or phase not in {"seed-fill", "depth-0", "depth-1", "depth-2", "commit-recovery-only"}
            or attempt_class not in {
                "SCIENCE_CHILD", "SCIENCE_PRELAUNCH_ZERO", "RECOVERY_ONLY_ZERO"
            }
            or (science_phase != (attempt_class != "RECOVERY_ONLY_ZERO"))
            or (phase == "commit-recovery-only")
            != (attempt_class == "RECOVERY_ONLY_ZERO")
            or attempt["terminal_event_type"] not in {
                "DONE", "ABANDONED", "REQUEUE",
            }
        ):
            raise ValueError("resource attempt identity/lifecycle is invalid")
        seen_attempt_ids.add(attempt_id)
        expected_retry = phase_counts.get(phase, 0)
        if attempt["retry_ordinal_for_phase"] != expected_retry:
            raise ValueError("resource retry ordinal is not the event-order ordinal")
        phase_counts[phase] = expected_retry + 1
        expected_stage: str | set[str] = (
            "P9v2-D1-T-C0-v8-seed-attempt-telemetry"
            if phase == "seed-fill" and launched_science
            else {
                "P9v2-D1-T-C0-v8-attempt-only",
                "P9v2-D1-T-C0-production-v8",
            }
            if phase in {"depth-0", "depth-1", "depth-2"} and launched_science
            else "PRELAUNCH_ZERO_NO_CPP_BY_DESIGN"
            if attempt_class == "SCIENCE_PRELAUNCH_ZERO"
            else "RECOVERY_ONLY_NO_CPP_BY_DESIGN"
        )
        telemetry_complete &= (
            attempt["cpp_telemetry_stage"] == expected_stage
            if isinstance(expected_stage, str)
            else attempt["cpp_telemetry_stage"] in expected_stage
        )
        has_logs = stdout_record is not None and stderr_record is not None
        has_cpp = attempt["cpp_telemetry_record"] is not None
        if launched_science:
            attempt_artifacts_complete &= (
                has_logs and has_cpp and attempt["cpp_telemetry_archived"]
            )
            if attempt["terminal_event_type"] == "DONE":
                attempt_artifacts_complete &= (
                    not attempt["residual_staging_directory_present"]
                    and not residual_records
                    and source_record is None
                )
        elif (
            has_logs
            or has_cpp
            or attempt["residual_staging_directory_present"]
            or residual_records
            or source_record is not None
        ):
            raise ValueError("zero-child resource attempt contains child artifacts")
        scheduler_identity_exact &= (
            attempt["cluster"] == "unicorn"
            and attempt["job_name"] == "nqs-d1-c0-v8"
            and attempt["partition"] == "kim"
            and attempt["account"] == "kim"
            and attempt["qos"] == "normal"
            and all(
                isinstance(attempt[key], str) and bool(attempt[key])
                for key in (
                    "slurm_job_id", "allocation_id", "node_list",
                    "runtime_manifest_sha256", "allocation_shape",
                    "sacct_state", "sacct_exit_code",
                )
            )
            and re.fullmatch(
                r"[0-9a-f]{64}", attempt["runtime_manifest_sha256"]
            ) is not None
            and (
                prepared_runtime_sha256 is None
                or attempt["runtime_manifest_sha256"]
                == prepared_runtime_sha256
            )
            and attempt["allocation_shape"] == EXPECTED_ALLOCATION_SHAPE
            and re.fullmatch(r"[1-9][0-9]*", attempt["slurm_job_id"])
            is not None
            and attempt["allocation_id"]
            == f"slurm-{attempt['slurm_job_id']}-restart-"
            + attempt["allocation_id"].rsplit("-", 1)[-1]
            and re.fullmatch(
                r"slurm-[1-9][0-9]*-restart-(?:0|[1-9][0-9]*)",
                attempt["allocation_id"],
            )
            is not None
            and attempt["attempt_id"].endswith(
                f"-{attempt['allocation_id']}"
            )
        )
        allocation_shape_exact &= (
            type(attempt["nodes"]) is int
            and attempt["nodes"] == 1
            and type(attempt["tasks"]) is int
            and attempt["tasks"] == 1
            and type(attempt["cpus_per_task"]) is int
            and attempt["cpus_per_task"] == 40
            and type(attempt["memory_gib"]) is int
            and attempt["memory_gib"] == 128
            and type(attempt["gpu_count"]) is int
            and attempt["gpu_count"] == 0
        )
        gpu_forbidden &= type(attempt["gpu_count"]) is int and attempt["gpu_count"] == 0
        submit = _utc(attempt["submit_utc"])
        start = _utc(attempt["start_utc"])
        end = _utc(attempt["end_utc"])
        chronology_valid &= (
            submit <= start <= end <= review_created
            and abs(
                (end - start).total_seconds()
                - float(attempt["slurm_elapsed_seconds"])
            )
            <= 1.0
        )
        if (
            attempt["attempted_actual_scalar_MVM"]
            != 3 * attempt["attempted_h_batched_actions"]
            or attempt["attempted_splus_abandoned_batched_actions"]
            != attempt["attempted_splus_started_batched_actions"]
            - attempt["attempted_splus_completed_batched_actions"]
            or attempt["basis_initializations_abandoned"]
            != attempt["basis_initializations_started"]
            - attempt["basis_initializations_completed"]
            or attempt["seed_rows_abandoned"]
            != attempt["seed_rows_started"] - attempt["seed_rows_completed"]
            or attempt["determinant_calls_started"]
            != 3 * attempt["seed_rows_started"]
            or attempt["determinant_calls_completed"]
            != 3 * attempt["seed_rows_completed"]
            or attempt["determinant_calls_abandoned"]
            != 3 * attempt["seed_rows_abandoned"]
        ):
            raise ValueError("resource attempt work arithmetic is inconsistent")
        if attempt_class != "SCIENCE_CHILD" and any(
            attempt[key] != 0
            for key in (
                "attempted_h_batched_actions", "attempted_actual_scalar_MVM",
                "attempted_splus_started_batched_actions",
                "attempted_splus_completed_batched_actions",
                "attempted_splus_abandoned_batched_actions",
                "basis_initializations_started",
                "basis_initializations_completed",
                "basis_initializations_abandoned",
                "seed_rows_started", "seed_rows_completed", "seed_rows_abandoned",
                "determinant_calls_started", "determinant_calls_completed",
                "determinant_calls_abandoned",
            )
        ):
            raise ValueError("recovery-only lifecycle contains science work")
        if attempt["conservative_elapsed_seconds"] != max(
            attempt["runner_elapsed_seconds"], attempt["slurm_elapsed_seconds"]
        ) or attempt["conservative_max_rss_gib"] != max(
            attempt["slurm_max_rss_gib"],
            0.0 if attempt["cpp_max_rss_gib"] is None else attempt["cpp_max_rss_gib"],
        ):
            raise ValueError("resource attempt conservative measurement mismatch")
    flattened_event_hashes = [
        record["sha256"]
        for attempt in attempts
        for record in attempt["event_records"]
    ]
    if (
        len(flattened_event_hashes) != parents["event_chain_record_count"]
        or len(set(flattened_event_hashes)) != len(flattened_event_hashes)
        or flattened_event_hashes[-1] != parents["event_chain_tip_sha256"]
    ):
        raise ValueError("resource attempt records do not cover the event chain")
    if repository_root is not None:
        result_event_tip = parent_result["parent_records"]["event_chain"]["tip"]
        run_root = (
            Path(repository_root) / result_event_tip["path"]
        ).resolve(strict=True).parent.parent
        rescanned = _scan_c0_attempt_artifacts(
            Path(repository_root),
            run_root,
            {
                str(attempt["attempt_id"]): {"phase": attempt["phase"]}
                for attempt in attempts
            },
        )
        for attempt in attempts:
            artifact = rescanned[str(attempt["attempt_id"])]
            expected_artifacts = {
                "attempt_identity_record": artifact["identity_record"],
                "stdout_record": artifact["stdout_record"],
                "stderr_record": artifact["stderr_record"],
                "cpp_telemetry_record": artifact["cpp_record"],
                "residual_staging_records": artifact["residual_staging_records"],
                "residual_staging_directory_present": artifact[
                    "residual_staging_directory_present"
                ],
                "source_duplicate_record": artifact["source_duplicate_record"],
                "cpp_telemetry_archived": artifact["cpp_telemetry_archived"],
            }
            if any(attempt[key] != value for key, value in expected_artifacts.items()):
                raise ValueError(
                    "resource attempt artifact manifest differs from full rescan"
                )
        previous_sha: str | None = None
        previous_time: datetime | None = None
        global_index = 0
        allowed_after = {
            None: {"STARTED"},
            "STARTED": {"COMPLETED", "ABANDONED", "TERMINATED"},
            "COMPLETED": {"DONE", "ABANDONED"},
            "TERMINATED": {"REQUEUE"},
            "REQUEUE": {"STARTED"},
            "ABANDONED": {"STARTED"},
            "DONE": {"STARTED"},
        }
        previous_type: str | None = None
        for attempt in attempts:
            for local_index, record in enumerate(attempt["event_records"]):
                event_path, event_sha = _validate_record_file(
                    record, repository_root=repository_root
                )
                event = _load_json(Path(repository_root) / event_path)
                event_type = event.get("event_type")
                event_time = _utc(event.get("created_utc"))
                if (
                    set(event)
                    != {
                        "schema_version", "stage", "event_index", "event_type",
                        "created_utc", "phase", "depth", "science_identity",
                        "attempt_identity", "parent_event_sha256", "details",
                    }
                    or event.get("schema_version") != 8
                    or event.get("stage") != "P9v2-D1-T-C0-v8-event"
                    or event.get("event_index") != global_index
                    or event.get("parent_event_sha256") != previous_sha
                    or event_type not in allowed_after[previous_type]
                    or event.get("phase") != attempt["phase"]
                    or not isinstance(event.get("attempt_identity"), Mapping)
                    or event["attempt_identity"].get("attempt_id")
                    != attempt["attempt_id"]
                    or not isinstance(event.get("science_identity"), Mapping)
                    or event["science_identity"].get("run_id") != payload["run_id"]
                    or event["science_identity"].get("prepared_sha256")
                    != parents["prepared_sha256"]
                    or (local_index == 0 and event_type != "STARTED")
                    or (
                        local_index == len(attempt["event_records"]) - 1
                        and event_type != attempt["terminal_event_type"]
                    )
                    or event_time > review_created
                    or (
                        previous_time is not None
                        and event_time < previous_time
                    )
                ):
                    raise ValueError("resource event semantic replay failed")
                if local_index == len(attempt["event_records"]) - 1:
                    details = event["details"]
                    measured = details.get(
                        "runner_monotonic_elapsed_seconds_this_attempt"
                    )
                    if measured is None:
                        if (
                            attempt["runner_elapsed_measurement"]
                            != "UNAVAILABLE_HISTORICAL_CRASH_RAW_SACCT_COVERS"
                            or float(attempt["runner_elapsed_seconds"]) != 0.0
                        ):
                            raise ValueError("runner elapsed missingness was relabelled")
                    elif (
                        attempt["runner_elapsed_measurement"]
                        != "FINAL_EVENT_MONOTONIC"
                        or _strict_finite_number(
                            measured,
                            "terminal runner monotonic elapsed",
                            nonnegative=True,
                        )
                        != float(attempt["runner_elapsed_seconds"])
                    ):
                        raise ValueError("runner elapsed differs from terminal event")
                previous_sha = event_sha
                previous_type = str(event_type)
                previous_time = event_time
                global_index += 1
        expected_raw_rows = []
        for attempt in attempts:
            expected_raw_rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "slurm_job_id": attempt["slurm_job_id"],
                    "cluster": attempt["cluster"],
                    "job_name": attempt["job_name"],
                    "submit_utc": attempt["submit_utc"],
                    "start_utc": attempt["start_utc"],
                    "end_utc": attempt["end_utc"],
                    "elapsed_seconds": attempt["slurm_elapsed_seconds"],
                    "max_rss_gib": attempt["slurm_max_rss_gib"],
                    "partition": attempt["partition"],
                    "account": attempt["account"],
                    "qos": attempt["qos"],
                    "nodes": attempt["nodes"],
                    "tasks": attempt["tasks"],
                    "cpus_per_task": attempt["cpus_per_task"],
                    "memory_gib": attempt["memory_gib"],
                    "gpu_count": attempt["gpu_count"],
                    "allocation_id": attempt["allocation_id"],
                    "node_list": attempt["node_list"],
                    "allocation_shape": attempt["allocation_shape"],
                    "state": attempt["sacct_state"],
                    "exit_code": attempt["sacct_exit_code"],
                }
            )
        raw_path, _ = _validate_record_file(
            payload["raw_sacct"], repository_root=repository_root
        )
        raw = _load_json(Path(repository_root) / raw_path)
        reparsed_raw_rows = _parse_raw_sacct_snapshot(
            raw,
            run_id=payload["run_id"],
            expected_attempts=[
                {
                    "terminal": attempt["terminal_event_type"],
                    "attempt_identity": {
                        "attempt_id": attempt["attempt_id"],
                        "slurm_job_id": attempt["slurm_job_id"],
                        "node_list": attempt["node_list"],
                        "allocation_id": attempt["allocation_id"],
                        "runtime_manifest_sha256": attempt[
                            "runtime_manifest_sha256"
                        ],
                        "allocation_shape": attempt["allocation_shape"],
                    }
                }
                for attempt in attempts
            ],
            latest_allowed_utc=review_created,
        )
        if reparsed_raw_rows != expected_raw_rows:
            raise ValueError("raw sacct literal replay differs from resource review")
        committed_seed_rows_by_attempt: dict[str, int] = {}
        committed_seed_invocations_by_attempt: dict[str, set[str]] = {}
        attempts_by_id = {str(item["attempt_id"]): item for item in attempts}
        full_record = parent_result["parent_records"]["full_seed_fill_done"]
        full_payload = _load_json(Path(repository_root) / full_record["path"])
        seed_root = Path(full_payload["seed_chain_root"]).resolve(strict=True)
        for ordinal in range(CHUNK_COUNT):
            metadata = _load_json(
                seed_root
                / "chunks"
                / f"chunk-{ordinal:04d}"
                / "METADATA.json"
            )
            seed_identity = metadata["attempt_identity"]
            if not isinstance(seed_identity, Mapping):
                raise ValueError("committed seed metadata lacks attempt identity")
            attempt_id = seed_identity["attempt_id"]
            expected_attempt = attempts_by_id.get(str(attempt_id))
            expected_identity = (
                None
                if expected_attempt is None
                else {
                    "attempt_id": expected_attempt["attempt_id"],
                    "allocation_id": expected_attempt["allocation_id"],
                    "slurm_job_id": expected_attempt["slurm_job_id"],
                    "node_list": expected_attempt["node_list"],
                    "runtime_manifest_sha256": expected_attempt[
                        "runtime_manifest_sha256"
                    ],
                    "allocation_shape": expected_attempt["allocation_shape"],
                }
            )
            if (
                expected_attempt is None
                or expected_attempt["phase"] != "seed-fill"
                or expected_attempt["attempt_class"] != "SCIENCE_CHILD"
                or dict(seed_identity) != expected_identity
            ):
                raise ValueError(
                    "committed seed chunk is detached from its seed attempt"
                )
            committed_seed_rows_by_attempt[attempt_id] = (
                committed_seed_rows_by_attempt.get(attempt_id, 0)
                + int(metadata["count"])
            )
            committed_seed_invocations_by_attempt.setdefault(
                attempt_id, set()
            ).add(str(metadata["producer_invocation_id"]))
        committed_depth_by_attempt: dict[str, int] = {}
        checkpoint_roots: set[Path] = set()
        for depth, marker_record in enumerate(
            parent_result["parent_records"]["generation_markers"]
        ):
            marker_path = (
                Path(repository_root) / marker_record["path"]
            ).resolve(strict=True)
            checkpoint_roots.add(marker_path.parent.parent)
            marker = _load_json(marker_path)
            marker_identity = marker["attempt_identity"]
            if not isinstance(marker_identity, Mapping):
                raise ValueError("committed depth marker lacks attempt identity")
            attempt_id = marker_identity["attempt_id"]
            expected_attempt = attempts_by_id.get(str(attempt_id))
            expected_identity = (
                None
                if expected_attempt is None
                else {
                    "attempt_id": expected_attempt["attempt_id"],
                    "allocation_id": expected_attempt["allocation_id"],
                    "slurm_job_id": expected_attempt["slurm_job_id"],
                    "node_list": expected_attempt["node_list"],
                    "runtime_manifest_sha256": expected_attempt[
                        "runtime_manifest_sha256"
                    ],
                    "allocation_shape": expected_attempt["allocation_shape"],
                }
            )
            if (
                expected_attempt is None
                or expected_attempt["phase"] != f"depth-{depth}"
                or expected_attempt["attempt_class"] != "SCIENCE_CHILD"
                or dict(marker_identity) != expected_identity
            ):
                raise ValueError(
                    "committed depth marker is detached from its depth attempt"
                )
            if attempt_id in committed_depth_by_attempt:
                raise ValueError("one resource attempt claims two committed depths")
            committed_depth_by_attempt[attempt_id] = depth
        if set(committed_seed_rows_by_attempt).intersection(
            committed_depth_by_attempt
        ):
            raise ValueError("seed and depth commits reuse an attempt identity")
        if len(checkpoint_roots) != 1:
            raise ValueError("resource markers do not share one checkpoint root")
        rescanned_operational = _scan_c0_operational_residuals(
            Path(repository_root),
            run_root=run_root,
            seed_root=seed_root,
            checkpoint_root=checkpoint_roots.pop(),
        )
        if rescanned_operational != operational_residuals:
            raise ValueError(
                "resource operational residual inventory differs from full rescan"
            )
        work_reconciled = True
        for attempt in attempts:
            attempt_id = attempt["attempt_id"]
            phase = attempt["phase"]
            committed_rows_for_attempt = committed_seed_rows_by_attempt.get(
                attempt_id, 0
            )
            if phase == "seed-fill":
                work_reconciled &= (
                    attempt["seed_rows_completed"] == committed_rows_for_attempt
                    and attempt["determinant_calls_completed"]
                    == 3 * committed_rows_for_attempt
                )
                if attempt["cpp_telemetry_record"] is not None:
                    cpp_path, _ = _validate_record_file(
                        attempt["cpp_telemetry_record"],
                        repository_root=repository_root,
                    )
                    cpp = _load_json(Path(repository_root) / cpp_path)
                    invocations = committed_seed_invocations_by_attempt.get(
                        attempt_id, set()
                    )
                    work_reconciled &= (
                        not invocations
                        or invocations == {str(cpp["producer_invocation_id"])}
                    )
            elif phase in {"depth-0", "depth-1", "depth-2"}:
                expected_depth = int(phase[-1])
                committed = (
                    committed_depth_by_attempt.get(attempt_id) == expected_depth
                )
                if committed:
                    work_reconciled &= (
                        attempt["terminal_event_type"] == "DONE"
                        and attempt["cpp_telemetry_stage"]
                        == "P9v2-D1-T-C0-production-v8"
                        and attempt["attempted_h_batched_actions"] == 2
                    )
                elif attempt["terminal_event_type"] == "DONE":
                    work_reconciled = False
        work_reconciled &= (
            sum(committed_seed_rows_by_attempt.values()) == FULL_DIMENSION
            and set(committed_depth_by_attempt.values()) == {0, 1, 2}
            and set(committed_seed_rows_by_attempt).issubset(seen_attempt_ids)
            and set(committed_depth_by_attempt).issubset(seen_attempt_ids)
        )
    totals = payload["attempted_totals"]
    slurm_by_allocation: dict[str, float] = {}
    slurm_allocation_fingerprints: dict[str, tuple[Any, ...]] = {}
    for item in attempts:
        allocation = item["allocation_id"]
        elapsed = float(item["slurm_elapsed_seconds"])
        fingerprint = tuple(
            item[key]
            for key in (
                "cluster", "job_name", "partition", "account", "qos",
                "submit_utc", "start_utc", "end_utc", "slurm_elapsed_seconds",
                "slurm_max_rss_gib", "nodes", "tasks", "cpus_per_task",
                "memory_gib", "gpu_count", "slurm_job_id", "node_list",
                "allocation_shape", "sacct_state", "sacct_exit_code",
            )
        )
        if (
            allocation in slurm_allocation_fingerprints
            and slurm_allocation_fingerprints[allocation] != fingerprint
        ):
            raise ValueError("duplicated normalized sacct allocation rows disagree")
        slurm_allocation_fingerprints[allocation] = fingerprint
        slurm_by_allocation[allocation] = elapsed
    expected_totals = {
        "attempt_count": len(attempts),
        "retry_count": sum(item["retry_ordinal_for_phase"] > 0 for item in attempts),
        "runner_elapsed_seconds_sum": float(sum(item["runner_elapsed_seconds"] for item in attempts)),
        "slurm_elapsed_seconds_sum": float(sum(slurm_by_allocation.values())),
        "conservative_elapsed_seconds_max_of_sums": max(
            float(sum(item["runner_elapsed_seconds"] for item in attempts)),
            float(sum(slurm_by_allocation.values())),
        ),
        "maximum_of_cpp_and_slurm_peak_rss_gib": float(max(item["conservative_max_rss_gib"] for item in attempts)),
        "H_batched_actions_started": sum(item["attempted_h_batched_actions"] for item in attempts),
        "attempted_actual_scalar_MVM": sum(item["attempted_actual_scalar_MVM"] for item in attempts),
        "Splus_batched_actions_started": sum(item["attempted_splus_started_batched_actions"] for item in attempts),
        "Splus_batched_actions_completed": sum(item["attempted_splus_completed_batched_actions"] for item in attempts),
        "Splus_batched_actions_abandoned": sum(item["attempted_splus_abandoned_batched_actions"] for item in attempts),
        "basis_initializations_started": sum(item["basis_initializations_started"] for item in attempts),
        "basis_initializations_completed": sum(item["basis_initializations_completed"] for item in attempts),
        "basis_initializations_abandoned": sum(item["basis_initializations_abandoned"] for item in attempts),
        "seed_rows_started": sum(item["seed_rows_started"] for item in attempts),
        "seed_rows_completed": sum(item["seed_rows_completed"] for item in attempts),
        "seed_rows_abandoned": sum(item["seed_rows_abandoned"] for item in attempts),
        "determinant_calls_started": sum(item["determinant_calls_started"] for item in attempts),
        "determinant_calls_completed": sum(item["determinant_calls_completed"] for item in attempts),
        "determinant_calls_abandoned": sum(item["determinant_calls_abandoned"] for item in attempts),
    }
    if not isinstance(totals, Mapping) or dict(totals) != expected_totals:
        raise ValueError("resource review attempted totals were not replayed exactly")
    committed = payload["committed_science"]
    expected_committed = {
        "seed_rows": FULL_DIMENSION,
        "determinant_calls": COMMITTED_DETERMINANT_CALLS,
        "physical_H_batched_actions": 3,
        "physical_H_scalar_MVM": 9,
        "null_H_batched_actions": 3,
        "null_H_scalar_MVM": 9,
        "total_H_batched_actions": 6,
        "total_committed_scalar_MVM": 18,
    }
    committed_exact = isinstance(committed, Mapping) and dict(committed) == expected_committed and all(
        type(value) is int for value in committed.values()
    )
    gates = payload["gates"]
    if not isinstance(gates, Mapping) or set(gates) != _RESOURCE_GATE_KEYS or any(type(value) is not bool for value in gates.values()):
        raise ValueError("resource review gate map is malformed")
    recomputed_gates = {
        "attempt_event_cpp_sacct_bijection": (
            len(seen_attempt_ids) == len(attempts)
            and all(
                attempt["cpp_telemetry_validation"] !=
                "MISSING_REQUIRED_CPP_TELEMETRY"
                for attempt in attempts
            )
        ),
        "scheduler_identity_exact": bool(scheduler_identity_exact),
        "allocation_shape_exact": bool(allocation_shape_exact),
        "scheduler_chronology_valid": bool(chronology_valid),
        "cpp_telemetry_complete": bool(telemetry_complete),
        "attempt_artifacts_complete": bool(attempt_artifacts_complete),
        "attempt_work_reconciled_to_commits": (
            bool(work_reconciled)
            if work_reconciled is not None
            else bool(gates["attempt_work_reconciled_to_commits"])
        ),
        "committed_science_exact": bool(committed_exact),
        "cumulative_elapsed_within_limit": expected_totals["conservative_elapsed_seconds_max_of_sums"] <= 18000.0,
        "peak_rss_within_limit": expected_totals["maximum_of_cpp_and_slurm_peak_rss_gib"] <= 89.6,
        "gpu_forbidden": bool(gpu_forbidden),
    }
    if dict(gates) != recomputed_gates:
        raise ValueError("resource review gate values differ from recomputation")
    expected_status = "C0_RESOURCE_PASS" if all(gates.values()) else "C0_RESOURCE_NO_GO"
    if payload["status"] != expected_status:
        raise ValueError("resource review status differs from its gates")
    if payload["interpretation"] != "null is an implementation invariant only; resource review contains no energy or physical/null timing comparison":
        raise ValueError("resource review interpretation boundary changed")
    json.dumps(dict(payload), allow_nan=False, sort_keys=True)
    return dict(payload)


_FINAL_RESULT_KEYS = {
    "schema_version",
    "stage",
    "created_utc",
    "status",
    "run_id",
    "parent_records",
    "numerical_verdict",
    "resource_verdict",
    "ownership_endpoint",
    "c1_unlock",
    "claim_boundary",
    "headline_promotion",
}


def build_and_validate_c0_final_result(
    *,
    repository_root: Path,
    created_utc: str,
    numerical_result_record: Mapping[str, Any],
    resource_review_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable final C0 status from RESULT and resource review."""

    repository_root = Path(repository_root)
    result_path, result_sha = _validate_record_file(
        numerical_result_record, repository_root=repository_root
    )
    review_path, review_sha = _validate_record_file(
        resource_review_record, repository_root=repository_root
    )
    result = validate_c0_numerical_result_payload(
        _load_json(repository_root / result_path),
        repository_root=repository_root,
    )
    review = validate_c0_resource_review_payload(
        _load_json(repository_root / review_path),
        repository_root=repository_root,
    )
    created = _utc(created_utc)
    if (
        review["run_id"] != result["run_id"]
        or review["parent_records"]["result"]
        != {"path": result_path, "sha256": result_sha}
        or created < _utc(result["created_utc"])
        or created < _utc(review["created_utc"])
    ):
        raise ValueError("FINAL_RESULT parents/run/chronology disagree")
    resource_pass = review["status"] == "C0_RESOURCE_PASS"
    payload = {
        "schema_version": 8,
        "stage": "P9v2-D1-T-C0-v8-FINAL_RESULT",
        "created_utc": created_utc,
        "status": "C0_NUMERICAL_PASS" if resource_pass else "C0_RESOURCE_NO_GO",
        "run_id": result["run_id"],
        "parent_records": {
            "numerical_result": {"path": result_path, "sha256": result_sha},
            "resource_review": {"path": review_path, "sha256": review_sha},
        },
        "numerical_verdict": result["numerical_verdict"],
        "resource_verdict": review["status"],
        "ownership_endpoint": result["ownership_endpoint"],
        "c1_unlock": dict(result["c1_unlock"]),
        "claim_boundary": dict(result["claim_boundary"]),
        "headline_promotion": {
            "promoted": False,
            "reason": "C0 alone supplies no NQS, phase, QSL, locality, carrier, accuracy, or efficiency headline",
        },
    }
    return validate_c0_final_result_payload(payload)


def validate_c0_final_result_payload(
    payload: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Validate and optionally rehash/replay both FINAL_RESULT parents."""

    if not isinstance(payload, Mapping) or set(payload) != _FINAL_RESULT_KEYS:
        raise ValueError("C0 FINAL_RESULT has missing or extra fields")
    if (
        payload["schema_version"] != 8
        or payload["stage"] != "P9v2-D1-T-C0-v8-FINAL_RESULT"
        or payload["status"] not in {"C0_NUMERICAL_PASS", "C0_RESOURCE_NO_GO"}
        or payload["numerical_verdict"] != "C0_NUMERICAL_PASS"
        or payload["resource_verdict"] not in {
            "C0_RESOURCE_PASS", "C0_RESOURCE_NO_GO"
        }
        or not isinstance(payload["run_id"], str)
        or not payload["run_id"]
    ):
        raise ValueError("C0 FINAL_RESULT identity/verdict is malformed")
    final_created = _utc(payload["created_utc"])
    parents = payload["parent_records"]
    if not isinstance(parents, Mapping) or set(parents) != {
        "numerical_result", "resource_review"
    }:
        raise ValueError("C0 FINAL_RESULT parent schema mismatch")
    result_path, _ = _validate_record_file(
        parents["numerical_result"], repository_root=repository_root
    )
    review_path, _ = _validate_record_file(
        parents["resource_review"], repository_root=repository_root
    )
    from su2_nqs.benchmarks.xdiag_seeded_block_krylov import (
        validate_partitioned_ownership_payload,
    )

    validate_partitioned_ownership_payload(payload["ownership_endpoint"])
    if payload["c1_unlock"] != {
        "criterion": "C0_NUMERICAL_PASS",
        "unlocked": True,
        "ownership_status_does_not_lock_c1": True,
    }:
        raise ValueError("FINAL_RESULT changed the numerical-only C1 unlock")
    if payload["headline_promotion"] != {
        "promoted": False,
        "reason": "C0 alone supplies no NQS, phase, QSL, locality, carrier, accuracy, or efficiency headline",
    }:
        raise ValueError("FINAL_RESULT overstates the C0 claim boundary")
    if repository_root is not None:
        result = validate_c0_numerical_result_payload(
            _load_json(Path(repository_root) / result_path),
            repository_root=repository_root,
        )
        review = validate_c0_resource_review_payload(
            _load_json(Path(repository_root) / review_path),
            repository_root=repository_root,
        )
        resource_pass = review["status"] == "C0_RESOURCE_PASS"
        expected_status = (
            "C0_NUMERICAL_PASS" if resource_pass else "C0_RESOURCE_NO_GO"
        )
        if (
            result["run_id"] != payload["run_id"]
            or review["run_id"] != payload["run_id"]
            or review["parent_records"]["result"]
            != parents["numerical_result"]
            or payload["status"] != expected_status
            or payload["resource_verdict"] != review["status"]
            or payload["ownership_endpoint"] != result["ownership_endpoint"]
            or payload["claim_boundary"] != result["claim_boundary"]
            or payload["c1_unlock"] != result["c1_unlock"]
            or final_created < _utc(result["created_utc"])
            or final_created < _utc(review["created_utc"])
        ):
            raise ValueError("FINAL_RESULT differs from its immutable parents")
    json.dumps(dict(payload), allow_nan=False, sort_keys=True)
    return dict(payload)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write/fsync/replace a JSON marker on one filesystem."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload), handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "ACCEPTANCE_PATH",
    "ACCEPTANCE_SHA256",
    "CHUNK_COUNT",
    "CHUNK_SIZE",
    "COMMITTED_DETERMINANT_CALLS",
    "CONFIG_PATH",
    "CONFIG_SHA256",
    "EXACT_SMALL_SCOPE",
    "FORMAL_KRYLOV_SCOPE",
    "FORMAL_SEED_SCOPE",
    "FULL_DIMENSION",
    "LAST_CHUNK_COUNT",
    "PreparedValidation",
    "SYNTHETIC_SCOPE",
    "SeedChainSummary",
    "aggregate_and_validate_c0_resource_review",
    "atomic_json",
    "capture_and_publish_c0_raw_sacct_snapshot",
    "build_and_validate_c0_final_result",
    "build_and_validate_c0_numerical_result",
    "construct_c0_ownership_context",
    "expected_chunk_range",
    "load_frozen_config",
    "read_row_major_seed_payloads",
    "sha256_file",
    "validate_p2_evidence",
    "validate_c0_attempt_telemetry",
    "validate_c0_final_result_payload",
    "validate_c0_numerical_result_payload",
    "validate_c0_resource_review_payload",
    "validate_prepared",
    "validate_review_chronology",
    "validate_seed_chunk_chain",
]
