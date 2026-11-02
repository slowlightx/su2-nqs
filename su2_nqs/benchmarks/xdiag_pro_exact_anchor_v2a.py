"""Prospective PRO-v2a contract layered on the frozen XDiag v1 anchor.

This module deliberately reuses the already exact-small-validated PRO adapter
and runner while giving the narrowed retry its own immutable configuration,
source identity, output namespace, and decision boundary.  It does not mutate
or reinterpret v1 output.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

from .xdiag_exact_anchor import load_and_validate_config as load_v1_config
from .xdiag_exact_anchor import sha256_file, task_table


STAGE = "P9v2-D1-N36-XDiag-PRO-exact-anchor-v2a"
PARENT_CONFIG_RELATIVE = (
    "configs/physics_win/p9v2_d1_n36_xdiag_exact_anchor_v1.json"
)
PARENT_CONFIG_SHA256 = (
    "158464782b319335405c43933d92e07256fa773fd63020f98c6065cdec704dac"
)
RUN_NAMESPACE = "p9v2_d1_xdiag_pro_exact_anchor_v2a"


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_reused_sources(
    raw: Mapping[str, object], root: Path
) -> tuple[dict[str, str], ...]:
    records = raw.get("reused_implementation")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("v2a must pin all six reused implementation artifacts")
    checked: list[dict[str, str]] = []
    paths: set[str] = set()
    for record_value in records:
        record = _require_mapping(record_value, "reused implementation record")
        role = record.get("role")
        relative = record.get("path")
        expected = record.get("sha256")
        if not all(isinstance(item, str) and item for item in (role, relative, expected)):
            raise ValueError("reused implementation record is incomplete")
        assert isinstance(role, str)
        assert isinstance(relative, str)
        assert isinstance(expected, str)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("reused implementation path escaped repository") from error
        if relative in paths or not path.is_file():
            raise ValueError("reused implementation path is duplicate or missing")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"reused implementation hash mismatch: {relative}")
        paths.add(relative)
        checked.append(
            {"role": role, "path": relative, "sha256": observed}
        )
    return tuple(checked)


def load_and_validate_config(path: Path, *, root: Path) -> dict[str, object]:
    """Load v2a and materialize its strictly limited overrides over v1."""

    raw_value = json.loads(path.read_text())
    if not isinstance(raw_value, dict):
        raise ValueError("v2a config must be a JSON object")
    raw = raw_value
    if raw.get("schema_version") != 1 or raw.get("stage") != STAGE:
        raise ValueError("unexpected PRO-v2a schema or stage")
    if raw.get("registration_status") != (
        "prospective_authorized_this_turn_before_first_v2a_submission"
    ):
        raise ValueError("v2a authorization/registration status changed")

    parent = _require_mapping(raw.get("parent_contract"), "parent_contract")
    if dict(parent) != {
        "path": PARENT_CONFIG_RELATIVE,
        "sha256": PARENT_CONFIG_SHA256,
    }:
        raise ValueError("v2a parent contract identity changed")
    parent_path = root / PARENT_CONFIG_RELATIVE
    if not parent_path.is_file() or sha256_file(parent_path) != PARENT_CONFIG_SHA256:
        raise ValueError("frozen v1 parent config hash mismatch")
    reused = _validate_reused_sources(raw, root)

    observation = _require_mapping(
        raw.get("v1_task0_observation"), "v1_task0_observation"
    )
    if observation.get("evidence_status") != "archived_hash_bound_remote_result":
        raise ValueError("v1 task-0 observation is not archived evidence")
    for prefix in ("result", "raw"):
        relative = observation.get(f"{prefix}_path")
        expected = observation.get(f"{prefix}_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("v1 task-0 evidence record is incomplete")
        evidence_path = (root / relative).resolve()
        try:
            evidence_path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("v1 task-0 evidence path escaped repository") from error
        if not evidence_path.is_file() or sha256_file(evidence_path) != expected:
            raise ValueError(f"v1 task-0 {prefix} evidence hash mismatch")
    result = json.loads((root / str(observation["result_path"])).read_text())
    residuals = result.get("direct_residual_norms")
    if not isinstance(residuals, list) or len(residuals) != 32:
        raise ValueError("v1 task-0 evidence lacks its 32 residuals")
    limit = float(observation.get("registered_direct_residual_limit", -1.0))
    leading = 0
    for value in residuals:
        if float(value) > limit:
            break
        leading += 1
    if (
        result.get("task", {}).get("task_id") != observation.get("task_id")
        or int(result.get("niterations", -1)) != observation.get("max_iterations")
        or int(result.get("task", {}).get("requested_eigenpairs", -1))
        != observation.get("requested_eigenpairs")
        or leading
        != observation.get(
            "leading_consecutive_roots_passing_registered_direct_residual_gate"
        )
        or float(max(residuals[:leading]))
        != observation.get("maximum_direct_residual_in_passing_prefix")
        or int(observation.get("first_failing_root_zero_based_index", -1))
        != leading
        or float(max(residuals)) != observation.get("largest_direct_residual")
        or float(result.get("elapsed_seconds", -1.0))
        != observation.get("elapsed_seconds")
    ):
        raise ValueError("v1 task-0 observation does not match archived RESULT")

    overrides = _require_mapping(raw.get("registered_overrides"), "overrides")
    if set(overrides) != {
        "spectrum_tasks",
        "spin_classification",
        "canary_gate",
        "remote_resources",
    }:
        raise ValueError("v2a contains an unregistered override block")
    spectrum_override = _require_mapping(
        overrides["spectrum_tasks"], "spectrum override"
    )
    memory = _require_mapping(spectrum_override.get("memory_model"), "memory model")
    if set(spectrum_override) != {
        "requested_eigenpairs",
        "analysis_prefix",
        "reseed_comparison_eigenpairs",
        "max_iterations",
        "memory_model",
    }:
        raise ValueError("v2a spectrum override changed scope")
    if (
        spectrum_override.get("requested_eigenpairs") != 20
        or spectrum_override.get("analysis_prefix") != 16
        or spectrum_override.get("reseed_comparison_eigenpairs") != 20
        or spectrum_override.get("max_iterations") != 640
    ):
        raise ValueError("v2a numerical retry parameters changed")
    expected_basis_gib = 21029820 * 640 * 8 / (1024**3)
    expected_block_gib = 21029820 * 20 * 8 / (1024**3)
    if (
        int(memory.get("coefficient_bytes", 0)) != 8
        or int(memory.get("retained_krylov_vectors", 0)) != 640
        or abs(float(memory.get("nup18_krylov_basis_gib", -1)) - expected_basis_gib)
        > 5.0e-5
        or abs(
            float(memory.get("nup18_single_20_vector_block_gib", -1))
            - expected_block_gib
        )
        > 5.0e-5
        or abs(
            float(memory.get("possible_inplace_trim_transient_gib", -1))
            - expected_basis_gib
        )
        > 5.0e-5
        or int(memory.get("minimum_slurm_memory_gib", 0)) != 256
    ):
        raise ValueError("v2a memory model changed or is inconsistent")

    spin_override = _require_mapping(
        overrides["spin_classification"], "spin-classification override"
    )
    if set(spin_override) != {"conditional_extension"} or (
        "never silently widen v2a" not in str(spin_override["conditional_extension"])
    ):
        raise ValueError("v2a extension policy changed")
    canary_override = _require_mapping(overrides["canary_gate"], "canary override")
    if set(canary_override) != {
        "purpose",
        "task_id",
        "required_time_review",
        "resource_review_marker",
        "resource_evidence_copy",
        "resource_sacct_copy",
        "sacct_columns",
        "sacct_solver_step_suffix",
        "sacct_command_template",
        "maximum_elapsed_seconds",
        "maximum_max_rss_gib",
        "paired_task_index",
        "paired_task_id",
        "paired_marker",
    } or (
        canary_override.get("task_id") != "g0p09_nup18_seed42017"
        or canary_override.get("required_time_review")
        != "Elapsed leaves margin before USR1 at 07:45:00"
        or canary_override.get("resource_review_marker")
        != "CANARY_RESOURCE_REVIEW.json"
        or canary_override.get("resource_evidence_copy")
        != "CANARY_RESOURCE_EVIDENCE.json"
        or canary_override.get("resource_sacct_copy") != "CANARY_SACCT.txt"
        or canary_override.get("sacct_columns")
        != [
            "JobID",
            "JobIDRaw",
            "JobName",
            "State",
            "ExitCode",
            "ElapsedRaw",
            "AllocCPUS",
            "ReqMem",
            "MaxRSS",
        ]
        or canary_override.get("sacct_solver_step_suffix") != ".0"
        or canary_override.get("sacct_command_template")
        != (
            "LC_ALL=C sacct -j {slurm_array_job_id}_{slurm_array_task_id} "
            "--units=K --parsable2 --format=JobID,JobIDRaw,JobName,State,"
            "ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS"
        )
        or canary_override.get("maximum_elapsed_seconds") != 21600
        or canary_override.get("maximum_max_rss_gib") != 160.0
        or canary_override.get("paired_task_index") != 2
        or canary_override.get("paired_task_id")
        != "g0p09_nup19_seed42017"
        or canary_override.get("paired_marker")
        != "PAIR_CLASSIFICATION_PASS.json"
    ):
        raise ValueError("v2a canary contract changed")
    resource_override = _require_mapping(
        overrides["remote_resources"], "resource override"
    )
    if (
        resource_override.get("wall_time_per_task") != "08:00:00"
        or resource_override.get("array_concurrency") != 1
        or resource_override.get("conservative_node_hours_if_serial") != 64
    ):
        raise ValueError("v2a remote resource contract changed")
    run = _require_mapping(raw.get("run_contract"), "run_contract")
    if (
        run.get("run_namespace") != RUN_NAMESPACE
        or run.get("stable_run_id_required") is not True
        or run.get("checkpoint_granularity")
        != "completed_coupling_sector_seed_task"
        or run.get("iteration_level_restart") is not False
        or run.get("task0_only_before_operator_review") is not True
        or run.get("remote_submission_authorized") is not True
        or run.get("authorized_phase_before_operator_review")
        != "task0_only_after_precheck_and_exact_small_gate"
    ):
        raise ValueError("v2a run/checkpoint contract changed")

    config = deepcopy(load_v1_config(parent_path))
    config["stage"] = STAGE
    config["registration_status"] = str(raw["registration_status"])
    config["purpose"] = str(raw["purpose"])
    config["scientific_scope"] = str(raw["scientific_scope"])
    config["parent_contract"] = dict(parent)
    config["reused_implementation"] = [dict(item) for item in reused]
    config["v1_task0_observation"] = deepcopy(observation)
    config["run_contract"] = dict(run)
    config["decision_contract"] = deepcopy(raw["decision_contract"])
    for key, value in spectrum_override.items():
        config["spectrum_tasks"][key] = deepcopy(value)
    config["spin_classification"].update(spin_override)
    config["canary_gate"]["purpose"] = str(canary_override["purpose"])
    for key in (
        "task_id",
        "resource_review_marker",
        "resource_evidence_copy",
        "resource_sacct_copy",
        "sacct_columns",
        "sacct_solver_step_suffix",
        "sacct_command_template",
        "maximum_elapsed_seconds",
        "maximum_max_rss_gib",
        "paired_task_index",
        "paired_task_id",
        "paired_marker",
    ):
        config["canary_gate"][key] = deepcopy(canary_override[key])
    reviews = list(config["canary_gate"]["required_operator_review"])
    reviews[1] = str(canary_override["required_time_review"])
    config["canary_gate"]["required_operator_review"] = reviews
    config["remote_resources"].update(resource_override)

    spectrum = config["spectrum_tasks"]
    if (
        spectrum["couplings"] != [0.09, 0.125]
        or spectrum["n_up_values"] != [18, 19]
        or spectrum["seeds"] != [42017, 78193]
        or spectrum["solver"] != "xdiag_lanczos_pro_fixed_cap_direct_residual"
        or spectrum["allowed_criteria"] != ["maxiterations"]
        or spectrum["maximum_direct_residual"] != 1e-8
        or config["remote_resources"]["memory"] != "256G"
        or config["remote_resources"]["cpus_per_task"] != 40
    ):
        raise ValueError("v2a changed a frozen v1 physical or numerical invariant")
    tasks = task_table(config)
    if len(tasks) != 8 or tasks[0].task_id != "g0p09_nup18_seed42017":
        raise ValueError("v2a task product or canary identity changed")
    return config


__all__ = [
    "PARENT_CONFIG_RELATIVE",
    "PARENT_CONFIG_SHA256",
    "RUN_NAMESPACE",
    "STAGE",
    "load_and_validate_config",
]
