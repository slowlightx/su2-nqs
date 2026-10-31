"""Seal a corrected CT12 A100 runtime-budget retry, not a new physics trial.

The failed original attempt stopped before its fit draw.  Its receipts, source
workspace and zero-update parent remain immutable; the retry uses a new output
and the same unconsumed statistical role streams.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/remote/ct12_direction_gate_retry_20261029"
RESULT = ROOT / "projects/clebsch_tree_nqs/results/ct12_direction_gate_retry_20261029"
WORKSPACE = "/home/yx789/nqs-ct12-kim-direction-gate-retry-20261029"
OUTPUT = "/home/yx789/nqs-wavefunctions/nqs/ct12_kim_direction_gate_retry_20261029/j0125/run01"


def run() -> dict:
    """Prepare one10h direction check and a certificate-gated18h short pilot."""
    from projects.clebsch_tree_nqs.workflows import ct12_direction_gate as workflow

    CONFIG.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    old_config = ROOT / "configs/remote/ct12_direction_gate_20261029"
    original = json.loads((old_config / "j0125.json").read_text())
    evidence = RESULT / "previous_attempt"
    old_stop = json.loads((evidence / "gate_result.json").read_text())
    stage = json.loads((evidence / "read_only_stage_check.json").read_text())
    if (old_stop["committed_updates"] != 0
            or "insufficient reserved time" not in old_stop["error"]
            or not stage["gate_stopped_before_fit_and_confirmation"]
            or any(stage[name] for name in ("fit_draw_exists", "fit_products_exists", "provisional_candidate_exists"))):
        raise ValueError("Only the documented zero-fit budget stop can reuse its unconsumed roles")
    p = deepcopy(original)
    p.update(runtime_budget_profile="runtime_budget_retry_v1",
        gate_wall_cap_seconds=36000, pilot_wall_cap_seconds=64800,
        projection_chunk=16, H_chunk=8, action_chunk=4, score_chunk=1,
        authorization_note=("User2026-10-29 explicitly requests corrected CT12 time-budget/query-batch "
            "resubmission. Same global37 parent, active widths, optimizer/RNG, fullA1/H, scientific "
            "record counts and passed-certificate requirement. At most10A100h gate plus18A100h "
            "conditional eight-step pilot; no B200 production or CT6 mutation."),
        budget_retry_evidence=dict(zero_fit_records_consumed=True, previous_gate_job_id=797344,
            previous_preflight_sha256=file_sha256(evidence / "gate_cost_preflight.json"),
            previous_gate_result_sha256=file_sha256(evidence / "gate_result.json"),
            previous_replay_sha256=file_sha256(evidence / "device_replay.json"),
            same_unconsumed_role_streams_retained=True),
        runtime_batch_basis=dict(
            A100_reference="projects/clebsch_tree_nqs/results/remote_status_20261027_2024/ct12_j0125",
            same_active_profile=True, projection_chunk=16, H_chunk=8, fit_records_in_reference=512,
            measured_reference_warm_H_seconds_per_center=1.0288537955281163,
            measured_reference_peak_device_bytes=6320002304,
            reference_is_not_a_new1024row_or_JVP_memory_certificate=True),
    )
    workflow.validate_plan(p, executing=True)
    invariant_keys = ("parent_directory", "parent_global_step", "parent_seals", "parent_parameter_sha256",
        "active_real_parameters", "model", "active_profile", "j2", "damping", "learning_rate",
        "epsilon", "target_FS", "maximum_FS", "fit_records", "calibration_records",
        "calibration_split", "bilateral_records", "bilateral_groups", "confirmation_records",
        "confirmation_groups", "audit_records", "audit_groups", "steps", "seeds",
        "maximum_peak_device_bytes")
    if any(p[k] != original[k] for k in invariant_keys):
        raise ValueError("A runtime-only retry cannot alter physical or statistical invariants")
    path = CONFIG / "j0125.json"
    if path.exists() and json.loads(path.read_text()) != p:
        raise ValueError("Existing sealed retry plan differs")
    atomic_json(path, p)
    atomic_json(CONFIG / "runtime_packages.json", json.loads((old_config / "runtime_packages.json").read_text()))
    entries = [Path(workflow.__file__), Path(__file__), ROOT / "tests/test_ct12_direction_gate.py"]
    source = transitive_local_source_manifest(entries, root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    files = set(source) | {"AGENTS.md", "configs/remote/ct12_production32_20261028/j0125.json"}
    files.update(str((CONFIG / name).relative_to(ROOT)) for name in (
        "j0125.json", "run.sbatch", "verify_deployment.py", "runtime_packages.json"))
    files.update(str(f.relative_to(ROOT)) for f in sorted(evidence.glob("*.json")))
    manifest = {name: file_sha256(ROOT / name) for name in sorted(files)}
    atomic_json(CONFIG / "source_manifest.json", manifest)
    (CONFIG / "sync_files.txt").write_text("\n".join(sorted(files | {
        str((CONFIG / "source_manifest.json").relative_to(ROOT))})) + "\n")
    report = dict(task=p["task"], runtime_budget_profile=p["runtime_budget_profile"],
        workspace=WORKSPACE, output=OUTPUT, parent_global_step=37, active_real_parameters=1495080,
        parent_optimizer_and_next_rng_unchanged=True, scientific_invariant_keys=list(invariant_keys),
        gate_A100_hours_upper=10, conditional_pilot_A100_hours_upper=18,
        total_new_requested_A100_hours_upper=28, requested_node=None,
        old_failed_attempts_preserved=[797344, 797345],
        existing_CT6_jobs_unchanged=True, new_B200_jobs=0,
        source_manifest_sha256=file_sha256(CONFIG / "source_manifest.json"),
        deployment_files=len(files)+1, conditional_dependency="afterany plus mandatory passed scientific/replay certificate",
        no_confirmatory_or_terminal_data_consumed_by_previous_attempt=True,
        same_unconsumed_seeds_not_a_new_selected_trial=True)
    atomic_json(RESULT / "prepared_bundle.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
