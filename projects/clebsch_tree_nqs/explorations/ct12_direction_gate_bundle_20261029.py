"""Seal one same-parent CT12 direction gate and its conditional eight-step pilot.

This builder performs no SSH, GPU query or Slurm submission.  The independently
audited global37 parent and the old production implementations are immutable.
"""
from __future__ import annotations

import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/remote/ct12_direction_gate_20261029"
LOCAL_PARENT = ROOT / "projects/clebsch_tree_nqs/results/ct_sector_physical_seed_20261029/ct12_completed"
REMOTE_PARENT = "/home/yx789/nqs-wavefunctions/nqs/ct12_production32_20261028/j0125/run01"
WORKSPACE = "/home/yx789/nqs-ct12-kim-direction-gate-20261029"
OUTPUT = "/home/yx789/nqs-wavefunctions/nqs/ct12_kim_direction_gate_20261029/j0125/run01"
PARENT_FILES = (
    "checkpoint.pkl", "registration.json", "endpoint_frozen.json", "audit_result.json",
    "terminal_audit/audit.json", "terminal_audit/untouched_products.npz",
)


def run() -> dict:
    """Prepare, validate and hash the finite 6h gate plus conditional 14h pilot."""
    from projects.clebsch_tree_nqs.workflows import ct12_direction_gate as workflow

    CONFIG.mkdir(parents=True, exist_ok=True)
    registration = json.loads((LOCAL_PARENT / "registration.json").read_text())
    old_plan = registration["plan"]
    plan = dict(
        task=workflow.TASK, execution_authorized=True, kind="ct12", device="A100", j2=.125,
        model=old_plan["model"], active_profile=old_plan["active_profile"],
        active_real_parameters=1495080,
        parent_directory=REMOTE_PARENT, parent_global_step=37,
        parent_seals={name: file_sha256(LOCAL_PARENT / name) for name in PARENT_FILES},
        parent_parameter_sha256="382fe5fe76a35a675434ed36d5336bce0f0562586568b63f639b520ac595238e",
        damping=.5, learning_rate=.03, epsilon=.125, target_FS=.003, maximum_FS=.003,
        fit_records=1024, calibration_records=1024, calibration_split=512,
        bilateral_records=512, bilateral_groups=32,
        confirmation_records=1024, confirmation_groups=32,
        audit_records=4096, audit_groups=64, steps=8,
        projection_chunk=4, action_chunk=4, score_chunk=1, H_chunk=2,
        gate_wall_cap_seconds=21600, pilot_wall_cap_seconds=50400,
        maximum_peak_device_bytes=70 * 2**30,
        seeds={name: 3930295001 + i for i, name in enumerate((
            "gate_stream", "bilateral", "confirmation", "confirmation_statistics",
            "audit", "audit_statistics"))},
        physical_state="unchanged strict-SU2 singlet, complete coherent Gamma-A1 and full physical H",
        no_requeue=True, automatic_next_allocation=False,
        authorization_note=("User2026-10-29: one full direction confirmation at identical frozen parent and width; "
            "only a passed, replayed confirmation permits eight new full-network updates. "
            "No B200 production, width change, node binding or alteration of CT6 jobs."),
        gate_candidate_committed=False,
        adaptation="fixed gate-validated controls; not a fully adaptive production experiment",
        failure_policy="preserve evidence and refuse pilot; no resampling to force passage",
    )
    workflow.validate_plan(plan)
    # The latest saved 32-step history supplies the actual used native roles,
    # not just the five externally named seeds in the old registration.
    history_path = (ROOT / "projects/clebsch_tree_nqs/results/ct_status_20261029_1020/"
        "snapshot_final/snapshot.json")
    history = json.loads(history_path.read_text())["runs"]["ct12_j0125"]["steps"]
    historical_roles = set(old_plan["seeds"].values())
    historical_transitions = {}
    for name, item in history.items():
        if name.endswith("/transition.json"):
            historical_roles.update(item["document"]["seeds"])
            historical_transitions[name] = item["sha256"]
    parent_state, _ = workflow.restart.load_state(LOCAL_PARENT / "checkpoint.pkl")
    stream_check = workflow.separate_role_streams(plan, parent_state[2], historical_roles)
    if set(stream_check["future_native_role_seeds"]) & historical_roles:
        raise ValueError("Future pilot reuses a historical native role")
    plan_path = CONFIG / "j0125.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("An existing sealed CT12 direction-gate plan differs")
    atomic_json(plan_path, plan)
    runtime = {
        "jax": "0.11.1", "jaxlib": "0.11.1", "numpy": "2.5.3", "scipy": "1.18.1",
        "jax-cuda12-plugin": "0.11.1", "jax-cuda12-pjrt": "0.11.1",
        "nvidia-cublas-cu12": "12.9.2.10", "nvidia-cuda-runtime-cu12": "12.9.79",
    }
    atomic_json(CONFIG / "runtime_packages.json", runtime)
    for name in ("run.sbatch", "verify_deployment.py"):
        if not (CONFIG / name).is_file():
            raise FileNotFoundError("Reviewed launcher/verifier required: " + name)
    entries = [Path(workflow.__file__), Path(__file__), ROOT / "tests/test_ct12_direction_gate.py"]
    sources = transitive_local_source_manifest(entries, root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    files = set(sources) | {"AGENTS.md"}
    # The focused test deliberately reads the unmodified historical profile.
    files.add("configs/remote/ct12_production32_20261028/j0125.json")
    files.update(str((CONFIG / name).relative_to(ROOT)) for name in (
        "j0125.json", "runtime_packages.json", "run.sbatch", "verify_deployment.py"))
    manifest = {name: file_sha256(ROOT / name) for name in sorted(files)}
    atomic_json(CONFIG / "source_manifest.json", manifest)
    (CONFIG / "sync_files.txt").write_text("\n".join(sorted(
        files | {str((CONFIG / "source_manifest.json").relative_to(ROOT))})) + "\n")
    report = dict(task=plan["task"], workspace=WORKSPACE, output=OUTPUT,
        parent_global_step=37, active_real_parameters=1495080, requested_node=None,
        gate_requested_A100_hours=6, conditional_pilot_requested_A100_hours=14,
        total_new_requested_A100_hours_upper=20, unchanged_existing_jobs=True,
        source_manifest_sha256=file_sha256(CONFIG / "source_manifest.json"),
        dependency="afterany; mandatory passed scientific and replay certificate before pilot",
        gate_proposal_is_not_a_training_commit=True, final_audit_records=4096,
        historical_seed_check=dict(snapshot_sha256=file_sha256(history_path),
            transition_seals=historical_transitions, distinct_historical_roles=len(historical_roles),
            gate_pilot_terminal_roles_disjoint_from_last32_update_history=True),
        deployment_files=len(files) + 1)
    result = ROOT / "projects/clebsch_tree_nqs/results/ct12_direction_gate_20261029"
    result.mkdir(parents=True, exist_ok=True)
    atomic_json(result / "prepared_bundle.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
