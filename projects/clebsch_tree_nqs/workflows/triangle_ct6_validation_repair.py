"""Train-only, sealed validation repair for an already replayed CT6 v8 state.

Keep the original scientific registration and numerical trajectory. Permit
only the named replay-validation change after checking old/new workflow ASTs,
all numerical dependencies, runtime identity, and original artifact seals.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from . import triangle_clebsch_continuation as workflow

REPAIR_TYPE = 'CT6_v8_replay_gate_bounds_validation_only_v1'
WORKFLOW_PATH = 'projects/clebsch_tree_nqs/workflows/triangle_clebsch_continuation.py'
VALIDATION_FUNCTIONS = frozenset({'checked_direction_release', 'direction_gate_replay_matches'})
SEALED_ARTIFACTS = frozenset({'registration.json', 'checkpoint.pkl', 'pilot.json',
    'process_replay.json', 'direction_check.json', 'direction_check.npz',
    'next_transition_fixture.pkl', 'preflight.json', 'sampler_branch.json'})


def assert_validation_only_source_change(original: str, repaired: str) -> None:
    """Require every AST node outside the two named validators to be equal."""
    def stripped(source: str) -> ast.Module:
        module = ast.parse(source)
        module.body = [node for node in module.body if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in VALIDATION_FUNCTIONS)]
        return module

    if ast.dump(stripped(original), include_attributes=False) != ast.dump(
            stripped(repaired), include_attributes=False):
        raise ValueError('repair changes code outside the two replay validators')


def validate_execution_identity(original: dict, current: dict) -> None:
    """Keep plan, physical parent, node/runtime, and other source hashes exact."""
    if set(original) != set(current):
        raise ValueError('repair execution identity fields changed')
    if canonical_sha256({k: v for k, v in original.items() if k != 'sources'}) != canonical_sha256(
            {k: v for k, v in current.items() if k != 'sources'}):
        raise ValueError('repair changed scientific plan, parent, or numerical runtime')
    old_sources, new_sources = original['sources'], current['sources']
    if (set(old_sources) != set(new_sources) or any(old_sources[k] != new_sources[k]
            for k in old_sources if k != WORKFLOW_PATH)):
        raise ValueError('repair changed a numerical source dependency')


def checked_repair_record(manifest: dict, manifest_sha256: str, *, plan: dict,
                          original: dict, current: dict) -> dict:
    """Verify the explicit repair seal, then return immutable execution provenance."""
    if canonical_sha256(manifest) != manifest_sha256 or manifest.get('repair_type') != REPAIR_TYPE:
        raise ValueError('explicit validation repair manifest seal/type mismatch')
    if (manifest.get('plan_sha256') != canonical_sha256(plan)
            or manifest.get('registration_sha256') != canonical_sha256(original)
            or manifest.get('launcher_sha256') != file_sha256(Path(__file__))
            or set(manifest.get('original_artifacts', {})) != SEALED_ARTIFACTS):
        raise ValueError('repair manifest does not seal its plan, registration, launcher and artifacts')
    previous = Path(manifest['original_workflow_snapshot'])
    current_path = Path(workflow.__file__)
    if (file_sha256(previous) != manifest['original_workflow_sha256']
            or file_sha256(current_path) != manifest['repaired_workflow_sha256']
            or original['sources'][WORKFLOW_PATH] != manifest['original_workflow_sha256']
            or current['sources'][WORKFLOW_PATH] != manifest['repaired_workflow_sha256']):
        raise ValueError('repair source snapshots do not match registered hashes')
    assert_validation_only_source_change(previous.read_text(), current_path.read_text())
    validate_execution_identity(original, current)
    return dict(repair_type=REPAIR_TYPE, manifest_sha256=manifest_sha256, manifest=manifest,
        original_registration_sha256=canonical_sha256(original),
        actual_execution_identity_sha256=canonical_sha256(current),
        actual_sources=current['sources'], actual_numerical_runtime=current['numerical_runtime'],
        allowed_changed_functions=sorted(VALIDATION_FUNCTIONS),
        original_registration_preserved=True, original_pilot_and_replay_preserved=True,
        numerical_method_changed=False, new_direction_draw=False, optimizer_reset=False)


def run(plan: dict, plan_sha256: str, manifest: dict, manifest_sha256: str) -> dict:
    """Resume exactly the sealed state; only the training phase is exposed."""
    started = perf_counter()
    workflow.validate_plan(plan)
    if plan['task'] != workflow.EVIDENCE_GUARDED_TASK or canonical_sha256(plan) != plan_sha256:
        raise ValueError('repair requires the unchanged registered CT6 v8 plan')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(plan['campaign_deadline_utc']):
        raise RuntimeError('original campaign deadline reached')
    jax.config.update('jax_enable_x64', True)
    if plan['require_gpu']:
        devices = jax.local_devices()
        if (jax.default_backend() != 'gpu' or len(devices) != 1
                or 'A100' not in devices[0].device_kind):
            raise RuntimeError('exactly one allocated A100 required')
    output = Path(plan['output'])
    original, registration = workflow._sealed_json(output/'registration.json')
    original = {k: v for k, v in original.items() if k != 'sha256'}
    parent, lineage = workflow.load_parent(plan)
    current = dict(plan_sha256=plan_sha256, case=plan['case'], stage=plan['stage'],
        arity=plan['arity'], model=plan['model'], j2_over_j1=plan['j2_over_j1'],
        binary_rank=plan['binary_rank'], parent=lineage,
        sources=transitive_local_source_manifest((Path(workflow.__file__),), root=workflow.ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
        numerical_runtime=workflow.numerical_runtime_fingerprint())
    if current['numerical_runtime']['node'] != plan['runtime_contract']['node']:
        raise ValueError('repair must retain the original pinned node')
    record = checked_repair_record(manifest, manifest_sha256, plan=plan, original=original, current=current)
    for name, digest in manifest['original_artifacts'].items():
        if name != 'checkpoint.pkl' and file_sha256(output/name) != digest:
            raise ValueError('sealed original repair artifact changed: '+name)
    repair_path = output/'execution_repair.json'
    if repair_path.exists() and json.loads(repair_path.read_text()) != record:
        raise ValueError('immutable validation repair provenance changed')
    state, meta = workflow.shared.restore_state(output/'checkpoint.pkl', registration)
    completed = meta['completed_steps']
    if (not isinstance(completed, int) or not 0 <= completed <= plan['training']['steps']
            or completed != len(meta['history'])
            or meta['absolute_completed_steps'] != meta['parent_absolute_completed_steps']+completed
            or int(state[1][0]) != meta['optimizer_base_step']+completed
            or np.shape(state[3]) != (256, 36) or np.shape(state[5]) != (128, 36)
            or meta.get('sampler_branch_sha256') != file_sha256(output/'sampler_branch.json')):
        raise ValueError('repair checkpoint optimizer/sampler/history counters differ')
    if completed == 0:
        if file_sha256(output/'checkpoint.pkl') != manifest['original_artifacts']['checkpoint.pkl']:
            raise ValueError('repair zero-update checkpoint differs from the sealed pilot state')
    elif not repair_path.exists() or meta.get('execution_repair_sha256') != file_sha256(repair_path):
        raise ValueError('partial continuation is not from this sealed validation repair')
    # Validate the existing pilot/replay and raw confirmation BEFORE compiling
    # training kernels or committing any optimizer update. No new samples here.
    workflow.checked_direction_release(output, meta, plan, registration)
    if not repair_path.exists():
        atomic_json(repair_path, record)
    meta['execution_repair_sha256'] = file_sha256(repair_path)
    runtime = workflow.runtime(plan, parent[0])
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_stop)
    result = workflow.run_direction_phases(plan, 'train', state, meta, runtime, output,
        registration, lineage, stop, started)
    if result.get('status') == 'fixed_steps_complete':
        result['execution_repair_sha256'] = file_sha256(repair_path)
        result['numerical_method_changed_by_repair'] = False
        atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--repair-manifest', type=Path, required=True)
    parser.add_argument('--repair-manifest-sha256', required=True)
    args = parser.parse_args()
    answer = run(json.loads(args.plan.read_text()), args.plan_sha256,
                 json.loads(args.repair_manifest.read_text()), args.repair_manifest_sha256)
    print(json.dumps({k: v for k, v in answer.items() if k != 'history'}), flush=True)
    if answer['status'] == 'safe_pause':
        raise SystemExit(75)
