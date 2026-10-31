"""Host-portable CT6 continuation with fixed-fixture replay on every allocation.

The historical scientific plan and registration remain untouched. Only the
hostname is excluded from numerical-runtime identity; all other runtime and
source fields remain sealed. Replay the original state/RNG and direction
fixture without committing an update, then train in the SAME allocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
from time import perf_counter
from typing import Iterator

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint
from . import triangle_ct6_validation_repair as repair
from . import triangle_clebsch_continuation as workflow

PORTABLE_TYPE = 'CT6_v8_hostname_portable_fixed_fixture_replay_v1'
RUNTIME_POLICY = 'any_Kim_A10080GB_same_runtime_except_hostname_fixed_replay'


def without_hostname(runtime: dict) -> dict:
    """Remove only the observed hostname, never software/device/environment."""
    if not isinstance(runtime.get('node'), str) or not runtime['node']:
        raise ValueError('numerical runtime must record its observed hostname')
    return {k: v for k, v in runtime.items() if k != 'node'}


def validate_portable_identity(original: dict, current: dict) -> None:
    """Allow hostname-only runtime variation and the already audited repair."""
    if canonical_sha256(without_hostname(original['numerical_runtime'])) != canonical_sha256(
            without_hostname(current['numerical_runtime'])):
        raise ValueError('portable execution changed non-host numerical runtime')
    compatible = deepcopy(current)
    compatible['numerical_runtime']['node'] = original['numerical_runtime']['node']
    repair.validate_execution_identity(original, compatible)


def checked_portable_record(manifest: dict, seal: str, *, plan: dict,
                            original: dict, current: dict) -> dict:
    """Seal a host-agnostic execution identity; observed hosts live in replays."""
    if (canonical_sha256(manifest) != seal or manifest.get('type') != PORTABLE_TYPE
            or manifest.get('runtime_policy') != RUNTIME_POLICY
            or manifest.get('launcher_sha256') != file_sha256(Path(__file__))
            or manifest.get('plan_sha256') != canonical_sha256(plan)
            or manifest.get('replay_budget_seconds') != 7200
            or manifest.get('train_budget_seconds') != plan['budget']['train_seconds']):
        raise ValueError('portable execution manifest seal/policy/budget mismatch')
    validate_portable_identity(original, current)
    # Reuse the prior source/AST/artifact-manifest validation, without claiming
    # its normalized hostname is the actual runtime. Its return is discarded.
    compatible = deepcopy(current)
    compatible['numerical_runtime']['node'] = original['numerical_runtime']['node']
    prior = manifest['original_repair_manifest']
    repair.checked_repair_record(prior, manifest['original_repair_manifest_sha256'],
        plan=plan, original=original, current=compatible)
    frozen = Path(manifest['frozen_checkpoint_path'])
    if (file_sha256(frozen) != prior['original_artifacts']['checkpoint.pkl']
            or frozen.resolve() == (Path(plan['output'])/'checkpoint.pkl').resolve()):
        raise ValueError('portable replay requires a separate sealed frozen checkpoint')
    return dict(type=PORTABLE_TYPE, manifest_sha256=seal, manifest=manifest,
        original_registration_sha256=canonical_sha256(original),
        runtime_policy=RUNTIME_POLICY, numerical_runtime_without_hostname=without_hostname(
            current['numerical_runtime']), actual_sources=current['sources'],
        original_registration_preserved=True, original_pilot_and_replay_preserved=True,
        scientific_plan_changed=False, optimizer_reset=False, new_direction_draw=False,
        replay_required_on_every_allocation=True)


@contextmanager
def single_writer(output: Path) -> Iterator[None]:
    """Protect portable continuations against concurrent checkpoint writers."""
    with (output/'.portable_writer.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another portable continuation owns this output') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_replay_evidence(report: dict, expected_gate: dict, actual_gate: dict,
                            tolerance: float) -> None:
    """Keep existing exact-discrete/relative-float and scientific gate checks."""
    if (not 0 < tolerance <= 2e-11
            or report.get('protocol') != 'exact_discrete_relative_float_v1'
            or report.get('tolerance') != tolerance or report.get('passed') is not True
            or not report.get('leaves') or actual_gate.get('passed') is not True
            or not workflow.direction_gate_replay_matches(expected_gate, actual_gate,
                tolerance=tolerance)):
        raise ValueError('portable fixed-fixture replay or scientific gate failed')
    for row in report['leaves']:
        if row.get('passed') is not True or row.get('finite') is not True:
            raise ValueError('portable replay contains a failed/nonfinite leaf')
        if row['discrete']:
            if row.get('bitwise_exact') is not True:
                raise ValueError('portable replay changed a discrete state/RNG decision')
        else:
            relative, absolute = row.get('relative_vector_error'), row.get('absolute_error')
            diagnostic = (row.get('natural_unit_scalar_diagnostic') and row['shape'] == []
                          and row['path'].startswith('[0][1]'))
            if not (relative is not None and math.isfinite(relative) and relative <= tolerance
                    or diagnostic and absolute is not None and math.isfinite(absolute)
                    and absolute <= tolerance):
                raise ValueError('portable replay numerical error exceeds original tolerance')


def fixed_fixture_replay(plan: dict, runtime: dict, frozen_state: tuple,
                         output: Path) -> dict:
    """Re-execute the original RNG fixture, not a new scientific confirmation."""
    started = perf_counter()
    proposed, info, timing = runtime['transition'](frozen_state)
    moments, payload = runtime['direction_confirmation'](frozen_state[0], proposed[0])
    moments = workflow.with_parent_energy_reference(moments, workflow.frozen_parent_energy_reference(plan))
    gate = workflow.registered_direction_gate(plan, moments)
    expected = load_checkpoint(output/'next_transition_fixture.pkl')['params']
    fixture = (workflow.shared.transition_fixture(proposed, info), payload)
    report = workflow.compare_replay(expected, fixture, tolerance=plan['tolerance'],
        diagnostic_prefixes=('[0][1]',))
    expected_gate = json.loads((output/'direction_check.json').read_text())['gate']
    failure = None
    try:
        require_replay_evidence(report, expected_gate, gate, plan['tolerance'])
    except ValueError as exc:
        failure = str(exc)
    peak = max(int((d.memory_stats() or {}).get('peak_bytes_in_use', 0)) for d in jax.local_devices())
    if peak > plan['budget']['maximum_peak_device_bytes'] or (plan['require_gpu'] and peak <= 0):
        failure = 'portable replay exceeds registered memory guard'
    return dict(status='failed' if failure else 'passed', failure=failure,
        report=report, independently_recomputed_gate=gate, seconds=perf_counter()-started,
        transition_timing=timing, peak_device_bytes=peak, optimizer_updates_committed=0,
        full_H_training_centers=plan['training']['samples'],
        full_H_parent_centers=plan['direction_check']['samples'],
        full_H_JVP_tangent_centers=plan['direction_check']['samples'],
        original_fixed_rng_reexecuted=True, new_scientific_confirmation_draw=False)


def run(plan: dict, plan_sha256: str, manifest: dict, manifest_sha256: str) -> dict:
    """Replay and train serially on any compatible allocated Kim A10080GB."""
    workflow.validate_plan(plan)
    if plan['task'] != workflow.EVIDENCE_GUARDED_TASK or canonical_sha256(plan) != plan_sha256:
        raise ValueError('portable continuation requires unchanged historical CT6 v8 plan')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(plan['campaign_deadline_utc']):
        raise RuntimeError('original campaign deadline reached')
    jax.config.update('jax_enable_x64', True)
    if plan['require_gpu']:
        devices = jax.local_devices()
        if (jax.default_backend() != 'gpu' or len(devices) != 1
                or 'A100' not in devices[0].device_kind or '80GB' not in devices[0].device_kind):
            raise RuntimeError('one allocated A10080GB required; hostname is unrestricted')
    output = Path(plan['output'])
    with single_writer(output):
        return _run_locked(plan, plan_sha256, manifest, manifest_sha256, output)


def _run_locked(plan: dict, plan_sha256: str, manifest: dict,
                manifest_sha256: str, output: Path) -> dict:
    original, registration = workflow._sealed_json(output/'registration.json')
    original = {k: v for k, v in original.items() if k != 'sha256'}
    parent, lineage = workflow.load_parent(plan)
    current = dict(plan_sha256=plan_sha256, case=plan['case'], stage=plan['stage'],
        arity=plan['arity'], model=plan['model'], j2_over_j1=plan['j2_over_j1'],
        binary_rank=plan['binary_rank'], parent=lineage,
        sources=transitive_local_source_manifest((Path(workflow.__file__),), root=workflow.ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
        numerical_runtime=workflow.numerical_runtime_fingerprint())
    record = checked_portable_record(manifest, manifest_sha256,
        plan=plan, original=original, current=current)
    originals = manifest['original_repair_manifest']['original_artifacts']
    for name, digest in originals.items():
        if name != 'checkpoint.pkl' and file_sha256(output/name) != digest:
            raise ValueError('sealed original portable artifact changed: '+name)
    record_path = output/'portable_execution.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError('portable execution registration changed')
    state, meta = workflow.shared.restore_state(output/'checkpoint.pkl', registration)
    completed = meta['completed_steps']
    if (not isinstance(completed, int) or not 0 <= completed <= plan['training']['steps']
            or completed != len(meta['history'])
            or meta['absolute_completed_steps'] != meta['parent_absolute_completed_steps']+completed
            or int(state[1][0]) != meta['optimizer_base_step']+completed
            or np.shape(state[3]) != (256, 36) or np.shape(state[5]) != (128, 36)
            or meta.get('sampler_branch_sha256') != file_sha256(output/'sampler_branch.json')):
        raise ValueError('portable checkpoint optimizer/sampler/history counters differ')
    if completed == 0:
        if file_sha256(output/'checkpoint.pkl') != originals['checkpoint.pkl']:
            raise ValueError('portable zero-update checkpoint differs from frozen state')
    elif (not record_path.exists()
            or meta.get('portable_execution_sha256') != file_sha256(record_path)):
        raise ValueError('partial checkpoint is not from this portable continuation')
    workflow.checked_direction_release(output, meta, plan, registration)
    if (output/'results.json').exists():
        result = json.loads((output/'results.json').read_text())
        if (completed != plan['training']['steps'] or result.get('status') != 'fixed_steps_complete'
                or result.get('completed_steps') != completed
                or result.get('checkpoint_sha256') != file_sha256(output/'checkpoint.pkl')
                or result.get('params_sha256') != meta['params_sha256']
                or result.get('terminal_audit', {}).get('params_sha256') != meta['params_sha256']
                or result.get('registration_sha256') != registration
                or result.get('portable_execution_sha256') != file_sha256(record_path)):
            raise ValueError('existing portable endpoint provenance differs')
        return dict(result, cached_verified_endpoint=True)
    frozen_state, frozen_meta = workflow.shared.restore_state(Path(manifest['frozen_checkpoint_path']), registration)
    if frozen_meta['completed_steps'] != 0:
        raise ValueError('portable fixed replay checkpoint is not the zero-update state')
    if not record_path.exists():
        atomic_json(record_path, record)
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_stop)
    runtime = workflow.runtime(plan, parent[0])
    replay = fixed_fixture_replay(plan, runtime, frozen_state, output)
    replay.update(actual_numerical_runtime=current['numerical_runtime'],
        portable_execution_sha256=file_sha256(record_path),
        frozen_checkpoint_sha256=originals['checkpoint.pkl'],
        fixture_sha256=originals['next_transition_fixture.pkl'],
        checked_utc=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if replay['seconds'] > manifest['replay_budget_seconds'] or stop[0]:
        replay.update(status='failed', failure='portable replay time budget/interruption')
    replay_path = output/'portable_replays'/(canonical_sha256(replay)+'.json')
    atomic_json(replay_path, replay)
    atomic_json(output/'portable_runtime_latest.json', dict(replay_path=str(replay_path),
        replay_sha256=file_sha256(replay_path), status=replay['status']))
    print(json.dumps(dict(portable_replay_status=replay['status'],
        observed_node=current['numerical_runtime']['node'], seconds=replay['seconds'],
        replay_sha256=file_sha256(replay_path))), flush=True)
    if replay['status'] != 'passed':
        raise ArithmeticError('portable fixed-fixture replay failed; no training started')
    meta.update(portable_execution_sha256=file_sha256(record_path),
        portable_replay_sha256=file_sha256(replay_path), portable_replay_path=str(replay_path))
    result = workflow.run_direction_phases(plan, 'train', state, meta, runtime, output,
        registration, lineage, stop, perf_counter())
    if result.get('status') == 'fixed_steps_complete':
        replay_ledger = []
        for path in sorted((output/'portable_replays').glob('*.json')):
            row = json.loads(path.read_text())
            if row.get('portable_execution_sha256') == file_sha256(record_path):
                replay_ledger.append(dict(path=str(path), sha256=file_sha256(path),
                    **{key: row[key] for key in ('status', 'full_H_training_centers',
                        'full_H_parent_centers', 'full_H_JVP_tangent_centers')}))
        result.update(portable_execution_sha256=file_sha256(record_path),
            portable_replay_sha256=file_sha256(replay_path),
            portable_replay_path=str(replay_path), numerical_method_changed_by_portability=False,
            additional_portable_replay_ledger=replay_ledger,
            additional_portable_replay_work={key: sum(row[key] for row in replay_ledger)
                for key in ('full_H_training_centers', 'full_H_parent_centers', 'full_H_JVP_tangent_centers')})
        atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--portable-manifest', type=Path, required=True)
    parser.add_argument('--portable-manifest-sha256', required=True)
    args = parser.parse_args()
    answer = run(json.loads(args.plan.read_text()), args.plan_sha256,
        json.loads(args.portable_manifest.read_text()), args.portable_manifest_sha256)
    print(json.dumps({k: v for k, v in answer.items() if k != 'history'}), flush=True)
    if answer['status'] == 'safe_pause':
        raise SystemExit(75)
