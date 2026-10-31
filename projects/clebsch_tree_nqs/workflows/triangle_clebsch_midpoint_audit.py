"""Independent, audit-only CT6 midpoint reads of immutable training snapshots.

This controller never invokes a transition, changes a training checkpoint, or
pretends that a midpoint completed the registered 128-update endpoint.  The
physical amplitude, full Hamiltonian, sampler, and moment estimator are reused
from the original sealed production deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as production
from projects.clebsch_tree_nqs.workflows import triangle_optimizer_comparison as shared

AUDIT_SPEC = dict(samples=16384, chains=256, refresh_proposals=1440, spacing_proposals=36)


def capture_checkpoint(source: Path, destination: Path, registration: dict) -> dict:
    """Copy one open inode; a concurrent atomic live-path replacement is safe.

    Destination is exclusive, never overwritten, and made read-only after
    checking complete parameter/optimizer/RNG seals.  Failed captures remain
    available for diagnosis and are never silently reused.
    """
    seal = registration['sha256']
    if canonical_sha256({k: v for k, v in registration.items() if k != 'sha256'}) != seal:
        raise ValueError('source registration is not self-consistent')
    if source.resolve() == destination.resolve():
        raise ValueError('snapshot cannot replace the live checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as inp, destination.open('xb') as out:
        before = os.fstat(inp.fileno())
        while data := inp.read(1024 * 1024):
            out.write(data)
            digest.update(data)
        out.flush()
        os.fsync(out.fileno())
        after = os.fstat(inp.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('opened source inode changed in place during capture')
    if destination.stat().st_size != before.st_size or file_sha256(destination) != digest.hexdigest():
        raise ValueError('snapshot byte/hash verification failed')
    state, metadata = shared.restore_state(destination, seal)
    validate_counters(metadata, state)
    destination.chmod(0o444)
    return dict(source_path=str(source), snapshot_path=str(destination),
        captured_utc=datetime.now(timezone.utc).isoformat(), bytes=before.st_size,
        source_open_inode=before.st_ino, source_open_mtime_ns=before.st_mtime_ns,
        checkpoint_sha256=digest.hexdigest(), registration_sha256=seal,
        source_plan_sha256=registration['plan_sha256'],
        params_sha256=metadata['params_sha256'], optimizer_sha256=metadata['optimizer_sha256'],
        sampler_rng_sha256=metadata['sampler_rng_sha256'],
        completed_steps=metadata['completed_steps'],
        absolute_completed_steps=metadata['absolute_completed_steps'],
        parent_absolute_completed_steps=metadata['parent_absolute_completed_steps'],
        optimizer_base_step=metadata['optimizer_base_step'],
        snapshot_selection='current complete checkpoint at preregistered capture time, not energy selected')


def validate_counters(metadata: dict, state: tuple) -> None:
    """Retain actual midpoint and lifetime counters, including optimizer state."""
    n = metadata['completed_steps']
    if (isinstance(n, bool) or not isinstance(n, int) or not 2 <= n < 128
            or n != len(metadata['history'])
            or metadata['absolute_completed_steps'] != metadata['parent_absolute_completed_steps'] + n
            or int(state[1][0]) != metadata['optimizer_base_step'] + n):
        raise ValueError('actual midpoint/history/optimizer counters disagree')


def validate_plan(plan: dict, source_plan: dict) -> None:
    """Release only the two bounded, fresh-chain, same-state midpoint audits."""
    if plan['task'] != 'ct6_immutable_midpoint_audit_v1' or plan['audit'] != AUDIT_SPEC:
        raise ValueError('audit protocol must remain 16384 samples on 256 fresh chains')
    if (plan['maximum_seconds'] != 5400 or plan['requested_A100_hours'] != 1.5
            or plan['campaign_total_A100_hours'] != 240
            or not math.isfinite(plan['aggregate_reserved_A100_hours'])
            or not 3 <= plan['aggregate_reserved_A100_hours'] <= plan['campaign_total_A100_hours']):
        raise ValueError('midpoint audit allocation exceeds its approved fixed budget')
    if not math.isfinite(plan['cost_forecast_seconds']) or not 0 < plan['cost_forecast_seconds'] <= 4500:
        raise ValueError('audit requires a finite measured cost forecast within the allocation')
    production.validate_plan(source_plan)
    if (source_plan['model'] != dict(length=6, multiplicity_cap=28, maximum_q=10)
            or source_plan['task'] != production.WIDE_PRODUCTION_TASK
            or source_plan['j2_over_j1'] not in (0., .125)
            or source_plan['training']['steps'] != 128):
        raise ValueError('only existing CT6 width28 production states are in scope')
    seed = plan['seed']
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            or seed in plan['forbidden_seeds'] or seed in source_plan['seeds'].values()):
        raise ValueError('fresh uint32 audit seed must not reuse earlier streams')
    if canonical_sha256(source_plan) != plan['source_plan_sha256']:
        raise ValueError('source physical/training plan changed')
    if plan['snapshot']['source_plan_sha256'] != plan['source_plan_sha256']:
        raise ValueError('snapshot registration belongs to another source plan')
    live_output = Path(source_plan['output']).resolve()
    if Path(plan['snapshot']['snapshot_path']).resolve().is_relative_to(live_output):
        raise ValueError('audit must not read the mutable training checkpoint')
    if Path(plan['output']).resolve().is_relative_to(live_output):
        raise ValueError('audit output must not change the training directory')


def verify_sources(plan: dict) -> None:
    """Verify the original frozen deployment, not the evolving local worktree."""
    root = Path(plan['source_root'])
    manifest_path = root / plan['source_manifest_path']
    if file_sha256(manifest_path) != plan['source_manifest_sha256']:
        raise ValueError('sealed production manifest changed')
    for name, digest in json.loads(manifest_path.read_text()).items():
        if file_sha256(root / name) != digest:
            raise ValueError('sealed production file changed: ' + name)
    if file_sha256(Path(__file__)) != plan['wrapper_sha256']:
        raise ValueError('audit-only wrapper changed')
    if not Path(production.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError('production runtime imported from the wrong deployment')


def run(plan: dict, plan_sha256: str) -> dict:
    """Audit one frozen midpoint; no optimizer call, no training gate bypass."""
    started = perf_counter()
    if canonical_sha256(plan) != plan_sha256:
        raise ValueError('midpoint audit registration changed')
    source_plan = json.loads(Path(plan['source_plan_path']).read_text())
    validate_plan(plan, source_plan)
    verify_sources(plan)
    if datetime.now(timezone.utc) + timedelta(seconds=plan['maximum_seconds']) > datetime.fromisoformat(plan['campaign_deadline_utc']):
        raise RuntimeError('audit allocation would exceed the unchanged campaign deadline')
    jax.config.update('jax_enable_x64', True)
    if (jax.default_backend() != 'gpu' or len(jax.devices()) != 1
            or 'A100' not in jax.devices()[0].device_kind):
        raise RuntimeError('exactly one approved A100 allocation required')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise FileExistsError('completed audit is immutable; no replacement or repeat selection')
    snapshot = plan['snapshot']
    path = Path(snapshot['snapshot_path'])
    if file_sha256(path) != snapshot['checkpoint_sha256']:
        raise ValueError('immutable midpoint bytes changed')
    state, metadata = shared.restore_state(path, snapshot['registration_sha256'])
    validate_counters(metadata, state)
    for field in ('params_sha256', 'optimizer_sha256', 'sampler_rng_sha256',
                  'completed_steps', 'absolute_completed_steps', 'parent_absolute_completed_steps'):
        if metadata[field] != snapshot[field]:
            raise ValueError('frozen midpoint identity changed: ' + field)
    if metadata['registration_sha256'] != snapshot['registration_sha256']:
        raise ValueError('snapshot source registration changed')
    rt = production.runtime(source_plan, state[0])
    if len(rt['images']) != 432 or len(rt['bonds']) != (108 if source_plan['j2_over_j1'] == 0 else 216):
        raise ValueError('full A1/full Hamiltonian contract changed')
    atomic_json(output / 'started.json', dict(plan_sha256=plan_sha256, snapshot=snapshot,
        started_utc=datetime.now(timezone.utc).isoformat(),
        device_kinds=[d.device_kind for d in jax.devices()], jax_version=jax.__version__,
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        optimizer_updates_committed=0, complete_A1_images=432, complete_H_bonds=len(rt['bonds']),
        estimated_seconds=plan['cost_forecast_seconds'], maximum_seconds=plan['maximum_seconds']))
    if plan['cost_forecast_seconds'] + 600 + 90 > plan['maximum_seconds'] - (perf_counter() - started):
        raise RuntimeError('measured audit forecast plus compile/save margin exceeds fixed allocation')
    audit = shared.terminal_audit(rt, state[0], plan['audit'], plan['seed'], output)
    if file_sha256(path) != snapshot['checkpoint_sha256'] or pytree_sha256(state[0]) != snapshot['params_sha256']:
        raise ValueError('immutable audit state changed during evaluation')
    result = dict(status='midpoint_audit_complete', plan_sha256=plan_sha256,
        snapshot=snapshot, j2_over_j1=source_plan['j2_over_j1'], arity=source_plan['arity'],
        model=source_plan['model'], audit=audit, optimizer_updates_committed=0,
        complete_A1_images=432, complete_H_bonds=len(rt['bonds']),
        audit_full_H_centers=plan['audit']['samples'], additional_H_preflight_centers=0,
        independent_initial_chains=plan['audit']['chains'], temporal_records_per_chain=64,
        sampling_proposals_per_chain=3744, fresh_initial_amplitude_centers=256,
        proposal_amplitude_centers=958464,
        variance_node_complete=False, variance_scope='Born local-energy second moment; exact-node contributions not certified',
        mixing_certified=False, endpoint_selection='capture time only, no energy-based selection',
        elapsed_seconds=perf_counter() - started,
        finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / 'results.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.plan.read_text()), args.plan_sha256)), flush=True)


if __name__ == '__main__':
    main()
