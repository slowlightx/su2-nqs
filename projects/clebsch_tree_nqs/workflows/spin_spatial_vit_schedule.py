"""Materialize fixed continuation stages only after their parent is complete.

The schedule fixes endpoints, audit seeds and time caps before any training.
Only future checkpoint hashes are filled in later; energies never select a
stage, parent or hyperparameter. Numerical training and resume use the existing
NetKet workflow unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from .spin_spatial_vit_quality import ROOT, read_sealed
from .spin_spatial_vit_netket import validate_continuation_protocol


def materialize_stage(plan: dict, index: int) -> dict:
    """Return a sealed-parent continuation without reading audit outcomes."""
    if isinstance(index, bool) or not 0 <= index < len(plan['stages']):
        raise ValueError('invalid registered stage')
    stages = plan['stages']
    outputs = [s['output'] for s in stages]
    seeds = [x for s in stages for x in s['audit_seeds']]
    if len(outputs) != len(set(outputs)) or len(seeds) != len(set(seeds)):
        raise ValueError('stage outputs and audit seeds must be distinct')
    first = ROOT / plan['parent']['directory']
    original = read_sealed(first / 'registration.json')
    origin_progress = read_sealed(first / 'progress.json')
    for name, value in [('registration_sha256', original['result_sha256']),
                        ('progress_sha256', origin_progress['result_sha256'])]:
        if value != plan['parent'][name]:
            raise ValueError('initial schedule parent changed: ' + name)
    count = sum(s['steps'] for s in original['config']['stages'])
    root_checkpoint = first / 'quality' / f'step{count:05d}.pkl'
    if file_sha256(root_checkpoint) != plan['parent']['checkpoint_sha256']:
        raise ValueError('initial schedule checkpoint changed')
    expected = copy.deepcopy(original['config'])
    # Recursion through completed ancestry also checks every prior config.
    for i in range(index + 1):
        parent = first if i == 0 else ROOT / stages[i-1]['output']
        registration = read_sealed(parent / 'registration.json')
        progress = read_sealed(parent / 'progress.json')
        training = read_sealed(parent / 'training.json')
        if registration['config'] != expected:
            raise ValueError('parent config departed from fixed schedule')
        if (training['status'] != 'fixed_steps_complete'
                or training['completed_steps'] != count
                or progress['completed_steps'] != count
                or training['final_params_sha256'] != progress['final_params_sha256']
                or training['registration_sha256'] != registration['result_sha256']):
            raise ValueError('parent endpoint is incomplete or inconsistent')
        stage = stages[i]
        additional = stage['additional_steps']
        child = copy.deepcopy(expected)
        child.pop('netket_resume_from', None)
        child['output'] = stage['output']
        child['stages'].append(dict(projection='gamma_a1', steps=additional))
        child['quality_milestones'] = sorted(set(child['quality_milestones'] +
            list(range(count+250, count+additional, 250)) + [count+additional]))
        child['maximum_training_seconds'] += stage['time_extension_seconds']
        child['audit']['seeds'] = stage['audit_seeds']
        child['audit']['bootstrap_seed'] = stage['bootstrap_seed']
        relative = f'quality/step{count:05d}.pkl'
        child['netket_continue_from'] = dict(directory=str(parent), completed_steps=count,
            checkpoint_path=relative, checkpoint_sha256=file_sha256(parent/relative),
            registration_sha256=registration['result_sha256'],
            progress_sha256=progress['result_sha256'], additional_steps=additional,
            time_budget_extension_seconds=stage['time_extension_seconds'])
        validate_continuation_protocol(expected, child)
        expected = child
        count += additional
    return expected


def prepare(plan_path: Path, index: int, expected_plan_sha256: str) -> Path:
    """Write once, or verify an identical generated config; never overwrite."""
    if file_sha256(plan_path) != expected_plan_sha256:
        raise ValueError('submitted schedule bytes changed')
    plan = json.loads(plan_path.read_text())
    spec = materialize_stage(plan, index)
    destination = ROOT / plan['stages'][index]['config']
    if destination.exists():
        if json.loads(destination.read_text()) != spec:
            raise FileExistsError('generated config differs from frozen stage')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, spec)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    path = prepare(args.plan, args.stage, args.plan_sha256)
    print(path)
