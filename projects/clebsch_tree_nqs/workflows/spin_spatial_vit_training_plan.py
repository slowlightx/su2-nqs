"""Materialize a frozen, audited-parent training plan without selecting weights.

Reuse the sealed same-H parameter repair loader and existing NetKet driver.
No active source/config is edited; a future endpoint is materialized only
after completion and its declared audit file exists. Audit NUMBERS cannot
select weights or hyperparameters. This is a new optimization experiment,
with fresh sampler/RNG/work counters by default. Explicit full-state and
cross-coupling modes reuse the corresponding validated loaders instead.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .spin_spatial_vit_quality import ROOT, read_sealed, validate_spec
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256


def materialize(plan: dict) -> dict:
    """Validate a frozen future parent and populate ONLY its immutable hashes."""
    parent = Path(plan['parent']['directory'])
    registration = read_sealed(parent/'registration.json')
    if registration['result_sha256'] != plan['parent']['registration_sha256']:
        raise ValueError('parent registration changed')
    progress = read_sealed(parent/'progress.json')
    training = read_sealed(parent/'training.json')
    step = plan['parent']['completed_steps']
    if (step != sum(s['steps'] for s in registration['config']['stages'])
            or progress['completed_steps'] != step or training['completed_steps'] != step
            or progress['status'] != 'fixed_steps_complete' or training['status'] != 'fixed_steps_complete'
            or progress['final_params_sha256'] != training['final_params_sha256']
            or training['registration_sha256'] != registration['result_sha256']):
        raise ValueError('parent endpoint incomplete or inconsistent')
    # Older full-physics audits are sealed as results.json; later energy-only
    # audits use energy.json. Require an explicit choice, never silently fall
    # back between different measurements or use whichever has lower energy.
    audit_file = plan['parent'].get('audit_file', 'energy.json')
    if audit_file not in {'energy.json', 'results.json'}:
        raise ValueError('declare an energy.json or results.json parent audit')
    audit = read_sealed(parent/'audits'/f'step{step:05d}'/audit_file)
    if (audit['checkpoint_sha256'] != file_sha256(parent/'quality'/f'step{step:05d}.pkl')
            or audit['registration_sha256'] != registration['result_sha256']
            or audit['params_sha256'] != training['final_params_sha256']
            or audit['step'] != step or audit['audit_projection'] != 'gamma_a1'):
        raise ValueError('declared audit does not belong to frozen parent')
    if audit_file == 'results.json' and audit.get('training_projection') != 'gamma_a1':
        raise ValueError('full-physics parent audit must describe the trained full-A1 state')
    if ('child_config' in plan) == ('child_overrides' in plan):
        raise ValueError('declare exactly one child_config or child_overrides')
    child = copy.deepcopy(plan.get('child_config', registration['config']))
    if 'child_overrides' in plan:
        child.update(copy.deepcopy(plan['child_overrides']))
    mode = plan.get('mode', 'parameter_repair')
    if mode not in {'parameter_repair', 'full_state_continuation', 'coupling_warm_start'}:
        raise ValueError('unknown audited-parent training mode')
    keys = ('model','model_family','coordinate_convention','spin_inversion_even')
    if mode != 'coupling_warm_start':
        keys += ('j2_over_j1',)
    for key in keys:
        if child.get(key) != registration['config'].get(key):
            raise ValueError('same-model training plan changed '+key)
    if child['output'] == str(parent) or any(s['projection'] != 'gamma_a1' for s in child['stages']):
        raise ValueError('new independent full-A1 output required')
    if set(child['audit']['seeds']) & set(registration['config']['audit']['seeds']):
        raise ValueError('untouched audit seeds required')
    declaration = dict(directory=str(parent),completed_steps=step,
        checkpoint_path=f'quality/step{step:05d}.pkl',
        checkpoint_sha256=file_sha256(parent/'quality'/f'step{step:05d}.pkl'),
        registration_sha256=registration['result_sha256'],params_sha256=training['final_params_sha256'])
    if mode == 'full_state_continuation':
        from .spin_spatial_vit_netket import validate_continuation_protocol
        continuation = plan['continuation']
        if set(continuation) != {'additional_steps', 'time_budget_extension_seconds'}:
            raise ValueError('declare only fixed steps and explicit cumulative time extension')
        declaration.pop('params_sha256')
        declaration.update(continuation, progress_sha256=progress['result_sha256'])
        child['netket_continue_from'] = declaration
        validate_continuation_protocol(registration['config'], child)
    else:
        # These modes explicitly reset optimizer/sampler/RNG. The old parent's
        # initializer is ancestry, not a second executable initialization.
        for key in ('repair_parameters_from', 'coupling_warm_start_from',
                    'netket_continue_from', 'netket_resume_from', 'initial_checkpoint'):
            child.pop(key, None)
        if mode == 'coupling_warm_start':
            if child['j2_over_j1'] == registration['config']['j2_over_j1']:
                raise ValueError('coupling warm start requires a different Hamiltonian')
            declaration['parent_j2_over_j1'] = registration['config']['j2_over_j1']
            child['coupling_warm_start_from'] = declaration
        else:
            child['repair_parameters_from'] = declaration
    validate_spec(child)
    return child


def prepare(path: Path, expected_sha256: str) -> Path:
    """Populate future hashes once; refuse different plans or existing configs."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('frozen optimization plan bytes changed')
    plan = json.loads(path.read_text())
    spec = materialize(plan)
    output = ROOT/plan['generated_config']
    if output.exists():
        if json.loads(output.read_text()) != spec:
            raise FileExistsError('generated optimizer protocol changed')
    else:
        atomic_json(output,spec)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    print(prepare(args.plan,args.plan_sha256))
