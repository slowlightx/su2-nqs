"""Fixed-budget audited-parent stages for the separate triangular ClebschTree.

Only continuation bookkeeping is added: reuse the existing exact-S0/full-A1
trainer, sampler, complete Hamiltonian and optimizer without changing them.
Future parents are pinned by their already frozen registration or the exact
preceding stage definition. An incomplete or unsafe pause cannot release a
dependent stage. Audit values do not select parameters or hyperparameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from .spin_triangular_clebsch_train import ROOT, checked_arity_endpoint, train


STAGE_KEYS = frozenset({
    'additional_steps', 'samples', 'audit_samples', 'audit_chains',
    'audit_burn_proposals', 'terminal_audit_seed', 'output', 'generated_config',
})


def materialize_stage(plan: dict, index: int, *, root: Path = ROOT) -> dict:
    """Restore one complete same-model/H state, including optimizer and RNG.

    All charges remain doubled integers; neither the retained q channels nor
    their reduced tensors change. Increasing sample counts changes statistical
    precision, not the physical family. Training chain count stays fixed.
    """
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(plan['stages']):
        raise ValueError('stage index outside the registered fixed budget')
    template = plan['template_config']
    stage = plan['stages'][index]
    if set(stage) != STAGE_KEYS:
        raise ValueError('stage must contain only the declared statistical/work fields')
    if index:
        previous = materialize_stage(plan, index-1, root=root)
        directory = Path(previous['output'])
        expected_step = previous['steps']
        expected_registration = None
    else:
        previous = None
        directory = Path(plan['parent']['directory'])
        expected_step = plan['parent']['completed_steps']
        expected_registration = plan['parent']['registration_sha256']
    registration = json.loads((directory/'registration.json').read_text())
    seal = registration['sha256']
    if seal != canonical_sha256({k: v for k, v in registration.items() if k != 'sha256'}):
        raise ValueError('parent registration changed')
    if expected_registration is not None and seal != expected_registration:
        raise ValueError('wrong registered first parent')
    parent_spec = registration['config']
    if previous is not None and parent_spec != previous:
        raise ValueError('parent is not the exact preceding frozen stage')
    for key in ('model', 'arm', 'arity_comparison', 'j2_over_j1', 'chains',
                'projection_chunk', 'damping', 'learning_rate', 'maximum_FS_sine',
                'spacing_proposals'):
        if parent_spec[key] != template[key]:
            raise ValueError('same-model continuation changed '+key)
    result = json.loads((directory/'results.json').read_text())
    if (result.get('status') != 'fixed_steps_complete'
            or parent_spec['steps'] != expected_step
            or result.get('completed_steps') != expected_step
            or result.get('registration_sha256') != seal
            or result.get('checkpoint_rng_sampler_replay_exact') is not True
            or result.get('checkpoint_optimizer_exact') is not True):
        raise ValueError('parent endpoint incomplete or checkpoint replay failed')
    audit_path = directory/'terminal_audit.json'
    audit = json.loads(audit_path.read_text())
    if (audit.get('independent_from_training') is not True
            or result.get('terminal_audit') != audit):
        raise ValueError('complete independent parent audit required')
    if stage['terminal_audit_seed'] == audit['seed']:
        raise ValueError('fresh terminal audit seed required')
    seeds = [s['terminal_audit_seed'] for s in plan['stages']]
    if len(seeds) != len(set(seeds)):
        raise ValueError('stage audit seeds must be distinct')
    for key in ('additional_steps', 'samples', 'audit_samples', 'audit_chains',
                'audit_burn_proposals'):
        if isinstance(stage[key], bool) or not isinstance(stage[key], int) or stage[key] < 1:
            raise ValueError('positive integer required for '+key)
    if stage['samples'] % template['chains'] or stage['samples'] % 2:
        raise ValueError('samples must tile unchanged chains and full-H pairs')
    if stage['audit_samples'] % stage['audit_chains'] or stage['audit_samples'] % 2:
        raise ValueError('audit samples must tile independent chains and full-H pairs')
    output = Path(stage['output']).resolve()
    outputs = [Path(s['output']).resolve() for s in plan['stages']]
    if len(set(outputs)) != len(outputs) or any(
            output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents
            for p in (directory, Path(plan['parent']['directory']))):
        raise ValueError('new nonoverlapping stage output required')
    spec = deepcopy(template)
    spec.update({k: v for k, v in stage.items() if k != 'generated_config'})
    spec.update(endpoint_protocol='same_H_full_state_continuation',
                steps=expected_step+stage['additional_steps'],
                continuation_plan=dict(sha256=canonical_sha256(plan), stage=index),
                arity_endpoint_parent=dict(
                    directory=str(directory), registration_sha256=seal,
                    completed_steps=expected_step,
                    checkpoint_sha256=file_sha256(directory/'checkpoint.pkl'),
                    params_sha256=audit['params_sha256'],
                    audit_sha256=file_sha256(audit_path)))
    # Reuse the actual training loader: verifies parameter, optimizer, sampler,
    # history, audit and all checkpoint identities, not just JSON progress.
    _, _, endpoint = checked_arity_endpoint(MixedClebschConfig(**spec['model']), spec)
    states = endpoint['saved']['metadata']['sampler_states']
    if states.shape != (spec['chains'], spec['model']['length']**2):
        raise ValueError('continuation cannot silently resize the saved chain state')
    return spec


def prepare(path: Path, expected_sha256: str, index: int, *, root: Path = ROOT) -> Path:
    """Freeze populated future-parent hashes once; never mutate active output."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('registered continuation plan bytes changed')
    plan = json.loads(path.read_text())
    for name, digest in plan['execution_sources'].items():
        if file_sha256(root/name) != digest:
            raise ValueError('registered execution source changed: '+name)
    spec = materialize_stage(plan, index, root=root)
    destination = root/plan['stages'][index]['generated_config']
    if destination.exists():
        if json.loads(destination.read_text()) != spec:
            raise FileExistsError('generated fixed-stage configuration changed')
    else:
        atomic_json(destination, spec)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--session-seconds', type=float, default=2400.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = prepare(args.plan, args.plan_sha256, args.stage)
    if args.prepare_only:
        print(config)
    else:
        spec = json.loads(config.read_text())
        result = train(spec, Path(spec['output']), session_seconds=args.session_seconds)
        if result['status'] != 'fixed_steps_complete':
            raise RuntimeError('safe checkpoint retained; fixed budget incomplete, do not release next stage')
        print(result['status'], result['completed_steps'], flush=True)
