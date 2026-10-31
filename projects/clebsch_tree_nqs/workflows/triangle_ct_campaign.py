"""Bounded Slurm campaign for the existing N36 exact-SU(2)/A1 tree.

Reuse triangle_clebsch_continuation for all numerical work. CPU dependency
jobs inspect independent fixed-endpoint audits, not training minima. They may
release at most three further64-step blocks before a fixed UTC deadline.
Selection audits are explicitly consumed; the last block uses a fresh larger
audit which is never used to choose a further training state.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from . import triangle_clebsch_continuation as train

ROOT = Path(__file__).resolve().parents[3]
SLURM = '/usr/local/slurm/current/bin'
SEALED_FILES = ('registration.json', 'results.json', 'terminal_audit.json',
                'checkpoint.pkl', 'process_replay.json')


def release_evidence(candidate: dict, parent: dict) -> dict:
    """Development release only, not a significance or GCNN victory test.

    A half-combined-SE energy gain with nonworse variance, OR a >=10% variance
    reduction without a >one-SE energy increase, can justify a bounded next
    block. This intentionally admits unresolved energy differences while
    rejecting simple energy/variance deterioration. Every branch is reported.
    """
    keys = ('energy_per_site', 'energy_chain_SE', 'variance_per_site')
    values = [r[k] for r in (candidate, parent) for k in keys]
    if not all(math.isfinite(x) for x in values) or any(
            r['energy_chain_SE'] < 0 or r['variance_per_site'] < 0 for r in (candidate, parent)):
        raise ValueError('invalid independent endpoint moments')
    delta = candidate['energy_per_site']-parent['energy_per_site']
    se = math.hypot(candidate['energy_chain_SE'], parent['energy_chain_SE'])
    variance_ratio = candidate['variance_per_site']/max(parent['variance_per_site'], 1e-300)
    energy_route = delta < -.5*se and variance_ratio <= 1.1
    variance_route = delta <= se and variance_ratio <= .9
    return dict(released=bool(energy_route or variance_route), delta_energy_per_site=delta,
        combined_SE=se, variance_ratio=variance_ratio,
        reason='energy_and_variance' if energy_route else 'variance_with_compatible_energy'
               if variance_route else 'no_development_release',
        significance_or_variance_error_test=False)


def choose_screen(rows: list[dict], parent: dict) -> tuple[dict | None, list[dict]]:
    """Choose one eligible branch conservatively; preserve failed/unresolved arms."""
    records = []
    for row in rows:
        if row.get('status') != 'verified':
            records.append(dict(case=row['case'], status=row['status']))
            continue
        records.append(dict(case=row['case'], status='verified',
            **release_evidence(row['audit'], parent)))
    eligible = {r['case'] for r in records if r.get('released')}
    options = [r for r in rows if r['case'] in eligible]
    chosen = min(options, key=lambda r: (r['audit']['energy_per_site']+
        r['audit']['energy_chain_SE'], r['case'])) if options else None
    return chosen, records


def validate_campaign(spec: dict) -> None:
    """Limit scope, reserved GPU-hours, number of stages, and absolute deadline."""
    if (spec['task'] != 'triangle_CT_six_day_20261022'
            or set(spec['couplings']) != {'j0', 'j0125'}
            or spec['production_blocks'] != 3 or spec['maximum_A100_hours'] != 240
            or spec['production_steps'] != 64 or spec['screen_steps'] != 16
            or spec['workspace'] != str(ROOT)):
        raise ValueError('campaign differs from the authorized bounded N36 plan')
    beginning = datetime.fromisoformat(spec['created_utc'])
    end = datetime.fromisoformat(spec['deadline_utc'])
    if beginning.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0) or not (
            timedelta(0) < end-beginning <= timedelta(days=6)):
        raise ValueError('six-day absolute UTC boundary required')
    if spec['phase_minutes'] != dict(pilot=180, replay=60, screen=480,
                                     production=1200, final=1920):
        raise ValueError('unexpected allocation schedule')
    screen = 4*(180+60+480)
    production = 2*(2*(180+60+1200)+(180+60+1920))
    if (screen+production)/60 > spec['maximum_A100_hours']:
        raise ValueError('conditional allocation ceiling exceeded')
    if any(len(row['screens']) != 2 for row in spec['couplings'].values()):
        raise ValueError('exactly two damping controls per coupling required')


def verify_endpoint(plan_path: Path) -> dict:
    """Validate actual frozen endpoint and complete optimizer/sampler identity."""
    plan = json.loads(plan_path.read_text()); train.validate_plan(plan)
    directory = Path(plan['output'])
    reg, seal = train._sealed_json(directory/'registration.json')
    result = json.loads((directory/'results.json').read_text())
    audit = json.loads((directory/'terminal_audit.json').read_text())
    replay = json.loads((directory/'process_replay.json').read_text())
    state, meta = train.shared.restore_state(directory/'checkpoint.pkl', seal)
    train.checked_training_release(directory, meta, plan, seal)
    if (reg['plan_sha256'] != canonical_sha256(plan)
            or result['status'] != 'fixed_steps_complete'
            or result['completed_steps'] != plan['training']['steps']
            or result['terminal_audit'] != audit
            or result['registration_sha256'] != seal
            or result['checkpoint_sha256'] != file_sha256(directory/'checkpoint.pkl')
            or result['params_sha256'] != pytree_sha256(state[0])
            or audit['params_sha256'] != pytree_sha256(state[0])
            or not audit['independent_from_training_and_calibration']
            or audit['samples'] != plan['audit']['samples']
            or audit['chains'] != plan['audit']['chains']
            or replay['status'] != 'passed' or replay.get('optimizer_updates_committed') != 0
            or len(meta['history']) != plan['training']['steps']
            or not all(h['accepted'] for h in meta['history'])
            or int(state[1][0]) != meta['optimizer_base_step']+meta['completed_steps']):
        raise ValueError('endpoint/audit/replay/state identity differs')
    return dict(status='verified', case=plan['stage'], plan_path=str(plan_path),
        plan=plan, audit=audit, absolute_completed_steps=meta['absolute_completed_steps'],
        sealed_files={name: file_sha256(directory/name) for name in SEALED_FILES},
        audit_raw_sha256=file_sha256(directory/'terminal_audit.npz'))


def next_plan(endpoint: dict, spec: dict, coupling: str, block: int) -> dict:
    """No predicted future file seals: derive only from the actual audited state."""
    if not 1 <= block <= spec['production_blocks']:
        raise ValueError('production block outside fixed budget')
    plan = deepcopy(endpoint['plan'])
    previous = plan['parent']
    plan['parent'] = dict(kind='continuation', directory=plan['output'],
        model=deepcopy(plan['model']), plan_path=endpoint['plan_path'],
        expected_identity={k: plan[k] for k in (
            'case', 'stage', 'arity', 'model', 'j2_over_j1', 'binary_rank')},
        absolute_completed_steps=endpoint['absolute_completed_steps'],
        sealed_files=endpoint['sealed_files'])
    if previous['model'] != plan['model']:
        raise ValueError('this campaign cannot widen the physical model')
    plan['stage'] = f'{coupling}_production{block:02d}'
    plan['training']['steps'] = spec['production_steps']
    final = block == spec['production_blocks']
    plan['audit']['samples'] = 262144 if final else 65536
    plan['seeds'] = dict(calibration=2610225000+block*100+(coupling == 'j0125')*10+1,
        expansion=2610225000+block*100+(coupling == 'j0125')*10+2,
        audit=2610225000+block*100+(coupling == 'j0125')*10+3)
    plan['budget']['train_seconds'] = 115200 if final else 72000
    plan['reset_optimizer'] = False
    plan['output'] = str(Path(spec['output'])/coupling/f'production{block:02d}')
    plan['interpretation'] = ('Fixed64 same-width/full-H/full-A1 updates; inherit parameters, '
        'direct-SR counter and both sampler/RNG streams. Earlier audits used for development '
        'release are consumed. Fresh final audit is not a checkpoint-selection set. No new head, '
        'width or carrier. Success is not assumed.')
    train.validate_plan(plan)
    return plan


def _submit(spec: dict, key: str, *, phase: str, minutes: int,
            config: Path | None = None, dependency: str | None = None,
            coupling: str = '', stage: str = '') -> str:
    """Idempotent explicit sbatch; ambiguous submissions fail closed, not retry."""
    now = datetime.now(timezone.utc)
    if now+timedelta(minutes=minutes+5) >= datetime.fromisoformat(spec['deadline_utc']):
        raise RuntimeError('not enough time remains before the six-day deadline')
    ledger = Path(spec['output'])/'submissions'/f'{key.replace("/", "__")}.json'
    if ledger.exists():
        old = json.loads(ledger.read_text())
        if old.get('job_id'):
            return old['job_id']
        raise RuntimeError('ambiguous previous submission; reconcile before retry: '+str(ledger))
    env = dict(NQS_PHASE=phase, NQS_CAMPAIGN=spec['campaign_path'],
               NQS_COUPLING=coupling, NQS_STAGE=stage, NQS_ALLOC_MINUTES=str(minutes))
    if config is not None:
        env.update(NQS_CONFIG=str(config),
                   NQS_PLAN_SHA256=canonical_sha256(json.loads(config.read_text())))
    job_name = 'ct6six_'+key.replace('/', '_')
    args = [f'{SLURM}/sbatch', '--parsable', '--partition=kim', '--cpus-per-task=4',
        f'--job-name={job_name}', f'--time={minutes}', '--kill-on-invalid-dep=yes',
        '--nice=1000', '--export=ALL,'+','.join(f'{k}={v}' for k, v in env.items()),
        '--output='+str(Path(spec['output'])/'logs'/'%x-%j.out'),
        '--error='+str(Path(spec['output'])/'logs'/'%x-%j.err')]
    if phase in ('advance', 'tests', 'parents'):
        args += ['--mem=8G']
    else:
        args += ['--mem=96G', '--gres=gpu:nvidia_a100_80gb_pcie:1',
                 '--nodelist='+spec['node'], '--signal=B:USR1@1800']
    if dependency:
        args += ['--dependency='+dependency]
    args.append(spec['runner'])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    (Path(spec['output'])/'logs').mkdir(parents=True, exist_ok=True)
    atomic_json(ledger, dict(status='submitting', command=args, requested_GPU_hours=(
        0 if phase in ('advance', 'tests', 'parents') else minutes/60)))
    proc = subprocess.run(args, text=True, capture_output=True)
    if proc.returncode:
        atomic_json(ledger, dict(status='submission_failed', command=args,
            stdout=proc.stdout, stderr=proc.stderr))
        raise RuntimeError('sbatch failed: '+proc.stderr)
    job = proc.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('ambiguous sbatch reply: '+proc.stdout)
    atomic_json(ledger, dict(status='submitted', job_id=job, command=args,
        requested_GPU_hours=0 if phase in ('advance', 'tests', 'parents') else minutes/60,
        submitted_utc=now.isoformat()))
    return job


def submit_chain(spec: dict, key: str, plan_path: Path, *, dependency: str | None,
                 coupling: str, stage: str) -> str:
    """Process-isolated pilot -> replay -> bounded fixed endpoint -> CPU gate."""
    plan = json.loads(plan_path.read_text())
    jobs = []
    for phase in ('pilot', 'replay', 'train'):
        minutes = plan['budget'][dict(pilot='pilot_seconds', replay='replay_seconds',
                                     train='train_seconds')[phase]]//60
        job = _submit(spec, key+'/'+phase, phase=phase, minutes=minutes,
                      config=plan_path, dependency=dependency)
        jobs.append(job); dependency='afterok:'+job
    if stage != 'screen':
        _submit(spec, key+'/advance', phase='advance', minutes=10,
                dependency='afterany:'+jobs[-1], coupling=coupling, stage=stage)
    return jobs[-1]


def advance(spec: dict, coupling: str, stage: str) -> dict:
    """One CPU gate per completed stage; no polling or login-node training."""
    if coupling not in spec['couplings']:
        raise ValueError('unknown coupling')
    output = Path(spec['output'])/coupling
    decision_path = output/f'decision_{stage}.json'
    if decision_path.exists():
        return json.loads(decision_path.read_text())
    row = spec['couplings'][coupling]
    if stage == 'screen':
        cases = []
        for path in row['screens']:
            try:
                cases.append(verify_endpoint(Path(path)))
            except (OSError, ValueError, KeyError) as error:
                cases.append(dict(case=Path(path).stem, status='failed_or_incomplete', error=str(error)))
        selected, records = choose_screen(cases, row['parent_audit'])
        block = 1
        report = dict(stage=stage, coupling=coupling, comparisons=records,
            selection_audits_consumed=True, best_of_screen_is_not_architecture_win=True)
    else:
        block_done = int(stage)
        try:
            selected = verify_endpoint(Path(spec['generated_configs'])/f'{coupling}_production{block_done:02d}.json')
        except (OSError, ValueError, KeyError) as error:
            report = dict(stage=stage, coupling=coupling, status='stopped_incomplete', error=str(error))
            atomic_json(decision_path, report); return report
        if block_done == spec['production_blocks']:
            report = dict(stage=stage, coupling=coupling, status='fixed_final_endpoint_archived',
                audit=selected['audit'], sealed_files=selected['sealed_files'],
                final_audit_not_used_for_further_training_selection=True,
                result_is_not_automatically_a_GCNN_win=True)
            atomic_json(decision_path, report); return report
        parent_audit = json.loads((Path(selected['plan']['parent']['directory'])/'terminal_audit.json').read_text())
        evidence = release_evidence(selected['audit'], parent_audit)
        report = dict(stage=stage, coupling=coupling, audit=selected['audit'],
            comparison=evidence, selection_audits_consumed=True)
        if not evidence['released']:
            selected = None
        block = block_done+1
    if selected is None:
        report.update(status='stopped_no_quality_release', no_automatic_widening=True,
                      existing_best_state_preserved=True)
    else:
        plan = next_plan(selected, spec, coupling, block)
        path = Path(spec['generated_configs'])/f'{coupling}_production{block:02d}.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('refuse to overwrite a sealed next-stage plan')
        atomic_json(path, plan)
        job = submit_chain(spec, f'{coupling}/production{block:02d}', path,
                           dependency=None, coupling=coupling, stage=str(block))
        report.update(status='next_bounded_block_submitted', selected_case=selected['case'],
                      next_plan=str(path), terminal_job=job)
    atomic_json(decision_path, report)
    return report


def submit_campaign(spec: dict) -> dict:
    """Initial four screens only; later training is audit-conditional."""
    test = _submit(spec, 'tests', phase='tests', minutes=20)
    jobs = {}
    for coupling, row in spec['couplings'].items():
        terminal = []
        for config in row['screens']:
            path = Path(config)
            terminal.append(submit_chain(spec, coupling+'/'+path.stem, path,
                dependency='afterok:'+test, coupling=coupling, stage='screen'))
        gate = _submit(spec, coupling+'/screen/advance', phase='advance', minutes=10,
            dependency='afterany:'+':'.join(terminal), coupling=coupling, stage='screen')
        jobs[coupling] = dict(screen_terminal_jobs=terminal, selection_gate=gate)
    result = dict(status='initial_screens_submitted', jobs=jobs,
        maximum_conditional_A100_hours=spec['maximum_A100_hours'],
        deadline_utc=spec['deadline_utc'], no_background_chat_monitor=True)
    atomic_json(Path(spec['output'])/'dispatch.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--command', choices=('submit', 'advance'), required=True)
    parser.add_argument('--coupling'); parser.add_argument('--stage')
    args = parser.parse_args(); spec = json.loads(args.campaign.read_text())
    validate_campaign(spec)
    if args.command == 'submit':
        answer = submit_campaign(spec)
    else:
        answer = advance(spec, args.coupling, args.stage)
    print(json.dumps(answer), flush=True)
