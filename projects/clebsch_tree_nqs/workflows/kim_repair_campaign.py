"""Bounded Kim repair campaign; preserve old runs and immutable parent states.

Prepare verifies and copies only declared parent inputs, then registers new
protocols. Submit releases long GPU jobs only after their actual full-model
pilot passes. No phase, convergence or variational-win claim is made here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
from collections.abc import Callable, Sequence

from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.frontier.artifacts import pytree_sha256, atomic_json
from su2_nqs.vmc.checkpoint import load_checkpoint
from .spin_spatial_vit_quality import ROOT, read_sealed, register, validate_spec

CASES = (('tri_j0', 'triangular', 0.), ('tri_j005', 'triangular', .05),
         ('tri_j01', 'triangular', .1), ('tri_j0175', 'triangular', .175),
         ('square_j0', 'square', 0.), ('square_j05', 'square', .5))
HOME_DATA = Path('/home/yx789/nqs-wavefunctions')
OLD_NQS = Path('/share/kim/yx789/nqs-vit-quality/results')
OLD_DMRG = Path('/share/kim/yx789/nqs-reference-dmrg/work/chi8192_from5120_v1')
CONFIGS = Path('projects/clebsch_tree_nqs/configs/remote/repair_20261010_v5')
REMOTE = Path('projects/clebsch_tree_nqs/remote')


def live_slurm_job(job_id: int, *, query: Callable[[list[str]], str] | None = None) -> dict[str, str]:
    """Read the live controller, never an undated accounting JobID match.

    Slurm can reuse IDs: sacct may return a months-old success for a currently
    running task. Missing live records fail closed; callers must separately
    verify dated receipts and scientific artifacts for expired parents.
    """
    if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
        raise ValueError('positive integer Slurm job ID required')
    command = ['/usr/local/slurm/current/bin/scontrol', 'show', 'job', '-o', str(job_id)]
    output = (query or (lambda cmd: subprocess.check_output(cmd, text=True)))(command)
    rows = [line for line in output.splitlines() if line.startswith('JobId=')]
    if len(rows) != 1:
        raise ValueError('one unambiguous live Slurm job required')
    fields = dict(re.findall(r'(\w+)=(\S+)', rows[0]))
    if fields.get('JobId') != str(job_id) or 'JobState' not in fields:
        raise ValueError('live Slurm job identity changed')
    return fields


def slurm_dependency_options(afterok: Sequence[int], *, afterany: Sequence[int] = ()) -> list[str]:
    """Keep server-enforced gates, including a safe-stop writer dependency.

    Never remove afterok just because an accounting query says COMPLETED.
    afterany is only for a known own task stopped at a checkpoint boundary,
    and does not substitute for the scientific afterok gate.
    """
    conditions = []
    for kind, ids in (('afterok', afterok), ('afterany', afterany)):
        if any(not isinstance(j, int) or isinstance(j, bool) or j <= 0 for j in ids):
            raise ValueError('positive integer dependency IDs required')
        if ids:
            conditions.append(kind+':'+':'.join(map(str, dict.fromkeys(ids))))
    return ['--dependency='+','.join(conditions), '--kill-on-invalid-dep=yes'] if conditions else []


def sync_manifest() -> dict:
    """Generate a code-only transfer list from the actual import closure."""
    entries = (Path(__file__), ROOT/'tests/test_spatial_vit_netket.py',
        ROOT/'tests/test_spatial_vit_quality.py', ROOT/'tests/test_kim_repair_campaign.py',
        ROOT/'tests/test_tensor_network_readout.py', ROOT/'tests/test_dmrg_archive.py',
        ROOT/'tests/test_dmrg_region_probe.py', Path(__file__).with_name('dmrg_continuation.py'),
        Path(__file__).with_name('dmrg_region_probe.py'))
    files = transitive_local_source_manifest(entries, root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    extras = ('AGENTS.md', 'pyproject.toml',
        str(REMOTE/'spatial_vit_recovery_a100.sbatch'), str(REMOTE/'dmrg_kim.sbatch'),
        str(REMOTE/'physics_readout_kim.sbatch'),
        'projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_tri_j005.json',
        'projects/clebsch_tree_nqs/configs/remote/spin_tensor_network_chi8192_kim.json',
        'projects/clebsch_tree_nqs/configs/remote/spin_dmrg_chi8192_spin_env_plan.json',
        'projects/clebsch_tree_nqs/configs/remote/spin_dmrg_chi8192_dimers_env_plan.json',
        'projects/clebsch_tree_nqs/configs/remote/spin_dmrg_chi8192_hexagon6_plan.json',
        'projects/clebsch_tree_nqs/configs/remote/readouts/tri_j005_magnetic.json',
        'projects/clebsch_tree_nqs/configs/remote/readouts/tri_j005_regional.json',
        'projects/clebsch_tree_nqs/configs/remote/readouts/square_j05_magnetic.json',
        'projects/clebsch_tree_nqs/configs/remote/readouts/square_j05_regional.json',
        'projects/clebsch_tree_nqs/configs/validation/spin_spatial_vit_quality.json',
        'projects/clebsch_tree_nqs/configs/validation/spin_spatial_vit_quality_continuation.json',
        'projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_netket_continuation_7600_n36.json')
    files.update({name: file_sha256(ROOT/name) for name in extras})
    fixture_spec = json.loads((ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_spatial_vit_quality_continuation.json').read_text())
    fixtures = [fixture_spec['initial_checkpoint'][key] for key in ('checkpoint', 'result', 'training_result')]
    files.update({name: file_sha256(ROOT/name) for name in fixtures})
    record = dict(remote_code_directory='/home/yx789/nqs-repair-20261010', files=files,
                  files_count=len(files), old_workspaces_untouched=True,
                  no_bulk_results_notebooks_or_secrets_transferred=True,
                  immutable_validation_fixture_files=fixtures,
                  immutable_validation_fixture_bytes=sum((ROOT/p).stat().st_size for p in fixtures))
    atomic_json(ROOT/REMOTE/'kim_repair_sync_manifest_20261010.json', record)
    (ROOT/REMOTE/'kim_repair_sync_files.txt').write_text('\n'.join(sorted(files))+'\n')
    return record


def write_once(path: Path, record: dict) -> None:
    """Create a replayable record; never silently overwrite an earlier plan."""
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise FileExistsError(f'conflicting campaign record: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write('\n')


def copy_verified(source: Path, destination: Path) -> None:
    """Copy a regular immutable input without changing or deleting its source."""
    if not source.is_file() or source.is_symlink():
        raise ValueError(f'explicit regular input required: {source}')
    expected = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != expected:
            raise FileExistsError(f'input copy differs: {destination}')
        return
    temporary = destination.with_name(destination.name+'.copying')
    if temporary.exists():
        raise FileExistsError(f'incomplete copy retained for inspection: {temporary}')
    shutil.copy2(source, temporary)
    if file_sha256(temporary) != expected or file_sha256(source) != expected:
        raise RuntimeError('parent changed or input copy is corrupt')
    temporary.replace(destination)


def migrate_nqs_parent(source: Path, name: str, step: int) -> tuple[dict, dict]:
    """Keep only registration, source snapshots and the selected quality state."""
    registration = read_sealed(source/'registration.json')
    destination = HOME_DATA/'inputs'/name
    copy_verified(source/'registration.json', destination/'registration.json')
    if (source/'training.json').exists():
        copy_verified(source/'training.json', destination/'training.json')
    for relative, digest in registration['source_manifest'].items():
        old = source/'source_snapshots'/relative
        if file_sha256(old) != digest:
            raise ValueError('parent frozen source changed')
        copy_verified(old, destination/'source_snapshots'/relative)
    relative = f'quality/step{step:05d}.pkl'
    copy_verified(source/relative, destination/relative)
    saved = load_checkpoint(destination/relative)
    if (saved['metadata']['registration_sha256'] != registration['result_sha256']
            or saved['metadata']['completed_steps'] != step):
        raise ValueError('selected parent is not the registered quality boundary')
    descriptor = dict(directory=str(destination), completed_steps=step,
        checkpoint_path=relative, checkpoint_sha256=file_sha256(destination/relative),
        registration_sha256=registration['result_sha256'],
        params_sha256=pytree_sha256(saved['params']['model']))
    return registration['config'], descriptor


def nqs_spec(parent: dict, descriptor: dict, *, name: str, geometry: str,
             j2: float, index: int) -> dict:
    """Same complete backbone, new bounded sampler/SR protocol and audit seeds."""
    if (name, geometry, j2) not in CASES or parent['model']['geometry'] != geometry:
        raise ValueError('only the six explicitly authorized cases are permitted')
    spec = copy.deepcopy(parent)
    for key in ('initial_checkpoint', 'coupling_warm_start_from', 'repair_parameters_from',
                'netket_continue_from', 'netket_resume_from', 'continuation_from',
                'initialization_revision'):
        spec.pop(key, None)
    spec.update(output=str(HOME_DATA/'nqs'/f'{name}_repair_v5'/'seed1'),
        j2_over_j1=j2, initialization_seed=2026101200+10*index,
        training_seed=2026101201+10*index, backend='netket_ntk', require_gpu=True,
        stages=[dict(projection='gamma_a1', steps=1000)],
        quality_milestones=[0, 20, 250, 500, 750, 1000],
        maximum_training_seconds=45000,
        maximum_audit_seconds_per_checkpoint=10800,
        maximum_total_audit_seconds=21600,
        sampling=dict(chains=128, snapshots_per_update=32,
                      proposals_between_snapshots=72, burn_proposals=3600),
        sr=dict(damping=.001, step_size=.00125))
    spec['netket'] = dict(version='3.22.3', on_the_fly=False, chunk_size=16,
        chunk_size_bwd=4, momentum=None, initialization='parent_configurations',
        exchange_kernel='uniform_all_pairs', replicate_initial_params=True,
        sampled_trust=dict(maximum_fs_sine=.02, minimum_ess_fraction=.95, halvings=6))
    spec['audit'] = dict(seeds=[2026101202+10*index, 2026101203+10*index],
        chains=64, snapshots=128, proposals_between_snapshots=144,
        burn_proposals=2880, query_batch=16, bootstrap_seed=2026101204+10*index,
        bootstrap_replicates=2000)
    spec['interpretation'] = dict(strict_su2=False, new_head=False,
        precision_or_phase_claim=False,
        protocol='Complete unchanged Spatial-ViT backbone. New registered repair '
        'from immutable weights; optimizer/RNG/work counters reset, old states '
        'preserved. Pure symmetric all-pair proposals (including null moves), '
        'explicit parent sampler installation and replicated GPU parameters. '
        'Full A1 from step0; conservative SR and sampled nonlinear FS/ESS bound. '
        'Bound is on training samples, not an equilibrium or held-out certificate. '
        '20 production steps form the gated pilot, not extra uncounted training. '
        'Only after pilot passes may it continue to fixed1000 and fresh audit.')
    if geometry == 'triangular':
        spec['coupling_warm_start_from'] = dict(**descriptor,
            parent_j2_over_j1=parent['j2_over_j1'])
    else:
        if parent['j2_over_j1'] != j2:
            raise ValueError('square repair must preserve its parent Hamiltonian')
        spec['repair_parameters_from'] = descriptor
    validate_spec(spec)
    return spec


def prepare() -> dict:
    """Run on Kim in the isolated code workspace, without submitting jobs."""
    triangle = migrate_nqs_parent(OLD_NQS/'spin_spatial_vit_gpu_continuation_7600_v1'/'seed1',
                                  'tri_j0125_step07600', 7600)
    square0 = migrate_nqs_parent(OLD_NQS/'spin_spatial_vit_square_j0_v2'/'seed1',
                                 'square_j0_step05500', 5500)
    square05 = migrate_nqs_parent(OLD_NQS/'spin_spatial_vit_square_n36_v1'/'seed1',
                                  'square_j05_step05500', 5500)
    plans = []
    for index, (name, geometry, j2) in enumerate(CASES):
        parent, descriptor = triangle if geometry == 'triangular' else square0 if j2 == 0 else square05
        spec = nqs_spec(parent, descriptor, name=name, geometry=geometry, j2=j2, index=index)
        relative = CONFIGS/f'{name}.json'
        write_once(ROOT/relative, spec)
        _, reg = register(ROOT/relative, Path(spec['output']))
        ceiling = -.45 if geometry == 'triangular' else -.65 if j2 == 0 else -.48
        readouts = []
        for kind in ('magnetic', 'regional'):
            template = 'tri_j005' if geometry == 'triangular' else 'square_j05'
            read_spec = json.loads((ROOT/f'projects/clebsch_tree_nqs/configs/remote/readouts/{template}_{kind}.json').read_text())
            read_spec.update(parent=spec['output'], step=1000,
                output=str(HOME_DATA/'readouts'/f'{name}_repair_v5'/kind),
                materialized_config=str(HOME_DATA/'readouts'/'configs'/f'{name}_repair_v5_{kind}.json'),
                bootstrap_seed=2026101300+10*index+(kind == 'regional'))
            read_spec['problem']['j2'] = j2
            read_config = CONFIGS/f'{name}_{kind}.json'
            write_once(ROOT/read_config, read_spec)
            readouts.append(str(read_config))
        plans.append(dict(name=name, config=str(relative), output=spec['output'],
            registration_sha256=reg['result_sha256'], initial_energy_ceiling=ceiling,
            readouts=readouts))
    # Archive complete sweep28 in HOME using the same verified byte-copy utility.
    # It includes the full optimizer/MPS and all sealed raw regional readouts.
    archive_receipt = ROOT/'logs'/'home-dmrg-parent.json'
    if not archive_receipt.exists():
        subprocess.run([str(ROOT/'.venv-tenpy/bin/python'), '-m',
            'projects.clebsch_tree_nqs.workflows.dmrg_archive',
            '--source-root', str(OLD_DMRG.parent), '--stage', OLD_DMRG.name,
            '--archive-root', str(HOME_DATA/'dmrg'/'archives'),
            '--receipt', str(archive_receipt)], check=True)
    archived = json.loads(archive_receipt.read_text())['archives'][0]
    snapshot = Path(archived['snapshot'])
    if any(file_sha256(snapshot/p) != digest for p, digest in archived['files_sha256'].items()):
        raise ValueError('HOME archive byte verification failed')
    result = json.loads((snapshot/'results.json').read_text())
    if archived['global_sweep'] != 28 or archived['source_status'] != 'complete':
        raise ValueError('expected complete immutable sweep28')
    old = json.loads((ROOT/'projects/clebsch_tree_nqs/configs/remote/spin_tensor_network_chi8192_kim.json').read_text())
    old.update(initial_checkpoint=str(snapshot/'checkpoint.h5'),
        initial_checkpoint_sha256=archived['checkpoint_sha256'],
        parent_output=str(snapshot), parent_protocol_sha256=archived['protocol_sha256'],
        output=str(HOME_DATA/'dmrg'/'work'/'chi8192_relax_v2'),
        archive_root=str(HOME_DATA/'dmrg'/'archives'),
        sweeps=2, mixer=False, chunk_seconds=82800, reserve_seconds=3600,
        initial_sweep_seconds_bound=20000, total_seconds_cap=82800,
        protocol='Samechi8192 mixerOFF exactly two sweeps29-30; full sweep28 '
        'MPS/optimizer inherited,32CPU512GiB,24h scheduler/23h active/1h reserve. '
        'Measured prior relaxation sweep4.07h suggests8-10h plus rawRDMs. '
        '448GiB guard; compare energy/discardedweight and identical three regions. '
        'HOME work and verified HOME immutable archive authorized2026-10-10. '
        'Old SHARE input/archive retained; no automatic chi increase.')
    dmrg_config = CONFIGS/'dmrg8192_relax.json'
    write_once(ROOT/dmrg_config, old)
    reads = []
    for observable in ('spin', 'dimers', 'hexagon6'):
        template = ROOT/f'projects/clebsch_tree_nqs/configs/remote/spin_dmrg_chi8192_{observable}{"" if observable == "hexagon6" else "_env"}_plan.json'
        spec = json.loads(template.read_text())
        spec['completed_parent']['directory'] = str(snapshot)
        name = f'chi8192_sweep28_{observable}_env_v3'
        spec['output'] = str(HOME_DATA/'dmrg'/'readouts'/name)
        spec['materialized_config'] = str(HOME_DATA/'dmrg'/'readouts'/'configs'/f'{name}.json')
        path = CONFIGS/f'dmrg8192_{observable}.json'
        write_once(ROOT/path, spec)
        reads.append(dict(name=name, config=str(path), seconds=spec['limits']['wall_seconds'],
            threads=spec['limits']['threads'], memory_gib=512 if observable == 'hexagon6' else 96,
            after_continuation=False))
        if observable != 'hexagon6':
            late = copy.deepcopy(spec)
            late['completed_parent'].update(directory=str(HOME_DATA/'dmrg'/'archives'/'chi8192_relax_v2'/'latest'), sweep=30)
            late_name = name.replace('sweep28', 'sweep30')
            late['output'] = str(HOME_DATA/'dmrg'/'readouts'/late_name)
            late['materialized_config'] = str(HOME_DATA/'dmrg'/'readouts'/'configs'/f'{late_name}.json')
            late_path = CONFIGS/f'dmrg8192_sweep30_{observable}.json'
            write_once(ROOT/late_path, late)
            reads.append(dict(name=late_name, config=str(late_path), seconds=late['limits']['wall_seconds'],
                threads=8, memory_gib=96, after_continuation=True))
    record = dict(task='kim_bounded_repair_20261010', workspace=str(ROOT),
        home_wavefunctions_user_authorized=True, old_share_archives_retained=True,
        nqs=plans, dmrg=dict(config=str(dmrg_config), archived_parent=archived,
            parent_energy_per_site=result['energy_per_site'], reads=reads),
        excluded='triangleJ2=.125 current production and squareJ2=.8 unchanged; '
        'no largeED, newhead, automaticchi increase or blind8192 H² contraction')
    write_once(ROOT/REMOTE/'kim_repair_preparation_20261010_v5.json', record)
    return record


def submit(preparation: dict, *, nqs_only: bool = False) -> dict:
    """Submit scoped, dependency-gated jobs and atomically retain each receipt."""
    receipt_path = ROOT/REMOTE/('kim_repair_nqs_retry_20261010.json' if nqs_only else 'kim_repair_launch_20261010.json')
    if receipt_path.exists():
        raise FileExistsError('campaign already submitted; inspect its receipt instead of duplicating')
    if nqs_only:
        prior = json.loads((ROOT/REMOTE/'kim_repair_launch_20261010.json').read_text())
        cpu_job = next(row['job_id'] for row in prior['jobs'] if row['name'] == 'repair-netket-cpu-tests')
        status = subprocess.check_output(['/usr/local/slurm/current/bin/sacct', '-u', 'yx789',
            '-S', '2026-10-10', '-j', cpu_job, '-n', '-X', '--format=State'], text=True).strip()
        if status != 'FAILED':
            raise ValueError('retry is permitted only after the failed CPU deployment gate')
        if any((Path(row['output'])/'checkpoint.pkl').exists() for row in preparation['nqs']):
            raise ValueError('a GPU repair already started; do not duplicate its state/trajectory')
    jobs = []
    def launch(name: str, options: list[str], script: str, args: list[str], *, env: dict | None = None) -> str:
        command = ['/usr/local/slurm/current/bin/sbatch', '--parsable', '--chdir='+str(ROOT),
                   '--job-name='+name, *options, str(REMOTE/script), *args]
        submitted = subprocess.check_output(command, text=True, env={**os.environ, **(env or {})}).strip()
        job = submitted.split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('unexpected sbatch receipt: '+submitted)
        jobs.append(dict(job_id=job, name=name, command=command, environment=env or {}))
        # Partial submission is recoverable/inspectable, never silently repeated.
        receipt = dict(preparation=preparation, jobs=jobs, all_submissions_complete=False)
        temporary = receipt_path.with_suffix('.pending')
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
        temporary.replace(receipt_path)
        print(json.dumps(jobs[-1]), flush=True)
        return job
    if not nqs_only:
        dmrg = launch('repair-dmrg8192', ['--time=24:00:00'], 'dmrg_kim.sbatch',
                      [preparation['dmrg']['config']])
        for row in preparation['dmrg']['reads']:
            options = ['--time=03:00:00', '--cpus-per-task='+str(row['threads']), '--mem='+str(row['memory_gib'])+'G']
            if row['after_continuation']:
                options.extend(['--dependency=afterok:'+dmrg, '--kill-on-invalid-dep=yes'])
            launch('repair-'+row['name'], options, 'physics_readout_kim.sbatch',
                   ['dmrg', row['config']])
    cpu_test = launch('repair-netket-cpu-tests',
        ['--constraint=cpu', '--gres=none', '--cpus-per-task=8', '--mem=64G', '--time=01:00:00'],
        'spatial_vit_recovery_a100.sbatch', ['cpu-test'])
    for row in preparation['nqs']:
        env = dict(NQS_CONFIG=row['config'], NQS_PREFLIGHT_ENDPOINT='20',
                   NQS_INITIAL_ENERGY_CEILING=str(row['initial_energy_ceiling']),
                   NQS_INITIAL_VARIANCE_CEILING='1.0', NQS_BOUNDED_PILOT_SCREEN='1')
        pilot = launch('repair-'+row['name']+'-pilot',
                       ['--time=02:00:00', '--dependency=afterok:'+cpu_test, '--kill-on-invalid-dep=yes'],
                       'spatial_vit_recovery_a100.sbatch', ['preflight'], env=env)
        train = launch('repair-'+row['name']+'-1000',
            ['--time=13:00:00', '--dependency=afterok:'+pilot, '--kill-on-invalid-dep=yes'],
            'spatial_vit_recovery_a100.sbatch', ['production'],
            env=dict(NQS_CONFIG=row['config'], NQS_UNTIL_STEP='1000', NQS_SESSION_SECONDS='45000'))
        audit = launch('repair-'+row['name']+'-audit',
            ['--time=03:00:00', '--gres=gpu:nvidia_a100_80gb_pcie:1', '--array=0-1',
             '--dependency=afterok:'+train, '--kill-on-invalid-dep=yes'],
            'spatial_vit_recovery_a100.sbatch', ['energy-stream'], env=dict(NQS_CONFIG=row['config']))
        seal = launch('repair-'+row['name']+'-seal',
            ['--constraint=cpu', '--gres=none', '--cpus-per-task=4', '--mem=32G',
             '--time=00:20:00', '--dependency=afterok:'+audit, '--kill-on-invalid-dep=yes'],
            'spatial_vit_recovery_a100.sbatch', ['aggregate'], env=dict(NQS_CONFIG=row['config']))
        launch('repair-'+row['name']+'-read',
            ['--time=02:00:00', '--dependency=afterok:'+seal, '--kill-on-invalid-dep=yes'],
            'physics_readout_kim.sbatch', ['nqs', *row['readouts']])
    receipt = dict(preparation=preparation, jobs=jobs, all_submissions_complete=True,
        nqs_only_deployment_retry=nqs_only, existing_dmrg_not_resubmitted=nqs_only)
    temporary = receipt_path.with_suffix('.pending')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    temporary.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sync-manifest', 'prepare', 'submit', 'resubmit-nqs'))
    args = parser.parse_args()
    if args.action == 'sync-manifest':
        result = sync_manifest()
    elif args.action == 'prepare':
        result = prepare()
    else:
        result = submit(json.loads((ROOT/REMOTE/'kim_repair_preparation_20261010_v5.json').read_text()),
                        nqs_only=args.action == 'resubmit-nqs')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
