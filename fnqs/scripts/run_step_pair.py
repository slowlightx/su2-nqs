"""Matched learning-rate repair after the fixed L16 coordinate experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import jax
import numpy as np
import scipy

from fnqs.artifacts import load_checkpoint, write_json
from fnqs.pilots.coordinate_sr import train
from fnqs.pilots.engine import Options, model_signature
from fnqs.pilots.lineage import ROOT, sha, sources, verify_sources, save_node, cost, check_cost, inclusive_cost
from fnqs.scripts.run_l16_pair import checked_node, audit, metrics
from fnqs.variants.coupling_units import CouplingUnits
from fnqs.variants.size_features import SizeFeatureConfig


def prepare(output: Path, parent: Path) -> None:
    """Declare six matched branches before reading any new observables."""
    if output.exists():
        raise FileExistsError('fresh output directory required')
    previous = json.loads((parent/'protocol.json').read_text())
    review = json.loads((parent/'review.json').read_text())
    verify_sources(previous['source_sha256'])
    if review['all200_development_observables_pass']:
        raise ValueError('all three development seeds already pass')
    base = Options(**previous['options'])
    runs = []
    for old in previous['runs']:
        node_path = (parent/'runs'/old['name']/'step_200/node.json').resolve()
        node = checked_node(node_path)
        for arm, learning_rate in (('slow', base.learning_rate), ('fast', 4.)):
            options = replace(base, learning_rate=learning_rate)
            schedule = [old['train_points'] for _ in range(100)]
            runs.append(dict(name=f'{arm}_seed_{old["seed"]}', arm=arm, seed=old['seed'],
                parent=str(node_path), parent_sha256=sha(node_path),
                checkpoint_sha256=node['checkpoint_sha256'], options=asdict(options),
                train_points=old['train_points'], schedule=schedule, cost=cost(schedule, options.samples)))
    manifest = {**sources(), **{name: sha(ROOT/name) for name in (
        'fnqs/scripts/run_l16_pair.py', 'fnqs/scripts/run_coordinate_repair.py',
        'fnqs/scripts/run_step_pair.py')}}
    output.mkdir(parents=True)
    for name in manifest:
        target = output/'runtime'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(output/'protocol.json', dict(kind='L16_matched_learning_rate_v17', n=16,
        model=previous['model'], units=previous['units'], runs=runs,
        primary_points=previous['primary_points'], reserved_points=previous['reserved_points'],
        source_sha256=manifest,
        input_sha256={str(parent/'protocol.json'):sha(parent/'protocol.json'), str(parent/'review.json'):sha(parent/'review.json')},
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                         jax=jax.__version__, platform=platform.platform(), backend=jax.default_backend(),
                         x64=bool(jax.config.jax_enable_x64)),
        grouping_features_width_loss_coordinate_shift_samples_clipping_unchanged=True,
        learning_rate_is_only_training_setting_change=True,
        optimizer_state='Stateless SR; both branches use the same complete parent physical weights and terminal PRNG.',
        selection_criterion='All three fast endpoints pass chi<=.1, C<=.03, E<=.002; report all paired differences regardless.',
        fixed_endpoint_updates=100, no_automatic_budget_extension=True,
        reference_labels_in_training=False, reserved_confirmation_measured=False, phase0_scientific_gate=False))


def run(output: Path) -> None:
    """Train all branches, seal all endpoints, then perform exhaustive audits."""
    protocol = json.loads((output/'protocol.json').read_text())
    verify_sources(protocol['source_sha256'])
    for name, digest in protocol['input_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('selection inputs changed')
    cfg = SizeFeatureConfig(**protocol['model'])
    units = CouplingUnits(**protocol['units'])
    for spec in protocol['runs']:
        folder = output/'runs'/spec['name']
        if (folder/'node.json').exists():
            checked_node(folder/'node.json')
            continue
        if (folder/'started.json').exists():
            raise RuntimeError('partial branch preserved; fresh protocol required')
        parent = Path(spec['parent'])
        node = checked_node(parent)
        if sha(parent) != spec['parent_sha256'] or node['checkpoint_sha256'] != spec['checkpoint_sha256']:
            raise ValueError('paired parent changed')
        p, _, key, _ = load_checkpoint(Path(node['checkpoint_path']))
        options = Options(**spec['options'])
        write_json(folder/'started.json', dict(protocol_sha256=sha(output/'protocol.json')))
        began = time.perf_counter()
        def callback(params, state, rng, history):
            if state['step'] == 1 or state['step'] % 25 == 0:
                print(spec['name'], state['step'], 'seconds', round(time.perf_counter()-began, 2),
                      'step_norm', history[-1]['step_norm'], flush=True)
        p, state, key, history = train(p, key, 16, spec['schedule'], cfg, units, options, callback)
        check_cost(state['work'], spec['cost'])
        save_node(folder, p, state, key, signature=model_signature(cfg, 16),
                  protocol_path=output/'protocol.json', train_points=spec['train_points'], history=history,
                  parents=[parent], stage='matched_learning_rate_warm_branch')
        write_json(folder/'timing.json', dict(training_seconds=time.perf_counter()-began))
    write_json(output/'endpoint_seal.json', dict(nodes={r['name']:sha(output/'runs'/r['name']/'node.json')
               for r in protocol['runs']}, reference_opened=False))
    rows = []
    for spec in protocol['runs']:
        folder = output/'runs'/spec['name']
        node = checked_node(folder/'node.json')
        if not (folder/'audit.json').exists():
            p, _, _, _ = load_checkpoint(folder/'checkpoint.npz')
            points = []
            for index, g in enumerate(protocol['primary_points']):
                points.append(audit(p, 16, g, cfg, folder/f'point_{index}.npz'))
                print('audit', spec['name'], index+1, flush=True)
            write_json(folder/'audit.json', dict(points=points, checkpoint_sha256=node['checkpoint_sha256'],
                       protocol_sha256=sha(output/'protocol.json')))
        data = json.loads((folder/'audit.json').read_text())
        rows.append(dict(seed=spec['seed'], arm=spec['arm'], primary=metrics(data['points']),
                         audit_sha256=sha(folder/'audit.json')))
    differences = []
    for seed in sorted({r['seed'] for r in rows}):
        slow = next(r for r in rows if r['seed']==seed and r['arm']=='slow')
        fast = next(r for r in rows if r['seed']==seed and r['arm']=='fast')
        paths = [output/'runs'/f'{arm}_seed_{seed}/checkpoint.npz' for arm in ('slow', 'fast')]
        keys = [load_checkpoint(path)[2] for path in paths]
        if not np.array_equal(*keys):
            raise ValueError('paired terminal PRNG differs')
        differences.append(dict(seed=seed, chi_improvement=slow['primary']['chi']-fast['primary']['chi'],
                                correlation_change=fast['primary']['correlation']-slow['primary']['correlation'],
                                terminal_prng_equal=True))
    endpoints = [output/'runs'/r['name']/'node.json' for r in protocol['runs']]
    nodes = [checked_node(p) for p in endpoints]
    work = {k:sum(n['exclusive_work'].get(k, 0) for n in nodes) for k in (
        'updates', 'task_evaluations', 'hamiltonian_actions', 'source_actions', 'born_samples', 'sr_correction_solves')}
    write_json(output/'review.json', dict(rows=rows, paired_differences=differences, exclusive_training_cost=work,
        inclusive_training_cost=inclusive_cost(endpoints),
        audit_cost=dict(basis_rows=120*65536, hamiltonian_actions=240, source_actions=120, born_samples=0),
        all_fast_development_observables_pass=all(r['primary']['observable_pass'] for r in rows if r['arm']=='fast'),
        all_slow_development_observables_pass=all(r['primary']['observable_pass'] for r in rows if r['arm']=='slow'),
        protocol_sha256=sha(output/'protocol.json'), reserved_confirmation_measured=False, phase0_scientific_gate=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.parent is None:
            parser.error('--parent required')
        prepare(args.output, args.parent)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
