"""Fixed-feature L32 factorial control of learning rate and update budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time

import jax
import numpy as np

from fnqs.artifacts import load_checkpoint, save_checkpoint, write_json
from fnqs.pilots.coordinate_sr import train
from fnqs.pilots.engine import Options, model_signature
from fnqs.pilots.lineage import ROOT, check_cost, cost, inclusive_cost, sha, validate_splits, verify_sources
from fnqs.scripts.run_frozen_confirmation import sampled_audit, verified as verify_parent
from fnqs.scripts.run_l16_pair import checked_node, freeze
from fnqs.variants.coupling_units import CouplingUnits
from fnqs.variants.size_features import SizeFeatureConfig


def segments(budgets: list[int]) -> list[tuple[int, int]]:
    """Return exclusive update counts along one cumulative-budget trajectory."""
    if (not budgets or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError('budgets must be increasing distinct positive integers')
    return [(end, end - start) for start, end in zip([0] + budgets[:-1], budgets)]


def exclusive_state(state: dict, previous_work: dict | None) -> dict:
    """Account for a continuous trajectory segment without changing its weights."""
    previous_work = previous_work or {key:0 for key in state['work']}
    work = {key:state['work'][key]-previous_work[key] for key in state['work']}
    if any(value < 0 for value in work.values()):
        raise ValueError('cumulative work must not decrease')
    return {**state, 'step':work['updates'], 'work':work}


def prepare(output: Path, parent: Path) -> None:
    """Freeze paired warm controls; all v18 opened fields are development data."""
    parent = parent.resolve()
    old = verify_parent(parent)
    review_path = ROOT / 'fnqs/results/progression_v18.json'
    review = json.loads(review_path.read_text())
    if review['phase0_scientific_gate']:
        raise ValueError('response repair is not justified by a passed gate')
    cfg = SizeFeatureConfig(**old['model'])
    if cfg.hidden_dim != 16 or cfg.size_features != 'legacy':
        raise ValueError('the fixed feature/width experiment requires the v18 legacy width16 parent')
    options = Options(**old['stages'][-1]['options'])
    if options.response_weight != 0:
        raise ValueError('energy-only control required')
    # These points have already been opened in v18. Never relabel them as a
    # method-independent confirmation of a subsequently selected remedy.
    development = old['reserved_points']
    future_reserved = [[1., 0., 0., float(.9503125 + .0025 * i)] for i in range(40)]
    validate_splits(old['train_points'], development, future_reserved)
    budgets = [100, 400]
    rates = [2., 4., 8.]
    runs = []
    for seed in old['seeds']:
        node_path = parent / 'runs' / f'L32_seed_{seed}' / 'fast/node.json'
        node = checked_node(node_path)
        if node['signature'] != model_signature(cfg, 32):
            raise ValueError('parent size/feature signature mismatch')
        for rate in rates:
            opt = replace(options, learning_rate=rate)
            runs.append(dict(name=f'lr{int(rate)}_seed_{seed}', seed=seed, learning_rate=rate,
                parent=str(node_path), parent_sha256=sha(node_path),
                checkpoint_sha256=node['checkpoint_sha256'], options=asdict(opt),
                schedule=[old['train_points']] * budgets[-1],
                exclusive_cost=cost([old['train_points']] * budgets[-1], opt.samples)))
    declaration = dict(kind='fixed_L32_learning_rate_budget_factorial_v19', n=32,
        seeds=old['seeds'], learning_rates=rates, budgets=budgets, runs=runs,
        model=asdict(cfg), coupling_units=old['coupling_units'], train_points=old['train_points'],
        development_points=development, development_is_opened_v18_reserved=True,
        reserved_points=future_reserved, reserved_confirmation_measured=False,
        audit=dict(born_samples=32768, energy_rows=2048, fd_rows=1024, chunk=256, bootstrap=200,
            seed_namespace=0xC01932, chi_gate=.1, correlation_gate=.03, energy_gate=.002,
            uncertainty_multiplier=2, common_sampling_key_across_paired_models=True),
        response_audit=dict(samples=8192, chunk=128, indices=[0, 19, 39], bootstrap=200,
                            seed_namespace=0xAE1932, diagnostic_not_chi_accuracy_certificate=True),
        selection=dict(primary='worst development chi relative error plus 2SE; all seeds individually',
            secondary='chi RMS, complete correlations, energy, finite-L peak and clipping fraction',
            candidate='one common learning rate at step400 passes all three seeds and all original observable gates',
            meaningful_paired_budget_improvement=.05,
            negative_result='if every long-budget rate fails, this fixed recipe has not been repaired within declared range'),
        budget_interpretation='100 and 400 are additional updates after each complete v18 endpoint',
        budget_endpoints_are_nested=True, learning_rate_only_between_trajectory_setting_change=True,
        continuous_optimizer_chart=True, intermediate_checkpoints_do_not_restart_training=True,
        unchanged=['features', 'width', 'energy-only loss', 'coupling units', 'task grouping',
                   'samples/task', 'SR shift/tolerance/maxiter', 'clipping norm'],
        optimizer_state='stateless SR; same full parent weights and terminal PRNG in every rate arm',
        no_adaptive_budget_extension=True, exact_labels_in_training=False,
        phase0_scientific_gate=False, confirmation_is_not_part_of_this_development_experiment=True,
        input_sha256={str(review_path):sha(review_path), str(parent/'protocol.json'):sha(parent/'protocol.json')})
    freeze(output, declaration)
    protocol = json.loads((output/'protocol.json').read_text())
    for name in ('fnqs/scripts/run_l32_controls.py', 'fnqs/scripts/review_l32_controls.py',
                 'fnqs/scripts/run_frozen_confirmation.py', 'fnqs/scripts/review_frozen_confirmation.py'):
        protocol['source_sha256'][name] = sha(ROOT/name)
        target = output/'runtime'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(output/'protocol.json', protocol)


def verified(output: Path) -> dict:
    """Validate the frozen recipe, method-selection inputs and parent sources."""
    protocol = json.loads((output/'protocol.json').read_text())
    verify_sources(protocol['source_sha256'])
    for name, digest in protocol['input_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('method selection or parent declaration changed')
    return protocol


def train_runs(output: Path, selected_seed: int | None = None) -> None:
    """Run a declared seed subset without changing the global protocol."""
    protocol = verified(output)
    if selected_seed is not None and selected_seed not in protocol['seeds']:
        raise ValueError('undeclared seed')
    cfg = SizeFeatureConfig(**protocol['model'])
    units = CouplingUnits(**protocol['coupling_units'])
    for spec in protocol['runs']:
        if selected_seed is not None and spec['seed'] != selected_seed:
            continue
        parent = Path(spec['parent'])
        node = checked_node(parent)
        if sha(parent) != spec['parent_sha256'] or node['checkpoint_sha256'] != spec['checkpoint_sha256']:
            raise ValueError('paired warm parent changed')
        options = Options(**spec['options'])
        run_folder = output/'runs'/spec['name']
        final_path = run_folder/f"step_{protocol['budgets'][-1]}/node.json"
        if final_path.exists():
            for budget in protocol['budgets']:
                checked_node(run_folder/f'step_{budget}/node.json')
            continue
        if (run_folder/'started.json').exists():
            raise RuntimeError('partial trajectory retained; no silent restart or budget extension')
        params, _, key, _ = load_checkpoint(Path(node['checkpoint_path']))
        write_json(run_folder/'started.json', dict(protocol_sha256=sha(output/'protocol.json')))
        started_at = time.perf_counter()
        previous_work = None
        previous_budget = 0

        def callback(p, state, rng, history):
            nonlocal parent, previous_work, previous_budget
            step = state['step']
            if step % 50 == 0 or step == 1:
                save_checkpoint(run_folder/'partial.npz', p, state, rng,
                    dict(history=history, protocol_sha256=sha(output/'protocol.json')))
                print(spec['name'], step, 'seconds', round(time.perf_counter()-started_at, 1),
                      'step_norm', history[-1]['step_norm'], flush=True)
            if step in protocol['budgets']:
                folder = run_folder/f'step_{step}'
                local_state = exclusive_state(state, previous_work)
                local_history = [{**row, 'step':index+1} for index, row in enumerate(history[previous_budget:step])]
                check_cost(local_state['work'], cost(spec['schedule'][previous_budget:step], options.samples))
                from fnqs.pilots.lineage import save_node
                save_node(folder, p, local_state, rng, signature=model_signature(cfg, 32),
                    protocol_path=output/'protocol.json', train_points=protocol['train_points'],
                    history=local_history, parents=[parent], stage=f'L32_budget_step_{step}')
                write_json(folder/'timing.json', dict(elapsed_branch_training_seconds=time.perf_counter()-started_at,
                    elapsed_since_branch_start=True, includes_previous_segments=previous_budget>0))
                previous_work = dict(state['work'])
                previous_budget = step
                parent = folder/'node.json'

        _, state, _, _ = train(params, key, 32, spec['schedule'], cfg, units, options, callback)
        check_cost(state['work'], spec['exclusive_cost'])
    if selected_seed is not None:
        nodes = {f"{spec['name']}/step_{budget}":sha(output/'runs'/spec['name']/f'step_{budget}/node.json')
                 for spec in protocol['runs'] if spec['seed'] == selected_seed for budget in protocol['budgets']}
        write_json(output/f'seed_{selected_seed}_complete.json', dict(nodes=nodes, new_references_opened=False))


def seal(output: Path) -> dict:
    """Require all declared rate/budget endpoints before any new measurement."""
    protocol = verified(output)
    nodes = {}
    for spec in protocol['runs']:
        for budget in protocol['budgets']:
            path = output/'runs'/spec['name']/f'step_{budget}/node.json'
            checked_node(path)
            nodes[f"{spec['name']}/step_{budget}"] = sha(path)
    receipt = dict(nodes=nodes, protocol_sha256=sha(output/'protocol.json'), new_references_opened=False)
    path = output/'endpoint_seal.json'
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError('endpoint seal changed')
    else:
        write_json(path, receipt)
    return receipt


def measurement_specs(protocol: dict) -> list[dict]:
    """Include equal-audit-budget warm baselines and every declared endpoint."""
    specs = []
    for seed in protocol['seeds']:
        run = next(run for run in protocol['runs'] if run['seed'] == seed)
        specs.append(dict(name=f'baseline_seed_{seed}', seed=seed, learning_rate=None, budget=0,
                          node=run['parent']))
    for run in protocol['runs']:
        for budget in protocol['budgets']:
            specs.append(dict(name=f"{run['name']}_step_{budget}", seed=run['seed'],
                learning_rate=run['learning_rate'], budget=budget,
                node=f"runs/{run['name']}/step_{budget}/node.json"))
    return specs


def measure(output: Path) -> None:
    """Equal independent Born audits of baseline and all development endpoints."""
    protocol = verified(output)
    seal(output)
    cfg = SizeFeatureConfig(**protocol['model'])
    receipt = []
    for spec in measurement_specs(protocol):
        path = Path(spec['node'])
        if not path.is_absolute():
            path = output/path
        node = checked_node(path)
        params, _, _, _ = load_checkpoint(Path(node['checkpoint_path']))
        folder = output/'measurements'/spec['name']
        points = []
        for index, couplings in enumerate(protocol['development_points']):
            archive = folder/f'point_{index}.npz'
            row_path = archive.with_suffix('.json')
            if row_path.exists():
                row = json.loads(row_path.read_text())
                if row['archive_sha256'] != sha(archive) or row['checkpoint_sha256'] != node['checkpoint_sha256']:
                    raise ValueError('measurement archive or model changed')
            else:
                key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
                    jax.random.PRNGKey(spec['seed']), protocol['audit']['seed_namespace']), 32), index)
                row = sampled_audit(params, 32, couplings, cfg, key, archive,
                                    protocol['audit'], spec['seed']+index+32)
                if max(row['fd_score_relative_rms_error']) > 5e-4:
                    raise RuntimeError('physical coupling derivative finite-difference guard failed')
                row.update(checkpoint_sha256=node['checkpoint_sha256'], protocol_sha256=sha(output/'protocol.json'),
                    split='development_previous_v18_opened_fields', draw_key=np.asarray(key).tolist(),
                    paired_bootstrap_seed=spec['seed']+index+32)
                write_json(row_path, row)
            points.append(row)
        summary = dict(chi=max(row['chi_upper_error'] for row in points),
            chi_rms=float(np.sqrt(np.mean([row['chi_relative_error']**2 for row in points]))),
            correlation=max(row['correlation_upper_error'] for row in points),
            energy=max(row['energy_upper_error'] for row in points))
        summary['passed'] = (summary['chi'] <= .1 and summary['correlation'] <= .03 and summary['energy'] <= .002)
        write_json(folder/'audit.json', dict(points=points, checkpoint_sha256=node['checkpoint_sha256']))
        receipt.append({**spec, **summary, 'audit_sha256':sha(folder/'audit.json')})
        print('audit', spec['name'], 'chi_upper', round(summary['chi'], 5), flush=True)
    endpoints = [output/'runs'/run['name']/'step_400/node.json' for run in protocol['runs']]
    write_json(output/'measurements.json', dict(rows=receipt, inclusive_cost=inclusive_cost(endpoints),
        protocol_sha256=sha(output/'protocol.json'), development_only=True, reserved_confirmation_measured=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'seal', 'measure'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parent', type=Path, default=ROOT/'fnqs/results/phase0_frozen_confirmation_v18')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'prepare':
        prepare(output, args.parent)
    elif args.command == 'train':
        train_runs(output, args.seed)
    elif args.command == 'seal':
        seal(output)
    else:
        measure(output)


if __name__ == '__main__':
    main()
