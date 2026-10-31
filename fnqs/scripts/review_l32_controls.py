"""Independently replay and compare the predeclared L32 budget/rate controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fnqs.artifacts import load_checkpoint, write_json
from fnqs.diagnostics.frozen_stability import local_peak
from fnqs.diagnostics.response import bootstrap_response
from fnqs.exact import TFIMReference
from fnqs.pilots.engine import draw
from fnqs.pilots.lineage import inclusive_cost, sha
from fnqs.pilots.measurements import sampled_records
from fnqs.scripts.review_frozen_confirmation import independent_local_energy
from fnqs.scripts.run_l16_pair import checked_node
from fnqs.scripts.run_l32_controls import measurement_specs, seal, verified
from fnqs.variants.size_features import SizeFeatureConfig


def replay(output: Path, protocol: dict) -> dict:
    """Recompute saved estimators and independently enumerate single-spin H flips."""
    cfg = SizeFeatureConfig(**protocol['model'])
    moment = action = correlation = covariance = fd_error = 0.
    bindings = {}
    count = 0
    for spec in measurement_specs(protocol):
        path = Path(spec['node'])
        if not path.is_absolute():
            path = output/path
        node = checked_node(path)
        params, _, _, _ = load_checkpoint(Path(node['checkpoint_path']))
        folder = output/'measurements'/spec['name']
        points = json.loads((folder/'audit.json').read_text())['points']
        if len(points) != len(protocol['development_points']):
            raise ValueError('development point count changed')
        for index, (row, g) in enumerate(zip(points, protocol['development_points'])):
            archive = folder/f'point_{index}.npz'
            if row['archive_sha256'] != sha(archive) or row['couplings'] != g or row['n'] != 32:
                raise ValueError('raw measurement binding or labels changed')
            if row['checkpoint_sha256'] != node['checkpoint_sha256'] or row['protocol_sha256'] != sha(output/'protocol.json'):
                raise ValueError('audit model/protocol binding changed')
            with np.load(archive, allow_pickle=False) as z:
                b, s, e, fd = z['bits'], z['score'], z['energy'], z['fd_scores']
                if (b.shape != (protocol['audit']['born_samples'], 32)
                        or not np.array_equal(z['couplings'], g) or not np.all((b == 0) | (b == 1))
                        or len(e) != protocol['audit']['energy_rows'] or len(fd) != protocol['audit']['fd_rows']):
                    raise ValueError('Born labels or fixed prefix budget changed')
                chi = np.mean((s-s.mean())**2)
                zz = 2*b.astype(float)-1
                c = np.stack([np.mean(zz*np.roll(zz, -r, axis=1), axis=1) for r in range(32)], axis=1)
                joint = np.column_stack(((s-s.mean())**2, c))
                centered = joint-joint.mean(axis=0)
                cov = centered.T@centered/(len(s)*(len(s)-1))
                covariance = max(covariance, float(np.max(abs(cov-z['joint_covariance']))))
                se = max(float(z['chi_bootstrap_replicates'].std(ddof=1)), float(np.sqrt(cov[0, 0])))
                moment = max(moment, abs(chi-row['chi_total']), abs(e.mean()-row['energy_total']),
                    abs(se-row['chi_se']), abs(e.std(ddof=1)/np.sqrt(len(e))-row['energy_se']))
                correlation = max(correlation, float(np.max(abs(c.mean(axis=0)-row['correlations']))))
                manual = independent_local_energy(params, b[:16], np.asarray(g), cfg)
                action = max(action, float(np.max(abs(manual-e[:16]))))
                ss = s[:len(fd)];ds = fd-fd.mean(axis=0)-(ss-ss.mean())[:, None]
                fd_error = max(fd_error, float(np.max(np.sqrt(np.mean(ds**2, axis=0)/ss.var()))))
            count += 1
            bindings[str(archive)] = sha(archive)
        print('replayed', spec['name'], flush=True)
    if moment > 2e-11 or action > 2e-11 or correlation > 1e-12 or covariance > 1e-10 or fd_error > 5e-4:
        raise RuntimeError('independent L32 estimator/action replay failed')
    result = dict(complete=True, points=count, moment_max_discrepancy=float(moment),
        operator_max_discrepancy=float(action), correlation_max_discrepancy=correlation,
        covariance_max_discrepancy=covariance, fd_relative_max=fd_error,
        tolerances=dict(moment=2e-11, operator=2e-11, correlation=1e-12, covariance=1e-10, fd_relative=5e-4),
        cost=dict(hamiltonian_actions=count, source_actions=0, born_samples=0, local_energy_rows=16*count),
        validated_sha256=bindings)
    write_json(output/'independent_replay.json', result)
    return result


def responses(output: Path, protocol: dict) -> dict:
    """Independent paired-key physical-response residual/HF diagnostics."""
    cfg = SizeFeatureConfig(**protocol['model']);options = protocol['response_audit'];rows = []
    for spec in measurement_specs(protocol):
        path = Path(spec['node'])
        if not path.is_absolute():
            path = output/path
        node = checked_node(path)
        params, _, _, _ = load_checkpoint(Path(node['checkpoint_path']))
        for index in options['indices']:
            archive = output/'response_audit'/spec['name']/f'point_{index}.npz'
            receipt = archive.with_suffix('.json')
            g = protocol['development_points'][index]
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if row['archive_sha256'] != sha(archive) or row['checkpoint_sha256'] != node['checkpoint_sha256']:
                    raise ValueError('response audit archive/model changed')
            else:
                key = jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
                    jax.random.PRNGKey(spec['seed']), options['seed_namespace']), 32), index)
                bits = draw(params, key, options['samples'], 32, jnp.asarray(g), cfg)
                blocks = [sampled_records(params, bits[i:i+options['chunk']], jnp.asarray(g), cfg)
                          for i in range(0, options['samples'], options['chunk'])]
                raw = {key:np.concatenate([np.asarray(block[key]) for block in blocks]) for key in blocks[0]}
                row = bootstrap_response(raw, spec['seed']+32+index, options['bootstrap'])
                if max(row['fd_score_relative_rms_error']) > 5e-4:
                    raise RuntimeError('independent physical response AD/FD guard failed')
                archive.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(archive, bits=np.asarray(bits), couplings=g, **raw)
                row.update(name=spec['name'], seed=spec['seed'], budget=spec['budget'],
                    learning_rate=spec['learning_rate'], couplings=g,
                    archive_sha256=sha(archive), checkpoint_sha256=node['checkpoint_sha256'])
                write_json(receipt, row)
            rows.append(row)
        print('response', spec['name'], flush=True)
    result = dict(rows=rows, residual_is_not_chi_error_bound=True,
        cost=dict(born_samples=len(rows)*options['samples'], hamiltonian_actions=2*len(rows), source_actions=len(rows)))
    write_json(output/'response_review.json', result)
    return result


def training_summary(node_path: Path, output: Path, cap: float) -> dict:
    """Include only this new branch, excluding its shared warm training parent."""
    records = []
    path = node_path
    while path.is_relative_to(output/'runs'):
        node = checked_node(path)
        records = node['history'] + records
        path = Path(node['parents'][0]['path'])
    norms = np.asarray([row['step_norm'] for row in records])
    return dict(updates=len(records), step_norm_median=float(np.median(norms)),
        clipping_fraction=float(np.mean(norms >= cap*(1-1e-10))),
        true_sr_residual_max=max(row['true_sr_residual'] for row in records),
        training_response_rms_tail=float(np.sqrt(np.mean([row['response_loss'] for row in records[-50:]]))))


def review(output: Path) -> dict:
    """Report all factorial cells; do not confuse development with confirmation."""
    protocol = verified(output);seal(output)
    data = json.loads((output/'measurements.json').read_text());rows = data['rows']
    replayed = replay(output, protocol)
    response = responses(output, protocol)
    fields = np.array([point[3] for point in protocol['development_points']])
    ref_peak = json.loads((Path(protocol['runs'][0]['parent']).parents[3]/'peak_reference_check.json').read_text())
    exact_peak = next(row['continuous_peak'] for row in ref_peak['references'] if row['n'] == 32)
    endpoints = []
    for row in rows:
        points = json.loads((output/'measurements'/row['name']/'audit.json').read_text())['points']
        row['peak'] = local_peak(fields, [point['chi_total'] for point in points], 7)
        row['peak']['reference'] = exact_peak
        row['peak']['signed_bias'] = row['peak']['h_peak']-exact_peak
        rr = [r for r in response['rows'] if r['name'] == row['name']]
        row['physical_response_max'] = max(r['response_relative_residual'] for r in rr)
        if row['budget']:
            path = output/row['node'];endpoints.append(path)
            opt = next(run['options'] for run in protocol['runs'] if run['name'] in row['name'])
            row['training'] = training_summary(path, output, opt['max_step'])
    comparisons = []
    for seed in protocol['seeds']:
        baseline = next(row for row in rows if row['seed'] == seed and row['budget'] == 0)
        for rate in protocol['learning_rates']:
            short, long = [next(row for row in rows if row['seed'] == seed and row['learning_rate'] == rate
                               and row['budget'] == budget) for budget in protocol['budgets']]
            short_key = load_checkpoint(output/short['node'].replace('node.json','checkpoint.npz'))[2]
            long_key = load_checkpoint(output/long['node'].replace('node.json','checkpoint.npz'))[2]
            comparisons.append(dict(seed=seed, learning_rate=rate,
                chi_upper_improvement_long_vs_short=short['chi']-long['chi'],
                chi_rms_improvement_long_vs_short=short['chi_rms']-long['chi_rms'],
                chi_upper_improvement_long_vs_baseline=baseline['chi']-long['chi'],
                chi_rms_improvement_long_vs_baseline=baseline['chi_rms']-long['chi_rms'],
                long_pass=long['passed'],
                budget_keys_distinct=not np.array_equal(short_key, long_key)))
    # Rate arms at the same cumulative budget consume exactly the same PRNG
    # schedule. Generated Born configurations may differ because theta differs.
    for seed in protocol['seeds']:
        for budget in protocol['budgets']:
            pair = [row for row in rows if row['seed'] == seed and row['budget'] == budget]
            keys = [load_checkpoint(output/row['node'].replace('node.json','checkpoint.npz'))[2] for row in pair]
            if not all(np.array_equal(keys[0], key) for key in keys[1:]):
                raise ValueError('matched-rate terminal PRNG differs')
    passing_rates = [rate for rate in protocol['learning_rates'] if all(row['passed'] for row in rows
                      if row['learning_rate'] == rate and row['budget'] == protocol['budgets'][-1])]
    useful_rates = [rate for rate in protocol['learning_rates'] if all(
        row['chi_upper_improvement_long_vs_short'] >= protocol['selection']['meaningful_paired_budget_improvement']
        for row in comparisons if row['learning_rate'] == rate)]
    graph = {}
    from fnqs.pilots.lineage import read_node
    for path in endpoints:
        graph.update(read_node(path))
    new_nodes = [node for path, node in graph.items() if path.is_relative_to(output/'runs')]
    keys = list(new_nodes[0]['exclusive_work'])
    new_work = {key:sum(node['exclusive_work'].get(key, 0) for node in new_nodes) for key in keys}
    expected = {key:sum(run['exclusive_cost'][key] for run in protocol['runs']) for key in protocol['runs'][0]['exclusive_cost']}
    for key, value in expected.items():
        if new_work[key] != value:
            raise ValueError('nested budgets double counted or actual training work differs')
    all_points = [point for row in rows for point in json.loads(
        (output/'measurements'/row['name']/'audit.json').read_text())['points']]
    result = dict(complete=True, rows=rows, comparisons=comparisons,
        common_learning_rates_passing_all_seeds=passing_rates,
        common_learning_rates_with_5pp_budget_improvement_all_seeds=useful_rates,
        fixed_recipe_repaired_in_development=bool(passing_rates),
        development_only=True, scientific_independent_confirmation_pass=False, phase0_scientific_gate=False,
        all_rate_terminal_prng_pairs_equal=True, exclusive_work=new_work,
        new_checkpoint_count=len(new_nodes), inclusive_cost=inclusive_cost(endpoints),
        validation_costs=dict(measurement={key:sum(point[key] for point in all_points)
            for key in ('born_samples','hamiltonian_actions','source_actions','basis_rows')},
            independent_replay=replayed['cost'], physical_response=response['cost']),
        independent_replay={key:value for key,value in replayed.items() if key != 'validated_sha256'},
        protocol_sha256=sha(output/'protocol.json'), measurement_sha256=sha(output/'measurements.json'),
        response_sha256=sha(output/'response_review.json'), reviewer_source_sha256=sha(Path(__file__)),
        no_automatic_extension=True,
        interpretation='A failed bounded factorial tests this GRU/energy-only recipe, not all FNQS architectures or losses.')
    write_json(output/'review.json', result)
    plot(output, protocol, rows)
    return result


def plot(output: Path, protocol: dict, rows: list[dict]) -> None:
    """Export per-seed budget/rate response errors and realized clipping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), layout='constrained')
    for column, seed in enumerate(protocol['seeds']):
        baseline = next(row for row in rows if row['seed'] == seed and row['budget'] == 0)
        for rate in protocol['learning_rates']:
            pair = sorted([row for row in rows if row['seed'] == seed and row['learning_rate'] == rate], key=lambda row:row['budget'])
            axes[0,column].plot([0]+[r['budget'] for r in pair], [100*baseline['chi']]+[100*r['chi'] for r in pair],
                                'o-', label=f'learning rate {rate:g}')
            axes[1,column].plot([r['budget'] for r in pair], [100*r['training']['clipping_fraction'] for r in pair], 'o-')
        axes[0,column].axhline(10, color='black', ls='--', lw=.8)
        axes[0,column].set(title=f'L32, seed {seed}', ylabel='worst chi relative error + 2SE (%)')
        axes[1,column].set(xlabel='additional updates after v18 parent', ylabel='updates at norm cap (%)')
        axes[0,column].legend(fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.savefig(output/'budget_rate_controls.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    result = review(parser.parse_args().output.resolve())
    print(json.dumps(dict(complete=result['complete'], candidate_rates=result['common_learning_rates_passing_all_seeds'])))
