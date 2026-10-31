"""Replay saved Born rows and report paired fixed-size pilot results/costs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import jax
import numpy as np
from fnqs.artifacts import write_json,load_checkpoint
from fnqs.diagnostics.response import bootstrap_response
from fnqs.observables import sampled_correlations
from fnqs.pilots.lineage import sha,read_node,inclusive_cost,check_cost,verify_holdout


def analyze(directory: Path, transfer: Path) -> dict:
    protocol_path=directory/'protocol.json';p=json.loads(protocol_path.read_text())
    rows=[]; endpoints=[]; hashes={}; largest=0.; counts={"born_samples":0,"basis_rows":0,"hamiltonian_actions":0,"source_actions":0}
    for run in p['runs']:
        folder=directory/'runs'/run['name'];node=folder/'node.json';nodes=read_node(node)
        check_cost(nodes[node.resolve()]['exclusive_work'],run['cost'])
        endpoints.append(node)
        auditpath=folder/'audit.json';audit=json.loads(auditpath.read_text())
        if audit['checkpoint_sha256']!=sha(folder/'checkpoint.npz') or audit['protocol_sha256']!=sha(protocol_path):
            raise ValueError('audit endpoint identity mismatch')
        if len(audit['points'])!=len(run['audit_points']):raise ValueError('audit grid count mismatch')
        for i,(point,expected) in enumerate(zip(audit['points'],run['audit_points'])):
            if point['couplings']!=expected['couplings'] or point['split']!=expected['split'] or point['n']!=run['n']:
                raise ValueError('audit labels mismatch')
            for k in counts:counts[k]+=point[k]
            if run['track']=='phase0':
                archive=folder/f'point_{i}_rows.npz'
                with np.load(archive,allow_pickle=False) as data:
                    if data['bits'].shape!=(4096,16) or not np.array_equal(data['couplings'],expected['couplings']):
                        raise ValueError('raw Born labels/counts mismatch')
                    raw={k:data[k] for k in ('score','energy','h_score','source','fd_scores')}
                    replay=bootstrap_response(raw,point['audit_seed']+100000,200)
                    c,se=sampled_correlations(data['bits'])
                for k in ('chi_total','response_relative_residual','hf_derivative_defect'):
                    largest=max(largest,abs(point[k]-replay[k]),abs(point['bootstrap_se'][k]-replay['bootstrap_se'][k]))
                largest=max(largest,abs(point['c_half']-c[8]),abs(point['c_half_se']-se[8]))
                hashes[str(archive)]=sha(archive)
        hashes[str(auditpath)]=sha(auditpath);hashes[str(node)]=sha(node)
        for family in sorted({x['family'] for x in audit['points']}):
            development=[x for x in audit['points'] if x['family']==family and x['split']=='development']
            rows.append(dict(track=run['track'],run=run['name'],seed=run['seed'],family=family,
                chi=max(x['chi_relative_error'] for x in development),
                chi_plus_2se=max(x.get('chi_relative_error_plus_2se',x['chi_relative_error']) for x in development),
                rho=max(x['response_relative_residual'] for x in development),
                energy=max(x['energy_per_site_error'] for x in development),
                correlation=max(x.get('c_half_absolute_error_plus_2se',x.get('c_half_absolute_error',0)) for x in development)))
    if largest>1e-12:raise ValueError('saved row replay discrepancy')
    pairs=[]
    for seed in (113,127,139):
        for family in 'ABC':
            shared=next(r for r in rows if r['run']==f'ABC_seed_{seed}' and r['family']==family)
            separate=next(r for r in rows if r['run']==f'{family}_seed_{seed}')
            pairs.append(dict(seed=seed,family=family,shared_chi=shared['chi'],separate_chi=separate['chi'],
                              shared_minus_separate=shared['chi']-separate['chi']))
    repair=[]
    for seed in (71,83,97):
        path=directory/'parents'/f'seed_{seed}'/'audit.json';base=json.loads(path.read_text())
        hashes[str(path)]=sha(path)
        parent=directory/'parents'/f'seed_{seed}'/'node.json'
        if base['checkpoint_sha256']!=sha(parent.parent/'checkpoint.npz'):raise ValueError('baseline parent mismatch')
        for i,point in enumerate(base['points']):
            archive=parent.parent/f'point_{i}_rows.npz'
            with np.load(archive,allow_pickle=False) as data:
                raw={k:data[k] for k in ('score','energy','h_score','source','fd_scores')}
                replay=bootstrap_response(raw,point['audit_seed']+100000,200)
            largest=max(largest,abs(replay['chi_total']-point['chi_total']),abs(replay['response_relative_residual']-point['response_relative_residual']))
            hashes[str(archive)]=sha(archive)
            for k in counts:counts[k]+=point[k]
        repair.append(dict(seed=seed,baseline_chi_plus_2se=max(x['chi_relative_error_plus_2se'] for x in base['points']),
            baseline_rho=max(x['response_relative_residual'] for x in base['points']),
            energy=next(r for r in rows if r['run']==f'energy_seed_{seed}'),
            response=next(r for r in rows if r['run']==f'response_seed_{seed}')))
    if largest>1e-12:raise ValueError('baseline replay discrepancy')
    tc=json.loads((transfer/'summary.json').read_text())
    tp=json.loads((transfer/'protocol.json').read_text())
    transfer_nodes=[]
    for seed in tp['seeds']:
        for fold in tp['folds']:
            base=transfer/f"held_{fold['held_family']}"/f'seed_{seed}'
            verify_holdout([base/'pretrain/node.json'],fold['held_family'])
            seal=json.loads((base/'endpoint_seal.json').read_text())
            for name,digest in seal['nodes'].items():
                if sha(Path(name))!=digest:raise ValueError('transfer endpoint seal changed')
                read_node(Path(name));transfer_nodes.append(Path(name))
    if inclusive_cost(transfer_nodes)!=tc['aggregate_training_cost']:raise ValueError('transfer inclusive cost mismatch')
    exclusive={k:sum(read_node(n)[n.resolve()]['exclusive_work'].get(k,0) for n in endpoints) for k in counts}
    report=dict(kind='fixed_size_pilot_review_v11',complete_runs=len(endpoints),expected_runs=18,rows=rows,
        phase1_pairs=pairs,phase0_pairs=repair,exclusive_training_cost=exclusive,
        inclusive_parameter_ancestor_training_cost=inclusive_cost(endpoints),audit_cost=counts,
        phase2_runner_verified=True,phase2_canary_cost=tc['aggregate_training_cost'],
        artifacts_sha256=hashes,protocol_sha256=sha(protocol_path),transfer_summary_sha256=sha(transfer/'summary.json'),
        analysis_source_sha256=sha(Path(__file__)),largest_saved_row_replay_discrepancy=largest,replay_tolerance=1e-12,
        python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,
        phase1_capacity_limit='One shared model versus three family models; equal queries/draws, unequal aggregate capacity/update counts.',
        cost_limits='Logical H actions are counted; differentiation FLOPs and historical method-selection research are not ancestral training work.',
        phase0_scientific_gate=False,universality_established=False,reserved_confirmation_measured=False,
        no_new_born_draws_or_hamiltonian_actions_in_saved_row_replay=True)
    write_json(directory/'review.json',report)
    return report


def plot(directory: Path, report: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.7))
    for seed,color in zip((113,127,139),('C0','C1','C2')):
        pairs=[p for p in report['phase1_pairs'] if p['seed']==seed]
        axes[0].plot(range(3),[100*p['shared_chi'] for p in pairs],'o-',color=color,label=f'ABC {seed}')
        axes[0].plot(range(3),[100*p['separate_chi'] for p in pairs],'x--',color=color)
    axes[0].set_xticks(range(3),list('ABC'));axes[0].set_title('Phase 1: worst development chi error')
    axes[0].set_ylabel('Relative error (%)');axes[0].legend(fontsize=8)
    for i,(metric,title) in enumerate((('chi_plus_2se','Phase 0: chi error + 2SE'),('rho','Phase 0: response residual')),1):
        for row,color in zip(report['phase0_pairs'],('C0','C1','C2')):
            baseline=row['baseline_chi_plus_2se' if metric=='chi_plus_2se' else 'baseline_rho']
            axes[i].plot(range(3),100*np.array([baseline,row['energy'][metric],row['response'][metric]]),'o-',color=color,label=str(row['seed']))
        axes[i].set_xticks(range(3),['Parent','Energy','Response']);axes[i].set_ylabel('%');axes[i].set_title(title)
        if i==1:axes[i].axhline(10,color='grey',ls=':',label='original chi threshold')
        axes[i].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory/'comparison.png',dpi=180);plt.close(fig)
    write_json(directory/'figure.json',dict(input_sha256=sha(directory/'review.json'),
        figure_sha256=sha(directory/'comparison.png'),plot_source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--transfer',type=Path,required=True)
    args=parser.parse_args();report=analyze(args.directory,args.transfer);plot(args.directory,report)
    print({'complete_runs':report['complete_runs'],'row_replay_error':report['largest_saved_row_replay_discrepancy']})
