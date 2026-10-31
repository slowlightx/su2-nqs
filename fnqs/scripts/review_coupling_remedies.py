"""Verify saved endpoints/rows with independent parity actions; plot results."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from fnqs.artifacts import write_json,load_checkpoint
from fnqs.diagnostics.response import summarize_response
from fnqs.pilots.factorial import metrics,decision
from fnqs.pilots.lineage import verify_sources,read_node,sha,check_cost
from fnqs.hamiltonians import IsingCouplings
from fnqs.sectors import parity_hamiltonian,lift_parity_state


def verify_stage(directory: Path) -> dict:
    """Recompute weighted moments and verify every archive against sector H."""
    protocol=json.loads((directory/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    review=json.loads((directory/'review.json').read_text())
    if review['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('review protocol mismatch')
    cases=[]
    if 'trajectories' in protocol:
        for t in protocol['trajectories']:
            for step in t['milestones']:
                cases.append((directory/'trajectories'/t['name']/f'step_{step}',t['endpoint_costs'][str(step)],protocol['primary_points'],protocol['secondary_points']))
    else:
        cases=[(directory/'runs'/r['name'],r['cost'],r['primary_points'],r['secondary_points']) for r in protocol['runs']]
    moment_error=0.;action_error=0.;corr_error=0.;fd_relative=0.;norm_error=0.;hashes={};count=0;cache={}
    seal=json.loads((directory/'endpoint_seal.json').read_text())['nodes']
    for folder,expected,primary,secondary in cases:
        node=folder/'node.json';data=read_node(node)[node.resolve()];check_cost(data['exclusive_work'],expected)
        if sha(node) not in seal.values():raise ValueError('endpoint not in pre-reference seal')
        path=folder/'audit.json';result=json.loads(path.read_text())
        if result['checkpoint_sha256']!=sha(folder/'checkpoint.npz') or result['protocol_sha256']!=sha(directory/'protocol.json'):
            raise ValueError('audit checkpoint binding mismatch')
        expected_points=[('primary',g) for g in primary]+[('secondary',g) for g in secondary]
        if len(result['points'])!=len(expected_points):raise ValueError('point count mismatch')
        for i,(point,(split,g)) in enumerate(zip(result['points'],expected_points)):
            if point['split']!=split or point['couplings']!=g or point['n']!=protocol['n']:raise ValueError('point labels mismatch')
            archive=folder/f'point_{i}.npz'
            if sha(archive)!=point['archive_sha256']:raise ValueError('Born archive changed')
            with np.load(archive,allow_pickle=False) as arrays:
                psi=arrays['psi'];bits=arrays['bits'];raw={k:arrays[k] for k in ('score','energy','h_score','source','fd_scores')}
                if len(psi)!=4096 or not np.array_equal(arrays['couplings'],g):raise ValueError('basis labels mismatch')
            refpath=directory/'references'/('L12_'+'_'.join(str(x) for x in g)+'.npz')
            if sha(refpath)!=point['reference_sha256']:raise ValueError('reference changed')
            replay=summarize_response(raw,psi**2)
            for key in ('chi_total','energy_total','energy_variance','response_relative_residual','hf_derivative_defect'):
                moment_error=max(moment_error,abs(replay[key]-point[key]))
            norm_error=max(norm_error,abs(float(np.sum(psi**2))-1))
            fd_relative=max(fd_relative,max(replay['fd_score_relative_rms_error']))
            label=tuple(g)
            if label not in cache:
                cache[label]=(parity_hamiltonian(12,IsingCouplings(*g),1),
                              parity_hamiltonian(12,IsingCouplings(0,0,0,1),1))
            h,v=cache[label];half=2048
            if np.max(abs(psi-psi[::-1]))>1e-13 or np.max(abs(raw['score']-raw['score'][::-1]))>1e-13:
                raise ValueError('Z2 state or tangent violation')
            for key,operator,vector in [('energy',h,psi),('h_score',h,psi*raw['score']),('source',v,psi)]:
                lifted=lift_parity_state(operator@(np.sqrt(2)*vector[:half]),12,1)
                action_error=max(action_error,float(np.max(abs(lifted-psi*raw[key]))))
            z=2*bits.astype(np.int32)-1
            corr=np.array([np.sum(psi**2*np.mean(z*np.roll(z,-r,axis=1),axis=1)) for r in range(12)])
            corr_error=max(corr_error,float(np.max(abs(corr-point['correlations']))))
            hashes[str(archive)]=sha(archive);count+=1
        hashes[str(node)]=sha(node);hashes[str(path)]=sha(path)
    if moment_error>1e-12 or corr_error>1e-12 or action_error>2e-11 or norm_error>1e-10:
        raise RuntimeError('saved-data exact validation failed')
    report=dict(complete=True,points=count,moment_max_discrepancy=moment_error,moment_tolerance=1e-12,
        independent_sector_action_max_discrepancy=action_error,action_tolerance=2e-11,
        full_correlation_max_discrepancy=corr_error,correlation_tolerance=1e-12,
        normalization_max_error=norm_error,normalization_tolerance=1e-10,
        fd_score_relative_rms_max=fd_relative,fd_guard=5e-4,validated_sha256=hashes,
        validation_cost=dict(born_samples=0,hamiltonian_actions=2*count,source_actions=count),
        protocol_sha256=sha(directory/'protocol.json'),review_sha256=sha(directory/'review.json'),
        validation_source_sha256=sha(Path(__file__)))
    write_json(directory/'replay_validation.json',report);return report


def plot(base: Path,probe: Path,family: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b=json.loads((base/'review.json').read_text());p=json.loads((probe/'review.json').read_text());f=json.loads((family/'review.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for seed,color in zip((113,127,139),('C0','C1','C2')):
        for window,marker in [('narrow','o'),('wide','x')]:
            r=sorted([r for r in b['rows'] if r['seed']==seed and r['window']==window],key=lambda r:r['step'])
            axes[0].plot([r['step'] for r in r],[100*r['primary']['chi'] for r in r],marker+'-',color=color,
                         label=f'{seed} {window}')
        row=next(r for r in p['rows'] if r['seed']==seed)
        axes[1].plot([0,1],[100*row['sequential_primary']['chi'],100*row['primary']['chi']],'o-',color=color,label=str(seed))
        shared=next(r for r in f['rows'] if r['run']==f'ABC_seed_{seed}')
        separate=[next(r for r in f['rows'] if r['run']==f'{fam}_seed_{seed}') for fam in 'ABC']
        axes[2].plot(range(3),[100*shared['families'][fam]['primary']['chi'] for fam in 'ABC'],'o-',color=color,label=f'ABC {seed}')
        axes[2].plot(range(3),[100*r['families'][fam]['primary']['chi'] for r,fam in zip(separate,'ABC')],'x--',color=color)
    axes[0].set_title('Coverage x optimization budget');axes[0].set_xlabel('Updates');axes[0].legend(fontsize=7)
    axes[1].set_title('Same 600 tasks / 38400 Born draws');axes[1].set_xticks([0,1],['Sequential 600','Joint 120']);axes[1].legend(fontsize=8)
    axes[2].set_title('Conditional joint-SR family retest');axes[2].set_xticks(range(3),list('ABC'));axes[2].legend(fontsize=8)
    for ax in axes:ax.set_ylabel('Worst central chi relative error (%)');ax.grid(alpha=.2)
    axes[2].axhline(10,color='grey',ls=':',lw=1)
    fig.tight_layout();path=family/'remedy_comparison.png';fig.savefig(path,dpi=170);plt.close(fig)
    write_json(family/'figure.json',dict(figure_sha256=sha(path),
        input_sha256={str(p/'review.json'):sha(p/'review.json') for p in (base,probe,family)},
        plot_source_sha256=sha(Path(__file__))))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factorial',type=Path,required=True);parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--families',type=Path,required=True)
    args=parser.parse_args()
    for path in (args.factorial,args.probe,args.families):
        report=verify_stage(path);print(path,'validated',report['points'],'actions',report['independent_sector_action_max_discrepancy'],flush=True)
    plot(args.factorial,args.probe,args.families)

if __name__=='__main__':main()
