"""Independent sparse-Pauli replay of sealed L16 raw audit vectors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from fnqs.artifacts import write_json
from fnqs.pilots.lineage import sha,verify_sources
from fnqs.scripts.run_l16_pair import checked_node,metrics


def sparse_tfim(n: int=16) -> tuple:
    """Construct ZZ diagonal and -sum X using integer labels, independently."""
    d=1<<n;labels=np.arange(d);diagonal=np.zeros(d)
    for bit in range(n):
        mismatch=((labels>>bit)&1)^((labels>>((bit+1)%n))&1)
        diagonal+=2*mismatch-1
    rows=np.tile(labels,n);cols=np.concatenate([labels^(1<<bit) for bit in range(n)])
    v=csr_matrix((-np.ones(n*d),(rows,cols)),shape=(d,d))
    return diagonal,v


def validate(directory: Path) -> dict:
    """Verify original archives, vectors, moments, costs and endpoint seals."""
    protocol=json.loads((directory/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    endpoint_seal=json.loads((directory/'endpoint_seal.json').read_text())['nodes']
    folders=[]
    for run in protocol['runs']:
        for step in run.get('milestones',[None]):
            folder=directory/'runs'/run['name']
            key=run['name']
            if step is not None:folder=folder/f'step_{step}';key=f"{key}/step_{step}"
            if endpoint_seal.get(key)!=sha(folder/'node.json'):raise ValueError('endpoint seal changed')
            folders.append(folder)
    diagonal,v=sparse_tfim();action_error=moment_error=corr_error=norm_error=fd_error=0.;hashes={};count=0
    for folder in folders:
        node=checked_node(folder/'node.json');data=json.loads((folder/'audit.json').read_text())
        if data['checkpoint_sha256']!=node['checkpoint_sha256'] or data['protocol_sha256']!=sha(directory/'protocol.json'):
            raise ValueError('checkpoint audit binding mismatch')
        if [r['couplings'] for r in data['points']]!=protocol['primary_points']:raise ValueError('audit fields changed')
        for i,point in enumerate(data['points']):
            archive=folder/f'point_{i}.npz'
            if point['archive_sha256']!=sha(archive):raise ValueError('raw audit archive changed')
            with np.load(archive,allow_pickle=False) as raw:
                psi=raw['psi'];s=raw['score'];bits=raw['bits'];g=raw['couplings'];fd=raw['fd_scores']
                saved={k:raw[k] for k in ('energy','h_score','source')}
            if not np.array_equal(g,point['couplings']) or not np.array_equal(g[:3],[1,0,0]):raise ValueError('non-TFIM labels')
            labels=np.arange(65536)
            expected=((labels[:,None]>>np.arange(15,-1,-1))&1).astype(np.int32)
            if not np.array_equal(bits,expected):raise ValueError('basis bit convention changed')
            norm=float(psi@psi);p=psi**2/norm;mu=float(p@s)
            vp=v@psi;hp=diagonal*psi+g[3]*vp
            hs=diagonal*(psi*s)+g[3]*(v@(psi*s))
            for name,vector in [('energy',hp),('h_score',hs),('source',vp)]:
                action_error=max(action_error,float(np.max(abs(vector-psi*saved[name]))))
            E=float(psi@hp/norm);tau=psi*(s-mu)
            residual=hs-mu*hp-E*tau+vp;residual-=psi*(psi@residual/norm)
            source=vp-psi*(psi@vp/norm)
            checks=dict(chi_total=float(tau@tau/norm),energy_total=E,energy_variance=float((hp-E*psi)@(hp-E*psi)/norm),
                        response_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(source)),
                        hf_derivative_defect=float(2*tau@(hp-E*psi)/norm))
            moment_error=max(moment_error,max(abs(checks[k]-point[k]) for k in checks))
            centered_fd=fd-p@fd;centered_s=s-mu
            errors=np.sqrt(np.sum(p[:,None]*(centered_fd-centered_s[:,None])**2,axis=0)/checks['chi_total'])
            fd_error=max(fd_error,float(np.max(errors)))
            z=2*bits-1
            C=np.array([p@np.mean(z*np.roll(z,-r,axis=1),axis=1) for r in range(16)])
            corr_error=max(corr_error,float(np.max(abs(C-point['correlations']))));norm_error=max(norm_error,abs(norm-1))
            hashes[str(archive)]=sha(archive);count+=1
        hashes[str(folder/'audit.json')]=sha(folder/'audit.json')
    if action_error>2e-11 or moment_error>2e-11 or corr_error>1e-12 or norm_error>1e-10 or fd_error>5e-4:
        raise RuntimeError('independent L16 replay validation failed')
    report=dict(complete=True,points=count,independent_action_max_error=action_error,independent_moment_max_error=moment_error,
                all_correlation_replay_max_error=corr_error,normalization_max_error=norm_error,input_fd_relative_rms_max=fd_error,
                tolerances=dict(action=2e-11,moments=2e-11,correlation=1e-12,normalization=1e-10,fd_relative_rms=5e-4),
                cost=dict(born_samples=0,hamiltonian_actions=2*count,source_actions=count),
                validated_archive_sha256=hashes,protocol_sha256=sha(directory/'protocol.json'),
                review_sha256=sha(directory/'review.json'),validation_source_sha256=sha(Path(__file__)))
    write_json(directory/'replay_validation.json',report);print(directory,count,'action',action_error,'moments',moment_error,flush=True);return report


def plot(pair: Path,repair: Path) -> None:
    """Show paired errors and full response curves, with unequal budget labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=json.loads((pair/'review.json').read_text());updated=json.loads((repair/'review.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex='row',sharey='row')
    bindings={}
    for column,seed in enumerate((71,83,97)):
        configs=[('Sequential1050',pair/'runs'/f'sequential_seed_{seed}','C1x--'),
                 ('Joint50',pair/'runs'/f'joint_seed_{seed}','C0o-'),
                 ('Chart50',repair/'runs'/f'units_seed_{seed}/step_50','C2s-'),
                 ('Chart200 (4x tasks)',repair/'runs'/f'units_seed_{seed}/step_200','C3^-')]
        values=[]
        for label,folder,style in configs:
            path=folder/'audit.json';data=json.loads(path.read_text())['points'];bindings[str(path)]=sha(path)
            values.append(100*metrics(data)['chi'])
            axes[1,column].plot([p['couplings'][3] for p in data],[p['chi_reference_ratio'] for p in data],style,ms=3,label=label)
        axes[0,column].bar(range(4),values,color=['C1','C0','C2','C3'])
        axes[0,column].set_xticks(range(4),['Seq','Joint','Chart50','Chart200'],rotation=20)
        axes[0,column].axhline(10,color='grey',ls=':');axes[0,column].set_title(f'L16 parent seed {seed}')
        axes[1,column].axhline(1,color='black',lw=1);axes[1,column].axhspan(.9,1.1,color='grey',alpha=.13)
        axes[1,column].set_xlabel('Transverse field h')
        for row in (0,1):axes[row,column].grid(alpha=.2)
    axes[0,0].set_ylabel('Worst chi relative error (%)');axes[1,0].set_ylabel('chi / independent JW reference')
    axes[1,2].legend(fontsize=7);fig.tight_layout();path=repair/'comparison.png';fig.savefig(path,dpi=170);plt.close(fig)
    write_json(repair/'figure.json',dict(figure_sha256=sha(path),input_sha256=bindings,plot_source_sha256=sha(Path(__file__)),new_hamiltonian_actions=0,new_born_samples=0))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,required=True);parser.add_argument('--repair',type=Path)
    args=parser.parse_args();validate(args.pair)
    if args.repair is not None:validate(args.repair);plot(args.pair,args.repair)


if __name__=='__main__':main()
