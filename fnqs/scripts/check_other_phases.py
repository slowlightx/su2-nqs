"""Read-only evidence/holdout/cost and independent-Pauli review of Phases 1/2."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from fnqs.artifacts import load_checkpoint,write_json
from fnqs.diagnostics.response import summarize_response
from fnqs.pilots.lineage import ROOT,sha,read_node,verify_sources,verify_holdout,inclusive_cost,check_cost,validate_splits
from fnqs.pilots.measurements import amplitudes
from fnqs.hamiltonians import IsingCouplings
from fnqs.sectors import parity_hamiltonian,lift_parity_state
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.basis import all_bits


def phase1(directory: Path) -> dict:
    """Replay all v14 raw amplitudes using independent sparse parity operators."""
    p=json.loads((directory/'protocol.json').read_text());verify_sources(p['source_sha256'])
    for name,digest in p['input_bindings'].items():
        if sha(ROOT/name)!=digest:raise ValueError('phase1 method selection changed')
    old=json.loads((ROOT/'fnqs/results/fixed_size_pilots_v11/protocol.json').read_text())
    review=json.loads((directory/'review.json').read_text());seal=json.loads((directory/'endpoint_seal.json').read_text())['nodes']
    if review['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('phase1 protocol changed')
    count=0;moment=0.;action=0.;correlation=0.;normalization=0.;fd=0.;nodes=[];bindings={};cache={}
    for run in p['runs']:
        folder=directory/'runs'/run['name'];node=Path(run['reused_node']) if run.get('reused_node') else folder/'node.json'
        graph=read_node(node);meta=graph[node.resolve()];check_cost(meta['exclusive_work'],run['cost']);nodes.append(node)
        if seal.get(run['name'])!=sha(node):raise ValueError('phase1 endpoint seal mismatch')
        validate_splits(run['train_points'],run['primary_points']+run['secondary_points'],
            [g for cut in old['cuts'] if cut['family'] in run['families'] for g in cut['reserved']])
        data=json.loads((folder/'audit.json').read_text())
        if data['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('phase1 audit binding changed')
        if run.get('reused_node'):
            if data['reused_node_sha256']!=sha(node):raise ValueError('reused phase1 ancestor changed')
        elif data['checkpoint_sha256']!=meta['checkpoint_sha256']:raise ValueError('phase1 checkpoint changed')
        expected=[('primary',g) for g in run['primary_points']]+[('secondary',g) for g in run['secondary_points']]
        if len(data['points'])!=len(expected):raise ValueError('phase1 point count')
        for i,(row,(split,g)) in enumerate(zip(data['points'],expected)):
            path=folder/f'point_{i}.npz'
            if row['couplings']!=g or row['split']!=split or sha(path)!=row['archive_sha256']:raise ValueError('phase1 raw labels/hash')
            refpath=directory/'references'/('L12_'+'_'.join(str(x) for x in g)+'.npz')
            if sha(refpath)!=row['reference_sha256']:raise ValueError('phase1 reference changed')
            with np.load(path,allow_pickle=False) as z:
                psi=z['psi'];bits=z['bits'];raw={k:z[k] for k in ('score','energy','h_score','source','fd_scores')}
                if not np.array_equal(bits,all_bits(12)) or not np.array_equal(z['couplings'],g):raise ValueError('phase1 basis labels')
            result=summarize_response(raw,psi**2)
            for key in ('chi_total','energy_total','energy_variance','response_relative_residual','hf_derivative_defect'):
                moment=max(moment,abs(row[key]-result[key]))
            label=tuple(g)
            if label not in cache:cache[label]=(parity_hamiltonian(12,IsingCouplings(*g),1),parity_hamiltonian(12,IsingCouplings(0,0,0,1),1))
            h,v=cache[label]
            if np.max(abs(psi-psi[::-1]))>1e-13 or np.max(abs(raw['score']-raw['score'][::-1]))>1e-13:raise ValueError('phase1 Z2 parity')
            for key,op,vec in [('energy',h,psi),('h_score',h,psi*raw['score']),('source',v,psi)]:
                action=max(action,float(np.max(abs(lift_parity_state(op@(np.sqrt(2)*vec[:2048]),12,1)-psi*raw[key]))))
            spins=2*bits.astype(float)-1
            corr=np.array([psi**2@np.mean(spins*np.roll(spins,-r,axis=1),axis=1) for r in range(12)])
            correlation=max(correlation,float(np.max(abs(corr-row['correlations']))));normalization=max(normalization,abs(psi@psi-1))
            fd=max(fd,max(result['fd_score_relative_rms_error']));count+=1;bindings[str(path)]=sha(path)
        bindings[str(node)]=sha(node);bindings[str(folder/'audit.json')]=sha(folder/'audit.json')
    if moment>1e-12 or action>2e-11 or correlation>1e-12 or normalization>1e-10 or fd>5e-4:raise RuntimeError('phase1 numerical replay failed')
    return dict(artifact_integrity_pass=True,points=count,moment_max_discrepancy=moment,action_max_discrepancy=action,
        correlation_max_discrepancy=correlation,normalization_max_error=normalization,fd_relative_max=fd,
        tolerances=dict(moment=1e-12,action=2e-11,correlation=1e-12,normalization=1e-10,fd_relative=5e-4),
        training_cost=inclusive_cost(nodes),validation_cost=dict(born_samples=0,hamiltonian_actions=2*count,source_actions=count),
        reserved_confirmation_measured=False,science_pass=False,large_BC_class_confirmed=False,
        limitations=['L12 only','B center passes but wide response fails','A/C responses remain inaccurate',
                     'no independent confirmation','B/C independent class references only through L14'],
        rows=review['rows'],validated_sha256=bindings,protocol_sha256=sha(directory/'protocol.json'))


def dense_pauli(n: int,g: list) -> tuple[np.ndarray,np.ndarray]:
    """Independent explicit tensor-product Hamiltonian and dH/dh."""
    x=np.array([[0.,1.],[1.,0.]]);z=np.diag([-1.,1.])
    def op(sites):
        a=np.ones((1,1))
        for s in range(n):a=np.kron(a,sites.get(s,np.eye(2)))
        return a
    h=np.zeros((1<<n,1<<n));v=np.zeros_like(h)
    for s in range(n):
        h-=g[0]*op({s:z,(s+1)%n:z})+g[1]*op({s:z,(s+2)%n:z})+g[2]*op({s:x,(s+1)%n:x})
        v-=op({s:x})
    return h+g[3]*v,v


def phase2(directory: Path) -> dict:
    """Check each physical-family holdout, matched budget and all 120 audits."""
    p=json.loads((directory/'protocol.json').read_text());verify_sources(p['source_sha256'])
    summary=json.loads((directory/'summary.json').read_text());cfg=SizeFeatureConfig(**p['model']);n=p['n']
    if summary['protocol_sha256']!=sha(directory/'protocol.json'):raise ValueError('phase2 protocol changed')
    nodes=[];count=0;moment=0.;reference=0.;norm=0.;bindings={};pairs=[]
    for seed in p['seeds']:
        for fold in p['folds']:
            base=directory/f"held_{fold['held_family']}"/f'seed_{seed}';parent=base/'pretrain/node.json'
            verify_holdout([parent],fold['held_family'])
            seal=json.loads((base/'endpoint_seal.json').read_text())['nodes']
            seal={Path(k).resolve():v for k,v in seal.items()}
            endpoints=[('frozen_zero_shot',parent,fold['held_audit_points'])]
            for arm in fold['arms']:
                path=base/arm['name']/'node.json';meta=read_node(path)[path.resolve()]
                expected=[str(parent.resolve())] if arm['kind']=='fewshot' else []
                if [x['path'] for x in meta['parents']]!=expected:raise ValueError('phase2 ancestry policy')
                validate_splits(arm['train_points'],fold['held_audit_points'],[]);check_cost(meta['exclusive_work'],arm['cost'])
                endpoints.append((arm['name'],path,fold['held_audit_points']))
            for k in (1,2,4,8):
                aa=base/f'fewshot_{k}/node.json';bb=base/f'scratch_{k}/node.json'
                ma=read_node(aa)[aa.resolve()];mb=read_node(bb)[bb.resolve()]
                ka=load_checkpoint(aa.parent/'checkpoint.npz')[2];kb=load_checkpoint(bb.parent/'checkpoint.npz')[2]
                if ma['exclusive_work']!=mb['exclusive_work'] or [r['tasks'] for r in ma['history']]!=[r['tasks'] for r in mb['history']] or not np.array_equal(ka,kb):raise ValueError('phase2 paired work/schedule/key')
                pairs.append(dict(held_family=fold['held_family'],shots=k,matched=True,
                    adaptation_born=ma['exclusive_work']['born_samples'],inclusive_fewshot_born=inclusive_cost([aa])['work']['born_samples']))
            for i,arm in enumerate(fold['pointwise']):endpoints.append((f'pointwise_{i}',base/f'pointwise_{i}/node.json',arm['train_points']))
            for name,path,points in endpoints:
                meta=read_node(path)[path.resolve()];nodes.append(path)
                if seal.get(path.resolve())!=sha(path):raise ValueError('phase2 endpoint seal changed')
                data=json.loads((base/f'{name}_audit.json').read_text())
                if data['node_sha256']!=sha(path) or data['inclusive_training_cost']!=inclusive_cost([path]):raise ValueError('phase2 audit/cost changed')
                params,_,_,_=load_checkpoint(path.parent/'checkpoint.npz');bits=all_bits(n)
                if len(data['points'])!=len(points):raise ValueError('phase2 point count')
                for row,g in zip(data['points'],points):
                    if row['couplings']!=g:raise ValueError('phase2 coupling labels')
                    logs,s=amplitudes(params,jnp.asarray(bits),jnp.asarray(g),cfg);psi=np.exp(np.asarray(logs));s=np.asarray(s)
                    h,v=dense_pauli(n,g);e,vec=np.linalg.eigh(h);ground=vec[:,0];overlap=vec[:,1:].T@v@ground
                    chi=float(np.sum((overlap/(e[1:]-e[0]))**2));reference=max(reference,abs(chi-row['reference_chi_total']))
                    raw=dict(score=s,energy=h@psi/psi,h_score=h@(psi*s)/psi,source=v@psi/psi,fd_scores=s[:,None])
                    result=summarize_response(raw,psi**2)
                    for key in ('chi_total','energy_total','energy_variance','response_relative_residual'):
                        moment=max(moment,abs(result[key]-row[key]))
                    norm=max(norm,abs(psi@psi-1));count+=1
                bindings[str(path)]=sha(path);bindings[str(base/f'{name}_audit.json')]=sha(base/f'{name}_audit.json')
    cost=inclusive_cost(nodes)
    if cost!=summary['aggregate_training_cost'] or moment>2e-11 or reference>5e-10 or norm>1e-10:raise RuntimeError('phase2 independent review failed')
    return dict(artifact_integrity_pass=True,transitive_holdouts_pass=True,matched_adaptation_budget_pass=True,
        points=count,training_cost=cost,pairs=pairs,moment_max_discrepancy=moment,
        independent_dense_reference_chi_max_discrepancy=reference,normalization_max_error=norm,
        tolerances=dict(moment=2e-11,reference_chi=5e-10,normalization=1e-10),
        validation_cost=dict(born_samples=0,basis_rows=count*(1<<n),hamiltonian_actions=2*count,source_actions=count,
                             dense_reference_solves=count),
        engineering_complete=True,science_pass=False,limitations=['L6 only','one seed','no cost-to-accuracy campaign',
        'all-family Phase1 weights cannot be held-family pretraining parents','pointwise directly trains at audit fields'],
        rows=summary['rows'],validated_sha256=bindings,protocol_sha256=sha(directory/'protocol.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh review output required')
    args.output.mkdir(parents=True)
    one=phase1(ROOT/'fnqs/results/joint_family_solution_v14');write_json(args.output/'phase1.json',one)
    two=phase2(ROOT/'fnqs/results/transfer_runner_canary_v11');write_json(args.output/'phase2.json',two)
    write_json(args.output/'receipt.json',dict(source_sha256=sha(Path(__file__)),phase1_sha256=sha(args.output/'phase1.json'),
                                             phase2_sha256=sha(args.output/'phase2.json')))
    print('phase1/2 artifact and independent numerical checks passed',flush=True)
