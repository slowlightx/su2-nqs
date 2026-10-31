"""Frozen-feature L16 paired SR validation with exhaustive response audits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from fnqs.artifacts import load_checkpoint, write_json
from fnqs.diagnostics.finite_chain import finite_bits, vector_action, full_correlations
from fnqs.diagnostics.response import summarize_response
from fnqs.exact import TFIMReference
from fnqs.pilots.engine import Options, train, model_signature
from fnqs.pilots.factorial import finite_scores
from fnqs.pilots.lineage import ROOT, sha, sources, verify_sources, read_node, save_node, cost, check_cost, validate_splits, inclusive_cost
from fnqs.pilots.measurements import amplitudes
from fnqs.variants.size_features import SizeFeatureConfig, parameter_counts

SEEDS = (71,83,97)


def checked_node(path: Path) -> dict:
    """Validate the transitive graph and return this node's own metadata."""
    read_node(path)
    return json.loads(path.read_text())


def matched_schedules(fields: list[list[float]], rounds: int) -> dict[str, list]:
    """Preserve task order/exposure; only group tasks into different updates."""
    if isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<1 or not fields:
        raise ValueError('positive rounds and fields required')
    if any(len(g)!=4 or not np.isfinite(g).all() for g in fields):
        raise ValueError('finite four-component fields required')
    joint = [fields for _ in range(rounds)]
    sequential = [[g] for tasks in joint for g in tasks]
    return dict(sequential=sequential,joint=joint)


def freeze(output: Path, declaration: dict) -> None:
    """Record versions and all numerical sources before any training."""
    if output.exists():
        raise FileExistsError('fresh output directory required')
    manifest={**sources(),'fnqs/scripts/run_l16_pair.py':sha(Path(__file__))}
    output.mkdir(parents=True)
    for name in manifest:
        target=output/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(output/'protocol.json',{**declaration,'source_sha256':manifest,
        'environment':dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,
                           scipy=scipy.__version__,jax=jax.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))})


def prepare(output: Path, rounds: int=50) -> None:
    """Declare paired warm branches of three independent frozen L16 parents."""
    oldpath=ROOT/'fnqs/results/fixed_size_pilots_v11/protocol.json'
    old=json.loads(oldpath.read_text())
    template=next(r for r in old['runs'] if r['name']=='energy_seed_71')
    fields=template['train_points']
    dev=json.loads(Path(old['phase0_parent_protocol']).read_text())['development_fields']
    primary=[[1.,0.,0.,h] for h in dev]
    reserved=[[1.,0.,0.,h] for h in old['phase0_reserved_confirmation_fields']]
    validate_splits(fields,primary,reserved)
    schedules=matched_schedules(fields,rounds)
    runs=[];parents=[]
    for seed in SEEDS:
        parent=ROOT/f'fnqs/results/fixed_size_pilots_v11/parents/seed_{seed}/node.json'
        node=checked_node(parent)
        if node['signature']!=model_signature(SizeFeatureConfig(**template['model']),16):
            raise ValueError('parent feature signature mismatch')
        parents.append(dict(seed=seed,node=str(parent),node_sha256=sha(parent),checkpoint_sha256=node['checkpoint_sha256']))
        for name,schedule in schedules.items():
            runs.append(dict(name=f'{name}_seed_{seed}',arm=name,seed=seed,parent=str(parent),parent_sha256=sha(parent),
                schedule=schedule,cost=cost(schedule,template['options']['samples']),train_points=fields))
    freeze(output,dict(kind='fixed_feature_L16_paired_SR_v15',n=16,seeds=list(SEEDS),rounds=rounds,
        model=template['model'],options=template['options'],parameter_counts=parameter_counts(SizeFeatureConfig(**template['model'])),
        runs=runs,parents=parents,primary_points=primary,reserved_points=reserved,
        input_sha256={str(oldpath):sha(oldpath)},
        grouping_is_only_training_change=True,total_task_H_V_Born_work_matched=True,number_of_updates_matched=False,
        audit='All 65536 configurations, all C(r), fixed-theta JVP/FD, independent JW; no MC measurement noise.',
        selection_criterion='Three paired chi-worst improvements >= .02 absolute; no C(r) worsening > .005 and no energy failure.',
        joint_method_selection_uses_development_only=True,reference_labels_in_training=False,
        no_automatic_budget_extension=True,reserved_confirmation_measured=False,phase0_scientific_gate=False))


def audit(params: dict, n: int, g: list[float], config: SizeFeatureConfig, archive: Path, chunk: int=1024) -> dict:
    """Exhaustive full-space audit, independent of local-energy sampling code."""
    x=np.asarray(g,dtype=np.float64)
    if x.shape!=(4,) or not np.isfinite(x).all() or not np.array_equal(x[:3],[1.,0.,0.]) or x[3]<=0:
        raise ValueError('JW audit requires j1=1,j2=jx=0,h>0')
    if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:
        raise ValueError('positive integer audit chunk required')
    bits=finite_bits(n)
    logs=[];scores=[];fd=[]
    for start in range(0,len(bits),chunk):
        b=jnp.asarray(bits[start:start+chunk]);x=jnp.asarray(g)
        log,s=amplitudes(params,b,x,config)
        logs.append(np.asarray(log));scores.append(np.asarray(s));fd.append(np.asarray(finite_scores(params,b,x,config)))
    psi=np.exp(np.concatenate(logs));score=np.concatenate(scores);fd_scores=np.concatenate(fd)
    norm_error=abs(float(psi@psi)-1)
    if norm_error>1e-10 or np.any(psi<=0):
        raise RuntimeError('normalization/positive-amplitude guard failed')
    hpsi=vector_action(psi,n,g);hs=vector_action(psi*score,n,g);v=vector_action(psi,n,[0.,0.,0.,1.])
    raw=dict(score=score,energy=hpsi/psi,h_score=hs/psi,source=v/psi,fd_scores=fd_scores)
    result=summarize_response(raw,psi**2)
    if max(result['fd_score_relative_rms_error'])>5e-4:
        raise RuntimeError('input derivative FD guard failed')
    ref=TFIMReference(n,g[3]);corr=full_correlations(psi,bits);reference_corr=np.array([ref.correlation(r) for r in range(n)])
    archive.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(archive,psi=psi,bits=bits,couplings=np.asarray(g),**raw)
    return {**result,'n':n,'couplings':g,'normalization_error':norm_error,
        'chi_relative_error':abs(result['chi_total']/ref.fidelity_susceptibility()-1),
        'reference_chi_total':ref.fidelity_susceptibility(),'chi_reference_ratio':result['chi_total']/ref.fidelity_susceptibility(),
        'reference_energy':ref.energy(),'energy_relative_error':abs(result['energy_total']-ref.energy())/abs(ref.energy()),
        'correlations':corr.tolist(),'reference_correlations':reference_corr.tolist(),
        'max_correlation_absolute_error':float(np.max(abs(corr-reference_corr))),
        'archive_sha256':sha(archive),'born_samples':0,'basis_rows':len(bits),'hamiltonian_actions':2,'source_actions':1}


def metrics(rows: list[dict]) -> dict:
    """Keep worst errors as the criterion, with descriptive RMS separately."""
    return dict(chi=max(r['chi_relative_error'] for r in rows),chi_rms=float(np.sqrt(np.mean([r['chi_relative_error']**2 for r in rows]))),
                correlation=max(r['max_correlation_absolute_error'] for r in rows),energy=max(r['energy_relative_error'] for r in rows),
                rho=max(r['response_relative_residual'] for r in rows),
                observable_pass=all(r['chi_relative_error']<=.1 and r['max_correlation_absolute_error']<=.03 and r['energy_relative_error']<=.002 for r in rows))


def run(output: Path, stage: str='all') -> None:
    """Execute declared arms, seal all before audit, never overwrite a failure."""
    protocol=json.loads((output/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    for name,digest in protocol['input_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('input changed')
    config=SizeFeatureConfig(**protocol['model']);options=Options(**protocol['options'])
    if stage in ('all','train'):
        for r in protocol['runs']:
            folder=output/'runs'/r['name']
            if (folder/'node.json').exists():read_node(folder/'node.json');continue
            if (folder/'started.json').exists():raise RuntimeError('partial training preserved; use a new protocol')
            parent=Path(r['parent']);node=checked_node(parent)
            if sha(parent)!=r['parent_sha256']:raise ValueError('parent changed')
            params,_,_,_=load_checkpoint(Path(node['checkpoint_path']))
            write_json(folder/'started.json',dict(protocol_sha256=sha(output/'protocol.json')))
            began=time.perf_counter()
            def callback(p,state,key,history):
                if state['step']==1 or state['step']%25==0:
                    print(r['name'],state['step'],'seconds',round(time.perf_counter()-began,2),'SR',history[-1]['true_sr_residual'],flush=True)
            p,s,k,h=train(params,jax.random.PRNGKey(r['seed']+2000),16,r['schedule'],config,options,callback)
            check_cost(s['work'],r['cost'])
            save_node(folder,p,s,k,signature=model_signature(config,16),protocol_path=output/'protocol.json',
                      train_points=r['train_points'],history=h,parents=[parent],stage='L16_paired_warm_branch')
            write_json(folder/'timing.json',dict(training_seconds=time.perf_counter()-began))
        write_json(output/'endpoint_seal.json',dict(nodes={r['name']:sha(output/'runs'/r['name']/'node.json') for r in protocol['runs']},reference_opened=False))
    if stage in ('all','audit'):
        if not (output/'endpoint_seal.json').exists():raise ValueError('all endpoints must be sealed first')
        sealed=json.loads((output/'endpoint_seal.json').read_text())['nodes']
        if sealed!={r['name']:sha(output/'runs'/r['name']/'node.json') for r in protocol['runs']}:
            raise ValueError('sealed endpoint identity changed')
        for r in protocol['runs']:
            folder=output/'runs'/r['name'];read_node(folder/'node.json')
            if (folder/'audit.json').exists():continue
            params,_,_,_=load_checkpoint(folder/'checkpoint.npz');rows=[]
            for i,g in enumerate(protocol['primary_points']):
                rows.append(audit(params,16,g,config,folder/f'point_{i}.npz'))
                print('audit',r['name'],i+1,flush=True)
            write_json(folder/'audit.json',dict(points=rows,checkpoint_sha256=sha(folder/'checkpoint.npz'),protocol_sha256=sha(output/'protocol.json')))
        review(output)


def review(output: Path) -> dict:
    """Verify actual costs/paired keys; do not promote a local observable pass."""
    protocol=json.loads((output/'protocol.json').read_text());rows=[]
    for r in protocol['runs']:
        folder=output/'runs'/r['name'];node=checked_node(folder/'node.json')
        data=json.loads((folder/'audit.json').read_text())
        if data['checkpoint_sha256']!=node['checkpoint_sha256'] or data['protocol_sha256']!=sha(output/'protocol.json'):
            raise ValueError('audit binding changed')
        rows.append(dict(run=r['name'],seed=r['seed'],arm=r['arm'],primary=metrics(data['points']),audit_sha256=sha(folder/'audit.json')))
    pairs=[]
    for seed in protocol['seeds']:
        a,b=[next(r for r in rows if r['seed']==seed and r['arm']==arm) for arm in ('sequential','joint')]
        ka=load_checkpoint(output/'runs'/a['run']/'checkpoint.npz')[2];kb=load_checkpoint(output/'runs'/b['run']/'checkpoint.npz')[2]
        if not np.array_equal(ka,kb):raise ValueError('paired random-key consumption differs')
        pairs.append(dict(seed=seed,chi_improvement=a['primary']['chi']-b['primary']['chi'],
                          correlation_change=b['primary']['correlation']-a['primary']['correlation'],terminal_prng_equal=True))
    supported=all(p['chi_improvement']>=.02 and p['correlation_change']<=.005 for p in pairs) and all(r['primary']['energy']<=.002 for r in rows if r['arm']=='joint')
    nodes=[checked_node(output/'runs'/r['name']/'node.json') for r in protocol['runs']]
    actual={key:sum(node['exclusive_work'].get(key,0) for node in nodes) for key in
            ('updates','task_evaluations','hamiltonian_actions','source_actions','born_samples','sr_correction_solves')}
    result=dict(rows=rows,pairs=pairs,consistent_joint_improvement=supported,exclusive_training_cost=actual,
        inclusive_training_cost=inclusive_cost([output/'runs'/r['name']/'node.json' for r in protocol['runs']]),
        audit_cost=dict(born_samples=0,basis_rows=len(rows)*len(protocol['primary_points'])*65536,
                        hamiltonian_actions=len(rows)*len(protocol['primary_points'])*2,source_actions=len(rows)*len(protocol['primary_points'])),
        protocol_sha256=sha(output/'protocol.json'),reserved_confirmation_measured=False,phase0_scientific_gate=False)
    write_json(output/'review.json',result);print('consistent joint improvement',supported,flush=True);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','audit','run','review'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=50)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output,args.rounds)
    elif args.command=='review':review(args.output)
    else:run(args.output,dict(train='train',audit='audit',run='all')[args.command])


if __name__=='__main__':main()
