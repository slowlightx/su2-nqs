"""Fresh-initialization confirmation of the unchanged v17 selected recipe."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from fnqs.artifacts import load_checkpoint, save_checkpoint, write_json
from fnqs.initialization import initialize_run
from fnqs.models import GRUConfig
from fnqs.exact import TFIMReference
from fnqs.pilots.coordinate_sr import train as coordinate_train
from fnqs.pilots.engine import Options, draw, model_signature
from fnqs.pilots.lineage import ROOT, sha, verify_sources, read_node, save_node, inclusive_cost, validate_splits
from fnqs.pilots.measurements import amplitudes
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.variants.coupling_units import CouplingUnits
from fnqs.diagnostics.confirmation import energy_rows, fd_rows, summarize_rows
from fnqs.scripts.run_l16_pair import freeze, audit as exact_audit
from fnqs.scripts.size_capacity_controls import build_protocol, field_schedule, selected_tasks, advance

SEEDS=[2026100511,2026100513,2026100517]


def adapt_state(state: dict) -> dict:
    """Translate actual original-training work without changing optimizer state."""
    w=state['work']
    return {**state,'work':dict(updates=int(state['step']),task_evaluations=w['gradient_hamiltonians'],
        hamiltonian_actions=w['hamiltonian_evaluations'],source_actions=0,born_samples=w['born_samples'],
        basis_rows=w['basis_state_evaluations'],sr_correction_solves=w['sr_refinement_solves'])}


def prepare(output: Path) -> None:
    """Freeze splits, full recipe, seeds, costs and analyses before training."""
    base=build_protocol(SEEDS)
    cfg=SizeFeatureConfig(16,64,'legacy')
    train=[[1.,0.,0.,h] for h in base['train_fields']]
    reserved=[[1.,0.,0.,h] for h in base['reserved_confirmation_fields']]
    validate_splits(train,[[1.,0.,0.,h] for h in base['development_fields']],reserved)
    selection=ROOT/'fnqs/results/phase0_step_pair_v17/review.json'
    declaration=dict(kind='independent_frozen_recipe_confirmation_v18',sizes=base['sizes'],seeds=SEEDS,
        model=asdict(cfg),base_protocol=base,train_points=train,reserved_points=reserved,
        calibration_points=[[1.,0.,0.,1.]],calibration_is_not_holdout=True,
        stages=[dict(name='pretrain',updates=1000,samples=128,options=base['training_options']),
                dict(name='coordinate',updates=200,options=asdict(Options())),
                dict(name='fast',updates=100,options=asdict(Options(learning_rate=4.)))],
        coupling_units=dict(center=1.,scale=.05),coordinate_prng_reset_offset=2000,
        independent_initialization=True,paired_initialization_across_sizes=True,
        training_labels_used=False,optimization_loss='original energy only',
        per_endpoint_expected_work=dict(updates=1300,task_evaluations=8300,hamiltonian_actions=14600,
            source_actions=6300,born_samples=659200,basis_rows=0),
        audit=dict(born_samples=65536,energy_rows=4096,fd_rows=2048,chunk=256,bootstrap=200,
            seed_namespace=0xC0F018,exact_n=16,exact_chunk=1024,
            chi_gate=.1,correlation_gate=.03,energy_gate=.002,uncertainty_multiplier=2),
        analysis=dict(windows=[.6,.8,1.],degrees=[4,6],corrections=[False,True],
            primary=dict(window=.8,degree=6,correction=True),eta_r_min=[2,3,4],
            eta_primary_r_min=4,peak_windows=[5,7,9],bootstrap=200,
            bootstrap_seed=2026100523,chi_fit_weight='fixed per-size observed mean',
            eta_field='interpolation at freely fitted hc',
            stability_limits=dict(hc=.02,nu=.05,eta=.03),
            minimum_size_drop_requires_four_sizes=True,
            size_bootstrap_available=False,confidence_limits='three initialization seeds; common bias is not covered'),
        method_selection_input=dict(path=str(selection),sha256=sha(selection)),
        no_adaptive_budget_extension=True,conditional_L48_extension=False,
        final_endpoints_sealed_before_confirmation=True,confirmation_measured=False,
        phase0_scientific_gate=False)
    freeze(output,declaration)
    # Extend the manifest before any stochastic work, then copy actual launchers.
    proto=json.loads((output/'protocol.json').read_text())
    import shutil
    for name in ('fnqs/scripts/run_frozen_confirmation.py','fnqs/scripts/size_capacity_controls.py',
                 'fnqs/scripts/analyze_frozen_confirmation.py'):
        proto['source_sha256'][name]=sha(ROOT/name)
        target=output/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    schedules={}
    for seed in SEEDS:
        path=output/f'schedule_{seed}.npy'
        np.save(path,field_schedule(seed,1000,base['sizes'],base['train_fields']),allow_pickle=False)
        schedules[str(seed)]=sha(path)
    proto['schedule_sha256']=schedules
    write_json(output/'protocol.json',proto)


def verified(output: Path) -> dict:
    """Reject changed recipe, source, schedule or method-selection receipt."""
    p=json.loads((output/'protocol.json').read_text());verify_sources(p['source_sha256'])
    if sha(Path(p['method_selection_input']['path']))!=p['method_selection_input']['sha256']:
        raise ValueError('method-selection input changed')
    for seed,digest in p['schedule_sha256'].items():
        if sha(output/f'schedule_{seed}.npy')!=digest:raise ValueError('schedule changed')
    return p


def train_all(output: Path) -> None:
    """Repeat all 1300 updates; seal all nine endpoints before held references."""
    p=verified(output);cfg=SizeFeatureConfig(**p['model']);base=p['base_protocol']
    for seed in p['seeds']:
        schedule=np.load(output/f'schedule_{seed}.npy')
        for n in p['sizes']:
            run=output/'runs'/f'L{n}_seed_{seed}';began=time.perf_counter();parent=None
            for stage in p['stages']:
                folder=run/stage['name'];node=folder/'node.json'
                if node.exists():
                    read_node(node);params,state,key,_=load_checkpoint(folder/'checkpoint.npz');parent=node;continue
                if (folder/'started.json').exists():raise RuntimeError('partial trajectory retained; no silent restart')
                write_json(folder/'started.json',dict(protocol_sha256=sha(output/'protocol.json')))
                if stage['name']=='pretrain':
                    params,state,key,_=initialize_run(seed,GRUConfig(16,64));history=[]
                    for epoch in range(stage['updates']):
                        tasks=selected_tasks(base,{'sizes':[n]},schedule,epoch)
                        params,state,key,records=advance(params,state,key,GRUConfig(16,64),base,{'sizes':[n]},schedule,1)
                        records[0]['tasks']=[t.couplings.vector().tolist() for t in tasks]
                        records[0]['true_sr_residual']=records[0]['sr_relative_residual']
                        history.extend(records)
                        if (epoch+1)%100==0:
                            save_checkpoint(folder/'partial.npz',params,state,key,dict(history=history,protocol_sha256=sha(output/'protocol.json')))
                            print(run.name,'pretrain',epoch+1,'seconds',round(time.perf_counter()-began,1),flush=True)
                    write_json(folder/'original_work.json',state['work']);state=adapt_state(state)
                else:
                    if stage['name']=='coordinate':key=jax.random.PRNGKey(seed+p['coordinate_prng_reset_offset'])
                    def callback(a,s,k,h):
                        if s['step']%25==0:
                            save_checkpoint(folder/'partial.npz',a,s,k,dict(history=h,protocol_sha256=sha(output/'protocol.json')))
                            print(run.name,stage['name'],s['step'],'seconds',round(time.perf_counter()-began,1),flush=True)
                    params,state,key,history=coordinate_train(params,key,n,[p['train_points']]*stage['updates'],cfg,
                        CouplingUnits(**p['coupling_units']),Options(**stage['options']),callback)
                save_node(folder,params,state,key,signature=model_signature(cfg,n),protocol_path=output/'protocol.json',
                    train_points=p['train_points'],history=history,parents=[] if parent is None else [parent],stage=stage['name'])
                parent=node
            actual=inclusive_cost([parent])['work']
            if any(actual[k]!=v for k,v in p['per_endpoint_expected_work'].items()):raise ValueError('full recipe work mismatch')
            write_json(run/'timing.json',dict(seconds=time.perf_counter()-began))
    nodes={f'L{n}_seed_{seed}':sha(output/'runs'/f'L{n}_seed_{seed}'/'fast/node.json')
           for seed in p['seeds'] for n in p['sizes']}
    write_json(output/'endpoint_seal.json',dict(nodes=nodes,confirmation_references_opened=False,
        protocol_sha256=sha(output/'protocol.json')))


def sampled_audit(params: dict,n: int,g: list,cfg: SizeFeatureConfig,key: jax.Array,
                  archive: Path,options: dict,bootstrap_seed: int) -> dict:
    """Independent iid Born QGT/all-C with disjoint prefix energy and FD work."""
    count=options['born_samples'];chunk=options['chunk'];x=jnp.asarray(g)
    bits=np.asarray(draw(params,key,count,n,x,cfg));scores=[];energies=[];finite=[]
    for start in range(0,count,chunk):
        b=jnp.asarray(bits[start:start+chunk]);scores.append(np.asarray(amplitudes(params,b,x,cfg)[1]))
        if start<options['energy_rows']:energies.append(np.asarray(energy_rows(params,b,x,cfg)))
        if start<options['fd_rows']:finite.append(np.asarray(fd_rows(params,b,x,cfg)))
    score=np.concatenate(scores);energy=np.concatenate(energies);fd=np.concatenate(finite)
    stats=summarize_rows(bits,score,energy,fd,bootstrap_seed,options['bootstrap'])
    if max(stats['fd_score_relative_rms_error'])>5e-4:raise RuntimeError('physical h finite-difference guard failed')
    archive.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(archive,bits=bits,score=score,energy=energy,fd_scores=fd,couplings=g,
        joint_covariance=stats['joint_covariance'],chi_bootstrap_replicates=stats['chi_bootstrap_replicates'])
    ref=TFIMReference(n,g[3]);c=np.asarray([ref.correlation(r) for r in range(n)]);chi=ref.fidelity_susceptibility()
    return {**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()},
        'n':n,'couplings':g,'reference_chi_total':chi,'chi_relative_error':abs(stats['chi_total']/chi-1),
        'reference_energy':ref.energy(),'energy_relative_error':abs(stats['energy_total']/ref.energy()-1),
        'reference_correlations':c.tolist(),'max_correlation_absolute_error':float(np.max(abs(stats['correlations']-c))),
        'chi_upper_error':float((abs(stats['chi_total']-chi)+2*stats['chi_se'])/chi),
        'correlation_upper_error':float(np.max(abs(stats['correlations']-c)+2*stats['correlation_se'])),
        'energy_upper_error':float((abs(stats['energy_total']-ref.energy())+2*stats['energy_se'])/abs(ref.energy())),
        'archive_sha256':sha(archive),'born_samples':count,'basis_rows':0,'hamiltonian_actions':1,'source_actions':0}


def audit_all(output: Path) -> None:
    """Open holdout labels only after validating the complete endpoint seal."""
    p=verified(output);seal=json.loads((output/'endpoint_seal.json').read_text())
    cfg=SizeFeatureConfig(**p['model']);records=[]
    for seed in p['seeds']:
        for n in p['sizes']:
            run=output/'runs'/f'L{n}_seed_{seed}';node=run/'fast/node.json';read_node(node)
            if seal['nodes'][run.name]!=sha(node):raise ValueError('endpoint seal changed')
            params,_,_,_=load_checkpoint(run/'fast/checkpoint.npz');rows=[]
            for i,g in enumerate(p['reserved_points']+p['calibration_points']):
                receipt=run/'audit'/f'point_{i}.json';archive=run/'audit'/f'point_{i}.npz'
                if receipt.exists():
                    row=json.loads(receipt.read_text())
                    if row['archive_sha256']!=sha(archive):raise ValueError('audit archive changed')
                else:
                    if n==16:
                        row=exact_audit(params,n,g,cfg,archive,p['audit']['exact_chunk'])
                        row.update(chi_se=0.,correlation_se=[0.]*n,energy_se=0.,joint_covariance=np.zeros((n+1,n+1)).tolist(),
                            chi_upper_error=row['chi_relative_error'],correlation_upper_error=row['max_correlation_absolute_error'],
                            energy_upper_error=row['energy_relative_error'])
                    else:
                        key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),p['audit']['seed_namespace']),n),i)
                        row=sampled_audit(params,n,g,cfg,key,archive,p['audit'],seed+i+n)
                    row.update(checkpoint_sha256=sha(run/'fast/checkpoint.npz'),protocol_sha256=sha(output/'protocol.json'),
                               split='reserved_confirmation' if i<len(p['reserved_points']) else 'training_field_calibration')
                    write_json(receipt,row)
                rows.append(row)
                print('confirmation',run.name,i+1,'chi',round(row['chi_upper_error'],4),flush=True)
            write_json(run/'audit.json',dict(points=rows,checkpoint_sha256=sha(run/'fast/checkpoint.npz')))
            held=rows[:len(p['reserved_points'])]
            records.append(dict(n=n,seed=seed,chi=max(r['chi_upper_error'] for r in held),
                correlation=max(r['correlation_upper_error'] for r in held),energy=max(r['energy_upper_error'] for r in held),
                passed=all(r['chi_upper_error']<=.1 and r['correlation_upper_error']<=.03 and r['energy_upper_error']<=.002 for r in held)))
    endpoints=[output/'runs'/f'L{n}_seed_{s}'/'fast/node.json' for s in p['seeds'] for n in p['sizes']]
    write_json(output/'confirmation.json',dict(rows=records,independent_initializations=3,
        confirmed_observable_accuracy=all(r['passed'] for r in records),training_cost=inclusive_cost(endpoints),
        protocol_sha256=sha(output/'protocol.json'),phase0_scientific_gate=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','audit','run'])
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:
        if args.command in ('train','run'):train_all(args.output)
        if args.command in ('audit','run'):audit_all(args.output)


if __name__=='__main__':main()
