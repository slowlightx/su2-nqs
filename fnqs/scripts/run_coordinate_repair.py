"""Conditional affine-coordinate SR repair, fixed physical GRU and loss."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from functools import partial
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
from fnqs.artifacts import load_checkpoint,write_json
from fnqs.pilots.engine import Options,model_signature
from fnqs.pilots.coordinate_sr import train
from fnqs.pilots.lineage import ROOT,sha,sources,verify_sources,save_node,cost,check_cost,inclusive_cost
from fnqs.pilots.measurements import amplitudes
from fnqs.scripts.run_l16_pair import checked_node,audit,metrics
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.variants.coupling_units import CouplingUnits,to_units,to_raw,physical_log_psi
from fnqs.diagnostics.finite_chain import finite_bits


@partial(jax.jit,static_argnames=('config','units'))
def unit_amplitudes(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig,units: CouplingUnits) -> tuple:
    """JVP is with respect to the physical h, before encoding."""
    return jax.jvp(lambda x:physical_log_psi(params,bits,x,config,units),(g,),(jnp.array([0.,0.,0.,1.]),))


def prepare(output: Path,parent: Path) -> None:
    """Seal 50/200 same-trajectory milestones after the fixed L16 review."""
    if output.exists():raise FileExistsError('fresh output directory required')
    previous=json.loads((parent/'protocol.json').read_text());review=json.loads((parent/'review.json').read_text())
    verify_sources(previous['source_sha256'])
    joint=[r for r in review['rows'] if r['arm']=='joint']
    if len(joint)!=3:raise ValueError('complete three-seed paired review required')
    if all(r['primary']['observable_pass'] for r in joint):
        raise ValueError('all joint development observables already pass; coordinate repair not needed')
    units=CouplingUnits(1.,.05);runs=[]
    for seed in previous['seeds']:
        control=next(r for r in previous['runs'] if r['seed']==seed and r['arm']=='joint')
        schedule=[control['train_points'] for _ in range(200)]
        runs.append(dict(name=f'units_seed_{seed}',seed=seed,parent=control['parent'],parent_sha256=control['parent_sha256'],
            train_points=control['train_points'],schedule=schedule,milestones=[50,200],
            endpoint_costs={str(k):cost(schedule[:k],previous['options']['samples']) for k in (50,200)},
            control_node=str((parent/'runs'/control['name']/'node.json').resolve()),
            control_audit=str((parent/'runs'/control['name']/'audit.json').resolve())))
    launchers=['fnqs/scripts/run_l16_pair.py','fnqs/scripts/run_coordinate_repair.py']
    manifest={**sources(),**{p:sha(ROOT/p) for p in launchers}}
    output.mkdir(parents=True)
    for name in manifest:
        target=output/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(output/'protocol.json',dict(kind='physical_GRU_affine_optimizer_coordinates_v16',n=16,runs=runs,
        model=previous['model'],options=previous['options'],units=asdict(units),source_sha256=manifest,
        primary_points=previous['primary_points'],reserved_points=previous['reserved_points'],
        input_sha256={str(parent/'review.json'):sha(parent/'review.json'),str(parent/'protocol.json'):sha(parent/'protocol.json')},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,jax=jax.__version__,platform=platform.platform(),backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64)),
        comparison='At step50, same initial physical model, fields, H/V/Born budget, updates, loss and nominal SR settings as joint v15; only optimizer coordinates differ.',
        budget_stage='Step200 is a predeclared larger budget along the same trajectory; no matched raw-coordinate 200 control, no unique coordinate attribution at this endpoint.',
        physical_model_features_width_loss_unchanged=True,parameter_count_unchanged=True,
        damping_and_clipping_are_coordinate_dependent=True,reference_labels_in_training=False,
        outputs_are_original_raw_parameter_API=True,phase0_scientific_gate=False,reserved_confirmation_measured=False))


def run(output: Path) -> None:
    """Run declared cumulative trajectories, seal every endpoint, then audit."""
    protocol=json.loads((output/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    for name,digest in protocol['input_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('development selection input changed')
    cfg=SizeFeatureConfig(**protocol['model']);options=Options(**protocol['options']);units=CouplingUnits(**protocol['units'])
    for r in protocol['runs']:
        folder=output/'runs'/r['name']
        if (folder/'step_200/node.json').exists():checked_node(folder/'step_200/node.json');continue
        if (folder/'started.json').exists():raise RuntimeError('partial trajectory preserved; fresh protocol required')
        parent=Path(r['parent']);node=checked_node(parent)
        if sha(parent)!=r['parent_sha256']:raise ValueError('raw parent changed')
        p,_,_,_=load_checkpoint(Path(node['checkpoint_path']));q=to_units(p,units);back=to_raw(q,units)
        parameter_error=max(float(np.max(abs(np.asarray(p[k])-np.asarray(back[k])))) for k in p)
        bits=jnp.asarray(finite_bits(16)[::128]);amplitude_error=score_error=0.
        for h in (.95,1.,1.05):
            g=jnp.array([1.,0.,0.,h]);a,s=amplitudes(p,bits,g,cfg);b,t=unit_amplitudes(q,bits,g,cfg,units)
            amplitude_error=max(amplitude_error,float(np.max(abs(np.asarray(a)-np.asarray(b)))))
            score_error=max(score_error,float(np.max(abs(np.asarray(s)-np.asarray(t)))))
        if max(parameter_error,amplitude_error,score_error)>1e-11:raise RuntimeError('initial physical equivalence failed')
        write_json(folder/'equivalence.json',dict(parent_sha256=sha(parent),parameter_roundtrip_max_error=parameter_error,
            log_amplitude_max_error=amplitude_error,physical_h_score_max_error=score_error,tolerance=1e-11,
            model_rows=512*3,hamiltonian_actions=0,born_samples=0))
        write_json(folder/'started.json',dict(protocol_sha256=sha(output/'protocol.json')))
        began=time.perf_counter()
        def callback(p,state,key,history):
            step=state['step']
            if step==1 or step%25==0:print(r['name'],step,'seconds',round(time.perf_counter()-began,2),flush=True)
            if step in r['milestones']:
                check_cost(state['work'],r['endpoint_costs'][str(step)])
                save_node(folder/f'step_{step}',p,state,key,signature=model_signature(cfg,16),protocol_path=output/'protocol.json',
                    train_points=r['train_points'],history=history,parents=[parent],stage='equivalent_coordinate_cumulative_warm_branch')
        train(p,jax.random.PRNGKey(r['seed']+2000),16,r['schedule'],cfg,units,options,callback)
        write_json(folder/'timing.json',dict(training_seconds=time.perf_counter()-began))
    endpoints={f"{r['name']}/step_{k}":sha(output/'runs'/r['name']/f'step_{k}/node.json') for r in protocol['runs'] for k in r['milestones']}
    write_json(output/'endpoint_seal.json',dict(nodes=endpoints,reference_opened=False))
    rows=[]
    for r in protocol['runs']:
        for step in r['milestones']:
            folder=output/'runs'/r['name']/f'step_{step}';checked_node(folder/'node.json')
            if not (folder/'audit.json').exists():
                p,_,_,_=load_checkpoint(folder/'checkpoint.npz');points=[]
                for i,g in enumerate(protocol['primary_points']):
                    points.append(audit(p,16,g,cfg,folder/f'point_{i}.npz'))
                    print('audit',r['name'],step,i+1,flush=True)
                write_json(folder/'audit.json',dict(points=points,checkpoint_sha256=sha(folder/'checkpoint.npz'),protocol_sha256=sha(output/'protocol.json')))
            points=json.loads((folder/'audit.json').read_text())['points'];control=checked_node(Path(r['control_node']))
            control_rows=json.loads(Path(r['control_audit']).read_text())['points']
            m=metrics(points);c=metrics(control_rows)
            row=dict(seed=r['seed'],step=step,primary=m,control_primary=c,chi_improvement=c['chi']-m['chi'],
                     correlation_change=m['correlation']-c['correlation'],audit_sha256=sha(folder/'audit.json'))
            if step==50:
                a=load_checkpoint(folder/'checkpoint.npz')[2];b=load_checkpoint(Path(control['checkpoint_path']))[2]
                if not np.array_equal(a,b):raise ValueError('matched step50 terminal PRNG differs')
                row['terminal_prng_matches_joint50']=True
            rows.append(row)
    terminal=[output/'runs'/r['name']/'step_200/node.json' for r in protocol['runs']]
    nodes=[checked_node(p) for p in terminal]
    keys=('updates','task_evaluations','hamiltonian_actions','source_actions','born_samples','sr_correction_solves')
    work={k:sum(n['exclusive_work'].get(k,0) for n in nodes) for k in keys}
    write_json(output/'review.json',dict(rows=rows,exclusive_training_cost=work,inclusive_training_cost=inclusive_cost(terminal),
        audit_cost=dict(born_samples=0,basis_rows=len(rows)*len(protocol['primary_points'])*65536,
                        hamiltonian_actions=2*len(rows)*len(protocol['primary_points']),source_actions=len(rows)*len(protocol['primary_points'])),
        coordinate50_consistent_improvement=all(r['chi_improvement']>=.02 and r['correlation_change']<=.005 and r['primary']['energy']<=.002 for r in rows if r['step']==50),
        all50_development_observables_pass=all(r['primary']['observable_pass'] for r in rows if r['step']==50),
        all200_development_observables_pass=all(r['primary']['observable_pass'] for r in rows if r['step']==200),
        protocol_sha256=sha(output/'protocol.json'),phase0_scientific_gate=False,reserved_confirmation_measured=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.parent is None:parser.error('--parent required')
        prepare(args.output,args.parent)
    else:run(args.output)


if __name__=='__main__':main()
